"""
Main training script for SSL
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import time
import warnings
import os
import random


from Semi_Supervised.data_loaders import create_dataloaders
import Semi_Supervised.augmentation
from monai.transforms import (
    KeepLargestConnectedComponent,
    FillHoles,
    RemoveSmallObjects,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Get rid of duplicate DLL issues


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combined loss with Cross-Entropy and Dice loss.
        Args:
        - alpha: Weight for the Dice loss in the combined loss (default: 0.5).
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Compute the combined loss.
        Args:
        - pred: Predicted logits of shape (N, C, H, W, D).
        - target: Ground truth labels of shape (N, H, W, D).
        Returns:
        - Combined loss value.
        """
        ce_loss = self.ce(pred, target)
        dice = dice_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * ce_loss


class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, focal_alpha=1.0):
        """
        Combines Focal Loss and Dice Loss.
        Args:
        - alpha: Weight for Dice loss (default: 0.5).
        - beta: Weight for Focal loss (default: 0.5).
        - gamma: Focusing parameter for Focal loss.
        - focal_alpha: Weighting factor for Focal loss to balance class importance.
        - class_weights: Class weights for Focal loss.
        - smooth: Smoothing factor for Dice loss to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, pred, target):
        """
        Compute the Focal loss.
        Args:
        - pred: Predicted logits of shape (N, C, H, W, D).
        - target: Ground truth labels of shape (N, H, W, D).
        Returns:
        - Focal loss value.
        """
        pred = F.log_softmax(pred, dim=1)  # Log probabilities
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 4, 1, 2, 3)  # Convert to one-hot

        ce_loss = -target_one_hot * pred  # Cross-entropy loss
        pt = torch.exp(-ce_loss)  # Probabilities of correct class
        focal_loss = self.focal_alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

    def forward(self, pred, target):
        """
        Compute the combined loss.
        Args:
        - pred: Predicted logits of shape (N, C, H, W, D).
        - target: Ground truth labels of shape (N, H, W, D).
        Returns:
        - Combined loss value.
        """
        dice = dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.alpha * dice + self.beta * focal


def dice_loss(pred, target, smooth=1e-5):
    """
    Compute the Dice loss.
    Args:
    - pred: Predicted logits of shape (N, C, H, W, D).
    - target: Ground truth labels of shape (N, H, W, D).
    - smooth: Smoothing factor to avoid division by zero.
    Returns:
    - Dice loss value.
    """
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 4, 1, 2, 3)  # Convert to one-hot

    intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def get_unet3D(in_channels=1, n_classes=9):
    """
    Get a standard UNet 3D model from Monai
    :param in_channels:
    :param n_classes:
    :return:
    """
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    )
    return model


def load_pretrained_weights(model: torch.nn.Module, modelStateDictPath: Path, config) -> \
        (torch.nn.Module, torch.optim, int, float, list[float]):
    """
    Load pretrained weights (from a previous run) back onto a model and optimizer
    :param model: model to have weights loaded onto
    :param modelStateDictPath: Path to .pth / .pt file containing weights
    :param config: Training config
    :return: model, optimizer, scheduler, epoch, best dice score, all_dice_scores
    """

    # Load model onto device
    model.to(config.device)

    path = Path(modelStateDictPath)
    data = torch.load(path)
    epoch = 0
    dice_score = 0
    all_dice_scores = []
    try:
        epoch = data["epoch"]
        print(f"""
            Model Loaded From {path}
            Validation Loss: {data["val_loss"]}
            Dice Score: {data["dice_score"]}
            Epoch: {epoch}
            ALL Dice Scores: {data["all_dice_scores"]}
                    """)
        model.load_state_dict(data["model_state_dict"])
        dice_score = data["dice_score"]
        all_dice_scores = data["all_dice_scores"]
    except KeyError:
        print("Model not loaded from pretrained as data not saved properly!")

    # Set up preloaded optimizer
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_weights = data.get("optimizer_state_dict", None)
    if optimizer_weights:
        optimizer.load_state_dict(optimizer_weights)
    else:
        print("No optimizer weights found in checkpoint, using default.")

    # Set up preloaded scheduler
    scheduler = config.scheduler(optimizer, T_max=config.num_epochs/4)
    scheduler_weights = data.get("scheduler_state_dict", None)
    if scheduler_weights:
        scheduler.load_state_dict(scheduler_weights)
    else:
        print("No scheduler weights found, using default.")

    return model, optimizer, scheduler, int(epoch), dice_score, all_dice_scores


def get_unet3D_larger(
        in_channels=1,
        n_classes=9,
        base_channels=64,
        num_res_units=3,
        dropout_prob=0.2
) -> (torch.nn.Module, torch.optim, int):
    """
    Loads a unet3D model and optimizer

    Parameters:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        n_classes (int): Number of output classes for segmentation.
        base_channels (int): Number of filters in the first layer, will double at each subsequent layer.
        num_res_units (int): Number of residual units per layer.
        dropout_prob (float): Dropout probability for regularization.

    Returns:
        model (torch.nn.Module): A 3D UNet model.
        optimizer (torch.optim): An optimizer
    """
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm

    # Define the number of channels in each layer (doubles at each level)
    channels = (
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
        base_channels * 16
    )

    # Define the strides for downsampling at each layer
    strides = (2, 2, 2, 2)

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act="RELU",
        norm=Norm.BATCH,
        dropout=dropout_prob,
    )
    return model


def load_swin_unetr(num_classes: int=9, pretrained=True) -> torch.nn.Module:
    """
    Loads the pretrained swin_unetr_btcv_segmentation model from Monai link: https://monai.io/model-zoo.html
    :param: num_classes: Number of output classes
    :return: model - loaded model with pretrained weights
    """
    from monai.bundle import ConfigParser, load
    import torch
    from pathlib import Path
    from monai.networks.blocks import UnetOutBlock

    # NOTE: AUTOMATICALLY RESCALES THE INPUT SIZE AS LONG AS MULTIPLE OF 32

    # Load the model from online resource
    model_dict = load("swin_unetr_btcv_segmentation")

    # Get path to the directory and model config
    torch_hub_dir = Path(torch.hub.get_dir())
    config_path = next(torch_hub_dir.glob("**/swin_unetr_btcv_segmentation/configs/train.json"))

    # Define and setup model according to instructions in downloaded model path
    config = ConfigParser()
    config.read_config(config_path)
    net = config.get_parsed_content("network_def", instantiate=True)

    # Load pretrained weights from download
    if pretrained:
        net.load_state_dict(model_dict)

    # Set last block for out 9 classes
    in_channels = net.out.conv.conv.in_channels
    net.out = UnetOutBlock(spatial_dims=3, in_channels=in_channels, out_channels=num_classes)

    return net


def softmax_confidence(predictions, T=2.0):
    """
    Calculate confidence from the softmax predictions.
    :param predictions: model output
    :param T: temperature - lower confidence score
    :return: confidence, pseudo_labels
    """
    probabilities = F.softmax(predictions / T, dim=1)
    confidence, pseudo_labels = torch.max(probabilities, dim=1)
    return confidence, pseudo_labels


class SimpleSSL:
    """
    Main script for Semi Supervised Learning
    """
    def __init__(self, model, criterion, optimizer, scheduler: torch.optim.lr_scheduler, device,
                 confidence_threshold=0.9, n_classes=9, data_dir=None, patience=15):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.best_val_loss = float('inf')
        self.n_classes = n_classes  # Including background class
        self.best_dice = 0
        self.all_dice_scores = []
        self.data_dir = data_dir  # For saving psuedo labels to during an epoch for RAM memory efficiency

        self.patience = patience
        self.epoch_since_last_improvement = 0
        self.best_dice_epoch = 0

        labels = list(range(1, n_classes))
        self.keep_largest = KeepLargestConnectedComponent(applied_labels=labels)
        self.fill_holes = FillHoles(applied_labels=labels)
        self.remove_small = RemoveSmallObjects(min_size=64)

    def generate_pseudo_labels(self, unlabeled_loader):
        """
        Generate pseudo labels for the data in the unlabeled loader.

        Saves values to disk as .pt file with torch.save for RAM usage issues (storing so many 3D files gets big)
        """
        self.model.eval()
        output_dir = Path(self.data_dir) if self.data_dir else None

        with torch.no_grad():
            for i, batch in enumerate(unlabeled_loader):
                print(f"\rProcessing unlabeled batch {i + 1}/{len(unlabeled_loader)}", end="")

                images = batch['image'].to(self.device)  # Shape: (B, 1, D, H, W)
                codes = batch['code']  # Unique identifiers for saving
                outputs = self.model(images)  # Shape: (B, 9, D, H, W)

                confidence, labels = softmax_confidence(outputs)  # Shape: (B, D, H, W)
                mask = confidence >= self.confidence_threshold

                # Save to disk or augment dataloader output
                for j in range(len(images)):
                    pseudo_data = {
                        'label': labels[j].cpu(),
                        'confidence': confidence[j].cpu(),
                        'mask': mask[j].cpu(),
                    }

                    if output_dir:
                        # Save pseudo-labels to disk
                        torch.save(pseudo_data, output_dir / f"{codes[j]}_pseudo.pt")

                # Free GPU memory
                del images, outputs, confidence, labels, mask
                torch.cuda.empty_cache()

    def post_process(self, preds):
        """
        Does post processing on segmentation predictions
        :param preds: (torch tensor: shape(B, D, H, W)) - tensor containing integer predictions for labels
        :return: post processed image
        """
        batch_size = preds.shape[0]
        processed_masks = []

        for batch_idx in range(batch_size):
            # Get single batch
            current_mask = preds[batch_idx]

            # Apply post-processing transforms
            processed = self.keep_largest(current_mask)
            processed = self.fill_holes(processed)
            processed = self.remove_small(processed)

            # Add processed mask to list
            processed_masks.append(processed)

        # Stack processed masks back into batch
        return torch.stack(processed_masks, dim=0)



    def train_epoch(self, labeled_loader, unlabeled_loader, unlabeled_train):
        """
        Train for 1 epoch on labeled data and unlabeled data if bool unlabeled_train is True
        :param labeled_loader: (Dataloader) - Dataloader for labeled data
        :param unlabeled_loader: (Dataloader) - Dataloader for unlabeled data
        :param unlabeled_train: (bool) - Should train on unlabeled data?
        :return: loss
        """

        self.model.train()
        total_loss = 0

        # Train on labeled data
        for i, batch in enumerate(labeled_loader):
            print(f"\rTraining labeled batch {i + 1}/{len(labeled_loader)}", end="")

            images = batch['image'].to(self.device)  # Shape: (B, 1, D, H, W)
            labels = batch['label'].to(self.device)  # Shape: (B, D, H, W)

            self.optimizer.zero_grad()
            outputs = self.model(images)  # Shape: (B, 9, D, H, W)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Train on pseudo-labeled data
        if unlabeled_loader is not None and unlabeled_train:
            for i, pseudo_batch in enumerate(unlabeled_loader):
                print(f"\rTraining unlabeled batch {i + 1}/{len(unlabeled_loader)}", end="")

                images = pseudo_batch['image'].to(self.device)  # Shape: (B, 1, D, H, W)
                labels = pseudo_batch['label'].to(self.device)  # Shape: (B, D, H, W)
                confidence = pseudo_batch['confidence'].to(self.device)  # Shape: (B, D, H, W)
                mask = pseudo_batch['mask'].to(self.device)  # Shape: (B, D, H, W)

                self.optimizer.zero_grad()
                outputs = self.model(images)  # Shape: (B, 9, D, H, W)

                loss = self.criterion(outputs, labels)
                loss = (loss * confidence * mask).mean()  # Apply confidence and mask
                loss = loss * 0.8  # Scale down pseudo-labeled loss - TODO: Make this dynamic
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (len(labeled_loader) + (len(unlabeled_loader) if unlabeled_loader else 0))

    def validate(self, val_loader, output_dir, epoch):
        """
        Validation stage with Dice score and regular loss
        """
        self.model.eval()
        total_loss = 0
        all_dice_scores = []
        all_post_dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                if not all(batch['is_labeled']):
                    # Somehow got unlabelled data in our val data
                    continue

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Calculate Dice score for each class
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                dice_scores = []
                for class_idx in range(self.n_classes):
                    dice = self.calculate_dice_score(
                        (preds == class_idx).float(),
                        (labels == class_idx).float()
                    )
                    dice_scores.append(dice)

                if epoch % 5 == 0:  # Apply post-processing every 5 epochs
                    post_preds = self.post_process(preds)

                    # Get New Dice Scores
                    dice_scores_post_processed = []
                    for class_idx in range(self.n_classes):
                        dice = self.calculate_dice_score(
                            (post_preds == class_idx).float(),
                            (labels == class_idx).float()
                        )
                        dice_scores_post_processed.append(dice)

                    all_post_dice_scores.append(dice_scores_post_processed)
                all_dice_scores.append(dice_scores)
                total_loss += loss.item()

        # Calculate average metrics
        avg_dice = np.mean(all_dice_scores, axis=0)
        avg_val_loss = total_loss / len(val_loader)

        # Get post metrics
        if epoch % 5 == 0:
            post_avg_dice = np.mean(all_post_dice_scores, axis=0)
            print("\nClass-wise Dice Scores (Raw → Post-processed):")
            for i, (og_dice, post_dice) in enumerate(zip(avg_dice, post_avg_dice)):
                diff = post_dice - og_dice
                print(f"Class {i + 1}: {og_dice:.4f} → {post_dice:.4f} "
                      f"({'↑' if diff > 0 else '↓'}{abs(diff):.4f})")


        avg_dice_all = np.mean(avg_dice)
        self.all_dice_scores.append(avg_dice_all)
        # Save best model
        if avg_dice_all > self.best_dice:
            self.best_dice = avg_dice_all
            self.best_dice_epoch = epoch
            self.save_checkpoint(output_dir, epoch, avg_val_loss, avg_dice_all, avg_dice)

            # Early stopping
            self.epoch_since_last_improvement = 0
        else:
            self.epoch_since_last_improvement += 1

        return avg_val_loss, avg_dice_all, avg_dice

    @staticmethod
    def calculate_dice_score(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def save_checkpoint(self, output_dir, epoch, val_loss, avg_dice, class_dice):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'dice_score': avg_dice,
            'epoch': epoch + 1,
            'class_dice': class_dice.tolist(),
            'all_dice_scores': self.all_dice_scores
        }
        torch.save(checkpoint, Path(output_dir) / 'best_model.pth')
        print(f"\nSaved new best model with average Dice score: {avg_dice:.4f}")

    def train(self, labeled_loader, unlabeled_loader, val_loader, num_epochs, output_dir, epoch_start=0):
        """
        Main training loop
        :param labeled_loader:
        :param unlabeled_loader:
        :param val_loader:
        :param num_epochs:
        :param output_dir:
        :param epoch_start:
        :return:
        """
        for epoch in range(epoch_start, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            # Dynamically update confidence threshold to lower throughout training
            self.confidence_threshold = max(0.5, 0.95 - 0.4 * (epoch / num_epochs))

            # Generate pseudo-labels
            unlabeled_train = False
            if unlabeled_loader is not None and epoch > 18 and epoch % 2 == 0:  # Let model train for x epochs on just labelled before using unlabelled then only use unlabelled data every 2 epochs
            # if unlabeled_loader is not None:
                unlabeled_train = True
                self.generate_pseudo_labels(unlabeled_loader)

            # Train
            train_loss = self.train_epoch(labeled_loader, unlabeled_loader, unlabeled_train)
            print(f"\nTraining Loss: {train_loss:.4f}")

            # Validate
            val_loss, avg_dice, class_dice = self.validate(val_loader, output_dir, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            if epoch % 5 == 0:
                print("Class-wise Dice scores:")
                for i, dice in enumerate(class_dice):
                    print(f"Class {i}: {dice:.4f}")
            print(f"Average Dice: {avg_dice:.4f}")

            elapsed = time.time() - start_time

            # Update scheduler
            self.scheduler.step()
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"Epoch completed in {elapsed:.2f}s")

            if self.epoch_since_last_improvement >= self.patience:
                # Early stopping initiated
                print(f"Early stopping due to no model improvement at epoch {epoch} with best Dice {self.best_dice} "
                      f"at epoch {self.best_dice_epoch}")
                break


def train(option=1):
    """
    Main SSL training script
    :param option: which model type to run
    :return:
    """
    from config import SSLTrainingConfig
    # Get augmented images -- Comment/Uncomment if augmented images already made
    augmentation.delete_augmented_images("../data")
    augmentation.main()

    # Set save path
    save_path = ("UNet3D" if option == 0 or option == 1 else "SwinUnetr")

    # Do actual training
    config = SSLTrainingConfig()
    config.output_dir = config.output_dir / save_path  # Set different model saves based on option
    config.output_dir.mkdir(exist_ok=True)  # Make sure output dir exists

    print("Creating dataloaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = create_dataloaders(config)

    print("Initializing model...")

    # ============= OPTION 0 - SMALL UNET (not pretrained) =============
    if option == 0:
        model = get_unet3D()
        model.to(config.device)
        optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
        scheduler = config.scheduler(optimizer, T_max=config.num_epochs/4)
        epoch = 0
        best_dice = 0
        all_dice = []

    # ============= OPTION 1 - SMALL UNET (pretrained local) =============
    elif option == 1:
        model = get_unet3D()
        model, optimizer, scheduler, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                          modelStateDictPath=Path(f"{config.output_dir}/best_model.pth"),
                                                          config=config)
    # ============= OPTION 2 - swin unetr model from online weights =============
    elif option == 2:
        model = load_swin_unetr(num_classes=config.n_classes, pretrained=True)
        model.to(config.device)
        optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
        scheduler = config.scheduler(optimizer, T_max=config.num_epochs/4)
        epoch = 0
        best_dice = 0
        all_dice = []
    # ============= OPTION 3 - swin unetr from local weights =============
    elif option == 3:
        model = load_swin_unetr(num_classes=config.n_classes, pretrained=False)
        model, optimizer, scheduler, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                          modelStateDictPath=Path(
                                                              f"{config.output_dir}/best_model.pth"),
                                                          config=config)
    else:
        raise ValueError("Invalid option for training.")


    # SET CRITERION (2 options currently)
    # criterion = CombinedLoss()
    criterion = CombinedFocalDiceLoss()

    trainer = SimpleSSL(model, criterion, optimizer, scheduler, config.device, n_classes=config.n_classes, data_dir=config.data_dir)
    trainer.best_dice = best_dice
    trainer.all_dice_scores = all_dice
    trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs, config.output_dir, epoch_start=epoch)


def test():
    """
    Test the previously generated models
    :return:
    """
    # STEP 1 - Load untrained model
    # STEP 2 - Load trained model
    # STEP 3 - Test untrained model and trained model on various test scores
    # STEP 4 - Output results and plot graphs (can get all val dice scores during training if you want this graph also)
    pass


if __name__ == '__main__':
    train()
