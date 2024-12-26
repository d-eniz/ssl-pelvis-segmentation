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
from config import SSLTrainingConfig

from data_loaders import create_dataloaders
import augmentation

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


def load_pretrained_weights(model: torch.nn.Module, modelStateDictPath: Path, device: torch.device, lr: float) -> \
        (torch.nn.Module, torch.optim, int, float, list[float]):
    """
    Load pretrained weights onto a model
    :param model: model to have weights loaded onto
    :param modelStateDictPath: Path to .pth / .pt file containing weights
    :param device: Training config
    :param lr: Learning rate for optimizer
    :return: model, optimizer, epoch, best dice score
    """

    # Load model onto device
    model.to(device)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return model, optimizer, int(epoch), dice_score, all_dice_scores


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
    Loads the pretrained swin_unetr_btcv_segmentation model from Monai
    :param: num_classes: Number of output classes
    :return: model - loaded model with pretrained weights
    """
    from monai.bundle import ConfigParser, load
    import torch
    from pathlib import Path
    from monai.networks.blocks import UnetOutBlock

    # NOTE: AUTOMATICALLY RESCALES THE INPUT SIZE AS LONG AS MULTIPLE OF 32
    model_dict = load("swin_unetr_btcv_segmentation")

    torch_hub_dir = Path(torch.hub.get_dir())
    config_path = next(torch_hub_dir.glob("**/swin_unetr_btcv_segmentation/configs/train.json"))

    config = ConfigParser()
    config.read_config(config_path)
    net = config.get_parsed_content("network_def", instantiate=True)

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
    def __init__(self, model, criterion, optimizer, device, confidence_threshold=0.9, n_classes=9, data_dir=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.best_val_loss = float('inf')
        self.n_classes = n_classes  # Including background class
        self.best_dice = 0
        self.all_dice_scores = []
        self.data_dir = data_dir  # For saving psuedo labels to during an epoch for RAM memory efficiency

    def generate_pseudo_labels(self, unlabeled_loader):
        """
        Generate pseudo labels for the data in the unlabeled loader.

        Can implement this in future saving to disk
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

        print("\n")

    def train_epoch(self, labeled_loader, unlabeled_loader):
        """
        Modified training epoch to handle the new pseudo-label format.
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
        if unlabeled_loader is not None:
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

        print("\n")
        return total_loss / (len(labeled_loader) + (len(unlabeled_loader) if unlabeled_loader else 0))

    def validate(self, val_loader, output_dir, epoch):
        """Validation with additional metrics"""
        self.model.eval()
        total_loss = 0
        all_dice_scores = []

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

                all_dice_scores.append(dice_scores)
                total_loss += loss.item()

        # Calculate average metrics
        avg_dice = np.mean(all_dice_scores, axis=0)
        avg_val_loss = total_loss / len(val_loader)

        # Save best model
        avg_dice_all = np.mean(avg_dice)
        self.all_dice_scores.append(avg_dice_all)
        if avg_dice_all > self.best_dice:
            self.best_dice = avg_dice_all
            self.save_checkpoint(output_dir, epoch, avg_val_loss, avg_dice_all, avg_dice)

        return avg_val_loss, avg_dice_all, avg_dice

    def calculate_dice_score(self, pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def save_checkpoint(self, output_dir, epoch, val_loss, avg_dice, class_dice):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'dice_score': avg_dice,
            'epoch': epoch + 1,
            'class_dice': class_dice.tolist(),
            'all_dice_scores': self.all_dice_scores
        }
        torch.save(checkpoint, Path(output_dir) / 'best_model.pth')
        print(f"\nSaved new best model with average Dice score: {avg_dice:.4f}")

    def train(self, labeled_loader, unlabeled_loader, val_loader, num_epochs, output_dir, epoch_start=0):
        for epoch in range(epoch_start, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            # Dynamically update confidence threshold to lower throughout training
            self.confidence_threshold = max(0.5, 0.95 - 0.4 * (epoch / num_epochs))

            # Generate pseudo-labels
            if unlabeled_loader is not None and epoch > 18 and epoch % 2 == 0:  # Let model train for x epochs on just labelled before using unlabelled then only use unlabelled data every 2 epochs
            # if unlabeled_loader is not None:
                self.generate_pseudo_labels(unlabeled_loader)

            # Train
            train_loss = self.train_epoch(labeled_loader, unlabeled_loader)
            print(f"Training Loss: {train_loss:.4f}")

            # Validate
            val_loss, avg_dice, class_dice = self.validate(val_loader, output_dir, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            if epoch % 5 == 0:
                print("Class-wise Dice scores:")
                for i, dice in enumerate(class_dice):
                    print(f"Class {i}: {dice:.4f}")
            print(f"Average Dice: {avg_dice:.4f}")

            elapsed = time.time() - start_time
            print(f"Epoch completed in {elapsed:.2f}s")


def main():
    # Get augmented images
    # augmentation.delete_augmented_images("../data")
    # augmentation.main()

    # Do actual training
    config = SSLTrainingConfig()
    config.output_dir.mkdir(exist_ok=True)
    config.batch_size = 2
    config.num_workers = 2

    print("Creating dataloaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = create_dataloaders(config)

    print("Initializing model...")
    option = 1
    # ============= OPTION 1 - LOCAL PRETRAINED =============
    if option == 1:
        model = get_unet3D()
        model, optimizer, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                          modelStateDictPath=Path(f"{config.output_dir}/best_model.pth"),
                                                          device=config.device, lr=config.learning_rate)
    # ============= OPTION 2 - swin unetr model from online weights =============
    elif option == 2:
        model = load_swin_unetr(num_classes=config.n_classes, pretrained=True)
        model.to(config.device)
        epoch = 0
        best_dice = 0
        all_dice = []
    # ============= OPTION 3 - swin unetr from local weights =============
    elif option == 3:
        model = load_swin_unetr(num_classes=config.n_classes, pretrained=False)
        model, optimizer, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                          modelStateDictPath=Path(
                                                              f"{config.output_dir}/best_model.pth"),
                                                          device=config.device, lr=config.learning_rate)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = CombinedLoss()

    trainer = SimpleSSL(model, criterion, optimizer, config.device, n_classes=config.n_classes, data_dir=config.data_dir)
    trainer.best_dice = best_dice
    trainer.all_dice_scores = all_dice
    trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs, config.output_dir, epoch_start=epoch)


if __name__ == '__main__':
    main()
