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
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.ce(pred, target)


def load_pretrained_unet3d(in_channels=1, n_classes=9):
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

def softmax_confidence(predictions):
    """Calculate confidence from the softmax predictions."""
    probabilities = F.softmax(predictions, dim=1)
    confidence, pseudo_labels = torch.max(probabilities, dim=1)
    return confidence, pseudo_labels

class SimpleSSL:
    def __init__(self, model, criterion, optimizer, device, confidence_threshold=0.9, n_classes=9):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.best_val_loss = float('inf')
        self.n_classes = n_classes  # Including background class
        self.best_dice = 0

    def generate_pseudo_labels(self, unlabeled_loader):
        """
        Generate pseudo labels for the data in the unlabeled loader.
        """
        self.model.eval()
        pseudo_labels = []

        with torch.no_grad():
            for i, batch in enumerate(unlabeled_loader):
                print(f"\rProcessing unlabeled batch {i + 1}/{len(unlabeled_loader)}", end="")

                images = batch['image'].to(self.device)  # Shape: (B, 1, D, H, W)
                outputs = self.model(images)  # Shape: (B, 9, D, H, W)

                confidence, labels = softmax_confidence(outputs)  # Shape: (B, D, H, W)

                mask = confidence >= self.confidence_threshold

                pseudo_labels.append({
                    'image': images.cpu(),  # Move to CPU to free VRAM
                    'label': labels.cpu(),
                    'confidence': confidence.cpu(),  # Move to CPU to free VRAM
                    'mask': mask.cpu(),  # Move to CPU to free VRAM
                })

                # Explicitly delete tensors to free GPU memory
                del images, outputs, confidence, labels, mask
                torch.cuda.empty_cache()

        print("\n")
        return pseudo_labels

    def train_epoch(self, labeled_loader, unlabeled_data, unlabeled_loader):
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
        if unlabeled_data is not None:
            for i, pseudo_batch in enumerate(unlabeled_data):

                print(f"\rTraining unlabeled batch {i + 1}/{len(unlabeled_loader)}", end="")

                images = pseudo_batch['image'].to(self.device)  # Shape: (B, 1, D, H, W)
                labels = pseudo_batch['label'].to(self.device)  # Shape: (B, D, H, W)
                confidence = pseudo_batch['confidence'].to(self.device)  # Shape: (B, D, H, W)
                mask = pseudo_batch['mask'].to(self.device)  # Shape: (B, D, H, W)
                if not any(mask):
                    continue  # We have no confident guesses

                self.optimizer.zero_grad()
                outputs = self.model(images)  # Shape: (B, 9, D, H, W)

                loss = self.criterion(outputs, labels)
                loss = (loss * confidence * mask).mean()  # Apply confidence and mask
                loss = loss * 0.8  # Scale down pseudo-labeled loss
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
        }
        torch.save(checkpoint, Path(output_dir) / 'best_model.pth')
        print(f"\nSaved new best model with average Dice score: {avg_dice:.4f}")

    def train(self, labeled_loader, unlabeled_loader, val_loader, num_epochs, output_dir):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            # Generate pseudo-labels
            unlabeled_data = None
            if unlabeled_loader is not None and epoch > 10:  # Let model train for 5 epochs on just labelled before using unlabelled
                print("Generating pseudo-labels...")
                unlabeled_data = self.generate_pseudo_labels(unlabeled_loader)

            # Train
            train_loss = self.train_epoch(labeled_loader, unlabeled_data, unlabeled_loader)
            print(f"Training Loss: {train_loss:.4f}")

            # Validate
            val_loss, avg_dice, class_dice = self.validate(val_loader, output_dir, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            print("Class-wise Dice scores:")
            for i, dice in enumerate(class_dice):
                print(f"Class {i}: {dice:.4f}")
            print(f"Average Dice: {avg_dice:.4f}")

            elapsed = time.time() - start_time
            print(f"Epoch completed in {elapsed:.2f}s")

def main():
    # Get augmented images
    # augmentation.main()

    # Do actual training
    config = SSLTrainingConfig()
    config.output_dir.mkdir(exist_ok=True)

    print("Creating dataloaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = create_dataloaders(config)

    print("Initializing model...")
    model = load_pretrained_unet3d(1, config.n_classes).to(config.device)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = SimpleSSL(model, criterion, optimizer, config.device, n_classes=config.n_classes)
    trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs, config.output_dir)


if __name__ == '__main__':
    main()
