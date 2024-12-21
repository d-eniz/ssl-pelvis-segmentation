from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms.functional as TF
import nibabel as nib
import numpy as np
import time
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data parameters
    data_dir: Path = Path("../data")
    output_dir: Path = Path("output")
    target_size: Tuple[int, int, int] = (256, 256, 32)
    n_classes: int = 9

    # Training parameters
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 1e-3
    labeled_ratio: float = 0.2
    confidence_threshold: float = 0.9

    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Hardware parameters
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0
    pin_memory: bool = True

class PelvicMRDataset(Dataset):
    def __init__(self, data_dir, mode='train', labeled_ratio=0.2, target_size=(256, 256, 32)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.labeled_ratio = labeled_ratio
        self.target_size = target_size

        self.images = []
        self.labels = []
        self.is_labeled = []
        self._load_dataset()

    def _load_dataset(self):
        image_files = sorted(list(self.data_dir.glob("*_img.nii")))
        total_files = len(image_files)

        indices = range(total_files)   # Maybe randomly permute other than just list
        train_split = int(self.labeled_ratio * total_files)
        val_split = train_split + int(0.15 * total_files)

        if self.mode == 'train':
            selected_files = [image_files[i] for i in indices[:train_split]]
        elif self.mode == 'val':
            selected_files = [image_files[i] for i in indices[train_split:val_split]]
        elif self.mode == 'test':
            selected_files = [image_files[i] for i in indices[val_split:]]
        else:
            raise ValueError(r"`mode` must be either train, val or test")

        self.images = selected_files
        self.labels = [f.with_name(f.name.replace("_img", "_mask")) for f in self.images]
        self.is_labeled = [True] * len(self.images)  # Modify as per requirements

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            img = nib.load(str(self.images[idx])).get_fdata()
            img = torch.tensor(img)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).unsqueeze(0),
                size=self.target_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.unsqueeze(0).float()

            sample = {'image': img, 'is_labeled': self.is_labeled[idx]}

            if self.is_labeled[idx]:
                label = nib.load(str(self.labels[idx])).get_fdata()
                label = torch.tensor(label)
                label = torch.nn.functional.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(),
                    size=self.target_size,
                    mode='nearest'
                ).squeeze(0).long()
                label = label.squeeze(0)  # Remove the singleton channel dimension
                sample['label'] = label

            return sample

    def __len__(self):
        return len(self.images)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.ce(pred, target)


class MetricsCalculator:
    @staticmethod
    def calculate_dice(pred, target, n_classes):
        """Calculate Dice coefficient for each class"""
        dice_scores = []

        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

        for class_idx in range(n_classes):
            pred_class = (pred == class_idx).float()
            target_class = (target == class_idx).float()

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.append(dice.item())

        return dice_scores


def create_dataloaders(config):
    train_set = PelvicMRDataset(config.data_dir, 'train', config.labeled_ratio, config.target_size)
    val_set = PelvicMRDataset(config.data_dir, 'val', config.labeled_ratio, config.target_size)

    labeled_indices = [i for i, labeled in enumerate(train_set.is_labeled) if labeled]
    unlabeled_indices = [i for i, labeled in enumerate(train_set.is_labeled) if not labeled]

    labeled_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(labeled_indices),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    unlabeled_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(unlabeled_indices),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return labeled_loader, unlabeled_loader, val_loader

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


class SimpleSSL:
    def __init__(self, model, criterion, optimizer, device, confidence_threshold=0.9, n_classes=9):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.best_val_loss = float('inf')
        self.n_classes=n_classes
        self.best_dice = 0

    def generate_pseudo_labels(self, unlabeled_loader):
        self.model.eval()
        pseudo_labels = []
        confidence_masks = []

        with torch.no_grad():
            for batch in unlabeled_loader:
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)

                # Get confidence scores and predictions
                confidence, predictions = torch.max(probs, dim=1)

                # Apply confidence thresholding
                mask = confidence > self.confidence_threshold

                # Store results
                pseudo_labels.append({
                    'predictions': predictions,
                    'confidence': confidence,
                    'images': batch['image']
                })
                confidence_masks.append(mask)

        return pseudo_labels, confidence_masks

    def train_epoch(self, labeled_loader, unlabeled_data):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        # Train on labeled data
        for batch_idx, batch in enumerate(labeled_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Train on pseudo-labeled data
        if unlabeled_data:
            pseudo_labels, confidence_masks = unlabeled_data
            for data, mask in zip(pseudo_labels, confidence_masks):
                if not torch.any(mask):  # Skip if no confident predictions
                    continue

                images = data['images'].to(self.device)
                pseudo_label = data['predictions'].to(self.device)
                confidence = data['confidence'].to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Weight loss by confidence scores
                loss = self.criterion(outputs[mask], pseudo_label[mask]) * confidence[mask].mean()
                loss.backward()
                self.optimizer.step()

        return total_loss / len(labeled_loader)

    def validate(self, val_loader, output_dir, epoch):
        """Validation with additional metrics"""
        self.model.eval()
        total_loss = 0
        all_dice_scores = []
        metrics_calc = MetricsCalculator()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if not batch['is_labeled']:
                    continue

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate metrics
                dice_scores = metrics_calc.calculate_dice(outputs, labels, self.n_classes)

                all_dice_scores.append(dice_scores)

                print(f"\rValidating batch {batch_idx + 1}/{len(val_loader)}", end="")

        # Average metrics
        avg_dice = np.mean(all_dice_scores, axis=0)

        print("\nValidation Metrics:")
        for class_idx in range(self.n_classes):
            print(f"Class {class_idx}:")
            print(f"  Dice: {avg_dice[class_idx]:.4f}")

        avg_val_loss = total_loss / len(val_loader)

        # Save best model based on average Dice score
        avg_dice_all = np.mean(avg_dice)
        if avg_dice_all > self.best_dice:
            self.best_dice = avg_dice_all
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'dice_score': avg_dice_all,
                'epoch': epoch + 1,
                'class_dice': avg_dice.tolist(),
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"\nSaved new best model with average Dice score: {avg_dice_all:.4f}")

        return avg_val_loss, avg_dice_all

    def train(self, labeled_loader, unlabeled_loader, val_loader, num_epochs, output_dir):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            # Generate pseudo-labels
            unlabeled_data = None
            if unlabeled_loader is not None:
                print("Generating pseudo-labels...")
                unlabeled_data = self.generate_pseudo_labels(unlabeled_loader)

            # Train
            train_loss = self.train_epoch(labeled_loader, unlabeled_data)
            print(f"Training Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate(val_loader, output_dir, epoch)
            print(f"Validation Loss: {val_loss[0]:.4f} \n Avg Dice: {val_loss[1]}")

            elapsed = time.time() - start_time
            print(f"Epoch completed in {elapsed:.2f}s")


def main():
    config = TrainingConfig()
    config.output_dir.mkdir(exist_ok=True)

    print("Creating dataloaders...")
    labeled_loader, unlabeled_loader, val_loader = create_dataloaders(config)

    print("Initializing model...")
    model = load_pretrained_unet3d(1, config.n_classes).to(config.device)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = SimpleSSL(model, criterion, optimizer, config.device, n_classes=config.n_classes)
    trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs, config.output_dir)

if __name__ == '__main__':
    main()
