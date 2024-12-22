"""
Initial testing script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import warnings
import logging
import os

class Config:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__name__), "../data")  # Path relative to file directory if running from elsewhere
        self.output_dir = "output"
        self.target_size = (256, 256, 32)
        self.n_classes = 9
        self.batch_size = 1
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.labeled_ratio = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 0


def load_pretrained_unet3d(in_channels=1, n_classes=9):
    """
    Load pretrained 3D UNet model.
    Falls back to MONAI UNet if pretrained loading fails.

    Need to implement loading of pretrained weights here
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
    print("Successfully created MONAI UNet!")
    return model

class PelvicMRDataset(Dataset):
    def __init__(self, data_dir, mode='train', labeled_ratio=0.2, target_size=(256, 256, 32)):
        print(f"\nInitializing {mode} dataset...")
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.labeled_ratio = labeled_ratio
        self.target_size = target_size

        self.images = []
        self.labels = []
        self.is_labeled = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Scanning {self.data_dir} for image files...")
        image_files = sorted(list(self.data_dir.glob("*_img.nii")))
        print(f"Found {len(image_files)} image files")

        # Create dataset splits
        total = len(image_files)
        train_idx = int(0.7 * total)
        val_idx = int(0.85 * total)

        if self.mode == 'train':
            files = image_files[:train_idx]
            num_labeled = int(self.labeled_ratio * len(files))
            self.is_labeled = [True] * num_labeled + [False] * (len(files) - num_labeled)
            print(f"Training split:")
            print(f"- Total files: {len(files)}")
            print(f"- Labeled files: {num_labeled}")
            print(f"- Unlabeled files: {len(files) - num_labeled}")
        else:  # val or test
            files = image_files[train_idx:val_idx] if self.mode == 'val' else image_files[val_idx:]
            self.is_labeled = [True] * len(files)
            print(f"{self.mode} split: {len(files)} files (all labeled)")

        # Store all file paths
        self.images = files
        self.labels = [Path(str(f).replace('_img.nii', '_mask.nii')) for f in files]

        # Verify label files exist
        missing_labels = [label for label in self.labels if not label.exists()]
        if missing_labels:
            print(f"Warning: {len(missing_labels)} label files are missing")

    def __getitem__(self, idx):
        try:
            # Suppress nibabel warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)

                # Load and process image
                img = nib.load(str(self.images[idx])).get_fdata()
                img = zoom(img, [t/c for t, c in zip(self.target_size, img.shape)], order=1)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = torch.from_numpy(img).float().unsqueeze(0)

                result = {'image': img, 'is_labeled': self.is_labeled[idx]}

                # Load label if it's a labeled sample
                if self.is_labeled[idx]:
                    label = nib.load(str(self.labels[idx])).get_fdata()
                    label = torch.from_numpy(label).long()
                    result['label'] = label

                return result
        except Exception as e:
            print(f"\nError loading sample at index {idx}: {str(e)}")
            return None

    def __len__(self):
        return len(self.images)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.ce(pred, target)


def create_dataloaders(config):
    # Create datasets
    train_set = PelvicMRDataset(config.data_dir, 'train', config.labeled_ratio)
    val_set = PelvicMRDataset(config.data_dir, 'val')

    # Split training set into labeled and unlabeled
    labeled_indices = [i for i, labeled in enumerate(train_set.is_labeled) if labeled]
    unlabeled_indices = [i for i, labeled in enumerate(train_set.is_labeled) if not labeled]

    # Create whoops loaders
    labeled_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(labeled_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )

    unlabeled_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(unlabeled_indices),
        num_workers=0,
        pin_memory=True
    ) if unlabeled_indices else None

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return labeled_loader, unlabeled_loader, val_loader

class SimpleSSL:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def generate_pseudo_labels(self, unlabeled_loader):
        self.model.eval()
        all_pseudo_labels = []
        all_confidence_masks = []

        print("\nGenerating pseudo-labels...")
        print(f"Processing {len(unlabeled_loader)} unlabeled batches")

        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                if batch is None:
                    continue

                print(f"\rGenerating labels for batch {batch_idx + 1}/{len(unlabeled_loader)}", end="")

                images = batch['image'].to(self.device)

                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)

                confidence, predictions = torch.max(probs, dim=1)
                mask = confidence > 0.9

                all_pseudo_labels.append(predictions)
                all_confidence_masks.append(mask)

        print(f"\nGenerated {len(all_pseudo_labels)} sets of pseudo-labels")
        return all_pseudo_labels, all_confidence_masks

    def train(self, labeled_loader, unlabeled_loader, val_loader, num_epochs):
        print("\nStarting training...")
        print(f"- Device: {self.device}")
        print(f"- Training batches per epoch: {len(labeled_loader)}")
        print(f"- Unlabeled batches: {len(unlabeled_loader) if unlabeled_loader else 0}")
        print(f"- Validation batches: {len(val_loader)}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            # Generate pseudo-labels every 5 epochs
            if epoch % 5 == 0 and unlabeled_loader is not None:
                pseudo_labels, confidence_masks = self.generate_pseudo_labels(unlabeled_loader)

            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(labeled_loader):
                if batch is None:
                    continue

                print(f"\rTraining batch {batch_idx+1}/{len(labeled_loader)}", end="")

                self.optimizer.zero_grad()

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss/max(n_batches, 1)
            print(f"\nTraining loss: {avg_train_loss:.4f}")

            # Validation
            print("\nValidating...")
            self.model.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch is None:
                        continue

                    print(f"\rValidation batch {batch_idx+1}/{len(val_loader)}", end="")

                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss/max(val_batches, 1)
            print(f"\nValidation loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("New best model! Saving checkpoint...")
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f"Best validation loss so far: {best_val_loss:.4f}")

def main():
    # Setup
    config = Config()
    Path(config.output_dir).mkdir(exist_ok=True)

    print("Creating dataloaders...")
    labeled_loader, unlabeled_loader, val_loader = create_dataloaders(config)

    print("\nInitializing model...")
    model = load_pretrained_unet3d(
        in_channels=1,
        n_classes=config.n_classes
    ).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\nStarting training...")
    trainer = SimpleSSL(model, criterion, optimizer, config.device)
    trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs)

    print("Training completed!")

if __name__ == '__main__':
    main()
