"""
A script containing datasets and dataloaders for labeled training data, unlabelled training data, val data and test data
"""
import random
import warnings
from pathlib import Path
from typing import Tuple

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler


class PelvicMRDataset(Dataset):
    """
    Dataloader for the Pelvic MR Dataset
    NOTE: Only the training set contains unlabeled data.
    """
    def __init__(self, data_dir, mode='train', target_size=(256, 256, 32), train_test_val_split=(0.8, 0.1, 0.1), seed=42):
        """
        :param data_dir: Directory containing the data.
        :param mode: 'train' | 'val' | 'test' which dataset to load.
        :param target_size: Image shape of mask.
        :param train_test_val_split: Ratio of training, validation, and test data.
        :param seed: Seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        self.train_test_val_split = train_test_val_split
        self.seed = seed

        self.images = []
        self.labels = []
        self.is_labeled = []

        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset images from
        """
        #print(f"Scanning directory: {self.data_dir}")  # Debugging
        image_files = sorted(list(self.data_dir.glob("*_img.nii")))
        label_files = {f.stem.replace("_mask", "") for f in self.data_dir.glob("*_mask.nii")}

        # Debugging
        #print(f"Num images: {len(image_files)} - Num labels: {len(label_files)}")
        #if not image_files or not label_files:
        #    raise ValueError(f"No valid dataset files found in {self.data_dir}")

        labeled_images = [img for img in image_files if img.stem.replace("_img", "") in label_files]
        unlabeled_images = [img for img in image_files if img.stem.replace("_img", "") not in label_files]

        # Shuffle labeled and unlabeled images consistently
        random.seed(self.seed)
        random.shuffle(labeled_images)
        random.shuffle(unlabeled_images)

        # Calculate split indices for labeled data
        total_labeled = len(labeled_images)
        train_split_labeled = int(self.train_test_val_split[0] * total_labeled)
        val_split_labeled = train_split_labeled + int(self.train_test_val_split[2] * total_labeled)

        # Split labeled data
        labeled_train = labeled_images[:train_split_labeled]
        labeled_val = labeled_images[train_split_labeled:val_split_labeled]
        labeled_test = labeled_images[val_split_labeled:]

        # For training, add unlabeled images to the labeled training set
        if self.mode == 'train':
            total_train = labeled_train + unlabeled_images
            print(f"\nTotal length of training data: {len(total_train)} - Labeled: {len(labeled_train)} - Unlabeled: {len(unlabeled_images)}")
            random.shuffle(total_train)  # Shuffle combined training data
            self.images = total_train
        elif self.mode == 'val':
            print(f"Total length of val data {len(labeled_val)}")
            self.images = labeled_val
        elif self.mode == 'test':
            print(f"Total length of test data {len(labeled_test)}")
            self.images = labeled_test
        else:
            raise ValueError("`mode` must be either 'train', 'val', or 'test'.")

        # Set labels and is_labeled flags
        self.labels = [f.with_name(f.name.replace("_img", "_mask")) for f in self.images]
        self.is_labeled = [
            True if img.stem.replace("_img", "") in label_files else False
            for img in self.images
        ]

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)  # Catch torch load warnings

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

            sample = {'image': img, 'is_labeled': self.is_labeled[idx], 'code': str(self.images[idx].stem).strip("_img")}

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

            else:
                # Load pseudo-labels if they exist
                pseudo_label_path = Path(self.data_dir) / f"{sample['code']}_pseudo.pt"
                if pseudo_label_path.exists():
                    pseudo_data = torch.load(pseudo_label_path)

                    # Detach and set requires_grad=False for all tensor data
                    for key, value in pseudo_data.items():
                        if torch.is_tensor(value):
                            pseudo_data[key] = value.detach().requires_grad_(False)

                    sample.update(pseudo_data)

            return sample

    def __len__(self):
        return len(self.images)



def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create labeled, unlabeled, val and test dataloaders
    :param config: (TrainingConfig): Training config data
    :return: labeled_dataloader, unlabeled_dataloader, val_dataloader, test_dataloader
    """
    train_set = PelvicMRDataset(config.data_dir, mode='train', target_size=config.target_size,
                               train_test_val_split=config.train_test_val_split, seed=config.seed)
    val_set = PelvicMRDataset(config.data_dir, mode='val', target_size=config.target_size,
                             train_test_val_split=config.train_test_val_split, seed=config.seed)
    test_set = PelvicMRDataset(config.data_dir, mode='test', target_size=config.target_size,
                              train_test_val_split=config.train_test_val_split, seed=config.seed)

    # Get indices for labeled and unlabeled data
    labeled_indices = [i for i, is_labeled in enumerate(train_set.is_labeled) if is_labeled]
    unlabeled_indices = [i for i, is_labeled in enumerate(train_set.is_labeled) if not is_labeled]

    labeled_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(labeled_indices),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False  # Ensure we don't drop the last incomplete batch
    )

    if len(unlabeled_indices) > 0:
        unlabeled_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(unlabeled_indices),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False  # Ensure we don't drop the last incomplete batch
        )
        # Verify the total number of samples that will be processed
        total_unlabeled_samples = len(unlabeled_loader) * config.batch_size
        if total_unlabeled_samples < len(unlabeled_indices):
            total_unlabeled_samples += len(unlabeled_indices) % config.batch_size
    else:
        unlabeled_loader = None

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    return labeled_loader, unlabeled_loader, val_loader, test_loader