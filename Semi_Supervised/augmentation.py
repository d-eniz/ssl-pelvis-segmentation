"""
A script to carry out data augmentation to create unlabelled data

Make sure to check images, their size and how they look when adding new transforms as they can mess up for 3D images
"""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import nibabel as nib
from pathlib import Path
from Semi_Supervised.data_loaders import create_dataloaders
from Semi_Supervised.config import SSLTrainingConfig
import os
import numpy as np
from monai.transforms import (
    Compose,
    RandRotate90,
    RandAffine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGibbsNoise,
    RandCoarseDropout,
    SpatialPad,
    CenterSpatialCrop,
    NormalizeIntensity,
    EnsureType,
    EnsureChannelFirst,
    ScaleIntensity
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Get rid of duplicate DLL issues


def get_monai_transforms(prob=0.5, image_size=(160, 160, 32)):
    train_transforms = Compose([
        # Ensure proper channel dimension and type
        # EnsureChannelFirst(),
        EnsureType(),

        # Normalize to [0,1] range
        ScaleIntensity(minv=0.0, maxv=1.0),

        # Spatial transforms that preserve anatomical structures
        RandAffine(
            prob=prob,
            rotate_range=(0.26, 0.26, 0.26),  # about 15 deg
            scale_range=(0.05, 0.05, 0.05),
            translate_range=(5, 5, 5),
            padding_mode='border'
        ),

        # Intensity transforms (all work within [0,1] range)
        RandScaleIntensity(factors=0.1, prob=prob),
        RandShiftIntensity(offsets=0.1, prob=prob),
        RandAdjustContrast(gamma=(0.9, 1.1), prob=prob),

        # Noise and artifact simulation
        RandGibbsNoise(alpha=(0.0, 0.5), prob=prob / 3),

        # Dropout for robustness
        RandCoarseDropout(
            holes=5,
            spatial_size=(1, 1, 1),
            fill_value=0.0,
            prob=prob / 3,
        ),

        # Ensure consistent size
        SpatialPad(spatial_size=image_size),
        CenterSpatialCrop(roi_size=image_size),

        # Final normalization to ensure [0,1] range after all transforms
        ScaleIntensity(minv=0.0, maxv=1.0),

        # Ensure output is torch tensor
        EnsureType(data_type="tensor"),
    ])

    return train_transforms


def apply_augmentations_monai(image, transform):
    result = transform(image)
    return result


def save_augmented_image(code, augmented_image, save_dir):
    """
    Save an augmented image to a .nii file.
    :param code: Path of the original image file
    :param augmented_image: Tensor of the augmented image
    :param save_dir: Directory to save the augmented image
    """
    # Convert to numpy and remove channel dimension
    augmented_image_np = augmented_image.squeeze(0).numpy()


    # Create save path
    save_path = save_dir / f"{code}_aug_img.nii"

    # Save the image
    nib.save(nib.Nifti1Image(augmented_image_np, affine=None), str(save_path))
    print(f"\rSaved augmented image: {save_path}", end="")


def delete_augmented_images(data_path):
    """
    Deletes augmented images in the data directory by checking filenames ending with "_aug_img.nii".
    :param data_path: (Path) Path for the data
    """
    data_path = Path(data_path)

    if not data_path.is_dir():
        raise ValueError(f"The provided path {data_path} is not a valid directory.")

    # Iterate over all files in the directory
    for file in data_path.glob("*_aug_img.nii"):
        try:
            file.unlink()  # Delete the file
            print(f"\rDeleted: {file}", end="")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def main(ratio_unlabelled=1):
    """
    Main augmentation script
    :param ratio_unlabelled: ratio of labelled ot unlabelled data (1:num_unlabelled)
    :return:
    """
    # Step 1 - Load training data
    config = SSLTrainingConfig()
    labeled_dataloader, _, _, _ = create_dataloaders(config=config)

    # Directory to save augmented images
    save_dir = config.data_dir

    transform = get_monai_transforms(image_size=config.target_size)

    # Step 2 - Apply augmentations to data
    for count in range(1):
        for batch in labeled_dataloader:
            images = batch['image']  # Extract images
            paths = batch['code']  # Ensure `paths` contains the original file paths

            for i, image in enumerate(images):
                # Ensure image is in range [0, 1] for augmentation
                image = image.clamp(0, 1)

                # Apply augmentations
                # augmented_image = apply_augmentations(image)
                augmented_image = apply_augmentations_monai(image, transform)

                # Step 3 - Save augmented images back to file
                save_augmented_image(paths[i], augmented_image, save_dir)


if __name__ == "__main__":
    delete_augmented_images(SSLTrainingConfig.data_dir)
    main()


