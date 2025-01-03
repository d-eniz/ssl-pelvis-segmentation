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


def apply_augmentations_torch(image, p=0.5):
    """
    Apply augmentations suitable for 3D medical imaging data.

    **NOTE** THIS IS THE OLD PYTORCH WAY OF DOING THIS - REPLACED BY MONAI TRANSFORMS
    Args:
        image: Tensor of shape (C, H, W, D) - 3D image volume
        p: Probability of applying each augmentation

    Returns:
        Augmented image tensor of the same shape
    """
    if image.ndim != 4:
        raise ValueError("Input image must be a 4D tensor of shape (C, H, W, D).")

    # Store original shape
    orig_shape = image.shape
    device = image.device

    # Convert to (C, D, H, W) for 3D operations
    image = image.permute(0, 3, 1, 2)

    # # 1. Random rotation (max 10 degrees) - implemented manually for 3D
    angles = (torch.rand(3) * 10 - 5) * np.pi / 180  # Convert to radians

    # Create affine matrix for rotation
    cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
    cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
    cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

    # Rotation matrices
    rot_x = torch.tensor([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]], device=device)

    rot_y = torch.tensor([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]], device=device)

    rot_z = torch.tensor([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]], device=device)

    rotation_matrix = rot_z @ rot_y @ rot_x

    # Create full affine matrix
    affine_matrix = torch.eye(4, device=device)
    affine_matrix[:3, :3] = rotation_matrix

    # Create grid for sampling
    grid = F.affine_grid(affine_matrix[:3].unsqueeze(0),
                         image.unsqueeze(0).shape,
                         align_corners=False)

    # Apply transformation with trilinear interpolation
    image = F.grid_sample(image.unsqueeze(0),
                          grid,
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False).squeeze(0)

    # 4. Intensity augmentations
    # Gamma correction (0.9-1.1)
    gamma = 0.9 + torch.rand(1) * 0.2
    image = torch.pow(image, gamma)

    # Convert back to original format (C, H, W, D)
    image = image.permute(0, 2, 1, 3)

    # Ensure output matches input shape exactly
    if image.shape != orig_shape:
        image = F.interpolate(
            image.permute(0, 3, 1, 2).unsqueeze(0),
            size=orig_shape[1:],
            mode='trilinear',
            align_corners=False
        ).squeeze(0).permute(0, 2, 1, 3)

    return image


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


def main():
    # Step 1 - Load training data
    config = SSLTrainingConfig()
    labeled_dataloader, _, _, _ = create_dataloaders(config=config)

    # Directory to save augmented images
    save_dir = config.data_dir

    transform = get_monai_transforms(image_size=config.target_size)

    # Step 2 - Apply augmentations to data
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
    delete_augmented_images("../data")
    main()


