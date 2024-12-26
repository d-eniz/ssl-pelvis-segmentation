"""
A script to carry out data augmentation to create unlabelled data
"""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import nibabel as nib
from pathlib import Path
from data_loaders import create_dataloaders
from config import SSLTrainingConfig
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Get rid of duplicate DLL issues


def apply_augmentations(image, p=0.5):
    """
    Apply augmentations suitable for 3D medical imaging data.

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

    # # 2. Scaling (2-3%)
    # scale_factor = 0.99 + torch.rand(1) * 0.02  # Random scale between 0.99 and 1.01
    # new_size = [int(s * scale_factor) for s in image.shape[1:]]
    # image = F.interpolate(image.unsqueeze(0),
    #                       size=new_size,
    #                       mode='trilinear',
    #                       align_corners=False).squeeze(0)
    # # Resize back to original size
    # image = F.interpolate(image.unsqueeze(0),
    #                       size=orig_shape[1:],
    #                       mode='trilinear',
    #                       align_corners=False).squeeze(0)
    #
    # # 3. Translation (1-2%)
    # if torch.rand(1) < p:
    # shift = [(torch.rand(1) * 0.04 - 0.02) * s for s in image.shape[1:]]  # ±2% shift
    # grid = torch.ones(1, *image.shape[1:], 3, device=device)
    # grid[..., 0] = grid[..., 0] + shift[0]
    # grid[..., 1] = grid[..., 1] + shift[1]
    # grid[..., 2] = grid[..., 2] + shift[2]
    # image = F.grid_sample(image.unsqueeze(0),
    #                       grid,
    #                       mode='bilinear',
    #                       padding_mode='zeros',
    #                       align_corners=False).squeeze(0)
    #
    # 4. Intensity augmentations
    # Gamma correction (0.9-1.1)
    gamma = 0.9 + torch.rand(1) * 0.2
    image = torch.pow(image, gamma)
    #
    # # Random contrast (0.95-1.05)
    # contrast_factor = 0.95 + torch.rand(1) * 0.1
    # mean = torch.mean(image)
    # image = (image - mean) * contrast_factor + mean
    #
    # # 5. Random Gaussian noise
    # if torch.rand(1) < p:
    #     noise = torch.randn_like(image) * 0.02
    #     image = image + noise
    #     image = torch.clamp(image, 0, 1)

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
    # print(f"Saved augmented image: {save_path}")


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
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def main():
    # Step 1 - Load training data
    config = SSLTrainingConfig()
    config.target_size = (180, 180, 43)
    labeled_dataloader, _, _, _ = create_dataloaders(config=config)

    # Directory to save augmented images
    save_dir = config.data_dir

    # Step 2 - Apply augmentations to data
    for batch in labeled_dataloader:
        images = batch['image']  # Extract images
        paths = batch['code']  # Ensure `paths` contains the original file paths

        for i, image in enumerate(images):
            # Ensure image is in range [0, 1] for augmentation
            image = image.clamp(0, 1)

            # Apply augmentations
            augmented_image = apply_augmentations(image)

            # Step 3 - Save augmented images back to file
            save_augmented_image(paths[i], augmented_image, save_dir)



if __name__ == "__main__":
    delete_augmented_images("../data")
    main()


