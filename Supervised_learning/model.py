from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_unet3D(in_channels: int = 1, n_classes: int = 9) -> UNet:
    """
    Creates a 3D UNet model using MONAI.

    :param in_channels: Number of input channels (e.g., 1 for grayscale MR images).
    :param n_classes: Number of output segmentation classes.
    :return: MONAI UNet 3D Model.
    """
    model = UNet(
        spatial_dims=3,            # Use 3D convolutions
        in_channels=in_channels,   # Number of input channels
        out_channels=n_classes,    # Number of output classes
        channels=(32, 64, 128, 256, 512),  # Feature map channels per layer
        strides=(2, 2, 2, 2),      # Down-sampling strides
        num_res_units=2,           # Residual units per layer
        norm=Norm.BATCH            # Batch Normalization
    )
    return model
