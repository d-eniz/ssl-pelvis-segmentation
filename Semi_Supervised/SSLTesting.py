"""
A script to carry out testing on the models
"""
import os.path
from pathlib import Path

# from config import SSLTrainingConfig
from Semi_Supervised.SemiSupervisedLearning import *

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np


# Outline to show what data is returned
data_outline = [
    {
        "images": plt.subplots(),  # model i example input image, true label image and predicted label image
        "avgDice": float,  # model i avg Dice Score
        "classDice": [float],  # model i class-wise Dice Score
        "avgSSIM": float,  # model i average SSIM score
        "avgPSNR": float  # model i average PSNR score
    }
    # Other metrics implemented later
]

global metrics



def get_predictions(models, test_loader, config):
    """
    Perform predictions on the test set for a model
    :param models:
    :param test_loader:
    :param config:
    :return:
    """

    global metrics

    # Set all models to eval mode
    for model in models:
        model.eval()

    with torch.no_grad():
        scores = [{"dice_scores": [], "SSIM": [], "PSNR": []} for _ in range(len(models) + 1)]
        for batch_num, batch in enumerate(test_loader):

            print(f"\rProcessing batch {batch_num+1}/{len(test_loader)}", end="")

            probs = []

            if not all(batch['is_labeled']):
                # Somehow got unlabelled data in our test data
                continue

            images = batch['image'].to(config.device)
            labels = batch['label'].to(config.device)

            # Get predictions from both models
            for i, model in enumerate(models):
                logits = model(images)

                # Convert logits to probabilities
                prob = F.softmax(logits, dim=1)
                probs.append(prob)

                # Get metrics for one model
                dice_scores, SSIM, PSNR = calculate_metrics(prob, labels, config)

                scores[i]["dice_scores"].append(dice_scores)
                scores[i]["SSIM"].append(SSIM)
                scores[i]["PSNR"].append(PSNR)

                if batch_num == 12:
                    # Set graphs for the first image only
                    graphs = get_graphs(images.cpu().numpy(), labels.cpu().numpy(), prob)
                    metrics[i]["images"] = graphs

            # Ensemble by averaging probabilities
            ensemble_probs = torch.stack(probs, dim=0).mean(dim=0)
            dice_scores, SSIM, PSNR = calculate_metrics(ensemble_probs, labels, config)
            scores[-1]["dice_scores"].append(dice_scores)
            scores[-1]["SSIM"].append(SSIM)
            scores[-1]["PSNR"].append(PSNR)

            if batch_num == 12:
                # Set graphs for the first image only
                graphs = get_graphs(images.cpu().numpy(), labels.cpu().numpy(), prob)
                metrics[-1]["images"] = graphs

        # After all batch calculate average metrics as necessary
        for i, model_scores in enumerate(scores):
            class_dice = np.mean(model_scores["dice_scores"], axis=0)
            metrics[i]["classDice"] = class_dice
            metrics[i]["avgDice"] = np.mean(class_dice)
            metrics[i]["avgSSIM"] = np.mean(model_scores["SSIM"])
            metrics[i]["avgPSNR"] = np.mean(model_scores["PSNR"])


def calculate_metrics(probs, labels, config):
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    def calculate_dice_score(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def calculate_ssim(img1, img2):
        """Calculate SSIM between two images"""
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        # Normalize to [0, 1] if necessary
        img1_min, img1_max = img1.min(), img1.max()
        img2_min, img2_max = img2.min(), img2.max()
        img1 = (img1 - img1_min) / (img1_max - img1_min + 1e-5)
        img2 = (img2 - img2_min) / (img2_max - img2_min + 1e-5)

        # Convert tensors to NumPy
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()

        ssim_values = []
        for i in range(img1_np.shape[0]):  # Batch dimension
            score = ssim(
                img1_np[i],
                img2_np[i],
                data_range=1.0,  # Normalized range
                win_size=11,
                gaussian_weights=True,
                use_sample_covariance=False
            )
            ssim_values.append(score)

        return np.mean(ssim_values)

    def calculate_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Assuming normalized images
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    # Calculate Dice scores
    preds = torch.argmax(probs, dim=1)  # Shape: [batch_size, depth, height, width]
    dice_scores = []
    for class_idx in range(config.n_classes):
        dice = calculate_dice_score(
            (preds == class_idx).float(),
            (labels == class_idx).float()
        )
        dice_scores.append(dice)

    # One-hot encoding for 3D data
    flat_preds = preds.view(-1)  # Flatten the tensor
    pred_one_hot = F.one_hot(flat_preds, num_classes=config.n_classes)  # [N, num_classes]
    pred_one_hot = pred_one_hot.view(*preds.shape, config.n_classes).permute(0, 4, 1, 2, 3).float()

    flat_labels = labels.view(-1)
    label_one_hot = F.one_hot(flat_labels, num_classes=config.n_classes)
    label_one_hot = label_one_hot.view(*labels.shape, config.n_classes).permute(0, 4, 1, 2, 3).float()

    # Calculate SSIM and PSNR
    ssim_scores = []
    psnr_scores = []
    for class_idx in range(config.n_classes):
        ssim_score = calculate_ssim(pred_one_hot[:, class_idx], label_one_hot[:, class_idx])
        psnr_score = calculate_psnr(pred_one_hot[:, class_idx], label_one_hot[:, class_idx])
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    ssim_avg = sum(ssim_scores) / len(ssim_scores)
    psnr_avg = sum(psnr_scores) / len(psnr_scores)

    return dice_scores, ssim_avg, psnr_avg


def get_graphs(image, label, probs):
    """
    Get graphs for axis subplots showing slices of original image, true labels, and predicted labels.
    :param image: (np.ndarray) shape: (B,C,D,H,W) The 3D image data (1,1,160,160,32)
    :param label: (np.ndarray) shape: (B,D,H,W) The 3D true label data (1,160,160,32)
    :param probs: (torch.tensor) shape: (B,Classes,D,H,W) The probability tensor (1,9,160,160,32)
    :return: (plt.Figure) The matplotlib figure containing the subplots
    """
    # Squeeze out batch and channel dimensions for visualization
    image = np.squeeze(image[0])  # Remove batch and channel dims -> (160,160,32)
    label = np.squeeze(label[0])  # Remove batch dim -> (160,160,32)

    # Convert predicted probabilities to predicted label
    preds = torch.argmax(probs, dim=1).cpu().numpy()  # Get class predictions
    preds = np.squeeze(preds[0])  # Remove batch dim -> (160,160,32)

    # Get evenly spaced axial slice indices
    num_slices = image.shape[2]  # Number of axial slices
    slice_indices = np.linspace(0, num_slices - 1, 5, dtype=int)

    # Create the figure and axes for each plot (image, true label, predicted label)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Plot each axial slice
    for i, slice_idx in enumerate(slice_indices):
        # Image slices
        axes[0, i].imshow(image[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f"Image - slice {slice_idx}")
        axes[0, i].axis('off')

        # True label slices
        axes[1, i].imshow(label[:, :, slice_idx], cmap='viridis')
        axes[1, i].set_title(f"True Label - slice {slice_idx}")
        axes[1, i].axis('off')

        # Predicted label slices
        axes[2, i].imshow(preds[:, :, slice_idx], cmap='viridis')
        axes[2, i].set_title(f"Predicted Label - slice {slice_idx}")
        axes[2, i].axis('off')

    plt.tight_layout()
    return fig


def display_model_comparisons(data_outline, model_names, save_path=""):
    """
    Display a comparison table of metrics for multiple models with plots for each model.

    :param data_outline: List of dictionaries containing metrics and images for each model
    :param model_names: List of model names corresponding to data_outline entries
    """
    num_models = len(data_outline)

    # Create a larger figure with more height per row
    fig = plt.figure(figsize=(20, 6 * num_models))

    # Create GridSpec with much more space between rows
    grid = GridSpec(num_models, 2,
                    width_ratios=[1, 2.5],  # Give more space to the image side
                    height_ratios=[1] * num_models,
                    hspace=0.8,  # Much more vertical space between rows
                    wspace=0.2,  # More horizontal space between table and plots
                    figure=fig)

    for i, (model_data, model_name) in enumerate(zip(data_outline, model_names)):
        # Extract metrics
        avg_dice = model_data["avgDice"]
        class_dice = model_data["classDice"]
        avg_ssim = model_data["avgSSIM"]
        avg_psnr = model_data["avgPSNR"]

        # Format class-wise dice scores with line breaks
        class_dice_str = [f"{score:.4f}" for score in class_dice]
        wrapped_dice = '\n'.join([', '.join(class_dice_str[j:j + 3])
                                  for j in range(0, len(class_dice_str), 3)])

        # Table with metrics
        metrics_table_data = [
            ["Metric", "Value"],
            ["Model Name", model_name],
            ["Avg Dice Score", f"{avg_dice:.4f}"],
            ["Class-wise Dice", wrapped_dice],
            ["Avg SSIM", f"{avg_ssim:.4f}"],
            ["Avg PSNR", f"{avg_psnr:.4f}"]
        ]

        # Add table
        ax_table = fig.add_subplot(grid[i, 0])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=metrics_table_data,
            cellLoc="left",
            colWidths=[0.48, 0.65],
            loc="center",
            bbox=[0.03, 0.03, 0.92, 0.92]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.6, 2.8)

        # Adjust row heights, especially for class-wise dice
        for j in range(len(metrics_table_data)):
            for k in range(2):
                cell = table[(j, k)]
                if j == 3:  # Class-wise dice row
                    cell.set_height(cell.get_height() * 2.5)
                    cell._text.set_verticalalignment('center')
                cell.set_text_props(wrap=True)

        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor('#E6E6E6')
            table[(0, j)].set_text_props(weight='bold')

        # Add image subplot
        ax_image = fig.add_subplot(grid[i, 1])

        # Get the existing figure's axes
        source_fig = model_data["images"]

        # Create a 3x5 subplot layout for the visualization
        inner_grid = GridSpecFromSubplotSpec(3, 5, subplot_spec=grid[i, 1],
                                             hspace=0.3, wspace=0.3)

        # Copy each subplot from the source figure to our new figure
        for idx, source_ax in enumerate(source_fig.axes):
            row = idx // 5
            col = idx % 5

            # Create new subplot in our target figure
            new_ax = fig.add_subplot(inner_grid[row, col])

            # Copy the image data
            for im in source_ax.images:
                new_ax.imshow(im.get_array(), cmap=im.get_cmap())

            # Remove axes and labels
            new_ax.set_xticks([])
            new_ax.set_yticks([])

            # Only add minimal labels where needed
            if row == 0 and col == 2:  # Center of top row
                new_ax.set_title("Image", pad=5, fontsize=10)
            elif row == 1 and col == 2:  # Center of middle row
                new_ax.set_title("True Label", pad=5, fontsize=10)
            elif row == 2 and col == 2:  # Center of bottom row
                new_ax.set_title("Predicted", pad=5, fontsize=10)

        # Add a single title for this row
        fig.text(0.5, 0.955 - (i / num_models + 1), f"{model_name}",  # CHANGE NUMBRE HERE TO CHANGE PLOT
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Final layout adjustments
    plt.tight_layout()

    # Add more padding at the top of the figure
    fig.subplots_adjust(top=0.95)
    fig.savefig(save_path)

    fig.show()

def test_SSL(save_path="outputImage"):
    """
    Main test script
    :param: models - list of models to use for testing (preloaded)
    :return:
    """
    global metrics

    def get_unet3D():
        from monai.networks.nets import UNet
        from monai.networks.layers import Norm

        model = UNet(
            spatial_dims=3,  # Use 3D convolutions
            in_channels=1,  # Number of input channels
            out_channels=9,  # Number of output classes
            channels=(32, 64, 128, 256),  # Feature map channels per layer
            strides=(2, 2, 2),  # Down-sampling strides
            num_res_units=2,  # Residual units per layer
            norm=Norm.BATCH  # Batch Normalization
        )
        return model

    # Load whichever pretrained models you want
    models = []
    config = SSLTrainingConfig()
    _, _, val_loader, test_loader = create_dataloaders(config)

    og_path = config.output_dir
    config_sl = config

    # GET SL UNET3D
    model = get_unet3D()
    model_UNET_SL, optimizer, scheduler, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                                                           modelStateDictPath= Path(os.path.join(__file__, "../../Supervised_learning/output/unet_supervised.pth")),
                                                                                           config=config)

    models.append(model_UNET_SL)

    # SSL GET UNET3D
    config.output_dir = og_path / "UNet3D"
    model = get_unet3D()
    model_UNET, optimizer, scheduler, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                                                           modelStateDictPath=Path(
                                                                                               f"{config.output_dir}/best_model.pth"),
                                                                                           config=config)

    models.append(model_UNET)
    models.append(model_UNET)

    # GET SWIN UNETR
    config.output_dir = og_path / "SwinUnetr"
    model = load_swin_unetr(num_classes=config.n_classes, pretrained=False)
    model_swin, optimizer, scheduler, epoch, best_dice, all_dice = load_pretrained_weights(model,
                                                                                           modelStateDictPath=Path(
                                                                                               f"{config.output_dir}/best_model.pth"),
                                                                                           config=config)
    models.append(model_swin)

    metrics = [{} for _ in range(len(models) + 1)]  # plus 1 as last one is the ensemble model

    get_predictions(models, test_loader, config)

    # Format and show results
    model_names = ["UNet Model SL", "UNet Model SSL", "Swin UNETR Model SSL", "Ensemble Predictor"]

    display_model_comparisons(metrics, model_names, save_path)


if __name__ == "__main__":
    test_SSL()

