import torch
import matplotlib.pyplot as plt
import os

from Semi_Supervised.SemiSupervisedLearning import get_unet3D
from Semi_Supervised.data_loaders import create_dataloaders
from Supervised_learning.config import SLTrainingConfig

config = SLTrainingConfig()

model_path = "./Supervised_learning/output/unet_supervised.pth"
device = config.device

_, _, _, test_loader = create_dataloaders(config)

model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

visualization_data = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        visualization_data.append((images.cpu().numpy(), preds, labels_np))


def visualize_predictions(data, output_dir="./Supervised_learning/output/data"):
    os.makedirs(output_dir, exist_ok=True)

    for i, (images, preds, labels) in enumerate(data):
        sample_folder = os.path.join(output_dir, f"sample_{i}")
        os.makedirs(sample_folder, exist_ok=True)

        for j in range(images.shape[0]):
            batch_folder = os.path.join(sample_folder, f"batch_{j}")
            os.makedirs(batch_folder, exist_ok=True)

            img_folder = os.path.join(batch_folder, "image")
            gt_folder = os.path.join(batch_folder, "ground_truth")
            pred_folder = os.path.join(batch_folder, "prediction")

            os.makedirs(img_folder, exist_ok=True)
            os.makedirs(gt_folder, exist_ok=True)
            os.makedirs(pred_folder, exist_ok=True)

            for slice_idx in range(images.shape[-1]):
                img_save_path = os.path.join(img_folder, f"slice_{slice_idx}.png")
                plt.imsave(img_save_path, images[j, 0, :, :, slice_idx], cmap="gray")

                pred_save_path = os.path.join(pred_folder, f"slice_{slice_idx}.png")
                plt.imsave(pred_save_path, preds[j, :, :, slice_idx], cmap="viridis")

                gt_save_path = os.path.join(gt_folder, f"slice_{slice_idx}.png")
                plt.imsave(gt_save_path, labels[j, :, :, slice_idx], cmap="viridis")


visualize_predictions(visualization_data)
