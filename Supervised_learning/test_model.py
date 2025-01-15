import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
#from sklearn.metrics import confusion_matrix
import numpy as np

# Dynamically add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Semi_Supervised.SemiSupervisedLearning import get_unet3D
from Semi_Supervised.data_loaders import create_dataloaders

class ModelEvaluator:
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.device
        print(f"Device: {self.device}")
        self.model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        _, _, _, self.test_loader = create_dataloaders(config)
        self.criterion = nn.CrossEntropyLoss()

    def test(self):
        """
        Evaluate the model on the test dataset.
        """
        test_loss = 0
        dice_scores = []
        iou_scores = []

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                # Calculate Dice Score and IoU for each batch
                dice = self.dice_score(labels, preds)
                iou = self.iou_score(labels, preds)

                dice_scores.append(dice)
                iou_scores.append(iou)

        avg_test_loss = test_loss / len(self.test_loader)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)

        print(f"\nTest Loss: {avg_test_loss:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")

    def dice_score(self, y_true, y_pred):
        """Calculate Dice Score for a batch."""
        smooth = 1e-5
        y_true = y_true.float()
        y_pred = y_pred.float()
        intersection = (y_true * y_pred).sum()
        return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

    def iou_score(self, y_true, y_pred):
        """Calculate IoU for a batch."""
        smooth = 1e-5
        y_true = y_true.float()
        y_pred = y_pred.float()
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    """def confusion_matrix(self):
        Calculate confusion matrix for the test set.
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)"""


if __name__ == "__main__":
    # Load the configuration and model
    from Supervised_learning.config import SLTrainingConfig

    config = SLTrainingConfig()
    model_path = config.output_dir / "unet_supervised.pth"
    
    import torch

    model = torch.load(model_path, map_location="cpu")  # Load the model to CPU

    for name, param in model.items():
        print(f"{name}: {param.size()}")  # This prints the shape of the weight tensors

    

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    evaluator = ModelEvaluator(config, model_path)
    evaluator.test()  # Evaluate on the test set
    #evaluator.confusion_matrix()  # Optional: Display confusion matrix

"""
import torch
import torch.nn as nn
import sys
import os
import numpy as np
from pathlib import Path

# Dynamically add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Semi_Supervised.data_loaders import create_dataloaders

# Directly define the model architecture
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def load_trained_unet(in_channels=1, n_classes=9):
    
    Define the exact UNet architecture used during training.
    
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        channels=(32, 64, 128, 256, 512),  # Match the training architecture
        strides=(2, 2, 2, 2),              # Down-sampling strides
        num_res_units=2,                   # Number of residual units
        norm=Norm.BATCH                    # Batch Normalization
    )
    return model

class ModelEvaluator:
    def __init__(self, config, model_path):
        self.config = config
        self.device = config.device
        print(f"Device: {self.device}")

        # Initialize the model
        self.model = load_trained_unet(in_channels=1, n_classes=config.n_classes).to(self.device)


        # Load the model weights
        #state_dict = torch.load(model_path, map_location=self.device)
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)

        # Remove unexpected keys or handle mismatched layers
        self.model.load_state_dict(state_dict, strict=False)

        #self.model.load_state_dict(state_dict)
        self.model.eval()

        # Create the test data loader
        _, _, _, self.test_loader = create_dataloaders(config)
        self.criterion = nn.CrossEntropyLoss()

    def test(self):
        Evaluate the model on the test dataset.
        test_loss = 0
        dice_scores = []
        iou_scores = []

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                # Calculate Dice Score and IoU for each batch
                dice = self.dice_score(labels, preds)
                iou = self.iou_score(labels, preds)

                dice_scores.append(dice)
                iou_scores.append(iou)

        avg_test_loss = test_loss / len(self.test_loader)
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)

        print(f"\nTest Loss: {avg_test_loss:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")

    def dice_score(self, y_true, y_pred):
        Calculate Dice Score for a batch.
        smooth = 1e-5
        y_true = y_true.float()
        y_pred = y_pred.float()
        intersection = (y_true * y_pred).sum()
        return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

    def iou_score(self, y_true, y_pred):
        Calculate IoU for a batch.
        smooth = 1e-5
        y_true = y_true.float()
        y_pred = y_pred.float()
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        return (intersection + smooth) / (union + smooth)

if __name__ == "__main__":
    # Load the configuration and model
    from Supervised_learning.config import SLTrainingConfig

    config = SLTrainingConfig()
    model_path = config.output_dir / "unet_supervised.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Display the model's weight tensor shapes
    state_dict = torch.load(model_path, map_location="cpu")
    print("Model State Dictionary (Shapes):")
    for name, param in state_dict.items():
        print(f"{name}: {param.size()}")

    # Initialize the evaluator
    evaluator = ModelEvaluator(config, model_path)

    # Evaluate on the test set
    evaluator.test()


import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Semi_Supervised.SemiSupervisedLearning import get_unet3D
from Semi_Supervised.data_loaders import create_dataloaders
from Supervised_learning.config import SLTrainingConfig

def test_model():
    # Load configuration
    config = SLTrainingConfig()
    device = config.device
    
    print(f"Using device: {device}")

    # Load test data loader
    _, test_loader, _, _ = create_dataloaders(config)

    # Initialize the model
    model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(device)
    
    model_path = f"{config.output_dir}/unet_supervised.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    # Load the state dictionary with strict=False to ignore mismatched keys
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False, ignoring mismatched keys.")
    except RuntimeError as e:
        print("Error while loading model state_dict:", e)
        raise

    # Set the model to evaluation mode
    model.eval()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model()
"""
