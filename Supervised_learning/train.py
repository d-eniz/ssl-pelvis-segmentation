"""
THIS SCRIPT IS NO LONGER USED.
USE SupervisedLearning.py
AND main.py INSTEAD.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# from dataset import create_dataloaders
from Semi_Supervised.SemiSupervisedLearning import get_unet3D

# from config import TrainingConfig
from Semi_Supervised.data_loaders import create_dataloaders
from Supervised_learning.config import SLTrainingConfig

import time
import os


# Load Configuration
config = SLTrainingConfig()

# Create DataLoaders
labeled_loader, _, val_loader, _ = create_dataloaders(config)

# Initialize Model
device = config.device
print(f"Device: {device}\n")  # Debugging
# model = UNet(n_classes=config.n_classes).to(device)
model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = config.optimizer(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scheduler = config.scheduler(optimizer, T_max=config.num_epochs)

total_training_time = 0  # Track total training time

# Training Loop
for epoch in range(config.num_epochs):
    start_time = time.time()
    print(f"Epoch [{epoch+1}/{config.num_epochs}]")  # Debugging

    model.train()
    total_loss = 0

    for batch in labeled_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(labeled_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    # Validation Step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    epoch_time = time.time() - start_time
    total_training_time += epoch_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    remaining_time = total_training_time / (epoch + 1) * (config.num_epochs - epoch - 1)
    eta = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_time)
    )
    remaining_hours = int(remaining_time // 3600)
    remaining_minutes = int((remaining_time % 3600) // 60)
    print(f"ETA: {remaining_hours}h{remaining_minutes}m, @ {eta}\n")

    # Update Scheduler
    scheduler.step()

# Save the Model
config.output_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), f"{config.output_dir}/unet_supervised.pth")

print(
f"""
Total training time for {config.num_epochs} epochs: {total_training_time / 3600:.2f} hours
Model saved to {config.output_dir}
"""
    )
