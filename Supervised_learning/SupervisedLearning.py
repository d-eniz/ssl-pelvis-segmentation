import time
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Semi_Supervised.SemiSupervisedLearning import get_unet3D
from Semi_Supervised.data_loaders import create_dataloaders

class SupervisedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        print(f"Device: {self.device}\n")
        self.model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = config.optimizer(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = config.scheduler(self.optimizer, T_max=config.num_epochs)
        self.labeled_loader, _, self.val_loader, _ = create_dataloaders(config)
        self.total_training_time = 0

    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch in self.labeled_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.labeled_loader)
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def estimate_time(self, epoch, epoch_time):
        """Estimate the remaining training time."""
        self.total_training_time += epoch_time
        remaining_epochs = self.config.num_epochs - (epoch + 1)
        if remaining_epochs > 0:
            remaining_time = self.total_training_time / (epoch + 1) * remaining_epochs
            eta = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_time))
            remaining_hours = int(remaining_time // 3600)
            remaining_minutes = int((remaining_time % 3600) // 60)
            print(f"Processing Time: {epoch_time:.2f}s")
            print(f"ETA: {remaining_hours}h{remaining_minutes}m, @ {eta}")
        else:
            print("Training complete. No remaining epochs.")

    def train(self):
        """Train the model for all epochs."""
        for epoch in range(self.config.num_epochs):
            start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Timestamp for start of epoch
            print(f"\n[{start_timestamp}]: Epoch {epoch + 1}/{self.config.num_epochs} ({round((epoch + 1)/(self.config.num_epochs)*100)}%)")  # Display Timestamp

            train_loss, epoch_time = self.train_one_epoch(epoch)
            avg_val_loss = self.validate()

            # Display both training and validation losses
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")

            self.estimate_time(epoch, epoch_time)
            self.scheduler.step()

        self.save_model()
        print(
            f"Total training time for {self.config.num_epochs} epochs: "
            f"{self.total_training_time / 3600:.2f} hours"
        )

    def save_model(self):
        """Save the trained model."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = f"{self.config.output_dir}/unet_supervised.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")