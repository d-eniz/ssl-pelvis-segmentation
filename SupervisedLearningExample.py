"""
An example file to show how to use stuff already built to train fully supervised

Might not be as flexible as you want, but just an example script to show you how stuff works

**Note** paths to imported files will need to change if put in a different directory!!
"""

from Semi_Supervised.SemiSupervisedLearning import SimpleSSL, load_swin_unetr, CombinedLoss
from Semi_Supervised.config import SSLTrainingConfig
from Semi_Supervised.data_loaders import create_dataloaders

import torch

# Define the config to use for the script
config = SSLTrainingConfig()

config.output_dir.mkdir(exist_ok=True)  # Make sure output dir exists

# Create the dataloaders
labeled_loader, unlabeled_loader, val_loader, test_loader = create_dataloaders(config)
unlabeled_loader = None  # Don't want to use unlabeled data for supervised learning

# Define the model and set variables
model = load_swin_unetr(num_classes=config.n_classes, pretrained=True)
model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # Change this however you want
scheduler = config.scheduler(optimizer, T_max=config.num_epochs / 4)
epoch = 0
best_dice = 0
all_dice = []

# Set up loss function
criterion = CombinedLoss()  # Could also just use nn.CrossEntropyLoss() if don't want to include DICE in loss

# Set up the trainer script
trainer = SimpleSSL(model, criterion, optimizer, scheduler, config.device, n_classes=config.n_classes, data_dir=config.data_dir)

# Set the trainer to train
trainer.train(labeled_loader, unlabeled_loader, val_loader, config.num_epochs, config.output_dir, epoch_start=epoch)
# This will save the best model to the config output dir every time a new best DICE is reached
