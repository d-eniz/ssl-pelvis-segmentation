import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
#from dataset import create_dataloaders
from model import get_unet3D
#from config import TrainingConfig
from Semi_Supervised.data_loaders import create_dataloaders
from Semi_Supervised.config import SSLTrainingConfig as TrainingConfig

# Load Configuration
config = TrainingConfig()

# Create DataLoaders
labeled_loader, _, val_loader, _ = create_dataloaders(config)

# Initialize Model
device = config.device
#model = UNet(n_classes=config.n_classes).to(device)
model = get_unet3D(in_channels=1, n_classes=config.n_classes).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = config.scheduler(optimizer, T_max=config.num_epochs)

# Training Loop
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    
    for batch in labeled_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(labeled_loader)
    print(f"Epoch [{epoch+1}/{config.num_epochs}], Training Loss: {avg_loss:.4f}")
    
    # Validation Step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Update Scheduler
    scheduler.step()

# Save the Model
torch.save(model.state_dict(), f"{config.output_dir}/unet_supervised.pth")
