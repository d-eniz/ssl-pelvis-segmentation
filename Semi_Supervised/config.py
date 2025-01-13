from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class SSLTrainingConfig:
    """Configuration for training pipeline."""
    # Data parameters
    data_dir: Path = Path(__file__).parent / "../data"
    output_dir: Path = Path(__file__).parent / "output"
    target_size: Tuple[int, int, int] = (160, 160, 32)
    n_classes: int = 9

    # Training parameters
    batch_size: int = 3
    num_epochs: int = 100
    learning_rate: float = 1e-3
    labeled_ratio: float = 0.2
    confidence_threshold: float = 0.9

    # Dataset splits
    train_test_val_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42

    # Optimizer and Scheduler
    optimizer: torch.optim = torch.optim.AdamW
    weight_decay: float = 1e-4

    scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    # Hardware parameters
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0
    pin_memory: bool = True
