import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # ensure correct path to script
import torch
from Semi_Supervised import SemiSupervisedLearning, SSLTesting
from Supervised_learning.SupervisedLearning import SupervisedTrainer
from Supervised_learning.config import SLTrainingConfig
from Supervised_learning.test_model import ModelEvaluator
from pathlib import Path
warnings.filterwarnings("ignore")


# SUPERVISED LEARNING
SL_config = SLTrainingConfig()

SL_trainer = SupervisedTrainer(SL_config)
SL_trainer.train()

# Testing
model_path = SL_config.output_dir / "unet_supervised.pth"

evaluator = ModelEvaluator(SL_config, model_path)
evaluator.test()  # Evaluate on the test set


# SEMI SUPERVISED LEARNING (SSL)

# TRAIN WITH 1:1 labelled to unlabelled ratio
SemiSupervisedLearning.train(option=0, ratio_unlabelled=1)  # Train UNET -- training outputs printed
SemiSupervisedLearning.train(option=2, ratio_unlabelled=1)  # Train SwinUNetr -- training outputs printed
SSLTesting.test_SSL(save_path="1-1_RatioSSL")  # Test model and display results - saves image to cw2/1-1_RatioSSL.png

# TRAIN WITH 1:2 labelled to unlabelled ratio
SemiSupervisedLearning.train(option=0, ratio_unlabelled=2)  # Train UNET -- training outputs printed
SemiSupervisedLearning.train(option=2, ratio_unlabelled=2)  # Train SwinUNetr -- training outputs printed
SSLTesting.test_SSL(save_path="1-2_RatioSSL")  # Test model and display results - saves image to cw2/1-2_RatioSSL.png

