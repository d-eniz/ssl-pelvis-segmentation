import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # ensure correct path to script
from Semi_Supervised import SemiSupervisedLearning, SSLTesting


# SUPERVISED LEARNING



# SEMI SUPERVISED LEARNING (SSL)
# TRAIN WITH 1:1 labelled to unlabelled ratio
SemiSupervisedLearning.train(option=0, ratio_unlabelled=1)  # Train UNET -- training outputs printed
SemiSupervisedLearning.train(option=2, ratio_unlabelled=1)  # Train SwinUNetr -- training outputs printed
SSLTesting.test_SSL()  # Test model and display results

# TRAIN WITH 1:2 labelled to unlabelled ratio
SemiSupervisedLearning.train(option=0, ratio_unlabelled=2)  # Train UNET -- training outputs printed
SemiSupervisedLearning.train(option=2, ratio_unlabelled=2)  # Train SwinUNetr -- training outputs printed
SSLTesting.test_SSL()  # Test model and display results

