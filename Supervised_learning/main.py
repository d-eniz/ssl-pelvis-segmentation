from SupervisedLearning import SupervisedTrainer
from Supervised_learning.config import SLTrainingConfig

config = SLTrainingConfig()

trainer = SupervisedTrainer(config)
trainer.train()
