# mphy0041-pelvis-segmentation

**NOTE**: An example for using the SSL stuff for supervised learning without having to do anything new is given in [SupervisedLearningExample.py](Semi_Supervised%2FSupervisedLearningExample.py).
You may want more flexibility than what this currently offers, but is a starting point on doing things easily.

## Example `main` script to run everything:

```python
# Supervised Learning

from SupervisedLearning import SupervisedTrainer
from Supervised_learning.config import SLTrainingConfig

SL_config = SLTrainingConfig()

SL_trainer = SupervisedTrainer(SL_config)
SL_trainer.train()

# Semi Supervised Learning

from Semi_Supervised import SemiSupervisedLearning

SemiSupervisedLearning.train()  # Outputs steps to console

SemiSupervisedLearning.test()  # (Yet to be implemented)
```

## Output Examples

Supervised Learning:

![gif1](Supervised_learning\output\examples\sample_1_batch_1_mode_2.gif)

![gif2](Supervised_learning\output\examples\sample_8_batch_1_mode_2.gif)

![gif3](Supervised_learning\output\examples\sample_13_batch_0_mode_2.gif)

![gif4](Supervised_learning\output\examples\sample_16_batch_2_mode_2.gif)