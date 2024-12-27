# mphy0041-pelvis-segmentation


**NOTE**: An example for using the SSL stuff for supervised learning without having to do anything new is given in [SupervisedLearningExample.py](Semi_Supervised%2FSupervisedLearningExample.py).
You may want more flexibility than what this currently offers, but is a starting point on doing things easily.

### Example `main` script to run everything:

```python
# Do supervised learning however that is implemented

# Semi Supervised Learning

from Semi_Supervised import SemiSupervisedLearning

SemiSupervisedLearning.train()  # Outputs steps to console

SemiSupervisedLearning.test()  # (Yet to be implemented)
```