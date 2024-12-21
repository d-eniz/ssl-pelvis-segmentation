# Semi-Supervised
### Current methodologies, techniques and instructions


## Data Location

Data for this script currently set in `config` class at top of script
**NOTE** This will need to move to specified `cw2/data` dir as specified in guidance sheet.

To fit with marker expected structure, currently assumes repository root is cw2 folder.


## Methodologies

Currently investigating **Transfer Learning**, ie loading a pretrained model trained on a separate but related medical
dataset and finetuning the model using SSL on our dataset.

Thinking of using `MONAI` package for loading the models, but not sure on this yet.

## Packages


To get GPU support on compute node, currently have to pip install:
```shell
conda install pytorch-cuda=11.8 -c pytorch -c nvidia
```

Models can run on CPU so doesn't need to be one of our 3 pip installs, but can just be used for training models.

For current testing:

```shell
pip install monai~=1.4.0
```