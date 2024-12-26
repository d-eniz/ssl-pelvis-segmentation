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

SEMI SUPERVISED:

All data in our dataset is labelled!

So we can:
- do as the other report (70%) did and use unsupervised learning as part of a Y-Net autoencoder
- augment just original images and add these to the dataset as additional unlabelled images
- Train a GAN on the labelled images in order to generate new images
- Hybrid approach of both

## Packages


To get GPU support on compute node, currently have to pip install/conda install:
```shell
conda install pytorch-cuda=12.1 -c pytorch -c nvidia
```

Models can run on CPU so doesn't need to be one of our 3 pip installs, but can just be used for training models.

For current testing:

```shell
pip install monai~=1.4.0
pip install monai[einops]
pip install requests
```

## Testing

1. Train using SSL on not pretrained model - evaluate and see how it does
2. Train using SSL on pretrained model - evaluate and see how it does
3. Compare the 2 models
- Try different types of SSL
- Try different augmentation techniques

## Code versions:


