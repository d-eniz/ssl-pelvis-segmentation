# Semi-Supervised
### Current methodologies, techniques and instructions


## Data Location

Data for this script currently set in `config` class at top of script
**NOTE** This will need to move to specified `cw2/data` dir as specified in guidance sheet.

To fit with marker expected structure, currently assumes repository root is cw2 folder.

## Usage

- `data_loaders.py`:
  - `create_dataloaders()`: Returns: Labeled dataloader, unlabeled loader, val loader and test loader


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

## Current Approach

- Augment labeled images to create new labeled images (current double)
- Augmentations are a random rotation +- 15deg and brightness adjustment pow(0.9 - 1.1)
- Training: Trains for 20 epochs fully supervised then runs semi supervised every other epoch
- Loss is based 50/50 on training loss and dice score and updated accordingly
- Loss is weighted by 0.8 for unlabelled images (so as not to overfit to noise in unlabelled)
- Confidence scaling is applied so that loss for each voxel is weighted according to the models confidence on that voxel
- Only pixels the model is confident about (confidence level set dynamically by `max(0.5, 0.95 - 0.4 * (epoch / num_epochs))`)
- Dice score is calculated with (1e-5 smoothing)
- Intermediate pseudo labels are saved to files as .pt to save RAM usage
- Image size changed to 160x160x32 for Swin Unetr (Needs to be multiple of 32) (Image size massively affects memory usage)
- Models are saved for every new best DICE score and can be reloaded from best model checkpoint (with previous dice scores)
- 471 Train labeled, 58 validation and 60 Test. All validation and test are labelled always!
- Using Swin Unetr model (See Monai model zoo) which is a transformer based UNet architecture
- Validation set used as completely separate data set and validation loss and validation DICE calculated every epoch (class-wise DICE printed every 5 epochs)
- Model output layer is changed from classifying 14 organs to 9 (8 organs + background) for our images. (This layer is reset to needs to be retrained).
- Adam optimizer used with lr=1e-3

Potential Improvements:
- More augmentations
- Better augmentations
- Alternative SSL to just augmentations
- Dynamic scaling of labelled vs unlabeled loss weighting
- Dynamic/different learning rate (Hyper Param tuning)
- BENCHMARKING: Benchmarking original model, supervised trained model, SSL trained model on test set
- Class-Balanced Cross-Entropy Loss - Struggles to predict class 8 (0.6863) and 4 and 7 (~0.75) so can weight losses to more heavily emphasise these classes
- Focal Loss
- Multiple Metrics - structural similarity, IoU, peak signal to noise, DICE

## Best Results so Far:

- DICE: 0.859668 Semi Supervised
- DICE: 0.856 Full Supervised - 20 epochs

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


