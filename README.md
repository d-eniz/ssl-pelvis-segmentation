# Pelvis MRI Segmentation - Semi-Supervised Deep Learning

This repository contains code for pelvis segmentation using supervised and semi-supervised deep learning techniques. The project leverages MONAI and PyTorch to train 3D U-Net models for medical image segmentation on T2-weighted MRI scans.

This project was developed as coursework for UCL's [Machine Learning for Medical Imaging](https://github.com/YipengHu/MPHY0041) module.

![gif3](https://github.com/d-eniz/mphy0041-pelvis-segmentation/blob/main/Supervised_learning/output/examples/sample_1_batch_1_mode_2.gif?raw=true)

## Methodology

1. **Preprocessing** – MRI scans are loaded and preprocessed using MONAI, including normalization and augmentation.
2. **Supervised Training** – The model is trained with ground truth labels to learn segmentation patterns.
3. **Semi-Supervised Training** – A combination of labeled and unlabeled data is used to improve model generalization.
4. **Inference & Evaluation** – The trained model is evaluated on a test set, with performance metrics computed.

## Dataset

This project uses the **Cross-institution Male Pelvic Structures dataset**, which includes T2-weighted MRI images with annotations for eight anatomical structures. The data was collected from multiple institutions, and the segmentations were manually annotated by biomedical imaging researchers.

You can download the dataset here:
- [Cross-institution Male Pelvic Structures Dataset](https://zenodo.org/records/7013610)

## ⚙Setup & Dependencies

### Required Modules

The following Python packages are required for running the project:

- **PyTorch** (Deep learning framework)
- **Nibabel** (Medical imaging file handling)
- **MONAI** (Medical image processing and augmentation)
- **Einops** (Tensor manipulation)
- **scikit-image** (Image processing)
- **Matplotlib** (Visualization)
- **Requests** (HTTP requests handling)

### Installation

Create a Conda environment and install dependencies:

```shell
conda create -n mphy0041-cw2-pt -c conda-forge pytorch=2.4 torchvision=0.14 nibabel=5.3
conda activate mphy0041-cw2-pt

pip install monai[einops,skimage,matplotlib]
pip install requests
```

## Running the Code

### Locally (CPU/GPU)

1. Set up the environment (see installation steps)
2. Ensure the [dataset](https://zenodo.org/records/7013610) is placed inside cw2/data/
3. Run the training script:

```shell
cd path/to/cw2  # Replace with actual path
python main.py
```

### 🖥On UCL DIAS HPC Cluster

Use the provided SLURM batch script:

```
sbatch ML_RUN_CPU.sh
```

ML_RUN_CPU.sh:

```shell
#!/bin/bash -l
#SBATCH --partition COMPUTE
#SBATCH --nodes 1
#SBATCH -n12
#SBATCH --mem-per-cpu 40G
#SBATCH --time 24:00:00
#SBATCH --job-name ML_RUN_CPU
#SBATCH --output ML_RUN_CPU.log

XDG_RUNTIME_DIR=""
export port=$(shuf -i8000-9999 -n1)
export node=$(hostname -s)
export user=$(whoami)
export cluster=$(hostname -f | awk -F"." '{print $2}')

# Load Conda and activate environment
eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
conda create -n mphy0041-cw2-pt -c conda-forge pytorch=2.4 torchvision=0.14 nibabel=5.3
conda activate mphy0041-cw2-pt


pip install monai[einops,skimage,matplotlib]
pip install requests

cd /home/xzcapbel/MedicalPhysics/cw2

python main.py
```

## Usage Notice

This project is a coursework submission and is for display purposes only. It is not intended for commercial or production use. For research or development purposes, please refer to the official dataset page for proper usage and licensing.

## Project Contributors

This project was developed by:

- [Ben Elliott](https://github.com/Ben-elliot27)
- [Deniz Terzioglu](https://github.com/d-eniz)
- [George Doumenis Ramos](https://github.com/georgedoumenisramos)
- [Kajan Subakannan](https://github.com/KajanGH)
- [Pego Letebe](https://github.com/PgStar1)
- [Peniel Chew](https://github.com/penielc173)
