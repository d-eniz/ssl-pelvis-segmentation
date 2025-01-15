# mphy0041-pelvis-segmentation


MPHY0041 Medical Imaging Coursework 2 - Instructions

Uses PyTorch conda environment.

#### REQUIRED MODULES:
- monai (https://pypi.org/project/monai/)
- requests

#### Installed by
```shell
pip install monai[einops,skimage,matplotlib]
pip install requests
```

#### Expects data in folder `cw2/data`

### TO RUN ALL CODE:
(Both supervised learning and semi supervised learning)

1. create conda environment and activate
2. install required modules
3. cd to directory containing the code (cw2) and make sure it also has the data in (cw2/data)
4. run: `python main.py`




---
#### Full Run shell:
```shell
conda create -n mphy0041-cw2-pt -c conda-forge pytorch=2.4 torchvision=0.14 nibabel=5.3
conda activate mphy0041-cw2-pt

pip install monai[einops, skimage, matplotlib]
pip install requests


cd path/to/this_group/cw2  # CHANGE THIS TO ACTUAL PATH CONTAINING THIS cw2 and cw2/data

python main.py
```

---

The actual sbatch command used to run the full script for reference:

*Run on UCL DIAS HPC cluster*

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



Ran via:
```
sbatch ML_RUN_CPU.sh
```