# MSCNet:Cross-modal Structural Consistency for Ocean Eddy Detection
This is the implementation of paper: Cross-modal Structural Consistency for Ocean Eddy Detection.
> ⚠ **Note:** Note: The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.
---
## Datasets
Experiments on a multimodal ocean eddy [dataset](https://github.com/huanglab-research/Mesoscale-Eddy-Dataset) comprising sea level anomaly (SLA), sea surface height (SSH), sea surface temperature (SST) and chlorophyll-a concentration (CHL) observations.

## Installation -- Compiling CUDA operators
* The code are built upon the official [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR) repository.

```sh
conda create -n mscnet python=3.9 --y
conda activate mscnet
bash install.sh
```

## Trained Model
* Changed the pretrained model path in DQ.sh
```sh
CUDA_VISIBLE_DEVICES=5,6,7 bash scripts/DQ.sh /path/to/your/dataset
```
