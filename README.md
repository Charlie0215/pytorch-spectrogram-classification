# PyTorch implementation of spectrogram classification

### Dataset Preparation
UrbanSound8k:
1. download urbansound dataset and place in a folder called ```/data/```.
2. 
```
cd ./utils
python utils.py -dataset ravdess
```

### How to reproduce
1. set up the conda environment:
```bash
conda env create -f environment.yml #Create a conda environment named sent_classification.
conda activate sent_classification #Activate the sent_classification environment.
```
2. run:
```
python train.py
```
### Evaludation
Run notebook ```mobile_net_evaluation_metric``` in ```Visualization``` folder.
