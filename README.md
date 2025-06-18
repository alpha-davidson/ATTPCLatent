This repository implements a self-supervised pretraining strategy for 3D point cloud data using the PointNet architecture. The core idea is to partition each event (represented as a 4D point set with spatial coordinates and charge) into fixed spatial voxels, shuffle the voxel order, and train PointNet to reconstruct the original arrangement. By learning to reverse the shuffling process, the model captures meaningful spatial structures without requiring labeled data. The pretrained model can then be fine-tuned for downstream tasks such as track counting in nuclear physics experiments.

# Packages
*Runs seamlessly using these versions as of July 12, 2023. Subject to change according to conda compatibility.*

* python 3.9.19
* tensorflow 2.12.0
* scikit-learn 1.2.2
* numpy 1.23.5
* click 8.0.4
* matplotlib 3.7.1
* tqdm 4.65.0
* jupyter 1.0.0
* seaborn 0.12.2

# Git Setup
After accessing server remotely by `ssh`, set up this Git repository for use. Enter your Git username and access token details as required.

* To clone this repo and set up your remote and upstream correctly, do:
```
git clone https://github.com/alpha-davidson/ATTPCLatent
cd ATTPCLatent
git remote add upstream https://github.com/alpha-davidson/TPCNet
git remote set-url --push upstream DISABLED
```

* To verify everything worked as it's supposed to, type: `git remote -v`. This should produce the
following output:
```
origin	https://github.com/alpha-davidson/ATTPCLatent (fetch)
origin	https://github.com/alpha-davidson/ATTPCLatent (push)
upstream	https://github.com/alpha-davidson/TPCNet (fetch)
upstream	DISABLED (push)
```

## Setting up conda environment

`conda create -n tpcnet python=3.9.19`

## Installing packages

`conda activate tpcnet`

`conda create -n tpcnet python=3.9 nodejs=16`

`python3 -m pip install tensorflow[and-cuda]==2.12 scikit-learn==1.2.2 numpy==1.23.5 click==8.0.4 matplotlib==3.7.1 tqdm==4.65.0 jupyter==1.0.0 seaborn==0.12.2`

*Ensure versions match those listed above under [Packages](#Packages). If any version is incompatible with conda due to updates, install default versions (ex. `conda install numpy click tqdm ...`) and **update README** accordingly.*

# Folders

The repository currently contains three main folders: `data_processing`, `training`, and `evaluation and plotting`. To reproduce results, visit these folders in order.
* `data_processing`: Contains data pipelines for different experimental data, including Mg22, O16, C16, and C16+O16. It also includes a python notebook for event-wise voxel data exploration.
* `training`: Contains scripts for pretraining a PointNet-based model on jigsaw event data. It includes model architecture definitions and the training pipeline.
* `evaluation and plotting`: Contains scripts for evaluating a pre-trained model.
* `latent_layer_processing`: Contains scripts for extracting global features and plotting them. 

# Files

The `ATTPCLatent` folder currently containts two Python files:
* `plotting.py`: Used for plotting purposes
* `pointnet.py`: PointNet implementation

# Workflow to Reproduce Results
Start by collecting the relevant data from the data folder (which is above the DAVIDSON directory), and copying it (cp command) to the data_pipeline folder. 

## Data Pipelines
Each experiment has a data_pipeline folder that will generate numpy files with data that the model creation pipeline can understand. For example, there is a python script for preprocessing the raw data from the O16 experiment (originally an h5 file). Currently, this repo contains four data pipelines for O16, Mg22, C16, and O16+C16. See data pipeline README for more information.

## Training on Dataset
To train a pre-trained model after preprocessing the data, use `unscrambling_jigsaw.sh` to run the `pretrain_on_jigsaw_events.py` training pipeline. `{experiment}_model` folder will be created with weights after running the pipeline.

## Evaluation of Models' Performance
To evaluate a pre-trained model, use `evaluating_jigsaw.sh` to run the `evaluate_jigsaw_reconstruction.py` file. `{experiment}_plots` folder will be created with learning curve, histogram of reconstruction accuracy, and sample reconstructed events.

## Extraction of Models' Latent Representation
To extract a model's latent representation, use `extract_latent_layer.sh` to run the `global_features_extraction.py` file. `clustering_plots` folder will be created, containing the specified clustering results, such as t-SNE or k-means.