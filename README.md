This repository contains code for self-supervised learning of track geometries from TPC [data](https://alphadavidson.slack.com/files/U0146JVGQEB/F025QCGNURJ/output_digi_hdf_mg22_ne20pp_8mev.h5) (in this case, Mg-22, O-16, and C-16). Code is written in `python3` and uses the `tensorflow` package as the framework for the implementation.

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
## Git Setup
After accessing server remotely by `ssh`, set up this Git repository for use. Enter your Git username and access token details as required.
* Cloning the repository: `git clone https://github.com/alpha-davidson/TPCNet.git`
* Creating your own branch: `git checkout -b [insert branchname here]`

## Setting up conda environment

`conda create -n tpcnet python=3.9.19`

## Installing packages

`conda activate tpcnet`

`conda create -n tpcnet python=3.9 nodejs=16`

`python3 -m pip install tensorflow[and-cuda]==2.12 scikit-learn==1.2.2 numpy==1.23.5 click==8.0.4 matplotlib==3.7.1 tqdm==4.65.0 jupyter==1.0.0 seaborn==0.12.2`

*Ensure versions match those listed above under [Packages](#Packages). If any version is incompatible with conda due to updates, install default versions (ex. `conda install numpy click tqdm ...`) and **update README** accordingly.*


# Folders

The repository currently contains two main folders: `pretrain` and `downstream_tasks`. By the final version, `downstream_tasks` will be moved to a separate repository.

* `pretrain`: Contains the complete workflow for building a pre-training model, including data processing, training, and evaluation/plotting.
* `downstream_tasks`: Implements various downstream tasks, including both workflows using the pre-trained model and benchmark workflows without pre-training for comparison.

See specific workflow guides inside corresponding folders to reproduce results.