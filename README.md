# Improving Commonsense Fact Linking by Incorporating Relations Between Facts


### Overview

Language models are frequently augmented with external knowledge to perform better on tasks such as question answering, story generation, or dialogue modeling. Many approaches have already utilized retrieving facts from knowledge graphs, however, they treat each fact independently. In this work, we improve on fact linking task by considering relationships between the facts to adjust predictions. For that purpose, we develop two graph processing techniques that increase graph quality, and a data-balancing approach. Since different applications have different computational constraints and accuracy requirements, we test our approach on three models that contain different amounts of contextual information and operate at different scales. Our models are evaluated on ComFact dataset, where they on average perform similarly or better than models that consider facts independently. On some dataset portions of ComFact, the performance gain is even more than 5% according to F1-score.


## Code Setup


### Installation

**Requirements:**

- NVIDIA GPU (recommended), Linux, Python
- PyTorch, PyTorch Geometric, various Python packages; Instructions for installing these dependencies are found below

**Python environment:**
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/index.html) package manager

```bash
conda create -n fact-linking python=3.9
conda activate fact-linking
```

**Pytorch and PyTorch Geometric:**
Manually install [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) with CUDA support (CPU version is also supported). We have verified under PyTorch 1.12.1 with CUDA version 11.3.

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
``` 

**Clone fact-linking repository and install:**

```bash
git clone https://github.com/timpostuvan/fact-linking.git
cd fact-linking
pip install -r requirements.txt
```


### Download Datasets

Manually download [ComFact dataset](https://drive.google.com/file/d/1nbQiASv32WTGVo5TQHatJbxBlz2HtMRP/view?usp=sharing) and place `data/` under this root directory. Then, rename the folder:

```bash
mkdir data/.raw-comfact
mv data/* data/.raw-comfact/
mv data/.raw-comfact/ data/raw-comfact/
```


## Data Preprocessing

To preprocess the dataset, run the following code within `preprocessing/` folder:

```bash
bash preprocess.sh
```


## Training and Evaluation

To train and evaluate a model, run:

```bash
python train.py --config_path=${CONFIG_FILE} --dataset_portion=${DATASET_PORTION}
```

Parameters:
- configuration path of the model `${CONFIG_FILE}`: configuration files for our models are inside `configs/` folder
- data portion of ComFact `${DATASET_PORTION}`: "persona" | "mutual" | "roc" | "movie"

Example:
``` bash
python train.py --config_path=configs/two_tower_MLP_node_classification.yaml --dataset_portion=roc
```


## Analysis of ComFact Dataset

Scripts to analyze ComFact dataset are available inside `scripts/` folder.
