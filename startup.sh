#!/bin/bash

wandb login <insert your API key>

git config --global --add safe.directory '*'

python train.py $@

# RUNAI COMMAND: 
# runai submit fact-linking -i ic-registry.epfl.ch/lsir/fact-linking \
#       --pvc runai-lsir-postuvan-nlp4sd:/nlp4sd \
#       --working-dir /nlp4sd/postuvan/fact-linking
#       --gpu 1  \
#       --command  -- "./startup.sh --config_path=configs/QAGNN_node_classification.yaml"
