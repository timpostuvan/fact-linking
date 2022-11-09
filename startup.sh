#!/bin/bash

cd /nlpdata1/home/postuvan/fact-linking

wandb login <insert your API key>

git config --global --add safe.directory '*'

python train.py $@

# RUNAI COMMAND: 
# runai submit -p nlp fact-linking -i ic-registry.epfl.ch/nlp/postuvan/fact-linking \
#       --pvc runai-pv-nlpdata1:/nlpdata1  --gpu 1  \
#       --command -- sh -c /nlpdata1/home/postuvan/fact-linking/startup.sh \
