CONTEXT_SIZE=2
EXPERIMENTAL_SETTING=complete

echo "Grounding facts"
python grounding.py

echo "Generating embeddings"
python generate_embeddings.py

echo "Generating split: movie"
python preprocess.py --dataset_portion=movie --experimental_setting=${EXPERIMENTAL_SETTING} --context_size=${CONTEXT_SIZE}
echo "Generating split: mutual"
python preprocess.py --dataset_portion=mutual --experimental_setting=${EXPERIMENTAL_SETTING} --context_size=${CONTEXT_SIZE}
echo "Generating split: persona"
python preprocess.py --dataset_portion=persona --experimental_setting=${EXPERIMENTAL_SETTING} --context_size=${CONTEXT_SIZE}
echo "Generating split: roc"
python preprocess.py --dataset_portion=roc --experimental_setting=${EXPERIMENTAL_SETTING} --context_size=${CONTEXT_SIZE}
