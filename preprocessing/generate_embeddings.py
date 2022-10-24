import json
from os.path import join
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    data_folder = join("..", "data")
    processed_dataset_folder = join(data_folder, "processed-comfact")
    embeddings_folder = join(data_folder, "embeddings")

    fact_linking_data_files = {"persona": "persona_atomic_final_123.json", "roc": "roc_atomic_final_328.json",
                                "movie": "moviesum_atomic_final_81.json", "mutual": "mutual_atomic_final_237.json"}

    with open(join(processed_dataset_folder, "fact2idx.json"), "r") as f:
        fact2idx = json.load(f)
    
    idx2fact = {idx:fact for fact, idx in fact2idx.items()}
    ordered_facts = [idx2fact[idx] for idx in range(len(idx2fact))]

    assert set(ordered_facts) == set(fact2idx.keys())

    sentence_bert = SentenceTransformer("all-mpnet-base-v2")
    node_embeddings = sentence_bert.encode(ordered_facts, convert_to_numpy=True, show_progress_bar=True)
    node_embeddings = np.stack(node_embeddings, axis=0)
    
    Path(embeddings_folder).mkdir(parents=True, exist_ok=True)
    np.save(join(embeddings_folder, "augmented_graph_embeddings.npy"), node_embeddings)
