import json
from os.path import join
from pathlib import Path

from preprocessing_utils import convert_fact_to_text


if __name__ == "__main__":
    data_folder = join("..", "data")
    raw_dataset_folder = join(data_folder, "raw-comfact")
    processed_dataset_folder = join(data_folder, "processed-comfact")

    fact_linking_data_files = {"persona": "persona_atomic_final_123.json", "roc": "roc_atomic_final_328.json",
                                "movie": "moviesum_atomic_final_81.json", "mutual": "mutual_atomic_final_237.json"}

    all_facts = []
    for folder, file in fact_linking_data_files.items():
        with open(join(raw_dataset_folder, folder, file), "r") as f:
            linking_data = json.load(f)
        for sid, sample in linking_data.items():
            for idx in range(len(sample["facts"])):
                linked_facts = sample["facts"][str(idx)]
        
                for head, rel_tails in linked_facts.items():
                    for rt in rel_tails["triples"]:
                        relation = rt["relation"]
                        tail = rt["tail"]

                        textual_description = convert_fact_to_text(head, relation, tail)
                        all_facts.append(textual_description)

    all_facts = set(all_facts)
    print(f"Number of facts: {len(all_facts)}")
    fact2idx = {fact:idx for idx, fact in enumerate(all_facts)}
    Path(processed_dataset_folder).mkdir(parents=True, exist_ok=True)
    with open(join(processed_dataset_folder, "fact2idx.json"), "w+") as f:
        json.dump(fact2idx, f)
