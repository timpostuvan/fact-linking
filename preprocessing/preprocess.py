import argparse
from os.path import join
from pathlib import Path
import json
import pickle
from collections import OrderedDict, defaultdict
from preprocessing_utils import convert_fact_to_text

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM


data_folder = join("..", "data")
comfact_path = join(data_folder, "raw-comfact")
processed_dataset_folder = join(data_folder, "processed-comfact")

fact_grounding_path = join(processed_dataset_folder, "fact2idx.json")
head_pattern_map_path = join(processed_dataset_folder, "atomic_head_to_patterns_list.json")
tail_pattern_map_path = join(processed_dataset_folder, "atomic_tail_to_patterns_list.json")


def load_resources():
    with open(fact_grounding_path, "r") as f:
        _fact2idx  = json.load(f)

    _idx2fact = {idx:fact for fact, idx in _fact2idx.items()}

    with open(head_pattern_map_path, "r") as f:
        _head_pattern_map = json.load(f)

    with open(tail_pattern_map_path, "r") as f:
        _tail_pattern_map = json.load(f)

    return _fact2idx, _idx2fact, _head_pattern_map, _tail_pattern_map


def load_lm():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_fast=True)
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return tokenizer, model


fact2idx, idx2fact, head_pattern_map, tail_pattern_map = load_resources()
tokenizer, model = load_lm()
loss_fct = CrossEntropyLoss(reduction='none')


@torch.no_grad()
def get_lm_score(
    node_ids: list[int],
    context_text: str,
):
    context_enc = tokenizer.encode(
        context_text.lower(),
        return_tensors="pt",
        add_special_tokens=True
    )
    context_enc = context_enc.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    context_outputs = model.roberta(
        context_enc,
        output_attentions=False,
        output_hidden_states=False,
    )
    context_scores = model.lm_head(context_outputs[0])

    loss_fct = CrossEntropyLoss(reduction='none')
    context_loss = loss_fct(context_scores[0, :, :], context_enc.view(-1)).sum()
    scores = [-context_loss.detach().cpu().numpy()]

    num_nodes = len(node_ids)
    batch_size = 30

    for idx in range(0, num_nodes, batch_size):
        nodes = node_ids[idx: min(idx + batch_size, num_nodes)]
        textual_representations = [idx2fact[node_id] for node_id in nodes]
        sentences = [f"{context_text.lower()} {text_rep}" for text_rep in textual_representations]
        sentences_enc = tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        sentences_enc["attention_mask"] = sentences_enc["attention_mask"].to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        sentences_enc["input_ids"] = sentences_enc["input_ids"].to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        masked_outputs = model.roberta(
            **sentences_enc,
            output_attentions=False,
            output_hidden_states=False,
        )
        masked_scores = model.lm_head(masked_outputs[0])
        masked_lm_loss = loss_fct(masked_scores.transpose(1, 2), sentences_enc["input_ids"])
        masked_lm_loss = (masked_lm_loss * sentences_enc["attention_mask"]).sum(1)
        _scores = list(-masked_lm_loss.detach().cpu().numpy())
        scores += _scores
    node_ids.insert(0, -1)
    assert len(scores) == len(node_ids)
    idx2score = OrderedDict(sorted(list(zip(node_ids, scores)), key=lambda x: -x[1]))
    return idx2score


def process_sample(
    text_context_list: list[str], 
    central_node_ids: list[int], 
    context_node_ids: list[int],
    edges: list[tuple[int, int]], 
    edge_types: list[int]
):
    text_context = " ".join(text_context_list)
    idx2score = get_lm_score(central_node_ids + context_node_ids, text_context)
    nodes = central_node_ids + sorted(context_node_ids, key=lambda x: -idx2score[x])
    # node type: 1 - central node, 0 - context node
    node_types = [1 for _ in central_node_ids] +  [0 for _ in context_node_ids]

    return dict(
        text_context_list=text_context_list,
        nodes=nodes,
        node_types=node_types,
        edges=edges,
        edge_types=edge_types,
        idx2score=idx2score
    )


def generate_dataset(
    dataset_portion_path: str,
    data_split_paths: dict[str],
    experimental_setting: str,
    output_folder: str,
    context_size: int = 2,
):
    """
    This function will save
        (1) text context
        (2) fact/node ids
        (3) types of facts/nodes: central or context
        (4) edge list
        (5) edge types: head connections or tail connections
        (6) idx2score that maps a fact id to its relevance score given the text context
    to the output folder in python pickle format
    """

    processed_dataset = defaultdict(dict)
    with open(dataset_portion_path, "r") as f:
        linking_data = json.load(f)
    for sid, sample in tqdm(linking_data.items()):
        for context_central_idx in range(len(sample["facts"])):
            central_node_ids = set()
            all_node_ids = set()
            text_context_list = []

            augmented_head_entities = defaultdict(set)
            augmented_tail_entities = defaultdict(set)

            scope_start = max(context_central_idx - context_size, 0)
            if experimental_setting == "complete":
                scope_end = min(context_central_idx + context_size + 1, len(sample["facts"]))
            elif experimental_setting == "past":
                scope_end = min(context_central_idx + 1, len(sample["facts"]))
            else:
                raise ValueError(f"Unknown experimental setting {experimental_setting}")

            for idx in range(scope_start, scope_end):
                text = sample["text"][idx]
                text_context_list.append(text)
                
                linked_facts = sample["facts"][str(idx)]
                central_utterance = True if idx == context_central_idx else False
        
                for head, rel_tails in linked_facts.items():
                    head_pattern = " ".join(head_pattern_map[head])

                    for rt in rel_tails["triples"]:
                        relation = rt["relation"]
                        tail = rt["tail"]
                        tail_pattern = " ".join(tail_pattern_map[tail])

                        augmented_node = convert_fact_to_text(head, relation, tail)
                        augmented_node_idx = fact2idx[augmented_node]

                        all_node_ids.add(augmented_node_idx)
                        if central_utterance:
                            central_node_ids.add(augmented_node_idx)

                        augmented_head_entities[head_pattern].add(augmented_node_idx)
                        augmented_tail_entities[tail_pattern].add(augmented_node_idx)
            
            context_node_ids = sorted(all_node_ids - central_node_ids)
            central_node_ids = sorted(central_node_ids)
            
            edges = []
            # edge type: 0 - between head entities, 1 - between tail entities
            edge_types = []

            # add edges between facts with the same head in the augmented graph
            for head_pattern in augmented_head_entities:
                facts = augmented_head_entities[head_pattern]
                
                for first_node_idx in facts:
                    for second_node_idx in facts:                    
                        if first_node_idx == second_node_idx:
                            continue
                        
                        edges.append((first_node_idx, second_node_idx))
                        edge_types.append(0)

            # add edges between facts with the same tail in the augmented graph
            for tail_pattern in augmented_tail_entities:
                facts = augmented_tail_entities[tail_pattern]
                
                for first_node_idx in facts:
                    for second_node_idx in facts:                    
                        if first_node_idx == second_node_idx:
                            continue
                        
                        edges.append((first_node_idx, second_node_idx))
                        edge_types.append(1)

            processed_dataset[sid][context_central_idx] = process_sample(
                text_context_list=text_context_list,
                central_node_ids=central_node_ids,
                context_node_ids=context_node_ids,
                edges=edges,
                edge_types=edge_types
            )


    for split in data_split_paths:
        with open(data_split_paths[split], "r") as f:
            indices = json.load(f)
        
        samples = []
        for sid in indices:
            for idx, sample in processed_dataset[sid].items():
                samples.append(sample)

        with open(join(output_folder, f"{split}.pk"), "wb+") as out:
            pickle.dump(samples, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_portion", default="movie")           # [movie, mutual, persona, roc]
    parser.add_argument("--experimental_setting", default="complete")   # [complete, past]
    parser.add_argument("--context_size", type=int, default=2)
    args = parser.parse_args()

    dataset_portion_folder = join(comfact_path, args.dataset_portion)
    if args.dataset_portion == "movie":
        dataset_portion_path = join(dataset_portion_folder, "moviesum_atomic_final_81.json")
        train_path = join(dataset_portion_folder, "done_mid_train_58.json")
        dev_path = join(dataset_portion_folder, "done_mid_dev_11.json")
        test_path = join(dataset_portion_folder, "done_mid_test_12.json")
    elif args.dataset_portion == "mutual":
        dataset_portion_path = join(dataset_portion_folder, "mutual_atomic_final_237.json")
        train_path = join(dataset_portion_folder, "mutual_atomic_did_train_170.json")
        dev_path = join(dataset_portion_folder, "mutual_atomic_did_val_33.json")
        test_path = join(dataset_portion_folder, "mutual_atomic_did_test_34.json")
    elif args.dataset_portion == "persona":
        dataset_portion_path = join(dataset_portion_folder, "persona_atomic_final_123.json")
        train_path = join(dataset_portion_folder, "persona_atomic_did_train_90.json")
        dev_path = join(dataset_portion_folder, "persona_atomic_did_val_15.json")
        test_path = join(dataset_portion_folder, "persona_atomic_did_test_18.json")
    elif args.dataset_portion == "roc":
        dataset_portion_path = join(dataset_portion_folder, "roc_atomic_final_328.json")
        train_path = join(dataset_portion_folder, "done_sid_train_235.json")
        dev_path = join(dataset_portion_folder, "done_sid_dev_46.json")
        test_path = join(dataset_portion_folder, "done_sid_test_47.json")
    else:
        raise ValueError(f"Unknown dataset name {args.dataset_portion}")
    
    data_split_paths = dict(
        train=train_path,
        dev=dev_path,
        test=test_path
    ) 

    output_folder = join(processed_dataset_folder, f"{args.experimental_setting}-{args.context_size}", args.dataset_portion)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    generate_dataset(
        dataset_portion_path=dataset_portion_path,
        data_split_paths=data_split_paths,
        experimental_setting=args.experimental_setting,
        context_size=args.context_size,
        output_folder=output_folder
    )
