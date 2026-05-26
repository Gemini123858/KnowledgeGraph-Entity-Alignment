
import pickle
from compute import compute_last_token_vector, compute_all_kv, aggregate_vectors
import json
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import tqdm
from torch import Tensor
from typing import List, Tuple, Dict
import numpy as np
from torch.nn import functional as F
import os
from collections import defaultdict

def compute(emb_type: str, model_name: str, sample_prompt: int, selected_rels: bool, all_rels: bool, rel_num: int, input_path: str, out_path: str, kg1_name: str, kg2_name: str):
    if emb_type == "last_token":
        node2name_file = f"{input_path}/{kg1_name}_node2name.pkl"
        node2rels_file = f"{input_path}/{kg1_name}_node2rels.pkl"
        selected_rels_file = f"{input_path}/{kg1_name}_selected_rels.pkl"
        compute_last_token_vector(node2name_file, node2rels_file, kg1_name, sample_prompt, out_path, selected_rels_file if selected_rels else None, model_name)
        node2name_file = f"{input_path}/{kg2_name}_node2name.pkl"
        node2rels_file = f"{input_path}/{kg2_name}_node2rels.pkl"
        selected_rels_file = f"{input_path}/{kg2_name}_selected_rels.pkl"
        compute_last_token_vector(node2name_file, node2rels_file, kg2_name, sample_prompt, out_path, selected_rels_file if selected_rels else None, model_name)
    elif emb_type == "hidden_state" or emb_type == "attention_k":
        node2name_file = f"{input_path}/{kg1_name}_node2name.pkl"
        node2rels_file = f"{input_path}/{kg1_name}_node2rels.pkl"
        selected_rels_file = f"{input_path}/{kg1_name}_selected_rels.pkl"
        compute_all_kv(node2name_file, node2rels_file, kg1_name, sample_prompt, out_path, selected_rels_file if selected_rels else None, model_name, all_rels, rel_num)
        node2name_file = f"{input_path}/{kg2_name}_node2name.pkl"
        node2rels_file = f"{input_path}/{kg2_name}_node2rels.pkl"
        selected_rels_file = f"{input_path}/{kg2_name}_selected_rels.pkl"
        compute_all_kv(node2name_file, node2rels_file, kg2_name, sample_prompt, out_path, selected_rels_file if selected_rels else None, model_name, all_rels, rel_num)

def compute_results(kg1_node_id_to_vector:dict[str, Tensor], kg2_node_id_to_vector:dict[str, Tensor], matched_nodes_id_to_node:Dict[str, List[str]]):
    results = []
    for node_id1, matched_node_ids2 in tqdm.tqdm(matched_nodes_id_to_node.items()):
        vector1 = kg1_node_id_to_vector.get(node_id1)
        if vector1 is None:
            print(f"Node {node_id1} in icews has no vector, skip.")
            continue
        for node_id2 in matched_node_ids2:
            vector2 = kg2_node_id_to_vector.get(node_id2)
            if vector2 is None:
                print(f"Node {node_id2} in yago has no vector, skip.")
                continue
            sim = F.cosine_similarity(vector1.unsqueeze(0), vector2.unsqueeze(0)).item()
            results.append({
                "node_1": node_id1,
                "node_2": node_id2,
                "sim": sim
            })
    return results

def analysis_hitsk_mrr(results, model:str, sample_:str, layers:list[int], type_:str="k vectors", mapping:Dict[str, str]=None):
    grouped_results = defaultdict(list)
    for item in tqdm.tqdm(results, desc="Processing results"):
        grouped_results[item["node_1"]].append(item)
    # 计算Hits@K, MRR
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = 0
    for node_1, items in tqdm.tqdm(grouped_results.items(), desc="Calculating metrics"):
        # 按照sim从大到小排序
        items.sort(key=lambda x: x["sim"], reverse=True)
        # 获取node_1对应的yago节点集合
        matched_nodes = mapping.get(node_1)
        for rank, item in enumerate(items, start=1):
            if item["node_2"] in matched_nodes:
                mrr += 1 / rank
                if rank <= 1:
                    hits_at_k[1] += 1
                if rank <= 3:
                    hits_at_k[3] += 1
                if rank <= 5:
                    hits_at_k[5] += 1
                if rank <= 10:
                    hits_at_k[10] += 1
                break
    num_nodes = len(grouped_results)
    print("\"\"\"")
    print(f"Type: {type_}, Model: {model}, Sample: {sample_}, Layer: {layers}, Similarity: mean")
    print(f"Hits@1: {hits_at_k[1] / num_nodes:.4f}")
    print(f"Hits@3: {hits_at_k[3] / num_nodes:.4f}")
    print(f"Hits@5: {hits_at_k[5] / num_nodes:.4f}")
    print(f"Hits@10: {hits_at_k[10] / num_nodes:.4f}")
    print(f"MRR: {mrr / num_nodes:.4f}")
    print("\"\"\"")

def compute_hidden_state(layer_idxs:list[list], model:str, sample_:str, out_data_dir:str, matched_nodes_id_to_node:Dict[str, List[str]], aggregation_method:str="mean"):
    in_file_sample = "{kg_type}_all_layers_hidden_states({sample_})({model}).pkl"
    for layer_idx in layer_idxs:
        in_file = in_file_sample.format(kg_type="icews", sample_=sample_, model=model, layer_idx=layer_idx)
        kg1_node_id_to_vector = aggregate_vectors(in_file, out_data_dir,  layer_idx, mean=aggregation_method)
        in_file = in_file_sample.format(kg_type="yago", sample_=sample_, model=model, layer_idx=layer_idx)
        kg2_node_id_to_vector = aggregate_vectors(in_file, out_data_dir,layer_idx, mean=aggregation_method)
        results = compute_results(kg1_node_id_to_vector, kg2_node_id_to_vector, matched_nodes_id_to_node)
        analysis_hitsk_mrr(results, model=model, sample_=sample_, layers=layer_idx, type_="hidden states")

def compute_k(layer_idxs:list[list], model:str, sample_:str, out_data_dir:str, matched_nodes_id_to_node:Dict[str, List[str]], aggregation_method:str="mean"):
    in_file_sample = "{kg_type}_all_layers_k({sample_})({model}).pkl"
    for layer_idx in layer_idxs:
        in_file = in_file_sample.format(kg_type="icews", sample_=sample_, model=model, layer_idx=layer_idx)
        kg1_node_id_to_vector = aggregate_vectors(in_file, out_data_dir, layer_idx, mean=aggregation_method)
        in_file = in_file_sample.format(kg_type="yago", sample_=sample_, model=model, layer_idx=layer_idx)
        kg2_node_id_to_vector = aggregate_vectors(in_file, out_data_dir, layer_idx, mean=aggregation_method)
        results = compute_results(kg1_node_id_to_vector, kg2_node_id_to_vector, matched_nodes_id_to_node)
        analysis_hitsk_mrr(results, model=model, sample_=sample_, layers=layer_idx, type_="k vectors")

def compute_last_token(model:str, sample_:str, out_data_dir:str, matched_nodes_id_to_node:Dict[str, List[str]]):
    infile = "{kg_type}_last_token({sample_})({model}).pkl"
    in_file = infile.format(kg_type="icews", sample_=sample_, model=model)
    with open(out_data_dir + "/" + in_file, "rb") as f:
        kg1_node_id_to_vector = pickle.load(f)
    in_file = infile.format(kg_type="yago", sample_=sample_, model=model)
    with open(out_data_dir + "/" + in_file, "rb") as f:        
         kg2_node_id_to_vector = pickle.load(f)
    results = compute_results(kg1_node_id_to_vector, kg2_node_id_to_vector, matched_nodes_id_to_node)
    analysis_hitsk_mrr(results, model=model, sample_=sample_, layers="", type_="last token")
           

if __name__ == "__main__":
    # python decoder_only.py --dataset dbp_wiki/icews_wiki/icews_yago --model llama3.2 --prompt 1/2/3 --emb_type last_token/hidden_state/attention_k --use_select true/false --all_rels true/false --rel_num 15 \
    # --layers [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]] --agg_method mean/concat/max
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="icews_yago", help="dataset name")
    parser.add_argument("--model", type=str, default="llama3.2", help="model name")
    parser.add_argument("--prompt", type=str, default="1", help="prompt sampling method, 1/2/3")
    parser.add_argument("--emb_type", type=str, default="last_token", help="embedding type, last_token/hidden_state/attention_k")
    parser.add_argument("--use_select", type=str, default="false", help="whether to use selected rels")
    parser.add_argument("--all_rels", type=str, default="false", help="whether to use all rels or only rel_num rels")
    parser.add_argument("--rel_num", type=int, default=15, help="number of rels to use if not using all rels")
    parser.add_argument("--layers", type=list[list[int]], default=[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]], help="layers to aggregate for hidden_state and attention_k")
    parser.add_argument("--agg_method", type=str, default="mean", help="aggregation method, mean/concat/max")
    args = parser.parse_args()
    
    if args.dataset == "dbp_wiki":
        input_path = "../data/output_data/dbp_wiki"
        out_path = "../data/output_data/dbp_wiki/decoder_only"
        kg1_name = "dbp"
        kg2_name = "wiki"
        kg1id2kg2id_file = f"{input_path}/dbp2wiki.json"
        
    elif args.dataset == "icews_wiki":
        input_path = "../data/output_data/icews_wiki"
        out_path = "../data/output_data/icews_wiki/decoder_only"
        kg1_name = "icews"
        kg2_name = "wiki"
        kg1id2kg2id_file = f"{input_path}/icews2wiki.json"

    elif args.dataset == "icews_yago":
        input_path = "../data/output_data/icews_yago"
        out_path = "../data/output_data/icews_yago/decoder_only"
        kg1_name = "icews"
        kg2_name = "yago"
        kg1id2kg2id_file = f"{input_path}/icews2yago.json"
    else:
        raise ValueError("Unknown dataset. Please choose from dbp_wiki/icews_wiki/icews_yago.")
    
    print(f"Processing dataset: {args.dataset} with model {args.model}, prompt {args.prompt}, embedding type {args.emb_type}, use_select {args.use_select}, all_rels {args.all_rels}, rel_num {args.rel_num}")
    selected_rels:bool = True if args.use_select == "true" else False
    all_rels:bool = True if args.all_rels == "true" else False
    rel_num:int = args.rel_num
    sample_prompt:int = int(args.prompt)
    if not selected_rels:
        emb_type = f"(sample{sample_prompt})({args.model})(raw)"
    else:
        emb_type = f"(sample{sample_prompt})({args.model})"
    if all_rels:
        emb_type += "(all_rels)"
    out_file_h = f"{out_path}/{kg1_name}_all_layers_hidden_states{emb_type}.pkl"
    out_file_k= f"{out_path}/{kg1_name}_all_layers_k{emb_type}.pkl"
    out_file_last_token = f"{out_path}/{kg1_name}_last_token{emb_type}.pkl"

    if args.emb_type == "last_token":
        if os.path.exists(out_file_last_token):
            print(f"{out_file_last_token} already exists. Skipping computation.")
        else:
            compute(args.emb_type, args.model, sample_prompt, selected_rels, all_rels, rel_num, input_path, out_path, kg1_name, kg2_name)

    elif args.emb_type == "hidden_state":
        if os.path.exists(out_file_h):
            print(f"{out_file_h} already exists. Skipping computation.")
        else:
            compute(args.emb_type, args.model, sample_prompt, selected_rels, all_rels, rel_num, input_path, out_path, kg1_name, kg2_name)

    elif args.emb_type == "attention_k":
        if os.path.exists(out_file_k):
            print(f"{out_file_k} already exists. Skipping computation.")
        else:
            compute(args.emb_type, args.model, sample_prompt, selected_rels, all_rels, rel_num, input_path, out_path, kg1_name, kg2_name)

    with open(kg1id2kg2id_file, "rb") as f:
        ref_id_mapping = pickle.load(f)
    matched_nodes_id_to_node_file = f"{input_path}/clusters/matched_nodes_to_cluster_nodes.pkl"
    with open(matched_nodes_id_to_node_file, "rb") as f:
        matched_nodes_id_to_node = pickle.load(f)

    layers = args.layers
    sample_prompt = f"(sample{sample_prompt})"
    if args.emb_type == "last_token":
        compute_last_token(model=args.model, sample_=sample_prompt, out_data_dir=out_path, matched_nodes_id_to_node=matched_nodes_id_to_node)
    elif args.emb_type == "hidden_state":
        compute_hidden_state(layer_idxs=layers, model=args.model, sample_=sample_prompt, out_data_dir=out_path, matched_nodes_id_to_node=matched_nodes_id_to_node, aggregation_method=args.agg_method)
    elif args.emb_type == "attention_k":
        compute_k(layer_idxs=layers, model=args.model, sample_=sample_prompt, out_data_dir=out_path, matched_nodes_id_to_node=matched_nodes_id_to_node, aggregation_method=args.agg_method)
    