import pickle
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import tqdm
from torch import Tensor
import torch
import random
from preprocess import get_node_signature, get_relationship_signature, get_node_signature_simple, get_node_id
import requests
from openai import OpenAI
from typing import List, Union, Dict
import torch
import json

MODEL_PORT = {
    "qwen2.5": 8088,
    "qwen3": 8084,
    "llama3.1": 8083,
    "llama3.2": 8082
}

QWEN2_5_PORT = 8088
QWEN3_PORT = 8084
LLAMA3_1_PORT = 8083
LLAMA3_2_PORT = 8082

sample_prompt1 = "Focus on entity deduplication and disambiguation in knowledge graph, by extracting the core discriminative semantics of the entity in its context. Entity: {word}, context: {relations}. The core semanitcs of the entity \" {word} \" in the current context is (keep the information that can distinguish the entity from other similar entities do not just output the generalized theme):"
sample_prompt2 = "The entity \" {word} \" might have different meanings in different contexts (ambiguation), or might share the same essence of an entity with a different name (duplication). For example, in relationships: {relations}, the entity \" {word} \""
sample_prompt3 = "Focus on entity deduplication and disambiguation in knowledge graph. Entity: {word}, context: {relations}. The core semanitcs of the entity \" {word} \""

def get_all(text:str, aim_word:str, port:str=QWEN2_5_PORT):
    payload = {
        "text": text,
        "aim_word": aim_word,
        "vector_mode": "mean"
    }
    API_URL = f"http://localhost:{port}/get_all"
    r = requests.post(API_URL, json=payload)
    # 如果失败，查看详细的detail
    if r.status_code != 200:
        raise ValueError(f"API request failed with status code {r.status_code}: {r.text}")
    
    data = r.json()
    if data["status"] != "success":
        raise ValueError(f"API error: {data}")
    hidden_states = [torch.tensor(vec) for vec in data["hidden_states"]]
    k_vectors = [torch.tensor(vec) for vec in data["k_vectors"]]
    token_num = data["token_num"]
    return hidden_states, k_vectors, token_num

def get_last_token(text:str, aim_word:str, vector_mode:str="mean", port:str=QWEN2_5_PORT):
    payload = {
        "text": text,
        "aim_word": aim_word,
        "layer_idx": -1,
        "vector_mode": vector_mode
    }
    API_URL = f"http://localhost:{port}/get_last_word"
    r = requests.post(API_URL, json=payload)
    # 如果失败，查看详细的detail
    if r.status_code != 200:
        raise ValueError(f"API request failed with status code {r.status_code}: {r.text}")
    
    data = r.json()
    if data["status"] != "success":
        raise ValueError(f"API error: {data}")
    last_token = torch.tensor(data["last_token_vector"])
    return last_token

def get_all_kv(node_id:str, relations:list[Relationship], port, sample_num:int, all_rels:bool=False, rel_num:int=15, max_rel:int=1000):
    if all_rels is False:
        if len(relations) > rel_num:
            
            relations = random.sample(relations, rel_num)
    else:
        if len(relations) > max_rel:
            # import random
            relations = random.sample(relations, max_rel)

    sample_prompt = ""
    # 根据sample_num选择不同的prompt
    if sample_num == 1:
        sample_prompt = sample_prompt1
    elif sample_num == 2:
        sample_prompt = sample_prompt2
    elif sample_num == 3:
        sample_prompt = sample_prompt3
    else:
        raise ValueError(f"Unknown sample_num: {sample_num}")

    if len(relations) == 0:
        # 此时node_id即node_name
        aim_word = node_id
        text = sample_prompt.format(word=aim_word, relations="")
        all_layers_hidden_states, all_layers_k, token_num = get_all(text, aim_word, port)
        return all_layers_hidden_states, all_layers_k, 0, token_num
    rel_signs = [get_relationship_signature(rel) for rel in relations]
    rel_signs_str = ";".join(rel_signs)
    aim_node:Node = relations[0].source if relations[0].source.id == node_id else relations[0].target

    assert aim_node.id == node_id, f"aim_node.id {aim_node.id} != node_id {node_id}"

    aim_word = get_node_signature_simple(aim_node)
    text = sample_prompt.format(word=aim_word, relations=rel_signs_str)

    # print(len(relations))

    all_layers_hidden_states, all_layers_k, token_num = get_all(text, aim_word, port)
    # 统计输入的text的token数量
    return all_layers_hidden_states, all_layers_k, len(relations), token_num

def compute_all_kv(node2name_file:str, node2rel_file:str, kg_name:str, sample_num:int, out_dir:str, node2selected_rel_file:str=None, model:str="qwen2.5", all_rels:bool=False, rel_num:int=15,max_rel:int=1000):
    with open(node2name_file, "rb") as f:
        node2name = pickle.load(f)
    with open(node2rel_file, "rb") as f:
        node2rels = pickle.load(f)
    if node2selected_rel_file is not None:
        with open(node2selected_rel_file, "rb") as f:
            node2selected_rels = pickle.load(f)
    else:
        node2selected_rels = None
    
    if node2selected_rel_file is None:
        emb_type = f"(sample{sample_num})({model})(raw)"
    else:
        emb_type = f"(sample{sample_num})({model})"
    if all_rels:
        emb_type += "(all_rels)"

    # 同时统计平均关系数和平均token数,以及平均推理时间
    total_rels = 0
    total_tokens = 0
    # 统计一个总时间
    import time
    start_time = time.time()
    ans_hidden_states = {}
    ans_k = {}
    for node_id, node_name in tqdm.tqdm(node2name.items()):
        rels = []
        if node2selected_rels:
            rels = node2selected_rels.get(node_id, [])
        if len(rels) == 0:
            rels = node2rels.get(node_id, [])
        if len(rels) == 0:
            # continue
            all_layers_hidden_states, all_layers_k, selected_rel_num, token_num = get_all_kv(node_name, rels, MODEL_PORT[model], sample_num, all_rels, rel_num, max_rel)
            ans_hidden_states[node_id] = all_layers_hidden_states
            ans_k[node_id] = all_layers_k
            total_rels += selected_rel_num
            total_tokens += token_num
            continue
        all_layers_hidden_states, all_layers_k, selected_rel_num, token_num = get_all_kv(node_id, rels, MODEL_PORT[model], sample_num, all_rels, rel_num, max_rel)
        ans_hidden_states[node_id] = all_layers_hidden_states
        ans_k[node_id] = all_layers_k
        total_rels += selected_rel_num
        total_tokens += token_num
    
    end_time = time.time()
    total_time = end_time - start_time # 单位是秒
    avg_time = total_time / len(node2name) if len(node2name) > 0 else 0
    avg_rels = total_rels / len(node2name)
    avg_tokens = total_tokens / len(node2name)
    print(f"Average number of relations per node: {avg_rels}")
    print(f"Average number of tokens per input: {avg_tokens}")
    print(f"Average inference time per node: {avg_time}")
    out_file1 = f"{out_dir}/{kg_name}_all_layers_hidden_states{emb_type}.pkl"
    out_file2 = f"{out_dir}/{kg_name}_all_layers_k{emb_type}.pkl"
    with open(out_file1, "wb") as f:
        pickle.dump(ans_hidden_states, f)
    with open(out_file2, "wb") as f:
        pickle.dump(ans_k, f)
        

def get_last_token_vector(node_id:str, relations:list[Relationship], port, rel_num:int=15) -> Tensor:
    if len(relations) > rel_num:
        import random
        relations = random.sample(relations, rel_num)

    if len(relations) == 0:
        # 此时node_id即node_name
        aim_word = node_id
        text = sample_prompt1.format(word=aim_word, relations="")
        last_token_vector = get_last_token(text, aim_word, port=port)
        return last_token_vector
    rel_signs = [get_relationship_signature(rel) for rel in relations]
    rel_signs_str = ";".join(rel_signs)
    aim_node:Node = relations[0].source if relations[0].source.id == node_id else relations[0].target

    assert aim_node.id == node_id, f"aim_node.id {aim_node.id} != node_id {node_id}"

    aim_word = get_node_signature_simple(aim_node)
    text = sample_prompt1.format(word=aim_word, relations=rel_signs_str)

    last_token_vector = get_last_token(text, aim_word, port=port)
    return last_token_vector

def compute_last_token_vector(node2name_file:str, node2rel_file:str, kg_name:str, sample_num:int, out_dir:str, node2selected_rel_file:str=None, model:str="qwen2.5"):
    with open(node2name_file, "rb") as f:
        node2name = pickle.load(f)
    with open(node2rel_file, "rb") as f:
        node2rels = pickle.load(f)
    if node2selected_rel_file is not None:
        with open(node2selected_rel_file, "rb") as f:
            node2selected_rels = pickle.load(f)
    else:
        node2selected_rels = None
    
    emb_type = f"(sample{sample_num})({model})"
    ans = {}
    for node_id, node_name in tqdm.tqdm(node2name.items()):
        rels = []
        if node2selected_rels:
            rels = node2selected_rels.get(node_id, [])
        if len(rels) == 0:
            rels = node2rels.get(node_id, [])
        if len(rels) == 0:
            # continue
            last_token_vector = get_last_token_vector(node_name, rels, port=MODEL_PORT[model])
            ans[node_id] = last_token_vector
            continue
        last_token_vector = get_last_token_vector(node_id, rels, port=MODEL_PORT[model])
        ans[node_id] = last_token_vector

    out_file = f"{out_dir}/{kg_name}_last_token{emb_type}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(ans, f)


def aggregate_vectors(source_file:str, dir:str, layer_idx:list[int], mean:str="mean"):
    in_file = f"{dir}/{source_file}"
    with open(in_file, "rb") as f:
        node2vectors = pickle.load(f)
    """
    {
        node_id:[tensor_layer1, tensor_layer2, ...] # 每个tensor_layer的shape都是(embedding_dim,)
    }
    """
    target_layer_vectors = {}
    for node_id, vectors in tqdm.tqdm(node2vectors.items()):
        selected_vectors = [vectors[i] for i in layer_idx if i < len(vectors)]
        if mean == "mean":
            aggregated_vector = torch.mean(torch.stack(selected_vectors), dim=0)
        elif mean == "max":
            aggregated_vector, _ = torch.max(torch.stack(selected_vectors), dim=0)
        elif mean == "concat":
            aggregated_vector = torch.cat(selected_vectors, dim=0)
        else:
            raise ValueError(f"Unknown mean method: {mean}")
        target_layer_vectors[node_id] = aggregated_vector
    
    return target_layer_vectors
