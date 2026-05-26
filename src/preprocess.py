from pathlib import Path
import sys
import json
import numpy as np
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import math
from typing import Union, List, Dict, Any
import torch
import pickle
from torch import Tensor
from torch.nn import functional as F
from embedding import BailianEmbeddings
import os

def get_node_signature(node: Node) ->str:
    properties = {}
    for k,v in node.properties.items():
        properties[k] = v
    property_str = ",".join(f"{k}={v}" for k, v in properties.items() if k != "name")
    node_type = node.type if node.type != "unknown" else "Entity"
    signature = f"{node_type}:{node.properties["name"]}({property_str})"
    return signature

def get_relationship_signature(relationship: Relationship) -> str:
    node1_signature = get_node_signature(relationship.source)
    node2_signature = get_node_signature(relationship.target)
    signature = f"{node1_signature}-[{relationship.type}]->{node2_signature}"
    return signature

def get_node_signature_simple(node: Node) -> str:
    signature = f"{node.properties['name']}"
    return signature

def get_node_id(node: Node) -> str:
    return node.id

def gen_id2rel(kg: GraphDocument, kg_type: str, out_file):
    node2rels = {}
    for rel in kg.relationships:
        source_id = get_node_id(rel.source)
        target_id = get_node_id(rel.target)

        if source_id not in node2rels:
            node2rels[source_id] = []
        if target_id not in node2rels:
            node2rels[target_id] = []

        node2rels[source_id].append(rel)
        node2rels[target_id].append(rel)

    with open(out_file, "wb") as f:
        pickle.dump(node2rels, f)
    print(f"Saved node2rels mapping to {out_file}")
    return node2rels

def gen_id2selected_rel(node_embeddings: Dict[str, List[float]], other_node_embeddings: Dict[str, List[float]], node2rels: Dict[str, List[Relationship]], out_file):
    # 先计算两个kg中节点embedding的相似度三角阵
    # 转化为tensor进行计算
    node_ids = list(node_embeddings.keys())
    other_node_ids = list(other_node_embeddings.keys())
    node_emb_matrix = torch.tensor([node_embeddings[id] for id in node_ids])  # shape: (num_nodes, embedding_dim)
    other_node_emb_matrix = torch.tensor([other_node_embeddings[id] for id in other_node_ids])  # shape: (num_other_nodes, embedding_dim)
    import torch.nn.functional as F

    # 先对两组 embedding 进行 L2 归一化
    node_emb_norm = F.normalize(node_emb_matrix, p=2, dim=1)
    other_node_emb_norm = F.normalize(other_node_emb_matrix, p=2, dim=1)
    # [N, D] x [D, M] -> [N, M]
    similarity_matrix = torch.mm(node_emb_norm, other_node_emb_norm.T)
    print(f"Computed similarity matrix of shape {similarity_matrix.shape}")

    top_k = 10
    # 对每个节点找到 top-k 最相似的节点
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=-1)

    node2topk = {}
    for i, node_id in enumerate(node_ids):
        similar_nodes = []
        for j in range(top_k):
            score = top_k_values[i, j].item()
            other_id = other_node_ids[top_k_indices[i, j].item()]
            similar_nodes.append((other_id, score))
        node2topk[node_id] = similar_nodes

    print(f"Extracted top-{top_k} similar nodes mappings.")

    # ========== 挑选关系部分 ==========

    # 设置挑选规则参数
    sim_threshold = 0       # 邻居节点在另一个KG中的最低相似度要求
    max_rels_per_node = 15     # 控制关联邻居数目“适中”上限
    max_same_type_rels = 5     # 控制“类型尽可能多样化” (同种类型最多挑选这条数)

    selected_rels_for_node = {}

    import tqdm
    for node_id in tqdm.tqdm(node_ids):
        rels = node2rels[node_id]
        rel_candidates = []

        for rel in rels:
            src_id = get_node_id(rel.source)
            tgt_id = get_node_id(rel.target)
            # 获取除了当前 node 以外的那个邻居 node_id
            neighbor_id = tgt_id if src_id == node_id else src_id

            # 查找邻居节点在另一张图里的最高相似度（因为上面已经排过序，所以取 [0][1] 即最大值）
            neighbor_max_sim = node2topk[neighbor_id][0][1] if neighbor_id in node2topk else 0.0
            rel_candidates.append((rel, neighbor_max_sim, rel.type, neighbor_id))

        # 按邻居节点的相似度降序排序，优先挑选邻居在匹配图中有良好对应关系（高相似度）的关系
        rel_candidates.sort(key=lambda x: x[1], reverse=True)

        picked_rels = []
        type_counter = {}  # 记录各类型被选中的次数

        for rel_info in rel_candidates:
            rel, sim, rel_type, neighbor_id = rel_info

            if sim < sim_threshold:
                continue

            current_type_count = type_counter.get(rel_type, 0)

            # 保证关系多样化：同类型的关系不要超过设定数量
            if current_type_count < max_same_type_rels:
                picked_rels.append(rel)
                type_counter[rel_type] = current_type_count + 1

            # 保证数量适中：不能太多
            if len(picked_rels) >= max_rels_per_node:
                break

        selected_rels_for_node[node_id] = picked_rels

    # 打印一下结果概况
    valid_nodes = sum(1 for v in selected_rels_for_node.values() if len(v) > 0) # 这个是有至少一条关系被选中的节点数量
    avg_rels = sum(len(v) for v in selected_rels_for_node.values()) / len(selected_rels_for_node) if len(selected_rels_for_node) > 0 else 0
    print(f"Finished selecting relations.")
    print(f"Nodes with selected relations: {valid_nodes}/{len(node_ids)}")
    print(f"Average selected relations per node: {avg_rels:.2f}")
    # 保存结果
    with open(out_file, "wb") as f:
        pickle.dump(selected_rels_for_node, f)
    """
    {
        node_id: [rel1, rel2, ...],
    }
    """
    print(f"Saved selected relations to {out_file}")

def gen_node_embedding(kg: GraphDocument, outfile: str, api_key, dimensions: int = 2048):
    node_embeddings = {}
    node_signatures = []
    for node in kg.nodes:
        signature = get_node_signature(node)
        node_signatures.append(signature)
    print(f"Generating embeddings for {len(node_signatures)} nodes...")

    embedding = BailianEmbeddings(api_key=api_key)
    node_embs = embedding.embed_documents(node_signatures, dimensions)
    print(f"Generated embeddings for {len(node_embs)} nodes.")
    for node, emb in zip(kg.nodes, node_embs):
        id = get_node_id(node)
        node_embeddings[id] = emb

    # 保存为pkl文件
    with open(outfile, "wb") as f:
        pickle.dump(node_embeddings, f)
    print(f"Saved node embeddings to {outfile}")
    return node_embeddings

def gen_node2name(kg: GraphDocument, out_file):
    node2name = {}
    for node in kg.nodes:
        node_id = get_node_id(node)
        node_name = node.properties.get("name", "")
        node2name[node_id] = node_name
    with open(out_file, "wb") as f:
        pickle.dump(node2name, f)
    print(f"Saved node2name mapping to {out_file}")
    return node2name

if __name__ == "__main__":
    # python preproces.py --dataset dbp_wiki/icews_wiki/icews_yago
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dbp_wiki", help="dataset name: dbp_wiki/icews_wiki/icews_yago")
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "dbp_wiki":
        input_dir = "../data/input_data/dbp_wiki"
        out_dir = "../data/output_data/dbp_wiki"
        graph1 = "../data/input_data/dbp_wiki/dbp_graph.pkl"
        graph2 = "../data/input_data/dbp_wiki/wiki_graph.pkl"
        kg1_name = "dbp"
        kg2_name = "wiki"
    elif dataset == "icews_wiki":
        input_dir = "../data/input_data/icews_wiki"
        out_dir = "../data/output_data/icews_wiki"
        graph1 = "../data/input_data/icews_wiki/icews_graph.pkl"
        graph2 = "../data/input_data/icews_wiki/wiki_graph.pkl"
        kg1_name = "icews"
        kg2_name = "wiki"
    elif dataset == "icews_yago":
        input_dir = "../data/input_data/icews_yago"
        out_dir = "../data/output_data/icews_yago"
        graph1 = "../data/input_data/icews_yago/icews_graph.pkl"
        graph2 = "../data/input_data/icews_yago/yago_graph.pkl"
        kg1_name = "icews"
        kg2_name = "yago"
    else:
        raise ValueError("Unknown dataset. Please choose from dbp_wiki/icews_wiki/icews_yago.")
    print(f"Processing dataset: {dataset}")
    print(f"Loading graphs from {graph1} and {graph2}...")
    kg1 = pickle.load(open(graph1, "rb"))
    kg2 = pickle.load(open(graph2, "rb"))
    print(f"Loaded graphs. KG1 has {len(kg1.nodes)} nodes and {len(kg1.relationships)} relationships. KG2 has {len(kg2.nodes)} nodes and {len(kg2.relationships)} relationships.")

    print("Generating node embeddings for KG1...")
    api_key = os.getenv("BAILIAN_API_KEY")
    if api_key is None:
        raise ValueError("Please set the BAILIAN_API_KEY environment variable.")
    node_emb_kg1 = gen_node_embedding(kg1, f"{out_dir}/{kg1_name}_node_embeddings.pkl", api_key=api_key)
    print("Generating node embeddings for KG2...")
    node_emb_kg2 = gen_node_embedding(kg2, f"{out_dir}/{kg2_name}_node_embeddings.pkl", api_key=api_key)

    print("Generating node to relationships mapping for KG1...")
    node2rels_kg1 = gen_id2rel(kg1, kg1_name, f"{out_dir}/{kg1_name}_node2rels.pkl")
    print("Generating node to relationships mapping for KG2...")
    node2rels_kg2 = gen_id2rel(kg2, kg2_name, f"{out_dir}/{kg2_name}_node2rels.pkl")

    print("Generating selected relationships for KG1 based on node embedding similarity...")
    gen_id2selected_rel(node_emb_kg1, node_emb_kg2, node2rels_kg1, f"{out_dir}/{kg1_name}_selected_rels.pkl")
    print("Generating selected relationships for KG2 based on node embedding similarity...")
    gen_id2selected_rel(node_emb_kg2, node_emb_kg1, node2rels_kg2, f"{out_dir}/{kg2_name}_selected_rels.pkl")

    gen_node2name(kg1, f"{out_dir}/{kg1_name}_node2name.pkl")
    gen_node2name(kg2, f"{out_dir}/{kg2_name}_node2name.pkl")
    print("Preprocessing completed.")