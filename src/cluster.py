from pathlib import Path
import sys
import json
import numpy as np
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import math
from typing import Union, List, Dict, Tuple, Any
from preprocess import get_node_id, get_node_signature, get_relationship_signature, get_node_signature_simple
import pickle
from torch import Tensor
from torch.nn import functional as F
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pickle

def gen_clusters(node_embeddings: Dict[str, List[float]], kg: GraphDocument, out_file: str, cluster_num: int=300, overlap: int=3):
    
    node_ids = list(node_embeddings.keys())
    id_to_node = {get_node_id(n): n for n in kg.nodes}

    X = np.array([node_embeddings[nid] for nid in node_ids])

    # 依旧需要L2归一化来保证余弦相似度和欧式距离一致
    X_norm = normalize(X)
    num_clusters = cluster_num
    print(f"Start clustering {len(X_norm)} nodes into {num_clusters} clusters...")
    # 1. 训练KMeans获得质心向量 (代表类的embedding)
    kmeans = KMeans(n_clusters=num_clusters, random_state=20)
    kmeans.fit(X_norm)
    centroids = kmeans.cluster_centers_
    # 初始化保存聚类信息的列表
    clusters_info = [{"id": i, "nodes": [], "embedding": centroids[i]} for i in range(num_clusters)]
    top_k_overlap = overlap  # 每个节点会被归入最相似的5个簇中。如果依然漏掉，可以增大这个值。

    # 计算所有节点到所有质心的余弦相似度
    similarities = cosine_similarity(X_norm, centroids) # shape: (N, num_clusters)

    # np.argsort默认升序，所以最后几个是相似度最大的
    top_k_indices = np.argsort(similarities, axis=1)[:, -top_k_overlap:] # shape: (N, top_k_overlap)，每行是该节点最相似的top_k_overlap个簇的索引

    for i, node_id in enumerate(node_ids):
        node = id_to_node[node_id]
        # 将该节点加入到对它来说最相似的 top-k 个簇中
        for cluster_id in top_k_indices[i]:
            clusters_info[cluster_id]["nodes"].append(node)

    # 输出各类别的大小，观察是否均衡
    sizes = [len(c["nodes"]) for c in clusters_info]
    print("Cluster sizes:", sizes)
    print(f"Min size: {min(sizes)}, Max size: {max(sizes)}, Mean size: {np.mean(sizes)}")

    with open(out_file, "wb") as f:
        pickle.dump(clusters_info, f)
    """
    [
        {
            "id": 0,
            "nodes": [Node(...), Node(...), ...],  # 属于这个簇的节点列表
            "embedding": np.array([...])  # 代表这个簇的质心向量
        },
        ...
    ]
    """
    print(f"Saved cluster info to {out_file}")
    return clusters_info

def gen_node_to_cluster(node_embeddings_1: Dict[str, List[float]], clusters_info_2: List[Dict[str, Any]], out_file: str, cluster_num_for_each_node: int=3):
    kg1_to_kg2_node_to_cluster = {}
    import tqdm
    for node_id, embedding in tqdm.tqdm(node_embeddings_1.items()):
        # 计算当前节点和另一个kg中各个簇的相似度，将该节点分配给相似度最高的若干个簇
        assigned_cluster_id = [] # 可以分配给多个簇, 保存当前匹配到的簇id以及相似度
        embedding = Tensor(embedding)
        for cluster_info in clusters_info_2:
            cluster_embedding = Tensor(cluster_info["embedding"])
            sim = F.cosine_similarity(embedding, cluster_embedding, dim=0).item()
            if len(assigned_cluster_id) < cluster_num_for_each_node:
                assigned_cluster_id.append((cluster_info["id"], sim))
            else:
                # 如果当前簇的相似度比已分配簇中相似度最低的还高，则替换掉相似度最低的簇
                min_sim = min(assigned_cluster_id, key=lambda x: x[1])[1]
                if sim > min_sim:
                    min_index = assigned_cluster_id.index(min(assigned_cluster_id, key=lambda x: x[1]))
                    assigned_cluster_id[min_index] = (cluster_info["id"], sim)
        # 最后按照相似度从高到低排序
        kg1_to_kg2_node_to_cluster[node_id] = sorted(assigned_cluster_id, key=lambda x: x[1], reverse=True)
        # break # 先测试一个节点
    # 保存kg1_to_kg2_node_to_cluster
    with open(out_file, "wb") as f:
        pickle.dump(kg1_to_kg2_node_to_cluster, f)
    """
    dict[node_id, cluster_id] = {
        node_id: str,
        cluster_id: list[truple(cluster_id, similarity)]
    }
    """
    return kg1_to_kg2_node_to_cluster

def gen_recall_dict(kg1: GraphDocument, kg1_to_kg2_node_to_cluster: Dict[str, List[Tuple[int, float]]], clusters_info_2: List[Dict[str, Any]], kg1id2kg2id: Dict[str, str], out_file: str):
    id2node = {}
    for node in kg1.nodes:
        id2node[node.id] = node
    node_num = 0
    matched_num = 0
    matched_nodes = set()
    import tqdm
    for node_id, cluster_id_list in tqdm.tqdm(kg1_to_kg2_node_to_cluster.items()):
        matched_id = kg1id2kg2id.get(node_id)
        if not matched_id:
            continue
        node_num += 1
        for i in range(len(cluster_id_list)):
            cluster_id, sim = cluster_id_list[i]
            cluster_info = {}
            for cluster_ in clusters_info_2:
                if cluster_["id"] == cluster_id:
                    cluster_info = cluster_
                    break
            cluster_nodes:List[Node] = cluster_info["nodes"]
            same_name_nodes = [node for node in cluster_nodes if node.id == matched_id]
            if same_name_nodes:
                matched_num += 1
                matched_nodes.add(node_id)
                break
    print(f"len of kg1id2kg2id: {len(kg1id2kg2id)}") 
    print(f"Total nodes: {node_num}, Matched nodes: {matched_num}, Matching ratio: {matched_num/node_num:.4f}")

    matched_node_ids = set(matched_nodes)
    matched_nodes_to_cluster_node = {}
    """
    {
        node_id: set(Node_id) # 该节点匹配的簇中的所有节点
    }
    """
    for node_id in tqdm.tqdm(matched_node_ids):
        cluster_list = kg1_to_kg2_node_to_cluster[node_id] # list of (cluster_id, sim)
        cluster_list = [cluster_id for cluster_id, sim in cluster_list] # list of cluster_id
        matched_cluster_nodes = set()
        for cluster_id in cluster_list:
            cluster_info = {}
            for cluster_ in clusters_info_2:
                if cluster_["id"] == cluster_id:
                    cluster_info = cluster_
                    break
            cluster_nodes:List[Node] = cluster_info["nodes"]
            for node in cluster_nodes:
                # unhashable的Node对象不能直接放入set中，为Node自定义一个hash方法，直接根据节点的id来判断是否相同
                matched_cluster_nodes.add(node.id)
        matched_nodes_to_cluster_node[node_id] = matched_cluster_nodes

    # 计算平均每个matched_node匹配到的cluster_node数量
    total_cluster_nodes = sum(len(cluster_nodes) for cluster_nodes in matched_nodes_to_cluster_node.values())
    average_cluster_nodes = total_cluster_nodes / len(matched_nodes_to_cluster_node) if matched_nodes_to_cluster_node else 0
    print(f"Average number of cluster nodes matched to each matched node: {average_cluster_nodes:.2f}")
    with open(out_file, "wb") as f:
        pickle.dump(matched_nodes_to_cluster_node, f)


if __name__ == "__main__":
    # python cluster.py --dataset dbp_wiki/icews_wiki/icews_yago --cluster_num 300 --overlap 3
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dbp_wiki", help="dataset name: dbp_wiki/icews_wiki/icews_yago")
    parser.add_argument("--cluster_num", type=int, default=300, help="number of clusters to generate")
    parser.add_argument("--overlap", type=int, default=3, help="number of clusters each node can belong to")
    args = parser.parse_args()
    dataset = args.dataset
    cluster_num = args.cluster_num
    overlap = args.overlap
    if dataset == "dbp_wiki":
        in_data_dir = "../data/output_data/dbp_wiki"
        out_data_dir = "../data/output_data/dbp_wiki/clusters"
        source_dir = "../data/input_data/dbp_wiki"
        graph1_path = f"{source_dir}/dbp_graph.pkl"
        graph2_path = f"{source_dir}/wiki_graph.pkl"
        kg1_name = "dbp"
        kg2_name = "wiki"
        ref_pairs_path = f"{source_dir}/dbp2wiki.json"
    elif dataset == "icews_wiki":
        in_data_dir = "../data/output_data/icews_wiki"
        out_data_dir = "../data/output_data/icews_wiki/clusters"
        source_dir = "../data/input_data/icews_wiki"
        graph1_path = f"{source_dir}/icews_graph.pkl"
        graph2_path = f"{source_dir}/wiki_graph.pkl"
        kg1_name = "icews"
        kg2_name = "wiki"
        ref_pairs_path = f"{source_dir}/icews2wiki.json"
    elif dataset == "icews_yago":
        in_data_dir = "../data/output_data/icews_yago"
        out_data_dir = "../data/output_data/icews_yago/clusters"
        source_dir = "../data/input_data/icews_yago"
        graph1_path = f"{source_dir}/icews_graph.pkl"
        graph2_path = f"{source_dir}/yago_graph.pkl"
        kg1_name = "icews"
        kg2_name = "yago"
        ref_pairs_path = f"{source_dir}/icews2yago.json"
    else:
        raise ValueError("Unknown dataset. Please choose from dbp_wiki/icews_wiki/icews_yago.")
    
    with open(graph1_path, "rb") as f:
        kg1 = pickle.load(f)
    with open(graph2_path, "rb") as f:        
        kg2 = pickle.load(f)
    with open(ref_pairs_path, "r") as f:
        ref_pairs = json.load(f)

    node_embeddings1 = pickle.load(open(f"{in_data_dir}/{kg1_name}_node_embeddings.pkl", "rb"))
    # clusters_info_1 = gen_clusters(node_embeddings1, kg1, f"{out_data_dir}/{kg1_name}_clusters.pkl", cluster_num=cluster_num, overlap=overlap)
    node_embeddings2 = pickle.load(open(f"{in_data_dir}/{kg2_name}_node_embeddings.pkl", "rb"))
    clusters_info_2 = gen_clusters(node_embeddings2, kg2, f"{out_data_dir}/{kg2_name}_clusters.pkl", cluster_num=cluster_num, overlap=overlap)

    kg1_to_kg2_node_to_cluster = gen_node_to_cluster(node_embeddings1, clusters_info_2, f"{out_data_dir}/kg1_to_kg2_node_to_cluster.pkl", cluster_num_for_each_node=overlap)
    gen_recall_dict(kg1, kg1_to_kg2_node_to_cluster, clusters_info_2, ref_pairs, f"{out_data_dir}/matched_nodes_to_cluster_nodes.pkl")
