import requests
import time
import json
import tqdm
import os
import re
from urllib.parse import unquote
from typing import List
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship, Document
import pickle


def extract_id(url):
    url = url.strip('<>')
    return url.split('/')[-1]

def get_rel_triple(input_file)->list[tuple]:
    triples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"警告: 跳过格式不正确的行: {line}")
                continue
            head, relation, tail = parts
            
            head_url = head.strip('<>') # 将可能存在的 <> 去掉，保证干净的 URI
            relation_url = relation.strip('<>')
            tail_url = tail.strip('<>')
            
            head_id = extract_id(head_url)
            relation_id = extract_id(relation_url)
            tail_id = extract_id(tail_url)
            
            triples.append((head_id, relation_id, tail_id))
            
    return triples

def extract_value(value):
    # "1985"^^<http://www.w3.org/2001/XMLSchema#gYear>
    # 可能还需要对Unicode转义字符进行处理
    if value.startswith('"') and '"^^' in value:
        val_part = value.split('"^^')[0].strip('"') # 只保留引号内的部分
        return val_part
    # Karma 如果没有“”引起来的话直接就是纯文本了
    # value = value.encode('utf-8').decode('unicode_escape')
    return value

def get_wiki_info(unique_ids, lang='en', fallback_lang='zh', max_retries=3):
    
    api_url = "https://www.wikidata.org/w/api.php"
    id_to_info = {}
    batch_size = 50
    ids_list = list(unique_ids)
    # 使用 Session 复用连接，提升稳定性
    session = requests.Session()
    session.headers.update({
        "User-Agent": "id2name-notebook/1.0 (contact: local-notebook)",
        "Accept": "application/json"
    })
    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i:i + batch_size]
        ids_str = '|'.join(batch)

        params = {
            "action": "wbgetentities",
            "ids": ids_str,
            "format": "json",
            "props": "labels", # 修改请求字段，加上 descriptions
            "languages": f"{lang}|{fallback_lang}"
        }
    
        try:
            response = session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("接口返回空内容")
            data = response.json()
            if 'entities' in data:
                for entity_id, entity_data in data['entities'].items():
                    labels = entity_data.get('labels', {})
                    # 解析 label
                    label_value = entity_id
                    if lang in labels:
                        label_value = labels[lang]['value']
                    elif fallback_lang in labels:
                        label_value = labels[fallback_lang]['value']
                    # 保存为字典格式返回
                    id_to_info[entity_id] = {
                        "name": label_value
                    }
        except (requests.RequestException, ValueError) as e:
            print(f"批次 {i}-{i+len(batch)} 失败: {e}")
        time.sleep(0.2)
    success_num = 0
    for qid, info in id_to_info.items():
        if info["name"] != qid:
            success_num += 1
    print(f"成功获取到 {success_num}/{len(id_to_info)} 个有效的 name")
    return id_to_info

def get_attr_triple(input_file):
    triples = []
    with open(input_file, 'r') as f:
        triples = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"警告: 跳过格式不正确的行: {line}")
                continue
            qid, pid, value = parts
            qid = extract_id(qid)
            pid = extract_id(pid)
            value = extract_value(value)
            triples.append((qid, pid, value))

def process_dbp_wiki(data_path="../data/input_data/dbp_wiki"):
    def save_graph_doc(kg_type, rel_path, attr_path, output_path):
        triples = get_rel_triple(rel_path)
        attrs = get_attr_triple(attr_path)
        qids = set([h for h, _, _ in triples] + [t for _, _, t in triples] + [qid for qid, _, _ in attrs])
        pids = set([r for _, r, _ in triples] + [pid for _, pid, _ in attrs])
        if kg_type == "wiki":
            pid2info = get_wiki_info(pids)
            qid2info = get_wiki_info(qids)
        else:
            pid2info = {}
            qid2info = {}
            for h,r,t in triples:
                qid2info[h] = {"name": h.replace('_', ' ')}
                pid2info[r] = {"name": r.replace('_', ' ')}
                qid2info[t] = {"name": t.replace('_', ' ')}
            for h,r,v in attrs:
                qid2info[h] = {"name": h.replace('_', ' ')}
                pid2info[r] = {"name": r.replace('_', ' ')}
        # 将attr triples转换为 {qid: {pid: value}} 的格式，方便后续查询
        attr_dict={}
        for qid, pid, value in attrs:
            if qid not in attr_dict:
                attr_dict[qid] = {}
            attr_dict[qid][pid] = value
        # 转换为Graph
        relationships:List[Relationship] = []
        processed_nodes = {}
        for h,r,t in triples:
            if h not in processed_nodes:
                head_info = qid2info.get(h, {"name": h})
                attrs = attr_dict.get(h, {})
                attr_trans = {}
                for pid, value in attrs.items():
                    pid_name = pid2info.get(pid, {"name": pid})["name"]
                    attr_trans[pid_name] = value
                node = Node(
                    id=h,
                    properties={
                        "name": head_info["name"],
                        **attr_trans
                    }
                )
                processed_nodes[h] = node
            if t not in processed_nodes:
                tail_info = qid2info.get(t, {"name": t})
                attrs = attr_dict.get(t, {})
                attr_trans = {}
                for pid, value in attrs.items():
                    pid_name = pid2info.get(pid, {"name": pid})["name"]
                    attr_trans[pid_name] = value
                node = Node(
                    id=t,
                    properties={
                        "name": tail_info["name"],
                        **attr_trans
                    }
                )
                processed_nodes[t] = node
            rel_name = pid2info.get(r, {"name": r})["name"]
            relationship = Relationship(
                source=processed_nodes[h],
                type=rel_name,
                target=processed_nodes[t]
            )
            relationships.append(relationship)
        nodes = list(processed_nodes.values())
        source:Document = Document(page_content=data_path)
        graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=source)
        with open(output_path,"wb") as f:
            pickle.dump(graph_doc, f)
    
    wiki_triples = data_path + "/rel_triples_2"
    dbp_triples = data_path + "/rel_triples_1"
    wiki_attr_triples = data_path + "/attr_triples_2"
    dbp_attr_triples = data_path + "/attr_triples_1"
    ref_pairs = data_path + "/ent_links"
    output_path1 = data_path + "/dbp_graph.pkl"
    output_path2 = data_path + "/wiki_graph.pkl"
    save_graph_doc("dbp", dbp_triples, dbp_attr_triples, output_path1)
    save_graph_doc("wiki", wiki_triples, wiki_attr_triples, output_path2)
    # 由ref_pairs生成dbp2wiki.json
    dbp2wiki = {}
    with open(ref_pairs, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            dbp_id = extract_id(parts[0])
            wiki_id = extract_id(parts[1])
            dbp2wiki[dbp_id] = wiki_id
    with open(data_path + "/dbp2wiki.json", "w") as f:
        json.dump(dbp2wiki, f, indent=4)

                


def process_icews(data_path="../data/input_data/icews_wiki"):
    def load_ents(path):
        data = {} # id to name
        with open(path,'r') as f:
            for line in f:
                line = line.replace("\n", "").split("\t")
                ent_id = line[0]
                ent_name = line[1].split('/')[-1] # https://en.wikipedia.org/wiki/United_States -> United_States
                ent_name = ent_name.replace('_', ' ') # United_States -> United States
                # 将%C3%A9这种URL编码转换为é, 再将é转换为e
                ent_name = unquote(ent_name)
                data[ent_id] = ent_name
        print('load %s %d'%(path,len(data)))
        return data
    def load_rel(path):
        data = {} # id to name
        with open(path,'r') as f:
            for line in f:
                line = line.strip().split('\t')
                rel_id = line[0]
                rel_name = line[1]
                data[rel_id] = rel_name
        print('load %s %d'%(path,len(data)))
        return data
    def load_kg(path):
        data = [] # (h, r, t, start_time, end_time)
        with open(path,'r') as f:
            for line in f: # 对于 xxx\txxx xxx\txxx\txxx先按\t分为 xxx, xxx xxx, xxx, xxx 再对第二个xxx xxx按空格分为xxx和xxx
                line = line.replace("\n", "").split("\t")
                # 对第二个xxx xxx按空格分为xxx和xxx
                if len(line) == 4:
                    line[1] = line[1].split(' ')
                    h = line[0]
                    r = line[1][0]
                    t = line[1][1]
                    start_time = line[2]
                    end_time = line[3]
                else:
                    h = line[0]
                    r = line[1]
                    t = line[2]
                    start_time = line[3]
                    end_time = line[4]
                data.append((h,r,t,start_time,end_time))
        print('load %s %d'%(path,len(data)))
        return data
    
    def load_time(path):
        data = {} # id to time
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split('\t')
                id = line[0]
                time = line[1]
                data[id] = time
        print('load %s %d'%(path,len(data)))
        return data

    def process_specical_word(c):
        rules = [[['à','á','â','ã','ä','å','ā','ă','ạ','ả','ấ','ầ','ẩ','ẫ','ậ','ắ','ằ','ẵ','а'],'a'],
                [['Á','Å'],'A'],
                [['в'],'b'],
                [['ç','ć','Ċ','č','с'],'c'],
                [['Ç','Ć','Č'],'C'],
                [['ð'],'d'],
                [['Đ'],'D'],
                [['Æ','è','é','ê','ë','ė','ē','ę','ě','ế','ề','ễ','ệ'],'e'],
                [['É'],'E'],
                [['ğ'],'g'],
                [['н'],'h'],
                [['ì','í','î','ï','ĩ','ī','ı','ľ','ị','Î'],'i'],
                [['Ď'],'J'],
                [['ķ'],'k'],
                [['ł','İ'],'l'],
                [['Ľ','Ł'],'L'],
                [['ṃ'],'m'],
                [['ñ','ń','ň','ņ'],'n'],
                [['И'],'N'],
                [['ø','ò','ó','ô','õ','ö','ō','ő','ũ','ơ','ọ','ố','ồ','ổ','ộ','ớ','ờ','ở','ợ','о'],'o'],
                [['Ö','Ø','Ó','Ō'],'O'],
                [['р'],'p'],
                [['œ'],'oe'],
                [['ř'],'r'],
                [['Þ','ś','ş','š','ș'],'s'],
                [['Ś','Ş','Š'],'S'],
                [['ß'],'ss'],
                [['ț'],'t'],
                [['ù','ú','ü','ū','ű','ư','ụ','ủ','ứ','ừ','ữ','ự'],'u'],
                [['Ú','Ü'],'U'],
                [['ý','ỹ','ÿ','ỳ'],'y'],
                [['ž'],'z'],
                [['Ž'],'Z'],
                ]
        for rule,replace in rules:
            if c in rule:
                return replace
        return c

    id2time = load_time(time_)
    # 先处理kg1
    def save_graph_doc(entity_path, relation_path, kg_path, output_path):
        ents1 = load_ents(entity_path)
        rels1 = load_rel(relation_path)
        kg1 = load_kg(kg_path)
        id2node = {}
        relationships:List[Relationship] = []
        import tqdm
        for h,r,t,start_time,end_time in tqdm.tqdm(kg1):
            h_name = ents1[h]
            r_name = rels1[r]
            t_name = ents1[t]
            start_time = id2time[start_time]
            end_time = id2time[end_time]
            h_name = ''.join([process_specical_word(c) for c in h_name])
            r_name = ''.join([process_specical_word(c) for c in r_name])
            t_name = ''.join([process_specical_word(c) for c in t_name])
            if h not in id2node:
                id2node[h] = Node(id=h, name=h_name)
            if t not in id2node:
                id2node[t] = Node(id=t, name=t_name)
            rel = Relationship(
                source=id2node[h],
                type=r_name,
                target=id2node[t],
                properties={"start_time": start_time, "end_time": end_time}
            )
            relationships.append(rel)
        nodes = list(id2node.values())
        source = Document(page_content=data_path)
        graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=source)
        with open(output_path,"wb") as f:
            pickle.dump(graph_doc, f)

    if data_path.endswith("icews_wiki"):
        kg1_type = "icews"
        kg2_type = "wiki"
    else:
        kg1_type = "icews"
        kg2_type = "yago"
    entity1 = data_path + "/emt_ids_1"
    entity2 = data_path + "/emt_ids_2"
    relation1 = data_path + "/rel_ids_1"
    relation2 = data_path + "/rel_ids_2"
    kg1 = data_path + "/triplets_1"
    kg2 = data_path + "/triplets_2"
    time_ = data_path + "/time_id"
    ref_pairs = data_path + "/ref_ent_ids"
    output_path1 = data_path + "/{kg1_type}_graph.pkl"
    output_path2 = data_path + "/{kg2_type}_graph.pkl"
    save_graph_doc(entity1, relation1, kg1, output_path1)
    save_graph_doc(entity2, relation2, kg2, output_path2)
    with open(ref_pairs, "r") as f:
        ref_dict = {}
        for line in f:
            line = line.strip().split('\t')
            id1 = line[0]
            id2 = line[1]
            ref_dict[id1] = id2
    with open(data_path + f"{kg1_type}2{kg2_type}.json", "w") as f:
        json.dump(ref_dict, f, indent=4)

if __name__ == "__main__":
    # python raw2KG.py --dataset dbp_wiki/icews_wiki/icews_yago
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dbp_wiki", help="dataset name: dbp_wiki/icews_wiki/icews_yago")
    args = parser.parse_args()
    if args.dataset == "dbp_wiki":
        process_dbp_wiki("../data/input_data/dbp_wiki")
    elif args.dataset == "icews_wiki":
        process_icews("../data/input_data/icews_wiki")
    elif args.dataset == "icews_yago":
        process_icews("../data/input_data/icews_yago")