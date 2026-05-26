# import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
# from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
# from langchain_core.documents import Document
# import numpy as np
import requests
# from config import EMBEDDING_CONFIG,SIMILARITY_THRESHOLD
# import logging
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

import os
from openai import OpenAI
from typing import List
import logging

# logger = logging.getLogger(__name__)

class BailianEmbeddings:
    """百炼平台OpenAI兼容模式嵌入服务"""
    def __init__(self, api_key: str = None, model: str = "text-embedding-v4", url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化百炼嵌入服务
        
        参数:
            api_key: 百炼API Key（从环境变量DASHSCOPE_API_KEY读取或直接传入）
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.url = url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.url
        )
    
    def embed_documents(self, texts: List[str], dimensions: int = 2048) -> List[List[float]]:
        """为文档列表生成嵌入向量（添加分批处理）"""
        if not texts:
            return []
        
        # 百炼平台单次请求最多支持10个文本
        batch_size = 10
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        import tqdm
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # logger.debug(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch)} 个文本")
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=dimensions,
                    encoding_format="float"
                )
                
                # 提取嵌入向量并添加到结果列表
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
                
                # logger.debug(f"批次 {batch_num} 处理成功")
                
            except Exception as e:
                print(e)
                # logger.error(f"百炼嵌入请求失败（批次 {batch_num}）: {str(e)}")
                # 对于失败批次，返回空向量占位符
                embeddings.extend([[]] * len(batch))
                # logger.warning(f"已为批次 {batch_num} 添加空嵌入占位符")
        
        return embeddings
    
    def embed_query(self, text: str, dimensions: int = 1024) -> List[float]:
        """为单个查询生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            # logger.error(f"百炼查询嵌入失败: {str(e)}")
            raise

class LocalEmbeddings:
    """封装本地嵌入服务的类"""
    def __init__(self, base_url: str, model: str, api_key: str = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        # self.embedding_endpoint = f"{base_url}/embeddings"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
    def embed_documents(self, texts: List[str], dimensions: int = 2048) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        if not texts:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=dimensions,
                encoding_format="float"
            )
            embeddings = [embedding.embedding for embedding in response.data]
            return embeddings
        
        except requests.exceptions.RequestException as e:
            print(e)
            raise
        except (KeyError, ValueError) as e:
            print(e)
            raise
    def embed_query(self, text: str, dimensions: int = 1024) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
