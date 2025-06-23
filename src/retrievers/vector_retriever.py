from typing import List, Dict, Any
import os
import asyncio
from langchain_core.documents import Document
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from retrievers.base_retriever import RetrieverResult
from shared.common_fn import load_embedding_model
from logger import logger

# 设置tokenizers并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, graph, index_name: str = "vector", k: int = 5, score_threshold: float = 0.7):
        self.graph = graph
        self.index_name = index_name
        self.k = k
        self.score_threshold = score_threshold
        self.vector_store = self.initialize_vector_store()
    
    def initialize_vector_store(self) -> Neo4jVector:
        """初始化向量存储"""
        try:
            embedding_function, _ = load_embedding_model()
            vector_store = Neo4jVector.from_existing_index(
                embedding=embedding_function,
                index_name=self.index_name,
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
                graph=self.graph
            )
            return vector_store
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            return None
    
    async def retrieve(self, query: str, **kwargs) -> RetrieverResult:
        """执行向量检索"""
        try:
            if not self.vector_store:
                raise ValueError("向量存储未初始化")
            
            # 获取检索参数
            k = kwargs.get('k', self.k)
            score_threshold = kwargs.get('score_threshold', self.score_threshold)
            
            # 使用asyncio.to_thread包装同步调用
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=k
            )
            
            if not docs:
                return RetrieverResult([], [], "vector")
            
            # 提取文档和分数
            documents = []
            scores = []
            
            for doc, score in docs:
                if isinstance(doc, Document):
                    documents.append(doc)
                    scores.append(float(score))
            
            return RetrieverResult(
                documents=documents,
                scores=scores,
                retriever_name="vector"
            )
            
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return RetrieverResult([], [], "vector") 