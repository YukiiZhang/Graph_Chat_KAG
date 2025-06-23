from typing import List, Dict, Any
import asyncio
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph

from retrievers.vector_retriever import VectorRetriever
from retrievers.reranker import Reranker
from logger import logger

class RetrievalService:
    """检索服务类，负责管理所有检索相关的操作"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.vector_retriever = VectorRetriever(graph)

    def get_vector_retriever(self, k=5, score_threshold=0.7):
        """获取向量检索器"""
        try:

            vector_store = self.vector_retriever.vector_store

            if not vector_store:
                raise ValueError("向量存储未初始化")

            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": k,
                    "score_threshold": score_threshold
                }
            )

            logger.info("retrieval_service ： get_vector_retriever() -->  获取向量检索器成功")
            return retriever
        except Exception as e:
            logger.error(f"创建向量检索器失败: {str(e)}")
            return None

    async def vector_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """执行向量搜索"""
        try:
            # 使用向量检索器
            result = await self.vector_retriever.retrieve(query, k=k)
            
            if not result.documents:
                return {
                    "success": False,
                    "message": "未找到相关文档",
                    "documents": [],
                    "metadata": result.metadata
                }
            
            # 准备返回结果
            sources = []
            for doc, score in zip(result.documents, result.scores):
                source_info = doc.metadata.copy()
                source_info.update({
                    "score": score,
                    "retriever": result.retriever_name
                })
                sources.append(source_info)
            
            return {
                "success": True,
                "documents": result.documents,
                "sources": sources,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return {
                "success": False,
                "message": f"搜索过程出错: {str(e)}",
                "documents": [],
                "metadata": {}
            }

    async def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """执行混合检索"""
        try:
            # 1. 执行向量检索
            vector_results = await self.vector_retriever.retrieve(
                query,
                k=top_k * 2  # 获取更多候选文档用于重排序
            )
            
            # 2. 执行重排序
            reranker = Reranker()
            reranked_results = await reranker.rerank(
                query,
                [vector_results],
                top_k=top_k
            )
            
            if not reranked_results.documents:
                return {
                    "success": False,
                    "message": "未找到相关文档",
                    "documents": [],
                    "metadata": {
                        "vector_retriever": vector_results.metadata,
                        "reranker": reranked_results.metadata
                    }
                }
            
            # 3. 准备返回结果
            sources = []
            for doc, score in zip(reranked_results.documents, reranked_results.scores):
                source_info = doc.metadata.copy()
                source_info.update({
                    "score": score,
                    "retriever": reranked_results.retriever_name
                })
                sources.append(source_info)
            
            return {
                "success": True,
                "documents": reranked_results.documents,
                "sources": sources,
                "metadata": {
                    "vector_retriever": vector_results.metadata,
                    "reranker": reranked_results.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            return {
                "success": False,
                "message": f"检索过程出错: {str(e)}",
                "documents": [],
                "metadata": {}
            }
    
    async def get_context(self, query: str, top_k: int = 5, use_hybrid: bool = False):
        """获取检索上下文"""
        try:
            if use_hybrid:
                result = await self.hybrid_search(query, top_k)
            else:
                result = await self.vector_search(query, top_k)
            
            if not result["success"] or not result["documents"]:
                return ""
            
            # 合并文档内容
            return result, "\n".join([doc.page_content for doc in result["documents"]])
            
        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            return "" 