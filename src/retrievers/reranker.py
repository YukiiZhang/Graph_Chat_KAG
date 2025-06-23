from typing import List, Tuple
import asyncio
import os
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from retrievers.base_retriever import RetrieverResult
from logger import logger

# 设置tokenizers并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Reranker:
    """重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称，默认使用较小的base版本
        """
        self.model_name = model_name
        self.reranker = self._initialize_reranker()
    
    def _initialize_reranker(self) -> HuggingFaceCrossEncoder:
        """初始化重排序模型"""
        try:
            logger.info(f"正在初始化重排序模型: {self.model_name}")
            # logger.info("首次运行需要下载模型，这可能需要几分钟时间...")
            
            # 移除不支持的参数
            return HuggingFaceCrossEncoder(model_name=self.model_name)
        except Exception as e:
            logger.error(f"初始化重排序模型失败: {str(e)}")
            logger.error("请检查网络连接或尝试使用其他模型")
            return None
    
    async def rerank(self, query: str, retriever_results: List[RetrieverResult], top_k: int = 5) -> RetrieverResult:
        """对检索结果进行重排序"""
        try:
            if not self.reranker:
                raise ValueError("重排序模型未初始化")
            
            # 合并所有检索结果
            all_docs = []
            all_scores = []
            all_retriever_names = []
            
            for result in retriever_results:
                all_docs.extend(result.documents)
                all_scores.extend(result.scores)
                all_retriever_names.extend([result.retriever_name] * len(result.documents))
            
            if not all_docs:
                return RetrieverResult([], [], "reranked")
            
            # 准备重排序数据
            pairs = [(query, doc.page_content) for doc in all_docs]
            
            # 异步执行重排序
            rerank_scores = await asyncio.to_thread(
                self.reranker.score,
                pairs
            )
            
            # 将重排序分数与原始分数结合
            combined_scores = [
                (doc, rerank_score * 0.7 + orig_score * 0.3, retriever_name)
                for doc, rerank_score, orig_score, retriever_name
                in zip(all_docs, rerank_scores, all_scores, all_retriever_names)
            ]
            
            # 按分数排序并选择top_k
            sorted_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
            
            # 分离结果
            reranked_docs = [item[0] for item in sorted_results]
            reranked_scores = [item[1] for item in sorted_results]
            
            return RetrieverResult(
                documents=reranked_docs,
                scores=reranked_scores,
                retriever_name="reranked"
            )
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return RetrieverResult([], [], "reranked") 