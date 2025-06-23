from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class RetrieverResult:
    """检索结果类"""
    def __init__(self, documents: List[Document], scores: List[float], retriever_name: str):
        self.documents = documents
        self.scores = scores
        self.retriever_name = retriever_name
        self.metadata = {
            "retriever_name": retriever_name,
            "num_docs": len(documents),
            "avg_score": sum(scores) / len(scores) if scores else 0
        } 