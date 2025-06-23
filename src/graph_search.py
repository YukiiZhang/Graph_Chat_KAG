from typing import List, Dict, Any, Tuple
import asyncio
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from logger import logger
from shared.prompts import (
    RELATION_SCORE_PROMPT,
    ENTITY_SCORE_PROMPT
)

class GraphSearch:
    """图搜索类，实现基于图数据库的搜索功能"""
    
    def __init__(self, graph: Neo4jGraph, embedding_function):
        self.graph = graph
        self.embedding_function = embedding_function
        self.vector_store = Neo4jVector.from_existing_index(
            embedding=embedding_function,
            index_name="entity_embedding_index",
            node_label="__Entity__",
            text_node_property="id",
            embedding_node_property="embedding",
            graph=self.graph
        )
    
    async def find_topic_entities(self, query: str, top_k: int = 3) -> List[str]:
        """查找问题中的主题实体"""
        try:
            # 使用向量搜索找到相关实体
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=top_k
            )
            
            # 提取实体ID
            topic_entities = [doc.page_content for doc in docs if doc.page_content]
            return topic_entities
            
        except Exception as e:
            logger.error(f"查找主题实体失败: {str(e)}")
            return []

    def _clean_relations(self, relations_text: str) -> List[Dict[str, Any]]:
        """清理和解析关系文本"""
        # 使用更灵活的正则表达式模式
        pattern = r"{(?P<relation>[^{}]+)\s*\(Score:\s*(?P<score>[0-9.]+)\)}"
        relations = []

        for match in re.finditer(pattern, relations_text):
            try:
                relation = match.group("relation").strip()
                score = float(match.group("score"))

                # 可选：添加更严格的分数验证
                if 0.0 <= score <= 1.0:
                    relations.append({
                        "relation": relation,
                        "score": score
                    })
                else:
                    logger.warning(f"无效的分数值: {score}，关系: {relation}")
            except (AttributeError, ValueError) as e:
                logger.error(f"解析关系时出错: {e}，匹配内容: {match.group(0)}")

        return relations

    async def _get_entity_relations(self, entity_id: str) -> Tuple[List[str], List[str]]:
        """获取实体的关系"""
        try:
            # 只获取出边关系，减少查询复杂度
            query = """
            MATCH (e:__Entity__ {id: $entity_id})-[r]->(n:__Entity__)
            RETURN DISTINCT type(r) as rel_type
            """
            results = self.graph.query(query, {"entity_id": entity_id})
            return [result["rel_type"] for result in results if result.get("rel_type")], []
            
        except Exception as e:
            logger.error(f"获取实体关系失败: {str(e)}")
            return [], []
    
    async def _score_relations(self, query: str, entity_ids: List[str], relations: List[str], llm) -> Dict[str, List[Dict[str, Any]]]:
        """批量对关系进行评分"""
        try:
            # 构建提示词
            relations_text = "\n".join([
                f"{i+1}. {entity_id} -> {rel}"
                for i, (entity_id, rel) in enumerate(zip(entity_ids, relations))
            ])
            
            prompt = RELATION_SCORE_PROMPT.format(
                question=query,
                entity="多个实体",
                relations=relations_text
            )
            
            # 调用LLM获取评分
            response = await llm.ainvoke(prompt)
            
            # 解析评分结果
            scored_relations = self._clean_relations(response.content)
            
            # 按实体ID组织结果
            result = {}
            for entity_id, rel, score in zip(entity_ids, relations, scored_relations):
                if entity_id not in result:
                    result[entity_id] = []
                result[entity_id].append({
                    "relation": rel,
                    "score": score["score"]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"关系评分失败: {str(e)}")
            return {}
    
    async def _get_related_entities_with_scores(self, entity_id: str, relations: List[str]) -> List[Dict[str, Any]]:
        """一次性获取所有相关实体及其关系"""
        try:
            # 使用 UNWIND 优化查询，一次性获取所有关系
            query = """
            UNWIND $relations as rel
            MATCH (e1:__Entity__ {id: $entity_id})-[r]->(e2:__Entity__)
            WHERE type(r) = rel
            RETURN e2.id as related_id, type(r) as relation
            """
            results = self.graph.query(query, {
                "entity_id": entity_id,
                "relations": relations
            })
            return [{"id": r["related_id"], "relation": r["relation"]} for r in results]
            
        except Exception as e:
            logger.error(f"获取相关实体失败: {str(e)}")
            return []
    
    async def _score_entities(self, query: str, entity_ids: List[str], relations: List[str], llm) -> List[Dict[str, Any]]:
        """批量对实体进行评分"""
        try:
            # 构建提示词
            entities_text = "\n".join([
                f"{i+1}. {entity_id} ({rel})"
                for i, (entity_id, rel) in enumerate(zip(entity_ids, relations))
            ])
            
            prompt = ENTITY_SCORE_PROMPT.format(
                question=query,
                relation="多个关系",
                entities=entities_text
            )
            
            # 调用LLM获取评分
            response = await llm.ainvoke(prompt)
            
            # 解析评分结果
            scored_entities = self._clean_relations(response.content)
            
            # 组织结果
            result = []
            for entity_id, rel, score in zip(entity_ids, relations, scored_entities):
                result.append({
                    "relation": entity_id,
                    "score": score["score"]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"实体评分失败: {str(e)}")
            return []
    
    async def search(self, query: str, llm, max_depth: int = 3, top_k: int = 3) -> List[Dict[str, Any]]:
        """执行图搜索"""
        try:
            # 1. 找到主题实体
            topic_entities = await self.find_topic_entities(query, top_k)
            if not topic_entities:
                return []
            
            # 2. 初始化结果存储
            chains = []
            current_entities = topic_entities
            
            # 3. 开始深度搜索
            for idx, depth in enumerate(range(max_depth)):
                logger.info(f"正在进行第{idx+1}轮深度搜索")
                
                new_chains = []
                
                # 3.1 并行获取所有实体的关系
                entity_relations_tasks = [
                    self._get_entity_relations(entity_id)
                    for entity_id in current_entities
                ]
                entity_relations_results = await asyncio.gather(*entity_relations_tasks)
                
                # 3.2 对关系进行批量评分
                all_relations = []
                for entity_id, (out_relations, _) in zip(current_entities, entity_relations_results):
                    if out_relations:
                        all_relations.extend([(entity_id, rel) for rel in out_relations])
                
                if all_relations:
                    # 批量评分所有关系
                    scored_relations = await self._score_relations(
                        query,
                        [rel[0] for rel in all_relations],
                        [rel[1] for rel in all_relations],
                        llm
                    )
                    
                    # 3.3 获取相关实体
                    for entity_id, relations in scored_relations.items():
                        related_entities = await self._get_related_entities_with_scores(
                            entity_id,
                            [rel["relation"] for rel in relations[:top_k]]
                        )
                        
                        if related_entities:
                            # 批量评分所有实体
                            scored_entities = await self._score_entities(
                                query,
                                [e["id"] for e in related_entities],
                                [e["relation"] for e in related_entities],
                                llm
                            )
                            
                            # 构建三元组
                            for rel, entity in zip(relations[:top_k], scored_entities[:top_k]):
                                new_chains.append({
                                    "head": entity_id,
                                    "relation": rel["relation"],
                                    "tail": entity["relation"],
                                    "score": rel["score"] * entity["score"]
                                })
                
                # 更新当前实体和结果
                current_entities = [chain["tail"] for chain in new_chains]
                chains.extend(new_chains)
                
                logger.info(f"第{idx+1}轮深度搜索完成，找到{len(new_chains)}个新三元组")
            
            return chains
            
        except Exception as e:
            logger.error(f"图搜索失败: {str(e)}")
            return [] 