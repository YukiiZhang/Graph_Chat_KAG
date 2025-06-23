from logger import *
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_neo4j import Neo4jGraph
from typing import List, Dict, Any, Optional, Set, Tuple
import asyncio
import json
import os
from datetime import datetime
from langchain_community.graphs.graph_document import Node, Relationship
from document_loader import DocumentLoader
from shared.prompts import GRAPH_PROMPT
from shared.prompts import GRAPH_PROMPT_COMMON



def ensure_vector_indexes(graph, dimension: int):
    """
    确保Neo4j数据库中创建了必要的向量索引
    
    Args:
        graph: Neo4j图数据库连接
        dimension: 向量维度
    """
    try:
        # 检查并创建Chunk节点的向量索引
        create_chunk_index_query = """
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: $dimension,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        graph.query(create_chunk_index_query, {"dimension": dimension})
        
        # 检查并创建Entity节点的向量索引
        create_entity_index_query = """
        CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
        FOR (e:__Entity__)
        ON (e.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: $dimension,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        graph.query(create_entity_index_query, {"dimension": dimension})
        
        logger.info("向量索引创建成功")
    except Exception as e:
        logger.error(f"创建向量索引失败: {e}")
        raise

def create_graph_transformer(
    llm: BaseLanguageModel,
    allowed_nodes: Optional[List[str]] = None,
    allowed_relationships: Optional[List[str]] = None,
) -> LLMGraphTransformer:
    """
    创建知识图谱转换器
    """
    try:
        # 创建转换器 - 移除allowed_nodes和allowed_relationships参数
        transformer = LLMGraphTransformer(
            llm=llm,
            prompt=GRAPH_PROMPT,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            # ignore_tool_usage=True,
        )
        
        return transformer
        
    except Exception as e:
        logger.error(f"创建图转换器失败: {e}")
        raise

async def generate_graph(
    llm: BaseLanguageModel,
    content: str,
    graph: Neo4jGraph,
    embedding_function: Any,
    chunk_id: str
) -> Dict[str, Any]:
    """
    使用LLM从文本内容生成知识图谱
    """
    try:
        # 1. 创建图转换器
        llm_transformer = create_graph_transformer(llm)
        
        # 2. 将文本转换为Document对象
        documents = [Document(page_content=content)]
        
        # 3. 将文本转换为图文档
        try:
            graph_documents = await asyncio.to_thread(
                llm_transformer.convert_to_graph_documents,
                documents
            )
        except Exception as e:
            logger.error(f"转换图文档失败: {e}")
            return {"error": f"转换图文档失败: {str(e)}"}
        
        if not graph_documents:
            return {"error": "未能生成图数据"}
        
        # 处理节点
        for node in graph_documents[0].nodes:
            logger.info(f"处理节点: {node.id} ({node.type})")
            # 计算节点嵌入
            try:
                node_embedding = embedding_function.embed_query(node.id)
            except AttributeError:
                try:
                    node_embedding = embedding_function.encode(node.id)
                except AttributeError:
                    node_embedding = embedding_function(node.id)

            # 创建或更新节点
            create_node_query = """
            MERGE (e:__Entity__ {id: $id})
            SET e.type = $type,
                e.embedding = $embedding,
                e.description = $description
            WITH e
            CALL apoc.create.addLabels(e, [$type]) YIELD node
            RETURN node
            """
            try:
                node_type = str(node.type) if node.type else "Unknown"
                graph.query(create_node_query, {
                    "id": node.id,
                    "type": node_type,
                    "embedding": node_embedding,
                    "description": node.properties.get("description", "")
                })
                logger.info(f"成功创建节点: {node.id} (标签: __Entity__, {node_type})")
            except Exception as e:
                logger.error(f"创建节点失败: {node.id}, 错误: {e}")
                continue

            # 创建Chunk与Entity的关系
            create_chunk_entity_relation_query = """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:__Entity__ {id: $entity_id})
            MERGE (c)-[:HAS_ENTITY]->(e)
            """
            try:
                graph.query(create_chunk_entity_relation_query, {
                    "chunk_id": chunk_id,
                    "entity_id": node.id
                })
                logger.info(f"成功创建Chunk-Entity关系: {chunk_id} -[:HAS_ENTITY]-> {node.id}")
            except Exception as e:
                logger.error(f"创建Chunk-Entity关系失败: {chunk_id} -[:HAS_ENTITY]-> {node.id}, 错误: {e}")
                continue
        
        # 处理关系
        for rel in graph_documents[0].relationships:
            logger.info(f"处理关系: {rel.source.id} -[{rel.type}]-> {rel.target.id}")
            # 创建关系
            create_relation_query = """
            MATCH (source:__Entity__ {id: $source_id})
            MATCH (target:__Entity__ {id: $target_id})
            CALL apoc.create.relationship(source, $rel_type, $properties, target) YIELD rel
            RETURN rel
            """
            try:
                result = graph.query(create_relation_query, {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "rel_type": rel.type,
                    "properties": rel.properties or {}
                })
                logger.info(f"成功创建关系: {rel.source.id} -[{rel.type}]-> {rel.target.id}")
            except Exception as e:
                logger.error(f"创建关系失败: {rel.source.id} -[{rel.type}]-> {rel.target.id}, 错误: {e}")
                continue
        
        return {"status": "success", "message": "知识图谱生成完成"}
        
    except Exception as e:
        logger.error(f"生成知识图谱时发生错误: {e}")
        return {"error": str(e)}

async def generate_graph_data_from_llm(
    llm: BaseLanguageModel,
    fileName: str,
    graph,
    embedding_function,
    dimension
) -> Dict[str, Any]:
    """
    从LLM生成知识图谱数据

    generate_graph_data_from_llm
    ├── 从Neo4j获取文档的Chunk
    ├── 对每个Chunk异步调用generate_graph
    └── 合并所有图数据并返回
    """
    try:
        # 1. 从Neo4j获取文档的所有Chunk
        get_chunks_query = """
        MATCH (d:Document {fileName: $fileName})<-[:PART_OF]-(c:Chunk)
        RETURN c.id AS chunk_id, c.text AS text
        """
        chunks = graph.query(get_chunks_query, {"fileName": fileName})
        if not chunks:
            return {"error": f"未找到文件 '{fileName}' 的任何Chunk"}

        # 2. 对每个Chunk异步生成知识图谱
        tasks = []
        for chunk in chunks:
            task = generate_graph(
                llm=llm,
                content=chunk["text"],
                graph=graph,
                embedding_function=embedding_function,
                chunk_id=chunk["chunk_id"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 检查是否有任务失败
        errors = [res for res in results if isinstance(res, Exception) or (isinstance(res, dict) and res.get('error'))]
        if errors:
            error_messages = []
            for error in errors:
                if isinstance(error, Exception):
                    error_messages.append(str(error))
                else:
                    error_messages.append(error.get('error'))
            logger.error(f"生成知识图谱时发生错误: {'; '.join(error_messages)}")
            return {"error": f"生成部分图谱时出错: {'; '.join(error_messages)}"}

        # 4. 合并所有图数据（如果需要返回聚合结果）
        # 在当前实现中，每个generate_graph调用都直接修改数据库，
        # 因此我们只需要确认所有任务都成功完成。

        ensure_vector_indexes(graph, dimension)

        return {"status": "success", "message": f"文件'{fileName}'的知识图谱生成完成"}
        
    except Exception as e:
        logger.error(f"生成知识图谱数据时发生严重错误: {e}")
        return {"error": str(e)}

def generate_graph_statistics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成图数据统计信息
    
    Args:
        graph_data: 图数据字典
    
    Returns:
        统计信息字典
    """
    try:
        node_types = {}
        relationship_types = {}

        # 统计节点类型
        for node in graph_data["nodes"]:
            node_type = node["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # 统计关系类型
        for rel in graph_data["relationships"]:
            rel_type = rel["type"]
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            "total_nodes": len(graph_data["nodes"]),
            "total_relationships": len(graph_data["relationships"]),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"生成图统计信息失败: {e}")
        return {}
