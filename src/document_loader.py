import os
import zipfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from shared.create_chunks import ChunksCreator
from shared.common_fn import (
    load_embedding_model,
    create_folder_name_hashed,
    get_neo4j_connection,
    save_processed_data
)
from logger import logger
from dotenv import load_dotenv



# 加载环境变量
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_FUNCTION, DIMENSION = load_embedding_model(EMBEDDING_MODEL)


class DocumentLoader:
    """文档加载和向量化存储类"""

    def __init__(self, graph=None, url=None, username=None, password=None):
        """
        初始化文档加载器
        """
        if graph is None and (url is None or username is None or password is None):
            raise ValueError("必须提供graph对象或Neo4j连接信息(uri, username, password)")

        self.graph = graph if graph else get_neo4j_connection(url, username, password)
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
        }



    def load_document(self, file_path: str) -> List[LangchainDocument]:
        """
        加载文档并返回文档对象列表
        """
        _, ext = os.path.splitext(file_path)
        allowed_extensions = ['.pdf', '.txt', '.md']
        if ext.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported file format: {ext}. Only PDF, TXT, and MD are allowed.")



        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")



        loader_class = self.supported_extensions.get(ext.lower())
        if not loader_class:
            raise ValueError(f"No loader found for file type: {ext}")
        if loader_class == "TextLoader":
            loader = loader_class(encdoing='utf-8', file_path=file_path)
        else:
            loader = loader_class(file_path)

        try:
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共{len(documents)}页")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            raise ValueError(f"加载文档失败: {str(e)}")



    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        使用ChunksCreator将文档分割成更小的片段
        Args:
            documents: 文档列表
        Returns:
            分割后的文档片段列表
        """
        try:
            # 使用已有的ChunksCreator类进行文档分块
            chunks_creator = ChunksCreator(documents)
            chunks = chunks_creator.split_file_into_chunks()

            # 确保保留原始文档的元数据
            for chunk in chunks:
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                # 保留原始文档的元数据
                if documents and documents[0].metadata:
                    chunk.metadata.update(documents[0].metadata)

            logger.info(f"文档分割完成，共{len(chunks)}个片段")
            return chunks
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            raise

    def process_document(self, file_path: str, index_name: str, custom_metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        处理单个文档：加载、分割、存储
        """
        try:
            documents = self.load_document(file_path)
            if not documents:
                return False, {"error": "文档加载失败，内容为空"}

            chunks = self.split_documents(documents)
            if not chunks:
                return False, {"error": "文档分割失败"}

            # 添加自定义元数据
            if custom_metadata:
                for chunk in chunks:
                    chunk.metadata.update(custom_metadata)

            success = self.store_documents(chunks, index_name)
            if not success:
                return False, {"error": "文档存储失败"}

            return True, {"message": "文档处理成功"}
        except Exception as e:
            logger.error(f"处理文档失败: {file_path}, 错误: {str(e)}")
            return False, {"error": str(e)}

    def store_documents(self, documents: List[LangchainDocument],
                        index_name: str = "vector",
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        将文档存储到Neo4j向量数据库，并建立Document和Chunk节点及其关系

        Args:
            documents: 文档片段列表
            index_name: 向量索引名称
            metadata: 额外的元数据

        Returns:
            是否成功存储
        """
        try:
            # 为每个文档添加元数据
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # 如果没有文档，直接返回成功
            if not documents:
                logger.warning("没有文档需要存储")
                return True

            # 获取文件信息
            file_hash = documents[0].metadata.get("fileHash", "unknown")
            file_name = documents[0].metadata.get("fileName", "unknown")
            source = documents[0].metadata.get("source", "unknown")

            # 1. 创建Document节点
            create_document_query = """
            MERGE (d:Document {id: $file_hash})
            SET d.fileName = $file_name,
                d.fileHash = $file_hash,
                d.source = $source
            RETURN d
            """

            self.graph.query(create_document_query, {
                "file_hash": file_hash,
                "file_name": file_name,
                "source": source
            })

            # 2. 使用Neo4jVector存储文档块作为Chunk节点
            Neo4jVector.from_documents(
                documents=documents,
                embedding=EMBEDDING_FUNCTION,
                index_name=index_name,
                node_label="Chunk",  # 设置节点标签为Chunk
                text_node_property="text",  # 文本内容属性名
                embedding_node_property="embedding",  # 向量属性名
                graph=self.graph
            )

            # 3. 为每个Chunk节点添加属性并创建与Document的关系
            for i, doc in enumerate(documents):
                chunk_id = f"{file_hash}_{doc.metadata.get('chunk_index', i)}"

                # 更新Chunk节点属性
                update_chunk_query = """
                MATCH (c:Chunk {id: $chunk_id})
                SET c.fileName = $file_name,
                    c.fileHash = $file_hash,
                    c.source = $source
                RETURN c
                """

                self.graph.query(update_chunk_query, {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "source": source
                })

                # 创建Chunk -> Document的PART_OF关系
                create_relation_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (d:Document {id: $file_hash})
                MERGE (c)-[:PART_OF]->(d)
                """

                self.graph.query(create_relation_query, {
                    "chunk_id": chunk_id,
                    "file_hash": file_hash
                })

            # 保存处理记录到缓存
            cache_data = {
                "source": source,
                "chunks_count": len(documents),
                "index_name": index_name,
                "timestamp": str(datetime.now())
            }
            save_processed_data(cache_data, f"doc_process_{file_hash}.json")

            logger.info(f"成功将{len(documents)}个文档片段存储到Neo4j向量数据库")
            return True
        except Exception as e:
            logger.error(f"存储文档到Neo4j失败: {str(e)}")
            return False

    def process_document(self, file_path: str, index_name: str = "vector",
                         custom_metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        处理文档的完整流程：加载、分割、添加元数据

        Args:
            file_path: 文档路径
            index_name: 向量索引名称
            custom_metadata: 自定义元数据，将与自动生成的元数据合并

        Returns:
            (是否成功处理, 处理结果信息)
        """
        try:
            logger.info(f"开始处理文档: {file_path}")
            # 提取文件名作为元数据，优先使用原始文件名
            original_filename = custom_metadata.get("original_filename") if custom_metadata else None
            file_name = original_filename if original_filename else os.path.basename(file_path)
            file_hash = create_folder_name_hashed(file_name)

            # 加载文档
            documents = self.load_document(file_path)
            if not documents:
                logger.error(f"文档加载失败: {file_path}")
                return False, {"file_path": file_path, "error": "文档加载失败"}

            # 为原始文档添加基本元数据
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}

                # 添加基本元数据
                doc.metadata.update({
                    "source": file_path,
                    "fileName": file_name,
                    "fileHash": file_hash,
                    "documentType": os.path.splitext(file_name)[1][1:].lower()
                })

                # 添加自定义元数据
                if custom_metadata:
                    doc.metadata.update(custom_metadata)

            # 分割文档 - 元数据会在split_documents方法中自动传递
            split_docs = self.split_documents(documents)
            if not split_docs:
                logger.warning(f"文档分割后没有内容: {file_path}")
                return False, {
                    "file_path": file_path,
                    "file_name": file_name,
                    "chunks_count": 0,
                    "error": "文档分割后没有内容",
                    "success": False
                }

            # 为每个chunk添加嵌入向量和唯一ID
            for i, doc in enumerate(split_docs):
                # 添加chunk索引
                doc.metadata["chunk_index"] = i
                # 生成唯一ID
                doc.metadata["id"] = f"{file_hash}_{i}"
                # 计算chunk的嵌入向量
                doc.metadata["embedding"] = EMBEDDING_FUNCTION.embed_query(doc.page_content)

            # 存储文档到Neo4j向量数据库
            storage_success = self.store_documents(split_docs, index_name=index_name)
            if not storage_success:
                logger.error(f"文档存储失败: {file_path}")
                return False, {
                    "file_path": file_path,
                    "file_name": file_name,
                    "chunks_count": len(split_docs),
                    "error": "文档存储失败",
                    "success": False
                }

            result = {
                "file_path": file_path,
                "file_name": file_name,
                "chunks_count": len(split_docs),
                "success": True
            }

            logger.info(f"文档处理成功: {file_path}")
            return True, result

        except Exception as e:
            logger.error(f"处理文档失败: {file_path}, 错误: {str(e)}")
            return False, {"file_path": file_path, "error": str(e), "success": False}

    def process_directory(self, directory_path: str,
                          index_name: str = "vector") -> Dict[str, Any]:
        """
        处理目录中的所有支持的文档
        """
        results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "failed_files": [],
            "processed_files": []
        }

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)

                if ext.lower() in self.supported_extensions:
                    results["total"] += 1
                    success, result = self.process_document(
                        file_path,
                        index_name=index_name
                    )

                    if success:
                        results["success"] += 1
                        results["processed_files"].append(result)
                    else:
                        results["failed"] += 1
                        results["failed_files"].append(file_path)

        return results

    def clean_graph(self):
        cleanup_query = """
                        MATCH (n)-[r]-(m)
                        DELETE n,r,m
                        """
        self.graph.query(cleanup_query)
        logger.info("测试数据已清理")
