import os
import json
import time
import asyncio
from dotenv import load_dotenv

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from shared.common_fn import load_embedding_model
from shared.prompts import *
from llm import get_llm
from retrievers.vector_retriever import VectorRetriever
from logger import *
from retrieval_service import RetrievalService
from graph_search import GraphSearch

# 加载环境变量
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_FUNCTION, EMBEDDING_DIMENSION = load_embedding_model(EMBEDDING_MODEL)


class QASystem:
    def __init__(self, graph, model, session_id):
        self.graph = graph
        self.model = model
        self.session_id = session_id
        self.history = self.create_chat_history()
        self.llm, self.model_name = get_llm(model)
        self.retrieval_service = RetrievalService(graph)

        self.vector_retriever = VectorRetriever(graph)

    def initialize_schema(self):
        """初始化数据库模式"""
        try:
            session_constraint = """
            CREATE CONSTRAINT session_id_constraint IF NOT EXISTS
            FOR (s:Session)
            REQUIRE s.id IS UNIQUE
            """
            self.graph.query(session_constraint)

            message_constraints = [
                """
                CREATE CONSTRAINT message_content_not_null IF NOT EXISTS
                FOR (m:Message)
                REQUIRE m.content IS NOT NULL
                """,
                """
                CREATE CONSTRAINT message_role_not_null IF NOT EXISTS
                FOR (m:Message)
                REQUIRE m.role IS NOT NULL
                """
            ]

            for constraint in message_constraints:
                try:
                    self.graph.query(constraint)
                    logger.info("Message 属性约束创建成功")
                except Exception as e:
                    logger.warning(f"创建约束时出现警告: {str(e)}")
                    cleanup_query = """
                    MATCH (m:Message)
                    WHERE m.content IS NULL OR m.role IS NULL
                    DETACH DELETE m
                    """
                    self.graph.query(cleanup_query)
                    logger.info("已清理无效的 Message 节点")
                    # 重试创建约束
                    self.graph.query(constraint)
                    logger.info("Message 属性约束创建成功")

        except Exception as e:
            logger.error(f"初始化数据库模式时出错: {str(e)}")
            raise

    def create_chat_history(self):
        """创建聊天历史记录"""
        try:
            self.initialize_schema()

            # 确保Session节点存在
            create_session_query = """
            MERGE (s:Session {id: $session_id})
            RETURN s
            """
            self.graph.query(create_session_query, {"session_id": self.session_id})

            # 创建Neo4jChatMessageHistory实例
            history = Neo4jChatMessageHistory(
                graph=self.graph,
                session_id=self.session_id
            )

            return history
        except Exception as e:
            logger.error(f"创建聊天历史记录时出现ERROR: {e}")



    async def answer_with_vector_search(self, question: str, top_k: int = 5):
        """使用向量检索回答问题"""
        try:
            # 1. 获取检索上下文 + 获取检索元数据
            metadata, context = await self.retrieval_service.get_context(
                question, 
                top_k=top_k,
                use_hybrid=False
            )
            if not context:
                return {
                    "session_id": self.session_id,
                    "message": "抱歉，我没有找到相关的信息。",
                    "info": {
                        "model": self.model_name,
                        "mode": "vector",
                        "sources": []
                    },
                    "user": "chatbot"
                }
            
            # 2. 构建提示词
            prompt = ChatPromptTemplate.from_messages([
                ("system", CHAT_PROMPTS),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{question}")
            ])
            
            # 3. 创建RAG链
            chain = prompt | self.llm | StrOutputParser()

            # 4. 执行问答
            start_time = time.time()
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "context": context,
                    "messages": self.history.messages,
                    "question": question
                }
            )
            response_time = time.time() - start_time
            
            # 5. 更新历史记录
            self.history.add_user_message(question)
            self.history.add_ai_message(response)
            await self.summarize_history(self.history.messages)
            
            return {
                "session_id": self.session_id,
                "message": response,
                "info": {
                    "model": self.model_name,
                    "mode": "vector",
                    "sources": metadata.get("sources", []),
                    "retrieval_metadata": metadata.get("metadata", {}),
                    "response_time": response_time,
                },
                "user": "chatbot"
            }
            
        except Exception as e:
            logger.error(f"向量检索问答失败: {str(e)}")
            return {
                "session_id": self.session_id,
                "message": "抱歉，处理您的问题时出现了错误。",
                "info": {
                    "model": self.model_name,
                    "mode": "vector",
                    "error": str(e)
                },
                "user": "chatbot"
            }

    async def answer_with_hybrid_search(self, question: str, top_k: int = 5):
        """使用混合检索策略回答问题"""
        try:
            # 1. 获取检索上下文 + 获取检索元数据
            metadata, context = await self.retrieval_service.get_context(
                question, 
                top_k=top_k,
                use_hybrid=True
            )
            if not context:
                return {
                    "session_id": self.session_id,
                    "message": "抱歉，我没有找到相关的信息。",
                    "info": {
                        "model": self.model_name,
                        "mode": "hybrid",
                        "sources": []
                    },
                    "user": "chatbot"
                }
            
            # 2. 构建提示词
            prompt = ChatPromptTemplate.from_messages([
                ("system", CHAT_PROMPTS),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{question}")
            ])
            
            # 3. 创建RAG链
            chain = prompt | self.llm | StrOutputParser()

            start_time = time.time()
            # 4. 执行问答
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "context": context,
                    "messages": self.history.messages,
                    "question": question
                }
            )
            response_time = time.time() - start_time
            
            # 5. 更新历史记录
            self.history.add_user_message(question)
            self.history.add_ai_message(response)
            await self.summarize_history(self.history.messages)

            
            return {
                "session_id": self.session_id,
                "message": response,
                "info": {
                    "model": self.model_name,
                    "mode": "hybrid",
                    "sources": metadata.get("sources", []),
                    "retrieval_metadata": metadata.get("metadata", {}),
                    "response_time": response_time,
                },
                "user": "chatbot"
            }
            
        except Exception as e:
            logger.error(f"混合检索问答失败: {str(e)}")
            return {
                "session_id": self.session_id,
                "message": "抱歉，处理您的问题时出现了错误。",
                "info": {
                    "model": self.model_name,
                    "mode": "hybrid",
                    "error": str(e)
                },
                "user": "chatbot"
            }

    async def answer_with_graph_search(self, question: str, max_depth: int = 3, top_k: int = 3):
        """使用图搜索回答问题"""
        try:
            # 1. 初始化图搜索
            graph_search = GraphSearch(self.graph, EMBEDDING_FUNCTION)
            
            # 2. 执行图搜索
            chains = await graph_search.search(
                question,
                self.llm,
                max_depth=max_depth,
                top_k=top_k
            )
            
            if not chains:
                return {
                    "session_id": self.session_id,
                    "message": "抱歉，我没有找到相关的信息。",
                    "info": {
                        "model": self.model_name,
                        "mode": "graph",
                        "sources": []
                    },
                    "user": "chatbot"
                }
            
            # 3. 构建提示词
            chain_prompt = "\n".join([
                f"{chain['head']}, {chain['relation']}, {chain['tail']}"
                for chain in chains
            ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", GRAPH_PROMPTS),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "问题：{question}")
            ])
            
            # 4. 创建RAG链
            chain = prompt | self.llm | StrOutputParser()
            
            # 5. 执行问答
            start_time = time.time()
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "chains": chain_prompt,
                    "messages": self.history.messages,
                    "question": question
                }
            )
            response_time = time.time() - start_time
            
            # 6. 更新历史记录
            self.history.add_user_message(question)
            self.history.add_ai_message(response)
            await self.summarize_history(self.history.messages)
            
            # 7. 准备返回结果
            sources = []
            for chain in chains:
                source_info = {
                    "head": chain["head"],
                    "relation": chain["relation"],
                    "tail": chain["tail"],
                    "score": chain["score"]
                }
                sources.append(source_info)
            
            return {
                "session_id": self.session_id,
                "message": response,
                "info": {
                    "model": self.model_name,
                    "mode": "graph",
                    "sources": sources,
                    "response_time": response_time
                },
                "user": "chatbot"
            }
            
        except Exception as e:
            logger.error(f"图搜索问答失败: {str(e)}")
            return {
                "session_id": self.session_id,
                "message": "抱歉，处理您的问题时出现了错误。",
                "info": {
                    "model": self.model_name,
                    "mode": "graph",
                    "error": str(e)
                },
                "user": "chatbot"
            }

    async def summarize_history(self, messages):
        """异步总结聊天历史"""
        if not messages or len(messages) < 7:  # 只在历史记录达到一定长度时进行总结
            return False

        try:
            summarization_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("human",
                 "总结过往的聊天信息，要求专注两方面，一个是过往的信息的关键点或者用户提出的回答要求，另一个是要注意用户给出给出的信息，比如用户给出的自己的信息，如称呼，性别，年龄等。"
                 + "最终给出的模式为 用户提出了什么什么方面的问题，基于过往的聊天信息总结如下"
                 + "下面为一些例子：用户提出了关于 Python 虚拟环境管理的问题，对于问答的总结如下： "
                 + "**关键点：**用户小明希望将当前正在使用的 Python 虚拟环境的依赖包列表保存到 requirements.txt 文件中，并能在其他环境中复现当前环境配置。已说明其使用的是 Conda 环境，希望用一条命令完成导出。回答中建议使用 pip freeze > requirements.txt 或 conda list --export，并提醒 pip 和 conda 导出内容格式不同。"
                 + "**用户信息：**用户自称为小明，年龄19岁，使用的是 macOS 系统和 PyCharm 开发工具，对环境管理有一定了解，主要使用 Anaconda 管理项目环境。")
            ])

            summary = await asyncio.to_thread((summarization_prompt | self.llm).invoke, {"chat_history": messages})

            try:
                self.history.clear()
                self.history.add_user_message("当前对话摘要")
                self.history.add_message(summary)
            except Exception as e:
                logger.warning(f"更新聊天历史记录失败: {e}")
                # 如果更新失败，我们可以创建一个新的历史记录
                from langchain_core.chat_history import ChatMessageHistory
                self.history = ChatMessageHistory()
                self.history.add_user_message("当前对话摘要")
                self.history.add_message(summary)

            return True
        except Exception as e:
            logger.error(f"总结聊天历史失败: {e}")
            return False
    def clear_history(self):
        """清除聊天历史"""
        self.history.clear()
        logger.info(f"{self.session_id} 已清除聊天记录")