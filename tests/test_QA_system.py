import os
import sys
import time
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_neo4j import Neo4jChatMessageHistory
from src.QA_system import QASystem
from logger import logger
import json

# 加载环境变量
load_dotenv()

def test_neo4j_chat_history(graph: Neo4jGraph):
    """测试 Neo4j 聊天历史记录"""
    logger.info("开始测试 Neo4j 聊天历史记录功能")
    try:
        # 创建 QASystem 实例
        qa_system = QASystem(
            graph=graph,
            model="deepseek",  # 可以根据需要更改模型
            session_id="test_session_neo4j"
        )

        # 测试 QASystem 的聊天历史功能
        logger.info("测试 QASystem 的聊天历史功能")

        # 添加用户消息
        qa_system.history.add_user_message("你好，这是一个测试消息99")
        # qa_system.history.add_user_message("你好，这是一个测试消息2")
        # qa_system.history.add_user_message("你好，这是一个测试消息3")
        # qa_system.history.add_user_message("你好，这是一个测试消息4")
        # qa_system.history.add_user_message("你好，这是一个测试消息5")
        qa_system.history.add_ai_message("你好，我是AI助手")
        logger.info("已添加消息")
        print(qa_system.history.messages)

        # 清理测试数据
        choice = input("是否要删除测试数据？(y/n): ").strip().lower()
        if choice == 'y':
            qa_system.clear_history()
        else:
            logger.info("未执行测试数据清理操作")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        logger.info("Neo4j 聊天历史记录测试完成")

def test_vector_search(graph: Neo4jGraph):
    """测试向量搜索功能"""
    logger.info("开始测试向量搜索功能")
    try:
        # 创建 QASystem 实例
        qa_system = QASystem(
            graph=graph,
            model="deepseek",
            session_id="test_session_vector"
        )

        # 测试问题
        question = "屠呦呦是谁"
        logger.info(f"测试问题: {question}")

        # 执行向量搜索
        import asyncio
        response = asyncio.run(qa_system.answer_with_vector_search(question, top_k=2))
        
        # 打印结果
        logger.info("搜索结果:")
        logger.info(f"回答: {response['message']}")
        logger.info(f"检索模式: {response['info']['mode']}")
        logger.info(f"来源数量: {len(response['info']['sources'])}")
        
        # 打印检索元数据
        if response['info'].get('retrieval_metadata'):
            logger.info("检索元数据:")
            logger.info(json.dumps(response['info']['retrieval_metadata'], indent=2, ensure_ascii=False))

        # 清理测试数据
        choice = input("是否要删除测试数据？(y/n): ").strip().lower()
        if choice == 'y':
            qa_system.clear_history()
        else:
            logger.info("未执行测试数据清理操作")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        logger.info("向量搜索测试完成")

def test_hybrid_search(graph: Neo4jGraph):
    """测试向量搜索功能"""
    logger.info("开始测试向量搜索功能")
    try:
        # 创建 QASystem 实例
        qa_system = QASystem(
            graph=graph,
            model="deepseek",
            session_id="test_session_vector"
        )

        # 测试问题
        question = "屠呦呦是谁"
        logger.info(f"测试问题: {question}")

        # 执行向量搜索
        import asyncio
        response = asyncio.run(qa_system.answer_with_hybrid_search(question, top_k= 2))

        # 打印结果
        logger.info("搜索结果:")
        logger.info(f"回答: {response['message']}")
        logger.info(f"检索模式: {response['info']['mode']}")
        logger.info(f"来源数量: {len(response['info']['sources'])}")

        # 打印检索元数据
        if response['info'].get('retrieval_metadata'):
            logger.info("检索元数据:")
            logger.info(json.dumps(response['info']['retrieval_metadata'], indent=2, ensure_ascii=False))

        # 清理测试数据
        choice = input("是否要删除测试数据？(y/n): ").strip().lower()
        if choice == 'y':
            qa_system.clear_history()
        else:
            logger.info("未执行测试数据清理操作")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        logger.info("向量搜索测试完成")

def test_graph_search(graph: Neo4jGraph):
    logger.info("开始测试向量搜索功能")
    try:
        # 创建 QASystem 实例
        qa_system = QASystem(
            graph=graph,
            model="deepseek",
            session_id="test_session_graph"
        )

        # 测试问题
        question = "屠呦呦是谁"
        logger.info(f"测试问题: {question}")

        # 执行向量搜索
        import asyncio
        response = asyncio.run(qa_system.answer_with_graph_search(question, max_depth=2, top_k=3))

        # 打印结果
        logger.info("搜索结果:")
        logger.info(f"回答: {response['message']}")
        logger.info(f"检索模式: {response['info']['mode']}")
        logger.info(f"来源数量: {len(response['info']['sources'])}")

        # 打印检索元数据
        if response['info'].get('retrieval_metadata'):
            logger.info("检索元数据:")
            logger.info(json.dumps(response['info']['retrieval_metadata'], indent=2, ensure_ascii=False))


    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
    finally:
        logger.info("向量搜索测试完成")

if __name__ == "__main__":
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # 创建 Neo4j 连接
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    # test_neo4j_chat_history(graph)
    # test_vector_search(graph)
    # test_hybrid_search(graph)
    test_graph_search(graph)