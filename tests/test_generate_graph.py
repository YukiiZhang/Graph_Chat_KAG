import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from src.document_loader import DocumentLoader
from src.generate_graph_from_llm import generate_graph_data_from_llm
from src.shared.common_fn import load_embedding_model
from src.llm import get_llm
import asyncio
from logger import logger

# 加载环境变量
load_dotenv()


async def test_document_processing():
    """测试文档处理流程"""
    try:
        # 初始化Neo4j连接
        graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )

        # 初始化文档加载器
        loader = DocumentLoader(graph=graph)

        # 定义文件路径
        file_path = "../your_file"

        # 处理文档
        success, result = loader.process_document(
            file_path=file_path,
            index_name="vector"
        )

        if not success:
            logger.error(f"文档处理失败: {result.get('error', '未知错误')}")
            return

        # 初始化LLM和嵌入模型
        llm, model_name = get_llm("deepseek")
        embedding_function, dimension = load_embedding_model(os.getenv('EMBEDDING_MODEL_NAME'))

        # 生成知识图谱
        graph_data = await generate_graph_data_from_llm(
            llm=llm,
            fileName=os.path.basename(file_path),
            graph=graph,
            embedding_function=embedding_function,
            dimension=dimension
        )

        if "error" in graph_data:
            logger.error(f"生成知识图谱失败: {graph_data['error']}")
            return

        logger.info("知识图谱生成完成")


    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_document_processing())