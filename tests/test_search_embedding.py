from neo4j import GraphDatabase
import os
from src.shared.common_fn import load_embedding_model
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector


load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_FUNCTION, DIMENSION = load_embedding_model(EMBEDDING_MODEL)

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

vector_store = Neo4jVector.from_existing_index(
    embedding=EMBEDDING_FUNCTION,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="vector",  # 向量索引名
    node_label="Chunk",    # 节点标签
    text_node_property="text",  # 存储文本的属性
    embedding_node_property="embedding",  # 存储向量的属性
)

query = "屠呦呦是谁"
results = vector_store.similarity_search_with_score(query, k=3)

print(f"\n查询: '{query}' 的搜索结果:")
print("-" * 80)
for i, (doc, score) in enumerate(results, 1):
    print(f"相似度分数: {score:.4f}")
    print(f"内容: {doc.page_content[:50]}...")
    print("-" * 80)

"""
OUTPUT:

查询: '屠呦呦是谁' 的搜索结果:
--------------------------------------------------------------------------------
相似度分数: 0.9269
内容: 屠呦呦，女，汉族，中共党员，1930年12月出生，浙江宁波人。1955年毕业于北京医学院（现北京大学...
--------------------------------------------------------------------------------
相似度分数: 0.9171
内容: 2017年获2016年度国家最高科学技术奖，2018年获改革先锋称号，2019年被授予共和国勋章。
...
--------------------------------------------------------------------------------
相似度分数: 0.9109
内容: 
“呦呦鹿鸣，食野之蒿”，屠呦呦的父亲正是根据《诗经·小雅》中的这段话，为自己女儿取名“呦呦”。惊人...
--------------------------------------------------------------------------------
"""