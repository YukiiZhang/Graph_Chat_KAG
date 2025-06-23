from typing import Tuple
from openai import OpenAI
import hashlib
import os
import json
from typing import Dict, Any
from logger import logger
from modelscope import snapshot_download
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

from dotenv import load_dotenv
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_LOCAL_PATH = os.getenv("EMBEDDING_LOCAL_PATH")

def clean_api_response(response: str, api_type: str) -> str:
    """清理API响应"""
    if api_type == "DeepSeek":
        return response.replace("<｜end of sentence｜>", "").strip()
    return response.strip()


# 组合 URI 和文件名，生成哈希名
def create_folder_name_hashed(file_name):
    unique_string = f"{file_name}"
    hashed_name = hashlib.sha256(unique_string.encode()).hexdigest()[:16]  # 取前 16 位，防止过长
    return hashed_name


# 格式化时间
def formatted_time(current_time):
    return current_time.strftime('%Y-%m-%d %H:%M:%S %Z')


#设置本地缓存
def get_cache_dir():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


#保存数据
def save_processed_data(data: Dict[str, Any], file_name: str):
    cache_file = os.path.join(get_cache_dir(), file_name)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


#读取数据
def load_processed_data(file_name: str):
    cache_file = os.path.join(get_cache_dir(), file_name)
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_neo4j_connection(url, username, password):
    graph = Neo4jGraph(url=url, username=username, password=password, refresh_schema=False, sanitize=True)
    return graph

_embedding_model_cache = {}

def load_embedding_model(embedding_model_name=EMBEDDING_MODEL):
    if embedding_model_name in _embedding_model_cache:
        return _embedding_model_cache[embedding_model_name]

    if embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info(f"Embedding: Using OpenAI Embeddings , Dimension:{dimension}")
    else:
        try:
            dir_path = os.path.join(EMBEDDING_LOCAL_PATH, embedding_model_name)
            if not os.path.exists(dir_path):
                model_dir = snapshot_download(
                    repo_id=embedding_model_name,
                    local_dir=dir_path,
                )
                embeddings = HuggingFaceEmbeddings(model_name=model_dir)
            else:
                embeddings = HuggingFaceEmbeddings(model_name=dir_path)
            
            # 使用一个简短的文本来获取维度
            text = "test"
            embedding_vector = embeddings.embed_query(text)
            dimension = len(embedding_vector)
            logger.info(f"Embedding: {embedding_model_name} loaded successfully, Dimension:{dimension}")
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model_name}: {str(e)}")
            logger.info("Falling back to default model: all-MiniLM-L6-v2")
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            dimension = 384
            logger.info(f"Embedding: Using SentenceTransformer , Dimension:{dimension}")

    _embedding_model_cache[embedding_model_name] = (embeddings, dimension)
    return embeddings, dimension
