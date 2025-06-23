import os
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from logger import *


load_dotenv()


def get_llm(model_version: str) -> Tuple[object, str]:
    """
    通用 LLM 加载器：根据模型版本从 .env 中加载对应配置。
    支持多种模型提供商（如 Ollama、Doubao、Deepseek 等）
    """
    if not model_version:
        raise ValueError("模型版本不能为空，请检查环境变量设置")
        
    prefix = model_version.upper().replace('-', '_').replace('.', '_')

    model_name = os.getenv(f"{prefix}_MODEL_NAME")
    api_key = os.getenv(f"{prefix}_API_KEY")
    api_url = os.getenv(f"{prefix}_API_URL")

    if not model_name:
        raise ValueError(f"Missing environment variable: {prefix}_MODEL_NAME")

    if "QWEN" in prefix:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_url,
            model=model_name,
            temperature=0.8
        )
    elif "DEEPSEEK" in prefix:
        llm = ChatDeepSeek(
            api_key=api_key,
            base_url=api_url,
            model=model_name,
            temperature=0.8
        )
    elif "DOUBAO" in prefix:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_url,
            model=model_name,
            temperature=0.8
        )
    else:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_url,
            model=model_name,
            temperature=0.8
        )

    logger.info(f"Model initialized: {model_version} -> {model_name}")

    return llm, model_name


# if __name__ == '__main__':
#     llm, _ = get_llm("qwen_plus")

