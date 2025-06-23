import logging
import logger
import streamlit as st
import asyncio
from langchain_community.document_loaders import PyPDFLoader

# 设置 Streamlit 页面配置
st.set_page_config(
    layout="wide",
    page_title="Graph_chat",
    page_icon=":rocket:"
)

# 自定义 Streamlit 日志处理器
class StreamlitHandler(logging.Handler):
    """
    Streamlit 日志输出处理类
    """
    def __init__(self, st_container):
        super().__init__()
        self.st_container = st_container

    def emit(self, record):
        log_entry = self.format(record)
        self.st_container.write(log_entry)

async def main():
    st.write("# 欢迎使用 Graph_chat! 👋")

    st.sidebar.success("欢迎使用 Graph_chat! 🎉")

    st.markdown(
        """
        Graph-chat 是基于知识图谱的专业领域聊天机器人。
        **👈 请先设置Settings**，看看 Graph_chat 能做什么吧！
        ### 项目介绍
        - 此项目的数据集和Prompts为**军事领域**相关，请上传**相关文件**才能达到预期效果。
        - 此项目允许您上传文件(PDF,DOC,TXT)，生成**知识图谱**导入Neo4j图形数据库中，并使用自然语言执行查询。
        - **可以做到实时与Neo4j数据库交互，并实现Chat_bot**。
        ### 项目参考
        - Graph查询模式参考了[《Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph》](https://arxiv.org/abs/2307.07697)
    """
    )


# 运行 Streamlit 应用
if __name__ == "__main__":
    asyncio.run(main())
