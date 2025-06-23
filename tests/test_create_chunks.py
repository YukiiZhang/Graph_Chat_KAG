import pytest
from langchain.docstore.document import Document
from src.shared.create_chunks import ChunksCreator
import os
from pathlib import Path

# 测试数据
SAMPLE_TEXT = """这是第一段。

这是第二段。
这是第二段的第二行。

这是第三段。"""

SAMPLE_PDF_TEXT = """第1页内容
这是第一页的段落。

这是第一页的第二个段落。

第2页内容
这是第二页的内容。"""


def test_text_document_splitting():
    """测试文本文档的拆分功能"""
    # 创建测试文档
    doc = Document(
        page_content=SAMPLE_TEXT,
        metadata={"source": "test.txt"}
    )
    
    # 初始化 ChunksCreator
    creator = ChunksCreator([doc])
    
    # 执行拆分
    chunks = creator.split_file_into_chunks()
    
    # 验证结果
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("paragraph" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)

def test_pdf_document_splitting():
    """测试PDF文档的拆分功能"""
    # 创建测试PDF文档（模拟多页）
    pages = [
        Document(
            page_content="第1页内容\n这是第一页的段落。\n\n这是第一页的第二个段落。",
            metadata={"source": "test.pdf"}
        ),
        Document(
            page_content="第2页内容\n这是第二页的内容。",
            metadata={"source": "test.pdf"}
        )
    ]
    
    # 初始化 ChunksCreator
    creator = ChunksCreator(pages)
    
    # 执行拆分
    chunks = creator.split_file_into_chunks()
    
    # 验证结果
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("page_number" in chunk.metadata for chunk in chunks)
    assert all("paragraph" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)

def test_invalid_file_format():
    """测试无效文件格式的验证"""
    doc = Document(
        page_content="测试内容",
        metadata={"source": "test.xyz"}
    )
    
    # 验证是否抛出预期的异常
    with pytest.raises(ValueError, match="Unsupported file format"):
        ChunksCreator([doc])

def test_empty_pages():
    """测试空文档列表的处理"""
    creator = ChunksCreator([])
    chunks = creator.split_file_into_chunks()
    assert len(chunks) == 0

def test_chunk_size_and_overlap():
    """测试chunk大小和重叠设置"""
    # 创建一个较长的文本文档
    long_text = "这是一个测试段落。" * 50
    doc = Document(
        page_content=long_text,
        metadata={"source": "test.txt"}
    )
    
    # 设置较小的chunk大小
    os.environ['CHUNK_SIZE'] = '100'
    os.environ['CHUNK_OVERLAP'] = '20'
    
    creator = ChunksCreator([doc])
    chunks = creator.split_file_into_chunks()
    
    # 验证chunk大小
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    # 验证chunk数量
    assert len(chunks) > 1  # 确保文档被拆分成了多个chunks 