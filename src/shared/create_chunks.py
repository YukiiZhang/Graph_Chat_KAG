from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from logger import *
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import re

load_dotenv()

class ChunksCreator:
    def __init__(self, pages: list[Document]):
        """
        初始化类
        :param pages: 需要拆分的文档列表（每个元素是一个 Document 对象）
        """
        self.pages = pages

        # 读取 chunk 配置
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))

        # 支持的格式
        self.allowed_formats = {"pdf", "txt", "md"}

        # 拆分器初始化
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # PDF专用拆分器，使用RecursiveCharacterTextSplitter更适合处理PDF结构
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # 按段落、行、空格等层次拆分
        )

        # 验证文档格式
        self.validate_documents()

    def validate_documents(self):
        """
        确保所有文件格式均为允许范围
        """
        for doc in self.pages:
            source = doc.metadata.get("source", "")
            ext = Path(source).suffix.lower().replace(".", "")
            if ext not in self.allowed_formats:
                raise ValueError(f"Unsupported file format: {ext}. Only PDF, TXT, and DOC/DOCX are allowed.")

    def split_text_into_paragraphs(self, text):
        """
        将文本拆分为段落
        """
        # 使用连续的换行符作为段落分隔符
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def process_pdf_page(self, document, page_number):
        """
        处理PDF页面，保留页面结构并添加段落信息
        """
        # 使用PDF专用拆分器处理页面内容
        chunks = self.pdf_splitter.split_documents([document])
        
        # 为每个chunk添加页码和段落信息
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # 复制原始元数据
            chunk.metadata.update(document.metadata)
            chunk.metadata['page_number'] = page_number
            chunk.metadata['paragraph'] = i + 1  # 在页面内的段落编号
            result_chunks.append(chunk)
            
        return result_chunks

    def process_text_document(self, document):
        """
        处理文本文档，按段落拆分
        """
        # 获取文档内容
        text = document.page_content
        # 拆分为段落
        paragraphs = self.split_text_into_paragraphs(text)
        
        result_chunks = []
        # 处理每个段落
        for p_idx, paragraph in enumerate(paragraphs):
            # 如果段落长度超过chunk_size，需要进一步拆分
            if len(paragraph) > self.chunk_size:
                temp_doc = Document(page_content=paragraph, metadata={})
                paragraph_chunks = self.token_splitter.split_documents([temp_doc])
                
                # 为每个chunk添加元数据
                for chunk in paragraph_chunks:
                    chunk.metadata.update(document.metadata)
                    chunk.metadata['paragraph'] = p_idx + 1
                    result_chunks.append(chunk)
            else:
                # 段落长度合适，直接作为一个chunk
                doc = Document(page_content=paragraph, metadata=document.metadata.copy())
                doc.metadata['paragraph'] = p_idx + 1
                result_chunks.append(doc)
                
        return result_chunks

    def split_file_into_chunks(self):
        """
        拆分文档为chunks：根据文件类型采用不同的拆分策略
        """
        logging.info("开始拆分文件为小块...")

        if not self.pages:
            logging.warning("输入的 pages 列表为空，未拆分任何内容。")
            return []

        chunks = []
        first_source = self.pages[0].metadata.get("source", "")
        is_pdf = first_source.lower().endswith(".pdf")

        if is_pdf:
            # 对PDF文件，按页面处理，保留页面结构
            for i, document in enumerate(self.pages):
                page_number = i + 1
                page_chunks = self.process_pdf_page(document, page_number)
                chunks.extend(page_chunks)
        else:
            for document in self.pages:
                doc_chunks = self.process_text_document(document)
                chunks.extend(doc_chunks)

        logger.info(f"文档拆分完成，共生成 {len(chunks)} 个 chunks")
        return chunks
