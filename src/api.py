from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import asyncio
from langchain_neo4j import Neo4jGraph
import os
import uuid
from dotenv import load_dotenv
from QA_system import QASystem
from document_loader import DocumentLoader
from generate_graph_from_llm import generate_graph_data_from_llm
from shared.common_fn import load_embedding_model
from llm import get_llm
from logger import logger
import json
import tempfile

# 加载环境变量
load_dotenv()

# 获取嵌入模型
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME')
EMBEDDING_FUNCTION, EMBEDDING_DIMENSION = load_embedding_model(EMBEDDING_MODEL)

# 全局变量存储活跃的QA系统实例
active_qa_systems = {}

# 定义数据模型
class Neo4jConnectionConfig(BaseModel):
    uri: str
    username: str
    password: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    search_type: str = "vector"  # vector, hybrid, graph
    top_k: int = Field(default=5, ge=1, le=20)
    max_depth: Optional[int] = Field(default=3, ge=1, le=5)
    config: Optional[Dict[str, str]] = None  # Neo4j连接配置

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    error: Optional[str] = None

class GraphGenerationRequest(BaseModel):
    fileName: str
    model: str = "deepseek"
    config: Neo4jConnectionConfig
    skip_document_processing: Optional[bool] = True

class GraphGenerationResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None
    error: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    message: str
    info: Dict[str, Any]
    user: str = "chatbot"
# 创建 FastAPI 应用
app = FastAPI(
    title="知识图谱问答系统",
    description="基于知识图谱的智能问答系统API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 依赖项：获取Neo4j图数据库连接
def get_neo4j_graph(config: Neo4jConnectionConfig = None):
    """
    创建Neo4j图数据库连接
    """
    try:
        if config:
            # 使用前端提供的连接信息
            graph = Neo4jGraph(
                url=config.uri,
                username=config.username,
                password=config.password
            )
        else:
            # 使用环境变量中的连接信息
            graph = Neo4jGraph(
                url=os.getenv('NEO4J_URI'),
                username=os.getenv('NEO4J_USERNAME'),
                password=os.getenv('NEO4J_PASSWORD')
            )
        return graph
    except Exception as e:
        logger.error(f"Neo4j连接失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Neo4j连接失败: {str(e)}")

# Neo4j聊天记录操作函数
async def save_message_to_neo4j(graph: Neo4jGraph, session_id: str, role: str, content: str):
    """
    保存消息到Neo4j数据库
    """
    try:
        # 创建或获取Session节点
        session_query = """
        MERGE (s:Session {id: $session_id})
        RETURN s
        """
        graph.query(session_query, {"session_id": session_id})
        
        # 创建Message节点
        message_query = """
        MATCH (s:Session {id: $session_id})
        CREATE (m:Message {role: $role, content: $content, timestamp: datetime()})
        
        // 获取当前最后一个消息
        OPTIONAL MATCH (s)-[:LAST_MESSAGE]->(lastMsg:Message)
        
        // 如果存在最后一个消息，建立NEXT关系
        FOREACH (last IN CASE WHEN lastMsg IS NOT NULL THEN [lastMsg] ELSE [] END |
            CREATE (last)-[:NEXT]->(m)
        )
        
        // 删除旧的LAST_MESSAGE关系并创建新的
        FOREACH (last IN CASE WHEN lastMsg IS NOT NULL THEN [lastMsg] ELSE [] END |
            DELETE (s)-[:LAST_MESSAGE]->(last)
        )
        CREATE (s)-[:LAST_MESSAGE]->(m)
        
        RETURN m
        """
        graph.query(message_query, {
            "session_id": session_id,
            "role": role,
            "content": content
        })
        
    except Exception as e:
        logger.error(f"保存消息到Neo4j失败: {str(e)}")
        raise

async def load_chat_history_from_neo4j(graph: Neo4jGraph, session_id: str):
    """
    从Neo4j加载聊天历史
    """
    try:
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[:LAST_MESSAGE]->(lastMsg:Message)
        
        // 如果没有消息，返回空结果
        WITH s, lastMsg
        WHERE lastMsg IS NOT NULL
        
        // 从最后一个消息开始，沿着NEXT关系的反向遍历所有消息
        MATCH path = (firstMsg:Message)-[:NEXT*0..]->(lastMsg)
        WHERE NOT EXISTS((:Message)-[:NEXT]->(firstMsg))
        
        // 获取路径中的所有消息节点，按照NEXT关系的顺序
        WITH nodes(path) as messages
        UNWIND range(0, size(messages)-1) as idx
        WITH messages[idx] as msg, idx
        RETURN msg.role as role, msg.content as content
        ORDER BY idx ASC
        """
        
        result = graph.query(query, {"session_id": session_id})
        
        chat_history = []
        for record in result:
            role = "user" if record["role"] == "human" else "assistant"
            chat_history.append({
                "role": role,
                "content": record["content"]
            })
        
        return chat_history
        
    except Exception as e:
        logger.error(f"从Neo4j加载聊天历史失败: {str(e)}")
        return []

async def delete_session_from_neo4j(graph: Neo4jGraph, session_id: str):
    """
    从Neo4j删除会话及其所有消息
    """
    try:
        query = """
        MATCH (s:Session {id: $session_id})
        OPTIONAL MATCH (s)-[:LAST_MESSAGE]->(lastMsg:Message)
        OPTIONAL MATCH (lastMsg)<-[:NEXT*0..]-(msg:Message)
        DETACH DELETE s, lastMsg, msg
        """
        
        graph.query(query, {"session_id": session_id})
        
    except Exception as e:
        logger.error(f"从Neo4j删除会话失败: {str(e)}")
        raise

async def get_all_sessions_from_neo4j(graph: Neo4jGraph):
    """
    获取所有会话列表
    """
    try:
        query = """
        MATCH (s:Session)
        OPTIONAL MATCH (s)-[:LAST_MESSAGE]->(lastMsg:Message)
        RETURN s.id as session_id, lastMsg.content as last_message_content
        ORDER BY s.id DESC
        """
        
        result = graph.query(query)
        
        sessions = []
        for record in result:
            sessions.append({
                "session_id": record["session_id"],
                "last_message_content": record["last_message_content"]
            })
        
        return sessions
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {str(e)}")
        return []

# 获取或创建QA系统实例
def get_qa_system(session_id: str, model: str, graph: Neo4jGraph):
    """
    获取或创建QA系统实例
    """
    global active_qa_systems
    
    # 如果会话ID已存在，返回现有实例
    if session_id in active_qa_systems:
        return active_qa_systems[session_id]
    
    # 否则创建新实例
    qa_system = QASystem(graph=graph, model=model, session_id=session_id)
    active_qa_systems[session_id] = qa_system
    return qa_system

@app.post("/connect", response_model=Dict[str, Any])
async def connect_to_neo4j(config: Neo4jConnectionConfig):
    """
    连接到Neo4j数据库
    """
    try:
        graph = get_neo4j_graph(config)
        # 测试连接
        result = graph.query("MATCH (n) RETURN count(n) as count LIMIT 1")
        return {"success": True, "message": "连接成功", "node_count": result[0]["count"]}
    except Exception as e:
        logger.error(f"Neo4j连接测试失败: {str(e)}")
        return {"success": False, "message": "连接失败", "error": str(e)}

@app.post("/upload-and-generate", response_model=GraphGenerationResponse)
async def upload_and_generate(
    file: UploadFile = File(...),
    model: str = "deepseek",
    config: str = Form(...)
):
    """
    上传并处理文档
    """
    try:
        # 解析config
        try:
            config_data = json.loads(config)
            neo4j_config = Neo4jConnectionConfig(**config_data)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"无效的config格式: {e}")

        # 获取Neo4j连接
        graph = get_neo4j_graph(neo4j_config)
        
        # 创建临时文件
        temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            # 读取上传的文件内容并写入临时文件
            content = await file.read()
            temp_file.write(content)
        
        # 初始化文档加载器
        loader = DocumentLoader(graph=graph)
        
        # 处理文档
        success, result = loader.process_document(
            file_path=temp_file_path,
            index_name="vector",
            custom_metadata={"original_filename": file.filename}
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {result.get('error')}")

        # 获取LLM
        llm, model_name = get_llm(model)
        
        # 生成知识图谱
        graph_data = await generate_graph_data_from_llm(
            llm=llm,
            fileName=file.filename,
            graph=graph,
            embedding_function=EMBEDDING_FUNCTION,
            dimension=EMBEDDING_DIMENSION
        )
        
        # 删除临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        if "error" in graph_data:
            return GraphGenerationResponse(
                success=False,
                message="知识图谱生成失败",
                error=graph_data["error"]
            )
        
        return GraphGenerationResponse(
            success=True,
            message="文档处理和知识图谱生成成功",
            graph_data=graph_data
        )
        
    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        return DocumentUploadResponse(
            success=False,
            message="文档上传处理失败",
            error=str(e)
        )



@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model: str = "deepseek"  # 默认使用deepseek模型
):
    """
    与知识图谱问答系统交互
    """
    try:
        # 从请求体中获取Neo4j配置
        config = None
        if hasattr(request, 'config') and request.config:
            config = Neo4jConnectionConfig(**request.config)
        
        # 获取Neo4j连接
        graph = get_neo4j_graph(config)
        
        # 保存用户消息到Neo4j
        await save_message_to_neo4j(graph, request.session_id, "human", request.question)
        
        # 获取或创建QA系统实例
        qa_system = get_qa_system(request.session_id, model, graph)
        
        # 根据搜索类型选择不同的回答方法
        if request.search_type == "vector":
            response = await qa_system.answer_with_vector_search(request.question, top_k=request.top_k)
        elif request.search_type == "hybrid":
            response = await qa_system.answer_with_hybrid_search(request.question, top_k=request.top_k)
        elif request.search_type == "graph":
            response = await qa_system.answer_with_graph_search(
                request.question, 
                max_depth=request.max_depth, 
                top_k=request.top_k
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的搜索类型: {request.search_type}")
        
        # 保存AI回复到Neo4j
        await save_message_to_neo4j(graph, request.session_id, "ai", response["message"])
        
        return response
        
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        return {
            "session_id": request.session_id,
            "message": "抱歉，处理您的问题时出现了错误。",
            "info": {
                "model": model,
                "mode": request.search_type,
                "error": str(e)
            },
            "user": "chatbot"
        }

@app.post("/clear-history")
async def clear_chat_history(
    request: Dict[str, Any]
):
    """
    清除聊天历史
    """
    global active_qa_systems
    
    try:
        session_id = request.get("session_id")
        config = request.get("config")
        
        if not session_id:
            return {"success": False, "message": "缺少session_id参数"}
        
        # 从Neo4j删除聊天记录
        if config:
            neo4j_config = Neo4jConnectionConfig(**config)
            graph = get_neo4j_graph(neo4j_config)
            await delete_session_from_neo4j(graph, session_id)
        
        # 清除内存中的QA系统实例
        if session_id in active_qa_systems:
            qa_system = active_qa_systems[session_id]
            qa_system.clear_history()
            del active_qa_systems[session_id]
            
        return {"success": True, "message": f"会话 {session_id} 的聊天历史已清除"}
    except Exception as e:
        logger.error(f"清除聊天历史失败: {str(e)}")
        return {"success": False, "message": "清除聊天历史失败", "error": str(e)}

@app.get("/models")
async def get_available_models():
    """
    获取可用的LLM模型列表
    """
    return {
        "models": ["deepseek", "doubao", "qwen"]
    }

@app.post("/files")
async def get_files(
    config: Neo4jConnectionConfig
):
    """
    获取已上传的文件列表
    """
    try:
        # 获取Neo4j连接
        graph = get_neo4j_graph(config)
        
        # 查询所有文档节点
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (c:Chunk {fileName: d.fileName})
        WITH d, collect(c) as chunks, count(c) as chunk_count
        UNWIND chunks as chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
        WITH d, chunk_count, count(DISTINCT e) as entity_count
        RETURN d.fileName as filename, 
               datetime().epochSeconds as created_at,
               chunk_count,
               entity_count
        ORDER BY d.fileName DESC
        """
        
        result = graph.query(query)
        
        files = []
        for record in result:
            files.append({
                "filename": record["filename"],
                "created_at": record["created_at"],
                "chunk_count": record["chunk_count"],
                "entity_count": record["entity_count"]
            })
        
        return {
            "success": True,
            "files": files
        }
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        return {
            "success": False,
            "message": "获取文件列表失败",
            "error": str(e),
            "files": []
        }

@app.delete("/delete-file")
async def delete_file(
    filename: str,
    config: Neo4jConnectionConfig
):
    """
    删除文件
    """
    try:
        # 获取Neo4j连接
        graph = get_neo4j_graph(config)
        
        # 删除相关的文档节点和关系
        delete_query = """
        // 首先找到要删除的Document节点（根据fileName属性）
        MATCH (d:Document {fileName: $filename})
        // 找到所有相关的Chunk节点
        OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
        // 删除Document和Chunk节点及其所有关系
        DETACH DELETE d, c
        RETURN count(DISTINCT d) as deleted_docs, count(DISTINCT c) as deleted_chunks
        """
        
        result = graph.query(delete_query, {"filename": filename})
        
        # 清理孤立的__Entity__节点
        cleanup_query = """
        // 找到没有被任何Chunk节点指向的__Entity__节点
        MATCH (e:__Entity__)
        WHERE NOT EXISTS((:Chunk)-[:HAS_ENTITY]->(e))
        DETACH DELETE e
        RETURN count(e) as deleted_entities
        """
        
        cleanup_result = graph.query(cleanup_query)
        
        if result and result[0]["deleted_docs"] > 0:
            deleted_entities = cleanup_result[0]["deleted_entities"] if cleanup_result else 0
            return {
                "success": True,
                "message": f"文件 {filename} 删除成功",
                "filename": filename,
                "deleted_docs": result[0]["deleted_docs"],
                "deleted_chunks": result[0]["deleted_chunks"],
                "deleted_entities": deleted_entities
            }
        else:
            return {
                "success": False,
                "message": f"文件 {filename} 不存在",
                "filename": filename
            }
        
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}")
        return {
            "success": False,
            "message": "删除文件失败",
            "error": str(e)
        }

@app.post("/sessions")
async def get_sessions(
    config: Neo4jConnectionConfig
):
    """
    获取所有会话列表
    """
    try:
        graph = get_neo4j_graph(config)
        sessions = await get_all_sessions_from_neo4j(graph)
        
        return {
            "success": True,
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {str(e)}")
        return {
            "success": False,
            "message": "获取会话列表失败",
            "error": str(e),
            "sessions": []
        }

@app.post("/chat-history")
async def get_chat_history(
    session_id: str,
    config: Neo4jConnectionConfig
):
    """
    获取指定会话的聊天历史
    """
    try:
        graph = get_neo4j_graph(config)
        chat_history = await load_chat_history_from_neo4j(graph, session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "chat_history": chat_history
        }
        
    except Exception as e:
        logger.error(f"获取聊天历史失败: {str(e)}")
        return {
            "success": False,
            "message": "获取聊天历史失败",
            "error": str(e),
            "chat_history": []
        }

@app.get("/health")
async def health_check():
    """
    健康检查
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
