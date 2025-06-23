# 知识图谱问答系统

本项目是一个基于 Neo4j 图数据库和大语言模型（LLM）的智能知识图谱问答系统。  
用户可以上传文档，自动生成知识图谱，并通过自然语言与知识图谱进行智能问答。  
在生成知识图谱时，本项目默认使用的是军事相关的prompt，如有需要自己进行修改。

## ✨ 项目亮点

- **文档自动解析**：支持 TXT、PDF、MD 等多种格式文档上传。
- **知识图谱自动生成**：结合 LLM 自动抽取知识并构建图谱。
- **多模型支持**：可选 deepseek、doubao、qwen 等主流大模型。
- **Neo4j 图数据库**：高效存储与查询知识图谱。
- **Streamlit 前端**：界面友好，操作简单。
- **灵活问答**：支持向量、混合、图谱等多种检索方式。

---

## 🚀 快速开始


### 1. 安装依赖

建议使用 Python 3.10，推荐虚拟环境。

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `example.env` 为 `.env`，并根据实际情况填写 Neo4j 连接、嵌入模型等信息。

建议使用neo4j aura
```env
NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_password"

EMBEDDING_MODEL_NAME="your_embedding_model"
```

### 3. 启动后端服务

```bash
python src/api.py
```

### 4. 启动前端（Streamlit）

另开一个终端：

```bash
streamlit run src/Home.py
```


## 🗂️ 主要目录结构

```
Graph_chat/
│
├── src/
│   ├── api.py                # FastAPI 后端主入口
│   ├── Home.py               # Streamlit 前端主入口
│   ├── pages/                # Streamlit 多页面
│   ├── QA_system.py          # 问答系统核心逻辑
│   ├── document_loader.py    # 文档解析与入库
│   ├── generate_graph_from_llm.py # LLM生成知识图谱
│   ├── shared/               # 公共函数、配置
│   └── ...                   # 其它模块
├── requirements.txt
├── .env.example
└── README.md
```

## 提示
如果需要加入新的大模型，比如zhipu：  
1.在环境变量中配置zhipu的API信息  
2.在 llm.py 中添加zhipu的初始化逻辑  
3.在 api.py 中将"zhipu"添加到支持的模型列表  
4.在两个前端页面的模型选择下拉框中添加"zhipu"选项

---


## 📝 TODO

**todo1: 知识图谱可视化**：集成像 Neo4j Bloom 前端可视化库
**todo2: 模型热插拔**：支持在线切换/加载不同 LLM
