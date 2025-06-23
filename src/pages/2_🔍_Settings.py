import json
import streamlit as st
import requests
import os
import tempfile
import time

st.set_page_config(page_title="Settings", page_icon="🔍")

st.markdown("# ⚙️ 系统设置")

# 创建标签页
tab1, tab2 = st.tabs(["🗄️ Neo4j 数据库", "📁 文档管理"])

with tab1:
    st.markdown("## Neo4j 数据库设置")

    # 默认值（可根据实际情况修改）
    default_uri = st.session_state.get("neo4j_uri", "neo4j+s://0f99bb11.databases.neo4j.io")
    default_username = st.session_state.get("neo4j_username", "neo4j")
    default_password = st.session_state.get("neo4j_password", "")

    uri = st.text_input("Neo4j URI", value=default_uri)
    username = st.text_input("用户名", value=default_username)
    password = st.text_input("密码", value=default_password, type="password")

    col1, col2 = st.columns(2)
    with col1:
        # 保存到 session_state
        if st.button("💾 保存设置", use_container_width=True):
            st.session_state["neo4j_uri"] = uri
            st.session_state["neo4j_username"] = username
            st.session_state["neo4j_password"] = password
            st.success("设置已保存！")

    with col2:
        # 测试连接
        if st.button("🔗 测试连接", use_container_width=True):
            with st.spinner("正在测试连接..."):
                payload = {
                    "uri": uri,
                    "username": username,
                    "password": password
                }
                try:
                    resp = requests.post("http://localhost:8000/connect", json=payload, timeout=10)
                    data = resp.json()
                    if data.get("success"):
                        st.session_state["neo4j_connect_ok"] = True
                        st.success(f"✅ 连接成功，节点数: {data.get('node_count')}")
                    else:
                        st.session_state["neo4j_connect_ok"] = False
                        st.error(f"❌ 连接失败: {data.get('error', data.get('message', '未知错误'))}")
                except Exception as e:
                    st.session_state["neo4j_connect_ok"] = False
                    st.error(f"❌ 连接测试异常: {e}")

with tab2:
    st.markdown("## 📄 文档上传与知识图谱生成")
    
    # 检查Neo4j连接状态
    if not st.session_state.get("neo4j_connect_ok", False):
        st.warning("⚠️ 请先在 Neo4j 数据库标签页中配置并测试连接！")
        st.stop()
    
    # 文件上传区域
    st.markdown("### 📤 上传文档")
    uploaded_file = st.file_uploader(
        "选择要上传的文档",
        type=['txt', 'pdf', 'md'],
        help="支持的文件格式：TXT, PDF, MD"
    )
    
    # 模型选择
    selected_model = st.selectbox(
        "🤖 选择LLM模型",
        ["deepseek", "doubao", "qwen"],
        help="用于生成知识图谱的大语言模型"
    )
    
    # 上传和处理按钮
    if uploaded_file is not None:
        if st.button("📤 上传并生成知识图谱", use_container_width=True, type="primary"):
            with st.spinner("正在上传文档并生成知识图谱，这可能需要几分钟..."):
                try:
                    # 准备Neo4j配置
                    config = {
                        "uri": st.session_state.get("neo4j_uri"),
                        "username": st.session_state.get("neo4j_username"),
                        "password": st.session_state.get("neo4j_password")
                    }
                    
                    # 上传文件并生成图谱
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "model": selected_model,
                        "config": json.dumps(config)
                    }
                    
                    resp = requests.post(
                        "http://localhost:8000/upload-and-generate",
                        files=files,
                        data=data,
                        # timeout=600
                    )
                    
                    result = resp.json()
                    
                    if result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        if result.get("graph_data"):
                            st.json(result["graph_data"])
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('message')}: {result.get('error', '')}")
                        
                except Exception as e:
                    st.error(f"❌ 操作失败: {str(e)}")
    
    # 文件管理区域
    st.markdown("### 🗂️ 文件管理")
    
    # 获取并显示文件列表
    def get_file_list():
        try:
            config = {
                "uri": st.session_state.get("neo4j_uri"),
                "username": st.session_state.get("neo4j_username"),
                "password": st.session_state.get("neo4j_password")
            }
            
            resp = requests.post(
                "http://localhost:8000/files",
                json=config,
                timeout=30
            )
            
            result = resp.json()
            if result.get("success"):
                return result.get("files", [])
            else:
                st.error(f"获取文件列表失败: {result.get('message', '')}")
                return []
        except Exception as e:
            st.error(f"获取文件列表异常: {str(e)}")
            return []
    
    # 显示已上传文件列表
    st.markdown("#### 📋 已上传文件")
    
    if st.button("🔄 刷新文件列表", type="secondary"):
        st.rerun()
    
    files = get_file_list()
    
    if files:
        # 创建文件表格
        import pandas as pd
        df = pd.DataFrame(files)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            df,
            column_config={
                "filename": "文件名",
                "created_at": "上传时间",
                "chunk_count": "文档块数量",
                "entity_count": "实体数量"
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 删除文件功能
        st.markdown("#### 🗑️ 删除文件")
        
        # 文件选择下拉框
        filename_options = [f["filename"] for f in files]
        selected_file = st.selectbox(
            "选择要删除的文件",
            options=["请选择文件..."] + filename_options,
            help="从已上传的文件中选择要删除的文件"
        )
        
        if selected_file != "请选择文件...":
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ 删除", type="secondary", use_container_width=True):
                    with st.spinner(f"正在删除文件 {selected_file}..."):
                        try:
                            config = {
                                "uri": st.session_state.get("neo4j_uri"),
                                "username": st.session_state.get("neo4j_username"),
                                "password": st.session_state.get("neo4j_password")
                            }
                            
                            params = {
                                "filename": selected_file
                            }
                            
                            resp = requests.delete(
                                "http://localhost:8000/delete-file",
                                params=params,
                                json=config,
                                timeout=30
                            )
                            
                            result = resp.json()
                            
                            if result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('message')}: {result.get('error', '')}")
                                
                        except Exception as e:
                            st.error(f"❌ 删除文件失败: {str(e)}")
    else:
        st.info("📝 暂无已上传的文件")
        
    # 知识图谱生成进度跟踪
    st.markdown("#### 📊 知识图谱生成进度")
    
    if "graph_generation_progress" in st.session_state:
        progress_data = st.session_state["graph_generation_progress"]
        
        # 显示进度条
        progress_bar = st.progress(progress_data.get("progress", 0))
        st.write(f"状态: {progress_data.get('status', '未知')}")
        
        if progress_data.get("completed", False):
            st.success("✅ 知识图谱生成完成！")
            del st.session_state["graph_generation_progress"]
    else:
        st.info("💡 当前没有正在进行的知识图谱生成任务")