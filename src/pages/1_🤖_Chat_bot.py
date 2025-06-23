import streamlit as st
import requests
import time
import random
import string

st.set_page_config(page_title="Chat_bot", page_icon="🤖")

# API 基础 URL
API_BASE_URL = "http://localhost:8000"

# 检查 Neo4j 设置和连接状态
uri = st.session_state.get("neo4j_uri")
username = st.session_state.get("neo4j_username")
password = st.session_state.get("neo4j_password")
connect_ok = st.session_state.get("neo4j_connect_ok", False)

if not (uri and username and password and connect_ok):
    st.warning("请先在 Settings 页面设置并测试 Neo4j 连接！")
    st.stop()

# ------------------ 对话管理 ------------------
# 从Neo4j加载历史会话
def load_sessions_from_neo4j():
    try:
        config = {
            "uri": st.session_state.get("neo4j_uri"),
            "username": st.session_state.get("neo4j_username"),
            "password": st.session_state.get("neo4j_password")
        }
        
        resp = requests.post(
            "http://localhost:8000/sessions",
            json=config,
            timeout=10
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                return result.get("sessions", [])
    except Exception as e:
        st.error(f"加载历史会话失败: {e}")
    return []

# 从Neo4j加载指定会话的聊天历史
def load_chat_history_from_neo4j(session_id):
    try:
        config = {
            "uri": st.session_state.get("neo4j_uri"),
            "username": st.session_state.get("neo4j_username"),
            "password": st.session_state.get("neo4j_password")
        }
        
        params = {"session_id": session_id}
        
        resp = requests.post(
            "http://localhost:8000/chat-history",
            params=params,
            json=config,
            timeout=10
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                return result.get("chat_history", [])
    except Exception as e:
        st.error(f"加载聊天历史失败: {e}")
    return []



if 'sessions' not in st.session_state:
    st.session_state['sessions'] = {}
    # 加载Neo4j中的历史会话
    neo4j_sessions = load_sessions_from_neo4j()
    for session in neo4j_sessions:
        session_id = session["session_id"]
        # 为每个历史会话创建空的本地记录，实际内容会在切换时加载
        st.session_state['sessions'][session_id] = []

if 'current_session' not in st.session_state:
    # 如果有历史会话，选择最新的；否则创建新会话
    if st.session_state['sessions']:
        # 选择第一个会话（已按时间排序）
        st.session_state['current_session'] = list(st.session_state['sessions'].keys())[0]
    else:
        # 创建新会话
        session_id = f"chat_{int(time.time())}_{random.randint(1000,9999)}"
        st.session_state['sessions'][session_id] = []
        st.session_state['current_session'] = session_id

# 生成随机 session_id
def gen_session_id():
    return f"chat_{int(time.time())}_{''.join(random.choices(string.ascii_letters+string.digits, k=6))}"

# 侧边栏：对话列表与新建
with st.sidebar:
    st.markdown("## 💬 对话列表")
    session_ids = list(st.session_state['sessions'].keys())
    def get_chat_title(idx):
        return f"对话 {idx+1}"
    
    with st.container():
        for idx, sid in enumerate(session_ids):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                btn_label = get_chat_title(idx)
                if st.button(btn_label, key=f"chatbtn_{sid}", help="切换对话", use_container_width=True):
                    # 切换对话时从Neo4j加载聊天历史
                    if sid != st.session_state['current_session']:
                        chat_history = load_chat_history_from_neo4j(sid)
                        st.session_state['sessions'][sid] = chat_history
                    st.session_state['current_session'] = sid
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{sid}", help="删除对话", use_container_width=True):
                    # 删除对话
                    if len(session_ids) > 1:  # 至少保留一个对话
                        try:
                            # 通过API删除会话（包括Neo4j和内存中的QA系统）
                            payload = {
                                "session_id": sid,
                                "config": {
                                    "uri": uri,
                                    "username": username,
                                    "password": password
                                }
                            }
                            response = requests.post(f"{API_BASE_URL}/clear-history", json=payload)
                            if response.status_code == 200:
                                result = response.json()
                                if result.get("success"):
                                    # 从本地删除
                                    del st.session_state['sessions'][sid]
                                    # 如果删除的是当前对话，切换到其他对话
                                    if sid == st.session_state['current_session']:
                                        remaining_sessions = list(st.session_state['sessions'].keys())
                                        if remaining_sessions:
                                            st.session_state['current_session'] = remaining_sessions[0]
                                            # 加载新当前对话的历史
                                            chat_history = load_chat_history_from_neo4j(remaining_sessions[0])
                                            st.session_state['sessions'][remaining_sessions[0]] = chat_history
                                    st.success(f"对话已删除")
                                    st.rerun()
                                else:
                                    st.error(f"删除对话失败: {result.get('message', '未知错误')}")
                            else:
                                st.error("删除对话失败")
                        except Exception as e:
                            st.error(f"删除对话时出错: {e}")
                    else:
                        st.warning("至少需要保留一个对话")
    
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("➕ 新建对话", use_container_width=True):
        new_id = gen_session_id()
        st.session_state['sessions'][new_id] = []
        st.session_state['current_session'] = new_id
        st.rerun()

cur_session = st.session_state['current_session']
chat_history = st.session_state['sessions'][cur_session]

st.markdown(f"### 🤖 当前对话")

# ------------------ 对话历史滚动容器（带头像） ------------------
chat_box_style = """
    max-height: calc(100vh - 280px);
    overflow-y: auto;
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 160px;
    border: 1px solid #e1e5e9;
"""
user_avatar = "<img src='https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png' width='36' style='border-radius:50%;vertical-align:middle;'>"
llm_avatar = "<img src='https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png' width='36' style='border-radius:50%;vertical-align:middle;'>"

with st.container():
    st.markdown(f"<div style='{chat_box_style}'>", unsafe_allow_html=True)
    for msg in chat_history:
        if msg['role'] == 'user':
            st.markdown(f"<div style='display:flex;justify-content:flex-end;align-items:center;margin-bottom:8px;'><span style='background:#4F8BF9;color:white;padding:8px 12px;border-radius:8px 8px 2px 8px;max-width:70%;display:inline-block;'>{msg['content']}</span><span style='margin-left:8px'>{user_avatar}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='display:flex;justify-content:flex-start;align-items:center;margin-bottom:8px;'><span style='margin-right:8px'>{llm_avatar}</span><span style='background:#e9ecef;color:#222;padding:8px 12px;border-radius:8px 8px 8px 2px;max-width:70%;display:inline-block;'>{msg['content']}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ 悬浮输入区（大输入框+左下角按钮） ------------------
input_box_style = """
    position: fixed;
    left: 0; right: 0; bottom: 0;
    background: linear-gradient(to top, #ffffff 0%, #ffffff 90%, rgba(255,255,255,0.95) 100%);
    box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
    padding: 20px;
    z-index: 1000;
    border-top: 1px solid #e1e5e9;
"""

st.markdown(f"<div style='{input_box_style}'>", unsafe_allow_html=True)

# 主输入区域
user_input = st.text_area(
    "💭 请输入您的问题...", 
    key=f"input_{cur_session}", 
    height=120, 
    max_chars=2000, 
    label_visibility="collapsed",
    placeholder="在这里输入您的问题，支持多行输入..."
)

# 底部控制栏
col1, col2, col3, col4 = st.columns([2, 2, 6, 2])
with col1:
    llm_model = st.selectbox(
        "🤖 模型", 
        ["deepseek", "doubao", "qwen"], 
        key=f"llm_{cur_session}",
        help="选择大语言模型"
    )
with col2:
    search_type = st.selectbox(
        "🔍 检索", 
        ["vector", "hybrid", "graph"], 
        key=f"search_{cur_session}",
        help="选择检索方式"
    )
with col3:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
with col4:
    send_clicked = st.button(
        "📤 发送", 
        key=f"send_{cur_session}", 
        use_container_width=True,
        type="primary",
        help="发送消息 (Ctrl+Enter)"
    )

st.markdown("</div>", unsafe_allow_html=True)

# 发送逻辑
if 'send_flag' not in st.session_state:
    st.session_state['send_flag'] = {}
if send_clicked:
    st.session_state['send_flag'][cur_session] = True

if st.session_state['send_flag'].get(cur_session, False):
    if user_input.strip():
        # 记录用户消息
        chat_history.append({"role": "user", "content": user_input})
        st.session_state['sessions'][cur_session] = chat_history
        # 组织请求
        payload = {
            "session_id": cur_session,
            "question": user_input,
            "search_type": search_type,
            "top_k": 5,
            "max_depth": 3,
            "config": {
                "uri": uri,
                "username": username,
                "password": password
            }
        }
        params = {"model": llm_model}
        with st.spinner("LLM思考中..."):
            try:
                resp = requests.post("http://localhost:8000/chat", json=payload, params=params, timeout=60)
                data = resp.json()
                answer = data.get("message", "[无回复]")
            except Exception as e:
                answer = f"[请求失败] {e}"
        # 记录LLM回复
        chat_history.append({"role": "assistant", "content": answer})
        st.session_state['sessions'][cur_session] = chat_history
    st.session_state['send_flag'][cur_session] = False
    st.rerun()