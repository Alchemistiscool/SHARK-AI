import os
import time
import json
import sqlite3
import psutil
from io import BytesIO
import streamlit as st
from gtts import gTTS

# LangChain & Tooling Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

# ==========================================
# 1. DATABASE MANAGEMENT & AUTH SYSTEM
# ==========================================
DB_FILE = "shark_ai_workspace.db"


def init_db():
    """Initialize SQLite tables for accounts, chat sessions, and messages."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES sessions(chat_id)
        )
    """)
    # Seed default demo account if no users exist
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("User", "password123"))
    conn.commit()
    conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()
    conn.close()
    return user is not None


def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_sessions(username):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT chat_id, title FROM sessions WHERE username=? ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows


def create_session(chat_id, username, title="New Conversation"):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (chat_id, username, title) VALUES (?, ?, ?)",
                (chat_id, username, title))
    conn.commit()
    conn.close()


def delete_session(chat_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    cur.execute("DELETE FROM sessions WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()


def save_message_db(chat_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
    conn.commit()
    conn.close()


def load_messages_db(chat_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


init_db()

# ==========================================
# 2. PAGE CONFIG & CUSTOM GUI CSS
# ==========================================
st.set_page_config(
    page_title="SHARK AI -SWIM",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #05070c 100%);
        color: #e2e8f0;
    }
    div[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 255, 163, 0.15);
    }
    .shark-title {
        font-family: 'Courier New', monospace;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00FFA3, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 18px rgba(0, 255, 163, 0.3);
        margin-bottom: 0px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(0, 255, 163, 0.1);
        color: #00FFA3;
        border: 1px solid rgba(0, 255, 163, 0.3);
    }
    .stChatInputContainer input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(0, 255, 163, 0.3) !important;
        color: #00FFA3 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ==========================================
# 4. LOGIN & ACCOUNT MANAGEMENT INTERFACE
# ==========================================
if not st.session_state.authenticated:
    st.markdown('<div class="shark-title" style="text-align: center;">🦈 SHARK AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Login Portal</p>', unsafe_allow_html=True)

    col_login, _ = st.columns([1, 1])
    with col_login:
        tab_login, tab_signup = st.tabs(["🔒 Sign In", "📝 Create Account"])

        with tab_login:
            login_user = st.text_input("Username", key="login_u")
            login_pass = st.text_input("Password", type="password", key="login_p")
            if st.button("Access Terminal", use_container_width=True):
                if authenticate_user(login_user, login_pass):
                    st.session_state.authenticated = True
                    st.session_state.username = login_user

                    # Set up default initial chat for logged in user
                    user_sessions = get_user_sessions(login_user)
                    if user_sessions:
                        st.session_state.current_chat = user_sessions[0][0]
                    else:
                        init_chat_id = f"session_{int(time.time())}"
                        create_session(init_chat_id, login_user, "Default Neural Session")
                        st.session_state.current_chat = init_chat_id

                    st.session_state.messages = load_messages_db(st.session_state.current_chat)
                    st.success("Access Granted!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")

        with tab_signup:
            new_user = st.text_input("New Username", key="reg_u")
            new_pass = st.text_input("New Password", type="password", key="reg_p")
            if st.button("Register Account", use_container_width=True):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Account created successfully! Please sign in.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill out all fields.")
    st.stop()

# ==========================================
# 5. SIDEBAR CONTROL CENTER
# ==========================================
with st.sidebar:
    st.markdown('<div class="shark-title">🦈 SHARK AI</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="status-badge">USER: {st.session_state.username.upper()}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

    accent_color = st.color_picker("THEME ACCENT", "#00FFA3")

    # --- SESSION MANAGEMENT ---
    st.markdown("### 💬 Saved Chats")
    user_sessions = get_user_sessions(st.session_state.username)
    session_dict = {s[0]: s[1] for s in user_sessions}

    if user_sessions:
        selected_session = st.selectbox(
            "ACTIVE SESSION",
            options=list(session_dict.keys()),
            format_func=lambda x: session_dict.get(x, x),
            index=0 if st.session_state.current_chat not in session_dict else list(session_dict.keys()).index(
                st.session_state.current_chat)
        )

        if selected_session != st.session_state.current_chat:
            st.session_state.current_chat = selected_session
            st.session_state.messages = load_messages_db(selected_session)
            st.rerun()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ New Chat", use_container_width=True):
            new_id = f"session_{int(time.time())}"
            create_session(new_id, st.session_state.username, f"Session #{len(user_sessions) + 1}")
            st.session_state.current_chat = new_id
            st.session_state.messages = []
            st.rerun()
    with col_s2:
        if st.button("🗑️ Delete", use_container_width=True):
            delete_session(st.session_state.current_chat)
            remaining = get_user_sessions(st.session_state.username)
            if remaining:
                st.session_state.current_chat = remaining[0][0]
                st.session_state.messages = load_messages_db(remaining[0][0])
            else:
                new_id = f"session_{int(time.time())}"
                create_session(new_id, st.session_state.username, "Default Session")
                st.session_state.current_chat = new_id
                st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # --- MODEL SETTINGS ---
    st.markdown("### ⚙️ Engine Settings")
    ai_mode = st.selectbox(
        "OPERATIONAL MODE",
        ["Standard AI Assistant", "Mega Overdrive (Best AI)", "Water-Roleplay Chatbot", "Strict Code Architect"]
    )

    ollama_model = st.selectbox(
        "OLLAMA MODEL",
        [
            "qwen2.5",
            "phi3",
            "deepseek-coder",
            "deepseek-coder-v2:16b",
            "llama3.1",
            "llama3",
            "mistral",
            "codellama"
        ],
        index=0
    )

    temperature = st.slider("TEMPERATURE", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("MAX RESPONSE TOKENS", 256, 8192, 2048, 256)

    enable_web_search = st.toggle("🌐 DuckDuckGo Web Search", value=False)
    enable_tts = st.toggle("🔊 Speech Synthesis (TTS)", value=False)

    st.markdown("---")

    # --- DOCUMENT RAG ---
    st.markdown("### 📄 Knowledge Base (RAG)")
    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])

    if uploaded_file and st.button("⚡ Index Knowledge Base", use_container_width=True):
        with st.spinner("Processing & indexing file vectors..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(model=ollama_model)
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.doc_name = uploaded_file.name
            st.success(f"Indexed: {uploaded_file.name}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

    if st.session_state.doc_name:
        st.info(f"Active Document: `{st.session_state.doc_name}`")
        if st.button("❌ Clear Knowledge Base"):
            st.session_state.vector_store = None
            st.session_state.doc_name = None
            st.rerun()

    st.markdown("---")

    # --- RESOURCE MONITOR ---
    st.markdown("### 💻 System Hardware Monitor")
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("CPU Load", f"{cpu_usage}%")
    with col_m2:
        st.metric("RAM Usage", f"{ram_usage}%")

# ==========================================
# 6. DYNAMIC SYSTEM PROMPT CONSTRUCT
# ==========================================
if ai_mode == "Mega Overdrive (Best AI)":
    system_prompt = "You are SHARK AI running on Mega Overdrive. Answer with total accuracy, speed, and analytical rigor."
elif ai_mode == "Water-Roleplay Chatbot":
    system_prompt = "You are SHARK AI in Water-Roleplay mode. Use aquatic, wave, and deep-sea metaphors throughout your responses."
elif ai_mode == "Strict Code Architect":
    system_prompt = "You are an expert Software Architect. Produce clean, optimized, production-ready code with complete documentation."
else:
    system_prompt = "You are SHARK AI, a powerful, helpful, and concise neural assistant."

# ==========================================
# 7. MAIN CHAT INTERFACE
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🦈 SHARK AI "
             "1.0"
             )
    st.caption(f"Account: **{st.session_state.username}** | Model: **{ollama_model}** | Mode: **{ai_mode}**")
with col_h2:
    if st.session_state.messages:
        chat_data = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat Log",
            data=chat_data,
            file_name=f"shark_chat_{st.session_state.current_chat}.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")

# Render active message history
formatted_langchain_messages = []
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    label = f"👤 {st.session_state.username.upper()}" if role == "user" else "🦈 SHARK AI"

    with st.chat_message(role):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; font-weight: bold;'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(content)

    if role == "user":
        formatted_langchain_messages.append(HumanMessage(content=content))
    else:
        formatted_langchain_messages.append(AIMessage(content=content))

# ==========================================
# 8. CHAT EXECUTION ENGINE
# ==========================================
tools = [DuckDuckGoSearchRun()] if enable_web_search else []
llm = ChatOllama(model=ollama_model, temperature=temperature, num_predict=max_tokens)

if prompt := st.chat_input("Transmit message to SHARK AI..."):
    with st.chat_message("user"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; font-weight: bold;'>👤 {st.session_state.username.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(prompt)

    save_message_db(st.session_state.current_chat, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add document RAG context if active
    context_str = ""
    if st.session_state.vector_store is not None:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(prompt)
        if relevant_docs:
            context_str = "\n\n[DOCUMENT KNOWLEDGE CONTEXT]:\n" + "\n".join([d.page_content for d in relevant_docs])

    augmented_prompt = prompt + context_str
    formatted_langchain_messages.append(HumanMessage(content=augmented_prompt))

    with st.chat_message("assistant"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; fontimport os
import time
import json
import sqlite3
import psutil
from io import BytesIO
import streamlit as st
from gtts import gTTS

# LangChain & Tooling Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

# ==========================================
# 1. DATABASE MANAGEMENT & AUTH SYSTEM
# ==========================================
DB_FILE = "shark_ai_workspace.db"


def init_db():
    """Initialize SQLite tables for accounts, chat sessions, and messages."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES sessions(chat_id)
        )
    """)
    # Seed default demo account if no users exist
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("User", "password123"))
    conn.commit()
    conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()
    conn.close()
    return user is not None


def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_sessions(username):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT chat_id, title FROM sessions WHERE username=? ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows


def create_session(chat_id, username, title="New Conversation"):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (chat_id, username, title) VALUES (?, ?, ?)",
                (chat_id, username, title))
    conn.commit()
    conn.close()


def delete_session(chat_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    cur.execute("DELETE FROM sessions WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()


def save_message_db(chat_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
    conn.commit()
    conn.close()


def load_messages_db(chat_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


init_db()

# ==========================================
# 2. PAGE CONFIG & CUSTOM GUI CSS
# ==========================================
st.set_page_config(
    page_title="SHARK AI -SWIM",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #05070c 100%);
        color: #e2e8f0;
    }
    div[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 255, 163, 0.15);
    }
    .shark-title {
        font-family: 'Courier New', monospace;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00FFA3, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 18px rgba(0, 255, 163, 0.3);
        margin-bottom: 0px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(0, 255, 163, 0.1);
        color: #00FFA3;
        border: 1px solid rgba(0, 255, 163, 0.3);
    }
    .stChatInputContainer input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(0, 255, 163, 0.3) !important;
        color: #00FFA3 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ==========================================
# 4. LOGIN & ACCOUNT MANAGEMENT INTERFACE
# ==========================================
if not st.session_state.authenticated:
    st.markdown('<div class="shark-title" style="text-align: center;">🦈 SHARK AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Login Portal</p>', unsafe_allow_html=True)

    col_login, _ = st.columns([1, 1])
    with col_login:
        tab_login, tab_signup = st.tabs(["🔒 Sign In", "📝 Create Account"])

        with tab_login:
            login_user = st.text_input("Username", key="login_u")
            login_pass = st.text_input("Password", type="password", key="login_p")
            if st.button("Access Terminal", use_container_width=True):
                if authenticate_user(login_user, login_pass):
                    st.session_state.authenticated = True
                    st.session_state.username = login_user

                    # Set up default initial chat for logged in user
                    user_sessions = get_user_sessions(login_user)
                    if user_sessions:
                        st.session_state.current_chat = user_sessions[0][0]
                    else:
                        init_chat_id = f"session_{int(time.time())}"
                        create_session(init_chat_id, login_user, "Default Neural Session")
                        st.session_state.current_chat = init_chat_id

                    st.session_state.messages = load_messages_db(st.session_state.current_chat)
                    st.success("Access Granted!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")

        with tab_signup:
            new_user = st.text_input("New Username", key="reg_u")
            new_pass = st.text_input("New Password", type="password", key="reg_p")
            if st.button("Register Account", use_container_width=True):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Account created successfully! Please sign in.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill out all fields.")
    st.stop()

# ==========================================
# 5. SIDEBAR CONTROL CENTER
# ==========================================
with st.sidebar:
    st.markdown('<div class="shark-title">🦈 SHARK AI</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="status-badge">USER: {st.session_state.username.upper()}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

    accent_color = st.color_picker("THEME ACCENT", "#00FFA3")

    # --- SESSION MANAGEMENT ---
    st.markdown("### 💬 Saved Chats")
    user_sessions = get_user_sessions(st.session_state.username)
    session_dict = {s[0]: s[1] for s in user_sessions}

    if user_sessions:
        selected_session = st.selectbox(
            "ACTIVE SESSION",
            options=list(session_dict.keys()),
            format_func=lambda x: session_dict.get(x, x),
            index=0 if st.session_state.current_chat not in session_dict else list(session_dict.keys()).index(
                st.session_state.current_chat)
        )

        if selected_session != st.session_state.current_chat:
            st.session_state.current_chat = selected_session
            st.session_state.messages = load_messages_db(selected_session)
            st.rerun()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ New Chat", use_container_width=True):
            new_id = f"session_{int(time.time())}"
            create_session(new_id, st.session_state.username, f"Session #{len(user_sessions) + 1}")
            st.session_state.current_chat = new_id
            st.session_state.messages = []
            st.rerun()
    with col_s2:
        if st.button("🗑️ Delete", use_container_width=True):
            delete_session(st.session_state.current_chat)
            remaining = get_user_sessions(st.session_state.username)
            if remaining:
                st.session_state.current_chat = remaining[0][0]
                st.session_state.messages = load_messages_db(remaining[0][0])
            else:
                new_id = f"session_{int(time.time())}"
                create_session(new_id, st.session_state.username, "Default Session")
                st.session_state.current_chat = new_id
                st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # --- MODEL SETTINGS ---
    st.markdown("### ⚙️ Engine Settings")
    ai_mode = st.selectbox(
        "OPERATIONAL MODE",
        ["Standard AI Assistant", "Mega Overdrive (Best AI)", "Water-Roleplay Chatbot", "Strict Code Architect"]
    )

    ollama_model = st.selectbox(
        "OLLAMA MODEL",
        [
            "qwen2.5",
            "phi3",
            "deepseek-coder",
            "deepseek-coder-v2:16b",
            "llama3.1",
            "llama3",
            "mistral",
            "codellama"
        ],
        index=0
    )

    temperature = st.slider("TEMPERATURE", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("MAX RESPONSE TOKENS", 256, 8192, 2048, 256)

    enable_web_search = st.toggle("🌐 DuckDuckGo Web Search", value=False)
    enable_tts = st.toggle("🔊 Speech Synthesis (TTS)", value=False)

    st.markdown("---")

    # --- DOCUMENT RAG ---
    st.markdown("### 📄 Knowledge Base (RAG)")
    uploaded_file = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])

    if uploaded_file and st.button("⚡ Index Knowledge Base", use_container_width=True):
        with st.spinner("Processing & indexing file vectors..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(model=ollama_model)
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.doc_name = uploaded_file.name
            st.success(f"Indexed: {uploaded_file.name}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

    if st.session_state.doc_name:
        st.info(f"Active Document: `{st.session_state.doc_name}`")
        if st.button("❌ Clear Knowledge Base"):
            st.session_state.vector_store = None
            st.session_state.doc_name = None
            st.rerun()

    st.markdown("---")

    # --- RESOURCE MONITOR ---
    st.markdown("### 💻 System Hardware Monitor")
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("CPU Load", f"{cpu_usage}%")
    with col_m2:
        st.metric("RAM Usage", f"{ram_usage}%")

# ==========================================
# 6. DYNAMIC SYSTEM PROMPT CONSTRUCT
# ==========================================
if ai_mode == "Mega Overdrive (Best AI)":
    system_prompt = "You are SHARK AI running on Mega Overdrive. Answer with total accuracy, speed, and analytical rigor."
elif ai_mode == "Water-Roleplay Chatbot":
    system_prompt = "You are SHARK AI in Water-Roleplay mode. Use aquatic, wave, and deep-sea metaphors throughout your responses."
elif ai_mode == "Strict Code Architect":
    system_prompt = "You are an expert Software Architect. Produce clean, optimized, production-ready code with complete documentation."
else:
    system_prompt = "You are SHARK AI, a powerful, helpful, and concise neural assistant."

# ==========================================
# 7. MAIN CHAT INTERFACE
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🦈 SHARK AI "
             "1.0"
             )
    st.caption(f"Account: **{st.session_state.username}** | Model: **{ollama_model}** | Mode: **{ai_mode}**")
with col_h2:
    if st.session_state.messages:
        chat_data = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat Log",
            data=chat_data,
            file_name=f"shark_chat_{st.session_state.current_chat}.json",
            mime="application/json",
            use_container_width=True
        )

st.markdown("---")

# Render active message history
formatted_langchain_messages = []
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    label = f"👤 {st.session_state.username.upper()}" if role == "user" else "🦈 SHARK AI"

    with st.chat_message(role):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; font-weight: bold;'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(content)

    if role == "user":
        formatted_langchain_messages.append(HumanMessage(content=content))
    else:
        formatted_langchain_messages.append(AIMessage(content=content))

# ==========================================
# 8. CHAT EXECUTION ENGINE
# ==========================================
tools = [DuckDuckGoSearchRun()] if enable_web_search else []
llm = ChatOllama(model=ollama_model, temperature=temperature, num_predict=max_tokens)

if prompt := st.chat_input("Transmit message to SHARK AI..."):
    with st.chat_message("user"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; font-weight: bold;'>👤 {st.session_state.username.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(prompt)

    save_message_db(st.session_state.current_chat, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add document RAG context if active
    context_str = ""
    if st.session_state.vector_store is not None:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(prompt)
        if relevant_docs:
            context_str = "\n\n[DOCUMENT KNOWLEDGE CONTEXT]:\n" + "\n".join([d.page_content for d in relevant_docs])

    augmented_prompt = prompt + context_str
    formatted_langchain_messages.append(HumanMessage(content=augmented_prompt))

    with st.chat_message("assistant"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.7; color:{accent_color}; font-weight: bold;'>🦈 SHARK AI</span>",
            unsafe_allow_html=True,
        )

        status_label = "Searching Knowledge Networks..." if enable_web_search else "Generating Neural Response..."
        start_time = time.time()

        with st.spinner(status_label):
            try:
                if enable_web_search:
                    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
                    response = agent.invoke({"messages": formatted_langchain_messages})
                    reply_text = response["messages"][-1].content
                else:
                    execution_messages = [SystemMessage(content=system_prompt)] + formatted_langchain_messages
                    response = llm.invoke(execution_messages)
                    reply_text = response.content

                elapsed_time = round(time.time() - start_time, 2)

                st.markdown(reply_text)
                st.caption(f"⚡ Response generated in `{elapsed_time}s` using `{ollama_model}`")

                save_message_db(st.session_state.current_chat, "assistant", reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})

                if enable_tts and reply_text:
                    tts_bytes = BytesIO()
                    tts = gTTS(text=reply_text[:300], lang="en")
                    tts.write_to_fp(tts_bytes)
                    st.audio(how to screenshot linux arch gnometts_bytes.getvalue(), format="audio/mp3", autoplay=True)

            except Exception as e:
                st.error(f"[SYSTEM EXECUTION ERR-weight: bold;'>🦈 SHARK AI</span>",
            unsafe_allow_html=True,
        )

        status_label = "Searching Knowledge Networks..." if enable_web_search else "Generating Neural Response..."
        start_time = time.time()

        with st.spinner(status_label):
            try:
                if enable_web_search:
                    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
                    response = agent.invoke({"messages": formatted_langchain_messages})
                    reply_text = response["messages"][-1].content
                else:
                    execution_messages = [SystemMessage(content=system_prompt)] + formatted_langchain_messages
                    response = llm.invoke(execution_messages)
                    reply_text = response.content

                elapsed_time = round(time.time() - start_time, 2)

                st.markdown(reply_text)
                st.caption(f"⚡ Response generated in `{elapsed_time}s` using `{ollama_model}`")

                save_message_db(st.session_state.current_chat, "assistant", reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})

                if enable_tts and reply_text:
                    tts_bytes = BytesIO()
                    tts = gTTS(text=reply_text[:300], lang="en")
                    tts.write_to_fp(tts_bytes)
                    st.audio(how to screenshot linux arch gnometts_bytes.getvalue(), format="audio/mp3", autoplay=True)

            except Exception as e:
                st.error(f"[SYSTEM EXECUTION ERR
