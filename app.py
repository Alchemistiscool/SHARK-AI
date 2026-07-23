import streamlit as st
import warnings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="SHARK AI 0.5 ALPHA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SIDEBAR SETTINGS & DYNAMIC COLOR SELECTION ---
with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; margin-bottom: 20px; text-shadow: 0 0 10px rgba(255,255,255,0.3);'>⚡ SYSTEM CONFIG</h2>",
        unsafe_allow_html=True)

    accent_color = st.selectbox(
        "ACCENT COLOR THEME",
        options=["Cyber Cyan", "Matrix Green", "Neon Red", "Synth Purple", "Gold Alert"],
        index=0
    )

    COLOR_MAP = {
        "Cyber Cyan": "#00f0ff",
        "Matrix Green": "#00ff66",
        "Neon Red": "#ff0055",
        "Synth Purple": "#b000ff",
        "Gold Alert": "#ffaa00"
    }

    HEX_COLOR = COLOR_MAP[accent_color]

    st.divider()

    st.markdown("### 🧬 Neural Evolution")
    research_topic = st.text_input("RESEARCH TOPIC", placeholder="e.g. Quantum Computing")

    if st.button("EXECUTE UPGRADE"):
        if research_topic.strip():
            with st.spinner("Adapting neural weights..."):
                meta_prompt = f"Update this System Prompt to specialize in: {research_topic}\n\nCurrent Prompt:\n{st.session_state.get('system_prompt', '')}"
                llm_temp = ChatOllama(model="llama3.1", temperature=0.7)
                res = llm_temp.invoke([HumanMessage(content=meta_prompt)])
                st.session_state.system_prompt = res.content.strip()
                st.success(f"System adapted to: '{research_topic}'")

    st.divider()

    with st.expander("🔍 View Active Prompt"):
        st.code(st.session_state.get("system_prompt", ""), language="text")

    st.divider()

    if st.button("RESET SESSION"):
        st.session_state.messages = []
        st.rerun()

# --- 3. ULTIMATE ANIMATED CYBER CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;600&display=swap');

    :root {{
        --theme-color: {HEX_COLOR};
        --theme-glow: {HEX_COLOR}88;
        --theme-dim: {HEX_COLOR}22;
        --bg-dark: #010204;
        --bg-panel: rgba(4, 9, 16, 0.75);
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Fira Code', monospace !important;
        background-color: var(--bg-dark);
        color: #e0f7ff;
    }}

    .stApp {{
        background: radial-gradient(circle at 50% 0%, var(--theme-dim) 0%, var(--bg-dark) 80%);
    }}

    /* CRT Scanlines Overlay */
    .stApp::after {{
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(
            to bottom,
            rgba(255,255,255,0),
            rgba(255,255,255,0) 50%,
            rgba(0,0,0,0.12) 50%,
            rgba(0,0,0,0.12)
        );
        background-size: 100% 4px;
        z-index: 9999;
        pointer-events: none;
        opacity: 0.35;
    }}

    ::selection {{
        background: var(--theme-color);
        color: #000;
    }}

    .banner-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
        padding-top: 10px;
    }}

    @keyframes breathingGlow {{
        0% {{ box-shadow: 0 0 15px var(--theme-dim), inset 0 0 10px var(--theme-dim); border-color: var(--theme-dim); }}
        50% {{ box-shadow: 0 0 30px var(--theme-glow), inset 0 0 20px var(--theme-glow); border-color: var(--theme-color); }}
        100% {{ box-shadow: 0 0 15px var(--theme-dim), inset 0 0 10px var(--theme-dim); border-color: var(--theme-dim); }}
    }}

    /* Fixed Raw Preformatted ASCII Styling */
    .ascii-banner {{
        color: var(--theme-color);
        font-family: 'Fira Code', 'Courier New', monospace !important;
       
        word-break: normal !important;
        word-wrap: normal !important;
        background: var(--bg-panel);
        backdrop-filter: blur(10px);
        padding: 25px 35px;
        border-radius: 6px;
        border: 1px solid var(--theme-color);
        line-height: 1.2 !important;
        font-size: 11px !important;
        font-weight: 600;
        overflow-x: auto;
        text-shadow: 0 0 8px var(--theme-glow);
        animation: breathingGlow 4s infinite ease-in-out;
        display: block;
    }}

    section[data-testid="stSidebar"] {{
        background-color: rgba(2, 4, 8, 0.95);
        border-right: 1px solid var(--theme-dim);
    }}

    div[data-testid="stChatMessage"] {{
        background: var(--bg-panel) !important;
        backdrop-filter: blur(8px);
        border: 1px solid var(--theme-dim) !important;
        border-radius: 8px !important;
        padding: 15px 20px !important;
        margin-bottom: 15px !important;
        transition: all 0.3s ease;
    }}
    div[data-testid="stChatMessage"]:hover {{
        border-color: var(--theme-glow) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }}

    .stButton>button {{
        background: transparent !important;
        color: var(--theme-color) !important;
        border: 1px solid var(--theme-color) !important;
        border-radius: 3px !important;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 5px var(--theme-dim);
    }}
    .stButton>button:hover {{
        background: var(--theme-color) !important;
        color: var(--bg-dark) !important;
        box-shadow: 0 0 20px var(--theme-glow) !important;
        transform: translateY(-2px);
    }}

    .stTextInput>div>div>input, .stChatInputContainer textarea {{
        background-color: rgba(2, 6, 12, 0.8) !important;
        color: var(--theme-color) !important;
        border: 1px solid var(--theme-dim) !important;
        border-radius: 4px !important;
        font-size: 14px;
        letter-spacing: 0.5px;
    }}
    .stTextInput>div>div>input:focus, .stChatInputContainer textarea:focus {{
        border-color: var(--theme-color) !important;
        box-shadow: 0 0 10px var(--theme-glow) !important;
    }}

    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg-dark);
        border-left: 1px solid var(--theme-dim);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--theme-dim);
        border-radius: 0px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--theme-color);
        box-shadow: 0 0 10px var(--theme-glow);
    }}
</style>
""", unsafe_allow_html=True)

# --- 4. ASCII BANNER ---
COMPACT_SHARK_LOGO = r"""
  ___  _  _    _    ___  _  __    _   ___  0.5 ALPHA
 / __|| || |  /_\  | _ \| |/ /   /_\  |_ _|
 \__ \| __ | / _ \ |   /| ' <   / _ \  | | 
 |___/|_||_/_/   \_|_|_\|_|\_\ /_/ \_ |___|

                __/\_
         ____.-'     `~-.
        /__      SHARK    \
       <___)-.._____..--'

  ===================================
  Developed by alchem!st | July 2026
  ===================================
"""

st.markdown(
    f'<div class="banner-container"><pre class="ascii-banner">{COMPACT_SHARK_LOGO}</pre></div>',
    unsafe_allow_html=True
)

# --- 5. INITIALIZE STATE & AGENT ---
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """
    You are Shark AI (0.5 ALPHA)—an elite, highly versatile AI assistant possessing immense problem-solving power, senior-level coding expertise, and deep knowledge across all academic and creative disciplines.
    You speak with high intelligence, clarity, and decisive strength.
    """

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOllama(model="llama3.1", temperature=0.7)
agent = create_react_agent(llm, tools=[], prompt=st.session_state.system_prompt)

# --- 6. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    label = "👤 SYSTEM.USER" if role == "user" else "🦈 SHARK.AI.CORE"
    with st.chat_message(role):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.8; letter-spacing:1.5px; color:{HEX_COLOR};'>{label}</span>",
            unsafe_allow_html=True)
        st.markdown(msg.content)

# --- 7. CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("TRANSMIT COMMAND TO SHARK AI..."):
    with st.chat_message("user"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.8; letter-spacing:1.5px; color:{HEX_COLOR};'>👤 SYSTEM.USER</span>",
            unsafe_allow_html=True)
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        st.markdown(
            f"<span style='font-size:0.75em; opacity:0.8; letter-spacing:1.5px; color:{HEX_COLOR};'>🦈 SHARK.AI.CORE</span>",
            unsafe_allow_html=True)
        with st.spinner("Compiling response..."):
            try:
                response = agent.invoke({"messages": st.session_state.messages})
                reply_text = response["messages"][-1].content
                st.markdown(reply_text)
                st.session_state.messages.append(AIMessage(content=reply_text))
            except Exception as e:
                st.error(f"[FATAL EXCEPTION]: {e}")