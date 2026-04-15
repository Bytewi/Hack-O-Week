"""
streamlit_app.py — SIT Nagpur Student Support Chatbot
Run: streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from faq_engine      import FAQEngine
from context_manager import ConversationContext
from chatbot_core    import get_response
from analytics       import analyze_logs, get_log_count, export_csv

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIT Nagpur — Student Support",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
.stApp { background: #0f1117; }
* { font-family: 'Segoe UI', sans-serif; }

/* Header */
.sit-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid #3949ab;
    box-shadow: 0 4px 24px rgba(26,35,126,0.4);
}
.sit-header h1 {
    color: #fff;
    margin: 0 0 4px;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: .02em;
}
.sit-header p {
    color: #c5cae9;
    margin: 0;
    font-size: 0.86rem;
}

/* Chat bubbles */
.bubble-user {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: #fff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 76%;
    margin: 8px 0 4px auto;
    width: fit-content;
    font-size: 0.92rem;
    line-height: 1.55;
    white-space: pre-wrap;
    box-shadow: 0 2px 10px rgba(13,71,161,.35);
}
.bubble-bot {
    background: linear-gradient(135deg, #1c1f2e, #242838);
    border: 1px solid #2d3250;
    color: #e8eaf6;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    margin: 4px auto 8px 0;
    width: fit-content;
    font-size: 0.92rem;
    line-height: 1.65;
    white-space: pre-wrap;
    box-shadow: 0 2px 10px rgba(0,0,0,.35);
}

/* Welcome box */
.welcome-box {
    background: #1c1f2e;
    border: 1px dashed #2d3250;
    border-radius: 14px;
    padding: 20px 24px;
    color: #9fa8da;
    font-size: 0.9rem;
    line-height: 1.8;
    margin: 8px 0 18px;
}
.welcome-box strong { color: #7986cb; }

/* Input */
.stTextInput input {
    background: #1c1f2e !important;
    border: 1px solid #2d3250 !important;
    border-radius: 12px !important;
    color: #e8eaf6 !important;
    padding: 12px 16px !important;
    font-size: 0.93rem !important;
}
.stTextInput input:focus {
    border-color: #5c6bc0 !important;
    box-shadow: 0 0 0 3px rgba(92,107,192,.18) !important;
}

/* Buttons */
.stButton > button {
    background: #1c1f2e !important;
    border: 1px solid #2d3250 !important;
    color: #c5cae9 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    transition: all .2s;
}
.stButton > button:hover {
    background: #2d3250 !important;
    border-color: #5c6bc0 !important;
    color: #fff !important;
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: #0f1117 !important;
}
.sidebar-card {
    background: #1c1f2e;
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}
.sidebar-card h4 { color: #7986cb; margin: 0 0 10px; font-size: .87rem; }

/* Analytics */
.stCode, pre { font-size: .77rem !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Cached init ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return FAQEngine()

engine = load_engine()

# ── Session State ──────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "context"     not in st.session_state: st.session_state.context     = ConversationContext()
if "show_analytics" not in st.session_state: st.session_state.show_analytics = False
if "pending_q"   not in st.session_state: st.session_state.pending_q  = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <h4>🎓 SIT Nagpur Chatbot</h4>
        <span style="color:#9fa8da;font-size:.8rem;line-height:1.6">
            Ask anything about Symbiosis Institute<br>
            of Technology, Nagpur — fees, admissions,<br>
            hostel, exams, placements & more.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card"><h4>⚡ Quick Questions</h4>', unsafe_allow_html=True)

    QUICK_QS = [
        ("💰", "What are the fees at SIT Nagpur?"),
        ("📋", "What is the admission process?"),
        ("🎓", "What courses are offered?"),
        ("🏠", "What are the hostel facilities?"),
        ("💼", "Placement statistics?"),
        ("🏆", "Scholarships available?"),
        ("📚", "Tell me about the library"),
        ("⏰", "What are the college timings?"),
        ("📶", "Is there WiFi on campus?"),
        ("🏛️", "About SIT Nagpur"),
        ("⚽", "Sports and gym facilities?"),
        ("📝", "What is the exam schedule?"),
    ]
    for icon, q in QUICK_QS:
        short_label = q[:35] + ("…" if len(q) > 35 else "")
        if st.button(f"{icon} {short_label}", key=f"q_{q[:20]}", use_container_width=True):
            st.session_state.pending_q = q

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.context.reset()
            st.session_state.show_analytics = False
            st.rerun()

    if st.button("📥 Export CSV", use_container_width=True):
        msg = export_csv()
        st.success(msg)

    st.markdown(
        f'<div style="color:#4a5568;font-size:.74rem;text-align:center;margin-top:10px">'
        f'💬 {get_log_count()} queries logged</div>',
        unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sit-header">
    <h1>🎓 SIT Nagpur — Student Support</h1>
    <p>Symbiosis Institute of Technology, Nagpur &nbsp;·&nbsp; sitnagpur.edu.in</p>
    <p style="margin-top:6px;font-size:.81rem;color:#9fa8da">
        Ask about <strong>Fees · Admissions · Courses · Hostel · Exams · Placements</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Analytics Panel ────────────────────────────────────────────────────────────
if st.session_state.show_analytics:
    with st.expander("📊 Analytics Dashboard", expanded=True):
        st.code(analyze_logs(), language=None)

# ── Chat History ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        👋 <strong>Hello! I'm your SIT Nagpur Student Support Assistant.</strong><br><br>
        I can answer questions about:<br>
        &nbsp;&nbsp;💰 <strong>Fees & Payments</strong> &nbsp;·&nbsp;
        📋 <strong>Admissions</strong> &nbsp;·&nbsp;
        🎓 <strong>Courses & Specializations</strong><br>
        &nbsp;&nbsp;🏠 <strong>Hostel</strong> &nbsp;·&nbsp;
        💼 <strong>Placements</strong> &nbsp;·&nbsp;
        🏆 <strong>Scholarships</strong> &nbsp;·&nbsp;
        📚 <strong>Library</strong><br><br>
        Click a quick question on the left, or type your question below! ✨
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        text = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="bubble-user">👤&nbsp; {text}</div>', unsafe_allow_html=True)
    else:
        text = (msg["content"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>"))
        st.markdown(f'<div class="bubble-bot">🤖&nbsp; {text}</div>', unsafe_allow_html=True)

# ── Input Row ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c1:
    user_text = st.text_input(
        "msg", label_visibility="collapsed",
        placeholder="Ask about SIT Nagpur…  e.g. 'What are the BTech fees?'",
        key="chat_input"
    )
with c2:
    send = st.button("Send ➤", use_container_width=True)

# Handle quick question
if st.session_state.pending_q:
    user_text = st.session_state.pending_q
    send = True
    st.session_state.pending_q = None

# ── Process ────────────────────────────────────────────────────────────────────
if send and user_text.strip():
    query = user_text.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    result = get_response(query, engine, st.session_state.context)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
    })
    st.rerun()
