import streamlit as st
import os
import sys
import hashlib
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Inject Streamlit Cloud secrets into env (no-op during local dev)
try:
    for key in ["GOOGLE_API_KEY", "GROQ_API_KEY", "SARVAM_API_KEY", "WHISPER_MODEL"]:
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])
except Exception:
    pass

from utils.audio_processor import process_input
from core.transcriber import transcribe_auto
from core.summarise import summarise, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.engine import build_rag_chain, load_rag_chain, ask_question

# ── Constants ──────────────────────────────────────────────────────────────────
SESSIONS_FILE = "chroma-db/sessions.json"

def _session_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:8]

def _load_sessions() -> dict:
    if not Path(SESSIONS_FILE).exists():
        return {}
    with open(SESSIONS_FILE) as f:
        return json.load(f)

def _save_session(sid: str, source: str, title: str):
    Path("chroma-db").mkdir(exist_ok=True)
    sessions = _load_sessions()
    sessions[sid] = {"source": source, "title": title}
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    .sidebar-logo {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    /* Title card */
    .title-card {
        background: linear-gradient(135deg, #1e1b4b, #2e1065);
        border: 1px solid #4338ca;
        border-left: 4px solid #818cf8;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 24px;
    }
    .title-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #818cf8;
        margin-bottom: 6px;
    }
    .title-text {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    /* Feature cards on welcome screen */
    .feature-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 24px 16px;
        text-align: center;
        height: 100%;
        transition: border-color 0.2s;
    }
    .feature-icon { font-size: 2rem; margin-bottom: 10px; }
    .feature-title { font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
    .feature-desc { color: #8b949e; font-size: 0.85rem; line-height: 1.4; }
    /* Hero */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        text-align: center;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 48px;
    }
    /* Session buttons */
    .session-source {
        font-size: 0.72rem;
        color: #6e7681;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Tab styling override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        color: #8b949e;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom: 2px solid #818cf8 !important;
        background: transparent !important;
    }
    /* Chat */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────────
for key, default in [
    ("result", None),
    ("rag_chain", None),
    ("messages", []),
    ("active_session_title", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎬 AI Video<br>Assistant</div>', unsafe_allow_html=True)
    st.caption("Transcribe · Summarise · Chat")
    st.divider()

    mode = st.radio(
        "mode",
        ["▶ New Video", "💬 Resume Session"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── New Video Mode ─────────────────────────────────────────────────────────
    if "New" in mode:
        source_type = st.radio("Input", ["YouTube URL", "Local File"], horizontal=True)

        source = None
        if source_type == "YouTube URL":
            source = st.text_input(
                "url", placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
            cookies_file = None
            with st.expander("🍪 YouTube cookies (optional)"):
                st.caption("Use this only if YouTube blocks the link on Streamlit Cloud. Export cookies.txt from your browser using the [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) extension.")
                cookies_file = st.file_uploader("cookies.txt", type=["txt"], label_visibility="collapsed")
                if cookies_file:
                    st.success("Cookies loaded for this video only ✓")
        else:
            cookies_file = None
            uploaded = st.file_uploader(
                "Upload", type=["mp4", "mp3", "wav", "m4a", "webm"],
                label_visibility="collapsed",
            )
            if uploaded:
                suffix = Path(uploaded.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.close()
                source = tmp.name

        process_btn = st.button(
            "⚡ Process Video", type="primary",
            use_container_width=True, disabled=not source,
        )

        if process_btn and source:
            st.session_state.messages = []
            sid = _session_id(source)
            cookie_path = None
            try:
                if cookies_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_cookies:
                        tmp_cookies.write(cookies_file.getvalue())
                        cookie_path = tmp_cookies.name

                with st.status("Processing video…", expanded=True) as status:
                    st.write("🎵 Downloading & chunking audio…")
                    chunks = process_input(source, cookie_path=cookie_path)

                    st.write("🗣️ Transcribing audio…")
                    transcript = transcribe_auto(chunks)

                    st.write("📝 Generating summary…")
                    summary = summarise(transcript)
                    title = generate_title(summary)

                    st.write("🔍 Extracting insights…")
                    action_items = extract_action_items(transcript)
                    key_decisions = extract_key_decisions(transcript)
                    questions = extract_questions(transcript)

                    st.write("🧠 Building Q&A index…")
                    rag_chain = build_rag_chain(transcript, sid)
                    _save_session(sid, source, title)

                    status.update(label="✅ Done!", state="complete", expanded=False)

                st.session_state.result = {
                    "title": title,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": key_decisions,
                    "open_questions": questions,
                    "transcript": transcript,
                }
                st.session_state.rag_chain = rag_chain
                st.session_state.active_session_title = title
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if cookie_path and Path(cookie_path).exists():
                    Path(cookie_path).unlink()

    # ── Resume Session Mode ────────────────────────────────────────────────────
    else:
        sessions = _load_sessions()
        if not sessions:
            st.info("No past sessions yet.\nProcess a video first.")
        else:
            st.markdown("**Past Sessions**")
            for sid, info in reversed(list(sessions.items())):
                label = info["title"][:28] + "…" if len(info["title"]) > 28 else info["title"]
                if st.button(f"📹 {label}", key=f"sess_{sid}", use_container_width=True):
                    with st.spinner("Loading session…"):
                        st.session_state.rag_chain = load_rag_chain(sid)
                    st.session_state.result = {
                        "title": info["title"],
                        "summary": "", "action_items": "",
                        "key_decisions": "", "open_questions": "",
                        "transcript": "",
                    }
                    st.session_state.messages = []
                    st.session_state.active_session_title = info["title"]
                    st.rerun()

# ── Main Content ───────────────────────────────────────────────────────────────
if st.session_state.result is None:
    # Welcome / hero screen
    st.markdown('<div class="hero-title">🎬 AI Video Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Transcribe, summarise, and chat with any YouTube video or local file.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🎵", "Transcribe", "Auto-detects language.\nSupports Hindi & more via Sarvam AI."),
        ("📋", "Summarise", "Get a concise bullet-point\nsummary instantly."),
        ("🔍", "Extract", "Action items, key decisions\n& open questions."),
        ("💬", "Chat", "Ask anything about the\nvideo via RAG Q&A."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈  Paste a YouTube URL or upload a file in the sidebar to get started.", icon="💡")

else:
    result = st.session_state.result

    # Title card
    st.markdown(f"""
    <div class="title-card">
        <div class="title-label">🎬 Video</div>
        <div class="title-text">{result["title"]}</div>
    </div>""", unsafe_allow_html=True)

    # Results tabs (only if freshly processed, not resumed with empty data)
    if result["summary"]:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary", "✅ Action Items", "🔑 Key Decisions",
            "❓ Open Questions", "📄 Transcript",
        ])
        with tab1:
            st.markdown(result["summary"])
        with tab2:
            st.markdown(result["action_items"])
        with tab3:
            st.markdown(result["key_decisions"])
        with tab4:
            st.markdown(result["open_questions"])
        with tab5:
            st.text_area(
                "transcript", result["transcript"], height=320,
                label_visibility="collapsed",
            )
        st.divider()

    # Chat
    st.subheader("💬 Chat with this video")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about the video…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                with st.spinner(""):
                    answer = ask_question(st.session_state.rag_chain, question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.warning("Process a video first to enable chat.")
