import sys
from pathlib import Path

# Make sure the project root (parent of ui/) is on the import path,
# so "from app...." imports work no matter how/where streamlit is launched from.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import time
from datetime import datetime
from pathlib import Path

from app.pipeline import processPdf, answerQuestion

# Basic page setup
st.set_page_config(
    page_title="PDF Search Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&family=Nunito+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #56682f;
    border-right: 1px solid #6a7d3c;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown * {
    color: #f4f7ea !important;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e3efb6 !important;
    font-size: 1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 800;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    color: #e3efb6 !important;
    letter-spacing: 0.03em !important;
    text-transform: none !important;
    line-height: 1.3 !important;
}
.main .block-container {
    padding-top: 1.7rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}

/* Chat bubbles */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.9rem 0;
}

.chat-user .bubble {
    background: #56682f;
    color: #ffffff;
    padding: 0.95rem 1.2rem;
    border-radius: 20px 20px 6px 20px;
    max-width: 72%;
    font-size: 1.05rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.chat-agent {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    margin: 0.4rem 0 0.8rem 0;
}

.chat-agent .bubble {
    background: #f7faef;
    color: #27310f;
    padding: 1.1rem 1.1rem;
    border-radius: 18px 18px 18px 6px;
    max-width: 62%;
    font-size: 1.08rem;
    line-height: 1.65;
    border: 1px solid #c9d99a;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: #56682f;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-right: 12px;
    flex-shrink: 0;
}

/* Make first message bigger */
.welcome-bubble {
    max-width: 72% !important;
    font-size: 1.15rem !important;
    padding: 1.2rem 1.4rem !important;
    text-align: center !important;
    line-height: 1.7 !important;
}

.citation-badge {
    display: inline-block;
    background: #eef5d9;
    border: 1px solid #b8cf71;
    color: #46551d;
    border-radius: 8px;
    padding: 5px 11px;
    font-size: 0.88rem;
    font-family: 'Nunito Sans', sans-serif;
    margin-top: 8px;
    margin-right: 6px;
    font-weight: 700;
}

.abstention-box {
    background: #fff8e8;
    border-left: 4px solid #e7a400;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1rem;
    color: #5a4000;
    font-size: 1.02rem;
    margin-top: 4px;
}

.error-box {
    background: #fef1f1;
    border-left: 4px solid #c0392b;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1rem;
    color: #6b0e0e;
    font-size: 1.02rem;
    margin-top: 4px;
}

/* Sidebar file status */
.status-pill {
    display: inline-block;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.status-completed { background: #d8edd7; color: #1f5a2f; }
.status-indexing  { background: #fff2cc; color: #7a5700; }
.status-error     { background: #f8d7da; color: #721c24; }

.file-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #708443;
    font-size: 0.96rem;
}

.file-name {
    color: #edf7c8;
    font-weight: 700;
}

/* Metrics */
.metric-card {
    background: white;
    border: 1px solid #d2dfaa;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #46551d;
}

.metric-card .metric-label {
    font-size: 0.86rem;
    color: #6f8340;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

/* Titles */
.page-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #27310f;
    margin-bottom: 0.2rem;
}

.page-subtitle {
    font-size: 1.1rem;
    color: #66793a;
    margin-top: 2px;
    margin-bottom: 0.8rem;
}

/* White area around input + button */
div[data-testid="stForm"] {
    border: none !important;
    border-radius: 22px !important;
    padding: 0.8rem !important;
    background: #ffffff !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}

/* Input field: no border */
.stTextInput input {
    font-size: 1.05rem !important;
    padding: 10px 1.2rem !important;
    height: 2.5rem !important;
    min-height: 2.5rem !important;
    border-radius: 16px !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-family: 'Nunito', sans-serif !important;
    background: #f7f9f2 !important;
    text-align: center !important;
}

.stTextInput input:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Placeholder */
.stTextInput input::placeholder {
    font-size: 1rem !important;
    color: #647244 !important;
    text-align: center !important;
}

/* Send button */
.stButton button,
.stFormSubmitButton button {
    border-radius: 16px !important;
    background: #56682f !important;
    color: white !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    height: 4.6rem !important;
    font-family: 'Nunito', sans-serif !important;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #64783a !important;
    color: white !important;
}

section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid #738747 !important;
    border-radius: 14px !important;
    padding: 0.6rem !important;
}

/* File uploader text */
section[data-testid="stFileUploader"],
section[data-testid="stFileUploader"] * {
    color: #1f280c !important;
}

section[data-testid="stFileUploader"] button {
    color: #1f280c !important;
    border: 1px solid #aeb79a !important;
    background: #f5f7ef !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

section[data-testid="stFileUploader"] button * {
    color: #1f280c !important;
    fill: #1f280c !important;
    stroke: #1f280c !important;
    opacity: 1 !important;
}

section[data-testid="stFileUploader"] small,
section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] p,
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] div {
    color: #1f280c !important;
    opacity: 1 !important;
}

section[data-testid="stFileUploader"] [disabled],
section[data-testid="stFileUploader"] .disabled,
section[data-testid="stFileUploader"] button:disabled {
    color: #1f280c !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #1f280c !important;
}

/* Prototype note */
.prototype-note {
    font-size: 1rem;
    color: #55672d;
    line-height: 1.7;
}
</style>
""",
    unsafe_allow_html=True,
)

WELCOME_TEXT = "Hey, first upload your PDF. Then ask a question about it, and I’ll answer based on the document."

# Session variables
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "agent",
            "text": WELCOME_TEXT,
            "citations": [],
            "abstain": False,
            "error": None,
        }
    ]

if "eval_runs" not in st.session_state:
    st.session_state.eval_runs = []


def has_indexed_pdf() -> bool:
    return any(
        doc.get("status") == "completed"
        for doc in st.session_state.uploaded_docs.values()
    )


def format_backend_answer(result: dict) -> dict:
    return {
        "text": result.get("answer", ""),
        "citations": [
            {
                "doc": citation.get("document", "Unknown"),
                "page": citation.get("page_number", "?"),
            }
            for citation in result.get("citations", [])
        ],
        "abstain": result.get("abstained", False),
        "error": None,
    }


def process_pdf(uploaded_file):
    file_name = uploaded_file.name

    if not file_name.lower().endswith(".pdf"):
        return False, f"{file_name} is not a PDF file. Please upload a valid PDF."

    raw_file = uploaded_file.read()

    if len(raw_file) < 10:
        return False, f"{file_name} seems to be empty or corrupted."

    if not raw_file.startswith(b"%PDF"):
        return False, f"{file_name} is not a valid PDF file."

    upload_dir = Path("data/uploaded_pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_path = upload_dir / file_name

    with open(safe_path, "wb") as f:
        f.write(raw_file)

    st.session_state.uploaded_docs[file_name] = {
        "status": "indexing",
        "size_kb": round(len(raw_file) / 1024, 1),
        "upload_time": datetime.now().strftime("%H:%M:%S"),
    }

    st.session_state.uploaded_file_paths[file_name] = str(safe_path)

    result = processPdf(str(safe_path))

    if result["success"]:
        st.session_state.uploaded_docs[file_name]["status"] = "completed"
        st.session_state.uploaded_docs[file_name]["num_chunks"] = result.get("num_chunks", 0)
        return True, None
    else:
        st.session_state.uploaded_docs[file_name]["status"] = "error"
        return False, result["message"]


def reset_app():
    st.session_state.uploaded_docs = {}
    st.session_state.uploaded_file_paths = {}
    st.session_state.chat_history = [
        {
            "role": "agent",
            "text": WELCOME_TEXT,
            "citations": [],
            "abstain": False,
            "error": None,
        }
    ]
    st.session_state.eval_runs = []


# Sidebar
with st.sidebar:
    st.markdown("##  🤖 PDF Search Agent")
    st.markdown("---")
    st.markdown("### Uploaded PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    ok, err = process_pdf(uploaded_file)
                if not ok:
                    st.session_state.chat_history.append(
                        {
                            "role": "agent",
                            "text": "",
                            "citations": [],
                            "abstain": False,
                            "error": err,
                        }
                    )

    if st.session_state.uploaded_docs:
        for file_name, meta in st.session_state.uploaded_docs.items():
            status = meta.get("status", "error")
            status_class = f"status-{status}"
            status_label = {
                "completed": "✓ Indexed",
                "indexing": "Indexing...",
                "error": "Error",
            }.get(status, "Unknown")

            st.markdown(
                f"""
<div class="file-row">
    <span class="file-name">📄 {file_name}</span>
    <span class="status-pill {status_class}">{status_label}</span>
</div>
<div style="font-size:0.82rem;color:#edf7c8;padding:2px 0 10px 0;">
    {meta.get('size_kb', '-')} KB · uploaded at {meta.get('upload_time', '-')} · chunks: {meta.get('num_chunks', '-')}
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#edf7c8;font-size:0.95rem;padding:8px 0;">No PDF uploaded yet.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("Reset prototype", use_container_width=True):
        reset_app()
        st.rerun()

# Main area
tab_search, tab_admin = st.tabs(["🔍 Search", "⚙️ Administration & Evaluation"])

with tab_search:
    st.markdown(
        """
<div class="page-title">Document Q&A</div>
<div class="page-subtitle">Upload a PDF, ask a question, and get an answer grounded in the document.</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 13rem;'></div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><div class="bubble">{msg["text"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            if msg.get("error"):
                st.markdown(
                    f'<div class="chat-agent"><span class="avatar">🤖</span><div class="bubble"><div class="error-box">⚠️ {msg["error"]}</div></div></div>',
                    unsafe_allow_html=True,
                )
            elif msg.get("abstain"):
                st.markdown(
                    '<div class="chat-agent"><span class="avatar">🤖</span><div class="bubble"><div class="abstention-box">I could not find enough evidence in the uploaded PDF to answer this question. Please try another question or upload a more relevant document.</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                citations_html = ""
                for citation in msg.get("citations", []):
                    doc_label = citation.get("doc", "document")
                    page = citation.get("page", "-")
                    citations_html += f'<span class="citation-badge">📄 {doc_label}, page {page}</span>'

                answer_html = msg.get("text", "")
                if citations_html:
                    answer_html += "<br><br>" + citations_html

                extra_class = " welcome-bubble" if msg.get("text") == WELCOME_TEXT else ""

                st.markdown(
                    f'<div class="chat-agent"><span class="avatar">🤖</span><div class="bubble{extra_class}">{answer_html}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.form("question_form", clear_on_submit=True):
        cols = st.columns([8, 1.2])

        with cols[0]:
            question = st.text_input(
                "question",
                placeholder="Ask a question about the uploaded PDF…",
                label_visibility="collapsed",
            )

        with cols[1]:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted:
        q = question.strip()

        if not q:
            st.session_state.chat_history.append(
                {
                    "role": "agent",
                    "text": "",
                    "citations": [],
                    "abstain": False,
                    "error": "Please enter a question before submitting.",
                }
            )
            st.session_state.eval_runs.append(
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "query": "",
                    "answer": "",
                    "abstained": False,
                    "response_time_s": 0,
                    "num_citations": 0,
                    "error": "Empty question",
                }
            )
            st.rerun()

        if not has_indexed_pdf():
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "text": q,
                    "citations": [],
                    "abstain": False,
                    "error": None,
                }
            )
            st.session_state.chat_history.append(
                {
                    "role": "agent",
                    "text": "",
                    "citations": [],
                    "abstain": False,
                    "error": "Please upload a PDF first before asking a question.",
                }
            )
            st.session_state.eval_runs.append(
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "query": q,
                    "answer": "",
                    "abstained": False,
                    "response_time_s": 0,
                    "num_citations": 0,
                    "error": "No PDF uploaded",
                }
            )
            st.rerun()

        st.session_state.chat_history.append(
            {
                "role": "user",
                "text": q,
                "citations": [],
                "abstain": False,
                "error": None,
            }
        )

        with st.spinner("Searching in the uploaded PDF..."):
            start_time = time.time()
            backend_result = answerQuestion(q, top_k=5)
            response_time = round(time.time() - start_time, 2)

        answer = format_backend_answer(backend_result)
        answer["role"] = "agent"
        answer["response_time"] = response_time
        st.session_state.chat_history.append(answer)

        st.session_state.eval_runs.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": q,
                "answer": backend_result.get("answer", ""),
                "citations": backend_result.get("citations", []),
                "abstained": backend_result.get("abstained", False),
                "response_time_s": response_time,
                "num_citations": len(backend_result.get("citations", [])),
                "error": None,
            }
        )

        st.rerun()

with tab_admin:
    st.markdown(
        """
<div class="page-title">Administration & Evaluation Dashboard</div>
<div class="page-subtitle">Monitor uploaded documents, questions, abstentions, and evaluation data.</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    total_docs = len(st.session_state.uploaded_docs)
    indexed_docs = sum(
        1 for doc in st.session_state.uploaded_docs.values()
        if doc.get("status") == "completed"
    )
    total_queries = sum(
        1 for msg in st.session_state.chat_history
        if msg.get("role") == "user"
    )
    abstentions = sum(
        1 for run in st.session_state.eval_runs
        if run.get("abstained")
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_docs}</div><div class="metric-label">Uploaded PDFs</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{indexed_docs}</div><div class="metric-label">Indexed PDFs</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_queries}</div><div class="metric-label">Questions</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{abstentions}</div><div class="metric-label">Abstentions</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Evaluation log")

    if st.session_state.eval_runs:
        st.dataframe(
            st.session_state.eval_runs,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No evaluation data yet. Ask a question to create the first log entry.")

    st.markdown("---")
    st.subheader("Prototype note")
    st.markdown(
        '<div class="prototype-note">This version is now connected to the real backend pipeline for PDF processing, indexing, retrieval, answer generation, citations, and abstention handling. The next step is to improve logging, evaluation runs, and configuration-based execution.</div>',
        unsafe_allow_html=True,
    )