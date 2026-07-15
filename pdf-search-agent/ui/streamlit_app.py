import html
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure the project root (parent of ui/) is on the import path,
# so "from app...." imports work regardless of where Streamlit is started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

import streamlit as st

from app.pipeline import answerQuestion, indexer, processPdf


# Temporary PDFs uploaded through the Streamlit interface.
# This directory is separate from permanent sample and benchmark PDFs.
UPLOAD_DIR = PROJECT_ROOT / "data" / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# Basic page setup
st.set_page_config(
    page_title="CERTUS",
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
    background: #36431D;
    border-right: 1px solid #4a5a28;
}
/* Sidebar success message */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(227, 239, 182, 0.16) !important;
    border: 1px solid #a9c45f !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #e3efb6 !important;
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

.status-completed {
    background: #d8edd7;
    color: #1f5a2f !important;
}

.status-indexing {
    background: #fff2cc;
    color: #7a5700 !important;
}

.status-error {
    background: #f8d7da;
    color: #721c24 !important;
}
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
[data-testid="stImage"] {
    margin-bottom: 0 !important;
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

WELCOME_TEXT = "Welcome! Upload a PDF to get started. I'll answer only from your document, with citations and I'll say so if I can't find enough evidence."

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

if "reset_message" not in st.session_state:
    st.session_state.reset_message = None

if "reset_errors" not in st.session_state:
    st.session_state.reset_errors = []

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


def has_indexed_pdf() -> bool:
    return any(
        doc.get("status") == "completed"
        for doc in st.session_state.uploaded_docs.values()
    )


def format_backend_answer(result: dict) -> dict:
    abstention_reason = result.get("abstention_reason")

    infra_message = None
    if abstention_reason == "generation_error":
        infra_message = (
            "Answer generation is unavailable in this cloud demo because it "
            "requires a local Ollama server, which cannot run on Streamlit "
            "Cloud. Retrieval and citation matching work normally — please "
            "run the app locally (see the README) for full answer generation."
        )

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
        "infra_message": infra_message,
        "error": None,
    }
    
def delete_file_safely(file_path: Path) -> str | None:
    """
    Deletes a file safely.

    Returns None when deletion succeeds.
    Returns an error message when deletion fails.
    """
    try:
        file_path.unlink(missing_ok=True)
        return None
    except OSError as error:
        return f"Could not delete {file_path.name}: {error}"
    
def process_pdf(uploaded_file):
    file_name = uploaded_file.name

    if not file_name.lower().endswith(".pdf"):
        return False, f"{file_name} is not a PDF file. Please upload a valid PDF."

    raw_file = uploaded_file.read()

    if len(raw_file) < 10:
        return False, f"{file_name} seems to be empty or corrupted."

    if not raw_file.startswith(b"%PDF"):
        return False, f"{file_name} is not a valid PDF file."

    # Store Streamlit uploads only in the temporary user-upload directory.
    safe_path = UPLOAD_DIR / file_name

    try:
        safe_path.write_bytes(raw_file)
    except OSError as error:
        return False, f"Could not save {file_name}: {error}"

    st.session_state.uploaded_docs[file_name] = {
        "status": "indexing",
        "size_kb": round(len(raw_file) / 1024, 1),
        "upload_time": datetime.now().strftime("%H:%M:%S"),
    }

    st.session_state.uploaded_file_paths[file_name] = str(safe_path)

    result = processPdf(str(safe_path))

    if result.get("success"):
        st.session_state.uploaded_docs[file_name]["status"] = "completed"
        st.session_state.uploaded_docs[file_name]["num_chunks"] = result.get(
            "num_chunks",
            0,
        )
        return True, None

    # The PDF could not be indexed.
    st.session_state.uploaded_docs[file_name]["status"] = "error"
    st.session_state.uploaded_docs[file_name]["num_chunks"] = 0
    st.session_state.uploaded_file_paths.pop(file_name, None)

    deletion_error = delete_file_safely(safe_path)

    error_message = result.get(
        "message",
        f"An unknown error occurred while processing {file_name}.",
    )

    if deletion_error:
        error_message = f"{error_message} Cleanup warning: {deletion_error}"

    return False, error_message



def reset_app() -> dict:
    """
    Resets the live application.

    This clears:
    - the live ChromaDB collection,
    - temporary files from data/user_uploads,
    - uploaded-document state,
    - chat history,
    - the evaluation log.

    It does not affect benchmark PDFs or data/eval_chroma_db.
    """
    errors = []
    deleted_files = []

    try:
        indexer.clear()
    except Exception as error:
        errors.append(f"Could not clear the vector database: {error}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in UPLOAD_DIR.iterdir():
        if not file_path.is_file():
            continue

        deletion_error = delete_file_safely(file_path)

        if deletion_error:
            errors.append(deletion_error)
        else:
            deleted_files.append(file_path.name)

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

    # Create a new empty uploader after reset.
    st.session_state.uploader_version += 1

    return {
        "success": not errors,
        "deleted_files": deleted_files,
        "errors": errors,
    }
# Sidebar
with st.sidebar:
    st.image(str(ASSETS_DIR / "logo.png"), use_container_width=True)
    st.markdown(
        """
        <div style="
            color:#e3efb6;
            font-size:0.92rem;
            line-height:1.5;
            margin-top:-0.5rem;
            margin-bottom:1rem;
            text-align:center;
        ">
            Grounded answers from your documents
        </div>
        """,
        unsafe_allow_html=True,
    )


    if st.session_state.reset_message:
        st.success(st.session_state.reset_message)
        st.session_state.reset_message = None

    if st.session_state.reset_errors:
        for reset_error in st.session_state.reset_errors:
            st.error(reset_error)
        st.session_state.reset_errors = []

    st.markdown("### Uploaded PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_version}",
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

    if st.button("Start New Session", use_container_width=True):
        reset_result = reset_app()

        if reset_result["success"]:
            number_deleted = len(reset_result["deleted_files"])

            st.session_state.reset_message = (
                "Started a new session successfully."
                f"Deleted temporary files: {number_deleted}."
            )
        else:
            st.session_state.reset_message = (
                "The application state was reset, "
                "but some cleanup operations failed."
            )
            st.session_state.reset_errors = reset_result["errors"]

        st.rerun()

    st.markdown(
        """
<div style="
    position: fixed;
    bottom: 18px;
    left: 0;
    width: 29rem;
    text-align: center;
    color: #e3efb6;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
">
    SEP 2026 · TU Braunschweig
</div>
""",
        unsafe_allow_html=True,
    )

# Main area
tab_search, tab_admin, tab_help = st.tabs(
    [
        "🔍 Ask Questions",
        "📊 Activity & Metrics",
        "❓ Help & About",
    ]
)
with tab_search:
    st.image(str(ASSETS_DIR / "logo1.png"), width=200)
    st.markdown(
        """
    <div class="page-subtitle">Reliable document question answering with grounded citations.</div>
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
            elif msg.get("infra_message"):
                st.markdown(
                     f'<div class="chat-agent"><span class="avatar">🤖</span><div class="bubble"><div class="abstention-box">ℹ️ {msg["infra_message"]}</div></div></div>',
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
<div class="page-title">Activity & Metrics</div>
<div class="page-subtitle">
    Monitor uploaded documents, questions, abstentions, response times,
    and application activity.
</div>
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
    st.subheader("System information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-label">Embedding Model</div>
    <div style="margin-top:0.7rem;font-weight:700;color:#27310f;">
        all-MiniLM-L6-v2
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with info_col2:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-label">Language Model</div>
    <div style="margin-top:0.7rem;font-weight:700;color:#27310f;">
        Llama 3.2 via Ollama
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with info_col3:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-label">Vector Database</div>
    <div style="margin-top:0.7rem;font-weight:700;color:#27310f;">
        ChromaDB
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
# ==========================================================
# Help and about tab
# ==========================================================

with tab_help:
    st.markdown(
        """
<div class="page-title">Help & About</div>
<div class="page-subtitle">
    Learn how the PDF Search Agent works and how to interpret its answers.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("How does the PDF Search Agent work?", expanded=True):
        st.markdown(
            """
The application follows a Retrieval-Augmented Generation pipeline:

1. You upload one or more PDF documents.
2. The documents are parsed and divided into smaller text chunks.
3. Each chunk is converted into an embedding and stored in ChromaDB.
4. When you ask a question, the most relevant chunks are retrieved.
5. Llama 3.2 generates an answer using only the retrieved document context.
6. The answer includes citations showing the document and page used.
"""
        )

    with st.expander("Why did the system say it could not answer?"):
        st.markdown(
            """
The system abstains when it cannot find enough relevant evidence in the
uploaded documents.

This is intentional. Instead of guessing or using unsupported external
knowledge, the agent returns an abstention message. You can try:

- asking the question in a different way,
- uploading a more relevant document,
- or checking whether the requested information is actually present.
"""
        )

    with st.expander("How should I interpret citations?"):
        st.markdown(
            """
Citations identify the document and page used to support the generated answer.

The system validates generated citations against the chunks that were
actually retrieved. However, local language models may occasionally produce
an incomplete or incorrectly formatted citation, especially for long or
complex answers.

For important information, users should still review the cited page in the
original PDF.
"""
        )

    with st.expander("What happens to uploaded PDF files?"):
        st.markdown(
            """
Files uploaded through the interface are stored temporarily inside:

    data/user_uploads/

The **Start New Session** button:

- removes uploaded files,
- clears the live ChromaDB collection,
- clears the conversation,
- clears the activity log,
- and resets the file uploader.

Benchmark PDFs and the separate evaluation database are not affected.
"""
        )

    with st.expander("What technology powers the application?"):
        st.markdown(
            """
- **PDF parser:** PyMuPDF
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector database:** ChromaDB
- **Language model:** Llama 3.2
- **Model runtime:** Ollama
- **User interface:** Streamlit
"""
        )

    st.markdown("---")

    st.markdown(
     """
        <div class="prototype-note">
        This application was developed as part of a university Software
        Engineering Project at TU Braunschweig. It focuses on reliable, grounded
        question answering over PDF documents — combining semantic retrieval,
        citation-backed answers, abstention when evidence is insufficient, and
        measurable evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )