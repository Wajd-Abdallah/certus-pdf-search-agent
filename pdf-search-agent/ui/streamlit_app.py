import streamlit as st
import time
from datetime import datetime
# Basic page setup
st.set_page_config(
    page_title="PDF Search Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Small custom design for the prototype
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1b2d;
    border-right: 1px solid #1e3354;
}

[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7fb3d3 !important;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Main layout */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Chat messages */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.7rem 0;
}

.chat-user .bubble {
    background: #163b63;
    color: #ffffff;
    padding: 0.7rem 1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.5;
}

.chat-agent {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    margin: 0.7rem 0;
}

.chat-agent .bubble {
    background: #f3f8fd;
    color: #102033;
    padding: 0.9rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #d5e3ef;
}

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #163b63;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    margin-right: 10px;
    flex-shrink: 0;
}

.citation-badge {
    display: inline-block;
    background: #e7f0fa;
    border: 1px solid #aac4de;
    color: #164b76;
    border-radius: 7px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
    margin-right: 6px;
}

.abstention-box {
    background: #fff8e8;
    border-left: 4px solid #f0a500;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #5a4000;
    font-size: 0.92rem;
    margin-top: 4px;
}

.error-box {
    background: #fef0f0;
    border-left: 4px solid #c0392b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #6b0e0e;
    font-size: 0.92rem;
    margin-top: 4px;
}

.status-pill {
    display: inline-block;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.status-completed {
    background: #d4edda;
    color: #1a5c2e;
}

.status-indexing {
    background: #fff3cd;
    color: #7a5700;
}

.status-error {
    background: #f8d7da;
    color: #721c24;
}

.file-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #e0e8f0;
    font-size: 0.85rem;
}

.file-name {
    color: #2c4e6e;
    font-weight: 500;
}

.metric-card {
    background: white;
    border: 1px solid #d5e3ef;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1a3a5c;
}

.metric-card .metric-label {
    font-size: 0.78rem;
    color: #7a8fa0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.page-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f1b2d;
}

.page-subtitle {
    font-size: 0.88rem;
    color: #6a8399;
    margin-top: 3px;
}

.stButton button,
.stFormSubmitButton button {
    border-radius: 12px !important;
    background: #163b63 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #1f4f80 !important;
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)
# Session variables
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "agent",
            "text": "Hello! Please upload a PDF first, then you can ask questions about it.",
            "citations": [],
            "abstain": False,
            "error": None,
        }
    ]

if "eval_runs" not in st.session_state:
    st.session_state.eval_runs = []
# Helper functions
def has_indexed_pdf() -> bool:
    return any(
        doc.get("status") == "completed"
        for doc in st.session_state.uploaded_docs.values()
    )


def process_pdf(uploaded_file):
    """
    Simple prototype validation.
    Later this should be replaced by the real parser, chunker and indexer.
    """
    file_name = uploaded_file.name

    if not file_name.lower().endswith(".pdf"):
        return False, f"{file_name} is not a PDF file. Please upload a valid PDF."

    raw_file = uploaded_file.read()

    if len(raw_file) < 10:
        return False, f"{file_name} seems to be empty or corrupted."

    if not raw_file.startswith(b"%PDF"):
        return False, f"{file_name} is not a valid PDF file."

    size_kb = round(len(raw_file) / 1024, 1)

    st.session_state.uploaded_docs[file_name] = {
        "status": "indexing",
        "size_kb": size_kb,
        "upload_time": datetime.now().strftime("%H:%M:%S"),
    }

    # This only simulates indexing for the prototype.
    time.sleep(0.6)

    st.session_state.uploaded_docs[file_name]["status"] = "completed"

    return True, None


def generate_answer(question: str, docs: dict):
    """
    Prototype answer function.
    In the final version this should call the real RAG pipeline.
    """
    q_lower = question.lower().strip()

    if not has_indexed_pdf():
        return {
            "text": "",
            "citations": [],
            "abstain": False,
            "error": "Please upload and index a PDF document first before asking a question.",
        }

    unanswerable_keywords = [
        "weather",
        "stock",
        "president",
        "ceo",
        "recipe",
        "capital of",
        "today",
        "news",
        "sport",
        "film",
    ]

    if any(keyword in q_lower for keyword in unanswerable_keywords):
        return {
            "text": "",
            "citations": [],
            "abstain": True,
            "error": None,
        }

    completed_docs = [
        name for name, meta in docs.items()
        if meta.get("status") == "completed"
    ]

    doc_name = completed_docs[0] if completed_docs else "uploaded document"

    return {
        "text": (
            "Based on the uploaded document, the PDF Search Agent follows a "
            "Retrieval-Augmented Generation approach. In the final system, the PDF content "
            "will be extracted, divided into chunks, indexed, searched, and then used to "
            "generate an answer that is grounded in the document."
        ),
        "citations": [
            {"doc": doc_name, "page": "prototype"},
        ],
        "abstain": False,
        "error": None,
    }


def reset_app():
    st.session_state.uploaded_docs = {}
    st.session_state.chat_history = [
        {
            "role": "agent",
            "text": "Hello! Please upload a PDF first, then you can ask questions about it.",
            "citations": [],
            "abstain": False,
            "error": None,
        }
    ]
    st.session_state.eval_runs = []
# Sidebar
with st.sidebar:
    st.markdown("## 📄 PDF Search Agent")
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
<div style="font-size:0.72rem;color:#8faabb;padding:2px 0 8px 0;">
    {meta.get('size_kb', '-')} KB · uploaded at {meta.get('upload_time', '-')}
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
<div style="color:#6a8399;font-size:0.82rem;padding:8px 0;">
    No PDF uploaded yet.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("Reset prototype", use_container_width=True):
        reset_app()
        st.rerun()
# Main area
tab_search, tab_admin = st.tabs(
    [
        "🔍 Search",
        "⚙️ Administration & Evaluation",
    ]
)
# Search tab
with tab_search:
    st.markdown(
        """
<div class="page-title">Document Q&A</div>
<div class="page-subtitle">
    Upload a PDF and ask a question about its content.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
<div class="chat-user">
    <div class="bubble">{msg["text"]}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        else:
            if msg.get("error"):
                st.markdown(
                    f"""
<div class="chat-agent">
    <span class="avatar">🤖</span>
    <div class="bubble">
        <div class="error-box">⚠️ {msg["error"]}</div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            elif msg.get("abstain"):
                st.markdown(
                    """
<div class="chat-agent">
    <span class="avatar">🤖</span>
    <div class="bubble">
        <div class="abstention-box">
            I could not find enough evidence in the uploaded PDF to answer this question.
            Please try another question or upload a more relevant document.
        </div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            else:
                citations_html = ""

                for citation in msg.get("citations", []):
                    doc_label = citation.get("doc", "document")
                    page = citation.get("page", "-")

                    citations_html += (
                        f'<span class="citation-badge">📄 {doc_label}, page {page}</span>'
                    )

                answer_html = msg.get("text", "")

                if citations_html:
                    answer_html += "<br><br>" + citations_html

                st.markdown(
                    f"""
<div class="chat-agent">
    <span class="avatar">🤖</span>
    <div class="bubble">
        {answer_html}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("question_form", clear_on_submit=True):
        cols = st.columns([8, 1])

        with cols[0]:
            question = st.text_input(
                "question",
                placeholder="Ask a question about the uploaded PDF...",
                label_visibility="collapsed",
            )

        with cols[1]:
            submitted = st.form_submit_button(
                "Send",
                use_container_width=True,
            )

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
            answer = generate_answer(q, st.session_state.uploaded_docs)
            response_time = round(time.time() - start_time, 2)

        answer["role"] = "agent"
        answer["response_time"] = response_time

        st.session_state.chat_history.append(answer)

        st.session_state.eval_runs.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "query": q,
                "abstained": answer.get("abstain", False),
                "response_time_s": response_time,
                "num_citations": len(answer.get("citations", [])),
                "error": answer.get("error"),
            }
        )

        st.rerun()
# Admin tab
with tab_admin:
    st.markdown(
        """
<div class="page-title">Administration & Evaluation Dashboard</div>
<div class="page-subtitle">
    Overview of uploaded PDFs, queries, abstentions and prototype evaluation data.
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
            f"""
<div class="metric-card">
    <div class="metric-value">{total_docs}</div>
    <div class="metric-label">Uploaded PDFs</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-value">{indexed_docs}</div>
    <div class="metric-label">Indexed PDFs</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-value">{total_queries}</div>
    <div class="metric-label">Questions</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-value">{abstentions}</div>
    <div class="metric-label">Abstentions</div>
</div>
""",
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

    st.write(
        "This version is mainly a frontend prototype. The PDF validation and indexing are "
        "simulated. In the final version, this part should be connected to the real backend "
        "pipeline for parsing, chunking, retrieval and answer generation."
    )