"""
Must be imported FIRST, before any app.* imports, in every evaluation
script. Points the pipeline at an isolated ChromaDB collection so
evaluation runs never read or write the live Streamlit app's data,
keeping results reproducible regardless of what's currently uploaded
in the live app.
"""
import os

os.environ.setdefault("PDF_AGENT_CHROMA_DIR", "./data/eval_chroma_db")
