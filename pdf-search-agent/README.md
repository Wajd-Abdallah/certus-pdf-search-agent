# PDF Search Agent
This repository contains an initial version of our PDF Search Agent project.

The goal of the project is to build a modular Retrieval-Augmented Generation (RAG) system that can answer questions based on uploaded PDF documents and support its answers with citations
> **Note:** This is only an initial project structure and setup proposal.  

> It is not final and can still be changed during development if the team finds a better solution.

## Initial Project Structure
## Initial project Structure
pdf-search-agent/
├── app/
│   ├── parser.py         # Initial idea: PDF ingestion and text extraction
│   ├── chunker.py        # Initial idea: text chunking strategies
│   ├── indexer.py        # Initial idea: vector indexing
│   ├── retriever.py      # Initial idea: chunk retrieval
│   ├── generator.py      # Initial idea: answer generation with citations
│   └── pipeline.py       # Initial idea: full RAG pipeline
├── evaluation/           # Evaluation metrics, benchmarks, and test scripts
├── ui/
│   └── streamlit_app.py  # Initial idea for a simple user interface
├── data/                 # PDF storage and sample files
├── tests/                # Unit tests and integration tests
└── requirements.txt
## Setup 
( The following setup is a first proposal and may be adapted later.)

1. Clone the repository
2. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run ui/streamlit_app.py

## Features

- PDF ingestion and parsing
- Semantic chunk retrieval
- Citation-supported answers
- Abstention when evidence is insufficient
- Modular and testable pipeline

Comment: This list is not final.
Features may be added, removed, or refined as the project develops.

Current Status

This repository is currently in an early stage.
The structure, tools, and implementation details are still subject to discussion and improvement.