Klar, hier ist der komplette richtige README.md-Code ohne die äußeren Chat-Backticks:

# PDF Search Agent
A modular Retrieval-Augmented Generation (RAG) system that answers questions
based on uploaded PDF documents and supports its answers with citations.
> **Note:** This is only an initial project structure and setup proposal.
> It is not final and can still be changed during development if the team finds a better solution.
## Description
The PDF Search Agent allows users to upload PDF documents, ask questions, and receive
answers that are grounded in the document content and supported by citations.
If no sufficient evidence is found, the system will abstain from answering instead of
generating an unsupported response.
The system is built as a modular pipeline so that each component can be developed,
tested, and improved independently.
## Project Structure
```text
pdf-search-agent/
├── app/
│   ├── parser.py          # Initial idea: PDF ingestion and text extraction
│   ├── chunker.py         # Initial idea: text chunking strategies
│   ├── indexer.py         # Initial idea: vector indexing
│   ├── retriever.py       # Initial idea: chunk retrieval
│   ├── generator.py       # Initial idea: answer generation with citations
│   └── pipeline.py        # Initial idea: full RAG pipeline
├── evaluation/            # Evaluation metrics, benchmarks, and test scripts
├── ui/
│   └── streamlit_app.py   # Initial idea for a simple user interface
├── data/                  # PDF storage and sample files
├── tests/                 # Unit tests and integration tests
└── requirements.txt

Installation

The following setup is a first proposal and may be adapted later.

Requirements

* Python 3.10 or higher
* pip

Steps

1. Clone the repository:

git clone https://git.rz.tu-bs.de/isf/sep/sep-2026/iai_drpsa_pdf-search-agent_g1/code.git
cd code

2. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

Usage

Run the application:

streamlit run ui/streamlit_app.py

Then open your browser and go to http://localhost:8501￼.

From there you can:

* Upload PDF documents
* Enter questions
* View answers with citations or abstention messages

Features

This list is not final. Features may be added, removed, or refined as the project develops.

* PDF ingestion and parsing
* Semantic chunk retrieval
* Citation-supported answers
* Abstention when evidence is insufficient
* Modular and testable pipeline
* Evaluation suite with documented metrics

Project Status

This repository is currently in an early stage.
The structure, tools, and implementation details are still subject to discussion and improvement.

Authors

Group 1 — PDF Search Agent
Software Engineering Praktikum 2026, TU Braunschweig