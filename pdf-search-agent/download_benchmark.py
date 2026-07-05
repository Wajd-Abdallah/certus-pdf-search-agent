from huggingface_hub import hf_hub_download

files = [
    "pdf/arxiv/pdf_urls.json",
    "pdf/arxiv/queries.json",
    "pdf/arxiv/qrels.json",
    "pdf/arxiv/answers.json",
]

for f in files:
    path = hf_hub_download(
        repo_id="vectara/open_ragbench",
        repo_type="dataset",
        filename=f,
        local_dir="evaluation/data/open_rag_bench",
    )
    print(f"Downloaded: {path}")
