import json
import random
import urllib.request
from pathlib import Path
from collections import defaultdict

BASE = Path("evaluation/data/open_rag_bench/pdf/arxiv")
qrels = json.load(open(BASE / "qrels.json"))
answers = json.load(open(BASE / "answers.json"))
queries = json.load(open(BASE / "queries.json"))
pdf_urls = json.load(open(BASE / "pdf_urls.json"))

# Group query_ids by their target doc_id
queries_by_doc = defaultdict(list)
for qid, info in qrels.items():
    queries_by_doc[info["doc_id"]].append(qid)

# Pick 5 positive docs with the most queries (most useful for testing)
positive_docs = sorted(queries_by_doc.items(), key=lambda x: -len(x[1]))[:5]
positive_doc_ids = [doc_id for doc_id, _ in positive_docs]

# Pick 2 hard negatives: docs that exist but are never a query's target
all_doc_ids = set(pdf_urls.keys())
referenced_doc_ids = set(queries_by_doc.keys())
hard_negative_ids = list(all_doc_ids - referenced_doc_ids)
random.seed(42)
hard_negatives = random.sample(hard_negative_ids, 2)

print("Selected positive docs:", positive_doc_ids)
print("Selected hard negatives:", hard_negatives)

# Download the real PDFs
pdf_dir = Path("evaluation/data/open_rag_bench/pdfs")
pdf_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (student research project)"}
for doc_id in positive_doc_ids + hard_negatives:
    url = pdf_urls[doc_id]
    out_path = pdf_dir / f"{doc_id}.pdf"
    if out_path.exists():
        print(f"Already downloaded: {doc_id}")
        continue
    print(f"Downloading {doc_id} from {url} ...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(out_path, "wb") as f:
        f.write(response.read())
    print(f"  saved to {out_path}")

# Build answerable questions (2 per positive doc)
entries = []
for doc_id, qids in positive_docs:
    for qid in qids[:2]:
        entries.append({
            "question": queries[qid]["query"],
            "expected_answer": answers[qid],
            "expected_source": f"{doc_id}.pdf",
            "expected_page": None,
            "should_abstain": False,
        })

# Build abstention test cases: real queries whose true doc is NOT indexed
selected_ids = set(positive_doc_ids + hard_negatives)
unrelated_qids = [
    qid for qid, info in qrels.items()
    if info["doc_id"] not in selected_ids
]
random.shuffle(unrelated_qids)
for qid in unrelated_qids[:3]:
    entries.append({
        "question": queries[qid]["query"],
        "expected_answer": None,
        "expected_source": None,
        "expected_page": None,
        "should_abstain": True,
    })

out_file = Path("evaluation/data/benchmark_subset.json")
with out_file.open("w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

print(f"\nWrote {len(entries)} entries to {out_file}")
