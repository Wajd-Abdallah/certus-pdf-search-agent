"""
Regression tests for evaluation/retrieval_metrics.py, using small,
hand-constructed synthetic chunk lists with known correct answers.
These exist specifically to catch metric-calculation bugs (like an
nDCG value exceeding 1.0) automatically, rather than relying on
manual inspection of real evaluation runs.
"""

from evaluation.retrieval_metrics import recall_at_k, mrr, ndcg_at_k, evaluate_question


def make_chunk(source: str, page: int) -> dict:
    return {"source": source, "page": page, "text": "irrelevant text", "score": 0.0}


def test_recall_and_mrr_when_relevant_chunk_is_first():
    chunks = [
        make_chunk("paper.pdf", 3),
        make_chunk("other.pdf", 1),
    ]
    assert recall_at_k(chunks, "paper.pdf", 3) == 1.0
    assert mrr(chunks, "paper.pdf", 3) == 1.0
    assert ndcg_at_k(chunks, "paper.pdf", 3) == 1.0


def test_mrr_when_relevant_chunk_is_third():
    chunks = [
        make_chunk("other.pdf", 1),
        make_chunk("other.pdf", 2),
        make_chunk("paper.pdf", 3),
    ]
    assert recall_at_k(chunks, "paper.pdf", 3) == 1.0
    assert mrr(chunks, "paper.pdf", 3) == 1.0 / 3
    # nDCG must be strictly between 0 and 1 when the hit isn't at rank 1
    assert 0.0 < ndcg_at_k(chunks, "paper.pdf", 3) < 1.0


def test_all_metrics_zero_when_source_never_appears():
    chunks = [
        make_chunk("other.pdf", 1),
        make_chunk("other.pdf", 2),
    ]
    assert recall_at_k(chunks, "paper.pdf", 3) == 0.0
    assert mrr(chunks, "paper.pdf", 3) == 0.0
    assert ndcg_at_k(chunks, "paper.pdf", 3) == 0.0


def test_ndcg_never_exceeds_one_with_multiple_relevant_chunks():
    """
    Regression test for a real bug found in this project: when several
    retrieved chunks all come from the same (correct) document, nDCG
    was computed with a hardcoded ideal score of 1.0, allowing the
    result to exceed 1.0 (e.g. 2.9). nDCG must always stay in [0, 1].
    """
    chunks = [
        make_chunk("paper.pdf", 1),
        make_chunk("paper.pdf", 2),
        make_chunk("paper.pdf", 3),
        make_chunk("paper.pdf", 4),
    ]
    score = ndcg_at_k(chunks, "paper.pdf", None, k=4)
    assert 0.0 <= score <= 1.0
    assert score == 1.0  # all 4 relevant, all ranked first -- ideal case


def test_no_ground_truth_returns_all_zero():
    chunks = [make_chunk("paper.pdf", 1)]
    result = evaluate_question(chunks, None, None)
    assert result == {"Recall@k": 0.0, "MRR": 0.0, "nDCG@k": 0.0}


def test_partial_ground_truth_ignores_missing_page():
    """expected_page=None means only the source needs to match."""
    chunks = [make_chunk("paper.pdf", 99)]
    assert recall_at_k(chunks, "paper.pdf", None) == 1.0
