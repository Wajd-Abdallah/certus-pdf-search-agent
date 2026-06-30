from math import log2
from typing import Any, Dict, List


def _is_relevant(
    chunk: Dict[str, Any],
    expected_source: str | None,
    expected_page: int | None,
) -> bool:
    """
    Returns True if the retrieved chunk matches the expected source/page.
    """

    source_matches = (
        expected_source is None
        or chunk.get("source") == expected_source
    )

    page_matches = (
        expected_page is None
        or chunk.get("page") == expected_page
    )

    return source_matches and page_matches


def recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source: str | None,
    expected_page: int | None,
    k: int = 5,
) -> float:
    """
    Recall@k

    Returns:
        1.0 if a relevant chunk exists in the first k results.
        0.0 otherwise.
    """

    if expected_source is None and expected_page is None:
        return 0.0

    for chunk in retrieved_chunks[:k]:
        if _is_relevant(chunk, expected_source, expected_page):
            return 1.0

    return 0.0


def mrr(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source: str | None,
    expected_page: int | None,
) -> float:
    """
    Mean Reciprocal Rank (MRR)

    Returns:
        1 / rank of the first relevant chunk.
    """

    if expected_source is None and expected_page is None:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if _is_relevant(chunk, expected_source, expected_page):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source: str | None,
    expected_page: int | None,
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@k)

    Assumes binary relevance:
        relevant = 1
        not relevant = 0
    """

    if expected_source is None and expected_page is None:
        return 0.0

    dcg = 0.0

    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if _is_relevant(chunk, expected_source, expected_page):
            dcg += 1.0 / log2(rank + 1)

    # Only one relevant chunk is expected.
    idcg = 1.0

    return dcg / idcg


def evaluate_question(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source: str | None,
    expected_page: int | None,
    k: int = 5,
) -> Dict[str, float]:
    """
    Computes all retrieval metrics for a single benchmark question.
    """

    return {
        "Recall@k": recall_at_k(
            retrieved_chunks,
            expected_source,
            expected_page,
            k,
        ),
        "MRR": mrr(
            retrieved_chunks,
            expected_source,
            expected_page,
        ),
        "nDCG@k": ndcg_at_k(
            retrieved_chunks,
            expected_source,
            expected_page,
            k,
        ),
    }