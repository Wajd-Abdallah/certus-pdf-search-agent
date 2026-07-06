"""
Pure, dependency-light metric functions for answer-quality, cost, and
abstention-classification evaluation. Kept separate from
evaluation/full_eval.py (which orchestrates the full pipeline) so these
functions can be unit-tested in isolation, without importing app.pipeline
or Ollama.
"""

import re
from collections import Counter

STOPWORDS = {
    "the", "and", "for", "are", "was", "but", "with", "that", "this",
    "from", "have", "has", "had", "were", "will", "would", "there",
    "their", "about", "into", "than", "then", "they", "them",
}

ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS = 0.0003
ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS = 0.0009


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def compute_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_correct(citations: list, expected_source: str) -> bool:
    sources = {c.get("document") for c in citations}
    return expected_source in sources


def lexical_grounding_score(answer_text: str, retrieved_contexts: list[str]) -> float:
    if not answer_text or not retrieved_contexts:
        return 0.0
    answer_words = re.findall(r"\b\w{3,}\b", answer_text.lower())
    answer_tokens = {w for w in answer_words if w not in STOPWORDS}
    if not answer_tokens:
        return 0.0
    context_text = " ".join(retrieved_contexts)
    context_words = re.findall(r"\b\w{3,}\b", context_text.lower())
    context_tokens = {w for w in context_words if w not in STOPWORDS}
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def estimate_illustrative_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS
        + completion_tokens * ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS
    ) / 1000.0


def classify_abstention(should_abstain: bool, did_abstain: bool, abstention_reason: str | None) -> str:
    if abstention_reason == "generation_error":
        return "system_error"
    if should_abstain and did_abstain:
        return "true_positive"
    if should_abstain and not did_abstain:
        return "false_negative"
    if not should_abstain and not did_abstain:
        return "true_negative"
    return "false_positive"
