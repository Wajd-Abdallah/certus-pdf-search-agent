"""
Unit tests for evaluation/answer_metrics.py -- pure, isolated functions
covering answer-quality, cost estimation, and abstention classification.
"""

from evaluation.answer_metrics import (
    compute_exact_match,
    compute_f1,
    citation_correct,
    lexical_grounding_score,
    estimate_illustrative_cost_usd,
    classify_abstention,
)


def test_exact_match_true_for_identical_normalized_text():
    assert compute_exact_match("The Cat Sat.", "the cat sat") == 1.0


def test_exact_match_false_for_different_text():
    assert compute_exact_match("The cat sat", "The dog ran") == 0.0


def test_f1_is_one_for_identical_text():
    assert compute_f1("glucose and oxygen", "glucose and oxygen") == 1.0


def test_f1_is_zero_for_no_word_overlap():
    assert compute_f1("completely different words", "nothing shared here") == 0.0


def test_f1_is_between_zero_and_one_for_partial_overlap():
    score = compute_f1("the model produces glucose", "the process produces glucose and oxygen")
    assert 0.0 < score < 1.0


def test_f1_handles_empty_strings_without_raising():
    assert compute_f1("", "") == 1.0
    assert compute_f1("something", "") == 0.0


def test_citation_correct_true_when_expected_source_present():
    citations = [{"document": "paper.pdf", "page_number": 3}]
    assert citation_correct(citations, "paper.pdf") is True


def test_citation_correct_false_when_expected_source_absent():
    citations = [{"document": "other.pdf", "page_number": 1}]
    assert citation_correct(citations, "paper.pdf") is False


def test_lexical_grounding_high_for_matching_vocabulary():
    answer = "Photosynthesis produces glucose and oxygen"
    contexts = ["Photosynthesis is a process that produces glucose and oxygen from sunlight."]
    assert lexical_grounding_score(answer, contexts) > 0.5


def test_lexical_grounding_zero_for_empty_input():
    assert lexical_grounding_score("", ["some context"]) == 0.0
    assert lexical_grounding_score("some answer", []) == 0.0


def test_lexical_grounding_never_exceeds_one():
    answer = "cats dogs cats dogs"
    contexts = ["cats dogs cats dogs cats dogs"]
    assert 0.0 <= lexical_grounding_score(answer, contexts) <= 1.0


def test_cost_estimate_scales_with_output_tokens_more_than_input():
    cost_input_heavy = estimate_illustrative_cost_usd(prompt_tokens=1000, completion_tokens=10)
    cost_output_heavy = estimate_illustrative_cost_usd(prompt_tokens=10, completion_tokens=1000)
    assert cost_output_heavy > cost_input_heavy


def test_cost_estimate_zero_for_zero_tokens():
    assert estimate_illustrative_cost_usd(0, 0) == 0.0


def test_classify_abstention_true_positive():
    assert classify_abstention(should_abstain=True, did_abstain=True, abstention_reason="insufficient_context") == "true_positive"


def test_classify_abstention_true_negative():
    assert classify_abstention(should_abstain=False, did_abstain=False, abstention_reason=None) == "true_negative"


def test_classify_abstention_false_negative_is_flagged_as_risk_case():
    assert classify_abstention(should_abstain=True, did_abstain=False, abstention_reason=None) == "false_negative"


def test_classify_abstention_false_positive():
    assert classify_abstention(should_abstain=False, did_abstain=True, abstention_reason="insufficient_context") == "false_positive"


def test_classify_abstention_system_error_excluded_from_normal_outcomes():
    result = classify_abstention(should_abstain=False, did_abstain=True, abstention_reason="generation_error")
    assert result == "system_error"
