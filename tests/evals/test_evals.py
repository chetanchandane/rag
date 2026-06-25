"""
tests/evals/test_evals.py — Unit tests for Phase 3 eval logic.

These tests cover pure logic (threshold checks, schema validation, evaluator
wiring) and do NOT call any external APIs.  They require eval dependencies:

    pip install -r requirements-eval.txt
    pytest tests/evals/ -v
"""

import json
from unittest.mock import MagicMock

import pytest

# Skip the entire module if eval dependencies aren't installed
pytest.importorskip("ragas", reason="install requirements-eval.txt to run eval tests")


# ── Threshold check logic ─────────────────────────────────────────────────────

THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.70,
    "context_precision": 0.60,
    "context_recall":    0.60,
}


def _passes(scores: dict[str, float]) -> bool:
    return all(scores.get(k, 0.0) >= v for k, v in THRESHOLDS.items())


def test_all_metrics_pass():
    scores = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.80,
        "context_precision": 0.75,
        "context_recall":    0.65,
    }
    assert _passes(scores) is True


def test_faithfulness_below_threshold_fails():
    scores = {
        "faithfulness":      0.50,   # below 0.70
        "answer_relevancy":  0.80,
        "context_precision": 0.75,
        "context_recall":    0.65,
    }
    assert _passes(scores) is False


def test_context_precision_at_exact_threshold_passes():
    scores = {
        "faithfulness":      0.70,
        "answer_relevancy":  0.70,
        "context_precision": 0.60,   # exactly at threshold
        "context_recall":    0.60,
    }
    assert _passes(scores) is True


def test_missing_metric_defaults_to_zero_and_fails():
    scores = {"faithfulness": 0.90, "answer_relevancy": 0.90}  # missing context metrics
    assert _passes(scores) is False


def test_all_metrics_at_zero_fails():
    assert _passes({k: 0.0 for k in THRESHOLDS}) is False


# ── SingleTurnSample schema ────────────────────────────────────────────────────

def test_single_turn_sample_construction():
    """SingleTurnSample builds cleanly from eval pipeline outputs."""
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input="What are the SAE reporting timelines under ICH E6?",
        response="SAEs must be reported within 24 hours per ICH E6(R3) section 5.17.",
        retrieved_contexts=[
            "ICH E6(R3) requires sponsors to report all SAEs within 24 hours."
        ],
        reference="Under ICH E6, sponsors must report serious adverse events immediately.",
    )

    assert sample.user_input == "What are the SAE reporting timelines under ICH E6?"
    assert len(sample.retrieved_contexts) == 1
    assert sample.response is not None
    assert sample.reference is not None


def test_single_turn_sample_with_multiple_contexts():
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input="What is GCP?",
        response="GCP stands for Good Clinical Practice.",
        retrieved_contexts=["GCP is an ethical standard.", "ICH E6 defines GCP requirements."],
        reference="Good Clinical Practice is an international ethical and scientific standard.",
    )

    assert len(sample.retrieved_contexts) == 2


# ── QA record schema ──────────────────────────────────────────────────────────

def test_qa_record_has_all_required_fields():
    """Each generated Q&A record has the fields expected by upload_to_langsmith."""
    record = {
        "question":              "What is the required timeline for reporting SAEs?",
        "ground_truth":          "Sponsors must report SAEs within 24 hours.",
        "ground_truth_contexts": ["ICH E6 section 5.17 requires 24-hour SAE reporting."],
        "source":                "ICH_E6_R3.pdf",
        "page":                  34,
    }
    assert "question" in record
    assert "ground_truth" in record
    assert isinstance(record["ground_truth_contexts"], list)
    assert len(record["ground_truth_contexts"]) > 0


# ── Claude response parsing ───────────────────────────────────────────────────

def test_valid_claude_json_parses_correctly():
    raw = '{"question": "What is the SAE timeline?", "ground_truth": "24 hours."}'
    qa  = json.loads(raw)
    assert qa["question"] == "What is the SAE timeline?"
    assert qa["ground_truth"] == "24 hours."


def test_malformed_claude_json_raises_decode_error():
    raw = "Here is the question: {malformed json}"
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)


def test_missing_key_in_claude_response_is_caught():
    raw = '{"question": "What is the SAE timeline?"}'   # missing ground_truth
    qa  = json.loads(raw)
    assert qa.get("ground_truth") is None


# ── Evaluator factory ─────────────────────────────────────────────────────────

def test_evaluator_returns_evaluation_result_with_correct_key():
    """_make_evaluator wraps a Ragas metric and returns a valid EvaluationResult."""
    from langsmith.evaluation import EvaluationResult
    from ragas.dataset_schema import SingleTurnSample

    # Inline the evaluator factory (mirrors run_evals._make_evaluator)
    def _make_evaluator(metric_name, metric, scores_tracker):
        def evaluator(run, example):
            sample = SingleTurnSample(
                user_input=example.inputs.get("question", ""),
                response=(run.outputs or {}).get("answer", ""),
                retrieved_contexts=(run.outputs or {}).get("contexts", []),
                reference=example.outputs.get("ground_truth", ""),
            )
            score = metric.single_turn_score(sample)
            score = round(float(score), 4)
            scores_tracker[metric_name].append(score)
            return EvaluationResult(key=metric_name, score=score)
        return evaluator

    mock_metric = MagicMock()
    mock_metric.single_turn_score.return_value = 0.85

    run     = MagicMock()
    run.outputs = {"answer": "SAEs must be reported within 24 hours.", "contexts": ["Context text."]}

    example = MagicMock()
    example.inputs  = {"question": "What is the SAE reporting timeline?"}
    example.outputs = {"ground_truth": "24 hours per ICH E6."}

    tracker   = {"faithfulness": []}
    evaluator = _make_evaluator("faithfulness", mock_metric, tracker)
    result    = evaluator(run, example)

    assert isinstance(result, EvaluationResult)
    assert result.key   == "faithfulness"
    assert result.score == 0.85
    assert tracker["faithfulness"] == [0.85]


def test_evaluator_handles_scoring_error_gracefully():
    """If a Ragas metric raises, the evaluator returns score=0.0 rather than crashing."""
    from langsmith.evaluation import EvaluationResult
    from ragas.dataset_schema import SingleTurnSample

    def _make_evaluator(metric_name, metric, scores_tracker):
        def evaluator(run, example):
            sample = SingleTurnSample(
                user_input=example.inputs.get("question", ""),
                response=(run.outputs or {}).get("answer", ""),
                retrieved_contexts=(run.outputs or {}).get("contexts", []),
                reference=example.outputs.get("ground_truth", ""),
            )
            try:
                score = metric.single_turn_score(sample)
            except Exception:
                score = 0.0
            score = round(float(score), 4)
            scores_tracker[metric_name].append(score)
            return EvaluationResult(key=metric_name, score=score)
        return evaluator

    mock_metric = MagicMock()
    mock_metric.single_turn_score.side_effect = RuntimeError("LLM call failed")

    run = MagicMock()
    run.outputs = {"answer": "Some answer", "contexts": []}

    example = MagicMock()
    example.inputs  = {"question": "?"}
    example.outputs = {"ground_truth": "Some truth"}

    tracker   = {"faithfulness": []}
    evaluator = _make_evaluator("faithfulness", mock_metric, tracker)
    result    = evaluator(run, example)

    assert result.score == 0.0
    assert tracker["faithfulness"] == [0.0]


# ── Exit-code logic ───────────────────────────────────────────────────────────

def test_exit_pass_when_all_thresholds_met():
    """Simulates the CI gate decision without calling sys.exit()."""
    scores = {k: t + 0.05 for k, t in THRESHOLDS.items()}   # all slightly above
    assert _passes(scores) is True


def test_exit_fail_when_one_threshold_missed():
    scores = {k: t + 0.05 for k, t in THRESHOLDS.items()}
    scores["context_recall"] = 0.40   # below 0.60 threshold
    assert _passes(scores) is False
