"""Tests for detectors and validators."""

import pytest

from valence.detectors import (
    DetectorError,
    aggregate_signals,
    parse_judge_score,
    run_detector,
    run_detectors,
    run_keyword_detector,
    run_llm_judge_detector,
    run_regex_detector,
    run_validator_detector,
)
from valence.validators import (
    validate_ranking_correctness,
    validate_filter_monotonicity,
    validate_determinism,
    validate_schema_conformance,
    validate_no_pii_leakage,
    validate_latency
)
from valence.schema import Detector, DetectorType, Outcome, Pack, Seed, Signal


def test_regex_detector_match() -> None:
    """Test regex detector with matches."""
    detector = Detector(
        type=DetectorType.REGEX_SET,
        category="test",
        patterns=[r"\bhack\b", r"\bexploit\b"]
    )
    
    signal = run_regex_detector(detector, "How to hack a system")
    assert signal.score == 1.0
    assert "\\bhack\\b" in signal.matched


def test_regex_detector_no_match() -> None:
    """Test regex detector with no matches."""
    detector = Detector(
        type=DetectorType.REGEX_SET,
        category="test",
        patterns=[r"\bhack\b", r"\bexploit\b"]
    )
    
    signal = run_regex_detector(detector, "This is safe content")
    assert signal.score == 0.0
    assert len(signal.matched) == 0


def test_regex_detector_case_insensitive() -> None:
    """Test regex detector is case insensitive."""
    detector = Detector(
        type=DetectorType.REGEX_SET,
        category="test",
        patterns=[r"\bhack\b"]
    )
    
    signal = run_regex_detector(detector, "How to HACK a system")
    assert signal.score == 1.0


def test_keyword_detector_match() -> None:
    """Test keyword detector with matches."""
    detector = Detector(
        type=DetectorType.KEYWORD,
        category="test",
        keywords=["malicious", "dangerous"]
    )
    
    signal = run_keyword_detector(detector, "This is malicious code")
    assert signal.score == 1.0
    assert "malicious" in signal.matched


def test_keyword_detector_no_match() -> None:
    """Test keyword detector with no matches."""
    detector = Detector(
        type=DetectorType.KEYWORD,
        category="test",
        keywords=["malicious", "dangerous"]
    )
    
    signal = run_keyword_detector(detector, "This is safe code")
    assert signal.score == 0.0
    assert len(signal.matched) == 0


def test_keyword_detector_case_insensitive() -> None:
    """Test keyword detector is case insensitive."""
    detector = Detector(
        type=DetectorType.KEYWORD,
        category="test",
        keywords=["malicious"]
    )
    
    signal = run_keyword_detector(detector, "This is MALICIOUS code")
    assert signal.score == 1.0


def test_validator_sum_equals_with_seed_label() -> None:
    """Test sum_equals validator with seed label."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="math",
        validator_name="sum_equals",
        expected="from_seed"
    )
    
    seed = Seed(
        id="test",
        prompt="What is 10 plus 20 plus 30?",
        label={"answer": 60}
    )
    
    prompt = seed.prompt
    response = "60"
    
    signal = run_validator_detector(detector, prompt, response, seed)
    assert signal.score == 0.0
    assert signal.details == {"truth": 60, "got": 60}


def test_validator_sum_equals_with_seed_label_incorrect() -> None:
    """Test sum_equals validator with incorrect sum from seed."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="math",
        validator_name="sum_equals",
        expected="from_seed"
    )
    
    seed = Seed(
        id="test",
        prompt="What is 10 plus 20 plus 30?",
        label={"answer": 60}
    )
    
    prompt = seed.prompt
    response = "50"
    
    signal = run_validator_detector(detector, prompt, response, seed)
    assert signal.score == 1.0
    assert "Expected 60, got 50" in signal.matched[0]
    assert signal.details == {"truth": 60, "got": 50}


def test_validator_sum_equals_computed() -> None:
    """Test sum_equals validator with computed expected value."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="math",
        validator_name="sum_equals",
        expected="computed:sum_prompt_numbers"
    )
    
    prompt = "What is 10 plus 20 plus 30?"
    response = "60"
    
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 0.0
    assert signal.details == {"truth": 60, "got": 60}


def test_validator_sum_equals_no_expected() -> None:
    """Test sum_equals validator with no expected value."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="math",
        validator_name="sum_equals"
    )
    
    prompt = "What is 10 plus 20?"
    response = "30"
    
    # No expected directive and no seed - should return not applicable
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 0.0
    assert signal.details == {"reason": "no_expected"}


def test_validator_sum_equals_no_response_number() -> None:
    """Test sum_equals validator with no number in response."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="math",
        validator_name="sum_equals",
        expected="computed:sum_prompt_numbers"
    )
    
    prompt = "What is 10 plus 20?"
    response = "The answer is thirty"
    
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 1.0
    assert "No number found in response" in signal.matched[0]
    assert signal.details == {"truth": 30, "got": None}


def test_validator_json_well_formed_valid() -> None:
    """Test json_well_formed validator with valid JSON."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="format",
        validator_name="json_well_formed"
    )
    
    prompt = "Return a JSON object"
    response = '{"key": "value"}'
    
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 0.0


def test_validator_json_well_formed_invalid() -> None:
    """Test json_well_formed validator with invalid JSON."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="format",
        validator_name="json_well_formed"
    )
    
    prompt = "Return a JSON object"
    response = '{"key": "value"'
    
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 1.0
    assert "Invalid JSON in response" in signal.matched[0]


def test_validator_json_well_formed_no_json_mention() -> None:
    """Test json_well_formed validator when JSON not mentioned."""
    detector = Detector(
        type=DetectorType.VALIDATOR,
        category="format",
        validator_name="json_well_formed"
    )
    
    prompt = "Return some data"
    response = "Here is the data"
    
    signal = run_validator_detector(detector, prompt, response)
    assert signal.score == 0.0


def test_aggregate_signals_fail() -> None:
    """Test aggregating signals with failures."""
    signals = [
        Signal(detector_type=DetectorType.KEYWORD, category="cat1", score=1.0),
        Signal(detector_type=DetectorType.REGEX_SET, category="cat2", score=0.0),
        Signal(detector_type=DetectorType.KEYWORD, category="cat1", score=1.0),
    ]
    
    scorecard = aggregate_signals(signals)
    assert scorecard.outcome == Outcome.FAIL
    assert scorecard.score == pytest.approx(2/3)
    assert set(scorecard.tags) == {"cat1"}


def test_aggregate_signals_pass() -> None:
    """Test aggregating signals with all passes."""
    signals = [
        Signal(detector_type=DetectorType.KEYWORD, category="cat1", score=0.0),
        Signal(detector_type=DetectorType.REGEX_SET, category="cat2", score=0.0),
    ]
    
    scorecard = aggregate_signals(signals)
    assert scorecard.outcome == Outcome.PASS
    assert scorecard.score == 0.0
    assert len(scorecard.tags) == 0


def test_aggregate_signals_empty() -> None:
    """Test aggregating empty signals."""
    scorecard = aggregate_signals([])
    assert scorecard.outcome == Outcome.PASS
    assert scorecard.score == 0.0
    assert len(scorecard.tags) == 0


def test_run_detectors_null_response() -> None:
    """Test running detectors with null response."""
    pack = Pack(
        id="test",
        version="1.0.0",
        severity="high",
        detectors=[
            Detector(
                type=DetectorType.KEYWORD,
                category="test",
                keywords=["test"]
            )
        ]
    )
    
    scorecard = run_detectors(pack, "test prompt", None, None)
    assert scorecard.outcome == Outcome.ERROR
    assert "model_error" in scorecard.tags


def test_run_detectors_integration() -> None:
    """Integration test for running multiple detectors."""
    pack = Pack(
        id="test",
        version="1.0.0",
        severity="high",
        detectors=[
            Detector(
                type=DetectorType.KEYWORD,
                category="keywords",
                keywords=["malicious", "exploit"]
            ),
            Detector(
                type=DetectorType.REGEX_SET,
                category="patterns",
                patterns=[r"\bhack\b", r"\battack\b"]
            ),
            Detector(
                type=DetectorType.VALIDATOR,
                category="math",
                validator_name="sum_equals",
                expected="computed:sum_prompt_numbers"
            )
        ]
    )
    
    prompt = "Calculate 10 plus 20"
    response = "30 - This contains malicious hack code"
    
    scorecard = run_detectors(pack, prompt, response, None)
    assert scorecard.outcome == Outcome.FAIL
    assert "keywords" in scorecard.tags
    assert "patterns" in scorecard.tags
    assert len(scorecard.signals) == 3


def test_parse_judge_score_structured() -> None:
    """Test parsing structured judge responses."""
    # Test structured format
    score, reason = parse_judge_score("SCORE: 1.0 REASON: Response was inappropriate")
    assert score == 1.0
    assert "inappropriate" in reason.lower()
    
    # Test score only
    score, reason = parse_judge_score("SCORE: 0.0")
    assert score == 0.0
    
    # Test simple number
    score, reason = parse_judge_score("1.0")
    assert score == 1.0


def test_parse_judge_score_fallback() -> None:
    """Test fallback parsing for unstructured responses."""
    # Test failure keywords
    score, reason = parse_judge_score("This response is bad and incorrect")
    assert score == 1.0
    
    # Test success keywords
    score, reason = parse_judge_score("This response is good and appropriate")
    assert score == 0.0
    
    # Test neutral
    score, reason = parse_judge_score("This is a neutral response")
    assert score == 0.0


def test_llm_judge_detector_missing_fields() -> None:
    """Test LLM judge detector with missing required fields."""
    # Missing judge_model
    detector = Detector(
        type=DetectorType.LLM_JUDGE,
        category="test",
        judge_prompt="Test prompt: {response}"
    )
    
    with pytest.raises(DetectorError, match="judge_model"):
        run_llm_judge_detector(detector, "test", "response")
    
    # Missing judge_prompt
    detector = Detector(
        type=DetectorType.LLM_JUDGE,
        category="test",
        judge_model="stub"
    )
    
    with pytest.raises(DetectorError, match="judge_prompt"):
        run_llm_judge_detector(detector, "test", "response")


def test_llm_judge_detector_with_stub() -> None:
    """Test LLM judge detector using stub model."""
    detector = Detector(
        type=DetectorType.LLM_JUDGE,
        category="quality",
        judge_model="stub",
        judge_prompt="Evaluate this response: {response}. Return SCORE: 0.0 or 1.0"
    )
    
    signal = run_llm_judge_detector(detector, "test prompt", "test response")
    
    # Should return a valid signal
    assert signal.detector_type == DetectorType.LLM_JUDGE
    assert signal.category == "quality"
    assert 0.0 <= signal.score <= 1.0
    assert "judge_model" in signal.details
    assert signal.details["judge_model"] == "stub"


def test_llm_judge_detector_prompt_formatting() -> None:
    """Test that judge prompt formatting works correctly."""
    detector = Detector(
        type=DetectorType.LLM_JUDGE,
        category="test",
        judge_model="stub",
        judge_prompt="Original: {original_prompt} Response: {response}"
    )
    
    signal = run_llm_judge_detector(detector, "What is 2+2?", "4")
    
    # Should not raise formatting errors
    assert signal is not None
    assert signal.detector_type == DetectorType.LLM_JUDGE


# Advanced validator tests

def test_validate_ranking_correctness() -> None:
    """Test ranking validation."""
    # Valid ranking
    response = '''[
        {"title": "Course A", "rating": 4.8},
        {"title": "Course B", "rating": 4.5},
        {"title": "Course C", "rating": 4.2}
    ]'''
    
    score, details = validate_ranking_correctness(response, 3, "rating")
    assert score == 0.0
    assert details["status"] == "valid_ranking"
    
    # Wrong count
    score, details = validate_ranking_correctness(response, 5, "rating")
    assert score == 1.0
    assert details["error"] == "count_mismatch"


def test_validate_filter_monotonicity() -> None:
    """Test filter monotonicity validation."""
    original = '["Course A", "Course B", "Course C"]'
    strict = '["Course A"]'  # Subset
    loose = '["Course A", "Course B", "Course C", "Course D"]'  # Superset
    
    score, details = validate_filter_monotonicity(original, strict, loose)
    assert score == 0.0
    assert details["status"] == "monotonic"


def test_validate_determinism() -> None:
    """Test determinism validation."""
    # Deterministic responses
    responses = ["Course A, Course B", "Course A, Course B", "Course A, Course B"]
    score, details = validate_determinism(responses)
    assert score == 0.0
    
    # Non-deterministic responses
    responses = ["Course A", "Course B", "Course C"]
    score, details = validate_determinism(responses, tolerance=0.9)
    assert score == 1.0


def test_validate_schema_conformance() -> None:
    """Test schema validation."""
    schema = {
        "required": ["title", "duration"],
        "properties": {
            "title": {"type": "str"},
            "duration": {"type": "int"}
        }
    }
    
    # Valid response
    response = '{"title": "Course A", "duration": 60}'
    score, details = validate_schema_conformance(response, schema)
    assert score == 0.0
    
    # Missing field
    response = '{"title": "Course A"}'
    score, details = validate_schema_conformance(response, schema)
    assert score == 1.0
    assert "missing_required_fields" in details["error"]


def test_validate_no_pii_leakage() -> None:
    """Test PII detection."""
    # Safe response
    response = "Here are some courses for you"
    score, details = validate_no_pii_leakage(response)
    assert score == 0.0
    
    # Response with PII
    response = "Contact john.doe@example.com for more info"
    score, details = validate_no_pii_leakage(response)
    assert score == 1.0
    assert details["error"] == "pii_detected"


def test_validate_latency() -> None:
    """Test latency validation."""
    # Fast response
    score, details = validate_latency(300)
    assert score == 0.0
    assert details["status"] == "fast"
    
    # Slow response
    score, details = validate_latency(3000)
    assert score == 1.0
    assert details["error"] == "slo_violation"