"""Tests for evaluation runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valence.evalrun import EvalRunner, run_evaluation
from valence.model import Model
from valence.schema import Detector, DetectorType, EvalRecord, Lineage, Outcome, Pack, Scorecard, Seed
from valence.ci import calculate_new_failures, calculate_metamorphic_consistency_score
from valence.metamorphic import generate_filter_variants, run_metamorphic_test
from valence.util import simhash, hamming_distance, are_similar, compute_failure_fingerprint


def test_evalrunner_init() -> None:
    """Test EvalRunner initialization."""
    model = Model("stub")
    seeds = [Seed(id="s1", prompt="test")]
    packs = [Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["test"]
        )]
    )]
    
    runner = EvalRunner(
        model=model,
        seeds=seeds,
        packs=packs,
        output_dir=Path("/tmp/test"),
        max_generations=2,
        mutations_per_failure=3
    )
    
    assert runner.model == model
    assert runner.seeds == seeds
    assert runner.packs == packs
    assert runner.max_generations == 2
    assert runner.mutations_per_failure == 3
    assert runner.run_id.startswith("run-")


def test_evaluate_prompt_pass(tmp_path: Path) -> None:
    """Test evaluating a passing prompt."""
    model = Model("stub")
    pack = Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["NOMATCH"]
        )]
    )
    
    runner = EvalRunner(
        model=model,
        seeds=[],
        packs=[pack],
        output_dir=tmp_path
    )
    
    with patch.object(model, 'generate', return_value=("Safe response", 10.0)):
        record = runner.evaluate_prompt(
            prompt_id="test-1",
            prompt="Test prompt",
            pack=pack
        )
    
    assert record.id == "test-1"
    assert record.prompt == "Test prompt"
    assert record.response == "Safe response"
    assert record.scorecard.outcome == Outcome.PASS
    assert runner.metadata.total_passes == 1
    assert runner.metadata.total_failures == 0


def test_evaluate_prompt_fail_generates_mutations(tmp_path: Path) -> None:
    """Test that failing prompt generates mutations."""
    model = Model("stub")
    pack = Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["malicious"]
        )]
    )
    
    runner = EvalRunner(
        model=model,
        seeds=[],
        packs=[pack],
        output_dir=tmp_path,
        max_generations=1,
        mutations_per_failure=2
    )
    
    with patch.object(model, 'generate', return_value=("malicious response", 10.0)):
        record = runner.evaluate_prompt(
            prompt_id="test-1",
            prompt="Test prompt",
            pack=pack
        )
    
    assert record.scorecard.outcome == Outcome.FAIL
    assert runner.metadata.total_failures == 1
    assert len(runner.pending_evaluations) == 2
    
    for pending in runner.pending_evaluations:
        assert pending["id"].startswith("test-1.c")
        assert pending["lineage"].parent_id == "test-1"
        assert pending["lineage"].generation == 1


def test_evaluate_prompt_error(tmp_path: Path) -> None:
    """Test evaluating prompt with model error."""
    model = Model("stub")
    pack = Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["test"]
        )]
    )
    
    runner = EvalRunner(
        model=model,
        seeds=[],
        packs=[pack],
        output_dir=tmp_path
    )
    
    with patch.object(model, 'generate', return_value=(None, 10.0)):
        record = runner.evaluate_prompt(
            prompt_id="test-1",
            prompt="Test prompt",
            pack=pack
        )
    
    assert record.response is None
    assert record.error == "Model returned no response"
    assert runner.metadata.total_errors == 1


def test_run_complete_evaluation(tmp_path: Path) -> None:
    """Test running complete evaluation."""
    model = Model("stub")
    seeds = [
        Seed(id="s1", prompt="safe prompt"),
        Seed(id="s2", prompt="dangerous prompt")
    ]
    pack = Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["dangerous"]
        )]
    )
    
    runner = EvalRunner(
        model=model,
        seeds=seeds,
        packs=[pack],
        output_dir=tmp_path,
        max_generations=1,
        mutations_per_failure=2
    )
    
    def mock_generate(prompt: str) -> tuple:
        if "dangerous" in prompt.lower():
            return ("dangerous response", 10.0)
        return ("safe response", 10.0)
    
    with patch.object(model, 'generate', side_effect=mock_generate):
        metadata = runner.run()
    
    assert metadata.total_prompts >= 2
    assert metadata.total_passes >= 1
    assert metadata.total_failures >= 1
    assert metadata.completed_at is not None
    
    assert (tmp_path / "evals.jsonl").exists()
    assert (tmp_path / "metadata.json").exists()


def test_run_with_memory(tmp_path: Path) -> None:
    """Test running evaluation with memory."""
    model = Model("stub")
    seeds = [Seed(id="s1", prompt="fail prompt")]
    pack = Pack(
        id="p1",
        version="1.0.0",
        severity="low",
        detectors=[Detector(
            type=DetectorType.KEYWORD,
            category="test",
            keywords=["fail"]
        )]
    )
    
    memory_path = tmp_path / "memory.jsonl"
    
    runner = EvalRunner(
        model=model,
        seeds=seeds,
        packs=[pack],
        output_dir=tmp_path,
        memory_path=memory_path
    )
    
    with patch.object(model, 'generate', return_value=("fail response", 10.0)):
        runner.run()
    
    assert memory_path.exists()
    with open(memory_path) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    assert "fail prompt" in lines[0]


def test_run_evaluation_integration(tmp_path: Path) -> None:
    """Integration test for run_evaluation function."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text('[{"id": "test", "prompt": "test prompt"}]')
    
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text("""
id: test-pack
version: "1.0.0"
severity: low
detectors:
  - type: keyword
    category: test
    keywords: ["test"]
""")
    
    output_dir = tmp_path / "output"
    
    metadata = run_evaluation(
        model_name="stub",
        seeds_path=seeds_file,
        packs_path=pack_file,
        output_dir=output_dir,
        max_generations=1,
        mutations_per_failure=2
    )
    
    assert metadata.model == "stub"
    assert metadata.total_prompts >= 1
    assert output_dir.exists()
    assert (output_dir / "evals.jsonl").exists()
    assert (output_dir / "metadata.json").exists()


# Utility function tests

def test_simhash_identical_texts() -> None:
    """Test SimHash produces same hash for identical texts."""
    text = "Find me leadership courses"
    hash1 = simhash(text)
    hash2 = simhash(text)
    assert hash1 == hash2


def test_simhash_similar_texts() -> None:
    """Test SimHash produces similar hashes for similar texts."""
    text1 = "Find me leadership courses"
    text2 = "Show me leadership training"
    
    hash1 = simhash(text1)
    hash2 = simhash(text2)
    distance = hamming_distance(hash1, hash2)
    
    # Should be similar (low Hamming distance)
    assert distance <= 25


def test_are_similar() -> None:
    """Test similarity detection."""
    text1 = "What is the completion rate for leadership training?"
    text2 = "What's the completion rate for leadership courses?"
    
    assert are_similar(text1, text2)
    
    # Different texts should not be similar
    text3 = "Show me cybersecurity courses"
    assert not are_similar(text1, text3)


def test_compute_failure_fingerprint() -> None:
    """Test failure fingerprint computation."""
    prompt = "Find courses"
    response = "No results found"
    tags = {"search", "results"}
    
    fingerprint1 = compute_failure_fingerprint(prompt, response, tags)
    fingerprint2 = compute_failure_fingerprint(prompt, response, tags)
    
    # Same inputs should produce same fingerprint
    assert fingerprint1 == fingerprint2
    
    # Different inputs should produce different fingerprints
    fingerprint3 = compute_failure_fingerprint("Different prompt", response, tags)
    assert fingerprint1 != fingerprint3


# Metamorphic testing
def test_generate_filter_variants() -> None:
    """Test filter variant generation."""
    original = "Find top 5 courses under 2 hours"
    variants = generate_filter_variants(original)
    
    assert "original" in variants
    assert "strict" in variants
    assert "loose" in variants
    
    # Strict should be more restrictive
    assert "top" in variants["strict"] and "1" in variants["strict"] or "1 hour" in variants["strict"]


@patch('valence.metamorphic.Model')
def test_run_metamorphic_test(mock_model_class: MagicMock) -> None:
    """Test metamorphic test execution."""
    mock_model = MagicMock()
    mock_model.generate.return_value = ("Test response", 100, None)
    mock_model_class.return_value = mock_model
    
    score, details = run_metamorphic_test(mock_model, "Find top 5 courses", "determinism")
    
    assert "test_type" in details
    assert details["test_type"] == "determinism"
    assert 0.0 <= score <= 1.0


# CI integration tests
def test_calculate_new_failures() -> None:
    """Test new failure calculation."""
    baseline_fingerprints = {"hash1", "hash2"}
    
    current_evals = [
        EvalRecord(
            id="test1",
            prompt="test",
            model="test",
            response="fail",
            scorecard=Scorecard(outcome=Outcome.FAIL, score=1.0, tags=["test"]),
            lineage=Lineage(parent_id=None, generation=0)
        )
    ]
    
    # Mock the fingerprint calculation
    with patch('valence.util.compute_failure_fingerprint', return_value="hash3"):
        new_failures = calculate_new_failures(current_evals, baseline_fingerprints)
    
    assert len(new_failures) == 1
    assert new_failures[0].id == "test1"


def test_calculate_metamorphic_consistency_score() -> None:
    """Test metamorphic consistency calculation."""
    evals = [
        EvalRecord(
            id="test1",
            prompt="original",
            model="test",
            scorecard=Scorecard(outcome=Outcome.PASS, score=0.0, tags=[]),
            lineage=Lineage(parent_id=None, generation=0)
        ),
        EvalRecord(
            id="test1.c1", 
            prompt="mutated",
            model="test",
            scorecard=Scorecard(outcome=Outcome.PASS, score=0.0, tags=[]),
            lineage=Lineage(parent_id="test1", generation=1)
        )
    ]
    
    consistency = calculate_metamorphic_consistency_score(evals)
    assert consistency == 1.0  # Both pass, so consistent