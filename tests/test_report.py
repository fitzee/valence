"""Tests for report generation."""

from datetime import datetime
from pathlib import Path

import pytest

from valence.report import generate_report
from valence.schema import (
    EvalRecord,
    Lineage,
    Outcome,
    RunMetadata,
    Scorecard,
    Signal,
    DetectorType,
)
from valence.storage import RunStorage


def test_generate_report_integration(tmp_path: Path) -> None:
    """Integration test for report generation."""
    output_dir = tmp_path / "run"
    storage = RunStorage(output_dir)
    
    metadata = RunMetadata(
        run_id="test-run",
        model="stub",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        total_prompts=3,
        total_passes=1,
        total_failures=2,
        total_errors=0,
        max_generations=1,
        mutations_per_failure=2
    )
    storage.save_metadata(metadata)
    
    evals = [
        EvalRecord(
            id="parent-1",
            prompt="Dangerous prompt",
            response="dangerous response",
            scorecard=Scorecard(
                outcome=Outcome.STRONG,
                score=1.0,
                tags=["security"],
                signals=[Signal(
                    detector_type=DetectorType.KEYWORD,
                    category="security",
                    score=1.0,
                    matched=["dangerous"]
                )]
            ),
            model="stub",
            duration_ms=15.0
        ),
        EvalRecord(
            id="parent-1.c1",
            prompt="Modified prompt",
            response="safe response",
            scorecard=Scorecard(
                outcome=Outcome.WEAK,
                score=0.0,
                tags=[],
                signals=[]
            ),
            model="stub",
            lineage=Lineage(
                parent_id="parent-1",
                mutation_operator="plain-english",
                generation=1
            ),
            duration_ms=12.0
        ),
        EvalRecord(
            id="safe-1",
            prompt="Safe prompt",
            response="safe response",
            scorecard=Scorecard(
                outcome=Outcome.WEAK,
                score=0.0,
                tags=[],
                signals=[]
            ),
            model="stub",
            duration_ms=10.0
        ),
    ]
    
    for eval_record in evals:
        storage.append_eval(eval_record)
    
    report_path = tmp_path / "report.html"
    generate_report(output_dir, report_path)
    
    assert report_path.exists()
    html = report_path.read_text()
    
    # Check key content is present
    assert "test-run" in html
    assert "stub" in html
    assert "parent-1" in html
    assert "plain-english" in html
    assert "security" in html
    
    # Check for colored scale outcomes
    assert "weak" in html or "mild" in html or "strong" in html
    
    # Check self-contained HTML structure
    assert "<style>" in html
    assert "<script>" in html
    assert "<!DOCTYPE html>" in html