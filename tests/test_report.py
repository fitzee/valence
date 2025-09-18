"""Tests for report generation."""

from datetime import datetime
from pathlib import Path

import pytest

from valence.report import (
    build_families,
    calculate_detector_stats,
    calculate_insights,
    calculate_timing_stats,
    generate_report,
)
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


def test_build_families() -> None:
    """Test building failure families."""
    evals = [
        EvalRecord(
            id="parent-1",
            prompt="Parent prompt",
            response="fail",
            scorecard=Scorecard(
                outcome=Outcome.FAIL,
                score=1.0,
                tags=["bad"],
                signals=[]
            ),
            model="stub",
            lineage=Lineage()
        ),
        EvalRecord(
            id="parent-1.c1",
            prompt="Child prompt 1",
            response="pass",
            scorecard=Scorecard(
                outcome=Outcome.PASS,
                score=0.0,
                tags=[],
                signals=[]
            ),
            model="stub",
            lineage=Lineage(
                parent_id="parent-1",
                mutation_operator="plain-english",
                generation=1
            )
        ),
        EvalRecord(
            id="parent-1.c2",
            prompt="Child prompt 2",
            response="fail",
            scorecard=Scorecard(
                outcome=Outcome.FAIL,
                score=1.0,
                tags=["bad"],
                signals=[]
            ),
            model="stub",
            lineage=Lineage(
                parent_id="parent-1",
                mutation_operator="role-expert",
                generation=1
            )
        ),
    ]
    
    families = build_families(evals)
    assert len(families) == 1
    
    family = families[0]
    assert family["parent"]["id"] == "parent-1"
    assert family["parent"]["status"] == "fail"
    assert len(family["children"]) == 2
    
    child_statuses = {c["id"]: c["status"] for c in family["children"]}
    assert child_statuses["parent-1.c1"] == "pass"
    assert child_statuses["parent-1.c2"] == "fail"


def test_calculate_insights() -> None:
    """Test calculating insights from families."""
    families = [
        {
            "parent": {"id": "p1", "prompt": "p1", "status": "fail"},
            "children": [
                {"id": "p1.c1", "prompt": "c1", "status": "pass", "operator": "op1"},
                {"id": "p1.c2", "prompt": "c2", "status": "fail", "operator": "op2"},
            ]
        },
        {
            "parent": {"id": "p2", "prompt": "p2", "status": "fail"},
            "children": [
                {"id": "p2.c1", "prompt": "c1", "status": "fail", "operator": "op1"},
                {"id": "p2.c2", "prompt": "c2", "status": "fail", "operator": "op2"},
            ]
        },
        {
            "parent": {"id": "p3", "prompt": "p3", "status": "fail"},
            "children": []
        }
    ]
    
    insights = calculate_insights(families)
    assert insights["fragile_prompts"] == 1
    assert insights["persistent_failures"] == 1
    assert insights["mutation_success_rate"] == 25.0


def test_calculate_detector_stats() -> None:
    """Test calculating detector statistics."""
    evals = [
        EvalRecord(
            id="e1",
            prompt="p1",
            response="r1",
            scorecard=Scorecard(
                outcome=Outcome.FAIL,
                score=1.0,
                tags=["security", "malicious"],
                signals=[]
            ),
            model="stub"
        ),
        EvalRecord(
            id="e2",
            prompt="p2",
            response="r2",
            scorecard=Scorecard(
                outcome=Outcome.FAIL,
                score=1.0,
                tags=["security"],
                signals=[]
            ),
            model="stub"
        ),
        EvalRecord(
            id="e3",
            prompt="p3",
            response="r3",
            scorecard=Scorecard(
                outcome=Outcome.PASS,
                score=0.0,
                tags=[],
                signals=[]
            ),
            model="stub"
        ),
    ]
    
    stats = calculate_detector_stats(evals)
    assert len(stats) == 2
    
    stat_dict = {s["category"]: s for s in stats}
    assert stat_dict["security"]["count"] == 2
    assert stat_dict["security"]["percentage"] == 100.0
    assert stat_dict["malicious"]["count"] == 1
    assert stat_dict["malicious"]["percentage"] == 50.0


def test_calculate_timing_stats() -> None:
    """Test calculating timing statistics."""
    evals = [
        EvalRecord(
            id="e1",
            prompt="p1",
            response="r1",
            scorecard=None,
            model="stub",
            duration_ms=10.0
        ),
        EvalRecord(
            id="e2",
            prompt="p2",
            response="r2",
            scorecard=None,
            model="stub",
            duration_ms=20.0
        ),
        EvalRecord(
            id="e3",
            prompt="p3",
            response="r3",
            scorecard=None,
            model="stub",
            duration_ms=100.0
        ),
    ]
    
    metadata = RunMetadata(
        run_id="test",
        model="stub",
        started_at=datetime(2024, 1, 1, 0, 0, 0),
        completed_at=datetime(2024, 1, 1, 0, 0, 10),
        max_generations=1,
        mutations_per_failure=4
    )
    
    stats = calculate_timing_stats(evals, metadata)
    assert stats["avg_eval_ms"] == pytest.approx(43.33, rel=0.01)
    assert stats["p95_eval_ms"] == 100.0
    assert stats["total_duration_s"] == 10.0


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
                outcome=Outcome.FAIL,
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
                outcome=Outcome.PASS,
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
                outcome=Outcome.PASS,
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
    
    assert "test-run" in html
    assert "stub" in html
    assert "33.3%" in html or "33.33%" in html
    assert "parent-1" in html
    assert "plain-english" in html
    assert "security" in html