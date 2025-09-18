"""CI integration utilities."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from valence.schema import EvalRecord, Outcome
from valence.storage import RunStorage

logger = logging.getLogger(__name__)


class CIGateError(Exception):
    """CI gate violation error."""
    pass


class CIMetrics:
    """CI metrics and thresholds."""
    
    def __init__(self):
        self.new_failure_threshold = 5
        self.metamorphic_consistency_threshold = 0.8
        self.p95_latency_slo_ms = 2000
        self.pass_rate_threshold = 0.85
    
    @classmethod
    def from_config(cls, config_path: Path) -> "CIMetrics":
        """Load CI metrics from configuration file."""
        instance = cls()
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                instance.new_failure_threshold = config.get("new_failure_threshold", 5)
                instance.metamorphic_consistency_threshold = config.get("metamorphic_consistency_threshold", 0.8)
                instance.p95_latency_slo_ms = config.get("p95_latency_slo_ms", 2000)
                instance.pass_rate_threshold = config.get("pass_rate_threshold", 0.85)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid CI config: {e}, using defaults")
        
        return instance


def calculate_new_failures(
    current_evals: List[EvalRecord],
    baseline_fingerprints: set[str]
) -> List[EvalRecord]:
    """
    Identify new failure fingerprints not seen in baseline.
    """
    from valence.util import compute_failure_fingerprint
    
    new_failures = []
    
    for eval_record in current_evals:
        if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.FAIL:
            tags = set(eval_record.scorecard.tags)
            fingerprint = compute_failure_fingerprint(
                eval_record.prompt,
                eval_record.response or "",
                tags
            )
            
            if fingerprint not in baseline_fingerprints:
                new_failures.append(eval_record)
    
    return new_failures


def calculate_metamorphic_consistency_score(evals: List[EvalRecord]) -> float:
    """
    Calculate metamorphic consistency score for CI gates.
    """
    from collections import defaultdict
    
    # Group by base prompt family
    families = defaultdict(list)
    
    for eval_record in evals:
        # Find root prompt ID
        if eval_record.lineage.parent_id is None:
            root_id = eval_record.id
        else:
            root_id = eval_record.lineage.parent_id
            while '.' in root_id:
                root_id = root_id.rsplit('.', 1)[0]
        
        families[root_id].append(eval_record)
    
    # Calculate consistency
    consistent_families = 0
    total_families_with_mutations = 0
    
    for family_records in families.values():
        if len(family_records) > 1:  # Has mutations
            total_families_with_mutations += 1
            outcomes = [r.scorecard.outcome for r in family_records if r.scorecard]
            
            # Consistent if all pass or all fail
            if outcomes and (all(o == Outcome.PASS for o in outcomes) or 
                           all(o == Outcome.FAIL for o in outcomes)):
                consistent_families += 1
    
    return (consistent_families / total_families_with_mutations 
            if total_families_with_mutations > 0 else 1.0)


def calculate_p95_latency(evals: List[EvalRecord]) -> float:
    """Calculate 95th percentile latency."""
    latencies = []
    
    for eval_record in evals:
        if eval_record.duration_ms:
            latencies.append(eval_record.duration_ms)
    
    if not latencies:
        return 0.0
    
    latencies.sort()
    idx = int(len(latencies) * 0.95)
    return latencies[idx]


def check_ci_gates(
    run_dir: Path,
    baseline_dir: Optional[Path] = None,
    config_path: Optional[Path] = None
) -> Dict[str, any]:
    """
    Check CI gates and return results.
    """
    # Load configuration
    metrics = CIMetrics.from_config(config_path) if config_path else CIMetrics()
    
    # Load current run
    storage = RunStorage(run_dir)
    metadata = storage.load_metadata()
    evals = storage.load_evals()
    
    if not metadata or not evals:
        raise CIGateError("No evaluation data found")
    
    results = {
        "gates": {},
        "summary": {"passed": True, "violations": []},
        "metrics": {}
    }
    
    # Gate 1: New failure fingerprints
    if baseline_dir and baseline_dir.exists():
        baseline_storage = RunStorage(baseline_dir)
        baseline_evals = baseline_storage.load_evals()
        
        # Get baseline fingerprints
        from valence.util import compute_failure_fingerprint
        baseline_fingerprints = set()
        
        for eval_record in baseline_evals:
            if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.FAIL:
                tags = set(eval_record.scorecard.tags)
                fingerprint = compute_failure_fingerprint(
                    eval_record.prompt,
                    eval_record.response or "",
                    tags
                )
                baseline_fingerprints.add(fingerprint)
        
        new_failures = calculate_new_failures(evals, baseline_fingerprints)
        new_failure_count = len(new_failures)
        
        results["gates"]["new_failures"] = {
            "count": new_failure_count,
            "threshold": metrics.new_failure_threshold,
            "passed": new_failure_count <= metrics.new_failure_threshold,
            "details": [{"id": f.id, "prompt": f.prompt[:100]} for f in new_failures[:5]]
        }
        
        results["metrics"]["new_failure_count"] = new_failure_count
        
        if new_failure_count > metrics.new_failure_threshold:
            results["summary"]["passed"] = False
            results["summary"]["violations"].append(f"New failures: {new_failure_count} > {metrics.new_failure_threshold}")
    
    # Gate 2: Metamorphic consistency
    consistency_score = calculate_metamorphic_consistency_score(evals)
    
    results["gates"]["metamorphic_consistency"] = {
        "score": consistency_score,
        "threshold": metrics.metamorphic_consistency_threshold,
        "passed": consistency_score >= metrics.metamorphic_consistency_threshold
    }
    
    results["metrics"]["metamorphic_consistency"] = consistency_score
    
    if consistency_score < metrics.metamorphic_consistency_threshold:
        results["summary"]["passed"] = False
        results["summary"]["violations"].append(f"Consistency: {consistency_score:.3f} < {metrics.metamorphic_consistency_threshold}")
    
    # Gate 3: p95 latency SLO
    p95_latency = calculate_p95_latency(evals)
    
    results["gates"]["p95_latency"] = {
        "latency_ms": p95_latency,
        "slo_ms": metrics.p95_latency_slo_ms,
        "passed": p95_latency <= metrics.p95_latency_slo_ms
    }
    
    results["metrics"]["p95_latency_ms"] = p95_latency
    
    if p95_latency > metrics.p95_latency_slo_ms:
        results["summary"]["passed"] = False
        results["summary"]["violations"].append(f"p95 latency: {p95_latency:.1f}ms > {metrics.p95_latency_slo_ms}ms")
    
    # Gate 4: Overall pass rate
    total_prompts = metadata.total_prompts
    pass_rate = metadata.total_passes / total_prompts if total_prompts > 0 else 0
    
    results["gates"]["pass_rate"] = {
        "rate": pass_rate,
        "threshold": metrics.pass_rate_threshold,
        "passed": pass_rate >= metrics.pass_rate_threshold
    }
    
    results["metrics"]["pass_rate"] = pass_rate
    
    if pass_rate < metrics.pass_rate_threshold:
        results["summary"]["passed"] = False
        results["summary"]["violations"].append(f"Pass rate: {pass_rate:.3f} < {metrics.pass_rate_threshold}")
    
    return results


def generate_ci_report(gate_results: Dict[str, any], output_path: Path) -> None:
    """Generate CI-friendly report."""
    with open(output_path, 'w') as f:
        # Summary
        status = "PASS" if gate_results["summary"]["passed"] else "FAIL"
        f.write(f"CI Gate Status: {status}\n")
        f.write("=" * 40 + "\n\n")
        
        # Violations
        if gate_results["summary"]["violations"]:
            f.write("VIOLATIONS:\n")
            for violation in gate_results["summary"]["violations"]:
                f.write(f"❌ {violation}\n")
            f.write("\n")
        
        # Gate details
        f.write("GATE DETAILS:\n")
        for gate_name, gate_data in gate_results["gates"].items():
            status_icon = "✅" if gate_data["passed"] else "❌"
            f.write(f"{status_icon} {gate_name.replace('_', ' ').title()}\n")
            
            if gate_name == "new_failures":
                f.write(f"   Count: {gate_data['count']} (threshold: {gate_data['threshold']})\n")
            elif gate_name == "metamorphic_consistency":
                f.write(f"   Score: {gate_data['score']:.3f} (threshold: {gate_data['threshold']})\n")
            elif gate_name == "p95_latency":
                f.write(f"   Latency: {gate_data['latency_ms']:.1f}ms (SLO: {gate_data['slo_ms']}ms)\n")
            elif gate_name == "pass_rate":
                f.write(f"   Rate: {gate_data['rate']:.3f} (threshold: {gate_data['threshold']})\n")
            
            f.write("\n")
        
        # Metrics summary
        f.write("METRICS:\n")
        for metric_name, value in gate_results["metrics"].items():
            if isinstance(value, float):
                f.write(f"  {metric_name}: {value:.3f}\n")
            else:
                f.write(f"  {metric_name}: {value}\n")


def auto_triage_failures(
    new_failures: List[EvalRecord],
    output_path: Optional[Path] = None
) -> List[Dict[str, str]]:
    """Auto-triage new failures for ticket creation."""
    tickets = []
    
    for failure in new_failures:
        ticket = {
            "title": f"Evaluation failure: {failure.id}",
            "description": f"""
**Prompt:** {failure.prompt}

**Response:** {failure.response[:500] if failure.response else 'None'}{'...' if failure.response and len(failure.response) > 500 else ''}

**Failed Detectors:** {', '.join(failure.scorecard.tags) if failure.scorecard else 'None'}

**Failure Score:** {failure.scorecard.score if failure.scorecard else 'N/A'}

**Mutation Info:**
- Parent: {failure.lineage.parent_id or 'None'}
- Generation: {failure.lineage.generation}
- Mutation Op: {failure.lineage.mutation_operator or 'None'}

**Raw I/O:**
```
Input: {failure.prompt}
Output: {failure.response or 'None'}
```
""",
            "labels": failure.scorecard.tags if failure.scorecard else [],
            "priority": "high" if failure.scorecard and failure.scorecard.score > 0.8 else "medium"
        }
        
        tickets.append(ticket)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(tickets, f, indent=2)
    
    return tickets


def ci_main(args: List[str]) -> int:
    """
    Main entry point for CI integration.
    """
    if len(args) < 2:
        print("Usage: valence ci <run_dir> [baseline_dir] [config_path]", file=sys.stderr)
        return 1
    
    run_dir = Path(args[1])
    baseline_dir = Path(args[2]) if len(args) > 2 else None
    config_path = Path(args[3]) if len(args) > 3 else None
    
    try:
        results = check_ci_gates(run_dir, baseline_dir, config_path)
        
        # Generate report
        report_path = run_dir / "ci_report.txt"
        generate_ci_report(results, report_path)
        
        # Auto-triage if there are new failures
        if "new_failures" in results["gates"] and results["gates"]["new_failures"]["count"] > 0:
            # This would need the actual failure records, simplified for now
            print(f"Found {results['gates']['new_failures']['count']} new failures")
        
        # Print summary
        if results["summary"]["passed"]:
            print("✅ All CI gates passed")
            return 0
        else:
            print("❌ CI gate violations found:")
            for violation in results["summary"]["violations"]:
                print(f"   {violation}")
            return 1
            
    except Exception as e:
        print(f"CI check failed: {e}", file=sys.stderr)
        return 2