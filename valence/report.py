"""HTML report generation for evaluation results."""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from jinja2 import Template

from valence.schema import EvalRecord, Outcome, RunMetadata
from valence.storage import RunStorage, StorageError
from valence.util import compute_failure_fingerprint

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Valence Evaluation Report - {{ run_id }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .pass-rate {
            color: {% if pass_rate > 70 %}#10b981{% elif pass_rate > 40 %}#f59e0b{% else %}#ef4444{% endif %};
        }
        .family {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .family-header {
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }
        .family-tree {
            margin-left: 20px;
        }
        .eval-item {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f3f4f6;
        }
        .eval-item:last-child {
            border-bottom: none;
        }
        .status-icon {
            width: 24px;
            height: 24px;
            margin-right: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-weight: bold;
        }
        .status-pass {
            background: #d1fae5;
            color: #065f46;
        }
        .status-fail {
            background: #fee2e2;
            color: #991b1b;
        }
        .status-error {
            background: #fef3c7;
            color: #92400e;
        }
        .prompt-text {
            flex: 1;
            font-family: monospace;
            font-size: 0.9em;
            color: #4b5563;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .mutation-tag {
            background: #e5e7eb;
            color: #374151;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .insights {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .insights h2 {
            color: #4b5563;
            margin-bottom: 15px;
        }
        .insight-item {
            padding: 10px;
            margin: 10px 0;
            background: #f9fafb;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }
        table {
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        th {
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #374151;
        }
        td {
            padding: 12px;
            border-top: 1px solid #e5e7eb;
        }
        tr:hover {
            background: #f9fafb;
        }
        .detector-hits {
            margin: 20px 0;
        }
        .timing-info {
            margin: 20px 0;
        }
        footer {
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Valence Evaluation Report</h1>
        <div>Model: <strong>{{ model }}</strong> | Run ID: <strong>{{ run_id }}</strong></div>
        <div>{{ timestamp }}</div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-label">Total Evaluations</div>
            <div class="stat-value">{{ total }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Pass Rate</div>
            <div class="stat-value pass-rate">{{ "%.1f"|format(pass_rate) }}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Failures</div>
            <div class="stat-value" style="color: #ef4444;">{{ failures }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Passes</div>
            <div class="stat-value" style="color: #10b981;">{{ passes }}</div>
        </div>
    </div>

    {% if families %}
    <h2>Failure Families</h2>
    {% for family in families %}
    <div class="family">
        <div class="family-header">
            Parent: {{ family.parent.id }}
            <span class="status-icon status-{{ family.parent.status }}">
                {% if family.parent.status == 'pass' %}✓{% elif family.parent.status == 'fail' %}✗{% else %}!{% endif %}
            </span>
        </div>
        <div class="eval-item">
            <span class="prompt-text">{{ family.parent.prompt[:100] }}...</span>
        </div>
        {% if family.children %}
        <div class="family-tree">
            <div style="margin-top: 10px; font-weight: 600; color: #6b7280;">Mutations:</div>
            {% for child in family.children %}
            <div class="eval-item">
                <span class="status-icon status-{{ child.status }}">
                    {% if child.status == 'pass' %}✓{% elif child.status == 'fail' %}✗{% else %}!{% endif %}
                </span>
                <span class="prompt-text">{{ child.prompt[:80] }}...</span>
                <span class="mutation-tag">{{ child.operator }}</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}

    <div class="insights">
        <h2>Insights</h2>
        {% if fragile_prompts %}
        <div class="insight-item">
            <strong>Fragile Prompts:</strong> {{ fragile_prompts }} prompts failed but had passing mutations.
            These represent opportunities where simple reformulations succeed.
        </div>
        {% endif %}
        {% if persistent_failures %}
        <div class="insight-item">
            <strong>Persistent Failures:</strong> {{ persistent_failures }} prompts failed along with all mutations.
            These represent systematic detection patterns that resist simple bypasses.
        </div>
        {% endif %}
        <div class="insight-item">
            <strong>Mutation Effectiveness:</strong> {{ "%.1f"|format(mutation_success_rate) }}% of mutations changed the outcome.
        </div>
    </div>

    {% if detector_stats %}
    <div class="detector-hits">
        <h2>Top Detector Hits</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Hit Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {% for stat in detector_stats %}
                <tr>
                    <td>{{ stat.category }}</td>
                    <td>{{ stat.count }}</td>
                    <td>{{ "%.1f"|format(stat.percentage) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <div class="timing-info">
        <h2>Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Duration</td>
                    <td>{{ "%.2f"|format(total_duration_s) }} seconds</td>
                </tr>
                <tr>
                    <td>Average Eval Time</td>
                    <td>{{ "%.2f"|format(avg_eval_ms) }} ms</td>
                </tr>
                <tr>
                    <td>p95 Eval Time</td>
                    <td>{{ "%.2f"|format(p95_eval_ms) }} ms</td>
                </tr>
            </tbody>
        </table>
    </div>

    <footer>
        Generated by Valence v0.1.0
    </footer>
</body>
</html>
"""


class ReportError(Exception):
    """Report generation error."""
    pass


def build_families(evals: list[EvalRecord]) -> list[dict[str, Any]]:
    """Build family trees from evaluation records."""
    families = []
    parents = {}
    children = defaultdict(list)
    
    for record in evals:
        if record.lineage.parent_id:
            children[record.lineage.parent_id].append(record)
        else:
            parents[record.id] = record
    
    for parent_id, parent in parents.items():
        if parent.scorecard and parent.scorecard.outcome == Outcome.FAIL:
            family = {
                "parent": {
                    "id": parent.id,
                    "prompt": parent.prompt,
                    "status": parent.scorecard.outcome.value,
                },
                "children": [],
            }
            
            for child in children.get(parent_id, []):
                if child.scorecard:
                    family["children"].append({
                        "id": child.id,
                        "prompt": child.prompt,
                        "status": child.scorecard.outcome.value,
                        "operator": child.lineage.mutation_operator,
                    })
            
            families.append(family)
    
    return families


def calculate_insights(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate insights from failure families."""
    fragile = 0
    persistent = 0
    total_mutations = 0
    successful_mutations = 0
    
    for family in families:
        if not family["children"]:
            continue
        
        child_outcomes = [c["status"] for c in family["children"]]
        total_mutations += len(child_outcomes)
        
        passing_children = sum(1 for o in child_outcomes if o == "pass")
        successful_mutations += passing_children
        
        if passing_children > 0:
            fragile += 1
        elif all(o == "fail" for o in child_outcomes):
            persistent += 1
    
    mutation_success_rate = (
        (successful_mutations / total_mutations * 100) if total_mutations > 0 else 0
    )
    
    return {
        "fragile_prompts": fragile,
        "persistent_failures": persistent,
        "mutation_success_rate": mutation_success_rate,
    }


def calculate_detector_stats(evals: list[EvalRecord]) -> list[dict[str, Any]]:
    """Calculate detector hit statistics."""
    category_counts = Counter()
    total_failures = 0
    
    for record in evals:
        if record.scorecard and record.scorecard.outcome == Outcome.FAIL:
            total_failures += 1
            for tag in record.scorecard.tags:
                category_counts[tag] += 1
    
    stats = []
    for category, count in category_counts.most_common(10):
        percentage = (count / total_failures * 100) if total_failures > 0 else 0
        stats.append({
            "category": category,
            "count": count,
            "percentage": percentage,
        })
    
    return stats


def calculate_timing_stats(
    evals: list[EvalRecord], metadata: Optional[RunMetadata]
) -> dict[str, float]:
    """Calculate timing statistics."""
    durations = [e.duration_ms for e in evals if e.duration_ms is not None]
    
    if not durations:
        return {
            "total_duration_s": 0.0,
            "avg_eval_ms": 0.0,
            "p95_eval_ms": 0.0,
        }
    
    durations.sort()
    avg_ms = sum(durations) / len(durations)
    p95_index = int(len(durations) * 0.95)
    p95_ms = durations[min(p95_index, len(durations) - 1)]
    
    total_s = 0.0
    if metadata and metadata.started_at and metadata.completed_at:
        delta = metadata.completed_at - metadata.started_at
        total_s = delta.total_seconds()
    
    return {
        "total_duration_s": total_s,
        "avg_eval_ms": avg_ms,
        "p95_eval_ms": p95_ms,
    }


# Failure clustering and analytics functions

def cluster_failures_by_capability(evals: list[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    """Cluster failures by capability type."""
    clusters = {
        "retrieval": [],
        "ranking": [],
        "filtering": [],
        "parsing": [],
        "auth": [],
        "ux": [],
        "unknown": []
    }
    
    for eval_record in evals:
        if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.FAIL:
            tags = eval_record.scorecard.tags
            
            # Classify by capability
            if any(tag in ["search", "retrieval", "results"] for tag in tags):
                clusters["retrieval"].append(eval_record)
            elif any(tag in ["ranking", "top", "order"] for tag in tags):
                clusters["ranking"].append(eval_record)
            elif any(tag in ["filter", "constraint", "criteria"] for tag in tags):
                clusters["filtering"].append(eval_record)
            elif any(tag in ["json", "parse", "format"] for tag in tags):
                clusters["parsing"].append(eval_record)
            elif any(tag in ["auth", "permission", "access"] for tag in tags):
                clusters["auth"].append(eval_record)
            elif any(tag in ["ux", "interface", "usability"] for tag in tags):
                clusters["ux"].append(eval_record)
            else:
                clusters["unknown"].append(eval_record)
    
    return clusters


def calculate_failure_fingerprints(evals: list[EvalRecord]) -> Dict[str, int]:
    """Calculate failure fingerprint frequencies."""
    fingerprints = Counter()
    
    for eval_record in evals:
        if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.FAIL:
            tags = set(eval_record.scorecard.tags)
            fingerprint = compute_failure_fingerprint(
                eval_record.prompt,
                eval_record.response or "",
                tags
            )
            fingerprints[fingerprint] += 1
    
    return dict(fingerprints)


def calculate_metamorphic_consistency(evals: list[EvalRecord]) -> float:
    """Calculate metamorphic consistency score."""
    # Group by base prompt (before mutation)
    base_prompts = defaultdict(list)
    
    for eval_record in evals:
        # Find root prompt
        if eval_record.lineage.parent_id is None:
            base_id = eval_record.id
        else:
            # Walk up to find root
            base_id = eval_record.lineage.parent_id
            while '.' in base_id:
                base_id = base_id.rsplit('.', 1)[0]
        
        base_prompts[base_id].append(eval_record)
    
    # Calculate consistency within families
    consistent_families = 0
    total_families = 0
    
    for family_records in base_prompts.values():
        if len(family_records) > 1:
            total_families += 1
            outcomes = [r.scorecard.outcome for r in family_records if r.scorecard]
            
            # Check if outcomes are consistent (all pass or all fail)
            if outcomes and (all(o == Outcome.PASS for o in outcomes) or 
                           all(o == Outcome.FAIL for o in outcomes)):
                consistent_families += 1
    
    return consistent_families / total_families if total_families > 0 else 1.0


def calculate_detector_confusion_matrix(evals: list[EvalRecord]) -> Dict[str, Dict[str, int]]:
    """Calculate detector co-occurrence matrix."""
    # Get all unique detector tags
    all_tags = set()
    for eval_record in evals:
        if eval_record.scorecard:
            all_tags.update(eval_record.scorecard.tags)
    
    # Build co-occurrence matrix
    matrix = defaultdict(lambda: defaultdict(int))
    
    for eval_record in evals:
        if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.FAIL:
            tags = eval_record.scorecard.tags
            
            # Count co-occurrences
            for tag1 in tags:
                for tag2 in tags:
                    if tag1 != tag2:
                        matrix[tag1][tag2] += 1
    
    return dict(matrix)


def calculate_latency_percentiles(evals: list[EvalRecord]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    latencies = []
    
    for eval_record in evals:
        if eval_record.duration_ms:
            latencies.append(eval_record.duration_ms)
    
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0}
    
    latencies.sort()
    n = len(latencies)
    
    return {
        "p50": latencies[int(n * 0.5)],
        "p95": latencies[int(n * 0.95)],
        "p99": latencies[int(n * 0.99)]
    }


def generate_report(input_dir: Path, output_path: Path) -> None:
    """Generate HTML report from evaluation run."""
    storage = RunStorage(input_dir)
    
    try:
        metadata = storage.load_metadata()
        evals = storage.load_evals()
    except StorageError as e:
        raise ReportError(f"Failed to load evaluation data: {e}") from e
    
    if not metadata:
        raise ReportError("No metadata found in run directory")
    
    if not evals:
        raise ReportError("No evaluation records found")
    
    families = build_families(evals)
    insights = calculate_insights(families)
    detector_stats = calculate_detector_stats(evals)
    timing_stats = calculate_timing_stats(evals, metadata)
    
    # Enhanced analytics
    failure_clusters = cluster_failures_by_capability(evals)
    failure_fingerprints = calculate_failure_fingerprints(evals)
    metamorphic_consistency = calculate_metamorphic_consistency(evals)
    detector_confusion = calculate_detector_confusion_matrix(evals)
    latency_percentiles = calculate_latency_percentiles(evals)
    
    pass_rate = (
        (metadata.total_passes / metadata.total_prompts * 100)
        if metadata.total_prompts > 0
        else 0
    )
    
    timestamp = (
        metadata.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if metadata.completed_at
        else "In Progress"
    )
    
    template = Template(HTML_TEMPLATE)
    html = template.render(
        run_id=metadata.run_id,
        model=metadata.model,
        timestamp=timestamp,
        total=metadata.total_prompts,
        passes=metadata.total_passes,
        failures=metadata.total_failures,
        pass_rate=pass_rate,
        families=families,
        detector_stats=detector_stats,
        **insights,
        **timing_stats,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    
    logger.info(f"Report generated: {output_path}")