"""HTML report generation for evaluation results."""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from valence.schema import EvalRecord, Outcome, RunMetadata
from valence.storage import RunStorage

logger = logging.getLogger(__name__)


def generate_report(run_dir: Path, output_path: Path) -> None:
    """Generate HTML report from evaluation results."""
    try:
        storage = RunStorage(run_dir)
        evals = storage.load_evals()
        metadata = storage.load_metadata()
        
        if not evals:
            raise ValueError("No evaluation records found")
        
        # Build data structures
        families = build_compact_families(evals)
        insights = calculate_compact_insights(families, evals)
        score_analysis = analyze_score_distributions(evals)
        
        # Generate report data
        # Calculate corrected totals based on 1.0 threshold
        total_strong = sum(1 for e in evals if e.scorecard and e.scorecard.score > 0.66)
        total_weak = sum(1 for e in evals if e.scorecard and e.scorecard.score <= 0.33)
        
        report_data = {
            "run_id": metadata.run_id if metadata else "unknown",
            "model": metadata.model if metadata else "unknown",
            "total_prompts": len(evals),
            "total_strong": total_strong,
            "total_weak": total_weak,
            "families": families,
            "insights": insights,
            "score_analysis": score_analysis,
            "evals": [eval_to_compact_dict(e) for e in evals],
        }
        
        # Render template
        template = Template(COMPACT_HTML_TEMPLATE)
        html_content = template.render(**report_data)
        
        # Write output
        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Report generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


def build_compact_families(evals: List[EvalRecord]) -> List[Dict[str, Any]]:
    """Build family structures focused on mutations and scores."""
    parents = {}
    children = defaultdict(list)
    all_records = {r.id: r for r in evals}
    
    for record in evals:
        if record.lineage.parent_id:
            children[record.lineage.parent_id].append(record)
        else:
            parents[record.id] = record
    
    def build_mutation_tree(record_id: str, depth: int = 0) -> List[Dict[str, Any]]:
        """Recursively build mutation tree for all generations."""
        mutations = []
        if record_id in children:
            for child in children[record_id]:
                if child.scorecard:
                    # Determine outcome based on score ranges
                    if child.scorecard.score <= 0.33:
                        child_corrected_outcome = "weak"
                    elif child.scorecard.score <= 0.66:
                        child_corrected_outcome = "mild"
                    else:
                        child_corrected_outcome = "strong"
                    
                    # Build signals list for mutation
                    child_signals = []
                    if child.scorecard:
                        for signal in child.scorecard.signals:
                            child_signals.append({
                                "category": signal.category,
                                "score": signal.score,
                                "details": str(signal.details) if signal.details else ""
                            })
                    
                    # Recursively get children of this child
                    sub_mutations = build_mutation_tree(child.id, depth + 1)
                    
                    mutations.append({
                        "id": child.id,
                        "operator": child.lineage.mutation_operator,
                        "status": child_corrected_outcome,
                        "score": child.scorecard.score,
                        "prompt": child.prompt,
                        "response": child.response or "",
                        "duration_ms": child.duration_ms,
                        "signals": child_signals,
                        "generation": child.lineage.generation,
                        "children": sub_mutations,  # Add nested children
                    })
        return mutations
    
    families = []
    for parent_id, parent in parents.items():
        # Get all mutations recursively
        mutations = build_mutation_tree(parent_id)
        
        # Determine parent outcome based on score ranges
        if parent.scorecard and parent.scorecard.score <= 0.33:
            parent_corrected_outcome = "weak"
        elif parent.scorecard and parent.scorecard.score <= 0.66:
            parent_corrected_outcome = "mild"
        elif parent.scorecard:
            parent_corrected_outcome = "strong"
        else:
            parent_corrected_outcome = "weak"
        
        # Build signals list for parent
        parent_signals = []
        if parent.scorecard:
            for signal in parent.scorecard.signals:
                parent_signals.append({
                    "category": signal.category,
                    "score": signal.score,
                    "details": str(signal.details) if signal.details else ""
                })
        
        family = {
            "parent_id": parent_id,
            "parent_status": parent_corrected_outcome,
            "parent_score": parent.scorecard.score if parent.scorecard else 0.0,
            "parent_prompt": parent.prompt,
            "parent_response": parent.response or "",
            "parent_signals": parent_signals,
            "mutations": mutations,
            "mutation_count": len(mutations),
        }
        families.append(family)
    
    return families


def analyze_score_distributions(evals: List[EvalRecord]) -> Dict[str, Any]:
    """Analyze score distributions to show distance from thresholds."""
    scores = []
    weak_scores = []
    strong_scores = []
    
    for eval_record in evals:
        if eval_record.scorecard:
            score = eval_record.scorecard.score
            scores.append(score)
            
            if score <= 0.33:
                weak_scores.append(score)
            elif score > 0.66:
                strong_scores.append(score)
    
    # Calculate statistics
    analysis = {
        "total_evals": len(scores),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "weak_threshold": 0.33,
        "strong_threshold": 0.66,
        "close_calls": [],  # Scores within 0.1 of threshold
        "score_ranges": {
            "0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0, "0.4-0.5": 0,
            "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0
        }
    }
    
    # Count score ranges
    for score in scores:
        range_key = f"{int(score*10)/10:.1f}-{int(score*10)/10 + 0.1:.1f}"
        if range_key in analysis["score_ranges"]:
            analysis["score_ranges"][range_key] += 1
    
    # Find close calls (within 0.1 of thresholds)
    for eval_record in evals:
        if eval_record.scorecard:
            score = eval_record.scorecard.score
            # Check distance from both thresholds
            distance_from_weak = abs(score - 0.33)
            distance_from_strong = abs(score - 0.66)
            min_distance = min(distance_from_weak, distance_from_strong)
            
            if min_distance <= 0.1:
                analysis["close_calls"].append({
                    "id": eval_record.id,
                    "score": score,
                    "distance": min_distance,
                    "outcome": eval_record.scorecard.outcome.value
                })
    
    return analysis


def calculate_compact_insights(families: List[Dict[str, Any]], evals: List[EvalRecord]) -> Dict[str, Any]:
    """Calculate focused insights without bloat."""
    total_mutations = sum(f["mutation_count"] for f in families)
    successful_mutations = 0
    mutation_operators = Counter()
    detector_hits = Counter()
    
    for family in families:
        for mutation in family["mutations"]:
            mutation_operators[mutation["operator"]] += 1
            if mutation["status"] == "weak":
                successful_mutations += 1
    
    for eval_record in evals:
        if eval_record.scorecard and eval_record.scorecard.outcome == Outcome.STRONG:
            for signal in eval_record.scorecard.signals:
                if signal.score > 0.5:
                    detector_hits[signal.category] += 1
    
    return {
        "total_mutations": total_mutations,
        "successful_mutations": successful_mutations,
        "mutation_success_rate": (successful_mutations / total_mutations * 100) if total_mutations > 0 else 0,
        "mutation_operators": dict(mutation_operators),
        "top_detectors": dict(detector_hits.most_common(5)),
        "families_with_mutations": sum(1 for f in families if f["mutation_count"] > 0),
    }


def eval_to_compact_dict(eval_record: EvalRecord) -> Dict[str, Any]:
    """Convert evaluation to compact dictionary."""
    signals = []
    if eval_record.scorecard:
        for signal in eval_record.scorecard.signals:
            signals.append({
                "category": signal.category,
                "score": signal.score,
                "details": str(signal.details) if signal.details else ""
            })
    
    # Determine outcome based on score ranges
    corrected_outcome = None
    if eval_record.scorecard:
        if eval_record.scorecard.score <= 0.33:
            corrected_outcome = "weak"
        elif eval_record.scorecard.score <= 0.66:
            corrected_outcome = "mild"
        else:
            corrected_outcome = "strong"
    
    return {
        "id": eval_record.id,
        "prompt": eval_record.prompt[:200] + "..." if len(eval_record.prompt) > 200 else eval_record.prompt,
        "response": eval_record.response[:300] + "..." if eval_record.response and len(eval_record.response) > 300 else eval_record.response,
        "outcome": corrected_outcome,
        "score": eval_record.scorecard.score if eval_record.scorecard else None,
        "signals": signals,
        "parent_id": eval_record.lineage.parent_id,
        "mutation_operator": eval_record.lineage.mutation_operator,
        "duration_ms": eval_record.duration_ms,
    }


# Compact HTML template without emojis and minimal whitespace
COMPACT_HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Valence Report - {{ run_id }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px; line-height: 1.4; color: #333; background: #fafafa; padding: 10px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header { background: #2563eb; color: white; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header-meta { font-size: 13px; opacity: 0.9; }
        
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 15px; }
        .stat { background: white; padding: 15px; border-radius: 4px; border-left: 3px solid #2563eb; }
        .stat-number { font-size: 22px; font-weight: 700; color: #2563eb; }
        .stat-label { font-size: 12px; color: #666; margin-top: 2px; }
        
        .section { background: white; margin-bottom: 15px; border-radius: 4px; }
        .section-header { background: #f8fafc; padding: 10px 15px; border-bottom: 1px solid #e5e7eb; font-weight: 600; }
        .section-content { padding: 15px; }
        
        .table { width: 100%; border-collapse: collapse; }
        .table th, .table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .table th { background: #f8fafc; font-weight: 600; font-size: 13px; }
        .table td { font-size: 13px; }
        
        .score { padding: 2px 6px; border-radius: 3px; font-weight: 600; font-size: 12px; }
        .score-weak { background: #dcfce7; color: #166534; }
        .score-mild { background: #fef3c7; color: #92400e; }
        .score-strong { background: #fee2e2; color: #991b1b; }
        
        .prompt-text { max-width: 400px; word-break: break-word; }
        .response-text { max-width: 300px; word-break: break-word; color: #4b5563; }
        
        .mutation-list { list-style: none; }
        .mutation-item { padding: 5px 0; border-bottom: 1px solid #f3f4f6; }
        .mutation-item:last-child { border-bottom: none; }
        
        .insight-list { list-style: none; }
        .insight-item { padding: 8px; margin-bottom: 5px; background: #f8fafc; border-radius: 3px; border-left: 3px solid #3b82f6; }
        
        .tabs { display: flex; background: white; border-radius: 4px; margin-bottom: 15px; }
        .tab { flex: 1; padding: 12px; cursor: pointer; border: none; background: none; font-size: 14px; }
        .tab.active { background: #2563eb; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .score-bar { height: 4px; background: #e5e7eb; border-radius: 2px; position: relative; margin: 4px 0; }
        .score-fill { height: 100%; border-radius: 2px; }
        .score-fill.weak { background: #10b981; }
        .score-fill.mild { background: #f59e0b; }
        .score-fill.strong { background: #ef4444; }
        
        .compact-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .compact-grid.three { grid-template-columns: 1fr 1fr 1fr; }
        
        /* Tooltip styling */
        .tooltip { position: relative; cursor: help; }
        .tooltip-content {
            visibility: hidden; opacity: 0; position: absolute; z-index: 1000;
            bottom: 125%; left: 50%; transform: translateX(-50%);
            background: white; color: #333; padding: 8px 12px;
            border: 2px solid #333; border-radius: 6px; font-size: 12px; white-space: nowrap;
            transition: opacity 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .tooltip-content::after {
            content: ''; position: absolute; top: 100%; left: 50%;
            transform: translateX(-50%); border: 5px solid transparent;
            border-top-color: #333;
        }
        .tooltip:hover .tooltip-content { visibility: visible; opacity: 1; }
        .signal-item { display: block; margin: 2px 0; }
        .signal-weak { color: #166534; }
        .signal-mild { color: #92400e; }
        .signal-strong { color: #991b1b; }
        
        .family-group { border: 1px solid #e5e7eb; border-radius: 4px; margin-bottom: 10px; }
        .family-header { 
            background: #f8fafc; padding: 10px 15px; cursor: pointer; 
            border-bottom: 1px solid #e5e7eb; font-weight: 600;
            display: flex; align-items: center; gap: 10px;
        }
        .family-header:hover { background: #f1f5f9; }
        .toggle-icon { font-size: 12px; transition: transform 0.2s; }
        .toggle-icon.collapsed { transform: rotate(-90deg); }
        .mutation-count { font-size: 12px; color: #6b7280; font-weight: normal; }
        
        .family-content { display: block; }
        .family-content.collapsed { display: none; }
        
        .eval-row { 
            display: grid; 
            grid-template-columns: 200px 1fr 1fr 120px 100px; 
            gap: 15px; padding: 8px 15px; 
            border-bottom: 1px solid #f3f4f6; 
            align-items: start;
        }
        .eval-row:last-child { border-bottom: none; }
        .parent-row { background: #fefefe; font-weight: 500; }
        .child-row { background: #fafbfc; margin-left: 10px; }
        
        .eval-id { font-size: 13px; font-weight: 600; }
        .mutation-id { font-size: 12px; }
        .mutation-indent { color: #9ca3af; margin-right: 5px; }
        .mutation-operator { 
            font-size: 10px; color: #6b7280; background: #f3f4f6; 
            padding: 2px 4px; border-radius: 2px; margin-top: 2px; display: inline-block;
        }
        
        .eval-prompt, .eval-response { font-size: 12px; word-break: break-word; }
        .eval-response { color: #6b7280; }
        .eval-score { text-align: center; }
        .eval-outcome { text-align: center; }
        .duration { font-size: 10px; color: #9ca3af; margin-top: 2px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Valence Report</h1>
            <div class="header-meta">
                Run: {{ run_id }} | Model: {{ model }} | Prompts: {{ total_prompts }}
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-number">{{ total_prompts }}</div>
                <div class="stat-label">Total Prompts</div>
            </div>
            <div class="stat">
                <div class="stat-number">{{ total_strong }}</div>
                <div class="stat-label">Strong</div>
            </div>
            <div class="stat">
                <div class="stat-number">{{ total_weak }}</div>
                <div class="stat-label">Weak</div>
            </div>
            <div class="stat">
                <div class="stat-number">{{ "%.1f"|format((total_weak / total_prompts * 100) if total_prompts > 0 else 0) }}%</div>
                <div class="stat-label">Weak Rate</div>
            </div>
            <div class="stat">
                <div class="stat-number">{{ insights.total_mutations }}</div>
                <div class="stat-label">Mutations</div>
            </div>
            <div class="stat">
                <div class="stat-number">{{ "%.1f"|format(insights.mutation_success_rate) }}%</div>
                <div class="stat-label">Mutation Success</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('evaluations')">All Evaluations</button>
            <button class="tab" onclick="showTab('scores')">Score Analysis</button>
            <button class="tab" onclick="showTab('families')">Mutations</button>
        </div>

        <div id="evaluations" class="tab-content active">
            <div class="section">
                <div class="section-header">
                    Evaluation Results (Grouped by Family)
                    <div style="float: right; font-size: 12px; font-weight: normal;">
                        <span style="margin-right: 10px; color: #6b7280;">Fail Threshold: 1.0</span>
                        <button onclick="expandAll()" style="padding: 4px 8px; margin-right: 5px; font-size: 11px;">Expand All</button>
                        <button onclick="collapseAll()" style="padding: 4px 8px; font-size: 11px;">Collapse All</button>
                    </div>
                </div>
                <div class="section-content">
                    {% for family in families %}
                    <div class="family-group">
                        <div class="family-header" onclick="toggleFamily('{{ family.parent_id }}')">
                            <span class="toggle-icon" id="toggle-{{ family.parent_id }}">▼</span>
                            <strong>{{ family.parent_id }}</strong>
                            <span class="score score-{{ family.parent_status }} tooltip">{{ "%.2f"|format(family.parent_score) }}
                                {% if family.parent_signals %}
                                <div class="tooltip-content">
                                    {% for signal in family.parent_signals %}
                                    {% if signal.score <= 0.33 %}
                                    <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                    {% elif signal.score <= 0.66 %}
                                    <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                    {% else %}
                                    <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                    {% endif %}
                                    {% endfor %}
                                </div>
                                {% endif %}
                            </span>
                            <span class="mutation-count">({{ family.mutation_count }} mutations)</span>
                        </div>
                        
                        <div class="family-content" id="content-{{ family.parent_id }}">
                            <!-- Parent evaluation -->
                            <div class="eval-row parent-row">
                                <div class="eval-id">{{ family.parent_id }}</div>
                                <div class="eval-prompt">{{ family.parent_prompt[:200] }}{% if family.parent_prompt|length > 200 %}...{% endif %}</div>
                                <div class="eval-response">{{ family.parent_response[:150] }}{% if family.parent_response|length > 150 %}...{% endif %}</div>
                                <div class="eval-score">
                                    <span class="score score-{{ family.parent_status }} tooltip">{{ "%.2f"|format(family.parent_score) }}
                                        {% if family.parent_signals %}
                                        <div class="tooltip-content">
                                            {% for signal in family.parent_signals %}
                                            {% if signal.score <= 0.33 %}
                                            <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% elif signal.score <= 0.66 %}
                                            <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% else %}
                                            <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% endif %}
                                            {% endfor %}
                                        </div>
                                        {% endif %}
                                    </span>
                                    <div class="score-bar">
                                        <div class="score-fill {{ family.parent_status }}" 
                                             style="width: {{ (family.parent_score * 100)|int }}%"></div>
                                    </div>
                                </div>
                                <div class="eval-outcome">
                                    <span class="score score-{{ family.parent_status }} tooltip">{{ family.parent_status }}
                                        {% if family.parent_signals %}
                                        <div class="tooltip-content">
                                            {% for signal in family.parent_signals %}
                                            {% if signal.score <= 0.33 %}
                                            <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% elif signal.score <= 0.66 %}
                                            <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% else %}
                                            <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% endif %}
                                            {% endfor %}
                                        </div>
                                        {% endif %}
                                    </span>
                                </div>
                            </div>
                            
                            <!-- Child mutations with recursive display -->
                            {% macro render_mutations(mutations, indent_level=1) %}
                                {% for mutation in mutations %}
                                <div class="eval-row child-row" style="margin-left: {{ indent_level * 20 }}px;">
                                    <div class="eval-id mutation-id">
                                        <span class="mutation-indent">
                                            {% if indent_level == 1 %}└─{% elif indent_level == 2 %}  └─{% else %}    └─{% endif %}
                                        </span>{{ mutation.id }}
                                        <div class="mutation-operator">Gen {{ mutation.generation }}: {{ mutation.operator }}</div>
                                    </div>
                                    <div class="eval-prompt">{{ mutation.prompt[:200] }}{% if mutation.prompt|length > 200 %}...{% endif %}</div>
                                    <div class="eval-response">{{ mutation.response[:150] }}{% if mutation.response|length > 150 %}...{% endif %}</div>
                                    <div class="eval-score">
                                        <span class="score score-{{ mutation.status }} tooltip">{{ "%.2f"|format(mutation.score) }}
                                            {% if mutation.signals %}
                                            <div class="tooltip-content">
                                                {% for signal in mutation.signals %}
                                                {% if signal.score <= 0.33 %}
                                                <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% elif signal.score <= 0.66 %}
                                                <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% else %}
                                                <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% endif %}
                                                {% endfor %}
                                            </div>
                                            {% endif %}
                                        </span>
                                        <div class="score-bar">
                                            <div class="score-fill {{ mutation.status }}" 
                                                 style="width: {{ (mutation.score * 100)|int }}%"></div>
                                        </div>
                                    </div>
                                    <div class="eval-outcome">
                                        <span class="score score-{{ mutation.status }} tooltip">{{ mutation.status }}
                                            {% if mutation.signals %}
                                            <div class="tooltip-content">
                                                {% for signal in mutation.signals %}
                                                {% if signal.score <= 0.33 %}
                                                <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% elif signal.score <= 0.66 %}
                                                <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% else %}
                                                <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                                {% endif %}
                                                {% endfor %}
                                            </div>
                                            {% endif %}
                                        </span>
                                        {% if mutation.duration_ms %}
                                        <div class="duration">{{ "%.0f"|format(mutation.duration_ms) }}ms</div>
                                        {% endif %}
                                    </div>
                                </div>
                                {% if mutation.children %}
                                    {{ render_mutations(mutation.children, indent_level + 1) }}
                                {% endif %}
                                {% endfor %}
                            {% endmacro %}
                            
                            {{ render_mutations(family.mutations) }}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <div id="scores" class="tab-content">
            <div class="compact-grid">
                <div class="section">
                    <div class="section-header">Score Distribution</div>
                    <div class="section-content">
                        <table class="table">
                            <tr><th>Range</th><th>Count</th></tr>
                            {% for range, count in score_analysis.score_ranges.items() %}
                            <tr><td>{{ range }}</td><td>{{ count }}</td></tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-header">Close Calls (±0.1 from threshold)</div>
                    <div class="section-content">
                        {% for call in score_analysis.close_calls %}
                        <div class="insight-item">
                            <strong>{{ call.id }}</strong>: {{ "%.3f"|format(call.score) }} 
                            ({{ "%.3f"|format(call.distance) }} from threshold)
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>

        <div id="families" class="tab-content">
            <div class="section">
                <div class="section-header">Mutation Families ({{ insights.families_with_mutations }} with mutations)</div>
                <div class="section-content">
                    {% for family in families %}
                    <div style="border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 10px;">
                        <strong>{{ family.parent_id }}</strong> 
                        <span class="score score-{{ family.parent_status }} tooltip">{{ "%.2f"|format(family.parent_score) }}
                            {% if family.parent_signals %}
                            <div class="tooltip-content">
                                {% for signal in family.parent_signals %}
                                {% if signal.score <= 0.33 %}
                                <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                {% elif signal.score <= 0.66 %}
                                <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                {% else %}
                                <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                {% endif %}
                                {% endfor %}
                            </div>
                            {% endif %}
                        </span>
                        
                        {% if family.mutations %}
                        <div style="margin-top: 5px;">
                            <strong>Mutations:</strong>
                            {% macro render_summary_mutations(mutations, indent=20) %}
                                {% for mutation in mutations %}
                                <div style="margin-left: {{ indent }}px; padding: 3px 0;">
                                    Gen{{ mutation.generation }}: {{ mutation.operator }} → 
                                    <span class="score score-{{ mutation.status }} tooltip">{{ "%.2f"|format(mutation.score) }}
                                        {% if mutation.signals %}
                                        <div class="tooltip-content">
                                            {% for signal in mutation.signals %}
                                            {% if signal.score <= 0.33 %}
                                            <div class="signal-item signal-weak">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% elif signal.score <= 0.66 %}
                                            <div class="signal-item signal-mild">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% else %}
                                            <div class="signal-item signal-strong">{{ signal.category }}: {{ "%.1f"|format(signal.score) }}</div>
                                            {% endif %}
                                            {% endfor %}
                                        </div>
                                        {% endif %}
                                    </span>
                                    {% if mutation.children %}
                                        {{ render_summary_mutations(mutation.children, indent + 20) }}
                                    {% endif %}
                                </div>
                                {% endfor %}
                            {% endmacro %}
                            {{ render_summary_mutations(family.mutations) }}
                        </div>
                        {% else %}
                        <div style="color: #666; font-style: italic;">No mutations</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function toggleFamily(familyId) {
            const content = document.getElementById('content-' + familyId);
            const icon = document.getElementById('toggle-' + familyId);
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                icon.classList.remove('collapsed');
                icon.textContent = '▼';
            } else {
                content.classList.add('collapsed');
                icon.classList.add('collapsed');
                icon.textContent = '▶';
            }
        }
        
        // Add collapse/expand all functionality
        function collapseAll() {
            document.querySelectorAll('.family-content').forEach(el => {
                el.classList.add('collapsed');
            });
            document.querySelectorAll('.toggle-icon').forEach(el => {
                el.classList.add('collapsed');
                el.textContent = '▶';
            });
        }
        
        function expandAll() {
            document.querySelectorAll('.family-content').forEach(el => {
                el.classList.remove('collapsed');
            });
            document.querySelectorAll('.toggle-icon').forEach(el => {
                el.classList.remove('collapsed');
                el.textContent = '▼';
            });
        }
    </script>
</body>
</html>'''