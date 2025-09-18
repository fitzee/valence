"""Validators for ranking, filtering, and metamorphic testing."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ValidatorError(Exception):
    """Validator execution error."""
    pass


def validate_ranking_correctness(
    response: str,
    expected_count: int,
    ranking_key: Optional[str] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate that response contains properly ranked results.
    
    Returns:
        Tuple of (score, details) where score is 0.0 for pass, 1.0 for fail
    """
    try:
        # Try to parse response as JSON first
        if response.strip().startswith('[') or response.strip().startswith('{'):
            try:
                data = json.loads(response)
                if isinstance(data, dict) and 'results' in data:
                    items = data['results']
                elif isinstance(data, list):
                    items = data
                else:
                    items = []
            except json.JSONDecodeError:
                items = []
        else:
            # Fall back to parsing numbered list
            pattern = r'^\s*(\d+)\.\s+(.+)$'
            items = re.findall(pattern, response, re.MULTILINE)
        
        # Check count
        actual_count = len(items)
        if actual_count != expected_count:
            return 1.0, {
                "error": "count_mismatch",
                "expected": expected_count,
                "actual": actual_count
            }
        
        # Check uniqueness
        if len(items) != len(set(str(item) for item in items)):
            return 1.0, {"error": "duplicate_items"}
        
        # Check ranking if key provided
        if ranking_key and items:
            # Extract ranking values
            values = []
            for item in items:
                if isinstance(item, dict) and ranking_key in item:
                    values.append(item[ranking_key])
                elif isinstance(item, (list, tuple)) and len(item) > 1:
                    # Try to extract score from text
                    score_match = re.search(r'(\d+(?:\.\d+)?)', str(item))
                    if score_match:
                        values.append(float(score_match.group(1)))
            
            # Verify descending order
            if values and values != sorted(values, reverse=True):
                return 1.0, {
                    "error": "ranking_order",
                    "values": values,
                    "expected_order": sorted(values, reverse=True)
                }
        
        return 0.0, {"status": "valid_ranking", "count": actual_count}
        
    except Exception as e:
        logger.error(f"Ranking validation error: {e}")
        return 1.0, {"error": str(e)}


def validate_filter_monotonicity(
    original_response: str,
    strict_response: str,
    loose_response: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate filter monotonicity: strict ⊆ original ⊆ loose
    """
    try:
        def extract_ids(text: str) -> Set[str]:
            """Extract identifiable items from response."""
            # Try JSON parsing
            if text.strip().startswith('[') or text.strip().startswith('{'):
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        # Handle both strings and objects in lists
                        result = set()
                        for i, item in enumerate(data):
                            if isinstance(item, str):
                                result.add(item)
                            elif isinstance(item, dict):
                                result.add(str(item.get('id', str(i))))
                            else:
                                result.add(str(item))
                        return result
                    elif isinstance(data, dict) and 'results' in data:
                        return {str(item.get('id', str(i))) for i, item in enumerate(data['results'])}
                except:
                    pass
            
            # Fall back to line-based extraction
            lines = text.strip().split('\n')
            return {line.strip() for line in lines if line.strip()}
        
        original_ids = extract_ids(original_response)
        strict_ids = extract_ids(strict_response)
        loose_ids = extract_ids(loose_response)
        
        # Check subset relationships
        strict_subset = strict_ids.issubset(original_ids)
        original_subset = original_ids.issubset(loose_ids)
        
        if not strict_subset or not original_subset:
            return 1.0, {
                "error": "monotonicity_violation",
                "strict_subset_of_original": strict_subset,
                "original_subset_of_loose": original_subset,
                "strict_count": len(strict_ids),
                "original_count": len(original_ids),
                "loose_count": len(loose_ids)
            }
        
        return 0.0, {
            "status": "monotonic",
            "counts": {
                "strict": len(strict_ids),
                "original": len(original_ids),
                "loose": len(loose_ids)
            }
        }
        
    except Exception as e:
        logger.error(f"Filter monotonicity validation error: {e}")
        return 1.0, {"error": str(e)}


def validate_determinism(
    responses: List[str],
    tolerance: float = 0.95
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate that repeated queries produce stable responses.
    """
    if not responses or len(responses) < 2:
        return 0.0, {"status": "insufficient_samples"}
    
    # Normalize responses
    normalized = [re.sub(r'\s+', ' ', r.strip().lower()) for r in responses]
    
    # Check exact matches
    unique_responses = set(normalized)
    consistency_rate = 1.0 / len(unique_responses)
    
    if consistency_rate >= tolerance:
        return 0.0, {
            "status": "deterministic",
            "consistency_rate": consistency_rate,
            "unique_responses": len(unique_responses)
        }
    else:
        return 1.0, {
            "error": "non_deterministic",
            "consistency_rate": consistency_rate,
            "unique_responses": len(unique_responses),
            "total_samples": len(responses)
        }


def validate_schema_conformance(
    response: str,
    expected_schema: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate response against expected JSON schema.
    """
    try:
        # Parse response as JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            return 1.0, {"error": "invalid_json", "parse_error": str(e)}
        
        # Check required fields
        required_fields = expected_schema.get('required', [])
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return 1.0, {
                "error": "missing_required_fields",
                "missing": missing_fields
            }
        
        # Check field types
        field_types = expected_schema.get('properties', {})
        type_errors = []
        
        for field, expected_type in field_types.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if expected_type.get('type') and actual_type != expected_type['type']:
                    type_errors.append({
                        "field": field,
                        "expected": expected_type['type'],
                        "actual": actual_type
                    })
        
        if type_errors:
            return 1.0, {"error": "type_mismatch", "mismatches": type_errors}
        
        return 0.0, {"status": "valid_schema"}
        
    except Exception as e:
        logger.error(f"Schema validation error: {e}")
        return 1.0, {"error": str(e)}


def validate_no_pii_leakage(response: str) -> Tuple[float, Dict[str, Any]]:
    """
    Check for potential PII leakage in response.
    """
    pii_patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
        (r'\b\d{3}-\d{2}-\d{4}\b', 'ssn'),
        (r'\b(?:\d{3}[-.]?)?\d{3}[-.]?\d{4}\b', 'phone'),
        (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b', 'credit_card'),
    ]
    
    found_pii = []
    for pattern, pii_type in pii_patterns:
        matches = re.findall(pattern, response)
        if matches:
            found_pii.append({"type": pii_type, "count": len(matches)})
    
    if found_pii:
        return 1.0, {"error": "pii_detected", "found": found_pii}
    
    return 0.0, {"status": "no_pii_detected"}


def validate_latency(
    response_time_ms: float,
    p50_threshold: float = 500,
    p95_threshold: float = 2000
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate response latency against SLO thresholds.
    """
    if response_time_ms <= p50_threshold:
        return 0.0, {"status": "fast", "latency_ms": response_time_ms}
    elif response_time_ms <= p95_threshold:
        return 0.5, {"status": "acceptable", "latency_ms": response_time_ms}
    else:
        return 1.0, {
            "error": "slo_violation",
            "latency_ms": response_time_ms,
            "threshold_exceeded": "p95"
        }