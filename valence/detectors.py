"""Detector execution and result aggregation."""

import logging
import re
from typing import Any, Optional

from valence.schema import Detector, DetectorType, Outcome, Pack, Scorecard, Seed, Signal
from valence.util import extract_numbers, is_valid_json, timer

logger = logging.getLogger(__name__)


class DetectorError(Exception):
    """Detector execution error."""
    pass


def compute_expected_value(name: str, prompt: str) -> Any:
    """Compute expected value from reference function."""
    if name == "sum_prompt_numbers":
        numbers = extract_numbers(prompt)
        return sum(numbers) if numbers else None
    else:
        raise DetectorError(f"Unknown computed expected function: {name}")


def get_expected_value(
    detector: Detector, seed: Optional[Seed], prompt: str
) -> Optional[Any]:
    """Get expected value for a detector."""
    if not detector.expected:
        return None
    
    if detector.expected == "from_seed":
        if not seed or not seed.label:
            return None
        # For sum_equals, look for 'answer' in label
        if detector.validator_name == "sum_equals":
            return seed.label.get("answer")
        return None
    
    if isinstance(detector.expected, str) and detector.expected.startswith("computed:"):
        func_name = detector.expected[9:]  # Remove "computed:" prefix
        return compute_expected_value(func_name, prompt)
    
    if isinstance(detector.expected, list):
        # This is a "one_of" list
        return detector.expected
    
    return None


def run_regex_detector(detector: Detector, text: str) -> Signal:
    """Run regex_set detector on text."""
    if not detector.patterns:
        raise DetectorError("regex_set detector requires patterns")
    
    matched = []
    for pattern in detector.patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(pattern)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            continue
    
    score = 1.0 if matched else 0.0
    return Signal(
        detector_type=DetectorType.REGEX_SET,
        category=detector.category,
        score=score,
        matched=matched,
    )


def run_keyword_detector(detector: Detector, text: str) -> Signal:
    """Run keyword detector on text."""
    if not detector.keywords:
        raise DetectorError("keyword detector requires keywords")
    
    text_lower = text.lower()
    matched = []
    
    for keyword in detector.keywords:
        if keyword.lower() in text_lower:
            matched.append(keyword)
    
    score = 1.0 if matched else 0.0
    return Signal(
        detector_type=DetectorType.KEYWORD,
        category=detector.category,
        score=score,
        matched=matched,
    )


def run_validator_detector(
    detector: Detector, prompt: str, response: str, seed: Optional[Seed] = None
) -> Signal:
    """Run validator detector on prompt/response pair."""
    if not detector.validator_name:
        raise DetectorError("validator detector requires validator_name")
    
    validator_name = detector.validator_name
    score = 0.0
    matched = []
    
    if validator_name == "sum_equals":
        # Get expected value (from seed, computed, or one_of)
        expected = get_expected_value(detector, seed, prompt)
        
        if expected is None:
            # No expected value available - not applicable
            return Signal(
                detector_type=DetectorType.VALIDATOR,
                category=detector.category,
                score=0.0,
                matched=[],
                details={"reason": "no_expected"},
            )
        
        # Extract number from response
        response_numbers = extract_numbers(response)
        if response_numbers:
            first_response_num = response_numbers[0]
            if first_response_num != expected:
                score = 1.0
                matched = [f"Expected {expected}, got {first_response_num}"]
                details = {"truth": expected, "got": first_response_num}
            else:
                score = 0.0
                matched = []
                details = {"truth": expected, "got": first_response_num}
        else:
            # No number in response - this is a failure per spec
            score = 1.0
            matched = ["No number found in response"]
            details = {"truth": expected, "got": None}
    
    elif validator_name == "json_well_formed":
        details = None
        if "json" in prompt.lower():
            if not is_valid_json(response):
                score = 1.0
                matched = ["Invalid JSON in response"]
                details = {"error": "Invalid JSON format"}
    
    else:
        raise DetectorError(f"Unknown validator: {validator_name}")
    
    return Signal(
        detector_type=DetectorType.VALIDATOR,
        category=detector.category,
        score=score,
        matched=matched,
        details=details,
    )


def parse_judge_score(judgment: str) -> tuple[float, str]:
    """Parse score and reason from LLM judge response.
    
    Expected formats:
    - "SCORE: 1.0 REASON: explanation"
    - "Score: 0.0"
    - "1.0"
    - Raw text (fallback to simple parsing)
    """
    judgment = judgment.strip()
    
    # Try structured format first
    if "SCORE:" in judgment.upper():
        try:
            score_part = judgment.upper().split("SCORE:")[1].split("REASON:")[0].strip()
            score = float(score_part)
            
            if "REASON:" in judgment.upper():
                reason = judgment.upper().split("REASON:")[1].strip()
            else:
                reason = judgment
                
            return min(max(score, 0.0), 1.0), reason
        except (ValueError, IndexError):
            pass
    
    # Try simple score format
    try:
        # Look for first number that could be a score
        import re
        scores = re.findall(r'\b([01]\.?\d*)\b', judgment)
        if scores:
            score = float(scores[0])
            return min(max(score, 0.0), 1.0), judgment
    except ValueError:
        pass
    
    # Fallback: look for failure indicators in text
    failure_keywords = ["fail", "incorrect", "poor", "bad", "wrong", "invalid", "problematic"]
    success_keywords = ["pass", "correct", "good", "valid", "appropriate", "satisfactory"]
    
    judgment_lower = judgment.lower()
    
    failure_count = sum(1 for word in failure_keywords if word in judgment_lower)
    success_count = sum(1 for word in success_keywords if word in judgment_lower)
    
    if failure_count > success_count:
        return 1.0, judgment
    else:
        return 0.0, judgment


def run_llm_judge_detector(
    detector: Detector, prompt: str, response: str, seed: Optional[Seed] = None
) -> Signal:
    """Run LLM judge detector on prompt/response pair."""
    if not detector.judge_model:
        raise DetectorError("llm_judge detector requires judge_model")
    if not detector.judge_prompt:
        raise DetectorError("llm_judge detector requires judge_prompt")
    
    # Import here to avoid circular imports
    from valence.model import Model, ModelError
    
    try:
        # Initialize judge model
        judge_model = Model(detector.judge_model)
        
        # Format the judge prompt with variables
        judge_input = detector.judge_prompt.format(
            original_prompt=prompt,
            response=response,
            prompt=prompt,  # Alternative name
        )
        
        # Get judgment from LLM
        judgment, duration = judge_model.generate(judge_input)
        
        if judgment is None:
            # Judge model failed
            return Signal(
                detector_type=DetectorType.LLM_JUDGE,
                category=detector.category,
                score=0.0,
                matched=[],
                details={
                    "error": "Judge model returned no response",
                    "judge_model": detector.judge_model,
                    "judge_duration_ms": duration,
                }
            )
        
        # Parse score and reason from judgment
        score, reason = parse_judge_score(judgment)
        
        return Signal(
            detector_type=DetectorType.LLM_JUDGE,
            category=detector.category,
            score=score,
            matched=[reason] if score > 0 else [],
            details={
                "judgment": judgment,
                "parsed_reason": reason,
                "judge_model": detector.judge_model,
                "judge_duration_ms": duration,
            }
        )
        
    except ModelError as e:
        logger.error(f"Judge model error: {e}")
        return Signal(
            detector_type=DetectorType.LLM_JUDGE,
            category=detector.category,
            score=0.0,
            matched=[],
            details={
                "error": f"Judge model error: {str(e)}",
                "judge_model": detector.judge_model,
            }
        )
    except Exception as e:
        logger.error(f"LLM judge detector failed: {e}")
        return Signal(
            detector_type=DetectorType.LLM_JUDGE,
            category=detector.category,
            score=0.0,
            matched=[],
            details={
                "error": f"Judge detector error: {str(e)}",
                "judge_model": detector.judge_model,
            }
        )


def run_detector(
    detector: Detector, prompt: str, response: str, seed: Optional[Seed] = None
) -> Signal:
    """Run a single detector."""
    if detector.type == DetectorType.REGEX_SET:
        return run_regex_detector(detector, response)
    elif detector.type == DetectorType.KEYWORD:
        return run_keyword_detector(detector, response)
    elif detector.type == DetectorType.VALIDATOR:
        return run_validator_detector(detector, prompt, response, seed)
    elif detector.type == DetectorType.LLM_JUDGE:
        return run_llm_judge_detector(detector, prompt, response, seed)
    else:
        raise DetectorError(f"Unknown detector type: {detector.type}")


def aggregate_signals(signals: list[Signal]) -> Scorecard:
    """Aggregate detection signals into a scorecard."""
    if not signals:
        return Scorecard(
            outcome=Outcome.PASS,
            score=0.0,
            tags=[],
            signals=[],
        )
    
    scores = [s.score for s in signals]
    mean_score = sum(scores) / len(scores)
    
    tags = list(set(s.category for s in signals if s.score > 0))
    
    outcome = Outcome.FAIL if any(s.score == 1.0 for s in signals) else Outcome.PASS
    
    return Scorecard(
        outcome=outcome,
        score=mean_score,
        tags=tags,
        signals=signals,
    )


def run_detectors(
    pack: Pack,
    prompt: str,
    response: Optional[str],
    seed: Optional[Seed] = None,
) -> Scorecard:
    """Run all pack detectors and aggregate results."""
    if response is None:
        return Scorecard(
            outcome=Outcome.ERROR,
            score=0.0,
            tags=["model_error"],
            signals=[],
        )
    
    signals = []
    
    with timer() as t:
        for detector in pack.detectors:
            try:
                signal = run_detector(detector, prompt, response, seed)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Detector failed: {e}")
                continue
    
    duration = t["duration_ms"]
    
    if len(response) <= 2048:
        if duration > 15.0 and len(pack.detectors) <= 1000:
            logger.warning(
                f"Detector performance degraded: {duration:.2f}ms for {len(pack.detectors)} detectors"
            )
    
    return aggregate_signals(signals)