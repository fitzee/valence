"""Metamorphic testing utilities."""

import logging
from typing import Dict, List, Optional, Tuple

from valence.model import Model
from valence.validators import validate_filter_monotonicity, validate_determinism

logger = logging.getLogger(__name__)


class MetamorphicTestError(Exception):
    """Metamorphic testing error."""
    pass


def generate_filter_variants(original_prompt: str) -> Dict[str, str]:
    """
    Generate strict and loose variants of a filter query.
    
    Metamorphic relation: strict ⊆ original ⊆ loose
    """
    variants = {"original": original_prompt}
    
    # Generate strict variant (tighten constraints)
    strict_base = original_prompt
    if "under" in original_prompt or "<" in original_prompt:
        # Make duration/time constraints stricter
        import re
        match = re.search(r'(\d+)\s*(hours?|minutes?)', original_prompt)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            strict_value = max(1, value // 2)  # Half the time
            strict_base = re.sub(
                r'\d+\s*(hours?|minutes?)',
                f'{strict_value} {unit}',
                strict_base,
                count=1
            )
    
    if "top" in original_prompt.lower():
        # Reduce top N to make stricter
        match = re.search(r'top\s+(\d+)', original_prompt.lower())
        if match:
            n = int(match.group(1))
            strict_n = max(1, n // 2)
            strict_base = re.sub(
                r'top\s+\d+',
                f'top {strict_n}',
                strict_base,
                flags=re.IGNORECASE
            )
    
    variants["strict"] = strict_base
    
    # Generate loose variant (loosen constraints)
    loose_base = original_prompt
    if "under" in original_prompt or "<" in original_prompt:
        match = re.search(r'(\d+)\s*(hours?|minutes?)', original_prompt)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            loose_value = value * 2  # Double the time
            loose_base = re.sub(
                r'\d+\s*(hours?|minutes?)',
                f'{loose_value} {unit}',
                loose_base,
                count=1
            )
    
    if "top" in original_prompt.lower():
        match = re.search(r'top\s+(\d+)', original_prompt.lower())
        if match:
            n = int(match.group(1))
            loose_n = n * 2
            loose_base = re.sub(
                r'top\s+\d+',
                f'top {loose_n}',
                loose_base,
                flags=re.IGNORECASE
            )
    
    variants["loose"] = loose_base
    
    # Default variants if none generated
    if "strict" not in variants:
        variants["strict"] = original_prompt + " (highest rated only)"
    if "loose" not in variants:
        variants["loose"] = original_prompt.replace("top 5", "all").replace("under", "any duration")
    
    return variants


def run_metamorphic_test(
    model: Model,
    original_prompt: str,
    test_type: str = "filter_monotonicity"
) -> Tuple[float, Dict[str, any]]:
    """Run metamorphic test on a prompt."""
    try:
        if test_type == "filter_monotonicity":
            # Generate variants
            variants = generate_filter_variants(original_prompt)
            
            # Get responses
            responses = {}
            for variant_type, prompt in variants.items():
                response, _, error = model.generate(prompt)
                if error:
                    return 1.0, {"error": f"model_error_{variant_type}", "details": error}
                responses[variant_type] = response
            
            # Validate monotonicity
            score, details = validate_filter_monotonicity(
                responses.get("original", ""),
                responses.get("strict", ""),
                responses.get("loose", "")
            )
            
            return score, {
                "test_type": "filter_monotonicity",
                "variants": variants,
                **details
            }
            
        elif test_type == "determinism":
            # Run same prompt multiple times
            responses = []
            for i in range(3):
                response, _, error = model.generate(original_prompt)
                if error:
                    return 1.0, {"error": f"model_error_run_{i}", "details": error}
                responses.append(response)
            
            # Check determinism
            score, details = validate_determinism(responses)
            
            return score, {
                "test_type": "determinism",
                "runs": len(responses),
                **details
            }
            
        elif test_type == "order_invariance":
            # Shuffle list items in prompt
            import re
            import random
            
            lines = original_prompt.split('\n')
            list_items = [l for l in lines if re.match(r'^[\d\-\*]\s*', l.strip())]
            
            if len(list_items) > 1:
                # Create shuffled variant
                shuffled_items = list_items.copy()
                random.shuffle(shuffled_items)
                
                other_lines = [l for l in lines if l not in list_items]
                shuffled_prompt = '\n'.join(other_lines + shuffled_items)
                
                # Get responses
                original_response, _, err1 = model.generate(original_prompt)
                shuffled_response, _, err2 = model.generate(shuffled_prompt)
                
                if err1 or err2:
                    return 1.0, {"error": "model_error", "details": err1 or err2}
                
                # Compare responses (should be semantically equivalent)
                from valence.util import are_similar
                if are_similar(original_response, shuffled_response):
                    return 0.0, {"test_type": "order_invariance", "status": "invariant"}
                else:
                    return 1.0, {
                        "test_type": "order_invariance",
                        "error": "order_dependent",
                        "original_prompt": original_prompt,
                        "shuffled_prompt": shuffled_prompt
                    }
            else:
                return 0.0, {"test_type": "order_invariance", "status": "no_list_found"}
        
        else:
            raise MetamorphicTestError(f"Unknown test type: {test_type}")
            
    except Exception as e:
        logger.error(f"Metamorphic test error: {e}")
        return 1.0, {"error": str(e)}


def generate_metamorphic_suite(seed_prompt: str) -> List[Dict[str, str]]:
    """Generate a suite of metamorphic tests for a seed prompt."""
    suite = []
    
    # Filter monotonicity tests
    if any(word in seed_prompt.lower() for word in ["top", "under", "less than", "more than"]):
        suite.append({
            "test_type": "filter_monotonicity",
            "prompt": seed_prompt,
            "description": "Test filter subset relationships"
        })
    
    # Determinism tests
    suite.append({
        "test_type": "determinism",
        "prompt": seed_prompt,
        "description": "Test response stability"
    })
    
    # Order invariance tests
    if '\n' in seed_prompt or ',' in seed_prompt:
        suite.append({
            "test_type": "order_invariance",
            "prompt": seed_prompt,
            "description": "Test order independence"
        })
    
    return suite