"""Utility functions for hashing, timing, and text processing."""

import hashlib
import re
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Set


def normalize_text(text: str) -> str:
    """Normalize text for consistent hashing."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


def hash_prompt(prompt: str) -> str:
    """Generate SHA256 hash of normalized prompt."""
    normalized = normalize_text(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat() + "Z"


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager for timing operations."""
    result = {"duration_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result["duration_ms"] = (end - start) * 1000


def redact_sensitive(text: str) -> str:
    """Redact potentially sensitive information (stub for now)."""
    patterns = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"\b(?:\d{3}[-.]?)?\d{3}[-.]?\d{4}\b", "[PHONE]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "[CARD]"),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    import json
    from datetime import datetime
    
    def default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    
    return json.dumps(obj, default=default, ensure_ascii=False)


def extract_numbers(text: str) -> list[int]:
    """Extract all integers from text."""
    numbers = re.findall(r"-?\d+", text)
    result = []
    for num in numbers:
        try:
            result.append(int(num))
        except ValueError:
            continue
    return result


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    import json
    
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


# SimHash for similarity-based deduplication
def simhash(text: str, bit_size: int = 64) -> int:
    """Compute SimHash for near-duplicate detection."""
    # Tokenize text
    tokens = text.lower().split()
    if not tokens:
        return 0
    
    # Initialize bit vector
    v = [0] * bit_size
    
    for token in tokens:
        # Hash each token
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        
        # Update bit vector
        for i in range(bit_size):
            bit = (token_hash >> i) & 1
            if bit == 1:
                v[i] += 1
            else:
                v[i] -= 1
    
    # Generate fingerprint
    fingerprint = 0
    for i in range(bit_size):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    
    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two hashes."""
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def are_similar(text1: str, text2: str, threshold: int = 15) -> bool:
    """Check if two texts are similar based on SimHash."""
    hash1 = simhash(text1)
    hash2 = simhash(text2)
    distance = hamming_distance(hash1, hash2)
    return distance <= threshold


# Richer failure fingerprinting
def compute_failure_fingerprint(
    prompt: str,
    response: str,
    detector_tags: Set[str],
    http_status: int = 200
) -> str:
    """Compute rich failure fingerprint for deduplication."""
    # Normalize prompt and response
    prompt_norm = normalize_text(prompt)
    response_norm = normalize_text(response) if response else ""
    
    # Extract key features
    features = {
        "prompt_hash": hash_prompt(prompt),
        "response_length": len(response) if response else 0,
        "detector_tags": sorted(list(detector_tags)),
        "http_status": http_status,
        "has_numbers": bool(extract_numbers(response)) if response else False,
        "is_json": is_valid_json(response) if response else False,
    }
    
    # Create composite fingerprint
    fingerprint_str = f"{features['prompt_hash']}:{','.join(features['detector_tags'])}:{features['http_status']}"
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]