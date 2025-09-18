"""Pydantic models for seeds, packs, evals, signals, scorecards, and lineage."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class DetectorType(str, Enum):
    """Types of detectors available."""

    REGEX_SET = "regex_set"
    KEYWORD = "keyword"
    VALIDATOR = "validator"
    LLM_JUDGE = "llm_judge"


class Severity(str, Enum):
    """Severity levels for pack detectors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Outcome(str, Enum):
    """Evaluation outcome status."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class Detector(BaseModel):
    """Individual detector configuration."""

    type: DetectorType
    category: str
    patterns: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    validator_name: Optional[str] = None
    expected: Optional[str | list[Any]] = None  # "from_seed", "computed:<name>", or list for "one_of"
    judge_model: Optional[str] = None  # Model string for LLM judge (e.g., "openai:gpt-4o-mini")
    judge_prompt: Optional[str] = None  # Template for LLM judge evaluation

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: Optional[list[str]], info: Any) -> Optional[list[str]]:
        """Validate patterns are provided for regex_set."""
        if info.data.get("type") == DetectorType.REGEX_SET and not v:
            raise ValueError("patterns required for regex_set detector")
        return v

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v: Optional[list[str]], info: Any) -> Optional[list[str]]:
        """Validate keywords are provided for keyword detector."""
        if info.data.get("type") == DetectorType.KEYWORD and not v:
            raise ValueError("keywords required for keyword detector")
        return v

    @field_validator("validator_name")
    @classmethod
    def validate_validator_name(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate validator_name is provided for validator detector."""
        if info.data.get("type") == DetectorType.VALIDATOR and not v:
            raise ValueError("validator_name required for validator detector")
        return v

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate judge_model is provided for llm_judge detector."""
        if info.data.get("type") == DetectorType.LLM_JUDGE and not v:
            raise ValueError("judge_model required for llm_judge detector")
        return v

    @field_validator("judge_prompt")
    @classmethod
    def validate_judge_prompt(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate judge_prompt is provided for llm_judge detector."""
        if info.data.get("type") == DetectorType.LLM_JUDGE and not v:
            raise ValueError("judge_prompt required for llm_judge detector")
        return v


class Pack(BaseModel):
    """Evaluation pack configuration."""

    id: str
    version: str
    detectors: list[Detector]
    severity: Severity


class Seed(BaseModel):
    """Seed prompt for evaluation."""

    id: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    label: Optional[dict[str, Any]] = None  # Oracle/expected output for correctness checking


class Signal(BaseModel):
    """Detection signal from a single detector."""

    detector_type: DetectorType
    category: str
    score: float = Field(ge=0.0, le=1.0)
    matched: list[str] = Field(default_factory=list)
    details: Optional[dict[str, Any]] = None  # Additional details like {"truth": x, "got": y}


class Scorecard(BaseModel):
    """Aggregated results from all detectors."""

    outcome: Outcome
    score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)


class Lineage(BaseModel):
    """Lineage tracking for mutations."""

    parent_id: Optional[str] = None
    mutation_operator: Optional[str] = None
    generation: int = 0


class EvalRecord(BaseModel):
    """Complete evaluation record for a single prompt."""

    id: str
    prompt: str
    response: Optional[str] = None
    scorecard: Optional[Scorecard] = None
    lineage: Lineage = Field(default_factory=Lineage)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    model: str
    pack_id: Optional[str] = None
    error: Optional[str] = None


class RunMetadata(BaseModel):
    """Metadata for an evaluation run."""

    run_id: str
    model: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_prompts: int = 0
    total_failures: int = 0
    total_passes: int = 0
    total_errors: int = 0
    max_generations: int = 1
    mutations_per_failure: int = 4


class FailureRecord(BaseModel):
    """Record for failure memory."""

    prompt: str
    prompt_hash: str
    id: str
    parent_id: Optional[str] = None
    mutation_operator: Optional[str] = None
    timestamp: datetime
    model: str
    pack_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)