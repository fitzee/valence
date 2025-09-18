"""Safe JSONL appends and atomic writes."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from valence.schema import EvalRecord, FailureRecord, RunMetadata
from valence.util import safe_json_dumps


class StorageError(Exception):
    """Storage operation error."""
    pass


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, content: str) -> None:
    """Atomically write content to file."""
    ensure_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise StorageError(f"Failed to write {path}: {e}") from e


def append_jsonl(path: Path, record: Any) -> None:
    """Append record to JSONL file."""
    ensure_dir(path)
    
    if hasattr(record, "model_dump"):
        data = record.model_dump()
    elif isinstance(record, dict):
        data = record
    else:
        data = {"data": record}
    
    line = safe_json_dumps(data) + "\n"
    
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        raise StorageError(f"Failed to append to {path}: {e}") from e


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from JSONL file."""
    if not path.exists():
        return []
    
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise StorageError(f"Invalid JSON at line {line_num}: {e}") from e
    except Exception as e:
        if not isinstance(e, StorageError):
            raise StorageError(f"Failed to read {path}: {e}") from e
        raise
    
    return records


def load_json(path: Path) -> Any:
    """Load JSON file."""
    if not path.exists():
        raise StorageError(f"File not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise StorageError(f"Invalid JSON in {path}: {e}") from e
    except Exception as e:
        raise StorageError(f"Failed to read {path}: {e}") from e


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save data as JSON file."""
    from datetime import datetime
    
    def default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return str(o)
    
    content = json.dumps(data, indent=indent, ensure_ascii=False, default=default)
    atomic_write(path, content)


class MemoryStore:
    """Manages failure memory with deduplication."""
    
    def __init__(self, path: Path):
        """Initialize memory store."""
        self.path = path
        self._hashes: set[str] = set()
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing failure hashes."""
        if not self.path.exists():
            return
        
        records = read_jsonl(self.path)
        for record in records:
            if "prompt_hash" in record:
                self._hashes.add(record["prompt_hash"])
    
    def add_failure(self, failure: FailureRecord) -> bool:
        """Add failure to memory if not duplicate."""
        if failure.prompt_hash in self._hashes:
            return False
        
        append_jsonl(self.path, failure)
        self._hashes.add(failure.prompt_hash)
        return True
    
    def has_seen(self, prompt_hash: str) -> bool:
        """Check if prompt hash has been seen."""
        return prompt_hash in self._hashes


class RunStorage:
    """Manages storage for evaluation runs."""
    
    def __init__(self, output_dir: Path):
        """Initialize run storage."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evals_path = self.output_dir / "evals.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        self.scorecards_path = self.output_dir / "scorecards.jsonl"
        self.lineage_path = self.output_dir / "lineage.jsonl"
    
    def save_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata."""
        save_json(self.metadata_path, metadata.model_dump())
    
    def append_eval(self, record: EvalRecord) -> None:
        """Append evaluation record."""
        append_jsonl(self.evals_path, record)
        
        if record.scorecard:
            scorecard_data = {
                "id": record.id,
                "outcome": record.scorecard.outcome.value,
                "score": record.scorecard.score,
                "tags": record.scorecard.tags,
            }
            append_jsonl(self.scorecards_path, scorecard_data)
        
        if record.lineage.parent_id:
            lineage_data = {
                "id": record.id,
                "parent_id": record.lineage.parent_id,
                "mutation_operator": record.lineage.mutation_operator,
                "generation": record.lineage.generation,
            }
            append_jsonl(self.lineage_path, lineage_data)
    
    def load_metadata(self) -> Optional[RunMetadata]:
        """Load run metadata."""
        if not self.metadata_path.exists():
            return None
        
        data = load_json(self.metadata_path)
        return RunMetadata(**data)
    
    def load_evals(self) -> list[EvalRecord]:
        """Load all evaluation records."""
        records = read_jsonl(self.evals_path)
        return [EvalRecord(**r) for r in records]