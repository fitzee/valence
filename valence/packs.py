"""Load and validate YAML packs."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from valence.schema import Pack, Seed
from valence.storage import StorageError, load_json

logger = logging.getLogger(__name__)


class PackError(Exception):
    """Pack loading or validation error."""
    pass


def load_seeds(path: Path) -> list[Seed]:
    """Load seeds from JSON file."""
    if not path.exists():
        raise PackError(f"Seeds file not found: {path}")
    
    try:
        data = load_json(path)
    except StorageError as e:
        raise PackError(f"Failed to load seeds: {e}") from e
    
    if not isinstance(data, list):
        raise PackError("Seeds file must contain a JSON array")
    
    seeds = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise PackError(f"Seed {i} must be an object")
        
        if "id" not in item or "prompt" not in item:
            raise PackError(f"Seed {i} missing required fields (id, prompt)")
        
        try:
            seed = Seed(**item)
            seeds.append(seed)
        except ValidationError as e:
            raise PackError(f"Invalid seed {i}: {e}") from e
    
    if not seeds:
        raise PackError("No valid seeds found")
    
    ids = [s.id for s in seeds]
    if len(ids) != len(set(ids)):
        raise PackError("Duplicate seed IDs found")
    
    logger.info(f"Loaded {len(seeds)} seeds from {path}")
    return seeds


def load_pack_file(path: Path) -> list[Pack]:
    """Load packs from a single YAML file."""
    if not path.exists():
        raise PackError(f"Pack file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
    except yaml.YAMLError as e:
        raise PackError(f"Invalid YAML in {path}: {e}") from e
    except Exception as e:
        raise PackError(f"Failed to read {path}: {e}") from e
    
    packs = []
    for doc_num, doc in enumerate(docs, 1):
        if doc is None:
            continue
        
        if not isinstance(doc, dict):
            raise PackError(f"Document {doc_num} in {path} must be a mapping")
        
        try:
            pack = Pack(**doc)
            packs.append(pack)
        except ValidationError as e:
            raise PackError(f"Invalid pack in {path} (doc {doc_num}): {e}") from e
    
    return packs


def load_packs(path: Path) -> list[Pack]:
    """Load all packs from directory or file."""
    if path.is_file():
        return load_pack_file(path)
    
    if not path.is_dir():
        raise PackError(f"Pack path is neither file nor directory: {path}")
    
    yaml_files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
    
    if not yaml_files:
        raise PackError(f"No YAML files found in {path}")
    
    all_packs = []
    for yaml_file in yaml_files:
        try:
            packs = load_pack_file(yaml_file)
            all_packs.extend(packs)
            logger.info(f"Loaded {len(packs)} packs from {yaml_file}")
        except PackError as e:
            logger.error(f"Failed to load {yaml_file}: {e}")
            raise
    
    if not all_packs:
        raise PackError("No valid packs found")
    
    ids = [p.id for p in all_packs]
    if len(ids) != len(set(ids)):
        raise PackError("Duplicate pack IDs found")
    
    logger.info(f"Loaded {len(all_packs)} packs total")
    return all_packs


def validate_pack(pack: Pack) -> None:
    """Validate pack configuration."""
    if not pack.detectors:
        raise PackError(f"Pack {pack.id} has no detectors")
    
    for i, detector in enumerate(pack.detectors):
        if detector.type == "regex_set" and not detector.patterns:
            raise PackError(f"Pack {pack.id} detector {i}: regex_set requires patterns")
        
        if detector.type == "keyword" and not detector.keywords:
            raise PackError(f"Pack {pack.id} detector {i}: keyword requires keywords")
        
        if detector.type == "validator" and not detector.validator_name:
            raise PackError(f"Pack {pack.id} detector {i}: validator requires validator_name")
        
        if detector.type == "validator":
            valid_names = ["sum_equals", "json_well_formed"]
            if detector.validator_name not in valid_names:
                raise PackError(
                    f"Pack {pack.id} detector {i}: unknown validator '{detector.validator_name}'"
                )
        
        if detector.type == "llm_judge":
            if not detector.judge_model:
                raise PackError(f"Pack {pack.id} detector {i}: llm_judge requires judge_model")
            if not detector.judge_prompt:
                raise PackError(f"Pack {pack.id} detector {i}: llm_judge requires judge_prompt")
            
            # Validate judge_prompt contains expected placeholders
            required_placeholders = ["{response}"]
            optional_placeholders = ["{original_prompt}", "{prompt}"]
            
            has_required = all(placeholder in detector.judge_prompt for placeholder in required_placeholders)
            if not has_required:
                raise PackError(
                    f"Pack {pack.id} detector {i}: judge_prompt must contain {{response}} placeholder"
                )