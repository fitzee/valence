"""Tests for pack loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from valence.packs import PackError, load_pack_file, load_packs, load_seeds, validate_pack
from valence.schema import Pack


def test_load_seeds_valid(tmp_path: Path) -> None:
    """Test loading valid seeds."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text("""[
        {"id": "test-1", "prompt": "Test prompt 1"},
        {"id": "test-2", "prompt": "Test prompt 2", "metadata": {"key": "value"}},
        {"id": "test-3", "prompt": "Add 10 and 20", "label": {"answer": 30}}
    ]""")
    
    seeds = load_seeds(seeds_file)
    assert len(seeds) == 3
    assert seeds[0].id == "test-1"
    assert seeds[0].prompt == "Test prompt 1"
    assert seeds[1].metadata == {"key": "value"}
    assert seeds[2].label == {"answer": 30}


def test_load_seeds_duplicate_ids(tmp_path: Path) -> None:
    """Test loading seeds with duplicate IDs fails."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text("""[
        {"id": "test-1", "prompt": "Test prompt 1"},
        {"id": "test-1", "prompt": "Test prompt 2"}
    ]""")
    
    with pytest.raises(PackError, match="Duplicate seed IDs"):
        load_seeds(seeds_file)


def test_load_seeds_missing_fields(tmp_path: Path) -> None:
    """Test loading seeds with missing fields fails."""
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text("""[{"id": "test-1"}]""")
    
    with pytest.raises(PackError, match="missing required fields"):
        load_seeds(seeds_file)


def test_load_seeds_file_not_found() -> None:
    """Test loading from non-existent file."""
    with pytest.raises(PackError, match="not found"):
        load_seeds(Path("/nonexistent/seeds.json"))


def test_load_pack_file_valid(tmp_path: Path) -> None:
    """Test loading valid pack file."""
    pack_file = tmp_path / "test.yaml"
    pack_file.write_text("""
id: test-pack
version: "1.0.0"
severity: high
detectors:
  - type: regex_set
    category: test
    patterns: ["test.*pattern"]
  - type: keyword
    category: test2
    keywords: ["keyword1", "keyword2"]
""")
    
    packs = load_pack_file(pack_file)
    assert len(packs) == 1
    assert packs[0].id == "test-pack"
    assert packs[0].version == "1.0.0"
    assert packs[0].severity == "high"
    assert len(packs[0].detectors) == 2


def test_load_pack_file_multiple_docs(tmp_path: Path) -> None:
    """Test loading YAML file with multiple documents."""
    pack_file = tmp_path / "multi.yaml"
    pack_file.write_text("""
id: pack1
version: "1.0.0"
severity: low
detectors:
  - type: keyword
    category: cat1
    keywords: ["test"]
---
id: pack2
version: "2.0.0"
severity: medium
detectors:
  - type: validator
    category: val1
    validator_name: sum_equals
    expected: from_seed
---
id: pack3
version: "3.0.0"
severity: high
detectors:
  - type: validator
    category: val2
    validator_name: sum_equals
    expected: "computed:sum_prompt_numbers"
""")
    
    packs = load_pack_file(pack_file)
    assert len(packs) == 3
    assert packs[0].id == "pack1"
    assert packs[1].id == "pack2"
    assert packs[1].detectors[0].expected == "from_seed"
    assert packs[2].id == "pack3"
    assert packs[2].detectors[0].expected == "computed:sum_prompt_numbers"


def test_load_pack_file_invalid_yaml(tmp_path: Path) -> None:
    """Test loading invalid YAML fails."""
    pack_file = tmp_path / "invalid.yaml"
    pack_file.write_text("invalid: yaml: content:")
    
    with pytest.raises(PackError, match="Invalid YAML"):
        load_pack_file(pack_file)


def test_load_packs_from_directory(tmp_path: Path) -> None:
    """Test loading packs from directory."""
    (tmp_path / "pack1.yaml").write_text("""
id: pack1
version: "1.0.0"
severity: low
detectors:
  - type: keyword
    category: test
    keywords: ["test"]
""")
    
    (tmp_path / "pack2.yml").write_text("""
id: pack2
version: "1.0.0"
severity: high
detectors:
  - type: regex_set
    category: test
    patterns: ["pattern"]
""")
    
    packs = load_packs(tmp_path)
    assert len(packs) == 2
    ids = {p.id for p in packs}
    assert ids == {"pack1", "pack2"}


def test_load_packs_duplicate_ids(tmp_path: Path) -> None:
    """Test loading packs with duplicate IDs fails."""
    (tmp_path / "pack1.yaml").write_text("""
id: duplicate
version: "1.0.0"
severity: low
detectors:
  - type: keyword
    category: test
    keywords: ["test"]
""")
    
    (tmp_path / "pack2.yaml").write_text("""
id: duplicate
version: "2.0.0"
severity: high
detectors:
  - type: keyword
    category: test
    keywords: ["test"]
""")
    
    with pytest.raises(PackError, match="Duplicate pack IDs"):
        load_packs(tmp_path)


def test_validate_pack_no_detectors() -> None:
    """Test validating pack with no detectors fails."""
    pack = Pack(
        id="test",
        version="1.0.0",
        severity="low",
        detectors=[]
    )
    
    with pytest.raises(PackError, match="has no detectors"):
        validate_pack(pack)


def test_validate_pack_invalid_validator() -> None:
    """Test validating pack with unknown validator fails."""
    pack = Pack(
        id="test",
        version="1.0.0",
        severity="low",
        detectors=[{
            "type": "validator",
            "category": "test",
            "validator_name": "unknown_validator"
        }]
    )
    
    with pytest.raises(PackError, match="unknown validator"):
        validate_pack(pack)