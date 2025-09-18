# Valence Documentation

## Overview

Valence is an adaptive evaluation framework for testing LLMs and AI agents. When a test fails, it automatically generates variations to explore related failure patterns.

## Key Features

- **Adaptive testing**: Generates mutations of failing prompts to find related issues
- **Multiple detectors**: Keyword matching, regex patterns, custom validators, and LLM judges
- **Failure memory**: Remembers patterns to avoid redundant testing
- **Mutation tracking**: Maintains lineage between original prompts and variations

## Quick Start

```bash
# Basic evaluation
valence run --model openai:gpt-4o --seeds seeds.json --packs packs/ --out runs/test-001/

# Generate report
valence report --in runs/test-001/ --out runs/test-001/report.html
```

## Core Concepts

### Seeds
Starting prompts that represent typical user inputs:
```json
[
  {"id": "search-1", "prompt": "Find leadership courses"},
  {"id": "math-1", "prompt": "What's 15 + 25?"}
]
```

### Packs
YAML files defining detection strategies:
```yaml
id: safety-pack
version: "1.0.0"
severity: high
detectors:
  - type: keyword
    category: safety
    keywords: ["harmful", "dangerous"]
```

### Detectors
- **keyword**: Simple text matching
- **regex_set**: Pattern-based detection
- **validator**: Custom logic (math validation, JSON formatting)
- **llm_judge**: LLM-based evaluation for complex assessment

### Mutations
When a prompt fails detection, Valence generates variations:
- Length constraints ("Keep response under 2 sentences")
- Style changes ("Use plain English")
- Role modifications ("You are an expert in...")
- Content variations (semantic paraphrasing with LLMs)

## Command Reference

### valence run

```bash
valence run [options]
```

**Required:**
- `--model`: Model to test (`stub`, `openai:gpt-4o`, `anthropic:claude-3-sonnet`)
- `--seeds`: Path to seed prompts JSON file
- `--packs`: Path to detector packs directory
- `--out`: Output directory for results

**Optional:**
- `--max-gens`: Maximum mutation generations (default: 1)
- `--mutations-per-failure`: Mutations per failing prompt (default: 4)
- `--llm-mutations`: Enable semantic mutations (requires API key)
- `--mutation-model`: Model for generating mutations
- `--memory`: Path to failure memory file

### valence report

```bash
valence report --in <run_dir> --out <report.html>
```

Generates an HTML report with:
- Test results summary
- Failure analysis
- Mutation lineage trees
- Performance metrics

## Pack Configuration

### Basic Structure
```yaml
id: pack-name
version: "1.0.0" 
severity: low|medium|high
detectors:
  - type: keyword
    category: safety
    keywords: ["unsafe", "harmful"]
    
  - type: regex_set
    category: format
    patterns: ["\\d{3}-\\d{2}-\\d{4}"]  # SSN pattern
    
  - type: validator
    category: math
    validator_name: sum_equals
    expected: from_seed  # or computed:function_name
    
  - type: llm_judge
    category: quality
    judge_model: "openai:gpt-4o-mini"
    judge_prompt: |
      Evaluate this response: {response}
      Original request: {original_prompt}
      
      Score 1.0 for poor quality, 0.0 for good quality.
      Format: SCORE: X.X REASON: explanation
```

### Detector Types

**keyword**
- `keywords`: List of words/phrases to detect
- Case-insensitive substring matching

**regex_set**
- `patterns`: List of regex patterns
- Any match triggers detection

**validator**
- `validator_name`: Built-in validator function
- `expected`: Where to get expected values
  - `from_seed`: Use seed.label field
  - `computed:function_name`: Call reference function
  - List of acceptable values

**llm_judge**
- `judge_model`: Model for evaluation
- `judge_prompt`: Template with `{original_prompt}` and `{response}` variables

## Model Support

### Built-in Models
- `stub`: Deterministic responses for testing

### LLM Providers
- **OpenAI**: `openai:gpt-4o`, `openai:gpt-3.5-turbo`
- **Anthropic**: `anthropic:claude-3-sonnet`, `anthropic:claude-3-haiku`
- **Azure OpenAI**: `azure-openai:gpt-4`

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment"
```

## Output Format

### Run Directory Structure
```
runs/test-001/
├── metadata.json      # Run configuration and summary
├── evals.jsonl       # Individual evaluation records  
├── lineage.jsonl     # Mutation relationships
├── scorecards.jsonl  # Aggregated results per prompt family
└── report.html       # Generated report (after valence report)
```

### Evaluation Record
```json
{
  "id": "seed-001.c1",
  "prompt": "Modified test prompt",
  "response": "Model response text",
  "scorecard": {
    "outcome": "fail",
    "score": 1.0,
    "tags": ["safety"],
    "signals": [...]
  },
  "lineage": {
    "parent_id": "seed-001",
    "generation": 1,
    "mutation_operator": "plain-english"
  },
  "model": "openai:gpt-4o",
  "duration_ms": 150.5
}
```

## Best Practices

### Detector Design
- Start with simple keyword/regex detectors
- Use LLM judges for complex semantic evaluation
- Combine multiple detector types for comprehensive coverage

### Seed Selection
- Focus on boundary cases and edge conditions
- Include both positive and negative examples
- Keep seeds simple - mutations will create variations

### Performance
- Use fast models for initial testing (`claude-3-haiku`, `gpt-4o-mini`)
- Limit generations in CI/CD pipelines
- Cache results with memory system for iterative development

### Cost Management
- Use deterministic mutations by default
- Enable LLM mutations selectively for important test suites
- Monitor API usage when using LLM judges