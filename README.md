# Valence - An AI Evaluation Framework

Adaptive evaluation framework for testing LLMs and AI agents. Automatically generates variations of failing prompts to find related issues.

## Documentation

- **[Getting Started](docs/BEGINNER_GUIDE.md)** - Quick start guide
- **[Technical Docs](docs/DOCUMENTATION.md)** - Full technical reference
- **[Testing AI Agents](docs/AGENTIC_TESTING.md)** - Guide for testing autonomous systems
- **[LLM Judges](docs/LLM_JUDGE_SUMMARY.md)** - Using LLMs as evaluators
- **[Provider Setup](docs/LLM_PROVIDERS.md)** - Configuring AI providers

## Installation

```bash
# Basic installation
pip install -e .

# With LLM provider support
pip install -e ".[llm]"
```

## Quick Start

### 1. Test with stub model (no API needed)
```bash
valence run \
  --model stub \
  --seeds ./seeds.json \
  --packs ./packs/ \
  --out ./runs/test-001/
```

### 2. Test with real LLMs
```bash
# Set API keys
export OPENAI_API_KEY="your-key"

# Run evaluation
valence run \
  --model openai:gpt-4o \
  --seeds ./seeds.json \
  --packs ./packs/ \
  --out ./runs/openai-001/
```

### 3. Generate report
```bash
valence report --in ./runs/test-001/ --out ./runs/test-001/report.html
```

## Supported Models

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| Stub | `stub` | None needed |
| OpenAI | `gpt-4o`, `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-sonnet`, `claude-3-haiku` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `gpt-4` | `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

## Key Features

### Adaptive Testing
When tests fail, Valence generates mutations to explore failure patterns:
- Basic mutations: length constraints, role changes, output format
- Semantic mutations: paraphrasing, complexity changes (requires LLM)
- Noise mutations: typos, unicode substitution, whitespace
- Constraint mutations: conflicting requirements, nested filters

### Detection Methods
- **keyword**: Simple text matching
- **regex_set**: Pattern-based detection  
- **validator**: Math checking, JSON validation, etc.
- **llm_judge**: LLM-based semantic evaluation

### Expected Output Testing
Test against known correct answers:
```json
{
  "id": "math-1",
  "prompt": "What is 15 + 25?",
  "label": {"answer": 40}
}
```

## Project Structure

```
valence-evals/
├── valence/          # Core package
├── packs/            # Detector configurations
├── seeds.json        # Test prompts
├── tests/            # Test suite
└── runs/             # Results (gitignored)
```

## Detector Configuration

### Basic Pack Example
```yaml
id: basic-pack
version: "1.0.0"
severity: medium
detectors:
  - type: keyword
    category: safety
    keywords: ["error", "failed"]
    
  - type: validator
    category: math
    validator_name: sum_equals
    expected: from_seed
    
  - type: llm_judge
    category: quality
    judge_model: "openai:gpt-4o-mini"
    judge_prompt: |
      Is this response helpful?
      Response: {response}
      Score 0.0 for helpful, 1.0 for unhelpful.
```

## CLI Commands

### valence run
```bash
valence run [OPTIONS]

Required:
  --model TEXT       Model to test (stub, openai:gpt-4o, etc.)
  --seeds PATH       Seeds JSON file
  --packs PATH       Packs directory or file  
  --out PATH         Output directory

Optional:
  --max-gens INT     Max mutation generations (default: 1)
  --mutations-per-failure INT  Mutations per failure (default: 4)
  --llm-mutations    Enable semantic mutations
  --memory PATH      Failure memory file
```

### valence report
```bash
valence report --in <run_dir> --out <report.html>
```

### valence ci
```bash
valence ci <run_dir> [baseline_dir] --config ci_config.json
```

## Development

```bash
# Run tests
pytest

# With coverage
pytest --cov=valence --cov-report=term-missing

# Format code
black valence tests
ruff check valence tests
```

## Cost Management

When using real LLMs:
- Start with cheaper models (`gpt-3.5-turbo`, `claude-3-haiku`)
- Use `--max-gens 1` to limit mutations
- Monitor API usage through provider dashboards
- Use stub model for detector development

## Planned Features

### Conversational Testing
- Stateful evaluation modes for multi-turn conversations
- Context preservation across mutations
- Conversation coherence tracking

### Performance & Scale
- Parallel evaluation support
- Request rate limiting and retry logic
- Baseline comparison across runs

### Analysis
- Failure pattern clustering
- Regression detection between model versions
- Extended reporting formats

### Detection
- Additional validator types
- Custom detector framework
- Domain-specific evaluation packs

## License

MIT