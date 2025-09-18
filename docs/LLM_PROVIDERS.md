# LLM Provider Support

Valence supports multiple LLM providers for testing real AI systems.

## Supported Providers

### Stub (Default)
- Built-in deterministic model for testing framework
- No API key required
- Usage: `--model stub`

### OpenAI
- Models: GPT-3.5, GPT-4, GPT-4o series
- Environment: `OPENAI_API_KEY`
- Usage: `--model openai:gpt-4o` or `--model openai:gpt-3.5-turbo`

### Anthropic
- Models: Claude 3 Haiku, Sonnet, Opus
- Environment: `ANTHROPIC_API_KEY`
- Usage: `--model anthropic:claude-3-sonnet-20241022`

### Azure OpenAI
- Azure-hosted OpenAI models
- Environment variables:
  - `AZURE_OPENAI_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT`
- Usage: `--model azure-openai:gpt-4`

## Setup

### Install Dependencies
```bash
pip install valence-evals[llm]
# or separately:
pip install openai anthropic
```

### Set API Keys

**Environment variables:**
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export AZURE_OPENAI_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

**Or .env file:**
```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

## Usage Examples

### OpenAI Evaluation
```bash
valence run \
  --model openai:gpt-4o \
  --seeds ./seeds.json \
  --packs ./packs/ \
  --out ./runs/openai-test/
```

### Anthropic Claude
```bash
valence run \
  --model anthropic:claude-3-5-sonnet-20241022 \
  --seeds ./seeds.json \
  --packs ./packs/ \
  --out ./runs/claude-test/
```

### Development Testing
```bash
valence run \
  --model stub \
  --seeds ./seeds.json \
  --packs ./packs/ \
  --out ./runs/test/
```

## Model Parameters

Default settings (configurable in code):
- `temperature`: 0.7 (randomness)
- `max_tokens`: 500 (response length)
- `timeout`: 30 seconds (API timeout)

## Error Handling

- Missing API keys: Clear error message
- API failures: Logged and recorded in results
- Network timeouts: Graceful handling
- Failed responses: Marked with error status

## Cost Management

### Model Selection by Cost
- **Cheapest**: `gpt-3.5-turbo`, `claude-3-haiku`
- **Balanced**: `gpt-4o-mini`, `claude-3-sonnet`
- **Premium**: `gpt-4o`, `claude-3-opus`

### Cost Control Tips
- Use cheaper models for initial testing
- Limit mutations with `--max-gens 1` during development
- Monitor usage through provider dashboards
- Use stub model for detector validation

### Testing Strategy
1. Develop with stub model (free)
2. Validate with cheap models (`gpt-4o-mini`)
3. Final testing with target model
4. Production monitoring with appropriate model tier