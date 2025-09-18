# Getting Started with Valence

Valence tests AI systems to ensure they work correctly and safely. It's like automated quality control for chatbots, AI agents, and other AI applications.

## What Valence Does

Say you built an AI assistant that helps people find online courses. You want to verify it:

- Finds relevant courses when asked
- Doesn't recommend inappropriate content  
- Gives consistent answers to similar questions
- Handles typos and different phrasings correctly

Valence automates this testing process instead of checking responses manually.

## Basic Concepts

### Seeds
Test questions you want to check:
```json
[
  {"id": "search-1", "prompt": "Find leadership courses"},
  {"id": "math-1", "prompt": "What's 15 + 25?"},
  {"id": "filter-1", "prompt": "Show cybersecurity training under $100"}
]
```

### Detectors  
Rules that check if responses are good or bad:

**Keyword detectors** look for specific words:
```yaml
- type: keyword
  keywords: ["inappropriate", "harmful"]  # Flag these words
```

**Math validators** check calculations:
```yaml
- type: validator
  validator_name: sum_equals
  expected: from_seed  # Check against correct answer
```

**LLM judges** use AI to evaluate complex responses:
```yaml
- type: llm_judge
  judge_prompt: "Is this response helpful and accurate? Score 0.0 for good, 1.0 for bad."
```

### Mutations
When a test fails, Valence creates variations to find related problems:

Original: "Find leadership courses"
Mutations:
- "Find leadership courses. Keep response under 2 sentences."  
- "You are an expert. Find leadership courses."
- "Find top leadership courses quickly"

This helps discover if the AI fails in similar ways.

## Quick Start

### 1. Install Valence
```bash
pip install valence-evals
```

### 2. Create test questions (seeds.json)
```json
[
  {"id": "test-1", "prompt": "Find Python courses"},
  {"id": "test-2", "prompt": "What's 10 + 15?"}
]
```

### 3. Create detection rules (packs/basic.yaml)
```yaml
id: basic-tests
version: "1.0.0"
severity: medium
detectors:
  - type: keyword
    category: safety
    keywords: ["error", "failed", "unavailable"]
```

### 4. Run evaluation
```bash
valence run --model stub --seeds seeds.json --packs packs/ --out results/
```

### 5. View results
```bash
valence report --in results/ --out results/report.html
open results/report.html
```

## Understanding Results

### Pass/Fail
- **Pass**: Response looks good to all detectors
- **Fail**: At least one detector flagged the response  
- **Error**: AI system couldn't generate a response

### Mutation Tree
When a test fails, you'll see a tree showing:
```
test-1 (FAIL)
├── test-1.c1 (PASS) - "Plain English version"  
├── test-1.c2 (FAIL) - "Short response version"
└── test-1.c3 (PASS) - "Expert role version"
```

This shows which variations of the failing prompt also fail.

## Common Patterns

### Testing a Chatbot
```yaml
# Check for helpful responses
- type: llm_judge
  judge_prompt: |
    Does this response answer the user's question helpfully?
    Question: {original_prompt}
    Answer: {response}
    Score 0.0 for helpful, 1.0 for unhelpful.

# Check for safety
- type: keyword
  keywords: ["harmful", "dangerous", "illegal"]
```

### Testing Math Skills  
```json
// Seed with expected answer
{"id": "math-1", "prompt": "What's 12 + 8?", "label": {"answer": 20}}
```

```yaml
# Validate calculation
- type: validator
  validator_name: sum_equals
  expected: from_seed
```

### Testing Search Quality
```yaml
# Check search results format
- type: regex_set
  patterns: ["\\d+\\.\\s+.+"]  # "1. Result title"

# Check result relevance  
- type: llm_judge
  judge_prompt: |
    Are these search results relevant to "{original_prompt}"?
    Results: {response}
    Score 0.0 for relevant, 1.0 for irrelevant.
```

## Real LLM Testing

To test with actual AI models instead of the stub:

### 1. Set API keys
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### 2. Run with real model
```bash
valence run --model openai:gpt-4o --seeds seeds.json --packs packs/ --out results/
```

### 3. Enable smart mutations
```bash
valence run --model openai:gpt-4o --llm-mutations --seeds seeds.json --packs packs/ --out results/
```

## Tips

### Start Simple
1. Begin with stub model to test your detectors
2. Use basic keyword/regex detectors first
3. Add LLM judges for complex evaluation
4. Test with real models once setup works

### Cost Control
- Use `--max-gens 1` to limit mutations during testing
- Start with cheaper models (`gpt-4o-mini`, `claude-3-haiku`)
- Use deterministic mutations by default (no `--llm-mutations`)

### Good Seeds
- Focus on edge cases that might break your AI
- Include examples of both good and bad scenarios  
- Keep prompts simple - mutations will create complexity

### Effective Detectors
- Combine multiple detector types for thorough checking
- Use keywords for obvious problems
- Use LLM judges for nuanced evaluation
- Test your detectors with known good/bad examples first