# Testing AI Agents with Valence

Valence can evaluate AI agents, tools, and APIs using LLM judges for sophisticated assessment of complex responses.

## Setup

### 1. Configure Your Agent as a Model

Implement your agent as a custom model in `valence/model.py`:

```python
def call_agent_api(prompt: str, config: dict) -> str:
    """Call your agent's API endpoint."""
    response = requests.post(
        config["endpoint"], 
        json={"prompt": prompt},
        headers={"Authorization": f"Bearer {config['api_key']}"}
    )
    return response.json()["response"]
```

### 2. Set Environment Variables

```bash
export AGENT_API_URL="https://your-agent.com/api/chat"
export AGENT_API_KEY="your-api-key"

# For LLM judges
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Create Agent-Specific Tests

**seeds.json** - Agent tasks to test:
```json
[
  {
    "id": "search-1", 
    "prompt": "Find Python courses under $50"
  },
  {
    "id": "complex-1",
    "prompt": "Research competitors for project management software, create comparison table"
  },
  {
    "id": "task-1", 
    "prompt": "Schedule meeting with John for next Tuesday 2pm"
  }
]
```

**packs/agent.yaml** - Agent-specific evaluation:
```yaml
id: agent-evaluation
version: "1.0.0"
severity: high
detectors:
  # Task completion assessment
  - type: llm_judge
    category: completion
    judge_model: "openai:gpt-4o"
    judge_prompt: |
      Did the agent complete the requested task successfully?
      
      Task: {original_prompt}
      Agent Response: {response}
      
      Consider:
      - Did it understand the request?
      - Did it provide the requested information/action?
      - Is the response complete and accurate?
      
      Score 1.0 for task failure, 0.0 for success.
      Format: SCORE: X.X REASON: explanation
  
  # Response quality
  - type: llm_judge
    category: quality
    judge_model: "openai:gpt-4o-mini"
    judge_prompt: |
      Rate the quality of this agent response:
      {response}
      
      Consider:
      - Clarity and helpfulness
      - Appropriate level of detail
      - Professional tone
      
      Score 1.0 for poor quality, 0.0 for good quality.
      Format: SCORE: X.X REASON: explanation

  # Safety check
  - type: keyword
    category: safety
    keywords: ["error", "failed", "unavailable", "cannot process"]
```

## Running Agent Evaluations

### Basic Evaluation
```bash
valence run \
  --model your-agent \
  --seeds agent-seeds.json \
  --packs packs/agent.yaml \
  --out runs/agent-test/
```

### Advanced with Mutations
```bash
valence run \
  --model your-agent \
  --seeds agent-seeds.json \
  --packs packs/agent.yaml \
  --out runs/agent-test/ \
  --llm-mutations \
  --mutation-model openai:gpt-4o-mini \
  --max-gens 2
```

## Agent-Specific Detection Patterns

### Task Completion Assessment
```yaml
- type: llm_judge
  category: task_completion
  judge_prompt: |
    The user requested: {original_prompt}
    Agent provided: {response}
    
    Did the agent complete the specific task requested?
    Score 1.0 if task was not completed, 0.0 if completed successfully.
```

### Information Accuracy
```yaml
- type: llm_judge
  category: accuracy
  judge_prompt: |
    Evaluate the factual accuracy of this response: {response}
    Context: {original_prompt}
    
    Score 1.0 for inaccurate information, 0.0 for accurate.
```

### Communication Quality
```yaml
- type: llm_judge
  category: communication
  judge_prompt: |
    Rate the communication quality of: {response}
    
    Consider:
    - Clarity and coherence
    - Appropriate tone for the request
    - Completeness of information
    
    Score 1.0 for poor communication, 0.0 for good.
```

### Tool Usage Validation
```yaml
- type: llm_judge
  category: tool_usage
  judge_prompt: |
    Task: {original_prompt}
    Agent Action: {response}
    
    Did the agent use appropriate tools/methods for this task?
    Score 1.0 for inappropriate tool usage, 0.0 for appropriate.
```

## Example Use Cases

### Customer Service Agent
```yaml
detectors:
  - type: llm_judge
    category: helpfulness
    judge_prompt: |
      Customer query: {original_prompt}
      Agent response: {response}
      
      Is this response helpful for the customer?
      Score 1.0 for unhelpful, 0.0 for helpful.
  
  - type: keyword
    category: safety
    keywords: ["can't help", "don't know", "error occurred"]
```

### Research Agent
```yaml
detectors:
  - type: llm_judge
    category: research_quality
    judge_prompt: |
      Research request: {original_prompt}
      Findings: {response}
      
      Does this provide comprehensive, relevant research results?
      Score 1.0 for poor research, 0.0 for good research.
```

### Code Generation Agent
```yaml
detectors:
  - type: regex_set
    category: code_format
    patterns: ["```\\w+", "def \\w+\\(", "class \\w+:"]
  
  - type: llm_judge
    category: code_quality
    judge_prompt: |
      Code request: {original_prompt}
      Generated code: {response}
      
      Is this working, appropriate code for the request?
      Score 1.0 for bad code, 0.0 for good code.
```

## Best Practices

### Judge Model Selection
- **Fast evaluation**: `claude-3-haiku`, `gpt-4o-mini`
- **Complex reasoning**: `claude-3-sonnet`, `gpt-4o`
- **Critical assessment**: `claude-3-opus`

### Cost Management
- Use cheaper models for initial testing
- Combine keyword/regex filters with LLM judges
- Limit mutation generations during development

### Agent-Specific Tips
- Test edge cases that break typical workflows
- Include both simple and complex multi-step tasks  
- Validate tool/API integration points
- Check error handling and graceful degradation
- Test with malformed or ambiguous inputs

### Evaluation Strategy
1. Start with basic task completion checks
2. Add response quality assessment
3. Include safety and error handling tests
4. Test with edge cases and complex scenarios
5. Use mutations to explore failure boundaries