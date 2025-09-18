"""Model interface supporting stub and real LLM providers."""

import hashlib
import logging
import os
import random
from typing import Any, Optional

from valence.util import timer

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Model execution error."""
    pass


class StubModel:
    """Stub model with deterministic failure patterns."""
    
    def __init__(self, failure_rate: float = 0.3, error_rate: float = 0.05):
        """Initialize stub model.
        
        Args:
            failure_rate: Probability of generating a failing response (0.0-1.0)
            error_rate: Probability of returning None/error (0.0-1.0)
        """
        self.failure_rate = min(max(failure_rate, 0.0), 1.0)
        self.error_rate = min(max(error_rate, 0.0), 1.0)
        self.name = "stub"
    
    def generate(self, prompt: str, **kwargs: Any) -> tuple[Optional[str], float, Optional[str]]:
        """Generate response for prompt.
        
        Returns:
            Tuple of (response, duration_ms, error)
        """
        with timer() as t:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            seed = int(prompt_hash[:8], 16)
            random.seed(seed)
            
            if random.random() < self.error_rate:
                logger.debug(f"Model error for prompt hash {prompt_hash[:8]}")
                return None, t["duration_ms"], "Stub model simulated error"
            
            if random.random() < self.failure_rate:
                failing_responses = [
                    "I cannot help with that request.",
                    "That's inappropriate content.",
                    "Error: malicious intent detected",
                    "WARNING: unsafe operation",
                    "Access denied for security reasons",
                    '{"error": "forbidden", "code": 403}',
                    "System alert: potential threat",
                ]
                response = random.choice(failing_responses)
                logger.debug(f"Failing response for prompt hash {prompt_hash[:8]}")
            else:
                safe_responses = [
                    "Here's a helpful response to your question.",
                    "I'd be happy to assist with that.",
                    "The answer is 42.",
                    '{"result": "success", "data": [1, 2, 3]}',
                    "Task completed successfully.",
                    "Based on my analysis, everything looks good.",
                ]
                response = random.choice(safe_responses)
                logger.debug(f"Safe response for prompt hash {prompt_hash[:8]}")
            
            # Special handling for math prompts
            if "sum" in prompt.lower() or "add" in prompt.lower():
                numbers = []
                for word in prompt.split():
                    try:
                        numbers.append(int(word))
                    except ValueError:
                        continue
                if numbers:
                    if random.random() < 0.5:
                        response = str(sum(numbers))
                    else:
                        response = str(sum(numbers) + random.randint(-5, 5))
            
            # Special handling for JSON prompts
            if "json" in prompt.lower():
                if random.random() < 0.7:
                    response = '{"valid": true, "message": "JSON response"}'
                else:
                    response = '{"invalid": true, broken'
        
        return response, t["duration_ms"], None


class OpenAIModel:
    """OpenAI API model."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs: Any):
        """Initialize OpenAI model."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ModelError("OPENAI_API_KEY environment variable not set")
        
        try:
            import openai
        except ImportError:
            raise ModelError("openai package not installed. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 500)
        self.timeout = kwargs.get("timeout", 30)
    
    def generate(self, prompt: str, **kwargs: Any) -> tuple[Optional[str], float, Optional[str]]:
        """Generate response using OpenAI API."""
        with timer() as t:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content
                logger.debug(f"OpenAI response: {len(content)} chars, {response.usage.total_tokens} tokens")
                return content, t["duration_ms"], None
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return None, t["duration_ms"], f"OpenAI API error: {str(e)}"


class AnthropicModel:
    """Anthropic API model."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307", **kwargs: Any):
        """Initialize Anthropic model."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ModelError("ANTHROPIC_API_KEY environment variable not set")
        
        try:
            import anthropic
        except ImportError:
            raise ModelError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 500)
        self.timeout = kwargs.get("timeout", 30)
    
    def generate(self, prompt: str, **kwargs: Any) -> tuple[Optional[str], float, Optional[str]]:
        """Generate response using Anthropic API."""
        with timer() as t:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = response.content[0].text
                logger.debug(f"Anthropic response: {len(content)} chars")
                return content, t["duration_ms"], None
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                return None, t["duration_ms"], f"Anthropic API error: {str(e)}"


class AzureOpenAIModel:
    """Azure OpenAI API model."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs: Any):
        """Initialize Azure OpenAI model."""
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        
        if not all([api_key, endpoint, deployment]):
            raise ModelError(
                "Azure OpenAI requires AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, "
                "and AZURE_OPENAI_DEPLOYMENT environment variables"
            )
        
        try:
            import openai
        except ImportError:
            raise ModelError("openai package not installed. Run: pip install openai")
        
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
        )
        self.deployment = deployment
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 500)
        self.timeout = kwargs.get("timeout", 30)
    
    def generate(self, prompt: str, **kwargs: Any) -> tuple[Optional[str], float, Optional[str]]:
        """Generate response using Azure OpenAI API."""
        with timer() as t:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content
                logger.debug(f"Azure OpenAI response: {len(content)} chars")
                return content, t["duration_ms"], None
            except Exception as e:
                logger.error(f"Azure OpenAI API error: {e}")
                return None, t["duration_ms"], f"Azure OpenAI API error: {str(e)}"


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse model string into provider and model name.
    
    Args:
        model_string: Model string in format "provider:model" or "stub"
    
    Returns:
        Tuple of (provider, model_name)
    """
    if ":" in model_string:
        parts = model_string.split(":", 1)
        return parts[0], parts[1]
    return model_string, ""


class Model:
    """Unified model interface."""
    
    def __init__(self, model_string: str = "stub", **kwargs: Any):
        """Initialize model based on provider string.
        
        Args:
            model_string: Model specification (e.g., "stub", "openai:gpt-4", "anthropic:claude-3")
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)
        """
        provider, model_name = parse_model_string(model_string)
        self.provider = provider
        self.model_name = model_name or provider
        self.name = model_string
        
        if provider == "stub":
            self.impl = StubModel(**kwargs)
        elif provider == "openai":
            self.impl = OpenAIModel(model_name or "gpt-4o-mini", **kwargs)
        elif provider == "anthropic":
            self.impl = AnthropicModel(model_name or "claude-3-haiku-20240307", **kwargs)
        elif provider == "azure-openai":
            self.impl = AzureOpenAIModel(model_name or "gpt-4", **kwargs)
        else:
            raise ModelError(
                f"Unknown provider: {provider}. "
                f"Supported: stub, openai, anthropic, azure-openai"
            )
        
        logger.info(f"Initialized {provider} model: {self.name}")
    
    def generate(self, prompt: str) -> tuple[Optional[str], float]:
        """Generate response for prompt.
        
        Returns:
            Tuple of (response, duration_ms)
        """
        response, duration, error = self.impl.generate(prompt)
        if error:
            logger.warning(f"Model error: {error}")
        return response, duration