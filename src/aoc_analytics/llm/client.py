"""
LLM client abstraction supporting multiple providers.

Supports:
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude)
- Ollama (local models)
- OpenRouter (multi-provider)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    
    provider: Literal["openai", "anthropic", "ollama", "openrouter"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    
    def __post_init__(self):
        # Auto-detect API keys from environment
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Set default base URLs
        if self.base_url is None:
            if self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a completion for the given prompt."""
        pass
    
    @abstractmethod
    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate a JSON completion for the given prompt."""
        pass
    
    def extract_structured(
        self, 
        prompt: str, 
        schema: dict,
        system: Optional[str] = None
    ) -> dict:
        """Extract structured data matching a JSON schema."""
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
```json
{schema_str}
```

JSON response:"""
        return self.complete_json(full_prompt, system)


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content
    
    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
    
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
            
        response = self._client.messages.create(**kwargs)
        return response.content[0].text
    
    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        # Claude doesn't have native JSON mode, so we prompt for it
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no markdown or explanation."
        content = self.complete(json_prompt, system)
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())


class OllamaClient(LLMClient):
    """Ollama local LLM client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("httpx package required: pip install httpx")
    
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        response = self._httpx.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."
        content = self.complete(json_prompt, system)
        # Try to extract JSON from response
        if "{" in content:
            start = content.index("{")
            end = content.rindex("}") + 1
            content = content[start:end]
        return json.loads(content)


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Get an LLM client based on configuration."""
    if config is None:
        # Auto-detect based on available API keys
        if os.getenv("OPENAI_API_KEY"):
            config = LLMConfig(provider="openai", model="gpt-4o-mini")
        elif os.getenv("ANTHROPIC_API_KEY"):
            config = LLMConfig(provider="anthropic", model="claude-3-haiku-20240307")
        elif os.getenv("OPENROUTER_API_KEY"):
            config = LLMConfig(provider="openrouter", model="openai/gpt-4o-mini")
        else:
            # Default to Ollama for local
            config = LLMConfig(provider="ollama", model="llama3.2")
    
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "openrouter": OpenAIClient,  # OpenRouter uses OpenAI-compatible API
    }
    
    client_class = clients.get(config.provider)
    if client_class is None:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    return client_class(config)


# Singleton for reuse
_default_client: Optional[LLMClient] = None

def get_default_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = get_llm_client()
    return _default_client
