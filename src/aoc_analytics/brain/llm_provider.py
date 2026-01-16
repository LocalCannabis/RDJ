"""
LLM Provider - Unified interface for AI text generation.

Supports multiple backends:
- OpenAI (GPT-4, GPT-3.5)
- Ollama (local LLMs like llama3.2)

Set LLM_PROVIDER=openai and OPENAI_API_KEY to use OpenAI.
Defaults to Ollama if available, otherwise falls back to templates.
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    model: str
    provider: str
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class LLMProvider:
    """
    Unified LLM provider supporting OpenAI and Ollama.
    
    Usage:
        llm = LLMProvider()  # Auto-detects best available provider
        response = llm.generate("Summarize these sales patterns...")
        
        # Or force a specific provider:
        llm = LLMProvider(provider="openai")
    """
    
    # Default models
    OPENAI_MODEL = "gpt-4o-mini"  # Fast and cheap, great for summaries
    OLLAMA_MODEL = "llama3.2:3b"
    
    def __init__(
        self, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize LLM provider.
        
        Args:
            provider: 'openai', 'ollama', or None (auto-detect)
            model: Model name override
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._provider = provider or os.environ.get("LLM_PROVIDER")
        self._model = model
        
        # Auto-detect provider if not specified
        if self._provider is None:
            self._provider = self._detect_provider()
        
        # Set model based on provider
        if self._model is None:
            if self._provider == "openai":
                self._model = self.OPENAI_MODEL
            else:
                self._model = self.OLLAMA_MODEL
    
    def _detect_provider(self) -> str:
        """Auto-detect best available provider."""
        # Prefer OpenAI if API key is set
        if self.openai_api_key:
            return "openai"
        
        # Check if Ollama is available
        if self._check_ollama():
            return "ollama"
        
        # Fallback to template (no LLM)
        return "template"
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @property
    def provider(self) -> str:
        """Current provider name."""
        return self._provider
    
    @property
    def model(self) -> str:
        """Current model name."""
        return self._model
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return self._provider in ("openai", "ollama")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Creativity (0-1)
            max_tokens: Max response length
        
        Returns:
            LLMResponse with generated text
        """
        if self._provider == "openai":
            return self._generate_openai(prompt, system, temperature, max_tokens)
        elif self._provider == "ollama":
            return self._generate_ollama(prompt, system, temperature, max_tokens)
        else:
            return LLMResponse(
                text="",
                model="none",
                provider="template",
                success=False,
                error="No LLM provider available"
            )
    
    def _generate_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        try:
            import httpx
        except ImportError:
            try:
                import requests as httpx
            except ImportError:
                return LLMResponse(
                    text="",
                    model=self._model,
                    provider="openai",
                    success=False,
                    error="httpx or requests required for OpenAI"
                )
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            
            if response.status_code != 200:
                return LLMResponse(
                    text="",
                    model=self._model,
                    provider="openai",
                    success=False,
                    error=f"OpenAI API error: {response.status_code} - {response.text[:200]}"
                )
            
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                text=text,
                model=self._model,
                provider="openai",
                tokens_used=tokens,
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                text="",
                model=self._model,
                provider="openai",
                success=False,
                error=str(e)
            )
    
    def _generate_ollama(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using Ollama."""
        try:
            cmd = ["ollama", "run", self._model]
            
            full_prompt = prompt
            if system:
                full_prompt = f"System: {system}\n\nUser: {prompt}"
            
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode == 0:
                return LLMResponse(
                    text=result.stdout.strip(),
                    model=self._model,
                    provider="ollama",
                    success=True
                )
            else:
                return LLMResponse(
                    text="",
                    model=self._model,
                    provider="ollama",
                    success=False,
                    error=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return LLMResponse(
                text="",
                model=self._model,
                provider="ollama",
                success=False,
                error="Ollama timeout"
            )
        except Exception as e:
            return LLMResponse(
                text="",
                model=self._model,
                provider="ollama",
                success=False,
                error=str(e)
            )


# Global instance for convenience
_default_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get the default LLM provider (singleton)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = LLMProvider()
    return _default_provider


def generate(prompt: str, system: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function to generate text.
    
    Returns empty string if LLM unavailable.
    """
    provider = get_llm_provider()
    response = provider.generate(prompt, system, **kwargs)
    return response.text if response.success else ""


# Test function
def test_provider():
    """Test the LLM provider."""
    print("Testing LLM Provider...")
    print(f"  OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"  LLM_PROVIDER env: {os.environ.get('LLM_PROVIDER', 'not set')}")
    
    llm = LLMProvider()
    print(f"\n  Provider: {llm.provider}")
    print(f"  Model: {llm.model}")
    print(f"  Available: {llm.is_available}")
    
    if llm.is_available:
        print("\n  Testing generation...")
        response = llm.generate(
            "In one sentence, what makes cannabis retail unique?",
            system="You are a retail analytics expert.",
            max_tokens=100
        )
        print(f"  Success: {response.success}")
        if response.success:
            print(f"  Response: {response.text[:200]}...")
            print(f"  Tokens: {response.tokens_used}")
        else:
            print(f"  Error: {response.error}")
    
    return llm


if __name__ == "__main__":
    test_provider()
