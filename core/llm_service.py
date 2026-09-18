"""Source-selector adapters for local Ollama and hosted compatible APIs.

Neither adapter is allowed to author rendered research findings. They only
select from source turns that deterministic retrieval has already found.
"""

import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

from openai import OpenAI


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def get_ollama_base_url() -> str:
    """Return a loopback-only default, optionally overridden for a private host."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def list_local_models(base_url: Optional[str] = None) -> List[str]:
    """Return models from a running Ollama server without downloading anything."""
    url = f"{(base_url or get_ollama_base_url()).rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=1.5) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError, json.JSONDecodeError):
        return []
    return [model["name"] for model in payload.get("models", []) if model.get("name")]


class OpenAICompatibleModelService:
    """Minimal shared client for an evidence-selector endpoint."""

    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key or "not-configured",
        )

    def is_configured(self) -> bool:
        return bool(self.model.strip())

    def query(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "success": False,
                "content": "",
                "error": "No evidence-selector model is configured.",
                "model_used": "deterministic-fallback",
            }
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return {
                "success": True,
                "content": response.choices[0].message.content or "",
                "model_used": self.model,
            }
        except Exception as error:
            return {
                "success": False,
                "content": "",
                "error": f"Evidence-selector request failed: {error}",
                "model_used": self.model,
            }


class LocalModelService(OpenAICompatibleModelService):
    """OpenAI-compatible client pointed at a local Ollama server."""

    def __init__(self, model: str = "", base_url: Optional[str] = None):
        # Ollama ignores this placeholder key on its loopback endpoint.
        super().__init__(model, base_url or get_ollama_base_url(), "ollama")


class HostedModelService(OpenAICompatibleModelService):
    """Server-side model adapter; its credential must never enter the browser."""

    def is_configured(self) -> bool:
        return bool(self.model.strip() and self.base_url and self.api_key)


def get_hosted_model_service() -> Optional[HostedModelService]:
    """Return a hosted adapter only when all deployment secrets are present."""
    service = HostedModelService(
        model=os.getenv("CLOUD_MODEL_NAME", ""),
        base_url=os.getenv("CLOUD_MODEL_BASE_URL", ""),
        api_key=os.getenv("CLOUD_MODEL_API_KEY", ""),
    )
    return service if service.is_configured() else None
