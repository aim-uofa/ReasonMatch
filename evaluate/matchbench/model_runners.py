from __future__ import annotations

import abc
import time
from typing import Any, Optional

import requests


class BaseRunner(abc.ABC):
    """Abstract interface for text-only chat completions."""

    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def run(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return the raw string response for the given chat messages."""


class OpenAIChatRunner(BaseRunner):
    """Requests-based runner for OpenAI-compatible APIs (including vLLM)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        extra_kwargs: Optional[dict[str, Any]] = None,
        retries: int = 3,
        timeout: float = 300.0,
    ):
        super().__init__(model)
        self.session = requests.Session()
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.extra_kwargs = extra_kwargs or {}
        self.retries = max(1, retries)
        self.timeout = timeout

    def run(self, messages, temperature=0.0, max_tokens=None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.extra_kwargs,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        raise last_exc  # type: ignore[misc]


def VLLMChatRunner(
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    extra_kwargs: Optional[dict[str, Any]] = None,
    retries: int = 3,
    timeout: float = 600.0,
) -> OpenAIChatRunner:
    """Factory for vLLM servers — just an OpenAIChatRunner with different defaults."""
    return OpenAIChatRunner(
        model=model,
        api_key=api_key,
        base_url=base_url,
        extra_kwargs=extra_kwargs,
        retries=retries,
        timeout=timeout,
    )
