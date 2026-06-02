from __future__ import annotations

import abc
import base64
import os
import time
from io import BytesIO
from typing import Any, Optional

import requests
from PIL import Image


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


# ── Shared helpers used by both run_eval.py and chk_label_cnt.py ──


def encode_image(image: str | Image.Image) -> str:
    buffer = BytesIO()
    if isinstance(image, str):
        Image.open(image).save(buffer, format="JPEG")
    elif isinstance(image, Image.Image):
        image.save(buffer, format="JPEG")
    else:
        raise ValueError("Image should be a file path or PIL Image object.")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def resolve_model_id(args) -> str:
    if args.runner != "vllm":
        return args.model_id
    if not args.base_url:
        raise RuntimeError("--base_url is required for the vllm runner")

    model_id = args.model_id or ""
    if model_id and model_id.lower() != "auto":
        return model_id

    headers = {
        "Authorization": f"Bearer {(args.api_key or os.environ.get('OPENAI_API_KEY', '') or 'EMPTY')}",
    }
    endpoint = args.base_url.rstrip("/") + "/models"
    response = requests.get(endpoint, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    models = [m["id"] for m in data.get("data", []) if "id" in m]
    if not models:
        raise RuntimeError("No models available on vLLM server.")
    resolved = models[0]
    print(f"[evaluate] Using vLLM model '{resolved}'.")
    return resolved


def make_runner(args, model_id: str) -> BaseRunner:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if args.runner == "openai":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai runner")
        return OpenAIChatRunner(model=model_id, api_key=api_key, base_url=args.base_url)
    if args.runner == "vllm":
        if not args.base_url:
            raise RuntimeError("--base_url is required for the vllm runner")
        return VLLMChatRunner(
            model=model_id, base_url=args.base_url, api_key=api_key or "EMPTY"
        )
    raise ValueError(f"Unsupported runner: {args.runner}")


def sanitise_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def dataset_group_from_key(dataset_key: str | None, dataset_name: str | None) -> str:
    key = (dataset_key or dataset_name or "unknown").strip()
    if key.startswith("imc"):
        return "imc"
    return key
