"""Provider adapters with retries, usage accounting, and no hidden credentials."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from audit_llm_raters.config import secret_from_env


@dataclass
class ProviderResponse:
    content: str
    raw_response: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    def generate(self, system: str, user: str) -> ProviderResponse: ...


def _cost(config: dict[str, Any], tokens_in: int | None, tokens_out: int | None) -> float | None:
    pricing = config.get("pricing", {})
    if tokens_in is None or tokens_out is None or not pricing:
        return None
    return (
        tokens_in * float(pricing.get("input_per_million", 0))
        + tokens_out * float(pricing.get("output_per_million", 0))
    ) / 1_000_000


class HttpProvider:
    def __init__(self, config: dict[str, Any], timeout: float = 120):
        self.config = config
        self.model = config["model_version"]
        self.timeout = timeout
        self.max_retries = int(config.get("max_retries", 5))

    def _request(self, request: httpx.Request) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.send(request)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt + 1 < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else min(60, 2**attempt)
                        time.sleep(delay + random.random())
                    continue
                if response.is_error:
                    raise RuntimeError(f"Provider returned HTTP {response.status_code}: {response.text}")
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(min(60, 2**attempt) + random.random())
        raise RuntimeError(f"Provider request failed after {self.max_retries} attempts: {last_error}")


class OpenAICompatibleProvider(HttpProvider):
    """OpenAI, DeepSeek, DashScope-compatible, OpenRouter, and local vLLM."""

    def generate(self, system: str, user: str) -> ProviderResponse:
        headers = {"Content-Type": "application/json"}
        key_env = self.config.get("api_key_env")
        if key_env:
            headers["Authorization"] = f"Bearer {secret_from_env(key_env)}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": int(self.config.get("max_output_tokens", 200)),
        }
        if self.config.get("temperature", 0) is not None:
            body["temperature"] = float(self.config.get("temperature", 0))
        if self.config.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        body.update(self.config.get("extra_body", {}))
        url = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        payload = self._request(
            httpx.Request("POST", f"{url}/chat/completions", headers=headers, json=body)
        )
        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        return ProviderResponse(
            content=choice["message"].get("content") or "",
            raw_response=json.dumps(payload, ensure_ascii=False),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=_cost(self.config, tokens_in, tokens_out),
            metadata={"finish_reason": choice.get("finish_reason")},
        )


class AnthropicProvider(HttpProvider):
    def generate(self, system: str, user: str) -> ProviderResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": int(self.config.get("max_output_tokens", 200)),
        }
        if self.config.get("temperature", 0) is not None:
            body["temperature"] = float(self.config.get("temperature", 0))
        body.update(self.config.get("extra_body", {}))
        payload = self._request(
            httpx.Request(
                "POST",
                self.config.get("base_url", "https://api.anthropic.com/v1/messages"),
                headers={
                    "x-api-key": secret_from_env(self.config.get("api_key_env", "ANTHROPIC_API_KEY")),
                    "anthropic-version": self.config.get("anthropic_version", "2023-06-01"),
                    "Content-Type": "application/json",
                },
                json=body,
            )
        )
        content = "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        )
        usage = payload.get("usage", {})
        tokens_in = usage.get("input_tokens")
        tokens_out = usage.get("output_tokens")
        return ProviderResponse(
            content=content,
            raw_response=json.dumps(payload, ensure_ascii=False),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=_cost(self.config, tokens_in, tokens_out),
            metadata={"stop_reason": payload.get("stop_reason")},
        )


class GeminiProvider(HttpProvider):
    def generate(self, system: str, user: str) -> ProviderResponse:
        generation: dict[str, Any] = {
            "maxOutputTokens": int(self.config.get("max_output_tokens", 200)),
            "responseMimeType": "application/json",
        }
        if self.config.get("temperature", 0) is not None:
            generation["temperature"] = float(self.config.get("temperature", 0))
        generation.update(self.config.get("generation_config", {}))
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        base = self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta/models")
        payload = self._request(
            httpx.Request(
                "POST",
                f"{base.rstrip('/')}/{self.model}:generateContent",
                headers={
                    "x-goog-api-key": secret_from_env(self.config.get("api_key_env", "GEMINI_API_KEY")),
                    "Content-Type": "application/json",
                },
                json=body,
            )
        )
        candidates = payload.get("candidates", [])
        content = ""
        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")
            content = "".join(
                part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])
            )
        usage = payload.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount")
        tokens_out = usage.get("candidatesTokenCount")
        return ProviderResponse(
            content=content,
            raw_response=json.dumps(payload, ensure_ascii=False),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=_cost(self.config, tokens_in, tokens_out),
            metadata={"finish_reason": finish_reason},
        )


class MockProvider:
    """Deterministic offline provider for integration tests; never for empirical results."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, system: str, user: str) -> ProviderResponse:
        del system
        score = sum(user.encode()) % 5 + 1
        content = json.dumps({"score": score, "reason": "Synthetic response for pipeline testing."})
        return ProviderResponse(content, json.dumps({"mock": content}), len(user) // 4, 12, 0.0)


def make_provider(config: dict[str, Any]) -> Provider:
    provider = config.get("provider", "openai_compatible")
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(config)
    if provider == "anthropic":
        return AnthropicProvider(config)
    if provider == "gemini":
        return GeminiProvider(config)
    if provider == "mock":
        return MockProvider(config)
    raise ValueError(f"Unknown provider: {provider}")
