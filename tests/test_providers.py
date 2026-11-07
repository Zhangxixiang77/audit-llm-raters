import json

import httpx
import pytest

from audit_llm_raters.scoring import client as client_module
from audit_llm_raters.scoring.client import (
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
)


def _mock_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda timeout: real_client(transport=transport, timeout=timeout))


def test_openai_compatible_payload_retry_usage_and_cost(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "busy"})
        body = json.loads(request.content)
        assert request.url == "https://provider.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        assert body["model"] == "dated-model"
        assert body["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "essay prompt"},
        ]
        assert body["max_tokens"] == 123
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning_effort"] == "none"
        assert "temperature" not in body
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"score": 4}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    _mock_http(monkeypatch, handler)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-secret")
    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(client_module.random, "random", lambda: 0.0)
    provider = OpenAICompatibleProvider({
        "model_version": "dated-model", "base_url": "https://provider.test/v1",
        "api_key_env": "TEST_PROVIDER_KEY", "temperature": None, "max_output_tokens": 123,
        "extra_body": {"reasoning_effort": "none"}, "max_retries": 2,
        "pricing": {"input_per_million": 2, "output_per_million": 10},
    })
    result = provider.generate("system prompt", "essay prompt")
    assert calls == 2
    assert result.content == '{"score": 4}'
    assert result.tokens_in == 100
    assert result.tokens_out == 20
    assert result.cost == pytest.approx(0.0004)
    assert result.metadata["finish_reason"] == "stop"


def test_openai_compatible_supports_completion_token_field(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 200
        assert "max_tokens" not in body
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"score": 4}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    _mock_http(monkeypatch, handler)
    provider = OpenAICompatibleProvider({
        "model_version": "reasoning-model", "base_url": "https://provider.test/v1",
        "max_tokens_field": "max_completion_tokens",
    })
    assert provider.generate("system", "user").tokens_out == 5


def test_openai_compatible_fatal_client_error_is_not_retried(monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "bad credentials"})

    _mock_http(monkeypatch, handler)
    provider = OpenAICompatibleProvider({
        "model_version": "dated-model", "base_url": "https://provider.test/v1", "max_retries": 5,
    })
    with pytest.raises(RuntimeError, match="HTTP 401"):
        provider.generate("system", "user")
    assert calls == 1


def test_anthropic_payload_usage_and_cost(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert request.url == "https://anthropic.test/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-test-secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert body["system"] == "system prompt"
        assert body["messages"] == [{"role": "user", "content": "essay prompt"}]
        assert body["temperature"] == 0
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": '{"score": 3,'}, {"type": "text", "text": ' "reason": "ok"}'}],
            "usage": {"input_tokens": 50, "output_tokens": 10}, "stop_reason": "end_turn",
        })

    _mock_http(monkeypatch, handler)
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-test-secret")
    provider = AnthropicProvider({
        "model_version": "claude-dated", "base_url": "https://anthropic.test/v1/messages",
        "api_key_env": "TEST_ANTHROPIC_KEY", "temperature": 0,
        "extra_body": {"thinking": {"type": "disabled"}},
        "pricing": {"input_per_million": 3, "output_per_million": 15},
    })
    result = provider.generate("system prompt", "essay prompt")
    assert result.content == '{"score": 3, "reason": "ok"}'
    assert result.tokens_in == 50
    assert result.tokens_out == 10
    assert result.cost == pytest.approx(0.0003)
    assert result.metadata["stop_reason"] == "end_turn"


def test_gemini_payload_usage_and_cost(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert request.url == "https://gemini.test/v1beta/models/gemini-dated:generateContent"
        assert request.headers["x-goog-api-key"] == "gemini-test-secret"
        assert body["systemInstruction"] == {"parts": [{"text": "system prompt"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "essay prompt"}]}]
        generation = body["generationConfig"]
        assert generation["responseMimeType"] == "application/json"
        assert generation["thinkingConfig"] == {"thinkingBudget": 0}
        assert "temperature" not in generation
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": '{"score": 2}' }]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 70, "candidatesTokenCount": 14, "thoughtsTokenCount": 6,
            },
        })

    _mock_http(monkeypatch, handler)
    monkeypatch.setenv("TEST_GEMINI_KEY", "gemini-test-secret")
    provider = GeminiProvider({
        "model_version": "gemini-dated", "base_url": "https://gemini.test/v1beta/models",
        "api_key_env": "TEST_GEMINI_KEY", "temperature": None,
        "generation_config": {"thinkingConfig": {"thinkingBudget": 0}},
        "pricing": {"input_per_million": 2, "output_per_million": 10},
    })
    result = provider.generate("system prompt", "essay prompt")
    assert result.content == '{"score": 2}'
    assert result.tokens_in == 70
    assert result.tokens_out == 20
    assert result.cost == pytest.approx(0.00034)
    assert result.metadata["finish_reason"] == "STOP"
