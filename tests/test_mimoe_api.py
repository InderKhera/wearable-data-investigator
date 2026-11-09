import json

import httpx
import pytest

from wearable import mimoe_api

BASE = "http://localhost:8083/mimik-ai/openai/v1"
MODEL = mimoe_api.DEFAULT_MODEL


def completion(**fields):
    return {
        "model": MODEL,
        "choices": [{"message": {"content": "It measures acceleration."}, "finish_reason": "stop"}],
        **fields,
    }


def test_direct_probe_checks_model_and_reports_separate_measurements():
    paths = []

    def handle(request):
        paths.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer sample-token"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["temperature"] == 0 and payload["max_tokens"] == 40
        assert payload["messages"][0]["role"] == "user"
        return httpx.Response(200, json=completion(usage={
            "prompt_tokens": 42, "completion_tokens": 5, "total_tokens": 47,
            "ttft_ms": 8, "decode_token_per_second": 32.5,
            "ignored": "sample-token",
        }))

    with httpx.Client(transport=httpx.MockTransport(handle),
                      headers={"Authorization": "Bearer sample-token"}) as client:
        result = mimoe_api.probe(client, BASE, MODEL)

    assert paths == [
        ("GET", "/mimik-ai/openai/v1/models"),
        ("POST", "/mimik-ai/openai/v1/chat/completions"),
    ]
    assert result["served_model"] == MODEL
    assert result["finish_reason"] == "stop"
    assert result["client_wall_ms"] >= 0
    assert result["mimoe_usage"] == {
        "prompt_tokens": 42, "completion_tokens": 5, "total_tokens": 47,
        "ttft_ms": 8, "decode_token_per_second": 32.5,
    }
    assert "sample-token" not in json.dumps(result)


@pytest.mark.parametrize("response", [
    completion(),
    completion(usage={"prompt_tokens": True, "total_tokens": -1, "ttft_ms": "unknown"}),
])
def test_probe_leaves_unavailable_usage_null(response):
    def handle(request):
        data = {"data": [{"id": MODEL}]} if request.method == "GET" else response
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert mimoe_api.probe(client, BASE, MODEL)["mimoe_usage"] is None


def test_probe_does_not_substitute_a_missing_model():
    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [{"id": "another-model"}]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="not listed"):
            mimoe_api.probe(client, BASE, MODEL)


def test_probe_detects_a_served_model_mismatch():
    def handle(request):
        data = {"data": [{"id": MODEL}]} if request.method == "GET" else completion(model="another-model")
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="served model"):
            mimoe_api.probe(client, BASE, MODEL)


@pytest.mark.parametrize("url", [
    "https://example.com/v1", "http://user@localhost:8083/v1", "http://@localhost:8083/v1",
])
def test_settings_require_a_loopback_url_without_credentials(monkeypatch, url):
    monkeypatch.setenv("MIMOE_BASE_URL", url)
    monkeypatch.setenv("MIMOE_API_KEY", "sample-token")
    with pytest.raises(ValueError, match="loopback"):
        mimoe_api.settings()


def test_main_does_not_echo_authentication_response(monkeypatch, capsys):
    monkeypatch.setattr(mimoe_api.sys, "argv", ["mimoe_api"])
    monkeypatch.setenv("MIMOE_BASE_URL", BASE)
    monkeypatch.setenv("MIMOE_API_KEY", "sample-token")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(403, text=request.headers["authorization"])
    ))

    def make_client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        client.headers.update(kwargs["headers"])
        return client

    monkeypatch.setattr(mimoe_api.httpx, "Client", make_client)
    assert mimoe_api.main() == 1
    output = capsys.readouterr()
    assert "rejected authentication" in output.err
    assert "sample-token" not in output.err + output.out


@pytest.mark.parametrize("usage", [None, {"total_tokens": 55, "ttft_ms": 97}])
def test_stream_emits_text_and_reports_optional_usage(usage):
    chunks = [
        {"model": MODEL, "choices": [{"delta": {"role": "assistant"}}]},
        {"model": MODEL, "choices": [{"delta": {"content": "mo"}}]},
        {"model": MODEL, "choices": [{"delta": {"content": "tion"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": usage},
    ]

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        assert json.loads(request.content)["stream"] is True
        events = ": keepalive\n\n" + "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"},
                              text=events + "data: [DONE]\n\n")

    emitted = []
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = mimoe_api.stream_probe(client, BASE, MODEL, emitted.append)

    assert emitted == ["mo", "tion"]
    assert result["served_model"] == MODEL
    assert result["finish_reason"] == "stop"
    assert result["mimoe_usage"] == usage
    assert 0 <= result["client_first_text_ms"] <= result["client_wall_ms"]


def test_stream_rejects_missing_completion_marker():
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        chunk = {"model": MODEL, "choices": [{"delta": {"content": "motion"}}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\n")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="before its completion marker"):
            mimoe_api.stream_probe(client, BASE, MODEL, lambda _: None)
