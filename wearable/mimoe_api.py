"""Probe the configured local mimOE endpoint without LangChain or wearable data."""
import argparse
import json
from math import isfinite
import os
import sys
from time import perf_counter
from urllib.parse import urlparse

import httpx

DEFAULT_MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
DEFAULT_URL = "http://127.0.0.1:8083/mimik-ai/openai/v1"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
PERFORMANCE_FIELDS = ("ttft_ms", "token_per_second", "decode_token_per_second")


def settings():
    base = os.getenv("MIMOE_BASE_URL", DEFAULT_URL).rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("MIMOE_BASE_URL must point to a loopback HTTP(S) endpoint.")
    if base.endswith("/chat/completions"):
        raise ValueError("Use the base URL ending in /v1, without /chat/completions.")
    token = os.getenv("MIMOE_API_KEY")
    if not token:
        raise ValueError("Set MIMOE_API_KEY to the token shown by Studio.")
    return base, os.getenv("MIMOE_MODEL", DEFAULT_MODEL), token


def check_model(client, base, model):
    listing = client.get(f"{base}/models")
    listing.raise_for_status()
    try:
        models = [entry["id"] for entry in listing.json()["data"]]
    except (ValueError, TypeError, KeyError):
        raise ValueError("mimOE did not return a valid model list.") from None
    if model not in models:
        raise ValueError("MIMOE_MODEL is not listed by the local endpoint. Check Models in Studio.")


def completion_payload(model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": (
            "Choose one label, heart or motion, for the request: How much did my wrist move? "
            "Reply with the label only."
        )}],
        "temperature": 0,
        "max_tokens": 40,
    }


def trace_router(client, base, model):
    direct_path = "/mimik-ai/openai/v1"
    if not base.endswith(direct_path) or "/" in model:
        raise ValueError("Trace requires the local inference base URL and a bare local model ID.")
    router = base.removesuffix(direct_path) + "/mimik-airouter/openai/v1"
    aliases = client.get(f"{router}/aliasModels")
    aliases.raise_for_status()
    try:
        names = [alias["name"] for alias in aliases.json()["data"]]
    except (ValueError, TypeError, KeyError):
        raise ValueError("Could not verify model aliases on the local router.") from None
    if model in names:
        raise ValueError("Trace requires a local model ID that is not shadowed by a router alias.")
    return router


def probe(client, base, model, *, trace=False):
    check_model(client, base, model)
    if trace:
        base = trace_router(client, base, model)
    headers = {"x-session-id": "wearable-api-probe"} if trace else None
    start = perf_counter()
    response = client.post(f"{base}/chat/completions", json=completion_payload(model), headers=headers)
    elapsed_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    if trace and response.headers.get("x-mimoe-node-id") != "local":
        raise ValueError("The router did not confirm local inference.")

    try:
        body = response.json()
        choice = body["choices"][0]
        answer = choice["message"]["content"]
    except (ValueError, TypeError, KeyError, IndexError):
        raise ValueError("mimOE did not return a valid Chat Completions response.") from None
    if body.get("model") != model:
        raise ValueError("The served model did not match MIMOE_MODEL.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("mimOE returned no text for the probe.")

    return {
        "served_model": body["model"],
        "finish_reason": choice.get("finish_reason"),
        "answer": answer,
        "client_wall_ms": round(elapsed_ms, 1),
        "mimoe_usage": usage_metrics(body.get("usage")) or None,
    }


def usage_metrics(usage):
    metrics = {}
    if isinstance(usage, dict):
        for key in TOKEN_FIELDS:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                metrics[key] = value
        for key in PERFORMANCE_FIELDS:
            value = usage.get(key)
            if (type(value) is int and value >= 0) or (type(value) is float and value >= 0 and isfinite(value)):
                metrics[key] = value
    return metrics


def sse_data(lines):
    data = []
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def stream_probe(client, base, model, emit):
    check_model(client, base, model)
    metrics = {}
    served_model = finish_reason = first_text_ms = None
    start = perf_counter()
    with client.stream("POST", f"{base}/chat/completions",
                       json={**completion_payload(model), "stream": True}) as response:
        response.raise_for_status()
        if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
            raise ValueError("mimOE did not return an event stream.")
        for event in sse_data(response.iter_lines()):
            if event == "[DONE]":
                break
            try:
                chunk = json.loads(event)
                reported_model = chunk.get("model")
                metrics.update(usage_metrics(chunk.get("usage")))
                choices = chunk.get("choices", [])
                choice = choices[0] if choices else {}
                delta = choice.get("delta", {}).get("content")
                finish_reason = choice.get("finish_reason") or finish_reason
            except (ValueError, AttributeError, TypeError, KeyError):
                raise ValueError("mimOE returned an invalid streaming event.") from None
            if reported_model is not None:
                if reported_model != model:
                    raise ValueError("The served model did not match MIMOE_MODEL.")
                served_model = reported_model
            if isinstance(delta, str) and delta:
                if served_model is None:
                    raise ValueError("mimOE did not identify the served model before sending text.")
                if first_text_ms is None:
                    first_text_ms = (perf_counter() - start) * 1000
                emit(delta)
        else:
            raise ValueError("The mimOE stream ended before its completion marker.")

    if first_text_ms is None:
        raise ValueError("mimOE returned no text for the probe.")
    return {
        "served_model": served_model,
        "finish_reason": finish_reason,
        "client_first_text_ms": round(first_text_ms, 1),
        "client_wall_ms": round((perf_counter() - start) * 1000, 1),
        "mimoe_usage": metrics or None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stream", action="store_true", help="Print text as mimOE sends it")
    mode.add_argument("--trace", action="store_true", help="Use the local Studio router for this probe")
    args = parser.parse_args()
    try:
        base, model, token = settings()
        with httpx.Client(headers={"Authorization": f"Bearer {token}"},
                          trust_env=False, follow_redirects=False, timeout=25) as client:
            if args.stream:
                print("Streamed reply:")
                try:
                    result = stream_probe(client, base, model, lambda text: print(text, end="", flush=True))
                finally:
                    print()
            else:
                result = probe(client, base, model, trace=args.trace)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in {401, 403}:
            print("mimOE rejected authentication. Check MIMOE_API_KEY in Studio.", file=sys.stderr)
        else:
            print(f"mimOE returned HTTP {status}. Check the local base URL and model.", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print("mimOE timed out. Check that the selected model is loaded.", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print("Cannot reach mimOE. Start the local runtime and verify MIMOE_BASE_URL.", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
