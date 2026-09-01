"""Shared low-level Ollama HTTP call, used by both Stage 2's turn analysis
(starter/stage2.py) and RANKLLM's final re-rank (starter/rankllm.py) -- the only two
LLM call sites in the pipeline. No dependency on any other starter module."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 45


def _ollama_chat(messages: list[dict], schema: dict) -> tuple[dict | None, dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None, {"prompt_tokens": 0, "completion_tokens": 0}
    usage = {
        "prompt_tokens": int(body.get("prompt_eval_count") or 0),
        "completion_tokens": int(body.get("eval_count") or 0),
    }
    content = body.get("message", {}).get("content")
    if not content:
        return None, usage
    try:
        return json.loads(content), usage
    except json.JSONDecodeError:
        return None, usage
