"""Client for the vLLM server used only by the schedule planner.

Scheduling takes several steps: clock, deadlines, one-time events, and
work that has to be split. vLLM is the open inference server Red Hat AI
is built on. It keeps Nemotron on the GPU and lets that one hard call
think. Intake, resource, and safety stay on the original server.
"""

import os

from openai import OpenAI

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "not-needed")
VLLM_MODEL = os.environ.get(
    "VLLM_MODEL",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
)


NEMOTRON_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "http://127.0.0.1:8000/v1")


def open_client(base_url=None):
    """OpenAI client aimed at one local server. Do not retry a closed port."""
    settings = {
        "base_url": base_url or VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "max_retries": 0,
    }
    try:
        import httpx
        settings["timeout"] = httpx.Timeout(120.0, connect=1.0)
    except Exception:
        settings["timeout"] = 30
    return OpenAI(**settings)


def planner_urls():
    """vLLM first. The original Nemotron server is the backup when that port is down."""
    urls = []
    for url in (VLLM_BASE_URL, NEMOTRON_BASE_URL):
        cleaned = str(url).rstrip("/")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def unreachable(error):
    """True when the server never answered, not when it rejected the prompt."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    name = type(error).__name__
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout"}:
        return True
    text = str(error).lower()
    return "connection error" in text or "connection refused" in text or "timed out" in text


def reply_text(message):
    """Read a vLLM chat reply, including a separate reasoning field."""
    chunks = []
    content = message.content or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content:
        chunks.append(content)
    for key in ("reasoning_content", "reasoning"):
        value = getattr(message, key, None)
        if isinstance(value, str) and value.strip():
            chunks.append(value)
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip() and value not in chunks:
                chunks.append(value)
    return "\n".join(chunks)
