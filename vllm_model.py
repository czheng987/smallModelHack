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


def open_client():
    """OpenAI client aimed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


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
