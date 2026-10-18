"""Start vLLM's OpenAI-compatible server for the shared Nemotron model.

Red Hat AI Inference Server is built on this same vLLM project. The agents
call http://127.0.0.1:8000/v1 and do not load the model themselves.

The flags match the model's vLLM quick start:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
"""

import os
import shutil
import sys

from vllm_model import VLLM_MODEL

HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
PORT = os.environ.get("VLLM_PORT", "8000")


def serve_args():
    """Arguments for `vllm serve`. Host and port match the agents' client."""
    args = [
        "vllm",
        "serve",
        VLLM_MODEL,
        "--host",
        HOST,
        "--port",
        PORT,
        "--max-num-seqs",
        os.environ.get("VLLM_MAX_NUM_SEQS", "128"),
        "--enable-prefix-caching",
        "--async-scheduling",
        "--mamba-backend",
        "flashinfer",
        "--mamba-ssm-cache-dtype",
        "float16",
        "--enable-mamba-cache-stochastic-rounding",
        "--mamba-cache-philox-rounds",
        "5",
    ]
    parser = os.environ.get("VLLM_REASONING_PARSER", "nemotron_v3")
    if parser.strip():
        args.extend(["--reasoning-parser", parser.strip()])
    return args


def main():
    if shutil.which("vllm") is None:
        sys.exit(
            "vLLM is not installed. On the GPU machine run: pip install vllm\n"
            "Then start this server before intake_agent.py."
        )
    os.execvp("vllm", serve_args())


if __name__ == "__main__":
    main()
