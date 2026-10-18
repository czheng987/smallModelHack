# smallModelHack

Intake, schedule, resource, and safety all call one local Nemotron model. The model is served by [vLLM](https://www.redhat.com/en/topics/ai/what-is-vllm), the open inference server Red Hat AI is built on. vLLM keeps the model in GPU memory and batches requests with PagedAttention, so several agents can call it at the same time. The agents use the OpenAI client against that server. They do not load the weights themselves.

## Run the model

On the GPU machine:

```bash
pip install openai vllm
python serve_vllm.py
```

That listens on `http://127.0.0.1:8000/v1` for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. Then, in another terminal:

```bash
python intake_agent.py
```

`VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_HOST`, and `VLLM_PORT` override the defaults. Set `VLLM_REASONING_PARSER` empty if this vLLM build does not include `nemotron_v3`.
