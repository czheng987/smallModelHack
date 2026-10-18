# smallModelHack

Intake, resource, and safety call the original Nemotron server at `http://127.0.0.1:8000/v1`.

Scheduling is the call that needs several steps: the clock, deadlines, one-time events such as shopping, and work that has to be split across days. That call goes through [vLLM](https://www.redhat.com/en/topics/ai/what-is-vllm), the open inference server Red Hat AI is built on. vLLM keeps the model on the GPU so the schedule planner can think through the plan. The other agents do not use it.

## Run scheduling with vLLM

On the GPU machine, with the original Nemotron server already on port 8000:

```bash
pip install openai vllm
python serve_vllm.py
```

vLLM listens on `http://127.0.0.1:8001/v1`. Then run `python intake_agent.py`. Only `schedule_intake.py` calls port 8001, and it asks the model to think before it returns the plan.

`VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_HOST`, and `VLLM_PORT` override the defaults. Set `VLLM_REASONING_PARSER` empty if this vLLM build does not include `nemotron_v3`.
