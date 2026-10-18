# smallModelHack

A UNC Chapel Hill student support tracker. The student describes what they need in the React web development site. Safety, intake, the schedule builder, and the resource matcher run on a local Nemotron model and return a plan the site can show as Today, Week, and Month.

## Quick start

On the GPU machine, with the original Nemotron server already on port 8000.

```bash
git submodule update --init
pip install -r requirements.txt
pip install openai vllm fastapi uvicorn pydantic
python serve_vllm.py
```

In a second terminal:

```bash
source ~/vllm-env/bin/activate
uvicorn backend:app --port 8002 --reload
```

In a third terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the site at [http://localhost:5173](http://localhost:5173). On a remote machine, forward ports **5173** (the website) and **8002** (the API). Check the API with:

```bash
curl -s localhost:8002/api/health
```

The terminal chat is still available with `python intake_agent.py`. Type `exit` to leave it.

vLLM listens on `http://127.0.0.1:8001/v1`. Intake and the schedule planner call that server. Resource matching and safety stay on the original Nemotron server at `http://127.0.0.1:8000/v1`.

## Tech stack and architecture

| Piece | What it is |
| --- | --- |
| Website | React 19 and Vite. Today, Week, and Month views. |
| API | FastAPI (`backend.py`) on port 8002. Vite proxies `/api` to it. |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, OpenAI-compatible. |
| Fast inference | [vLLM](https://www.redhat.com/en/topics/ai/what-is-vllm) on port 8001. Red Hat AI inference is built on this project. |
| Phrase check | [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) scores the intake split on 100 in-process servers. |
| Campus list | `unc_resources.json`. The model does not invent phone numbers. |

```text
Browser (React, port 5173)
        |
        |  POST /api/plan
        v
backend.py (port 8002)
        |
        +-- Safety agent ---------> Nemotron :8000 ---- unc_resources.json
        +-- Intake agent ---------> vLLM :8001, then NeMo Gym phrase check
        +-- Schedule builder -----> vLLM :8001
        +-- Resource matcher -----> Nemotron :8000 ---- unc_resources.json
```

Safety runs first. In a life-safety case the API returns urgent help and skips the plan. Otherwise intake splits the message into `schedule[]` and `resource[]`, the schedule builder lays out timed blocks, and the resource matcher returns catalog entries.

## Reproduce the demo

No paid API key is required. Both local servers accept the placeholder key `not-needed`.

Copy this into a `.env` in the project folder if you want to override the defaults. The programs also read these variables from the shell.

```bash
# Original Nemotron server (resource matcher and safety)
NEMOTRON_BASE_URL=http://127.0.0.1:8000/v1

# vLLM server (intake phrasing and the schedule planner)
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_API_KEY=not-needed
VLLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
VLLM_HOST=127.0.0.1
VLLM_PORT=8001
VLLM_MAX_NUM_SEQS=128
# Set this empty if this vLLM build does not include nemotron_v3
VLLM_REASONING_PARSER=nemotron_v3

# Intake phrase check. 0.95 raises the bar to 95%.
PHRASE_PASS_RATE=0.90
PHRASE_SERVERS=100

# Campus catalog. Another school can point this at its own file.
RESOURCE_FILE=unc_resources.json
```

Start Nemotron on port 8000, then `python serve_vllm.py`, then `uvicorn backend:app --port 8002 --reload`, then `npm run dev` inside `frontend`. Submit a check-in on the site. A sample message:

> I have three papers due Friday and I need someone to talk to

The page should show study blocks before Friday and a campus resource for talking with someone.

`VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_HOST`, and `VLLM_PORT` override the defaults used by `serve_vllm.py` and the planner client.

## Datasets and synthetic data

This demo does not train on a student dataset, and it does not ship a synthetic student corpus.

| Data | Provenance |
| --- | --- |
| `unc_resources.json` | Hand-compiled catalog for UNC Chapel Hill, last checked 2026-10-17. Entries are public campus and community services: 988, UNC CAPS, UNC Police, Alert Carolina, Carolina Cupboard, the Inter-Faith Council food pantry, Dean of Students basic needs, Carolina Housing service desks, and University Career Services. Names, phones, and links are copied from those public pages. |
| `resources_servers/intake_phrases/data/example.jsonl` | Four hand-written intake examples used by the NeMo Gym phrase server (appointment and midterm, papers plus someone to talk to, counseling, and a thank-you). |
| `phrase_gym.py` | Hand-written checks that the phrase split keeps a noun on schedule items and puts resource needs in `resource[]`. |
| Model weights | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) on Hugging Face. |
| NeMo Gym | Git submodule of [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym). |

## Known limitations and next steps

The schedule builder makes a schedule pretty well. It tries its best to place work so everything is finished before the deadline. It has a hard time understanding that an event has to happen on a specific day. It mostly lists the deadline, even when the item is an event that should occur on that day.

Next steps:

- Pull from more resource databases so the agents can match what a student needs more closely.
- Implement this app for the RTP area so it can help students there.
