# smallModelHack

Estelle Supporter is a UNC Chapel Hill student support tracker. A student describes what they need on the website. One self-hosted Nemotron model runs four agents (Safety, Intake, Schedule, and Resources) and returns a plan the site can show as Today, Week, and Month.

## Quick start

These commands work on any GPU machine that can hold the model, including a personal machine or an NVIDIA Brev instance such as an H100. They are not tied to one cloud box. Each server listens on that machine’s own `localhost`.

You need:

- A GPU with enough memory for a 30B model. This demo uses one NVIDIA H100 80GB.
- The Nemotron server already running at `http://127.0.0.1:8000/v1`.
- Python, and Node.js for the website.
- This repo, with the NeMo Gym submodule checked out.

### 1. One-time setup

From the project folder:

```bash
git submodule update --init
pip install -r requirements.txt
pip install openai vllm fastapi uvicorn pydantic
cd frontend
npm install
cd ..
```

`pip install -r requirements.txt` installs NeMo Gym, which scores the intake phrase split. The second `pip install` is the model server, the OpenAI client, and the website API. `npm install` is only for the React site.

If vLLM lives in a virtualenv, activate it before the `pip` and `python` commands. On the demo machine that env is `~/vllm-env`. On your machine, use the path where you installed vLLM.

### 2. Start the three processes

Use a separate terminal for each one. Leave each process running.

**Terminal 1, the fast model server.** Intake and the schedule builder call this. It is Red Hat vLLM serving the same Nemotron weights.

```bash
python serve_vllm.py
```

Wait until it is listening on `http://127.0.0.1:8001/v1`. Safety and the resource matcher keep using the Nemotron server you already started on port 8000.

**Terminal 2, the website API.** This is the door between the page and the four agents. It runs Safety, then Intake, then Schedule, then Resources, and sends back tasks, deadlines, and campus resources.

```bash
uvicorn backend:app --port 8002 --reload
```

**Terminal 3, the website.**

```bash
cd frontend
npm run dev
```

Vite serves the page on port 5173 and forwards every `/api` request to port 8002, so the browser never talks to the model directly.

### 3. Open the site

Go to [http://localhost:5173](http://localhost:5173).

If the GPU machine is remote, forward port **5173** (the page) and port **8002** (the API), then open the forwarded 5173 address.

Confirm the API before you submit a check-in:

```bash
curl -s localhost:8002/api/health
```

A healthy API answers `{"ok": true}`. Then try a check-in such as “I have three papers due Friday and I need someone to talk to.” The plan should show study time before Friday and a campus resource for talking with someone.

The older terminal chat is still there. From the project folder, run `python intake_agent.py` and type `exit` to leave it. That chat prints its own timing lines. The website skips those extra timing calls.

## AI model

NVIDIA Nemotron 3.5 Lightning 30B-A3B (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) is the only model in Estelle Supporter. It powers all four agents: Safety, Intake, Schedule, and Resources.

We self-host it with Red Hat vLLM on a single NVIDIA H100 and call it through vLLM’s OpenAI-compatible API with the OpenAI Python SDK. No OpenAI-hosted model is used.

## Tools

- **Red Hat vLLM**, serving Nemotron with an OpenAI-compatible API. The intake check is a 100-vote agreement; vLLM serves the phrasing that those votes score.
- **NVIDIA Nemotron 3.5 Lightning**, the model above.
- **NVIDIA H100 via Brev**, the GPU this demo runs on. Any machine with enough GPU memory can run the same commands.
- **Docker.**
- **FastAPI and Uvicorn**, the Python API in `backend.py` (port 8002).
- **React and Vite**, the website in `frontend` (port 5173).
- **NeMo Gym**, the in-process servers that score the intake split. It is the `Gym` submodule from [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym).

## How we use Nemotron

**What it does.** Nemotron 3.5 Lightning (30B-A3B) turns a stressed UNC student’s check-in into a plan:

- Safety reads the student’s own words and returns structured JSON needs (category and severity) for crises, food, housing, loneliness, lockouts, and active threats.
- Intake splits the message into schedule tasks and resource requests.
- Schedule builds a day-by-day study plan up to each deadline.
- Resources matches verified UNC resources.

**Why this model.** It is a small language model with about 3B active parameters, so one model serves all four agents on a single H100 and answers quickly. A full check-in takes about 6–25 seconds. Student support data is sensitive, so a small self-hosted model keeps the check-in on our own GPU.

**How we keep it honest and fast.**

- The model classifies, and our code supplies the facts. Nemotron returns JSON needs only. Phone numbers and links come from the verified `unc_resources.json`, so the model cannot invent a hotline.
- A hallucination check compares any phone number or website in an answer with that verified list, and it blocks diagnosis language.
- Keyword backups for self-harm, danger, and an active threat still run when the model’s output cannot be parsed.
- Intake keeps a phrase split when 100 NeMo Gym servers agree. vLLM produces the phrasing those servers score, which is what keeps that check fast.
- The website skips a second, timing-only pass through the model. That pass only existed to print a before-and-after runtime. Removing it from the app path took a check-in from over 2 minutes to under 25 seconds.

## Tech stack and architecture

```text
Browser (React + Vite, port 5173)
        |
        |  POST /api/plan
        v
backend.py (FastAPI + Uvicorn, port 8002)
        |
        +-- Safety ---------> Nemotron :8000 ---- unc_resources.json
        +-- Intake ---------> vLLM :8001, then 100 NeMo Gym servers
        +-- Schedule -------> vLLM :8001
        +-- Resources ------> Nemotron :8000 ---- unc_resources.json
```

Safety runs first. In a life-safety case the API returns urgent help and skips the plan. Otherwise intake splits the message into `schedule[]` and `resource[]`, the schedule builder lays out timed blocks, and the resource matcher returns catalog entries. Same model weights on both ports: port 8000 is the original Nemotron server, and port 8001 is vLLM.

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
