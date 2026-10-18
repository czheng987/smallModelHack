import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from phrase_check import PASS_RATE, SERVER_COUNT, accepted_split, arrange_phrases
from resource_intake import get_resource_needs
from schedule_intake import get_schedule_needs
from vllm_model import NEMOTRON_BASE_URL, open_client


# Connect to Nemotron running on the Brev server. A closed port is not retried.
client = open_client(NEMOTRON_BASE_URL)
# A little variation so the servers can disagree when the phrasing is unstable.
PHRASE_TEMPERATURE = float(os.environ.get("PHRASE_TEMPERATURE", "0.2"))
INTAKE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

handoff = "none"


def message_text(message):
    """Read the reply, including a reasoning field when the server splits it out."""
    chunks = []
    content = message.content or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content:
        chunks.append(content)
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value)
    return "\n".join(chunks)


def phrase_list(value):
    """Keep only non-empty phrase strings."""
    if not isinstance(value, list):
        return []
    phrases = []
    for item in value:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.split())
        if phrase:
            phrases.append(phrase)
    return phrases


def parse_phrase_lists(reply):
    """Read schedule[] and resource[] from Nemotron's JSON reply."""
    text = re.sub(
        r"<think>.*?</think>",
        " ",
        reply or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return [], []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    return phrase_list(data.get("schedule")), phrase_list(data.get("resource"))


def assign_type(schedule, resource):
    """Name the route from whichever phrase arrays contain something."""
    if schedule and resource:
        return "both"
    if schedule:
        return "schedule"
    if resource:
        return "resource"
    return "none"


def _ask_for_phrases(user_text):
    """Ask one server for short schedule and resource phrases."""
    messages = [
        {
            "role": "system",
            "content": (
                "Break a UNC student's message into short phrases from their own words. "
                "Shorten each activity so a planner can read it. Keep every activity. "
                "Do not invent a task.\n"
                "Return JSON only, with two arrays:\n"
                '{"schedule": ["phrase"], "resource": ["phrase"]}\n'
                "schedule stores phrases about a scheduling problem: classes, exams, deadlines, "
                "study time, workload, or planning the week.\n"
                "resource stores phrases about needing mental health support: wanting someone to "
                "talk to, counseling, campus mental health resources, or feeling overwhelmed and "
                "asking for help. Do not diagnose. Do not name a disorder.\n"
                "Put each phrase in only one array. Leave an array empty when the message has "
                "nothing for that side.\n"
                "Examples:\n"
                "Student: I have three papers due Friday and I need someone to talk to\n"
                '{"schedule": ["three papers due Friday"], "resource": ["someone to talk to"]}\n'
                "Student: Where can I find counseling on campus?\n"
                '{"schedule": [], "resource": ["counseling on campus"]}\n'
                "Student: Can you help me plan study time around my labs?\n"
                '{"schedule": ["plan study time around my labs"], "resource": []}\n'
                "Student: thanks\n"
                '{"schedule": [], "resource": []}'
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]
    try:
        response = client.chat.completions.create(
            model=INTAKE_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=PHRASE_TEMPERATURE,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except TypeError:
        response = client.chat.completions.create(
            model=INTAKE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=PHRASE_TEMPERATURE,
        )
    except Exception as error:
        if getattr(error, "status_code", None) not in (400, 422):
            raise
        response = client.chat.completions.create(
            model=INTAKE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=PHRASE_TEMPERATURE,
        )
    schedule, resource = parse_phrase_lists(message_text(response.choices[0].message))
    return arrange_phrases(schedule, resource)


def _server_vote(user_text):
    """One server's phrases. A server that does not answer casts no vote."""
    try:
        return _ask_for_phrases(user_text)
    except Exception:
        return None


def split_request(user_text):
    """Phrase the message on many servers and keep it only when they agree."""
    print(f"Checking this phrasing across {SERVER_COUNT} servers.")
    with ThreadPoolExecutor(max_workers=SERVER_COUNT) as pool:
        votes = list(pool.map(_server_vote, [user_text] * SERVER_COUNT))
    ok, schedule, resource, rate = accepted_split(user_text, votes)
    if ok:
        print(
            f"Phrase check passed at {rate:.0%} agreement "
            f"across {SERVER_COUNT} servers."
        )
        return schedule, resource
    if rate >= PASS_RATE:
        print(
            f"Phrase check is {rate:.0%} agreement across {SERVER_COUNT} servers, "
            "but the phrases were not accurate enough, so they were not used."
        )
        return [], []
    print(
        f"Phrase check is {rate:.0%} agreement across {SERVER_COUNT} servers, "
        f"below {PASS_RATE:.0%}, so those phrases were not used."
    )
    return [], []


def _run_intake(intake, phrases):
    intake(list(phrases))


def run_intakes(schedule, resource):
    """Start the intakes whose phrase arrays have something. Both run together when both do."""
    jobs = []
    if resource:
        jobs.append((get_resource_needs, resource))
    if schedule:
        jobs.append((get_schedule_needs, schedule))
    if not jobs:
        return

    if len(jobs) == 1:
        _run_intake(jobs[0][0], jobs[0][1])
        return

    print("Starting resource intake and schedule intake at the same time.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_intake, intake, phrases) for intake, phrases in jobs]
        for future in futures:
            future.result()


def show_phrases(schedule, resource):
    print("schedule[]:")
    if schedule:
        for phrase in schedule:
            print(f"- {phrase}")
    else:
        print("- (empty)")
    print("resource[]:")
    if resource:
        for phrase in resource:
            print(f"- {phrase}")
    else:
        print("- (empty)")


def main():
    global handoff
    while True:
        user_need = input("What do you need help with? (Type 'exit' to quit): ")

        if user_need.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        schedule, resource = split_request(user_need)
        route = assign_type(schedule, resource)
        show_phrases(schedule, resource)
        print(f"Assigned: {route}")

        if route == "none":
            print("I can help with a scheduling problem, mental health resources, or both. Tell me a bit more about what you need.")
            continue

        if route == "resource":
            handoff = "Resource"
        elif route == "schedule":
            handoff = "Safety"
        else:
            handoff = "both"

        run_intakes(schedule, resource)


if __name__ == "__main__":
    main()
