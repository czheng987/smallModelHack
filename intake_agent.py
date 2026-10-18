import json
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from phrase_check import PASS_RATE, arrange_phrases, propose_split, passes, verify_split
from resource_intake import get_resource_needs
from schedule_intake import get_schedule_needs


# Connect to Nemotron running on the Brev server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)
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


def _ask_for_phrases(user_text, correction=""):
    """Ask Nemotron for short schedule and resource phrases."""
    correction_line = f"\n{correction}\n" if correction else ""
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
                f"{correction_line}"
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
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except TypeError:
        response = client.chat.completions.create(
            model=INTAKE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0,
        )
    except Exception as error:
        if getattr(error, "status_code", None) not in (400, 422):
            raise
        response = client.chat.completions.create(
            model=INTAKE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0,
        )
    schedule, resource = parse_phrase_lists(message_text(response.choices[0].message))
    return arrange_phrases(schedule, resource)


def split_request(user_text):
    """Break the message into phrases and return them only at the pass rate."""
    correction = ""
    best_reward = 0.0
    for _attempt in range(2):
        schedule, resource = _ask_for_phrases(user_text, correction)
        reward = verify_split(user_text, schedule, resource)
        best_reward = max(best_reward, reward)
        if passes(reward):
            print(f"Phrase check passed at {reward:.0%}.")
            return schedule, resource
        correction = (
            "The last split was not accurate enough. Keep every activity, "
            "do not invent a task, and shorten each phrase."
        )

    schedule, resource = propose_split(user_text)
    reward = verify_split(user_text, schedule, resource)
    if passes(reward):
        print(f"Phrase check passed at {reward:.0%}.")
        return schedule, resource

    shown = max(best_reward, reward)
    print(
        f"Phrase check is {shown:.0%}, below {PASS_RATE:.0%}, "
        "so those phrases were not used."
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
