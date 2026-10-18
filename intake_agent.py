import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from phrase_check import PASS_RATE, SERVER_COUNT, arrange_phrases
from resources_servers.intake_phrases.app import score_votes
from resource_intake import get_resource_needs
from schedule_intake import format_runtime, get_schedule_needs
from vllm_model import (
    NEMOTRON_BASE_URL,
    open_client,
    planner_urls,
    reply_text,
    unreachable,
)


# A little variation so the servers can disagree when the phrasing is unstable.
PHRASE_TEMPERATURE = float(os.environ.get("PHRASE_TEMPERATURE", "0.2"))
INTAKE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
# vLLM returns short phrases with thinking off. The original server thinks
# through each vote on its own, which is the slower run we time for "before".
VLLM_PHRASE_TOKENS = 160
UNASSISTED_PHRASE_TOKENS = 400

handoff = "none"


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


def _phrase_messages(user_text):
    """Shared prompt. vLLM caches this prefix across the whole batch."""
    return [
        {
            "role": "system",
            "content": (
                "Break a UNC student's message into short phrases from their own words. "
                "Shorten each activity so a planner can read it. Keep every activity. "
                "Do not invent a task.\n"
                "Each schedule phrase must include the noun for what is being scheduled. "
                "Do not return a bare verb such as \"study on\".\n"
                "Each resource phrase must name what the student needs, such as counseling "
                "or someone to talk to. Do not return only \"help\" or \"talk\".\n"
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


def _vote_from_message(message):
    schedule, resource = parse_phrase_lists(reply_text(message))
    return arrange_phrases(schedule, resource)


def _create_phrase(model_client, user_text, max_tokens, thinking, samples=1):
    """One completion. samples > 1 asks vLLM to draw the whole vote in one call."""
    request = {
        "model": INTAKE_MODEL,
        "messages": _phrase_messages(user_text),
        "max_tokens": max_tokens,
        "temperature": PHRASE_TEMPERATURE,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
    }
    if samples > 1:
        request["n"] = samples
    try:
        return model_client.chat.completions.create(**request)
    except TypeError:
        request.pop("extra_body", None)
        return model_client.chat.completions.create(**request)
    except Exception as error:
        if unreachable(error):
            raise
        status = getattr(error, "status_code", None)
        if status not in (400, 422):
            raise
        if samples > 1:
            raise _BatchUnsupported(error)
        request.pop("extra_body", None)
        return model_client.chat.completions.create(**request)


class _BatchUnsupported(Exception):
    """The server accepted a single reply and rejected n=100."""


def _one_vote(model_client, user_text, max_tokens, thinking):
    response = _create_phrase(model_client, user_text, max_tokens, thinking, samples=1)
    return _vote_from_message(response.choices[0].message)


def votes_batched(model_client, user_text, count):
    """All votes in one vLLM call. Fan out only if this server rejects n."""
    try:
        response = _create_phrase(
            model_client, user_text, VLLM_PHRASE_TOKENS, thinking=False, samples=count
        )
    except _BatchUnsupported:
        return votes_parallel(model_client, user_text, count)
    choices = list(response.choices or [])
    if len(choices) < count:
        return votes_parallel(model_client, user_text, count)
    return [_vote_from_message(choice.message) for choice in choices[:count]]


def votes_parallel(model_client, user_text, count):
    """Short votes at the same time, used when one batched call is not available."""
    def _vote(_index):
        try:
            return _one_vote(model_client, user_text, VLLM_PHRASE_TOKENS, thinking=False)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(_vote, range(count)))


def votes_sequential(model_client, user_text, count):
    """One vote at a time on the original server, with thinking left on."""
    votes = []
    for _ in range(count):
        try:
            votes.append(
                _one_vote(model_client, user_text, UNASSISTED_PHRASE_TOKENS, thinking=True)
            )
        except Exception as error:
            votes.append(None)
            if unreachable(error):
                break
    votes.extend([None] * (count - len(votes)))
    return votes


def timed_vllm_votes(user_text):
    """Phrase on vLLM. A closed port is not included in the runtime."""
    for url in planner_urls():
        started = time.perf_counter()
        try:
            votes = votes_batched(open_client(url), user_text, SERVER_COUNT)
        except Exception as error:
            if not unreachable(error):
                raise
            continue
        return votes, time.perf_counter() - started, True
    return [None] * SERVER_COUNT, 0.0, False


def timed_unassisted(user_text):
    """Time the same votes one by one, without the vLLM batch."""
    started = time.perf_counter()
    try:
        votes_sequential(open_client(NEMOTRON_BASE_URL), user_text, SERVER_COUNT)
    except Exception:
        pass
    return time.perf_counter() - started


def print_phrase_runtimes(without_seconds, with_seconds, reached):
    print(f"Runtime without Red Hat vLLM: {format_runtime(without_seconds)}")
    if reached:
        print(f"Runtime with Red Hat vLLM: {format_runtime(with_seconds)}")
    else:
        print("Runtime with Red Hat vLLM: unavailable")


def split_request(user_text):
    """Phrase the message with vLLM and keep it only when the servers agree."""
    print(f"Checking this phrasing across {SERVER_COUNT} servers with Red Hat vLLM.")
    votes, with_seconds, reached = timed_vllm_votes(user_text)
    print(f"Timing the same {SERVER_COUNT} votes on the original server, one at a time.")
    without_seconds = timed_unassisted(user_text)
    ok, schedule, resource, rate = score_votes(user_text, votes)
    if ok:
        print(
            f"Phrase check passed at {rate:.0%} agreement "
            f"across {SERVER_COUNT} servers."
        )
    elif rate >= PASS_RATE:
        print(
            f"Phrase check is {rate:.0%} agreement across {SERVER_COUNT} servers, "
            "but the phrases were not accurate enough, so they were not used."
        )
        schedule, resource = [], []
    else:
        print(
            f"Phrase check is {rate:.0%} agreement across {SERVER_COUNT} servers, "
            f"below {PASS_RATE:.0%}, so those phrases were not used."
        )
        schedule, resource = [], []
    print_phrase_runtimes(without_seconds, with_seconds, reached)
    return schedule, resource


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
