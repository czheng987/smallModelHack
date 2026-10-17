import re

from openai import OpenAI

from resource_intake import get_resource_needs
from schedule_intake import get_schedule_needs


# Connect to Nemotron running on the Brev server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)
INTAKE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

handoff = "none"
ROUTES = {"resource", "schedule", "both", "none"}


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


def parse_route(reply):
    """Pull resource, schedule, both, or none out of a normal model reply."""
    text = re.sub(
        r"<think>.*?</think>",
        " ",
        reply or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text.strip()]
    for line in reversed(lines):
        words = re.findall(r"\b(resource|schedule|both|none)\b", line.lower())
        if not words:
            continue
        if len(words) == 1:
            return words[0]
        labels = set(words)
        if "both" in labels or {"resource", "schedule"} <= labels:
            return "both"
        return words[-1]
    return "none"


def assign_request(user_text):
    """Ask Nemotron which intake the student's request belongs to."""
    messages = [
        {
            "role": "system",
            "content": (
                "You route a UNC student's request by its meaning. "
                "The student will usually not say the words resource or schedule. "
                "Reply with exactly one word: resource, schedule, both, or none.\n"
                "resource: campus help, counseling, someone to talk to, library databases, articles, or other campus services.\n"
                "schedule: classes, study time, exams, deadlines, workload, or planning the week.\n"
                "both: the student needs campus help and help planning their time.\n"
                "none: greetings, thanks, jokes, or anything that is not campus help or time planning.\n"
                "Examples:\n"
                "Student: I have a chem midterm Thursday and I don't know what to do tonight\n"
                "schedule\n"
                "Student: Where can I find psychology articles from my apartment?\n"
                "resource\n"
                "Student: I need sources for my essay and a plan to finish it before Friday\n"
                "both\n"
                "Student: hello\n"
                "none"
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
            max_tokens=64,
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
    reply = message_text(response.choices[0].message)
    route = parse_route(reply)
    if route not in ROUTES:
        return "none"
    return route


def main():
    global handoff
    while True:
        user_need = input("What do you need help with? (Type 'exit' to quit): ")

        if user_need.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break

        # Nemotron reads the request and assigns resource, schedule, both, or none.
        route = assign_request(user_need)
        print(f"Assigned: {route}")

        if route == "none":
            print("I can help with campus resources, your schedule, or both. Tell me a bit more about what you need.")
            continue

        if route in ("resource", "both"):
            print("Okay, let's help you with your resources!")
            handoff = "Resource"
            get_resource_needs()

        if route in ("schedule", "both"):
            print("Okay, let's help you with your schedule!")
            handoff = "Safety"
            get_schedule_needs()


if __name__ == "__main__":
    main()
