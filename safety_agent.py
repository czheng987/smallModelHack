import json
import os
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------
# PART 1: Connect to the shared Nemotron server
# ---------------------------------------------------------------
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)
SAFETY_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

CATEGORIES = ["mental_health", "food", "housing", "lockout"]
SEVERITIES = ["none", "moderate", "severe", "crisis"]  # lowest -> highest

# ---------------------------------------------------------------
# PART 2: Load the school's verified resources from a JSON file.
# The model NEVER writes phone numbers. They only come from this file.
# Another school can swap in its own file, e.g.:
#   RESOURCE_FILE=duke_resources.json python safety_agent.py
# ---------------------------------------------------------------
RESOURCE_FILE = Path(__file__).parent / os.environ.get("RESOURCE_FILE", "unc_resources.json")

with open(RESOURCE_FILE) as f:
    RESOURCES = json.load(f)

SCHOOL = RESOURCES["school"]
SERVICE_DESKS = RESOURCES["service_desks"]

# ---------------------------------------------------------------
# PART 3: Backup check. If the student uses any of these phrases,
# it's a crisis even if the model misses it.
# ---------------------------------------------------------------
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "want to die", "suicide", "suicidal",
    "hurt myself", "self harm", "self-harm",
    "following me", "being followed", "not safe", "unsafe", "in danger", "unalive", "stab myself"
]

# If the model's answer can't be read, assume the student needs support.
SAFE_DEFAULT = [{
    "category": "mental_health",
    "severity": "severe",
    "lockout_type": None,
    "reason": "Safety check could not be completed, so support resources are shown.",
}]

# ---------------------------------------------------------------
# PART 4: Instructions for Nemotron (the "job description")
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are the safety reviewer for a student support assistant.
Read the conversation and identify urgent student needs. Do NOT diagnose any condition.
Do NOT write phone numbers, websites, or resource names.

Categories: mental_health, food, housing, lockout
Severity:
- none: no real need
- moderate: a concern, but not urgent (stressed, worried about money, locked out in daytime)
- severe: an urgent basic need (hasn't eaten in days, losing housing soon, locked out at night with nowhere to go)
- crisis: any sign of self-harm, suicide, or immediate danger to the student
For lockout, set lockout_type to "room" or "car". For other categories, set it to null.

Return ONLY JSON, no other text, in exactly this format:
{"needs": [{"category": "food", "severity": "severe", "lockout_type": null, "reason": "one short sentence"}]}
If there are no needs, return {"needs": []}"""


def ask_model(transcript):
    """Send the conversation to Nemotron. Return a list of needs, or None if it fails."""
    try:
        response = client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=1500,
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        # Keep only the part between the first { and the last }
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        return clean_needs(data.get("needs", []))
    except Exception:
        return None


def clean_needs(raw_needs):
    """Throw away anything that isn't a valid category/severity."""
    needs = []
    for need in raw_needs:
        if not isinstance(need, dict):
            continue
        category = need.get("category")
        severity = need.get("severity")
        if category not in CATEGORIES or severity not in SEVERITIES:
            continue
        lockout_type = need.get("lockout_type")
        if category == "lockout" and lockout_type not in ("room", "car"):
            lockout_type = "room"
        needs.append({
            "category": category,
            "severity": severity,
            "lockout_type": lockout_type if category == "lockout" else None,
            "reason": str(need.get("reason", "")),
        })
    return needs


def has_crisis_keyword(text):
    text = text.lower()
    return any(phrase in text for phrase in CRISIS_KEYWORDS)


def find_service_desk(text):
    """If the student names their dorm, return that Service Desk."""
    text = text.lower()
    for dorm, info in SERVICE_DESKS.items():
        if dorm in text:
            return {"name": f"{info['desk']} Service Desk", "phone": info["phone"],
                    "details": "Issues temporary replacement keys"}
    return None


def add(resources, items):
    """Add resources without listing the same one twice."""
    names = [r["name"] for r in resources]
    for item in items:
        if item["name"] not in names:
            resources.append(item)
            names.append(item["name"])


# ---------------------------------------------------------------
# PART 5: The main function. Same name and input as before, so
# my teammate's files keep working.
# ---------------------------------------------------------------
def review_conversation(conversation):
    student_text = " ".join(turn.get("user", "") for turn in conversation)
    transcript = "\n".join(
        f"Student: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
        for turn in conversation
    ) or "No conversation to review."

    # Step 1: Nemotron judges the needs (or we fall back to the safe default)
    needs = ask_model(transcript)
    if needs is None:
        needs = SAFE_DEFAULT

    # Step 2: Is this a crisis? (model OR keyword backup)
    crisis = any(n["severity"] == "crisis" for n in needs) or has_crisis_keyword(student_text)

    # Step 3: Attach verified resources
    resources = []
    if crisis:
        add(resources, RESOURCES["crisis"])  # crisis help always comes first

    for need in needs:
        level = SEVERITIES.index(need["severity"])
        if need["category"] == "lockout":
            if level >= SEVERITIES.index("moderate"):
                if need["lockout_type"] == "room":
                    desk = find_service_desk(student_text)
                    if desk:
                        add(resources, [desk])
                add(resources, RESOURCES["lockout_" + need["lockout_type"]])
        elif level >= SEVERITIES.index("severe"):
            add(resources, RESOURCES[need["category"]])

    # Step 4: Return a labeled "form" (dictionary) the rest of the app can read
    return {
        "school": SCHOOL,
        "crisis": crisis,
        "allow": not crisis,  # False = stop the normal plan, show help first
        "needs": needs,
        "resources": resources,
    }


# ---------------------------------------------------------------
# PART 6: Tests. Runs ONLY when you type: python safety_agent.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    test_messages = [
        "Can you help me plan my study week?",
        "I'm kind of stressed about my midterms.",
        "I haven't eaten since Tuesday and I have no money for food.",
        "I'm getting evicted this Friday and I have nowhere to stay.",
        "I locked myself out of my room in Hinton James. It's 2pm.",
        "I locked my keys in my car in a campus lot. It's midnight and someone is following me.",
        "I don't want to be here anymore. I want to end my life.",
        "What's the best pizza place on Franklin Street?",
    ]
    for message in test_messages:
        result = review_conversation([{"user": message, "assistant": ""}])
        print("STUDENT:", message)
        print(json.dumps(result, indent=2))
        print("-" * 60)
