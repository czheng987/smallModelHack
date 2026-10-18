import json
import os
import re
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

CATEGORIES = ["mental_health", "community", "safety", "active_threat", "food", "housing", "lockout"]
LIFE_SAFETY = ("active_threat", "self_harm", "danger")  # these always come first and pause other agents
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
# it's a crisis even if the model misses it. Two lists, because
# self-harm and physical danger need different messages.
# ---------------------------------------------------------------
SELF_HARM_KEYWORDS = [
    "kill myself", "end my life", "want to die", "suicide", "suicidal",
    "hurt myself", "self harm", "self-harm", "unalive", "stab myself",
]
DANGER_KEYWORDS = [
    "following me", "being followed", "not safe", "unsafe", "in danger",
]
ACTIVE_THREAT_KEYWORDS = [
    "shooter", "shooting", "shots fired", "gunshots", "gun shots", "gunman",
    "has a gun", "with a gun", "armed person", "someone with a weapon",
]

# Danger and active-threat resources come from the JSON file.
DANGER_RESOURCES = RESOURCES.get("danger", [
    {"name": "Emergency services", "phone": "911", "details": "Call now if you are in danger"},
    {"name": "UNC Police", "phone": "919-962-8100", "details": "On campus, 24/7"},
])
ACTIVE_THREAT_RESOURCES = RESOURCES.get("active_threat", DANGER_RESOURCES)

# ---------------------------------------------------------------
# PART 3b: Checking OTHER agents' answers before the student sees them
#   1. Hallucination: any phone number or website that is not in our
#      verified JSON file gets flagged and removed.
#   2. Diagnosis: the app must never tell a student they have a
#      mental illness or tell them to take medication.
# ---------------------------------------------------------------
def _host(url):
    url = re.sub(r"^https?://", "", url.strip().lower())
    url = re.sub(r"^www\.", "", url)
    return url.split("/")[0].rstrip(".,;:)")


def _collect_verified(data, phones, hosts):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "phone" and isinstance(value, str):
                phones.add(re.sub(r"\D", "", value))
            elif key == "link" and isinstance(value, str):
                hosts.add(_host(value))
            else:
                _collect_verified(value, phones, hosts)
    elif isinstance(data, list):
        for item in data:
            _collect_verified(item, phones, hosts)


VERIFIED_PHONES = {"911", "988", "741741"}
VERIFIED_HOSTS = set()
_collect_verified(RESOURCES, VERIFIED_PHONES, VERIFIED_HOSTS)

PHONE_PATTERN = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
LINK_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:edu|org|com|gov|net)(?:/[^\s)]*)?", re.I)

CONDITIONS = (r"(?:clinical |major |severe |an? )?(?:depression|anxiety disorder|anxiety|bipolar(?: disorder)?|adhd|ptsd|ocd|"
              r"eating disorder|anorexia|bulimia|schizophrenia|borderline(?: personality disorder)?|panic disorder|"
              r"insomnia|mental illness|personality disorder|disorder)")
DIAGNOSIS_PATTERNS = [
    re.compile(r"\byou (?:might |may |probably |likely |clearly |definitely |could |seem to |appear to )?"
               r"(?:have|suffer from|are suffering from|are showing signs of|show signs of|are experiencing) " + CONDITIONS, re.I),
    re.compile(r"\b(?:sounds|seems|looks) like (?:you have|you're dealing with|you are dealing with|you suffer from) " + CONDITIONS, re.I),
    re.compile(r"\byou(?:'re| are) (?:clinically depressed|bipolar|schizophrenic|anorexic|bulimic)\b", re.I),
    re.compile(r"\bdiagnose you\b|\byour diagnosis\b", re.I),
    re.compile(r"\b(?:take|taking) (?:some |your )?(?:medication|meds|antidepressants?|"
               r"sertraline|zoloft|prozac|fluoxetine|lexapro|xanax|adderall|melatonin)\b", re.I),
]


def check_response(text):
    """Look for unverified contact info or diagnosis in another agent's answer.
    Returns (problems, safe_text) where safe_text has the bad lines removed."""
    problems = []
    safe_lines = []
    for line in (text or "").splitlines():
        line_problems = []
        for phone in PHONE_PATTERN.findall(line):
            if re.sub(r"\D", "", phone) not in VERIFIED_PHONES:
                line_problems.append(f"Unverified phone number: {phone.strip()}")
        for link in LINK_PATTERN.findall(line):
            if _host(link) not in VERIFIED_HOSTS:
                line_problems.append(f"Unverified website: {link}")
        for pattern in DIAGNOSIS_PATTERNS:
            match = pattern.search(line)
            if match:
                line_problems.append(f'Diagnosis or medication advice: "{match.group(0)}"')
        if line_problems:
            problems.extend(p for p in line_problems if p not in problems)
        else:
            safe_lines.append(line)
    return problems, "\n".join(safe_lines).strip()


# If the model's answer can't be read, guess from the student's own words.
# (The self-harm / danger / active-threat keyword checks still run either way.)
FALLBACK_WORDS = {
    "housing": ["evict", "evicted", "kicked out of my apartment", "nowhere to stay", "nowhere to sleep", "homeless"],
    "food": ["haven't eaten", "havent eaten", "no food", "can't afford food", "cant afford food", "starving"],
    "lockout": ["locked out", "locked my keys", "lock myself out", "locked myself out"],
}


def guess_needs(student_text):
    text = student_text.lower()
    needs = []
    for category, words in FALLBACK_WORDS.items():
        if any(word in text for word in words):
            needs.append({
                "category": category,
                "severity": "moderate" if category == "lockout" else "severe",
                "lockout_type": ("car" if " car" in text else "room") if category == "lockout" else None,
                "reason": "Found in the student's own words.",
            })
    return needs

# ---------------------------------------------------------------
# PART 4: Instructions for Nemotron (the "job description")
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are the safety reviewer for a student support assistant.
Read the STUDENT's messages and identify urgent needs in the student's own situation.
Do NOT diagnose any condition. Do NOT write phone numbers, websites, or resource names.
If the student is only asking for information or where to find a resource, and does not
describe their own urgent situation, return no needs.

Categories:
- mental_health: emotional distress, feeling hopeless or overwhelmed, self-harm, suicide
- community: feeling lonely or isolated, or struggling to make friends or find community
- safety: physical danger to the student (being followed, threatened, feeling unsafe where they are)
- active_threat: an active shooter, gunshots, or an armed person on or near campus
- food: ONLY when the student does not have enough to eat
- housing: ONLY when the student is being evicted, losing their housing, or has nowhere to stay.
  Roommate problems, housing applications, or questions about housing options are NOT housing needs.
- lockout: locked out of their room or car
Severity:
- none: no real need
- moderate: a concern, but not urgent
- severe: an urgent need that should be handled today
- crisis: self-harm, suicide, or immediate danger
Examples:
- "I'm stressed about my midterms" -> mental_health, moderate
- "I feel overwhelmed and can't sleep" -> mental_health, moderate
- "I haven't eaten in two days" -> food, severe
- "I'm being evicted Friday and have nowhere to go" -> housing, severe
- "Locked out of my dorm, it's 2pm" -> lockout (room), moderate
- "Locked out of my car at midnight and someone is following me" -> lockout (car), severe AND safety, crisis
- "I want to end my life" -> mental_health, crisis
- "I heard gunshots near the library" -> active_threat, crisis
- "I don't have any friends here and I feel lonely" -> community, moderate
- "Where is the food pantry on campus?" -> no needs (only asking for information)
- "What housing resources does UNC have?" -> no needs (only asking for information)
- "My roommate is really messy" -> no needs
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


def has_keyword(text, keywords):
    text = text.lower()
    return any(phrase in text for phrase in keywords)


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
    # Nemotron only judges the STUDENT's words (from the intake), never another agent's answer
    transcript = "\n".join(
        f"Student: {turn.get('user', '')}" for turn in conversation if turn.get("user", "").strip()
    ) or "No conversation to review."
    # Reviewer mode = another agent (resource/schedule) already wrote an answer for this student.
    # That agent handles everyday resources, so we only add life-safety help on top.
    reviewer_mode = any(turn.get("assistant", "").strip() for turn in conversation)

    # Step 1: Nemotron judges the needs (or we guess from the student's own words)
    needs = ask_model(transcript)
    if needs is None:
        needs = guess_needs(student_text)

    # Step 2: Which kinds of crisis are happening? (model OR keyword backup)
    def rated(category, level):
        return any(n["category"] == category and SEVERITIES.index(n["severity"]) >= SEVERITIES.index(level)
                   for n in needs)

    crisis_types = []
    if rated("mental_health", "crisis") or has_keyword(student_text, SELF_HARM_KEYWORDS):
        crisis_types.append("self_harm")
    if rated("active_threat", "severe") or has_keyword(student_text, ACTIVE_THREAT_KEYWORDS):
        crisis_types.append("active_threat")
    if rated("safety", "severe") or has_keyword(student_text, DANGER_KEYWORDS):
        crisis_types.append("danger")
    if rated("food", "severe"):
        crisis_types.append("food")
    if rated("housing", "severe"):
        crisis_types.append("housing")
    crisis = len(crisis_types) > 0

    # Step 2b: Check the other agents' answers for hallucinations or diagnosis
    response_problems = []
    safe_response = ""
    for turn in conversation:
        problems, safe_text = check_response(turn.get("assistant", ""))
        response_problems.extend(problems)
        safe_response = safe_text  # cleaned version of the latest answer

    # Step 3: Attach verified resources
    resources = []  # every resource, in order (for code that just needs the list)
    groups = []     # the same resources grouped by need, so each gets its own message

    def attach(group, items):
        already = [r["name"] for r in resources]
        new_items = [item for item in items if item["name"] not in already]
        add(resources, items)
        if new_items:
            groups.append({"group": group, "resources": new_items})

    # Life-safety help always comes first
    if "active_threat" in crisis_types:
        attach("active_threat", ACTIVE_THREAT_RESOURCES)
    if "self_harm" in crisis_types:
        attach("crisis", RESOURCES["crisis"])
    if "danger" in crisis_types:
        attach("danger", DANGER_RESOURCES)

    for need in needs:
        if reviewer_mode:
            break  # the other agent is already giving resources; only life-safety help is added
        level = SEVERITIES.index(need["severity"])
        if need["category"] in ("safety", "active_threat"):
            continue  # already handled by the life-safety groups above
        if need["category"] == "mental_health" and "self_harm" in crisis_types:
            continue  # crisis group already includes CAPS
        if need["category"] in ("mental_health", "community") and ("food" in crisis_types or "housing" in crisis_types):
            continue  # food/housing emergencies show only their own help, not CAPS
        if need["category"] == "community":
            if level >= SEVERITIES.index("moderate"):
                attach("community", RESOURCES.get("community", RESOURCES["mental_health"][:1]))
            continue
        if need["category"] == "lockout":
            if level >= SEVERITIES.index("moderate"):
                group = "lockout_" + need["lockout_type"]
                items = []
                if need["lockout_type"] == "room":
                    desk = find_service_desk(student_text)
                    if desk:
                        items.append(desk)
                attach(group, items + RESOURCES[group])
        elif level >= SEVERITIES.index("severe"):
            attach(need["category"], RESOURCES[need["category"]])

    # For food or housing emergencies, also point to people who know the student
    if (not reviewer_mode and ("food" in crisis_types or "housing" in crisis_types)
            and "also_reach_out" in RESOURCES):
        attach("also_reach_out", RESOURCES["also_reach_out"])

    # Always show the most urgent help first, and mental health / community (CAPS) last
    groups.sort(key=lambda g: GROUP_ORDER.index(g["group"]) if g["group"] in GROUP_ORDER else len(GROUP_ORDER))
    resources = [r for g in groups for r in g["resources"]]

    # Step 4: Return a labeled "form" (dictionary) the rest of the app can read
    return {
        "school": SCHOOL,
        "crisis": crisis,
        "crisis_types": crisis_types,  # e.g. ["active_threat"], ["self_harm"], ["danger"], ["food", "housing"]
        # False = don't show the other agent's answer as-is (life-safety crisis, or it had a problem).
        # Food/housing don't block the other agent: its resource list or plan is still shown.
        "allow": not any(t in LIFE_SAFETY for t in crisis_types) and not response_problems,
        "response_problems": response_problems,  # hallucinations / diagnosis found in the answer
        "safe_response": safe_response,          # the answer with those lines removed
        "needs": needs,
        "resources": resources,
        "groups": groups,
    }


# ---------------------------------------------------------------
# PART 5b: Show the result to the student in plain, caring words.
# review_conversation() gives the data; this function displays it.
# ---------------------------------------------------------------
# Order the student sees help in: life-safety first, then basic needs, CAPS last
GROUP_ORDER = ["active_threat", "crisis", "danger", "food", "housing", "also_reach_out",
               "lockout_room", "lockout_car", "mental_health", "community"]

GROUP_MESSAGES = {
    "crisis": {
        "intro": "Your life is valuable, and you don't have to face this alone. "
                 "Please reach out now. Support is available 24/7:",
        "outro": "What's happening right now does not decide your future, and life can change "
                 "in ways you can't see yet, sometimes faster than you'd expect. You deserve to "
                 "be heard, and there are people who want to be there for you. This may feel like "
                 "one of the darkest times, but brighter days are ahead, and you can make it through. "
                 "It's okay to be scared; reaching out is a brave first step.",
    },
    "active_threat": {
        "intro": "If there is an active shooter or armed person near you, act now: "
                 "RUN if you can get out safely. If you can't, HIDE: lock or barricade the door, "
                 "turn off lights, and silence your phone. FIGHT only as a last resort. "
                 "Call 911 as soon as it is safe to do so:",
        "outro": "Follow Alert Carolina messages for updates, and stay where you are safe until "
                 "police say it is clear.",
    },
    "danger": {
        "intro": "Your safety comes first. If you are in danger right now, call 911 or UNC Police. "
                 "If you see a Blue Call Box on campus (a black pole with a blue light on top), "
                 "pick up the receiver: it sends your location to UNC Police, and they will come "
                 "even if you can't speak. If you can, move toward a well-lit place with other people around you.",
        "outro": "You did the right thing by reaching out. Stay on the phone with them until you feel safe.",
    },
    "mental_health": {
        "intro": "It takes strength to share how you're feeling, and you don't have to carry this "
                 "alone. These are good places to start:",
    },
    "food": {
        "intro": "Not having enough to eat is an emergency, and you deserve help right away. "
                 "No one should have to go hungry, and asking for food help is completely okay. "
                 "Here's where you can get food today:",
    },
    "housing": {
        "intro": "Losing a safe place to stay is an emergency, and you don't have to figure this out alone. "
                 "Please reach out today. These offices help students in exactly this situation:",
    },
    "also_reach_out": {
        "intro": "You can also reach out to people who know you. They can help you get support "
                 "from the Dean of Students and other offices:",
    },
    "community": {
        "intro": "Feeling lonely or like you haven't found your people yet is really common in college, "
                 "and it's worth talking about. You don't have to figure it out alone:",
    },
    "lockout_room": {
        "intro": "Getting locked out happens to everyone. Here's how to get back in:",
    },
    "lockout_car": {
        "intro": "Locked out of your car? Here's who can help on campus:",
    },
}


def format_resource(r):
    """One resource -> 'name: phone (details)' with the link on its own line."""
    line = "  - " + r["name"]
    if r.get("phone"):
        line += ": " + r["phone"]
    if r.get("details"):
        line += " (" + r["details"] + ")"
    if r.get("link"):
        line += "\n      " + r["link"]
    return line


def print_safety_result(result):
    """Print the safety result as a readable message. Prints nothing if there are no needs."""
    # If another agent's answer had unverified info or a diagnosis, show the cleaned version
    if result.get("response_problems") and not result.get("crisis"):
        print()
        print("Note: part of this answer was removed because it could not be verified "
              "against our list of UNC resources, or it sounded like medical advice. "
              "Only verified information is shown:")
        if result.get("safe_response"):
            print(result["safe_response"])
    for group in result.get("groups", []):
        message = GROUP_MESSAGES.get(group["group"], {})
        print()
        print(message.get("intro", "Here are some places that can help:"))
        for r in group["resources"]:
            print(format_resource(r))
        if message.get("outro"):
            print()
            print(message["outro"])
    if result.get("groups"):
        print()


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
        "I just heard gunshots outside Davis Library. What do I do?",
        "What's the best pizza place on Franklin Street?",
        "Where is the food pantry on campus?",
        "What housing resources does UNC have?",
        "I'm getting evicted this Friday and I'm so stressed.",
        "I don't have any friends here and I feel really lonely.",
    ]
    print("=== Checking other agents' answers (hallucinations and diagnosis) ===")
    sample_answers = [
        "Carolina Cupboard: https://carolinacupboard.web.unc.edu\nCall CAPS at 919-966-3658",
        "Call the UNC Wellness Hotline at 919-555-0123 or visit uncwellnesshelp.com",
        "It sounds like you have depression. You should take melatonin.\nCAPS: 919-966-3658",
    ]
    for answer in sample_answers:
        problems, cleaned = check_response(answer)
        print("ANSWER:", answer.replace("\n", " | "))
        print("PROBLEMS:", problems or "none")
        print("-" * 60)

    print("=== Checking student messages ===")
    for message in test_messages:
        result = review_conversation([{"user": message, "assistant": ""}])
        print("STUDENT:", message)
        print("NEEDS:", [n["category"] + ":" + n["severity"] for n in result["needs"]],
              "| crisis:", result["crisis"], result["crisis_types"])
        print_safety_result(result)
        print("-" * 60)
