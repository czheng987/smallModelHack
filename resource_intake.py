import json
import re
from pathlib import Path

from safety_agent import print_safety_result, review_conversation
from vllm_model import VLLM_MODEL, open_client, reply_text

client = open_client()
RESOURCE_MODEL = VLLM_MODEL
RESOURCE_FILE = Path(__file__).with_name("unc_resources.json")
MIN_MATCHES = 2
MAX_MATCHES = 5

# Words that point a phrase at one section of unc_resources.json.
CATEGORY_WORDS = {
    "crisis": (
        "suicide", "suicidal", "kill myself", "end my life", "self-harm",
        "self harm", "hurt myself", "in danger", "emergency",
    ),
    "mental_health": (
        "counsel", "caps", "therap", "talk", "anxi", "stress", "overwhelm",
        "mental", "support", "lonely", "panic", "crisis", "help", "someone",
    ),
    "food": ("food", "hungry", "hunger", "eat", "pantry", "meal", "grocer"),
    "housing": ("hous", "evict", "rent", "homeless", "shelter", "nowhere"),
    "lockout_room": ("lock", "key", "dorm", "room", "shut out"),
    "lockout_car": ("car", "vehicle", "parking"),
}


def clean_phrases(resource):
    """Accept resource[] as a list of phrases."""
    if isinstance(resource, str):
        resource = [resource]
    if not isinstance(resource, list):
        return []
    phrases = []
    for item in resource:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.split())
        if phrase:
            phrases.append(phrase)
    return phrases


def load_catalog(path=RESOURCE_FILE):
    with open(path) as handle:
        data = json.load(handle)
    records = []
    for category in ("crisis", "mental_health", "food", "housing", "lockout_room", "lockout_car"):
        for item in data.get(category, []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            records.append({
                "category": category,
                "name": item.get("name", ""),
                "phone": item.get("phone", ""),
                "link": item.get("link", ""),
                "details": item.get("details", ""),
            })
    for dorm, info in data.get("service_desks", {}).items():
        if not isinstance(info, dict):
            continue
        records.append({
            "category": "service_desk",
            "name": f"{info.get('desk', dorm)} Service Desk",
            "phone": info.get("phone", ""),
            "link": "",
            "details": f"Housing desk for {dorm}. Issues temporary replacement keys.",
            "dorm": dorm,
        })
    return records


def phrase_words(phrases):
    return set(re.findall(r"[a-z0-9']+", " ".join(phrases).lower()))


def hint_matches(hint, words):
    """Match a stem to a whole word. Short words such as 'car' must be exact."""
    if hint in {"car", "rent"}:
        return hint in words
    return any(word.startswith(hint) for word in words)


def score_record(record, phrases):
    text = " ".join(phrases).lower()
    words = phrase_words(phrases)
    blob_words = set(re.findall(
        r"[a-z0-9']+",
        " ".join((
            record.get("category", ""),
            record.get("name", ""),
            record.get("details", ""),
            record.get("dorm", ""),
        )).lower(),
    ))
    score = 0
    for word in words:
        if len(word) < 4:
            continue
        if word in blob_words:
            score += 2
    category = record.get("category")
    if category == "service_desk":
        dorm = record.get("dorm", "")
        if dorm and dorm in text:
            score += 6
        return score
    for hint in CATEGORY_WORDS.get(category, ()):
        if hint_matches(hint, words):
            score += 3
    return score


def search_resources(phrases, catalog):
    """Rank JSON entries by how well they match resource[]."""
    ranked = []
    seen = set()
    for record in catalog:
        score = score_record(record, phrases)
        if score <= 0 or record["name"] in seen:
            continue
        seen.add(record["name"])
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in ranked]


def choose_names(phrases, candidates):
    """Ask Nemotron to pick 2-5 names from the matched JSON entries."""
    catalog_lines = []
    for record in candidates[:8]:
        catalog_lines.append(f"- {record['name']}: {record.get('details', '')}")
    numbered = "\n".join(f"- {phrase}" for phrase in phrases)
    response = client.chat.completions.create(
        model=RESOURCE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You match a student's resource[] phrases to campus resources. "
                    "Choose 2 to 5 names from the catalog. Use the names exactly. "
                    "Do not invent resources, phone numbers, or a diagnosis. "
                    'Return JSON only: {"resources": ["Name"]}'
                ),
            },
            {
                "role": "user",
                "content": f"resource[]:\n{numbered}\n\nCatalog:\n" + "\n".join(catalog_lines),
            },
        ],
        max_tokens=300,
        temperature=0,
    )
    text = reply_text(response.choices[0].message)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    data = json.loads(text[start:end + 1])
    names = data.get("resources", [])
    if not isinstance(names, list):
        return []
    return [name.strip() for name in names if isinstance(name, str) and name.strip()]


def select_resources(phrases, matches):
    """Keep between 2 and 5 matched resources when the catalog has that many."""
    if not matches:
        return []
    by_name = {}
    for record in matches:
        by_name.setdefault(record["name"], record)
    chosen = []
    try:
        for name in choose_names(phrases, matches):
            record = by_name.get(name)
            if record and record not in chosen:
                chosen.append(record)
    except Exception:
        chosen = []
    if len(chosen) < MIN_MATCHES:
        for record in matches:
            if record not in chosen:
                chosen.append(record)
            if len(chosen) >= MIN_MATCHES:
                break
    return chosen[:MAX_MATCHES]


def format_resource(index, record):
    lines = [f"{index}. {record['name']}"]
    if record.get("phone"):
        lines.append(f"   Phone: {record['phone']}")
    if record.get("link"):
        lines.append(f"   Link: {record['link']}")
    if record.get("details"):
        lines.append(f"   {record['details']}")
    return "\n".join(lines)


def asks_for_resources(phrases):
    """True only when the student is asking for resources, not just describing a problem."""
    text = " ".join(phrases).lower()
    markers = (
        "resource",
        "resources",
        "where can i",
        "where do i",
        "who can i",
        "who should i",
        "who do i",
        "can you find",
        "can you give",
        "help me find",
        "looking for",
        "counsel",
        "caps",
        "hotline",
        "phone number",
        "someone to talk",
        "someone i can talk",
        "do you know",
    )
    return any(marker in text for marker in markers)


def release_response(student_text, response_text):
    """Run the drafted resource list through the safety agent before showing it."""
    result = review_conversation([
        {"user": student_text, "assistant": response_text},
    ])
    print_safety_result(result)
    return not result.get("crisis", False)


def get_resource_needs(resource):
    """Read resource[] and list 2-5 matching entries from unc_resources.json."""
    resource = clean_phrases(resource)
    if not resource:
        return []

    student_text = "\n".join(resource)
    if not asks_for_resources(resource):
        result = review_conversation([
            {"user": student_text, "assistant": ""},
        ])
        if result.get("crisis"):
            print_safety_result(result)
        return []

    matches = search_resources(resource, load_catalog())
    chosen = select_resources(resource, matches)
    if not chosen:
        print("I could not find a matching resource in the campus list.")
        return []

    rendered = []
    blocks = []
    for index, record in enumerate(chosen, start=1):
        blocks.append(format_resource(index, record))
        rendered.append(record)
    response_text = "\n".join(blocks)
    if not release_response("\n".join(resource), response_text):
        return rendered

    print("Here are resources that match what you asked for:")
    print(response_text)
    return rendered
