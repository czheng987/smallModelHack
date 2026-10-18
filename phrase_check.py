"""Score intake phrases by agreement across many servers.

Each server phrases the same student message. The phrases are good when at
least 90% of the servers return the same split, or a very similar one. Set
PHRASE_PASS_RATE=0.95 to require 95%, and PHRASE_SERVERS to change how many
servers vote. The default is 100.
"""

import os
import re

# 90% of the servers must agree. Set PHRASE_PASS_RATE=0.95 to require 95%.
PASS_RATE = float(os.environ.get("PHRASE_PASS_RATE", "0.90"))
# How many servers phrase the message together.
SERVER_COUNT = max(1, int(os.environ.get("PHRASE_SERVERS", "100")))

GLUE_WORDS = {
    "a", "an", "the", "to", "for", "of", "on", "in", "at", "and", "or", "my",
    "me", "i", "you", "can", "could", "please", "help", "need", "needs", "with",
    "am", "is", "are", "it", "this", "that", "just", "want", "wants", "would",
    "like", "do", "does", "have", "has", "be", "from", "about", "into",
}
CHATTER = {"thanks", "thank", "hello", "hi", "hey", "ok", "okay", "bye", "goodbye"}
INTRO = re.compile(
    r"^(?:(?:please|can|could|you|i|i'm|im|am|need|needs|to|help|me|with|my|"
    r"just|want|wants|would|like|where|how|do|does|is|are|there|a|an|the|find|"
    r"have|has|got|get)\s+)+",
    re.IGNORECASE,
)
# A resource need names the help itself: counseling, food, a place to stay, a lockout.
# "sign housings" does not match. That is a calendar task and stays in schedule[].
RESOURCE_WORD = re.compile(
    r"\b("
    r"counsel(?:ing|or|ors)?|therap(?:y|ist|ists)|caps|mental|"
    r"anxiou\w*|overwhelm\w*|stress(?:ed|ful)?|depress\w*|"
    r"lonely|loneliness|panic(?:king)?|panicked|support|"
    r"crisis|suicidal|suicide|"
    r"hungry|hunger|homeless|evict\w*|shelter|lockout|pantry|"
    r"grocer(?:y|ies)?"
    r")\b|"
    r"\b(?:talk|speak|chat)\s+(?:to|with)\b|"
    r"\bsomeone to (?:talk|speak|listen)\b|"
    r"\blocked out\b|\bshut out\b|\blost (?:my )?keys?\b|"
    r"\bnowhere to (?:stay|live|sleep)\b|\bplace to stay\b|"
    r"\bhousing (?:resource|resources|help|support|insecurity)\b|"
    r"\b(?:need|find) (?:food|housing|a place to stay)\b|"
    r"\bcan(?:not|'t) afford (?:food|rent)\b",
    re.IGNORECASE,
)
# Calendar verbs. "sign housings" and "study for midterm" stay on the schedule side.
_SCHEDULE_ACTION = re.compile(
    r"\b(?:sign|signing|schedule|scheduling|study|studying|submit|submitting|"
    r"complete|completing|finish|finishing|review|reviewing|plan|planning|"
    r"work|working|read|reading|write|writing|attend|attending)\b",
    re.IGNORECASE,
)
_FOOD_WORD = re.compile(r"\bfood\b", re.IGNORECASE)
_HOUSING_WORD = re.compile(r"\bhous(?:e|ing|ings)?\b", re.IGNORECASE)
_RENT_WORD = re.compile(r"\brent\b", re.IGNORECASE)
_ACADEMIC = re.compile(
    r"\b(?:science|class|classes|exam|exams|midterm|final|finals|homework|"
    r"lab|labs|paper|papers|course|lecture|quiz|assignment)\b",
    re.IGNORECASE,
)
_LOCKOUT = re.compile(
    r"\blocked out\b|\blockout\b|\bshut out\b|\blost (?:my )?keys?\b|"
    r"\blocked\b.{0,30}\b(?:dorm|room|car|vehicle)\b|"
    r"\b(?:dorm|room|car|vehicle)\b.{0,30}\blocked\b",
    re.IGNORECASE,
)
_SOMEONE_NEED = re.compile(
    r"\bsomeone\b.+\b(?:talk|listen|speak|support|counsel)\b|"
    r"\b(?:talk|listen|speak|support|counsel)\b.+\bsomeone\b",
    re.IGNORECASE,
)
# Verbs alone are not a schedule item. "Study on" has no noun; "study for midterm" does.
ACTION_VERBS = {
    "study", "studying", "schedule", "scheduling", "sign", "signing", "plan",
    "planning", "work", "working", "finish", "finishing", "complete", "completing",
    "review", "reviewing", "practice", "practicing", "meet", "go", "going",
    "make", "making", "take", "taking", "prepare", "preparing", "read", "reading",
    "write", "writing", "submit", "submitting", "attend", "attending", "book",
    "booking", "register", "registering", "start", "starting", "focus", "focusing",
    "revise", "revising", "cram", "cramming", "organize", "organizing", "do",
    "doing", "get", "getting", "set", "setting", "look", "looking", "keep",
    "keeping", "put", "putting",
}
# "talk" or "find" does not say what the student needs. "someone" and "counseling" do.
RESOURCE_VERBS = {
    "talk", "talking", "find", "finding", "look", "looking", "feel", "feeling",
    "get", "getting", "see", "seeing", "speak", "speaking", "chat", "chatting",
    "help", "need", "want", "ask", "asking",
}


def shorten_phrase(phrase):
    """Drop filler so a planner sees the activity, not the whole sentence."""
    text = " ".join(str(phrase).split()).strip(" .?")
    previous = None
    while text and text != previous:
        previous = text
        text = INTRO.sub("", text).strip(" .?")
    return text


def tokens(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower().replace("'", ""))
    kept = []
    for word in words:
        if word in GLUE_WORDS or word in CHATTER:
            continue
        if len(word) >= 3 or any(character.isdigit() for character in word):
            kept.append(word)
    return kept


def task_chunks(message):
    """Activities named in the message, already shortened."""
    chunks = []
    for part in re.split(r",|&|\band\b", str(message), flags=re.IGNORECASE):
        piece = shorten_phrase(part)
        if tokens(piece):
            chunks.append(piece)
    return chunks


def prepare_phrases(phrases):
    """Split a combined model line and shorten each piece."""
    prepared = []
    for phrase in phrases or []:
        pieces = task_chunks(phrase)
        if not pieces:
            short = shorten_phrase(phrase)
            pieces = [short] if short else []
        for piece in pieces:
            if piece and piece not in prepared:
                prepared.append(piece)
    return prepared


def schedule_has_noun(phrase):
    """A schedule phrase has to name the thing, not stop at a verb like "study on"."""
    words = [
        word for word in tokens(phrase)
        if word not in ACTION_VERBS and word not in RESOURCE_VERBS
        and not word.isdigit() and any(character.isalpha() for character in word)
    ]
    return bool(words)


def resource_has_need(phrase):
    """A resource phrase has to name the need, not stop at "help" or "talk"."""
    words = [
        word for word in tokens(phrase)
        if word not in RESOURCE_VERBS and word not in ACTION_VERBS and not word.isdigit()
    ]
    return bool(words)


def drop_bare_phrases(schedule, resource):
    """Remove schedule verbs with no noun and resource lines with no need."""
    schedule_out = [phrase for phrase in (schedule or []) if schedule_has_noun(phrase)]
    resource_out = [phrase for phrase in (resource or []) if resource_has_need(phrase)]
    return schedule_out, resource_out


def is_resource_phrase(phrase):
    """True when the phrase names a campus resource need and belongs in resource[]."""
    if not resource_has_need(phrase):
        return False
    if RESOURCE_WORD.search(phrase) or _LOCKOUT.search(phrase) or _SOMEONE_NEED.search(phrase):
        return True
    # "sign housings" and "study for food science" name a calendar task.
    if _SCHEDULE_ACTION.search(phrase) or _ACADEMIC.search(phrase):
        return False
    return bool(_FOOD_WORD.search(phrase) or _HOUSING_WORD.search(phrase) or _RENT_WORD.search(phrase))


def arrays_are_correct(schedule, resource):
    """Resource needs belong in resource[]. Schedule tasks stay in schedule[]."""
    if any(is_resource_phrase(phrase) for phrase in (schedule or [])):
        return False
    if any(not is_resource_phrase(phrase) for phrase in (resource or [])):
        return False
    return True


def arrange_phrases(schedule, resource):
    """Shorten model phrases and put each one in schedule[] or resource[]."""
    schedule_out = []
    resource_out = []
    for phrase in prepare_phrases(list(schedule or []) + list(resource or [])):
        if is_resource_phrase(phrase):
            resource_out.append(phrase)
        elif schedule_has_noun(phrase):
            schedule_out.append(phrase)
    return schedule_out, resource_out


def propose_split(message):
    """Rule split used when the model reply does not clear the pass rate."""
    return arrange_phrases(task_chunks(message), [])


def _covers(chunk, phrases):
    needed = set(tokens(chunk))
    if not needed:
        return True
    for phrase in phrases:
        found = set(tokens(phrase))
        if len(needed & found) / len(needed) >= 0.5:
            return True
    return False


def _faithful(phrase, message_tokens):
    found = set(tokens(phrase))
    return not found or found <= message_tokens


def _short(phrase, message):
    if shorten_phrase(phrase) != phrase:
        return False
    if len(phrase) < len(str(message).strip()):
        return True
    lowered = str(message).lower()
    return "," not in message and " and " not in lowered and "&" not in message


def verify_split(message, schedule, resource):
    """Reward in [0, 1]. NeMo Gym returns this from verify() after array assignment."""
    if any(not schedule_has_noun(phrase) for phrase in (schedule or [])):
        return 0.5
    if any(not resource_has_need(phrase) for phrase in (resource or [])):
        return 0.5
    if not arrays_are_correct(schedule, resource):
        return 0.5
    phrases = prepare_phrases(list(schedule or []) + list(resource or []))
    expected = task_chunks(message)
    if not expected:
        return 1.0 if not phrases else 0.0
    if not phrases:
        return 0.0
    message_tokens = set(tokens(message))
    if any(not _faithful(phrase, message_tokens) for phrase in phrases):
        return 0.5
    covered = sum(1 for chunk in expected if _covers(chunk, phrases))
    short = sum(1 for phrase in phrases if _short(phrase, message))
    total = len(expected) + len(phrases)
    if total == 0:
        return 0.0
    return (covered + short) / total


def passes(reward, minimum=None):
    bar = PASS_RATE if minimum is None else minimum
    return reward >= bar


def split_keys(schedule, resource):
    """Activity keys, so small wording differences still count as the same phrase."""
    keys = set()
    for label, phrases in (("s", schedule or []), ("r", resource or [])):
        for phrase in phrases:
            key = " ".join(tokens(phrase))
            if key:
                keys.add(f"{label}:{key}")
    return keys


def splits_similar(left, right):
    """True when two servers returned the same activities, or at least 90% of them."""
    left_keys = split_keys(*left)
    right_keys = split_keys(*right)
    if not left_keys and not right_keys:
        return True
    if not left_keys or not right_keys:
        return False
    overlap = len(left_keys & right_keys) / len(left_keys | right_keys)
    return overlap >= PASS_RATE


def consensus(results):
    """Majority split, and the share of servers that match it.

    A missing vote (None) does not match. The share uses every server that was
    asked, so a server that never answered pulls the agreement down.
    """
    asked = len(results)
    votes = []
    for item in results:
        if item is None:
            continue
        votes.append(drop_bare_phrases(*item))
    if asked == 0 or not votes:
        return [], [], 0.0
    best = votes[0]
    best_hits = -1
    for candidate in votes:
        hits = sum(1 for other in votes if splits_similar(candidate, other))
        if hits > best_hits:
            best = candidate
            best_hits = hits
    return list(best[0]), list(best[1]), best_hits / asked


def accepted_split(message, results):
    """Keep the majority only when the servers agree and the phrases are accurate."""
    arranged = []
    for item in results:
        if item is None:
            arranged.append(None)
        else:
            arranged.append(arrange_phrases(*item))
    schedule, resource, rate = consensus(arranged)
    if not passes(rate):
        return False, schedule, resource, rate
    if not passes(verify_split(message, schedule, resource)):
        return False, schedule, resource, rate
    return True, schedule, resource, rate
