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
RESOURCE_WORD = re.compile(
    r"\b(counsel(?:ing|or)?|therap(?:y|ist)|caps|mental|anxiou\w*|overwhelm\w*|"
    r"stress(?:ed)?|depress\w*|lonely)\b|\btalk to\b|\bsomeone to talk\b",
    re.IGNORECASE,
)


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


def arrange_phrases(schedule, resource):
    """Shorten model phrases and put each one on the schedule or resource side."""
    schedule_out = []
    resource_out = []
    for phrase in prepare_phrases(list(schedule or []) + list(resource or [])):
        if RESOURCE_WORD.search(phrase):
            resource_out.append(phrase)
        else:
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
    """Reward in [0, 1]. NeMo Gym would return this from verify()."""
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
    votes = [item for item in results if item is not None]
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
    schedule, resource, rate = consensus(results)
    if not passes(rate):
        return False, schedule, resource, rate
    if not passes(verify_split(message, schedule, resource)):
        return False, schedule, resource, rate
    return True, schedule, resource, rate
