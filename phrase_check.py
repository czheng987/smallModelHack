"""Score intake phrases with the NeMo Gym reward and pass-rate pattern.

Each student message is one task. verify_split returns a reward from 0 to 1:
every activity is still there, nothing was invented, and each phrase is short
enough for the planner. A split is returned only when the reward reaches the
pass rate (90%, or 95% when PHRASE_PASS_RATE=0.95).
"""

import os
import re

# A live message is one task. 90% is the bar; set PHRASE_PASS_RATE=0.95 to require 95%.
PASS_RATE = float(os.environ.get("PHRASE_PASS_RATE", "0.90"))

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


def pass_rate(messages):
    """pass@1 for the rule splitter across these student messages."""
    if not messages:
        return 0.0
    rewards = [
        verify_split(message, *propose_split(message))
        for message in messages
    ]
    return sum(rewards) / len(rewards)
