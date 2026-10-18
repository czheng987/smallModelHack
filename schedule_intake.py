import json
import re
import time
from datetime import datetime, timedelta

from safety_agent import print_safety_result, review_conversation
from vllm_model import VLLM_MODEL, open_client, reply_text

client = open_client()
SCHEDULE_MODEL = VLLM_MODEL

# A focused block stays short so one task does not take over the day.
MAX_WORK_MINUTES = 45
BREAK_GAP_MINUTES = 10
# Four hours of work plus breaks is a full day. The rest waits for another day.
MAX_WORK_PER_DAY = 4 * 60
DAY_START = 9 * 60
DAY_END = 21 * 60


def clean_phrases(schedule):
    """Accept schedule[] as a list of phrases."""
    if isinstance(schedule, str):
        schedule = [schedule]
    if not isinstance(schedule, list):
        return []
    phrases = []
    for item in schedule:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.split())
        if phrase:
            phrases.append(phrase)
    return phrases


LIST_INTRO = re.compile(
    r"^(?:(?:please|can|you|i(?:'m| am)?|need|help|me|with|my|the|a|an|schedul\w+)\s+)+",
    re.IGNORECASE,
)


def tidy_activity(piece):
    """Drop list filler so 'help with scheduling shopping' becomes 'shopping'."""
    piece = " ".join(piece.split()).strip(" .")
    previous = None
    while piece and piece != previous:
        previous = piece
        piece = LIST_INTRO.sub("", piece).strip(" .")
    piece = re.sub(r"\bfix$", "", piece, flags=re.IGNORECASE).strip()
    return piece


def timeline_marker(text):
    """A short timeline phrase that listed tasks should keep."""
    patterns = (
        r"\bnext month\b",
        r"\bthis month\b",
        r"\ba month\b",
        r"\bone month\b",
        r"\bnext week\b",
        r"\bthis week\b",
        r"\b(?:a|an|one|two|three|four|\d+)\s+weeks?\b",
        r"\b\d+\s+days?\b",
        r"\b(?:today|tomorrow|tonight)\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text), re.IGNORECASE)
        if match:
            return match.group(0)
    return weekday_of(text)


def split_listed_activities(phrases, now=None):
    """Turn one sentence that lists several activities into one phrase each."""
    expanded = []
    for phrase in phrases:
        body = tidy_activity(phrase) or phrase
        pieces = [tidy_activity(part) for part in re.split(r",|\band\b", body, flags=re.IGNORECASE)]
        pieces = [part for part in pieces if significant_tokens(part)]
        if len(pieces) < 2:
            expanded.append(phrase)
            continue
        marker = timeline_marker(phrase)
        if marker:
            pieces = [
                part if timeline_marker(part) else f"{part} {marker}"
                for part in pieces
            ]
        expanded.extend(pieces)
    return expanded


def clock_to_minutes(value):
    text = str(value).strip().lower().replace(".", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm)?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def minutes_to_clock(minutes):
    minutes = minutes % (24 * 60)
    hour, minute = divmod(minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def normalize_plan(raw_days, stressed=False):
    """Drop overlapping blocks, cap long work, and leave room for breaks."""
    if not isinstance(raw_days, list):
        return []
    work_limit = 25 if stressed else MAX_WORK_MINUTES
    day_limit = 2 * 60 if stressed else MAX_WORK_PER_DAY
    plan = []
    for day in raw_days:
        if not isinstance(day, dict):
            continue
        label = str(day.get("day") or "Today").strip() or "Today"
        blocks = []
        for block in day.get("blocks") or []:
            if isinstance(block, str):
                block = {"task": block}
            if not isinstance(block, dict):
                continue
            start = clock_to_minutes(block.get("start") or block.get("start_time"))
            end = clock_to_minutes(block.get("end") or block.get("end_time"))
            if start is None and isinstance(block.get("time"), str) and "-" in block["time"]:
                left, right = block["time"].split("-", 1)
                start = clock_to_minutes(left)
                end = clock_to_minutes(right)
            task = block.get("task") or block.get("name") or block.get("title") or block.get("activity") or ""
            task = " ".join(str(task).split())
            if not task:
                continue
            is_break = "break" in task.lower()
            duration = 10 if is_break else work_limit
            if start is None:
                start = DAY_START
                end = start + duration
            elif end is None or end <= start:
                end = start + duration
            elif not is_break and end - start > work_limit:
                end = start + work_limit
            blocks.append({
                "start": start,
                "end": end,
                "task": task,
                "is_break": is_break,
            })
        blocks.sort(key=lambda block: block["start"])

        placed = []
        work_minutes = 0
        for block in blocks:
            duration = block["end"] - block["start"]
            start = block["start"]
            if placed:
                earliest = placed[-1]["end"]
                if not placed[-1]["is_break"] and not block["is_break"]:
                    earliest += BREAK_GAP_MINUTES
                start = max(start, earliest)
            if start < DAY_START:
                start = DAY_START
            end = start + duration
            if end > DAY_END:
                continue
            if not block["is_break"] and work_minutes + duration > day_limit:
                continue
            placed.append({
                "start": start,
                "end": end,
                "task": block["task"],
                "is_break": block["is_break"],
            })
            if not block["is_break"]:
                work_minutes += duration
        if placed:
            plan.append({"day": label, "blocks": placed})
    return plan


def render_plan(plan):
    lines = []
    for day in plan:
        lines.append(day["day"])
        for block in day["blocks"]:
            lines.append(
                f"  {minutes_to_clock(block['start'])} – {minutes_to_clock(block['end'])}  {block['task']}"
            )
        lines.append("")
    return "\n".join(lines).strip()


WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Words that describe feelings or glue a sentence together, not an assignment.
GLUE_WORDS = {
    "i", "im", "me", "my", "a", "an", "the", "and", "or", "to", "of", "for",
    "with", "on", "in", "at", "by", "is", "am", "are", "be", "been", "was",
    "have", "has", "had", "do", "does", "did", "so", "very", "really", "just",
    "too", "much", "lot", "this", "that", "these", "those", "it", "its",
    "stressed", "stress", "overwhelmed", "overwhelm", "anxious", "anxiety",
    "behind", "cant", "cannot", "panic", "burned", "burnt", "out", "exhausted",
    "swamped", "feeling", "feel", "felt", "need", "needs", "help", "please",
    "today", "tomorrow", "tonight", "due", "deadline", "about", "into", "from",
    "some", "any", "all", "also", "still", "really", "kinda", "kind", "of",
}
GLUE_WORDS.update(day.lower() for day in WEEKDAYS)


def significant_tokens(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower().replace("'", ""))
    tokens = set()
    for word in words:
        if word in GLUE_WORDS:
            continue
        if len(word) >= 3 or any(character.isdigit() for character in word):
            tokens.add(word)
    return tokens


def task_phrases(schedule):
    """Keep phrases that name work. Drop lines that only say the student is stressed."""
    return [phrase for phrase in schedule if significant_tokens(phrase)]


def shares_task(left, right):
    return bool(significant_tokens(left) & significant_tokens(right))


def weekday_of(text):
    for name in WEEKDAYS:
        if re.search(rf"\b{name}\b", str(text), re.IGNORECASE):
            return name
    return None


def is_canned_example(plan, schedule):
    """The old prompt sample was Thursday 4:00–4:25, Outline the paper."""
    work = [block for day in plan for block in day["blocks"] if not block["is_break"]]
    if len(plan) != 1 or len(work) != 1:
        return False
    block = work[0]
    student = " ".join(schedule).lower()
    copied = (
        plan[0]["day"].lower() == "thursday"
        and block["task"].lower() == "outline the paper"
        and block["start"] == 16 * 60
        and block["end"] == 16 * 60 + 25
    )
    if not copied:
        return False
    return "outline" not in student or "thursday" not in student


def plan_uses_student_tasks(plan, schedule, stressed=False):
    """Accept a model plan only when it schedules the student's own tasks."""
    phrases = task_phrases(schedule)
    work = [block for day in plan for block in day["blocks"] if not block["is_break"]]
    if not phrases or not work or is_canned_example(plan, schedule):
        return False
    for block in work:
        if not any(shares_task(block["task"], phrase) for phrase in phrases):
            return False
    for phrase in phrases:
        matched = [block for block in work if shares_task(block["task"], phrase)]
        if not matched:
            return False
        # A paper or exam is not done in one 25 minute slot.
        if session_count(phrase, stressed) > 1 and len(matched) < 2:
            return False
    return True


def load_json_values(text):
    """Pull every JSON object or list out of a model reply."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```json", "").replace("```", "")
    decoder = json.JSONDecoder()
    values = []
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            data, length = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            snippet = re.sub(r",\s*([}\]])", r"\1", text[index:])
            try:
                data, _ = decoder.raw_decode(snippet)
            except json.JSONDecodeError:
                index += 1
                continue
            values.append(data)
            index += 1
            continue
        values.append(data)
        index += max(length, 1)
    return values


def days_from_reply(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("days", "schedule", "plan"):
        if isinstance(data.get(key), list):
            return data[key]
    if any(key in data for key in ("blocks", "task", "name", "title")):
        return [data]
    return []


def current_moment(now=None):
    """Weekday and minute-of-day when the student messages."""
    if now is None:
        now = time.localtime()
    return WEEKDAYS[now.tm_wday], now.tm_hour * 60 + now.tm_min


def earliest_start(day_offset, now_minutes):
    """First minute a block may start. Today cannot begin in the past."""
    if day_offset > 0:
        return DAY_START
    if now_minutes < DAY_START:
        return DAY_START
    start = now_minutes + 5
    if start % 5:
        start += 5 - (start % 5)
    if start >= DAY_END:
        return None
    return start


def is_today_label(label, weekday, now=None):
    text = str(label).strip().lower()
    names = {weekday.lower(), "today", "tonight", format_plan_day(now, 0).lower()}
    return text in names


def plan_respects_clock(plan, now=None):
    weekday, now_minutes = current_moment(now)
    earliest = earliest_start(0, now_minutes)
    for day in plan:
        if not is_today_label(day["day"], weekday, now):
            continue
        for block in day["blocks"]:
            if earliest is None or block["start"] < earliest:
                return False
    return True


def parse_plan(reply, stressed=False, schedule=None, now=None):
    """Use a model plan only when it names the student's tasks, not the sample."""
    matched = None
    first = None
    for data in load_json_values(reply):
        plan = normalize_plan(days_from_reply(data), stressed=stressed)
        if not plan:
            continue
        if first is None:
            first = plan
        if (
            schedule is not None
            and plan_uses_student_tasks(plan, schedule, stressed)
            and plan_respects_clock(plan, now)
        ):
            matched = plan
            break
    if schedule is not None:
        return matched or []
    return first or []


ONE_TIME_EVENT = re.compile(
    r"\b(shop(?:ping)?|grocer(?:y|ies)|errands?|appointments?|haircuts?|"
    r"conferences?|meetings?|"
    r"pick\s?up|drop\s?off|laundry|pharmacy|bank|mail|packages?)\b",
    re.IGNORECASE,
)
STUDY_WORD = re.compile(
    r"\b(study|studying|review|revise|prep|prepare|preparing|practice)\b",
    re.IGNORECASE,
)
EXAM_WORD = re.compile(r"\b(midterms?|finals?|exams?|tests?)\b", re.IGNORECASE)
SIZE_WORD = r"small|short|quick|minor|little"
LARGE_WORD = r"large|big|major|long|huge"
WORK_NOUN = r"projects?|assignments?|papers?|essays?"


def explicitly_small(phrase):
    return re.search(rf"\b(?:{SIZE_WORD})\b", phrase, re.IGNORECASE) and re.search(
        rf"\b(?:{WORK_NOUN})\b", phrase, re.IGNORECASE
    )


def explicitly_large(phrase):
    return re.search(rf"\b(?:{LARGE_WORD})\b", phrase, re.IGNORECASE) and re.search(
        rf"\b(?:{WORK_NOUN})\b", phrase, re.IGNORECASE
    )


def activity_kind(phrase):
    """once: one sitting. chunks: split across periods. exam: study chunks plus one sitting."""
    if ONE_TIME_EVENT.search(phrase) and not EXAM_WORD.search(phrase) and not STUDY_WORD.search(phrase):
        return "once"
    if explicitly_large(phrase):
        return "chunks"
    if STUDY_WORD.search(phrase) and EXAM_WORD.search(phrase):
        return "chunks"
    if EXAM_WORD.search(phrase):
        return "exam"
    if explicitly_small(phrase):
        return "once"
    if re.search(r"\b(papers?|essays?|theses|thesis|capstones?|presentations?|projects?)\b", phrase, re.IGNORECASE):
        return "chunks"
    if (
        re.search(r"\b(assignments?|homework|quiz|quizzes|worksheets?|labs?|psets?|readings?)\b", phrase, re.IGNORECASE)
        or "problem set" in phrase.lower()
    ):
        return "once"
    if re.search(r"\b(two|three|four|several|many)\b", phrase, re.IGNORECASE):
        return "chunks"
    return "once"


def session_count(phrase, stressed):
    """How many work blocks this activity should become."""
    kind = activity_kind(phrase)
    if kind == "once":
        return 1
    count = 3 if stressed else 4
    if kind == "exam":
        return count + 1
    return count


def task_label(phrase):
    text = str(phrase)
    text = re.sub(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", " ", text)
    text = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"\.?\s+\d{1,2}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:a|an|one|two|three|four|\d+)\s+weeks?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+days?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:this|next|the|all|in|a|one)\s+(?:week|month)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    for name in list(WEEKDAYS) + ["today", "tomorrow", "tonight", "due", "deadline"]:
        text = re.sub(rf"\b{name}\b", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text or phrase


def minutes_to_hhmm(minutes):
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


MAX_PLAN_DAYS = 31
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def clock_moment(now=None):
    """Date and time of the student's message."""
    if now is None:
        now = time.localtime()
    return datetime(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def format_plan_day(now, offset):
    """Label one plan day so a later week is not the same Sunday."""
    day = clock_moment(now) + timedelta(days=offset)
    return f"{WEEKDAYS[day.weekday()]}, {day.strftime('%b')} {day.day}"


def cap_span(days):
    return max(0, min(int(days), MAX_PLAN_DAYS))


def days_until_month_end(moment):
    if moment.month == 12:
        first_next = datetime(moment.year + 1, 1, 1)
    else:
        first_next = datetime(moment.year, moment.month + 1, 1)
    last = first_next - timedelta(days=1)
    return (last.date() - moment.date()).days


def _dated_offset(year, month, day, moment, year_was_given):
    try:
        target = datetime(year, month, day)
    except ValueError:
        return None
    if target.date() < moment.date() and not year_was_given:
        try:
            target = datetime(year + 1, month, day)
        except ValueError:
            return None
    delta = (target.date() - moment.date()).days
    if delta < 0:
        return None
    return cap_span(delta)


def point_offset(text, now):
    """Days until one named day. None when the text is a range or has no day."""
    if not text:
        return None
    lowered = str(text).lower()
    moment = clock_moment(now)
    dated = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lowered)
    if dated:
        year_text = dated.group(3)
        year = int(year_text) if year_text else moment.year
        if year < 100:
            year += 2000
        return _dated_offset(year, int(dated.group(1)), int(dated.group(2)), moment, bool(year_text))
    for index, name in enumerate(MONTH_NAMES, start=1):
        match = re.search(rf"\b(?:{name}|{name[:3]})\.?\s+(\d{{1,2}})\b", lowered)
        if match:
            return _dated_offset(moment.year, index, int(match.group(1)), moment, False)
    if re.search(r"\b(today|tonight)\b", lowered):
        return 0
    if re.search(r"\btomorrow\b", lowered):
        return 1
    weekday = weekday_of(text)
    if not weekday:
        return None
    delta = (WEEKDAYS.index(weekday) - moment.weekday()) % 7
    # "next Saturday" on Sunday is the coming Saturday. Add a week only when that day is today.
    if re.search(r"\bnext\b", lowered) and not re.search(r"\bthis\b", lowered) and delta == 0:
        delta += 7
    return cap_span(delta)


def range_offset(text, now):
    """Days until the end of a week or month range. None when no range is named."""
    if not text:
        return None
    lowered = str(text).lower()
    moment = clock_moment(now)
    count = re.search(r"\b(\d+)\s+days?\b", lowered)
    if count:
        return cap_span(int(count.group(1)))
    weeks = re.search(r"\b(a|an|one|two|three|four|\d+)\s+weeks?\b", lowered)
    if weeks:
        word = weeks.group(1)
        number = int(word) if word.isdigit() else NUMBER_WORDS[word]
        return cap_span(number * 7)
    if re.search(r"\b(next month|in a month|a month|one month)\b", lowered):
        return MAX_PLAN_DAYS
    if re.search(r"\bthis month\b", lowered):
        return cap_span(days_until_month_end(moment))
    if re.search(r"\bnext week\b", lowered):
        return cap_span(((6 - moment.weekday()) % 7) + 7)
    if re.search(r"\b(this week|the week|all week)\b", lowered):
        return cap_span((6 - moment.weekday()) % 7)
    return None


def deadline_clauses(notes):
    """Split one answer into the separate deadlines the student named."""
    clauses = []
    for note in notes:
        for part in re.split(r",|\band\b", note, flags=re.IGNORECASE):
            part = " ".join(part.split()).strip(" .")
            if part:
                clauses.append(part)
    return clauses


def clause_for_phrase(phrase, notes):
    """The deadline clause that names this task, not the whole answer."""
    phrase_tokens = significant_tokens(phrase)
    best = None
    best_score = 0
    for clause in deadline_clauses(notes):
        score = len(phrase_tokens & significant_tokens(clause))
        if score > best_score:
            best = clause
            best_score = score
    return best


def mentioned_clock(text):
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", str(text), re.IGNORECASE)
    if not match:
        return None
    return clock_to_minutes(match.group(0))


def clock_limit(text):
    """Return (start at this minute, end by this minute) from words like 'at 9am' or 'by 5pm'."""
    minute = mentioned_clock(text)
    if minute is None:
        return None, None
    if re.search(r"\b(by|before)\b", str(text), re.IGNORECASE):
        return None, minute
    return minute, None


def phrase_horizon(phrase, deadline_notes, now):
    """End offset, whether it is one day, and the text that named the time."""
    clause = clause_for_phrase(phrase, deadline_notes)
    clauses = deadline_clauses(deadline_notes)
    sources = [phrase]
    if clause and clause != phrase:
        sources.insert(0, clause)
    elif not clause and len(clauses) == 1 and clauses[0] != phrase:
        # One deadline for the whole answer still applies when it names no single task.
        sources.insert(0, clauses[0])
    for source in sources:
        point = point_offset(source, now)
        if point is not None:
            return point, True, source
        broad = range_offset(source, now)
        if broad is not None:
            return broad, False, source
    return 0, False, phrase


def blocks_across(span, stressed):
    """Short work blocks spread across the timeline, at most a few each week."""
    if span <= 0:
        return 1 if stressed else 2
    weeks = max(1, (span + 6) // 7)
    per_week = 2 if stressed else 3
    cap = 8 if stressed else 12
    return min(cap, weeks * per_week)


def pick_days(day_offsets, count):
    """Choose that many days across the whole window, including the last day."""
    days = list(day_offsets) or [0]
    count = max(1, min(len(days), int(count)))
    if count >= len(days):
        return days
    if count == 1:
        return [days[0]]
    last = len(days) - 1
    picked = []
    for index in range(count):
        day = days[round(index * last / (count - 1))]
        if not picked or picked[-1] != day:
            picked.append(day)
    return picked or [days[0]]


def pace_loads(day_offsets, per_day, requested_days=None):
    """Several sittings on each chosen day. A long deadline is not one block a day."""
    days = list(day_offsets) or [0]
    per_day = max(2, min(3, int(per_day or 2)))
    if len(days) <= 7:
        chosen = days
    else:
        weeks = max(1, (len(days) + 6) // 7)
        default = min(len(days), weeks * 3)
        low = min(len(days), max(2, weeks * 2))
        high = min(len(days), weeks * 4)
        if requested_days is None:
            count = default
        else:
            try:
                count = int(requested_days)
            except (TypeError, ValueError):
                count = default
            count = max(low, min(high, count))
        chosen = pick_days(days, count)
    return [(day, per_day) for day in chosen]


def sessions_from_loads(loads, label, minutes):
    sessions = []
    for offset, count in loads:
        for _ in range(count):
            sessions.append({"offset": offset, "task": label, "minutes": minutes})
    return sessions


def sessions_per_day_from(guidance):
    """Study sits more than once a day. Three sittings is the full realistic day."""
    if not guidance:
        return 2
    try:
        number = int(guidance.get("sessions_per_day"))
    except (TypeError, ValueError):
        return 2
    return max(2, min(3, number))


def flag_is_true(value, default=True):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "0"}
    return bool(value)


def spread_sessions(day_offsets, count, label, minutes):
    """Place blocks across the whole timeline, including the last day."""
    if not day_offsets:
        day_offsets = [0]
    if count <= 1:
        chosen = [day_offsets[0]]
    elif count >= len(day_offsets):
        chosen = [day_offsets[index % len(day_offsets)] for index in range(count)]
    else:
        last = len(day_offsets) - 1
        chosen = [day_offsets[round(index * last / (count - 1))] for index in range(count)]
    buckets = {offset: [] for offset in day_offsets}
    for offset in chosen:
        buckets[offset].append(label)
    sessions = []
    for offset in day_offsets:
        for task in buckets[offset]:
            sessions.append({"offset": offset, "task": task, "minutes": minutes})
    return sessions


def with_clock(session, text):
    start_at, end_by = clock_limit(text)
    if start_at is not None:
        session["at"] = start_at
    if end_by is not None:
        session["by"] = end_by
    return session


def sessions_for_phrase(phrase, stressed, deadline_notes, today_name, now=None, guidance=None):
    """Turn one phrase into one sitting or several chunks across the named timeline."""
    del today_name  # dates come from the message clock, not the weekday name alone
    label = task_label(phrase)
    end, specific, source = phrase_horizon(phrase, deadline_notes, now)
    kind = activity_kind(phrase)
    chunk_minutes = 25 if stressed else MAX_WORK_MINUTES
    once_minutes = MAX_WORK_MINUTES if ONE_TIME_EVENT.search(phrase) else chunk_minutes
    studying = bool(STUDY_WORD.search(phrase) and EXAM_WORD.search(phrase))
    per_day = sessions_per_day_from(guidance)
    requested_days = None if not guidance else guidance.get("days")

    if kind == "once":
        # A month-long note does not turn shopping into a daily chore.
        offset = end if specific else 0
        return [with_clock({"offset": offset, "task": label, "minutes": once_minutes}, source)]

    if kind == "exam" or studying:
        event = re.sub(r"^study for\s+", "", label, flags=re.IGNORECASE)
        if end <= 0:
            count = per_day if studying else 1
            task = f"Study for {event}" if studying else event
            sessions = sessions_from_loads([(0, count)], task, chunk_minutes)
            if sessions:
                sessions[0] = with_clock(sessions[0], source)
                _, end_by = clock_limit(source)
                if end_by is not None:
                    for session in sessions[1:]:
                        session["by"] = end_by
            return sessions
        study_days = list(range(end)) or [0]
        loads = pace_loads(study_days, per_day, requested_days)
        sessions = sessions_from_loads(loads, f"Study for {event}", chunk_minutes)
        sessions.append(with_clock({"offset": end, "task": event, "minutes": chunk_minutes}, source))
        return sessions

    window = list(range(end + 1)) or [0]
    # vLLM can call a project one sitting. Study still repeats when the deadline has room.
    if not flag_is_true(None if not guidance else guidance.get("split"), default=True):
        offset = end if specific else 0
        return [with_clock({"offset": offset, "task": label, "minutes": chunk_minutes}, source)]
    repeat = flag_is_true(None if not guidance else guidance.get("repeat"), default=len(window) > 1)
    if not repeat:
        loads = [(window[0], per_day)]
    else:
        loads = pace_loads(window, per_day, requested_days)
    return sessions_from_loads(loads, label, chunk_minutes)


def guidance_for(phrase, guidance):
    """The vLLM decision for this task, matched by the student's own words."""
    if not guidance:
        return None
    for item in guidance:
        if not isinstance(item, dict):
            continue
        name = item.get("task") or item.get("name") or ""
        if shares_task(name, phrase):
            return item
    return None


def plan_from_phrases(schedule, stressed, deadline_notes=None, now=None, guidance=None):
    """Build a timed plan from the student's own tasks, starting at the message time."""
    deadline_notes = [note.strip() for note in (deadline_notes or []) if note and note.strip()]
    schedule = split_listed_activities(clean_phrases(schedule))
    phrases = list(task_phrases(schedule))
    work_note = re.compile(
        r"\b(papers?|essays?|projects?|exams?|midterms?|finals?|presentations?|"
        r"quiz|quizzes|reading|homework|lab|pset|assignment|problem set|shop(?:ping)?)\b",
        re.IGNORECASE,
    )
    for note in deadline_notes:
        if work_note.search(note) and not any(shares_task(note, phrase) for phrase in phrases):
            phrases.append(note)
    if not phrases:
        return []

    today_name, now_minutes = current_moment(now)
    day_limit = 2 * 60 if stressed else MAX_WORK_PER_DAY
    days = {}

    def ensure(label, offset):
        if label not in days:
            days[label] = {
                "offset": offset,
                "cursor": earliest_start(offset, now_minutes),
                "work": 0,
                "blocks": [],
            }
        return days[label]

    def place(offset, task, minutes, latest_offset, at_time=None, by_time=None):
        current = offset
        while current <= latest_offset:
            label = format_plan_day(now, current)
            day = ensure(label, current)
            start = day["cursor"]
            if start is None:
                if at_time is None and by_time is None:
                    current += 1
                    continue
                return False
            needs_break = bool(day["blocks"]) and "break" not in day["blocks"][-1]["task"].lower()
            break_len = BREAK_GAP_MINUTES if needs_break else 0
            if at_time is not None and not day["blocks"]:
                start = at_time
                needs_break = False
                break_len = 0
            if by_time is not None and start + break_len + minutes > by_time:
                return False
            if (
                start + break_len + minutes <= DAY_END
                and day["work"] + minutes <= day_limit
            ):
                if needs_break:
                    break_end = start + BREAK_GAP_MINUTES
                    day["blocks"].append({
                        "start": minutes_to_hhmm(start),
                        "end": minutes_to_hhmm(break_end),
                        "task": "Break",
                    })
                    start = break_end
                end = start + minutes
                day["blocks"].append({
                    "start": minutes_to_hhmm(start),
                    "end": minutes_to_hhmm(end),
                    "task": task,
                })
                day["cursor"] = end
                day["work"] += minutes
                return True
            if at_time is not None or by_time is not None:
                return False
            current += 1
        return False

    for phrase in phrases:
        kind = activity_kind(phrase)
        end, _specific, _source = phrase_horizon(phrase, deadline_notes, now)
        for session in sessions_for_phrase(
            phrase, stressed, deadline_notes, today_name, now, guidance=guidance_for(phrase, guidance)
        ):
            at_time = session.get("at")
            by_time = session.get("by")
            if at_time is not None or by_time is not None:
                latest = session["offset"]
            elif kind == "once":
                latest = min(session["offset"] + 2, MAX_PLAN_DAYS)
            elif session["task"].startswith("Study for "):
                latest = max(session["offset"], end - 1)
            else:
                latest = end
            place(
                session["offset"],
                session["task"],
                session["minutes"],
                min(MAX_PLAN_DAYS, max(latest, session["offset"])),
                at_time=at_time,
                by_time=by_time,
            )

    ordered = sorted(days.items(), key=lambda item: item[1]["offset"])
    raw_days = [
        {"day": label, "blocks": bucket["blocks"]}
        for label, bucket in ordered
        if bucket["blocks"]
    ]
    # Durations are already chosen, including a full block for a one-time errand.
    return normalize_plan(raw_days, stressed=False)


DEADLINE_HINT = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|tonight|due|deadline|week|month|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)
TIMELINE_HINT = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|tonight|week|month|\d{1,2}/\d{1,2})\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    re.IGNORECASE,
)


STRESS_HINT = re.compile(
    r"\b(stress(?:ed)?|overwhelm(?:ed)?|anxious|anxiety|behind|too much|"
    r"can't|cannot|panic|burned out|burnt out|exhausted|swamped)\b",
    re.IGNORECASE,
)


def sounds_stressed(schedule):
    return bool(STRESS_HINT.search(" ".join(schedule)))


def phrases_without_deadline(schedule):
    return [phrase for phrase in schedule if not DEADLINE_HINT.search(phrase)]


def collect_deadlines(schedule):
    """Ask for a deadline until the student gives one. Never ask how long a task takes."""
    if not phrases_without_deadline(schedule):
        return []
    print("I'll choose the study times from what you already said.")
    prompt = (
        "When are these due? A deadline is required. You can say "
        "'Friday', 'this week', or 'next month': "
    )
    while True:
        answer = input(prompt).strip()
        if TIMELINE_HINT.search(answer):
            return [answer]
        print("I still need a deadline or a timeline. Pressing enter will not skip this.")


def parse_task_analysis(reply, schedule):
    """Read the short vLLM decision. Ignore a reply that does not name the student's tasks."""
    phrases = task_phrases(clean_phrases(schedule))
    if not phrases:
        return []
    for data in load_json_values(reply or ""):
        tasks = []
        if isinstance(data, dict):
            raw = data.get("tasks") or data.get("analysis")
            if isinstance(raw, list):
                tasks = raw
        elif isinstance(data, list):
            tasks = data
        matched = []
        for item in tasks:
            if not isinstance(item, dict):
                continue
            name = item.get("task") or item.get("name") or ""
            if any(shares_task(name, phrase) for phrase in phrases):
                matched.append(item)
        if matched:
            return matched
    return []


def ask_for_plan(schedule, deadline_notes, stressed, now=None):
    numbered = "\n".join(f"- {phrase}" for phrase in schedule)
    deadlines = "\n".join(f"- {note}" for note in deadline_notes) or "- No extra deadlines were given."
    weekday, now_minutes = current_moment(now)
    clock = minutes_to_clock(now_minutes)
    load = (
        "The student sounds stressed. Keep each sitting to 25 minutes and about 2 hours in a day."
        if stressed
        else "Use 45 minute sittings and at most 4 hours of work in a day."
    )
    # A short decision is enough. The local planner places the clock times, so this
    # call does not spend tokens writing every block.
    request = {
        "model": SCHEDULE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You decide how a UNC student's tasks should be split. "
                    "You do not write clock times. Keep the reasoning to a few sentences.\n"
                    f"The student is messaging you on {weekday} at {clock}.\n"
                    f"{load}\n"
                    f"There are {len(schedule)} tasks. Look at each task and its deadline.\n"
                    "For each task decide:\n"
                    "- repeat: false for one sitting, such as shopping, an appointment, or a small assignment. "
                    "true when the work continues on later days.\n"
                    "- split: true when one day should hold more than one sitting.\n"
                    "- sessions_per_day: 2 or 3. Study should happen more than once a day, "
                    "with a break between sittings. Do not leave study as one 45 minute block.\n"
                    "- days: how many different days to use. Use more than one day when the deadline allows it. "
                    "Do not squeeze a multi-day deadline into one day, and do not spread study so thin "
                    "that a day has only one short block.\n"
                    "Use only the tasks they named. Do not diagnose a mental illness.\n"
                    "Return JSON only:\n"
                    '{"tasks":[{"task":"<student task>","repeat":true,"split":true,"sessions_per_day":2,"days":3}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Right now it is {weekday} {clock}.\n"
                    f"Task count: {len(schedule)}\n\n"
                    f"Tasks:\n{numbered}\n\n"
                    f"Deadlines:\n{deadlines}"
                ),
            },
        ],
        "max_tokens": 800,
        "temperature": 0.2,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    try:
        response = client.chat.completions.create(**request)
    except TypeError:
        request.pop("extra_body")
        response = client.chat.completions.create(**request)
    except Exception as error:
        if getattr(error, "status_code", None) not in (400, 422):
            raise
        request.pop("extra_body", None)
        response = client.chat.completions.create(**request)
    return reply_text(response.choices[0].message)


def release_response(student_text, response_text):
    """Run the drafted schedule through the safety agent before showing it."""
    result = review_conversation([
        {"user": student_text, "assistant": response_text},
    ])
    print_safety_result(result)
    return not result.get("crisis", False)


def get_schedule_needs(schedule):
    """Read schedule[] and print a plan with no overlaps and a lighter load."""
    schedule = clean_phrases(schedule)
    if not schedule:
        print("schedule[] is empty, so there is nothing to plan.")
        return ""

    stressed = sounds_stressed(schedule)
    moment = time.localtime()
    schedule = split_listed_activities(schedule, now=moment)
    print("Schedule intake is reading schedule[]:")
    for phrase in schedule:
        print(f"- {phrase}")

    weekday, now_minutes = current_moment(moment)
    print(f"Planning from {format_plan_day(moment, 0)} at {minutes_to_clock(now_minutes)}.")

    deadline_notes = collect_deadlines(schedule)

    before_started = time.perf_counter()
    local_plan = plan_from_phrases(schedule, stressed, deadline_notes, now=moment)
    before_seconds = time.perf_counter() - before_started

    vllm_started = time.perf_counter()
    reply = ""
    try:
        reply = ask_for_plan(schedule, deadline_notes, stressed, now=moment)
    except Exception as error:
        print(f"The schedule planner could not be reached: {error}")
    vllm_seconds = time.perf_counter() - vllm_started

    plan = local_plan
    if reply:
        guidance = parse_task_analysis(reply, schedule)
        if guidance:
            guided = plan_from_phrases(
                schedule, stressed, deadline_notes, now=moment, guidance=guidance
            )
            if guided:
                plan = guided
    print(f"Runtime without Red Hat vLLM: {before_seconds:.2f}s")
    print(f"Runtime with Red Hat vLLM: {vllm_seconds:.2f}s")
    if not plan:
        print("I don't see an assignment, class, or exam to put on a schedule.")
        return ""
    rendered = render_plan(plan)
    allowed = release_response("\n".join(schedule), rendered)
    if not allowed:
        return rendered

    print("Here is a schedule that leaves breaks and does not stack tasks on top of each other:")
    print(rendered)
    return rendered
