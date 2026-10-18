import json
import re

from openai import OpenAI

from safety_agent import print_safety_result, review_conversation

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
)

SCHEDULE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

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


def message_text(message):
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


def parse_plan(reply, stressed=False, schedule=None):
    """Use a model plan only when it names the student's tasks, not the sample."""
    matched = None
    first = None
    for data in load_json_values(reply):
        plan = normalize_plan(days_from_reply(data), stressed=stressed)
        if not plan:
            continue
        if first is None:
            first = plan
        if schedule is not None and plan_uses_student_tasks(plan, schedule, stressed):
            matched = plan
            break
    if schedule is not None:
        return matched or []
    return first or []


def session_count(phrase, stressed):
    """Short blocks, repeated when the task is a paper, exam, or a pile of work."""
    text = phrase.lower()
    big = re.search(
        r"\b(papers?|essays?|projects?|exams?|midterms?|finals?|presentations?)\b",
        text,
    )
    several = re.search(r"\b(two|three|four|several|many)\b", text)
    small = bool(re.search(r"\b(quiz|quizzes|reading|homework|lab|pset)\b", text))
    small = small or "problem set" in text
    if big or several:
        return 3 if stressed else 4
    if small or stressed:
        return 1
    return 2


def task_label(phrase):
    text = phrase
    for name in list(WEEKDAYS) + ["today", "tomorrow", "tonight", "due", "deadline"]:
        text = re.sub(rf"\b{name}\b", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split(" "))
    text = " ".join(text.split())
    return text or phrase


def shared_deadline(note):
    """A note like 'Friday' or 'all of them Friday' dates every undated task."""
    day = weekday_of(note)
    if not day:
        return None
    if not significant_tokens(note):
        return day
    if re.search(r"\b(all|everything|every|both|them|each)\b", note, re.IGNORECASE):
        return day
    return None


def phrase_day(phrase, deadline_notes):
    own = weekday_of(phrase)
    if own:
        return own
    if re.search(r"\btomorrow\b", phrase, re.IGNORECASE):
        return "Tomorrow"
    shared = [shared_deadline(note) for note in deadline_notes]
    shared = [day for day in shared if day]
    if len(set(shared)) == 1:
        return shared[0]
    for note in deadline_notes:
        day = weekday_of(note)
        if day and shares_task(phrase, note):
            return day
    return "Today"


def spill_target(label):
    if label in WEEKDAYS:
        index = WEEKDAYS.index(label)
        return WEEKDAYS[(index - 1) % 7]
    if label == "Today":
        return "Tomorrow"
    if label == "Tomorrow":
        return "Later"
    return None


def minutes_to_hhmm(minutes):
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def day_sort_key(label):
    if label == "Today":
        return (0, 0)
    if label == "Tomorrow":
        return (0, 1)
    if label in WEEKDAYS:
        return (1, WEEKDAYS.index(label))
    return (2, 0)


def plan_from_phrases(schedule, stressed, deadline_notes=None):
    """Build a timed plan from the student's own tasks."""
    deadline_notes = [note.strip() for note in (deadline_notes or []) if note and note.strip()]
    phrases = list(task_phrases(schedule))
    work_note = re.compile(
        r"\b(papers?|essays?|projects?|exams?|midterms?|finals?|presentations?|"
        r"quiz|quizzes|reading|homework|lab|pset|assignment|problem set)\b",
        re.IGNORECASE,
    )
    for note in deadline_notes:
        if work_note.search(note) and not any(shares_task(note, phrase) for phrase in phrases):
            phrases.append(note)
    if not phrases:
        return []

    work_limit = 25 if stressed else MAX_WORK_MINUTES
    day_limit = 2 * 60 if stressed else MAX_WORK_PER_DAY
    days = {}

    def ensure(label):
        if label not in days:
            days[label] = {"cursor": DAY_START, "work": 0, "blocks": []}
        return days[label]

    def place(label, task):
        current = label
        for _ in range(8):
            if current is None:
                return False
            day = ensure(current)
            start = day["cursor"]
            needs_break = bool(day["blocks"]) and "break" not in day["blocks"][-1]["task"].lower()
            break_len = BREAK_GAP_MINUTES if needs_break else 0
            if (
                start + break_len + work_limit <= DAY_END
                and day["work"] + work_limit <= day_limit
            ):
                if needs_break:
                    break_end = start + BREAK_GAP_MINUTES
                    day["blocks"].append({
                        "start": minutes_to_hhmm(start),
                        "end": minutes_to_hhmm(break_end),
                        "task": "Break",
                    })
                    start = break_end
                end = start + work_limit
                day["blocks"].append({
                    "start": minutes_to_hhmm(start),
                    "end": minutes_to_hhmm(end),
                    "task": task,
                })
                day["cursor"] = end
                day["work"] += work_limit
                return True
            current = spill_target(current)
        return False

    for phrase in phrases:
        label = task_label(phrase)
        day_name = phrase_day(phrase, deadline_notes)
        for _ in range(session_count(phrase, stressed)):
            place(day_name, label)

    ordered = sorted(days.items(), key=lambda item: day_sort_key(item[0]))
    raw_days = [
        {"day": label, "blocks": bucket["blocks"]}
        for label, bucket in ordered
        if bucket["blocks"]
    ]
    return normalize_plan(raw_days, stressed=stressed)


DEADLINE_HINT = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|tonight|due|deadline|\d{1,2}/\d{1,2})\b",
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
    """Ask one optional deadline question. Never ask how long a task takes."""
    if not phrases_without_deadline(schedule):
        return []
    print("I'll choose the study times from what you already said.")
    answer = input(
        "Do any of these have a deadline? You can say something like "
        "'chem exam Friday', or press enter to skip: "
    ).strip()
    if not answer or answer.lower() in {"skip", "none", "no", "n/a"}:
        return []
    return [answer]


def ask_for_plan(schedule, deadline_notes, stressed):
    numbered = "\n".join(f"- {phrase}" for phrase in schedule)
    deadlines = "\n".join(f"- {note}" for note in deadline_notes) or "- No extra deadlines were given."
    load = (
        "The student sounds stressed. Use 25 minute work blocks, extra breaks, "
        "and about 2 hours of work in a day."
        if stressed
        else "The student does not sound especially stressed. Use up to 45 minute blocks "
        "and at most 4 hours of work in a day."
    )
    response = client.chat.completions.create(
        model=SCHEDULE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You plan a realistic schedule for one UNC student. "
                    "You choose every start and end time. "
                    "The student does not tell you how long each assignment takes. "
                    "Do not ask them for a duration or a start time. "
                    "Estimate a realistic duration from the task itself.\n"
                    f"{load}\n"
                    "Rules:\n"
                    "- Use only the tasks they named. Do not invent extra exams or jobs.\n"
                    "- Do not diagnose a mental illness.\n"
                    "- Put a 10 minute break between work blocks.\n"
                    "- Do not overlap any blocks.\n"
                    "- Stay between 9:00 AM and 9:00 PM.\n"
                    "- If a deadline is missing, use Today and spread the work across the day.\n"
                    "- Copy each task from the student's own words. Do not invent a task.\n"
                    "- A paper, exam, or project needs several blocks, not one short block.\n"
                    "Return JSON only, with 24-hour times and no sample task:\n"
                    '{"days":[{"day":"Today","blocks":[{"start":"09:00","end":"09:45","task":"<one student task>"}]}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Student's words:\n{numbered}\n\n"
                    f"Deadlines:\n{deadlines}"
                ),
            },
        ],
        max_tokens=1200,
        temperature=0.2,
    )
    return message_text(response.choices[0].message)


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
    print("Schedule intake is reading schedule[]:")
    for phrase in schedule:
        print(f"- {phrase}")

    if not schedule:
        print("schedule[] is empty, so there is nothing to plan.")
        return ""

    stressed = sounds_stressed(schedule)
    deadline_notes = collect_deadlines(schedule)

    try:
        reply = ask_for_plan(schedule, deadline_notes, stressed)
        plan = parse_plan(reply, stressed=stressed, schedule=schedule)
    except Exception as error:
        print(f"The schedule planner could not be reached: {error}")
        plan = []

    if not plan:
        plan = plan_from_phrases(schedule, stressed, deadline_notes)
    if not plan:
        print("I don't see an assignment, class, or exam to put on a schedule.")
        return ""

    rendered = render_plan(plan)
    if not release_response("\n".join(schedule), rendered):
        return rendered

    print("Here is a schedule that leaves breaks and does not stack tasks on top of each other:")
    print(rendered)
    return rendered
