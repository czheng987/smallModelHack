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
    content = message.content or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


def clock_to_minutes(value):
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value).strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def minutes_to_clock(minutes):
    minutes = minutes % (24 * 60)
    hour, minute = divmod(minutes, 60)
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def normalize_plan(raw_days):
    """Drop overlapping blocks, cap long work, and leave room for breaks."""
    if not isinstance(raw_days, list):
        return []
    plan = []
    for day in raw_days:
        if not isinstance(day, dict):
            continue
        label = str(day.get("day") or "Today").strip() or "Today"
        blocks = []
        for block in day.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            start = clock_to_minutes(block.get("start"))
            end = clock_to_minutes(block.get("end"))
            task = " ".join(str(block.get("task") or "").split())
            if start is None or not task:
                continue
            if end is None or end <= start:
                end = start + 25
            is_break = "break" in task.lower()
            if not is_break and end - start > MAX_WORK_MINUTES:
                end = start + MAX_WORK_MINUTES
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
            if not block["is_break"] and work_minutes + duration > MAX_WORK_PER_DAY:
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


def parse_plan(reply):
    text = reply or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    return normalize_plan(data.get("days"))


DEADLINE_HINT = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|tonight|due|deadline|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)


def parse_missing_deadlines(reply, schedule):
    """Keep only deadline questions that refer to a task the student already named."""
    text = reply or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    missing = data.get("missing") if isinstance(data, dict) else None
    if not isinstance(missing, list):
        return []
    kept = []
    for item in missing:
        if not isinstance(item, str):
            continue
        name = " ".join(item.split())
        if not name:
            continue
        lowered = name.lower()
        if any(lowered in phrase.lower() or phrase.lower() in lowered for phrase in schedule):
            kept.append(name)
    return kept


def phrases_without_deadline(schedule):
    return [phrase for phrase in schedule if not DEADLINE_HINT.search(phrase)]


def find_missing_deadlines(schedule):
    """Ask which tasks have no day or date. Durations are never requested."""
    numbered = "\n".join(f"- {phrase}" for phrase in schedule)
    try:
        response = client.chat.completions.create(
            model=SCHEDULE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide which tasks have no deadline. "
                        "A deadline is a weekday, a date, or the words today, tomorrow, tonight, or due. "
                        "Do not ask how long a task takes. "
                        'Return JSON only: {"missing": ["task from the list"]}. '
                        "Use an empty list when every task already has a deadline."
                    ),
                },
                {"role": "user", "content": f"schedule[]:\n{numbered}"},
            ],
            max_tokens=300,
            temperature=0,
        )
        missing = parse_missing_deadlines(message_text(response.choices[0].message), schedule)
    except Exception:
        missing = phrases_without_deadline(schedule)
    return missing


def collect_deadlines(missing):
    """Ask only for due dates. The student does not estimate how long work will take."""
    if not missing:
        return []
    print("I will choose the study times. I only need the deadlines.")
    notes = []
    for task in missing:
        answer = input(
            f'When is "{task}" due? (type skip if there is no deadline): '
        ).strip()
        if answer and answer.lower() not in {"skip", "none", "no", "n/a"}:
            notes.append(f"{task} is due {answer}")
    return notes


def ask_for_plan(schedule, deadline_notes):
    numbered = "\n".join(f"- {phrase}" for phrase in schedule)
    deadlines = "\n".join(f"- {note}" for note in deadline_notes) or "- No extra deadlines were given."
    response = client.chat.completions.create(
        model=SCHEDULE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You plan a realistic schedule for one UNC student. "
                    "You choose every start and end time. "
                    "The student does not say how long each assignment takes, and you must not ask them to. "
                    "Estimate a realistic duration from the task itself.\n"
                    "Read the student's words for stress. If they sound overwhelmed, anxious, or overloaded, "
                    "use about 25 minute work blocks, add extra breaks, and spread the work across more days. "
                    "If they sound calm, 45 minute blocks are fine.\n"
                    "Rules:\n"
                    "- Use only the tasks they named. Do not invent extra exams or jobs.\n"
                    "- Do not diagnose a mental illness.\n"
                    "- Put a 10 minute break between work blocks.\n"
                    "- Do not overlap any blocks.\n"
                    "- Keep work to 4 hours or less in one day. If they sound stressed, keep it closer to 2 hours.\n"
                    "- Stay between 9:00 AM and 9:00 PM.\n"
                    "Return JSON only:\n"
                    '{"days": [{"day": "Thursday", "blocks": [{"start": "16:00", "end": "16:25", "task": "Outline the paper"}]}]}'
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

    deadline_notes = collect_deadlines(find_missing_deadlines(schedule))

    try:
        reply = ask_for_plan(schedule, deadline_notes)
        plan = parse_plan(reply)
    except Exception as error:
        print(f"The schedule planner could not be reached: {error}")
        return ""

    if not plan:
        print("I could not build a schedule from those tasks.")
        return ""

    rendered = render_plan(plan)
    if not release_response("\n".join(schedule), rendered):
        return rendered

    print("Here is a schedule that leaves breaks and does not stack tasks on top of each other:")
    print(rendered)
    return rendered
