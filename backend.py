"""
backend.py - the "kitchen door" between the React tracker and our agents.

React sends the student's check-in here. This file runs the agents in order
and sends back organized data (urgent help, tasks with dates, deadlines,
resources) that the Today / Week / Month views can display.

It only CALLS the team's agent files. It does not change them.

Run it:
    source ~/vllm-env/bin/activate
    cd ~/smallModelHack
    uvicorn backend:app --port 8002 --reload

Test it (in another terminal):
    curl -s localhost:8002/api/health
"""
import io
import re
import threading
from contextlib import redirect_stdout
from datetime import date, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# The team's agents (we only call their functions)
from intake_agent import split_request
from safety_agent import GROUP_MESSAGES, LIFE_SAFETY, RESOURCES, check_response, review_conversation
import schedule_intake
from schedule_intake import get_schedule_needs, minutes_to_clock, task_label
from resource_intake import load_catalog, search_resources, select_resources
import intake_agent

# SPEED: the intake and schedule agents run all their Nemotron calls a SECOND time, one by one,
# only to print a "with vs. without vLLM" timing comparison for the demo. The web app doesn't
# show that comparison, so we skip the second run here. Their files are not changed: running
# intake_agent.py or schedule_intake.py directly still prints the comparison.
intake_agent.timed_unassisted = lambda user_text: 0.0

# KEEP THE VOTE: intake throws away its answer when a strict wording check scores it just under 90%,
# even when all 100 votes agreed. We remember the agreed answer so the backend can still use it.
from phrase_check import PASS_RATE
_last_split = threading.local()


def saving_the_vote(original):
    """Wrap intake's vote check so we can see the agreed answer, even when it gets rejected."""
    def wrapper(*args, **kwargs):
        ok, schedule, resource, rate = original(*args, **kwargs)
        _last_split.value = (list(schedule or []), list(resource or []), rate)
        return ok, schedule, resource, rate
    return wrapper


# Works with both versions of intake_agent.py (older: accepted_split, newer: run_gym_servers)
for _name in ("run_gym_servers", "accepted_split"):
    if hasattr(intake_agent, _name):
        setattr(intake_agent, _name, saving_the_vote(getattr(intake_agent, _name)))
schedule_intake.ask_unassisted = lambda *args, **kwargs: None

app = FastAPI(title="Student Support Tracker API")

# Lets the React dev server (another port) call this API while we develop
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------
# What React sends us
# ---------------------------------------------------------------
class Deadline(BaseModel):
    title: str        # "Chem midterm"
    date: str         # "2026-10-23" (from the date picker)


class CheckIn(BaseModel):
    message: str = ""                 # "I have a chem midterm Thursday and I'm stressed"
    topics: list[str] = []            # cards the student tapped: ["Exams", "Food"]
    deadlines: list[Deadline] = []


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Topic cards the student tapped -> which verified resources to show (from unc_resources.json).
# Housing is not here on purpose: the Safety agent shows housing help only for eviction.
TOPIC_GROUPS = {
    "Food": "food",
    "Feeling down": "mental_health",
    "Lonely": "community",
    "Locked out": "lockout_room",
}
TOPIC_INTROS = {
    "food": "Here's where to get food on and near campus:",
    "mental_health": "Feeling down is worth talking about. Here's who you can reach out to:",
    "community": "Finding your people takes time, and you don't have to do it alone. Here's who can help:",
    "lockout_room": "Locked out? Here's how to get back in:",
}


def quietly(func, *args):
    """Run an agent function without its print() output cluttering the server log."""
    with redirect_stdout(io.StringIO()):
        return func(*args)


def label_to_date(label, today, fallback_index):
    """The schedule agent says 'Thursday' or 'Today'. Turn that into a real date."""
    text = (label or "").strip().lower()
    if text in ("today", "tonight"):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    for i, name in enumerate(WEEKDAYS):
        if name in text:
            days_ahead = (i - today.weekday()) % 7
            return today + timedelta(days=days_ahead)
    return today + timedelta(days=fallback_index)  # unknown label: keep the agent's order


# ---- Making the tracker readable: one clear task per subject per day ----
# Words that don't say WHICH class (so "exam on Oct 22" alone has no context)
GENERIC_WORDS = {"study", "studying", "for", "prepare", "prep", "review", "my", "an", "a", "the", "on", "by",
                 "at", "upcoming", "next", "i", "have", "got", "need", "to", "exam", "exams", "test", "tests",
                 "midterm", "midterms", "quiz", "quizzes", "final", "finals"}
# Leftover words at the END of a title after dates are removed ("Study for exam on" -> "Study for exam")
FILLER_END = {"on", "by", "at", "for", "in", "of", "and", "the", "a", "an", "is", "due", "before", "until", "to", "with"}
MAX_MINUTES_PER_TASK = 180
# A "task" made only of these words is just a leftover date, like "October 22 Thursday"
DATE_WORDS = set(WEEKDAYS) | {"january", "february", "march", "april", "may", "june", "july", "august",
                              "september", "october", "november", "december", "jan", "feb", "mar", "apr",
                              "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "today", "tomorrow",
                              "tonight", "due", "on", "by", "the", "this", "next", "week"}


# A phrase that names a PLACE to go (not a task to do) is a resource request: "career center at UNC"
PLACE_WORDS = {"center", "centre", "office", "services", "service", "resource", "resources", "pantry",
               "cupboard", "counseling", "caps", "clinic", "advisor", "advising"}
TASK_WORDS = {"exam", "test", "quiz", "midterm", "final", "assignment", "assn", "hw", "homework", "essay",
              "paper", "project", "lab", "study", "reading", "pset", "presentation", "interview"}
# Words that mean the student wants career help (shows University Career Services)
CAREER_WORDS = {"career", "careers", "interview", "interviews", "resume", "internship", "internships", "job", "jobs"}


def is_resource_request(phrase):
    words = words_of(phrase)
    return bool(words & PLACE_WORDS) and not (words & TASK_WORDS)


def is_junk(title):
    return all(w in DATE_WORDS or w.isdigit() for w in words_of(title))  # never ask for more than 3 hours on one subject in one day


def words_of(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_generic(phrase):
    """True for phrases like 'exam on October 22' that don't name the class."""
    return words_of(task_label(phrase)) <= GENERIC_WORDS


def same_task(phrase, title):
    """Does this phrase from the message talk about this deadline? ('data 120 midterm' vs 'Data 120 midterm')"""
    key = words_of(title) - GENERIC_WORDS or words_of(title) - FILLER_END
    return bool(key & words_of(phrase))


def clean_title(text):
    words = re.sub(r"[,;:.]+", " ", text).split()
    while len(words) > 1 and words[-1].lower() in FILLER_END:
        words.pop()
    title = " ".join(words) or text.strip()
    return title[:1].upper() + title[1:]


def duration_label(minutes):
    minutes = max(15, round(minutes / 15) * 15)  # round to the nearest 15 minutes
    hours, mins = divmod(minutes, 60)
    if hours == 0:
        return f"{mins} min"
    if mins == 30:
        return f"{hours}.5 hrs"
    text = f"{hours} hr" if hours == 1 else f"{hours} hrs"
    return f"{text} {mins} min" if mins else text


def merge_blocks(blocks):
    """The schedule agent splits study time into short sessions with breaks.
    Combine every session of the same subject on the same day into ONE task with its total time."""
    merged = {}
    for b in blocks:
        key = (b["date"], b["title"].lower())
        m = merged.setdefault(key, {"date": b["date"], "title": b["title"], "first": b["start"], "minutes": 0})
        m["first"] = min(m["first"], b["start"])
        m["minutes"] += b["end"] - b["start"]
    tasks = []
    for m in sorted(merged.values(), key=lambda m: (m["date"], m["first"])):
        minutes = min(m["minutes"], MAX_MINUTES_PER_TASK)
        slug = "-".join(sorted(words_of(m["title"])))[:60]
        tasks.append({
            "id": f"{m['date']}-{slug}",
            "date": m["date"],
            "start": to_clock24(m["first"]),
            "minutes": minutes,
            "duration_label": duration_label(minutes),
            "time_label": f"Starts {minutes_to_clock(m['first'])}",
            "title": m["title"],
            "is_break": False,
        })
    return tasks


_schedule_lock = threading.Lock()


def build_schedule(phrases):
    """Run the schedule agent's main function and catch the plan it builds.

    get_schedule_needs() normally prints the plan. Right before printing, it calls
    render_plan(plan). We briefly swap in a copy of render_plan that also saves
    the plan, so we get the organized days/blocks instead of printed text.
    This keeps working even if the schedule agent's inner steps change.
    """
    captured = []
    no_due_date = []
    original_render = schedule_intake.render_plan
    original_collect = schedule_intake.collect_deadlines

    def render_and_save(plan):
        captured.append(plan)
        return original_render(plan)

    def answer_without_asking(schedule):
        # The schedule agent normally TYPES a question in the terminal ("When are these due?")
        # and waits for an answer. Nobody can answer it from the web page, so the request hung
        # forever. Instead we answer "this week" and tell the student on the page.
        missing = schedule_intake.phrases_without_deadline(schedule)
        if not missing:
            return []
        no_due_date.extend(missing if isinstance(missing, (list, tuple)) else [])
        return ["this week"]

    # Give ONLY the tasks with no due date their own "this week". (If we answer the agent's
    # question for everything at once, it squeezes the whole plan into today.)
    missing = list(schedule_intake.phrases_without_deadline(phrases) or [])
    no_due_date.extend(missing)
    phrases = [f"{p} this week" if p in missing else p for p in phrases]

    with _schedule_lock:
        schedule_intake.render_plan = render_and_save
        schedule_intake.collect_deadlines = answer_without_asking
        try:
            quietly(get_schedule_needs, phrases)
        finally:
            schedule_intake.render_plan = original_render
            schedule_intake.collect_deadlines = original_collect
    return (captured[-1] if captured else []), no_due_date


def to_clock24(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/api/health")
def health():
    """Quick check that the backend is running."""
    return {"ok": True}


@app.post("/api/plan")
def plan(check_in: CheckIn):
    today = date.today()
    steps = []   # shown on React's "agents at work" screen
    notes = []   # friendly notes if a step could not finish

    # The student's words, plus any topic cards they tapped
    student_text = check_in.message.strip()
    if check_in.topics:
        student_text += f"\n(Topics I picked: {', '.join(check_in.topics)})"
    deadline_phrases = [
        f"{d.title} due {date.fromisoformat(d.date).strftime('%A %B %d')}" for d in check_in.deadlines
    ]

    # 1. SAFETY first, on the student's own words
    safety = quietly(review_conversation, [{"user": student_text, "assistant": ""}])
    steps.append({"agent": "Safety", "status": "done"})
    urgent = {
        "crisis": safety["crisis"],
        "crisis_types": safety["crisis_types"],
        "groups": [
            {
                "group": g["group"],
                "intro": GROUP_MESSAGES.get(g["group"], {}).get("intro", "Here are some places that can help:"),
                "outro": GROUP_MESSAGES.get(g["group"], {}).get("outro", ""),
                "resources": g["resources"],
            }
            for g in safety["groups"]
        ],
    }
    life_safety = any(t in LIFE_SAFETY for t in safety["crisis_types"])

    # In a life-safety crisis, show only the urgent help. No plan, no extra lists.
    if life_safety:
        steps += [{"agent": a, "status": "skipped"} for a in ("Intake", "Schedule", "Resources")]
        return {"today": today.isoformat(), "urgent": urgent, "tasks": [], "deadlines": [],
                "resources": [], "steps": steps, "notes": notes}

    # 1b. TOPIC CARDS: a tapped card counts as mentioning that need, even with no words
    shown_groups = {g["group"] for g in urgent["groups"]}
    shown_names = {r["name"] for g in urgent["groups"] for r in g["resources"]}
    for topic in check_in.topics:
        group = TOPIC_GROUPS.get(topic)
        if not group or group in shown_groups:
            continue
        if group == "community":  # no separate community list yet: CAPS is the fallback
            items = RESOURCES.get("community") or RESOURCES.get("mental_health", [])[:1]
        else:
            items = RESOURCES.get(group, [])
        items = [r for r in items if r["name"] not in shown_names]
        if items:
            urgent["groups"].append({"group": group, "intro": TOPIC_INTROS[group], "outro": "", "resources": items})
            shown_groups.add(group)
            shown_names.update(r["name"] for r in items)

    # Career help: mock interviews, resumes, internships
    if words_of(student_text) & CAREER_WORDS and RESOURCES.get("career"):
        urgent["groups"].append({
            "group": "career",
            "intro": "For mock interviews, resumes, and internships, University Career Services can help:",
            "outro": "",
            "resources": RESOURCES["career"],
        })

    # 2. INTAKE: split the message into schedule phrases and resource phrases
    try:
        _last_split.value = None
        schedule_phrases, resource_phrases = quietly(split_request, student_text) if student_text else ([], [])
        # Intake returned nothing, but the votes agreed: use the agreed answer instead of nothing
        if not schedule_phrases and not resource_phrases and _last_split.value:
            agreed_schedule, agreed_resource, rate = _last_split.value
            if rate >= PASS_RATE:
                schedule_phrases, resource_phrases = agreed_schedule, agreed_resource
        # Move "career center at UNC"-style phrases from the schedule list to the resource list
        resource_phrases = list(resource_phrases) + [p for p in schedule_phrases if is_resource_request(p)]
        schedule_phrases = [p for p in schedule_phrases if not is_resource_request(p)]
        steps.append({"agent": "Intake", "status": "done"})
    except Exception:
        schedule_phrases, resource_phrases = [], []
        steps.append({"agent": "Intake", "status": "error"})
        notes.append("We couldn't read your message just now. Your deadlines are still on the tracker.")
    # If a deadline was added with the date picker, it replaces the same exam in the message
    # (so it is planned once), and vague ones like "exam on Oct 22" are dropped.
    if check_in.deadlines:
        schedule_phrases = [
            p for p in schedule_phrases
            if not is_generic(p) and not any(same_task(p, d.title) for d in check_in.deadlines)
        ]
    schedule_phrases = list(schedule_phrases) + deadline_phrases

    # 3. SCHEDULE: build timed blocks, then give each one a real date
    tasks = []
    if schedule_phrases:
        try:
            days, no_due_date = build_schedule(schedule_phrases)
            if no_due_date:
                notes.append(
                    "No due date given for: " + ", ".join(task_label(p) for p in no_due_date)
                    + ". We spread it over this week. Add it under Upcoming deadlines to plan around the exact date."
                )
            blocks = []
            for index, day in enumerate(days):
                day_date = label_to_date(day["day"], today, index)
                for block in day["blocks"]:
                    if block["is_break"]:
                        continue  # breaks are folded into each task's total time
                    title = clean_title(block["task"])
                    if is_junk(title):
                        continue  # leftover date, not a real task
                    if check_response(title)[0]:
                        continue  # drop any block with made-up contact info or medical advice
                    blocks.append({"date": day_date.isoformat(), "start": block["start"],
                                   "end": block["end"], "title": title})
            tasks = merge_blocks(blocks)
            # On the day of the exam itself, the block is a short warm-up, not "Data 120 midterm".
            # (Exam day = a deadline from the date picker, or a day after "Study for <same thing>" days.)
            due_on = {(d.date, d.title.lower()) for d in check_in.deadlines}
            studied = {t["title"].lower()[len("study for "):] for t in tasks if t["title"].lower().startswith("study for ")}
            for t in tasks:
                if (t["date"], t["title"].lower()) in due_on or t["title"].lower() in studied:
                    t["title"] = f"Quick review before {t['title']}"
            steps.append({"agent": "Schedule", "status": "done"})
        except Exception:
            steps.append({"agent": "Schedule", "status": "error"})
            notes.append("The planner couldn't build a schedule right now. Try again in a minute.")
    else:
        steps.append({"agent": "Schedule", "status": "skipped"})

    # 4. RESOURCES: match verified UNC resources (skip ones Safety already showed)
    resources = []
    if resource_phrases:
        try:
            catalog = load_catalog()
            matches = search_resources(resource_phrases, catalog)
            chosen = quietly(select_resources, resource_phrases, matches)
            already_shown = {r["name"] for g in urgent["groups"] for r in g["resources"]}
            basic_need = any(t in ("food", "housing") for t in safety["crisis_types"])
            resources = [
                {k: r.get(k, "") for k in ("name", "phone", "link", "details")}
                for r in chosen
                if r["name"] not in already_shown
                # food/housing emergencies show only their own help, not CAPS
                and not (basic_need and r.get("category") in ("mental_health", "crisis"))
            ]
            steps.append({"agent": "Resources", "status": "done"})
        except Exception:
            steps.append({"agent": "Resources", "status": "error"})
            notes.append("We couldn't match resources right now.")
    else:
        steps.append({"agent": "Resources", "status": "skipped"})

    return {
        "today": today.isoformat(),
        "urgent": urgent,
        "tasks": tasks,
        "deadlines": [{"title": d.title, "date": d.date} for d in check_in.deadlines],
        "resources": resources,
        "steps": steps,
        "notes": notes,
    }
