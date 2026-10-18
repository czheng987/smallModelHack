"""Ask NeMo Gym to run one server per phrase vote.

The model is not called 100 times. Each vote is one NeMo Gym resources server.
Run: python phrase_gym.py
"""

from phrase_check import PASS_RATE, SERVER_COUNT, passes, propose_split, verify_split
from resources_servers.intake_phrases.app import run_gym_servers, score_votes

# Messages the servers must phrase the same way.
CASES = [
    "I need to schedule my appointment, study for midterm, and sign housings",
    "I have three papers due Friday and I need someone to talk to",
    "Where can I find counseling on campus?",
    "Can you help me plan study time around my labs?",
    "thanks",
    "homework 1, homework 2, homework 3, homework 4, homework 5, "
    "homework 6, homework 7, homework 8, homework 9, homework 10, and study for midterm",
]


EXPECTED = {
    CASES[0]: (
        ["schedule my appointment", "study for midterm", "sign housings"],
        [],
    ),
    CASES[1]: (["three papers due Friday"], ["someone to talk to"]),
    CASES[2]: ([], ["counseling on campus"]),
    CASES[3]: (["plan study time around my labs"], []),
    CASES[4]: ([], []),
    CASES[5]: (
        [f"homework {number}" for number in range(1, 11)] + ["study for midterm"],
        [],
    ),
}


def main():
    print(f"NeMo Gym servers: {SERVER_COUNT}")
    print(f"Required pass@1: {PASS_RATE:.0%}")
    for message in CASES:
        schedule, resource = propose_split(message)
        ok, got_schedule, got_resource, rate = run_gym_servers(
            message, schedule, resource, SERVER_COUNT
        )
        print(f"{rate:.0%}  {message}")
        want_schedule, want_resource = EXPECTED[message]
        if not ok or got_schedule != want_schedule or got_resource != want_resource:
            print("NeMo Gym did not accept this split.")
            print(got_schedule, got_resource)
            return 1

    base = propose_split(CASES[0])
    other = propose_split(CASES[1])
    agree_count = int(SERVER_COUNT * 0.91)
    mostly = [base] * agree_count + [other] * (SERVER_COUNT - agree_count)
    ok, _schedule, _resource, rate = score_votes(CASES[0], mostly)
    print(f"{rate:.0%}  91% of servers matched")
    if not ok:
        print("A match above the bar was rejected.")
        return 1

    disagree_count = int(SERVER_COUNT * 0.80)
    mixed = [base] * disagree_count + [other] * (SERVER_COUNT - disagree_count)
    ok, _schedule, _resource, rate = score_votes(CASES[0], mixed)
    print(f"{rate:.0%}  80% of servers matched")
    if ok:
        print("Disagreement was accepted.")
        return 1

    message = "I need to study for midterm"
    exact_count = max(1, int(SERVER_COUNT * 0.70))
    near = (
        [(["study for midterm"], [])] * exact_count
        + [(["study for the midterm"], [])] * (SERVER_COUNT - exact_count)
    )
    ok, schedule, _resource, rate = score_votes(message, near)
    print(f"{rate:.0%}  similar wording")
    if not ok or schedule != ["study for midterm"]:
        print("Similar phrasing was treated as disagreement.")
        return 1

    invented = [(["buy a textbook"], []) for _ in range(SERVER_COUNT)]
    ok, _schedule, _resource, rate = score_votes(CASES[0], invented)
    print(f"{rate:.0%}  agreed on an invented task")
    if ok:
        print("The servers agreed, but the phrases were not accurate.")
        return 1

    bare = [(["study for midterm", "Study on"], []) for _ in range(SERVER_COUNT)]
    ok, schedule, _resource, rate = score_votes(
        "I need to study for midterm and study on",
        bare,
    )
    print(f"{rate:.0%}  dropped a verb with no noun")
    if not ok or schedule != ["study for midterm"]:
        print("A schedule phrase without a noun was kept.")
        return 1

    only_verb = [(["Study on"], []) for _ in range(SERVER_COUNT)]
    ok, schedule, _resource, rate = score_votes("I need to study for midterm", only_verb)
    print(f'{rate:.0%}  only "Study on"')
    if ok or schedule:
        print("A verb with no noun passed the phrase check.")
        return 1

    bare_need = [([], ["talk"]) for _ in range(SERVER_COUNT)]
    ok, _schedule, resource, rate = score_votes("I need someone to talk to", bare_need)
    print(f"{rate:.0%}  resource verb without the need")
    if ok or resource:
        print("A resource phrase without the need was kept.")
        return 1

    named_need = [([], ["someone to talk to"]) for _ in range(SERVER_COUNT)]
    ok, _schedule, resource, rate = score_votes("I need someone to talk to", named_need)
    print(f"{rate:.0%}  resource names the need")
    if not ok or resource != ["someone to talk to"]:
        print("A resource phrase that names the need was rejected.")
        return 1

    placed = [
        (
            "I need to study for midterm and I need someone to talk to",
            [(["someone to talk to", "study for midterm"], [])],
            ["study for midterm"],
            ["someone to talk to"],
            "resource need moved out of schedule[]",
        ),
        (
            "I need to study for midterm and I need someone to talk to",
            [(["someone to talk to"], ["study for midterm"])],
            ["study for midterm"],
            ["someone to talk to"],
            "swapped arrays put back",
        ),
        (
            "I need to study for midterm",
            [([], ["study for midterm"])],
            ["study for midterm"],
            [],
            "schedule task moved out of resource[]",
        ),
        (
            "I need to talk with someone",
            [(["talk with someone"], [])],
            [],
            ["talk with someone"],
            "talk with someone",
        ),
        (
            "I need support",
            [(["support"], [])],
            [],
            ["support"],
            "support",
        ),
        (
            "Where can I find the food pantry?",
            [(["food pantry"], [])],
            [],
            ["food pantry"],
            "food pantry",
        ),
        (
            "I need housing resources",
            [(["housing resources"], [])],
            [],
            ["housing resources"],
            "housing resources",
        ),
        (
            "I am locked out of my dorm",
            [(["locked out of my dorm"], [])],
            [],
            ["locked out of my dorm"],
            "locked out of my dorm",
        ),
        (
            "I need to sign housings and I need housing resources",
            [(["sign housings", "housing resources"], [])],
            ["sign housings"],
            ["housing resources"],
            "sign housings stays on the schedule",
        ),
        (
            "I have a paper due Friday and I am locked out of my dorm",
            [(["paper due Friday", "locked out of my dorm"], [])],
            ["paper due Friday"],
            ["locked out of my dorm"],
            "lockout moved out of schedule[]",
        ),
    ]
    for message, votes, want_schedule, want_resource, label in placed:
        ok, schedule, resource, rate = score_votes(message, votes * SERVER_COUNT)
        print(f"{rate:.0%}  {label}")
        if not ok or schedule != want_schedule or resource != want_resource:
            print(schedule, resource)
            print("Phrase landed in the wrong array.")
            return 1

    raw = verify_split(
        "I need to study for midterm and I need someone to talk to",
        ["someone to talk to", "study for midterm"],
        [],
    )
    if passes(raw):
        print("A resource need left in schedule[] passed the checker.")
        return 1

    print("NeMo Gym phrase check passed the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
