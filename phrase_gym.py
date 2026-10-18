"""Offline agreement gate for intake phrasing.

One hundred servers phrase each student message. The split is good when at
least 90% of them return the same activities, or activities that overlap by
at least 90%. Set PHRASE_PASS_RATE=0.95 to require 95%, and PHRASE_SERVERS
to change the server count.
Run: python phrase_gym.py
"""

from phrase_check import PASS_RATE, SERVER_COUNT, accepted_split, propose_split

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


def votes_for(message, count=None):
    """Same split from every server. A stable phrasing agrees with itself."""
    split = propose_split(message)
    return [split for _ in range(count or SERVER_COUNT)]


def main():
    print(f"Servers: {SERVER_COUNT}")
    print(f"Required agreement: {PASS_RATE:.0%}")
    for message in CASES:
        ok, _schedule, _resource, rate = accepted_split(message, votes_for(message))
        print(f"{rate:.0%}  {message}")
        if not ok:
            print("The servers did not agree on an accurate split.")
            return 1

    base = propose_split(CASES[0])
    other = propose_split(CASES[1])
    agree_count = int(SERVER_COUNT * 0.91)
    mostly = [base] * agree_count + [other] * (SERVER_COUNT - agree_count)
    ok, _schedule, _resource, rate = accepted_split(CASES[0], mostly)
    print(f"{rate:.0%}  91% of servers matched")
    if not ok:
        print("A match above the bar was rejected.")
        return 1

    disagree_count = int(SERVER_COUNT * 0.80)
    mixed = [base] * disagree_count + [other] * (SERVER_COUNT - disagree_count)
    ok, _schedule, _resource, rate = accepted_split(CASES[0], mixed)
    print(f"{rate:.0%}  80% of servers matched")
    if ok:
        print("Disagreement was accepted.")
        return 1

    # "study for the midterm" and "study for midterm" are the same activity.
    message = "I need to study for midterm"
    exact_count = max(1, int(SERVER_COUNT * 0.70))
    near = (
        [(["study for midterm"], [])] * exact_count
        + [(["study for the midterm"], [])] * (SERVER_COUNT - exact_count)
    )
    ok, schedule, _resource, rate = accepted_split(message, near)
    print(f"{rate:.0%}  similar wording")
    if not ok or schedule != ["study for midterm"]:
        print("Similar phrasing was treated as disagreement.")
        return 1

    invented = [(["buy a textbook"], []) for _ in range(SERVER_COUNT)]
    ok, _schedule, _resource, rate = accepted_split(CASES[0], invented)
    print(f"{rate:.0%}  agreed on an invented task")
    if ok:
        print("The servers agreed, but the phrases were not accurate.")
        return 1

    print("Phrase check passed the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
