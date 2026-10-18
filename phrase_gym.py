"""Offline pass-rate gate for intake phrasing.

Same loop NeMo Gym uses for one environment: each student message is a task,
verify_split is the reward, and the splitter is kept only when pass@1 reaches
the bar (90% by default, 95% with PHRASE_PASS_RATE=0.95).
Run: python phrase_gym.py
"""

from phrase_check import PASS_RATE, pass_rate, verify_split

# Messages the rule splitter must phrase accurately.
CASES = [
    "I need to schedule my appointment, study for midterm, and sign housings",
    "I have three papers due Friday and I need someone to talk to",
    "Where can I find counseling on campus?",
    "Can you help me plan study time around my labs?",
    "thanks",
    "homework 1, homework 2, homework 3, homework 4, homework 5, "
    "homework 6, homework 7, homework 8, homework 9, homework 10, and study for midterm",
]


def main():
    rate = pass_rate(CASES)
    print(f"Phrase pass rate: {rate:.0%}")
    print(f"Required pass rate: {PASS_RATE:.0%}")
    if rate < PASS_RATE:
        print("The phrase splitter is below the pass rate, so it should not be used.")
        return 1
    # A dropped task and an invented task must not clear the bar.
    dropped = verify_split(
        CASES[0],
        ["schedule my appointment"],
        [],
    )
    invented = verify_split(
        CASES[0],
        ["schedule my appointment", "study for midterm", "sign housings", "buy a textbook"],
        [],
    )
    if dropped >= PASS_RATE or invented >= PASS_RATE:
        print("The verifier accepted an inaccurate split.")
        return 1
    print("Phrase check passed the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
