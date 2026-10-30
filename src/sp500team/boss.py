from __future__ import annotations

from sp500team.config import MISSED_LINES, ROSTER, STATUS_LINES

ORDER = [name for _key, name in ROSTER] + ["S&P 500", "ES"]
NAMES = {key: name for key, name in ROSTER}


def wake(store) -> list[str]:
    """Put a down feed back on the clock. A retired source stays retired."""
    found: list[str] = []
    for source in store.sources():
        if source.status != "down":
            continue
        source.status = "active"
        source.fail_streak = 0
        if source.weight <= 0:
            source.weight = 0.4
        store.save_source(source)
        name = NAMES.get(source.beat, source.id)
        if name not in found:
            found.append(name)
    return [name for name in ORDER if name in found]


def boss_sentence(restarted: list[str], still_stuck: list[str]) -> str:
    acted = [name for name in ORDER if name in restarted or name in still_stuck]
    stuck = [name for name in ORDER if name in still_stuck]
    if not acted:
        return "Watching. Every agent is working."
    if not stuck:
        return f"Restarted {_and(acted)}. Working again."
    if stuck == acted:
        return f"Restarted {_and(acted)}. Still stuck."
    return f"Restarted {_and(acted)}. {_and(stuck)} still stuck."


def missed(sentence: str) -> bool:
    return sentence in MISSED_LINES


def boss_stuck(sentence: str) -> bool:
    text = (sentence or "").lower()
    return "still stuck" in text or sentence in {"", "No check"}


def agree_to_rescue(sentence: str, outcomes: dict[str, str]) -> list[str]:
    """Agents with a real report who see a stuck boss. Quiet and market-shut only watch."""
    if not boss_stuck(sentence):
        return []
    voters = []
    for _key, name in ROSTER:
        line = outcomes.get(name, "No check")
        if line in STATUS_LINES or line == "No check":
            continue
        voters.append(name)
    return voters


def watch_line(sentence: str, voters: list[str]) -> str:
    if len(voters) != 1:
        return sentence
    return f"{sentence} {voters[0]} is watching the boss."


def crew_sentence(voters: list[str], still_stuck: list[str]) -> str:
    who = _and(voters)
    if still_stuck:
        return f"Restarted by {who}. Still stuck."
    return f"Restarted by {who}. Working again."


def _and(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"
