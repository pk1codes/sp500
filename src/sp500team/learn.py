from __future__ import annotations

from sp500team.config import BOARD_AGENT, ROSTER
from sp500team.models import Claim, Ensemble
from sp500team.scales import FAMILY_CAP


PROVED_CLOSES = 4


def learning_score(misses: list[float]) -> int | None:
    if len(misses) < 4:
        return None
    early, late = _miss_halves(misses)
    if early + late == 0:
        return 50
    return round(100 * early / (early + late))


def learning_proof(misses: list[float], closes: list[int] | None = None) -> dict | None:
    """The score, the two misses behind it, and whether enough closes back the word."""
    score = learning_score(misses)
    if score is None:
        return None
    early, late = _miss_halves(misses)
    distinct = len(set(closes or []))
    return {
        "score": score,
        "early": early,
        "late": late,
        "closes": distinct,
        "proved": distinct >= PROVED_CLOSES,
    }


def _miss_halves(misses: list[float]) -> tuple[float, float]:
    half = len(misses) // 2
    early_rows = misses[:half]
    late_rows = misses[half:]
    early = sum(abs(item) for item in early_rows) / len(early_rows)
    late = sum(abs(item) for item in late_rows) / len(late_rows)
    return early, late


def shift_trail(misses: list[float], keep: int = 8) -> list[float]:
    """Running copy of the points the next call will add."""
    values = []
    for end in range(1, len(misses) + 1):
        shift = call_shift(misses[:end])
        if shift is not None:
            values.append(shift)
    return values[-keep:]


def call_shift(misses: list[float]) -> float | None:
    if not misses:
        return None
    count = len(misses)
    bias = (sum(misses) / count) * (count / (count + 4))
    return max(-FAMILY_CAP, min(FAMILY_CAP, bias))


def learning_trail(misses: list[float], keep: int = 8) -> list[int]:
    scores = []
    for end in range(4, len(misses) + 1):
        score = learning_score(misses[:end])
        if score is not None:
            scores.append(score)
    return scores[-keep:]


def agent_misses(claims: list[Claim], ensembles: list[Ensemble]) -> dict[str, list[float]]:
    grouped = {key: [] for key, _name in ROSTER}
    grouped[BOARD_AGENT] = []
    for claim in sorted(claims, key=lambda row: (row.ts, row.id)):
        if claim.residual is None or claim.horizon == "week":
            continue
        if claim.source_id == "scout":
            grouped["scout"].append(claim.residual)
        elif claim.beat in grouped and claim.beat != "scout":
            grouped[claim.beat].append(claim.residual)
    for row in sorted(ensembles, key=lambda item: item.ts):
        if row.residual is not None:
            grouped[BOARD_AGENT].append(row.residual)
    return grouped
