from __future__ import annotations

import math
from collections import defaultdict

from sp500team.models import Claim, Source
from sp500team.scales import AGREE, SCORE_FLOOR, SIGMA_CAP, SIGMA_FLOOR_MIN


def claim_weight(claim: Claim, source: Source, shrink_until: int = 20) -> float:
    if source.status in {"retired", "down", "proposal"} or source.weight <= 0:
        return 0.0
    shrink = min(1.0, (source.n_scored + 2) / (shrink_until + 2))
    weight = source.weight * shrink * claim.confidence
    if source.status == "probation":
        weight = min(weight, 0.3)
    if source.kind == "pcr" and source.n_scored < shrink_until:
        weight *= 0.15
    return weight


def build_ensemble(claims: list[Claim], sources: dict[str, Source], sigma_floor: float) -> dict | None:
    grouped: dict[str, list[tuple[Claim, float]]] = defaultdict(list)
    for claim in claims:
        source = sources.get(claim.source_id)
        if source is None:
            continue
        weight = claim_weight(claim, source)
        if weight <= 0:
            continue
        grouped[claim.shock_key].append((claim, weight))
    if not grouped:
        return None

    shocks = []
    for key, rows in grouped.items():
        total = sum(weight for _, weight in rows)
        midpoint = sum(claim.midpoint * weight for claim, weight in rows) / total
        band = max(claim.points_high - claim.points_low for claim, _ in rows) / 2
        shocks.append({"key": key, "x": midpoint, "w": total, "band": band, "ids": [c.id for c, _ in rows]})

    agreed = _agreed_move(shocks)
    if agreed is not None:
        mu, sigma = agreed
    else:
        weight_sum = sum(shock["w"] for shock in shocks)
        mu = sum(shock["x"] * shock["w"] for shock in shocks) / weight_sum
        var = sum(shock["w"] * (shock["x"] - mu) ** 2 for shock in shocks) / weight_sum
        sigma = math.sqrt(var)
        if len(shocks) == 1:
            sigma = max(sigma, shocks[0]["band"])
    sigma = max(sigma, sigma_floor)
    return {"mu": mu, "sigma": sigma, "n_shocks": len(shocks), "ids": [i for shock in shocks for i in shock["ids"]]}


def _agreed_move(shocks: list[dict]) -> tuple[float, float] | None:
    chosen: list[dict] | None = None
    for sign in (1, -1):
        group = [shock for shock in shocks if shock["x"] * sign > 0 and abs(shock["x"]) >= AGREE]
        if len(group) < 3:
            continue
        if chosen is None or sum(shock["w"] for shock in group) > sum(shock["w"] for shock in chosen):
            chosen = group
    if chosen is None:
        return None
    weight_sum = sum(shock["w"] for shock in chosen)
    mu = sum(shock["x"] * shock["w"] for shock in chosen) / weight_sum
    return mu, abs(mu) * 0.6


def attributed_points(rows: list[tuple[float, float]], mu: float, actual: float) -> list[float]:
    total = sum(weight for _mid, weight in rows)
    if total <= 0:
        return [mid for mid, _weight in rows]
    error = actual - mu
    return [mid + error * (weight / total) for mid, weight in rows]


def learned_midpoint(prior: float, n: int, points_sum: float) -> float:
    if n <= 0:
        return prior
    shrink = n / (n + 4)
    learned = points_sum / n
    return (1 - shrink) * prior + shrink * learned


def adjust_sigma(sigma: float, floor: float, records: list[tuple[int, int]]) -> float:
    if not records:
        return sigma
    if any(outside > inside for inside, outside in records):
        return min(sigma * 1.15, SIGMA_CAP)
    if all(inside >= 3 and inside > outside for inside, outside in records):
        return max(sigma * 0.9, floor)
    return sigma


def apply_score(source: Source, residual: float, book_std: float) -> str | None:
    """Update a source from one residual. Return a log sentence when its status changes."""
    source.n_scored += 1
    source.residual_sum += residual
    source.residual_sq_sum += residual * residual
    mean = source.residual_sum / source.n_scored
    typical = max(book_std, SCORE_FLOOR)
    if abs(residual) <= typical:
        source.weight = min(2.0, source.weight * 1.05)
    else:
        source.weight = max(0.05, source.weight * 0.9)
    if source.n_scored >= 30 and book_std > 0 and abs(mean) > 1.5 * book_std:
        source.status = "retired"
        source.weight = 0
        return f"{source.id} retired. Its predictions stayed to one side."
    return None


def next_sigma_floor(floor: float, covered_flags: list[bool]) -> float:
    if len(covered_flags) < 30:
        return floor
    coverage = sum(1 for flag in covered_flags if flag) / len(covered_flags)
    if coverage < 0.68:
        return min(floor * 1.1, SIGMA_CAP)
    if coverage > 0.68:
        return max(floor * 0.9, SIGMA_FLOOR_MIN)
    return floor


def book_std(residuals: list[float]) -> float:
    if len(residuals) < 2:
        return 0.0
    mean = sum(residuals) / len(residuals)
    var = sum((item - mean) ** 2 for item in residuals) / len(residuals)
    return math.sqrt(var)
