from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Source:
    id: str
    beat: str
    kind: str
    url: str
    status: str
    weight: float
    n_scored: int = 0
    residual_sum: float = 0.0
    residual_sq_sum: float = 0.0
    fail_streak: int = 0
    domain: str = ""


@dataclass
class Claim:
    id: str
    beat: str
    source_id: str
    ts: str
    headline: str
    url: str
    shock_key: str
    direction: str
    points_low: float
    points_high: float
    midpoint: float
    confidence: float
    nifty_level: float | None
    score_after: str
    status: str
    actual_move: float | None = None
    residual: float | None = None
    body: str = ""
    ensemble_id: str | None = None
    horizon: str = "session"
    driver: float | None = None


@dataclass
class Event:
    id: int
    ts: str
    sentence: str
    impact: float
    issue: bool = False
    agent: str = ""


@dataclass
class Ensemble:
    id: str
    ts: str
    mu: float
    sigma: float
    n_shocks: int
    nifty_level: float | None
    actual: float | None = None
    residual: float | None = None
    horizon: str = "session"


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)
