"""Next-session S&P 500 forecast that refits after every close.

A call on day T uses only bars that had already settled by the New York close.
The next close is the target, never an input. Old sessions fade, so a new
regime can take the weights over. The printed center stays on unchanged
until the fit has beaten a flat guess. The width is the recent miss.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta

import httpx

from sp500team.ahead import _solve
from sp500team.config import ET
from sp500team.scales import (
    CAP,
    EVOLVE_SIGMA_FLOOR,
    LEAN,
    LIVE_GAP,
    PRIOR,
    QUIETER,
    RESID_PRIOR,
)

FORGET = 0.985
RIDGE = 25.0
WARM = 60
SKILL_LEAST = 40
SKILL_WINDOW = 60
PROOF_WINDOW = 80
LIVE_COUNT = 60
BAND_SAMPLE = 40
CORE = ("spx1", "spx5", "europe", "dxy", "brent", "vix")
LABELS = {
    "spx1": "Last session",
    "spx5": "Five sessions",
    "europe": "Europe",
    "dxy": "Dollar",
    "brent": "Brent",
    "vix": "VIX",
}
YAHOO = {
    "spx": "^GSPC",
    "europe": "^STOXX50E",
    "dxy": "DX-Y.NYB",
    "brent": "BZ=F",
    "vix": "^VIX",
}
SAME_DAY = {"spx", "vix", "brent", "europe"}


def skill_ratio(model: float, flat: float) -> float:
    """Share of the flat-guess miss the fit actually removed. Zero unless it wins."""
    if flat <= 1 or model > flat - 1:
        return 0.0
    return max(0.0, min(1.0, (flat - model) / flat))


def publish(raw: float, skill: float) -> float:
    return raw * max(0.0, min(1.0, skill))


def up_chance(mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((mu / sigma) / math.sqrt(2.0)))


def band_mass() -> float:
    """Share of a normal bell inside one sigma of its center."""
    return math.erf(1.0 / math.sqrt(2.0))


def coverage_scale(pairs: list[tuple[float, float]]) -> float:
    """Multiplier, from past misses only, that puts about 68% of them inside the band."""
    usable = [(abs(err), sigma) for err, sigma in pairs if sigma > 0]
    if len(usable) < BAND_SAMPLE:
        return 1.0
    target = band_mass()
    best_scale = 1.0
    best_gap = 1e9
    for step in range(50, 201):
        scale = step / 100
        held = sum(1 for err, sigma in usable if err <= scale * sigma) / len(usable)
        gap = abs(held - target)
        if gap < best_gap or (abs(gap - best_gap) <= 1e-12 and abs(scale - 1.0) < abs(best_scale - 1.0)):
            best_gap = gap
            best_scale = scale
    return best_scale


def live_ready(rows: list[dict]) -> bool:
    """True only after 60 live closes beat a flat guess by the frozen live gap."""
    if len(rows) < LIVE_COUNT:
        return False
    model = sum(abs(row["actual"] - row["mu"]) for row in rows) / len(rows)
    flat = sum(abs(row["actual"]) for row in rows) / len(rows)
    return model <= flat - LIVE_GAP


def gap_phrase(anchor_day: str, due: str) -> str:
    span = (date.fromisoformat(due) - date.fromisoformat(anchor_day)).days
    if span == 1:
        return "about 24 hours after the last close"
    return "the next cash close, after the break"


def next_session(day: str) -> str:
    found = date.fromisoformat(day) + timedelta(days=1)
    while found.weekday() >= 5:
        found += timedelta(days=1)
    return found.isoformat()


def same_day_names() -> set[str]:
    """Series whose daily bar has settled by the 16:00 New York close."""
    from sp500team.scales import DXY_SAME_DAY

    names = set(SAME_DAY)
    if DXY_SAME_DAY:
        names.add("dxy")
    return names


def feature_row(books: dict[str, list[tuple[str, float]]], day: str) -> dict[str, float] | None:
    """Drivers known at the New York close of `day`."""
    spx = [(bar_day, close) for bar_day, close in books.get("spx") or [] if bar_day <= day]
    if len(spx) < 6 or spx[-1][0] != day:
        return None
    values = {
        "spx1": spx[-1][1] - spx[-2][1],
        "spx5": spx[-1][1] - spx[-6][1],
    }
    known = same_day_names()
    for name in ("europe", "dxy", "brent", "vix"):
        change = _change(books.get(name) or [], day, strict=name not in known)
        if change is None:
            return None
        values[name] = change
    return values


def clip_books(books: dict[str, list[tuple[str, float]]], stamp: datetime) -> dict[str, list[tuple[str, float]]]:
    """Drop a session that has not closed, and any same-day bar that settles later."""
    day = stamp.date().isoformat()
    clock = stamp.timetz().replace(tzinfo=None)
    closed = stamp.weekday() < 5 and clock >= time(16, 0)
    known = same_day_names()
    clipped: dict[str, list[tuple[str, float]]] = {}
    for name, rows in books.items():
        kept = []
        for bar_day, close in _clean(rows):
            if bar_day < day or (bar_day == day and closed and name in known):
                kept.append((bar_day, close))
        clipped[name] = kept
    return clipped


class Fit:
    """Forgetting ridge. Moments and the solve use only rows already passed to update."""

    def __init__(self, prior: list[float], forget: float = FORGET, lam: float = RIDGE, warm: int = WARM):
        self.prior = [float(item) if item > 1e-8 else 1.0 for item in prior]
        self.forget = forget
        self.lam = lam
        self.warm = warm
        self.width = len(self.prior)
        self.n = 0
        self.mean = [0.0] * self.width
        self.var = [item * item for item in self.prior]
        size = self.width + 1
        self.xtx = [[0.0] * size for _ in range(size)]
        self.xty = [0.0] * size
        self.resid = RESID_PRIOR
        self.scale = 1.0
        self.abs_z: list[float] = []

    def predict(self, raw: list[float]) -> dict:
        scaled = self._scale(raw)
        if self.n < self.warm:
            return self._blank(self._sigma(0.0))
        beta = self._beta()
        if beta is None:
            return self._blank(self._sigma(0.0))
        parts = [beta[index + 1] * scaled[index] for index in range(self.width)]
        base = beta[0]
        raw_mu = base + sum(parts)
        sigma = self._sigma(self._lever([1.0] + scaled))
        cap = min(CAP, 2.5 * sigma)
        clipped = max(-cap, min(cap, raw_mu))
        if raw_mu and clipped != raw_mu:
            factor = clipped / raw_mu
            parts = [part * factor for part in parts]
            base *= factor
        return {"mu": clipped, "sigma": sigma, "fitted": True, "parts": parts, "base": base}

    def update(self, raw: list[float], actual: float, quoted: float, sigma: float) -> None:
        scaled = self._scale(raw)
        row = [1.0] + scaled
        weight = 1.0 - self.forget
        for index, value in enumerate(raw):
            delta = value - self.mean[index]
            self.mean[index] = self.forget * self.mean[index] + weight * value
            self.var[index] = self.forget * self.var[index] + weight * delta * delta
            floor = (self.prior[index] * 0.25) ** 2
            if self.var[index] < floor:
                self.var[index] = floor
        size = self.width + 1
        for left in range(size):
            self.xty[left] = self.forget * self.xty[left] + row[left] * actual
            for right in range(size):
                self.xtx[left][right] = self.forget * self.xtx[left][right] + row[left] * row[right]
        miss = actual - quoted
        self.resid = self.forget * self.resid + weight * miss * miss
        if self.resid < 1.0:
            self.resid = 1.0
        if sigma > 0:
            self.abs_z.append(abs(miss) / sigma)
            self.abs_z = self.abs_z[-40:]
            if self.n >= self.warm and self.n % 20 == 0 and len(self.abs_z) >= 30:
                self._calibrate()
        self.n += 1

    def slopes(self) -> list[float]:
        beta = self._beta()
        if beta is None:
            return [0.0] * self.width
        return beta[1:]

    def _blank(self, sigma: float) -> dict:
        return {
            "mu": 0.0,
            "sigma": sigma,
            "fitted": False,
            "parts": [0.0] * self.width,
            "base": 0.0,
        }

    def _scale(self, raw: list[float]) -> list[float]:
        scaled = []
        for index, value in enumerate(raw):
            center = self.mean[index] if self.n else 0.0
            spread = math.sqrt(self.var[index])
            if spread < 1e-8:
                spread = self.prior[index]
            scaled.append((value - center) / spread)
        return scaled

    def _sigma(self, lever: float) -> float:
        width = math.sqrt(self.resid) * self.scale * math.sqrt(1.0 + min(max(lever, 0.0), 1.0))
        return max(width, EVOLVE_SIGMA_FLOOR)

    def _beta(self) -> list[float] | None:
        size = self.width + 1
        matrix = [row[:] for row in self.xtx]
        matrix[0][0] += self.lam * 0.25
        for index in range(1, size):
            matrix[index][index] += self.lam
        return _solve(matrix, self.xty[:])

    def _lever(self, row: list[float]) -> float:
        size = len(row)
        matrix = [line[:] for line in self.xtx]
        matrix[0][0] += self.lam * 0.25
        for index in range(1, size):
            matrix[index][index] += self.lam
        solved = _solve(matrix, row[:])
        if solved is None:
            return 0.0
        return max(0.0, sum(left * right for left, right in zip(row, solved)))

    def _calibrate(self) -> None:
        target = math.sqrt(2.0 / math.pi)
        got = sum(self.abs_z) / len(self.abs_z)
        if got > target * 1.12:
            self.scale = min(3.0, self.scale * 1.04)
        elif got < target * 0.88:
            self.scale = max(0.5, self.scale * 0.97)


def replay(
    samples: list[dict],
    forget: float = FORGET,
    lam: float = RIDGE,
    warm: int = WARM,
    prior: list[float] | None = None,
    live_from: str = "",
) -> tuple[list[dict], Fit]:
    """Predict each row, then fold it in. Changing a row's outcome cannot change that row's call."""
    width = len(samples[0]["x"]) if samples else len(CORE)
    fit = Fit(prior or [1.0] * width, forget, lam, warm)
    past_pairs: list[tuple[float, float]] = []
    live_past: list[dict] = []
    rows: list[dict] = []
    for sample in samples:
        pred = fit.predict(sample["x"])
        ready = live_ready(live_past)
        published = pred["mu"] if ready and pred["fitted"] else 0.0
        band = pred["sigma"] * coverage_scale(past_pairs[-PROOF_WINDOW:])
        row = {
            "day": str(sample.get("day") or ""),
            "actual": float(sample["actual"]),
            "mu": pred["mu"] if pred["fitted"] else 0.0,
            "published": published,
            "sigma": band,
            "raw_sigma": pred["sigma"],
            "fitted": pred["fitted"],
            "x0": float(sample["x"][0]) if sample["x"] else 0.0,
        }
        rows.append(row)
        quoted = published if pred["fitted"] else 0.0
        fit.update(sample["x"], row["actual"], quoted, pred["sigma"])
        if pred["fitted"]:
            past_pairs.append((abs(row["actual"] - published), pred["sigma"]))
            if live_from and row["day"] >= live_from:
                live_past.append(row)
    return rows, fit


def study(books: dict[str, list[tuple[str, float]]], stamp: datetime, live_from: str = "") -> dict | None:
    clipped = clip_books(books, stamp)
    spx = clipped.get("spx") or []
    if len(spx) < 80:
        return None
    samples = _samples(clipped)
    if len(samples) < WARM + 5:
        return None
    rows, fit = replay(samples, prior=[PRIOR[name] for name in CORE], live_from=live_from)
    anchor_day, anchor = spx[-1]
    current = feature_row(clipped, anchor_day)
    if current is None:
        return None
    pred = fit.predict([current[name] for name in CORE])
    fitted = [row for row in rows if row["fitted"]]
    live_rows = [row for row in fitted if live_from and row["day"] >= live_from]
    pairs = [(abs(row["actual"] - row["published"]), row["raw_sigma"]) for row in fitted]
    scale = coverage_scale(pairs[-PROOF_WINDOW:])
    raw = pred["mu"] if pred["fitted"] else 0.0
    mu = raw if pred["fitted"] and live_ready(live_rows) else 0.0
    sigma = pred["sigma"] * scale
    factor = (mu / raw) if raw else 0.0
    parts = [(LABELS[name], pred["parts"][index] * factor) for index, name in enumerate(CORE)]
    parts.append(("Base", pred["base"] * factor))
    due = next_session(anchor_day)
    proof = _proof(fitted, live_from)
    return {
        "anchor": round(anchor, 2),
        "anchor_day": anchor_day,
        "due": due,
        "gap": gap_phrase(anchor_day, due),
        "mu": round(mu, 2),
        "raw": round(raw, 2),
        "sigma": round(sigma, 2),
        "band_scale": round(scale, 2),
        "skill": round(1.0 if mu else 0.0, 4),
        "up": round(up_chance(mu, sigma), 4),
        "parts": [[label, round(value, 2)] for label, value in parts],
        "proof": proof,
        "trail": _trail(fitted),
    }


def run_evolve(store, stamp: datetime, fetch=None) -> dict | None:
    """Refresh the daily bars at most once a day, replay, and store the call."""
    if _stale(store, stamp):
        loader = _fetch_books if fetch is None else fetch
        try:
            fresh = loader()
        except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError):
            fresh = None
        if fresh and all(len(fresh.get(name) or []) >= 80 for name in YAHOO):
            for name in YAHOO:
                store.save_evolve_bars(name, fresh[name])
            store.set_meta("evolve_fetched", stamp.date().isoformat())
    books = {name: store.evolve_bars(name) for name in YAHOO}
    if any(len(rows) < 80 for rows in books.values()):
        return evolve_view(store)
    if not store.meta("evolve_started"):
        store.set_meta("evolve_started", stamp.date().isoformat())
    view = study(books, stamp, store.meta("evolve_started") or "")
    if view is None:
        return evolve_view(store)
    store.set_meta("evolve_view", json.dumps(view))
    from sp500team.signoff import note_author

    note_author(store, view)
    return view


def evolve_view(store) -> dict | None:
    raw = store.meta("evolve_view")
    if not raw:
        return None
    try:
        found = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(found, dict) or "sigma" not in found:
        return None
    return found


def _samples(books: dict[str, list[tuple[str, float]]]) -> list[dict]:
    spx = books["spx"]
    closes = {day: close for day, close in spx}
    samples = []
    days = [day for day, _close in spx]
    for index in range(len(days) - 1):
        day = days[index]
        values = feature_row(books, day)
        if values is None:
            continue
        nxt = days[index + 1]
        samples.append(
            {
                "day": day,
                "actual": closes[nxt] - closes[day],
                "x": [values[name] for name in CORE],
            }
        )
    return samples


def _skill_from(rows: list[dict]) -> float:
    chunk = rows[-SKILL_WINDOW:]
    if len(chunk) < SKILL_LEAST:
        return 0.0
    model = sum(abs(row["actual"] - row["mu"]) for row in chunk) / len(chunk)
    flat = sum(abs(row["actual"]) for row in chunk) / len(chunk)
    return skill_ratio(model, flat)


def _proof(fitted: list[dict], live_from: str) -> dict:
    window = fitted[-PROOF_WINDOW:]
    count = len(window)
    blank = {
        "studied": len(fitted),
        "window": count,
        "fit": 0.0,
        "printed": 0.0,
        "flat": 0.0,
        "momentum": 0.0,
        "band": 0.0,
        "direction": None,
        "direction_n": 0,
        "earlier": None,
        "later": None,
        "improving": False,
        "quieter": False,
        "live_n": 0,
        "live_mae": None,
    }
    if count == 0:
        return blank

    def _avg(rows: list[dict], fn) -> float:
        return sum(fn(row) for row in rows) / len(rows)

    leans = [row for row in window if abs(row["published"]) >= max(LEAN, 0.2 * row["sigma"])]
    direction = None
    if leans:
        direction = sum(1 for row in leans if row["actual"] * row["published"] > 0) / len(leans)
    earlier = later = None
    improving = False
    quieter = False
    if len(fitted) >= 80:
        before = fitted[-80:-40]
        after = fitted[-40:]
        earlier = _avg(before, lambda row: abs(row["actual"] - row["mu"]))
        later = _avg(after, lambda row: abs(row["actual"] - row["mu"]))
        later_flat = _avg(after, lambda row: abs(row["actual"]))
        earlier_flat = _avg(before, lambda row: abs(row["actual"]))
        improving = later + 1 < earlier and later < later_flat
        quieter = (not improving) and later_flat + QUIETER < earlier_flat
    live = [row for row in fitted if live_from and row["day"] >= live_from]
    live_mae = None
    if live:
        live_mae = _avg(live, lambda row: abs(row["actual"] - row["published"]))
    return {
        "studied": len(fitted),
        "window": count,
        "fit": round(_avg(window, lambda row: abs(row["actual"] - row["mu"])), 1),
        "printed": round(_avg(window, lambda row: abs(row["actual"] - row["published"])), 1),
        "flat": round(_avg(window, lambda row: abs(row["actual"])), 1),
        "momentum": round(_avg(window, lambda row: abs(row["actual"] - row["x0"])), 1),
        "band": round(sum(1 for row in window if abs(row["actual"] - row["published"]) <= row["sigma"]) / count, 3),
        "direction": None if direction is None else round(direction, 3),
        "direction_n": len(leans),
        "earlier": None if earlier is None else round(earlier, 1),
        "later": None if later is None else round(later, 1),
        "improving": improving,
        "quieter": quieter,
        "live_n": len(live),
        "live_mae": None if live_mae is None else round(live_mae, 1),
    }


def _trail(fitted: list[dict]) -> list[list[float]]:
    points = []
    window = 40
    for end in range(window, len(fitted) + 1):
        if end != len(fitted) and (len(fitted) - end) % 5:
            continue
        chunk = fitted[end - window : end]
        band = sum(row["sigma"] for row in chunk) / window
        typical = sum(abs(row["actual"]) for row in chunk) / window
        points.append([round(band, 1), round(typical, 1)])
    return points[-36:]


def _change(rows: list[tuple[str, float]], day: str, strict: bool) -> float | None:
    if strict:
        closes = [close for bar_day, close in rows if bar_day < day]
    else:
        closes = [close for bar_day, close in rows if bar_day <= day]
    if len(closes) < 2:
        return None
    return closes[-1] - closes[-2]


def _clean(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    folded: dict[str, float] = {}
    for day, close in rows:
        folded[str(day)] = float(close)
    return sorted(folded.items())


def _stale(store, stamp: datetime) -> bool:
    if any(len(store.evolve_bars(name)) < 80 for name in YAHOO):
        return True
    return store.meta("evolve_fetched") != stamp.date().isoformat()


def _fetch_books() -> dict[str, list[tuple[str, float]]]:
    from sp500team.search import fetch_bars

    books = {}
    with httpx.Client(timeout=20, headers={"User-Agent": "sp500team/0.1"}) as client:
        for name, symbol in YAHOO.items():
            books[name] = fetch_bars(client, symbol)
    return books
