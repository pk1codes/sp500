from __future__ import annotations

from datetime import datetime, time, timedelta

from sp500team.config import ET

OPEN = time(9, 30)
CLOSE = time(16, 0)
AHEAD_AT = time(16, 30)


def next_open(stamp: datetime) -> datetime:
    return _next_mark(stamp, OPEN)


def session_close(stamp: datetime) -> datetime:
    return _next_mark(stamp, CLOSE)


def decision_due(stamp: datetime) -> datetime | None:
    """Next cash close for a weekday tick at or after 16:30 New York."""
    if stamp.weekday() >= 5:
        return None
    if stamp.timetz().replace(tzinfo=None) < AHEAD_AT:
        return None
    return session_close(stamp)


def cash_open(stamp: datetime) -> bool:
    """True during the weekday cash session, from 09:30 through 16:00."""
    if stamp.weekday() >= 5:
        return False
    clock = stamp.timetz().replace(tzinfo=None)
    return OPEN <= clock <= CLOSE


def last_cash_close(bars: list, stamp: datetime) -> float | None:
    today = stamp.date().isoformat()
    done = []
    for bar in bars:
        day, close, _opened = _parts(bar)
        if day < today or (day == today and _done_for_day(stamp)):
            done.append((day, close))
    if not done:
        return None
    done.sort()
    return done[-1][1]


def session_move(bars: list, day: str) -> float | None:
    for bar in bars:
        bar_day, close, opened = _parts(bar)
        if bar_day == day and opened is not None:
            return close - opened
    return None


def sessions_ahead(stamp: datetime, count: int = 5) -> datetime:
    day = stamp.date()
    left = count
    while left > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            left -= 1
    return datetime.combine(day, CLOSE, tzinfo=stamp.tzinfo or ET)


def forecast_move(bars: list, score_day: str, anchor: float | None) -> float | None:
    """Later close minus the level when the forecast was made."""
    if anchor is None:
        return None
    for bar in bars:
        day, close, _opened = _parts(bar)
        if day >= score_day:
            return close - anchor
    return None


def week_move(bars: list, score_day: str, anchor: float | None) -> float | None:
    if anchor is None:
        return None
    for bar in bars:
        day, close, _opened = _parts(bar)
        if day >= score_day:
            return close - anchor
    return None


def day_open(bars: list, day: str) -> float | None:
    for bar in bars:
        bar_day, _close, opened = _parts(bar)
        if bar_day == day:
            return opened
    return None


def _next_mark(stamp: datetime, mark: time) -> datetime:
    day = stamp.date()
    candidate = datetime.combine(day, mark, tzinfo=ET)
    if stamp.tzinfo is not None:
        candidate = candidate.astimezone(stamp.tzinfo)
    if stamp < candidate and day.weekday() < 5:
        return candidate
    day = day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    found = datetime.combine(day, mark, tzinfo=ET)
    if stamp.tzinfo is not None and stamp.tzinfo is not ET:
        return found.astimezone(stamp.tzinfo)
    return found


def _done_for_day(stamp: datetime) -> bool:
    return stamp.timetz().replace(tzinfo=None) >= CLOSE


def _parts(bar) -> tuple[str, float, float | None]:
    opened = bar[2] if len(bar) > 2 else None
    return bar[0], float(bar[1]), None if opened is None else float(opened)
