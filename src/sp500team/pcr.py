from __future__ import annotations

import csv
import math
import re
from datetime import datetime
from io import StringIO

import httpx

INDEX_PCR_URL = "https://www.cboe.com/markets/us/options/market-statistics/daily/"


def in_session(now: datetime, start: str, end: str) -> bool:
    if now.weekday() >= 5:
        return False
    clock = now.strftime("%H:%M")
    return start <= clock <= end


def fetch_index_pcr(client: httpx.Client, url: str = INDEX_PCR_URL) -> list[tuple[str, float]]:
    """CBOE index put/call rows as (New York session date, ratio)."""
    response = client.get(url, timeout=12, follow_redirects=True, headers={"User-Agent": "sp500team/0.1"})
    response.raise_for_status()
    text = response.text
    if "INDEX PUT/CALL RATIO" in text:
        return parse_index_page(text)
    return parse_index_pcr(text)


def parse_index_page(text: str) -> list[tuple[str, float]]:
    """Today's index put/call from the CBOE daily statistics page."""
    day = re.search(r'selectedDate\\?":\\?"(\d{4}-\d{2}-\d{2})', text)
    ratio = re.search(r'INDEX PUT/CALL RATIO\\?",\\?"value\\?":\\?"([0-9.]+)', text)
    if day is None or ratio is None:
        return []
    value = float(ratio.group(1))
    if not 0 < value < 10:
        return []
    return [(day.group(1), value)]


def parse_index_pcr(text: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    reader = csv.reader(StringIO(text))
    for raw in reader:
        if len(raw) < 2:
            continue
        day = _day(raw[0].strip())
        if day is None:
            continue
        ratio = _ratio(raw)
        if ratio is None:
            continue
        rows.append((day, ratio))
    rows.sort()
    return rows


def trailing_std(values: list[float]) -> float:
    if len(values) < 4:
        return 0.05
    mean = sum(values) / len(values)
    var = sum((item - mean) ** 2 for item in values) / len(values)
    return math.sqrt(var) or 0.05


def _day(text: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _ratio(raw: list[str]) -> float | None:
    for cell in reversed(raw[1:]):
        token = cell.strip().replace("%", "")
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if 0 < value < 10:
            return value
    return None
