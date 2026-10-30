from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse

import feedparser
import httpx

from sp500team.config import ET

UA = "sp500team/0.1 (local research; +https://localhost)"
LIVE_SECONDS = 300


def quote_stamp(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=ET)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace(".", "", 1).isdigit()):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone(ET)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=ET)
    except ValueError:
        pass
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=ET)
        except ValueError:
            continue
    return None


def quote_is_live(stamp: datetime | None, now: datetime, seconds: int = LIVE_SECONDS) -> bool:
    if stamp is None:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    return abs((now - stamp).total_seconds()) <= seconds


def fetch_feed(client: httpx.Client, url: str) -> list[dict]:
    response = client.get(url, timeout=12, follow_redirects=True)
    response.raise_for_status()
    return feed_items(response.content)


def feed_items(raw: bytes) -> list[dict]:
    parsed = feedparser.parse(raw)
    items = []
    for entry in parsed.entries:
        link = entry.get("link") or ""
        title = entry.get("title") or ""
        if not link or not title:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "published": _published(entry),
                "summary": entry.get("summary") or "",
            }
        )
    if items:
        return items
    return _listing(raw)


def _listing(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8", "replace")
    items = []
    for blob in re.findall(r'<script type="application/ld\+json">(.*?)</script>', text, re.S | re.I):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        for row in data.get("itemListElement") or []:
            title = str(row.get("name") or "").strip()
            link = str(row.get("url") or "").strip()
            if not title or not link:
                continue
            items.append({"title": title, "url": link, "published": None, "summary": ""})
    return items


def ny_day(stamp: float) -> str:
    """Calendar date in New York for a UTC unix timestamp."""
    return datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone(ET).date().isoformat()


def fetch_price(client: httpx.Client) -> tuple[float | None, list[tuple[str, float]]]:
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
        params={"interval": "1d", "range": "1mo"},
        timeout=12,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    stamps = result.get("timestamp") or []
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close") or []
    opens = list(quote.get("open") or [])
    if len(opens) < len(closes):
        opens.extend([None] * (len(closes) - len(opens)))
    bars = []
    for stamp, close, opened in zip(stamps, closes, opens):
        if close is None:
            continue
        day = ny_day(stamp)
        bars.append((day, float(close), None if opened is None else float(opened)))
    traded = quote_stamp(meta.get("regularMarketTime"))
    return (float(price) if price is not None else None), bars, traded


def fetch_bars(client: httpx.Client, symbol: str, span: str = "5y") -> list[tuple[str, float]]:
    """Daily closes. The date is the New York date of the bar's timestamp."""
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
        params={"interval": "1d", "range": span},
        timeout=20,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    closes = (result.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    folded: dict[str, float] = {}
    for stamp, close in zip(stamps, closes):
        if stamp is None or close is None:
            continue
        day = ny_day(stamp)
        folded[day] = float(close)
    return sorted(folded.items())


def fetch_level(client: httpx.Client, symbol: str) -> float | None:
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
        params={"interval": "1d", "range": "5d"},
        timeout=12,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    meta = response.json()["chart"]["result"][0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    return None if price is None else float(price)


def fetch_es(client: httpx.Client) -> dict | None:
    """ES futures, the price traded outside the cash session."""
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote("ES=F", safe=""),
        params={"interval": "1d", "range": "5d"},
        timeout=12,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    meta = response.json()["chart"]["result"][0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    previous = meta.get("chartPreviousClose")
    if previous is None:
        previous = meta.get("previousClose")
    change = 0.0 if previous is None else float(price) - float(previous)
    stamp = quote_stamp(meta.get("regularMarketTime"))
    return {
        "price": float(price),
        "change": change,
        "asof": stamp.isoformat() if stamp else "",
    }


def bar_clock(client: httpx.Client, symbol: str):
    """New York time of the latest daily bar. Used to decide same-day membership."""
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
        params={"interval": "1d", "range": "5d"},
        timeout=20,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    stamps = response.json()["chart"]["result"][0].get("timestamp") or []
    if not stamps:
        return None
    return datetime.fromtimestamp(stamps[-1], tz=timezone.utc).astimezone(ET).timetz().replace(tzinfo=None)


def fetch_spot_quote(client: httpx.Client, now: datetime | None = None) -> dict | None:
    now = now or datetime.now(ET)
    response = client.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC",
        params={"interval": "1m", "range": "1d"},
        timeout=12,
        headers={"User-Agent": "sp500team/0.1"},
    )
    response.raise_for_status()
    meta = response.json()["chart"]["result"][0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    stamp = quote_stamp(meta.get("regularMarketTime"))
    return {
        "price": float(price),
        "live": quote_is_live(stamp, now),
        "asof": stamp.isoformat() if stamp else "",
    }


def live_quotes(now: datetime | None = None) -> dict:
    now = now or datetime.now(ET)
    found = {"spx": None, "es": None}
    try:
        with httpx.Client(timeout=12, headers={"User-Agent": "sp500team/0.1"}) as client:
            try:
                found["spx"] = fetch_spot_quote(client, now)
            except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError):
                found["spx"] = None
            try:
                es = fetch_es(client)
            except (httpx.HTTPError, KeyError, ValueError, TypeError):
                es = None
            if es is not None:
                es["live"] = quote_is_live(quote_stamp(es.get("asof")), now)
                found["es"] = es
    except httpx.HTTPError:
        return found
    return found


def close_on_or_after(bars: list, day: str) -> float | None:
    for bar in bars:
        if bar[0] >= day:
            return float(bar[1])
    return None


def _published(entry) -> datetime | None:
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            stamp = parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ET)
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(ET)


def fresh(items: list[dict], since: datetime) -> list[dict]:
    kept = []
    for item in items:
        if item["published"] is None or item["published"] >= since:
            kept.append(item)
    kept.sort(key=lambda item: item["published"] or since, reverse=True)
    return kept[:8]


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def hrefs(summary: str) -> list[str]:
    import re

    return re.findall(r"https?://[^\"'\s<]+", summary or "")

