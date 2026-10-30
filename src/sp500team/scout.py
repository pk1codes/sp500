from __future__ import annotations

import re
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import feedparser

import httpx

QUERIES = (
    "S&P 500 news today",
    "Fed policy",
    "crude oil",
    "dollar",
    "US equity fund flows",
    "S&P earnings Apple Microsoft Nvidia Amazon Alphabet Meta Broadcom Berkshire Lilly JPMorgan",
)

JUNK_DOMAINS = {
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
    "quora.com",
    "linkedin.com",
    "wikipedia.org",
    "duckduckgo.com",
    "www.google.com",
    "google.com",
    "t.co",
    "news.google.com",
}

_RESULT = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.I | re.S,
)
_TAGS = re.compile(r"<[^>]+>")


def query_for(hour: int) -> str:
    return QUERIES[hour % len(QUERIES)]


def gather_stories(fetch) -> list[str]:
    """Top stories from every query. Each query keeps at most eight."""
    titles = []
    seen = set()
    for query in QUERIES:
        for row in list(fetch(query) or [])[:8]:
            title = (row.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            titles.append(title)
    return titles


def search_all(client: httpx.Client) -> list[str]:
    def fetch(query: str) -> list[dict]:
        try:
            return search_web(client, query)
        except (httpx.HTTPError, ValueError):
            return []

    return gather_stories(fetch)


def scout_sentence(stories: int, sources: int, failed: bool = False) -> str:
    if failed:
        return "Scout looked across the web. The search did not answer."
    story_word = "story" if stories == 1 else "stories"
    source_word = "source" if sources == 1 else "sources"
    return f"Scout looked across the web. {stories} new {story_word}, {sources} new {source_word}."


def agent_line(worked: set[str]) -> str:
    count = max(1, len(worked))
    word = "agent" if count == 1 else "agents"
    return f"{count} {word} worked this update."


def google_news_url(query: str) -> str:
    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )


def parse_google_feed(raw: str) -> list[dict]:
    parsed = feedparser.parse(raw)
    found = []
    seen = set()
    for entry in parsed.entries:
        source = entry.get("source") or {}
        href = source.get("href") or source.get("url") or ""
        title = " ".join((entry.get("title") or "").split())
        domain = bare_domain(href)
        if not title or not href.startswith("http") or domain in JUNK_DOMAINS or href in seen:
            continue
        seen.add(href)
        found.append({"title": title, "url": href, "domain": domain})
        if len(found) == 8:
            break
    return found


def fetch_google_news(client: httpx.Client, query: str) -> list[dict]:
    response = client.get(
        google_news_url(query),
        timeout=12,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 sp500team/0.1"},
    )
    response.raise_for_status()
    return parse_google_feed(response.text)


def search_web(client: httpx.Client, query: str) -> list[dict]:
    try:
        rows = fetch_google_news(client, query)
        if rows:
            return rows
    except (httpx.HTTPError, ValueError):
        pass
    return _duckduckgo(client, query)


def _duckduckgo(client: httpx.Client, query: str) -> list[dict]:
    response = client.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        timeout=12,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 sp500team/0.1"},
    )
    response.raise_for_status()
    return parse_results(response.text)


def parse_results(page: str) -> list[dict]:
    found = []
    seen = set()
    for href, raw_title in _RESULT.findall(page):
        url = unwrap(unescape(href))
        title = _TAGS.sub("", unescape(raw_title))
        title = " ".join(title.split())
        domain = bare_domain(url)
        if not title or not url.startswith("http") or domain in JUNK_DOMAINS or url in seen:
            continue
        seen.add(url)
        found.append({"title": title, "url": url, "domain": domain})
        if len(found) == 8:
            break
    return found


def unwrap(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else href


def bare_domain(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host
