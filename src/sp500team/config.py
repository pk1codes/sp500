from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ET = ZoneInfo("America/New_York")

BEAT_LABELS = {
    "global": "World markets",
    "crude": "Crude",
    "fx": "Dollar",
    "policy": "Policy",
    "flows": "Flows",
    "weights": "Heavyweights",
    "pcr": "Put-call ratio",
}

ROSTER = (
    ("global", "World"),
    ("crude", "Crude"),
    ("fx", "Dollar"),
    ("policy", "Policy"),
    ("flows", "Flows"),
    ("weights", "Weights"),
    ("pcr", "Options"),
    ("scout", "Scout"),
)
BOARD_AGENT = "board"
SCORE_AGENT = "score"
BOSS_AGENT = "boss"
QUIET = "Quiet"
SHUT = "Market shut"
NO_ANSWER = "No answer"
FEED_DOWN = "Feed down"
SEARCH_FAILED = "Search failed"
STATUS_LINES = {QUIET, SHUT, NO_ANSWER, FEED_DOWN, SEARCH_FAILED}
MISSED_LINES = {NO_ANSWER, FEED_DOWN, SEARCH_FAILED}


def now_et() -> datetime:
    return datetime.now(ET)


def find_root() -> Path:
    env = os.environ.get("SP500_HOME")
    if env:
        return Path(env)
    cwd = Path.cwd()
    if (cwd / "beats.yaml").exists():
        return cwd
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "beats.yaml").exists():
            return parent
    return cwd


def load_config(root: Path | None = None) -> dict:
    root = root or find_root()
    with (root / "beats.yaml").open() as handle:
        return yaml.safe_load(handle)


def load_dotenv(root: Path) -> None:
    path = root / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def impact_text(value: float) -> str:
    number = int(round(value))
    if number > 0:
        return f"+{number}"
    return str(number)


def impact_class(value: float) -> str:
    number = int(round(value))
    if number > 0:
        return "plus"
    if number < 0:
        return "minus"
    return "flat"


def clip(text: str, limit: int = 78) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"
