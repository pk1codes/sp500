from __future__ import annotations

import sqlite3
from pathlib import Path

from sp500team.models import Claim, Ensemble, Event, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    beat TEXT NOT NULL,
    kind TEXT NOT NULL,
    url TEXT NOT NULL,
    status TEXT NOT NULL,
    weight REAL NOT NULL,
    n_scored INTEGER NOT NULL DEFAULT 0,
    residual_sum REAL NOT NULL DEFAULT 0,
    residual_sq_sum REAL NOT NULL DEFAULT 0,
    fail_streak INTEGER NOT NULL DEFAULT 0,
    domain TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    beat TEXT NOT NULL,
    source_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    headline TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    shock_key TEXT NOT NULL,
    direction TEXT NOT NULL,
    points_low REAL NOT NULL,
    points_high REAL NOT NULL,
    midpoint REAL NOT NULL,
    confidence REAL NOT NULL,
    nifty_level REAL,
    score_after TEXT NOT NULL,
    status TEXT NOT NULL,
    actual_move REAL,
    residual REAL,
    body TEXT NOT NULL DEFAULT '',
    ensemble_id TEXT,
    horizon TEXT NOT NULL DEFAULT 'session',
    driver REAL
);
CREATE TABLE IF NOT EXISTS ensembles (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    n_shocks INTEGER NOT NULL,
    nifty_level REAL,
    actual REAL,
    residual REAL,
    horizon TEXT NOT NULL DEFAULT 'session'
);
CREATE TABLE IF NOT EXISTS levels (
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (day, name)
);
CREATE TABLE IF NOT EXISTS level_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    change REAL NOT NULL,
    move REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    sentence TEXT NOT NULL,
    impact REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pcr_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    expiry TEXT NOT NULL,
    oi_pcr REAL NOT NULL,
    vol_pcr REAL NOT NULL,
    session_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    domain TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS families (
    key TEXT PRIMARY KEY,
    n_scored INTEGER NOT NULL DEFAULT 0,
    residual_sum REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS misses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    title TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS family_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    signal REAL NOT NULL,
    move REAL NOT NULL,
    clean INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ahead_calls (
    day TEXT PRIMARY KEY,
    anchor REAL NOT NULL,
    due_day TEXT NOT NULL,
    predicted_move REAL NOT NULL,
    baseline_move REAL NOT NULL DEFAULT 0,
    model_move REAL NOT NULL,
    train_n INTEGER NOT NULL DEFAULT 0,
    actual_move REAL,
    status TEXT NOT NULL,
    parts TEXT NOT NULL DEFAULT '',
    model_abs REAL,
    base_abs REAL
);
CREATE TABLE IF NOT EXISTS ahead_snapshots (
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (day, name)
);
CREATE TABLE IF NOT EXISTS ahead_drivers (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    contribution REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ahead_topics (
    phrase TEXT PRIMARY KEY,
    direction INTEGER NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    misses INTEGER NOT NULL DEFAULT 0,
    alive INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ahead_topic_days (
    day TEXT NOT NULL,
    phrase TEXT NOT NULL,
    direction INTEGER NOT NULL,
    scored INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, phrase)
);
CREATE TABLE IF NOT EXISTS evolve_bars (
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    close REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def seed_sources(self, rows: list[dict]) -> None:
        for row in rows:
            domain = ""
            if row["kind"] == "rss":
                domain = _domain(row["url"])
            weight = 1.0 if row.get("important") else 0.4
            self.conn.execute(
                """
                INSERT OR IGNORE INTO sources
                (id, beat, kind, url, status, weight, domain)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (row["id"], row["beat"], row["kind"], row["url"], weight, domain),
            )
            if row.get("important"):
                self.conn.execute(
                    "UPDATE sources SET weight = 1.0 WHERE id = ? AND n_scored = 0 AND weight < 1.0",
                    (row["id"],),
                )
        self.conn.commit()

    def sources(self, kinds: set[str] | None = None) -> list[Source]:
        sql = "SELECT * FROM sources"
        rows = self.conn.execute(sql).fetchall()
        found = [_source(row) for row in rows]
        if kinds is None:
            return found
        return [row for row in found if row.kind in kinds]

    def source(self, source_id: str) -> Source | None:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return _source(row) if row else None

    def save_source(self, source: Source) -> None:
        self.conn.execute(
            """
            UPDATE sources
            SET status=?, weight=?, n_scored=?, residual_sum=?, residual_sq_sum=?,
                fail_streak=?, domain=?
            WHERE id=?
            """,
            (
                source.status,
                source.weight,
                source.n_scored,
                source.residual_sum,
                source.residual_sq_sum,
                source.fail_streak,
                source.domain,
                source.id,
            ),
        )
        self.conn.commit()

    def known_domains(self) -> set[str]:
        rows = self.conn.execute("SELECT domain FROM sources WHERE domain != ''").fetchall()
        return {row["domain"] for row in rows}

    def remember_link(self, domain: str, url: str, status: str) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO links (domain, url, status) VALUES (?, ?, ?)",
            (domain, url, status),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def add_probation_source(self, domain: str, url: str, beat: str, kind: str = "rss") -> bool:
        source_id = "prob-" + domain.replace(".", "-")
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO sources
            (id, beat, kind, url, status, weight, domain)
            VALUES (?, ?, ?, ?, 'probation', 0.3, ?)
            """,
            (source_id, beat, kind, url, domain),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def add_claim(self, claim: Claim) -> bool:
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO claims (
                id, beat, source_id, ts, headline, url, shock_key, direction,
                points_low, points_high, midpoint, confidence, nifty_level,
                score_after, status, body, ensemble_id, horizon, driver
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                claim.beat,
                claim.source_id,
                claim.ts,
                claim.headline,
                claim.url,
                claim.shock_key,
                claim.direction,
                claim.points_low,
                claim.points_high,
                claim.midpoint,
                claim.confidence,
                claim.nifty_level,
                claim.score_after,
                claim.status,
                claim.body,
                claim.ensemble_id,
                claim.horizon,
                claim.driver,
            ),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def attach_ensemble(self, claim_ids: list[str], ensemble_id: str) -> None:
        self.conn.executemany(
            "UPDATE claims SET ensemble_id=? WHERE id=?",
            [(ensemble_id, claim_id) for claim_id in claim_ids],
        )
        self.conn.commit()

    def scored_claims(self) -> list[Claim]:
        rows = self.conn.execute(
            "SELECT * FROM claims WHERE residual IS NOT NULL ORDER BY ts, id"
        ).fetchall()
        return [_claim(row) for row in rows]

    def due_claims(self, now_iso: str) -> list[Claim]:
        rows = self.conn.execute(
            """
            SELECT * FROM claims
            WHERE status='open' AND score_after <= ?
            ORDER BY ts
            """,
            (now_iso,),
        ).fetchall()
        return [_claim(row) for row in rows]

    def save_claim(self, claim: Claim) -> None:
        self.conn.execute(
            """
            UPDATE claims
            SET status=?, actual_move=?, residual=?, ensemble_id=?
            WHERE id=?
            """,
            (claim.status, claim.actual_move, claim.residual, claim.ensemble_id, claim.id),
        )
        self.conn.commit()

    def add_ensemble(self, item: Ensemble) -> None:
        self.conn.execute(
            """
            INSERT INTO ensembles (id, ts, mu, sigma, n_shocks, nifty_level, actual, residual, horizon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (item.id, item.ts, item.mu, item.sigma, item.n_shocks, item.nifty_level, item.actual, item.residual, item.horizon),
        )
        self.conn.commit()

    def latest_ensemble(self, horizon: str | None = None) -> Ensemble | None:
        if horizon is None:
            row = self.conn.execute("SELECT * FROM ensembles ORDER BY ts DESC LIMIT 1").fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM ensembles WHERE horizon = ? ORDER BY ts DESC LIMIT 1",
                (horizon,),
            ).fetchone()
        return _ensemble(row) if row else None

    def ensembles(self) -> list[Ensemble]:
        rows = self.conn.execute("SELECT * FROM ensembles ORDER BY ts").fetchall()
        return [_ensemble(row) for row in rows]

    def save_ensemble(self, item: Ensemble) -> None:
        self.conn.execute(
            "UPDATE ensembles SET actual=?, residual=? WHERE id=?",
            (item.actual, item.residual, item.id),
        )
        self.conn.commit()

    def add_event(self, ts: str, sentence: str, impact: float, issue: bool = False, agent: str = "") -> None:
        self.conn.execute(
            "INSERT INTO events (ts, sentence, impact, issue, agent) VALUES (?, ?, ?, ?, ?)",
            (ts, sentence, impact, 1 if issue else 0, agent),
        )
        self.conn.commit()

    def events(self, limit: int = 80) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Event(
                id=row["id"],
                ts=row["ts"],
                sentence=row["sentence"],
                impact=row["impact"],
                issue=bool(row["issue"]),
                agent=row["agent"] if "agent" in row.keys() else "",
            )
            for row in rows
        ]

    def add_pcr(self, ts: str, expiry: str, oi_pcr: float, vol_pcr: float, session_date: str, bucket: str = "week", oi_change: float = 0) -> None:
        self.conn.execute(
            """
            INSERT INTO pcr_snapshots (ts, expiry, oi_pcr, vol_pcr, session_date, bucket, oi_change)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, expiry, oi_pcr, vol_pcr, session_date, bucket, oi_change),
        )
        self.conn.commit()

    def recent_pcr(self, limit: int = 40, bucket: str | None = None) -> list[sqlite3.Row]:
        if bucket is None:
            return self.conn.execute(
                "SELECT * FROM pcr_snapshots ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM pcr_snapshots WHERE bucket = ? ORDER BY id DESC LIMIT ?",
            (bucket, limit),
        ).fetchall()

    def save_level(self, day: str, name: str, value: float) -> None:
        self.conn.execute(
            """
            INSERT INTO levels (day, name, value) VALUES (?, ?, ?)
            ON CONFLICT(day, name) DO UPDATE SET value=excluded.value
            """,
            (day, name, value),
        )
        self.conn.commit()

    def previous_level(self, name: str, day: str) -> float | None:
        row = self.conn.execute(
            "SELECT value FROM levels WHERE name = ? AND day < ? ORDER BY day DESC LIMIT 1",
            (name, day),
        ).fetchone()
        return float(row["value"]) if row else None

    def add_level_pair(self, name: str, change: float, move: float) -> None:
        self.conn.execute(
            "INSERT INTO level_pairs (name, change, move) VALUES (?, ?, ?)",
            (name, change, move),
        )
        self.conn.commit()

    def families(self) -> list[tuple[str, int, float]]:
        rows = self.conn.execute(
            "SELECT key, n_scored, residual_sum FROM families WHERE n_scored > 0 ORDER BY key"
        ).fetchall()
        return [(row["key"], int(row["n_scored"]), float(row["residual_sum"])) for row in rows]

    def family_stats(self, key: str) -> tuple[int, float]:
        row = self.conn.execute(
            "SELECT n_scored, residual_sum FROM families WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return 0, 0.0
        return int(row["n_scored"]), float(row["residual_sum"])

    def add_family_residual(self, key: str, residual: float) -> None:
        self.conn.execute(
            """
            INSERT INTO families (key, n_scored, residual_sum) VALUES (?, 1, ?)
            ON CONFLICT(key) DO UPDATE SET
                n_scored = n_scored + 1,
                residual_sum = residual_sum + excluded.residual_sum
            """,
            (key, residual),
        )
        self.conn.commit()

    def family_points(self, key: str) -> tuple[int, float]:
        row = self.conn.execute(
            "SELECT points_n, points_sum FROM families WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return 0, 0.0
        return int(row["points_n"] or 0), float(row["points_sum"] or 0)

    def add_family_points(self, key: str, points: float) -> None:
        self.conn.execute(
            """
            INSERT INTO families (key, points_n, points_sum) VALUES (?, 1, ?)
            ON CONFLICT(key) DO UPDATE SET
                points_n = points_n + 1,
                points_sum = points_sum + excluded.points_sum
            """,
            (key, points),
        )
        self.conn.commit()

    def family_coverage(self, key: str) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT inside_n, outside_n FROM families WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return 0, 0
        return int(row["inside_n"] or 0), int(row["outside_n"] or 0)

    def add_family_coverage(self, key: str, inside: bool) -> None:
        column = "inside_n" if inside else "outside_n"
        self.conn.execute(
            f"""
            INSERT INTO families (key, {column}) VALUES (?, 1)
            ON CONFLICT(key) DO UPDATE SET {column} = {column} + 1
            """,
            (key,),
        )
        self.conn.commit()

    def add_family_pair(self, key: str, signal: float, move: float, clean: bool = False) -> None:
        self.conn.execute(
            "INSERT INTO family_pairs (key, signal, move, clean) VALUES (?, ?, ?, ?)",
            (key, signal, move, 1 if clean else 0),
        )
        self.conn.commit()

    def family_pairs(self, key: str, clean_only: bool = False) -> list[tuple[float, float]]:
        query = "SELECT signal, move FROM family_pairs WHERE key = ?"
        if clean_only:
            query += " AND clean = 1"
        rows = self.conn.execute(query + " ORDER BY id", (key,)).fetchall()
        return [(float(row["signal"]), float(row["move"])) for row in rows]

    def add_misses(self, ts: str, titles: list[tuple[str, str, int]]) -> int:
        self.conn.executemany(
            "INSERT INTO misses (ts, title, kind, sign) VALUES (?, ?, ?, ?)",
            [(ts, title, kind, sign) for title, kind, sign in titles],
        )
        self.conn.commit()
        return len(titles)

    def miss_history(self) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT kind, sign FROM misses WHERE kind != '' AND sign != 0"
        ).fetchall()
        return [(row["kind"], int(row["sign"])) for row in rows]

    def miss_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM misses").fetchone()
        return int(row["n"])

    def ahead_call(self, day: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM ahead_calls WHERE day = ?", (day,)).fetchone()

    def ahead_calls(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM ahead_calls ORDER BY day").fetchall()

    def latest_ahead(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM ahead_calls WHERE status != 'discarded' ORDER BY day DESC LIMIT 1"
        ).fetchone()

    def add_ahead_call(
        self,
        day: str,
        anchor: float,
        due_day: str,
        predicted_move: float,
        model_move: float,
        train_n: int,
        parts: str,
    ) -> bool:
        cursor = self.conn.execute(
            """
            INSERT INTO ahead_calls (
                day, anchor, due_day, predicted_move, baseline_move, model_move, train_n, status, parts
            ) VALUES (?, ?, ?, ?, 0, ?, ?, 'open', ?)
            ON CONFLICT(day) DO NOTHING
            """,
            (day, anchor, due_day, predicted_move, model_move, train_n, parts),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def finish_ahead(self, day: str, actual: float | None, status: str) -> None:
        if actual is None:
            self.conn.execute(
                """
                UPDATE ahead_calls SET status = ?
                WHERE day = ? AND status = 'open'
                """,
                (status, day),
            )
        else:
            self.conn.execute(
                """
                UPDATE ahead_calls
                SET actual_move = ?, status = ?,
                    model_abs = abs(? - model_move),
                    base_abs = abs(?)
                WHERE day = ? AND status = 'open'
                """,
                (actual, status, actual, actual, day),
            )
        self.conn.commit()

    def topic(self, phrase: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM ahead_topics WHERE phrase = ?", (phrase,)).fetchone()

    def topic_seen_before(self, phrase: str, day: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 AS hit FROM ahead_topic_days WHERE phrase = ? AND day < ? LIMIT 1",
            (phrase, day),
        ).fetchone()
        return row is not None

    def save_topic_day(self, day: str, phrase: str, direction: int) -> None:
        self.conn.execute(
            """
            INSERT INTO ahead_topic_days (day, phrase, direction) VALUES (?, ?, ?)
            ON CONFLICT(day, phrase) DO NOTHING
            """,
            (day, phrase, direction),
        )
        self.conn.commit()

    def ensure_topic(self, phrase: str, direction: int) -> None:
        self.conn.execute(
            """
            INSERT INTO ahead_topics (phrase, direction) VALUES (?, ?)
            ON CONFLICT(phrase) DO NOTHING
            """,
            (phrase, direction),
        )
        self.conn.commit()

    def topic_days_on(self, day: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM ahead_topic_days WHERE day = ? ORDER BY phrase",
            (day,),
        ).fetchall()

    def mark_topic_day(self, day: str, phrase: str) -> None:
        self.conn.execute(
            "UPDATE ahead_topic_days SET scored = 1 WHERE day = ? AND phrase = ?",
            (day, phrase),
        )
        self.conn.commit()

    def mark_topic(self, phrase: str, hit: bool) -> None:
        column = "hits" if hit else "misses"
        self.conn.execute(
            f"UPDATE ahead_topics SET {column} = {column} + 1 WHERE phrase = ?",
            (phrase,),
        )
        self.conn.commit()

    def fade_topic(self, phrase: str) -> None:
        self.conn.execute("UPDATE ahead_topics SET alive = 0, direction = 0 WHERE phrase = ?", (phrase,))
        self.conn.commit()

    def last_session_pcr(self, day: str) -> float | None:
        rows = self.conn.execute(
            """
            SELECT oi_pcr FROM pcr_snapshots
            WHERE session_date = ? AND bucket = 'day' AND substr(ts, 12, 5) <= '16:00'
            ORDER BY ts
            """,
            (day,),
        ).fetchall()
        if not rows:
            return None
        return float(rows[-1]["oi_pcr"])

    def save_snapshots(self, day: str, values: dict[str, float]) -> None:
        self.conn.executemany(
            """
            INSERT INTO ahead_snapshots (day, name, value) VALUES (?, ?, ?)
            ON CONFLICT(day, name) DO NOTHING
            """,
            [(day, name, value) for name, value in values.items()],
        )
        self.conn.commit()

    def snapshots(self, day: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT name, value FROM ahead_snapshots WHERE day = ?",
            (day,),
        ).fetchall()
        return {row["name"]: float(row["value"]) for row in rows}

    def previous_snapshot(self, name: str, day: str) -> float | None:
        row = self.conn.execute(
            """
            SELECT value FROM ahead_snapshots
            WHERE name = ? AND day < ?
            ORDER BY day DESC LIMIT 1
            """,
            (name, day),
        ).fetchone()
        return None if row is None else float(row["value"])

    def ahead_drivers(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM ahead_drivers ORDER BY name").fetchall()

    def ensure_ahead_driver(self, name: str, status: str) -> None:
        self.conn.execute(
            """
            INSERT INTO ahead_drivers (name, status, contribution) VALUES (?, ?, 0)
            ON CONFLICT(name) DO NOTHING
            """,
            (name, status),
        )
        self.conn.commit()

    def set_ahead_driver(self, name: str, status: str, contribution: float) -> None:
        self.conn.execute(
            """
            INSERT INTO ahead_drivers (name, status, contribution) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                status = excluded.status,
                contribution = excluded.contribution
            """,
            (name, status, contribution),
        )
        self.conn.commit()

    def set_ahead_status(self, name: str, status: str) -> None:
        self.conn.execute("UPDATE ahead_drivers SET status = ? WHERE name = ?", (status, name))
        self.conn.commit()

    def save_evolve_bars(self, symbol: str, rows: list[tuple[str, float]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO evolve_bars (symbol, day, close) VALUES (?, ?, ?)
            ON CONFLICT(symbol, day) DO UPDATE SET close = excluded.close
            """,
            [(symbol, day, float(close)) for day, close in rows],
        )
        self.conn.commit()

    def evolve_bars(self, symbol: str) -> list[tuple[str, float]]:
        found = self.conn.execute(
            "SELECT day, close FROM evolve_bars WHERE symbol = ? ORDER BY day",
            (symbol,),
        ).fetchall()
        return [(row["day"], float(row["close"])) for row in found]

    def level_pairs(self, name: str) -> list[tuple[float, float]]:
        rows = self.conn.execute(
            "SELECT change, move FROM level_pairs WHERE name = ? ORDER BY id",
            (name,),
        ).fetchall()
        return [(float(row["change"]), float(row["move"])) for row in rows]

    def _migrate(self) -> None:
        self._migrate_pcr()
        self._migrate_columns("claims", {"horizon": "TEXT NOT NULL DEFAULT 'session'", "driver": "REAL"})
        self._migrate_columns("ensembles", {"horizon": "TEXT NOT NULL DEFAULT 'session'"})
        self._migrate_columns(
            "families",
            {
                "points_n": "INTEGER NOT NULL DEFAULT 0",
                "points_sum": "REAL NOT NULL DEFAULT 0",
                "inside_n": "INTEGER NOT NULL DEFAULT 0",
                "outside_n": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        self._migrate_columns("misses", {"kind": "TEXT NOT NULL DEFAULT ''", "sign": "INTEGER NOT NULL DEFAULT 0"})
        self._migrate_columns("family_pairs", {"clean": "INTEGER NOT NULL DEFAULT 0"})
        self._migrate_columns("ahead_calls", {"model_abs": "REAL", "base_abs": "REAL"})
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "issue" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN issue INTEGER NOT NULL DEFAULT 0")
        if "agent" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN agent TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _migrate_columns(self, table: str, columns: dict[str, str]) -> None:
        have = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.conn.commit()

    def _migrate_pcr(self) -> None:
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(pcr_snapshots)")}
        if "bucket" not in cols:
            self.conn.execute("ALTER TABLE pcr_snapshots ADD COLUMN bucket TEXT NOT NULL DEFAULT 'week'")
        if "oi_change" not in cols:
            self.conn.execute("ALTER TABLE pcr_snapshots ADD COLUMN oi_change REAL NOT NULL DEFAULT 0")
        self.conn.commit()

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()


def _domain(url: str) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return host.lower().split(":")[0]


def _source(row: sqlite3.Row) -> Source:
    return Source(
        id=row["id"],
        beat=row["beat"],
        kind=row["kind"],
        url=row["url"],
        status=row["status"],
        weight=row["weight"],
        n_scored=row["n_scored"],
        residual_sum=row["residual_sum"],
        residual_sq_sum=row["residual_sq_sum"],
        fail_streak=row["fail_streak"],
        domain=row["domain"],
    )


def _claim(row: sqlite3.Row) -> Claim:
    return Claim(
        id=row["id"],
        beat=row["beat"],
        source_id=row["source_id"],
        ts=row["ts"],
        headline=row["headline"],
        url=row["url"],
        shock_key=row["shock_key"],
        direction=row["direction"],
        points_low=row["points_low"],
        points_high=row["points_high"],
        midpoint=row["midpoint"],
        confidence=row["confidence"],
        nifty_level=row["nifty_level"],
        score_after=row["score_after"],
        status=row["status"],
        actual_move=row["actual_move"],
        residual=row["residual"],
        body=row["body"],
        ensemble_id=row["ensemble_id"],
        horizon=row["horizon"] if "horizon" in row.keys() else "session",
        driver=row["driver"] if "driver" in row.keys() else None,
    )


def _ensemble(row: sqlite3.Row) -> Ensemble:
    return Ensemble(
        id=row["id"],
        ts=row["ts"],
        mu=row["mu"],
        sigma=row["sigma"],
        n_shocks=row["n_shocks"],
        nifty_level=row["nifty_level"],
        actual=row["actual"],
        residual=row["residual"],
        horizon=row["horizon"] if "horizon" in row.keys() else "session",
    )
