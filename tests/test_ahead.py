import json
import re
from datetime import date, datetime, timedelta

import pytest

from sp500team.ahead import (
    choose_published,
    consider_edit,
    improvement_note,
    judge_due,
    project,
    run_ahead,
    scored_samples,
    settle_phrases,
)
from sp500team.scout import QUERIES, gather_stories
from sp500team.chart import render_page
from sp500team.config import ET
from sp500team.horizons import decision_due
from sp500team.store import Store


def test_monday_due_is_tuesday_and_friday_due_is_monday():
    monday = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    friday = datetime(2026, 10, 16, 16, 35, tzinfo=ET)
    assert decision_due(monday) == datetime(2026, 10, 20, 16, 0, tzinfo=ET)
    assert decision_due(friday) == datetime(2026, 10, 19, 16, 0, tzinfo=ET)
    assert decision_due(datetime(2026, 10, 19, 16, 29, tzinfo=ET)) is None
    assert decision_due(datetime(2026, 10, 19, 16, 30, tzinfo=ET)) is not None
    assert decision_due(datetime(2026, 10, 18, 19, tzinfo=ET)) is None


def test_second_tick_does_not_insert_another_call(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    store.save_snapshots("2026-10-16", {"europe_px": 5000.0, "dxy_px": 104.0, "brent_px": 70.0})
    stamp = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    bars = _bars("2026-10-19", [100, 110, 120, 130, 140, 150, 160, 170])
    prices = {"europe": 5100.0, "dxy": 104.5, "brent": 72.0}
    fetches = {"n": 0}

    def fetch():
        fetches["n"] += 1
        return dict(prices)

    run_ahead(store, stamp, bars, fetch)
    run_ahead(store, stamp, bars, lambda: {"europe": 9999.0, "dxy": 1.0, "brent": 1.0})
    call = store.ahead_call("2026-10-19")
    assert fetches["n"] == 1
    assert call["due_day"] == "2026-10-20"
    assert call["anchor"] == 170
    assert call["predicted_move"] == 0
    assert call["baseline_move"] == 0
    assert store.snapshots("2026-10-19")["europe"] == pytest.approx(100)
    assert store.snapshots("2026-10-19")["spx5"] == pytest.approx(50)
    assert store.conn.execute("SELECT COUNT(*) AS n FROM ahead_calls").fetchone()["n"] == 1
    for table in ("claims", "ensembles", "levels", "level_pairs", "family_pairs"):
        assert store.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 0
    store.close()


def test_missing_due_date_is_discarded_and_not_trained(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    store.add_ahead_call("2026-10-01", 100, "2026-10-02", 0, 1, 0, "[]")
    store.finish_ahead("2026-10-01", 5, "scored")
    store.save_snapshots("2026-10-01", {"spx5": 1.0})
    store.add_ahead_call("2026-10-19", 100, "2026-10-20", 0, 0, 0, "[]")
    bars = [("2026-10-19", 100.0), ("2026-10-21", 130.0)]
    stamp = datetime(2026, 10, 21, 16, 35, tzinfo=ET)
    run_ahead(store, stamp, bars, None)
    assert store.ahead_call("2026-10-19")["status"] == "discarded"
    assert store.ahead_call("2026-10-19")["actual_move"] is None
    assert [sample["day"] for sample in scored_samples(store)] == ["2026-10-01"]
    store.close()


def test_a_due_bar_that_has_aged_out_stays_open(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    store.add_ahead_call("2026-10-19", 100, "2026-10-20", 0, 0, 0, "[]")
    bars = [("2026-11-02", 100.0), ("2026-11-03", 101.0)]
    run_ahead(store, datetime(2026, 11, 3, 16, 35, tzinfo=ET), bars, None)
    assert store.ahead_call("2026-10-19")["status"] == "open"
    assert judge_due([("2026-10-19", 90.0)], "2026-10-20", datetime(2026, 10, 19, 16, 35, tzinfo=ET))[0] == "wait"
    store.close()


def test_walk_forward_does_not_train_on_its_own_day():
    rows = [([float(i)], float(i) * 2) for i in range(1, 15)]
    first, _contribs, _intercept = project(rows[:-1], rows[-1][0])
    rows[-1] = (rows[-1][0], 9999.0)
    second, _contribs, _intercept = project(rows[:-1], rows[-1][0])
    assert first == pytest.approx(second)
    moved, _contribs, _intercept = project(rows, rows[-1][0])
    assert moved != pytest.approx(first)


def test_published_move_stays_zero_until_the_untouched_window_wins():
    short = [_scored(i, train_n=30, actual=10, model=10) for i in range(1, 20)]
    assert choose_published(short, 4) is None
    young = [_scored(i, train_n=29, actual=10, model=10) for i in range(1, 21)]
    assert choose_published(young, 4) is None
    tied = [_scored(i, train_n=30, actual=10, model=0) for i in range(1, 21)]
    assert choose_published(tied, 4) is None
    ahead = [_scored(i, train_n=30, actual=10, model=8) for i in range(1, 21)]
    assert choose_published(ahead, 4) == 4
    frozen = ahead[:]
    frozen[0] = _scored(1, train_n=30, actual=10, model=8, day="2025-12-01")
    assert choose_published(frozen, 4, frozen[-1]["day"]) is None


def test_point_contributions_equal_the_predicted_move():
    history = [([1.0, 0.0], 10.0), ([2.0, 0.0], 20.0), ([3.0, 0.0], 30.0), ([4.0, 0.0], 40.0)]
    pred, contribs, intercept = project(history, [5.0, 1.0])
    assert pred == pytest.approx(sum(contribs) + intercept)
    page = render_page(
        None,
        [],
        [],
        None,
        0,
        ahead={"move": 0.8, "parts": [("Five sessions", 0.4), ("Base", 0.4)], "due": "2026-10-20"},
    )
    shown = [int(item) for item in re.findall(r"ahead-point[^>]*>([+-]?\d+)", page)]
    total = int(re.search(r"data-ahead='(-?\d+)'", page).group(1))
    assert sum(shown) == total == 1
    assert "Already in the move" in page
    assert "<h2>Next close" not in page
    assert "Due 2026-10-20" in page


def test_a_live_call_shows_parts_that_sum_to_the_move(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    for index in range(50):
        day = (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        store.add_ahead_call(day, 1000, day, 2.0 * (index + 1), 2.0 * (index + 1), 30, "[]")
        store.finish_ahead(day, 2.0 * (index + 1), "scored")
        store.save_snapshots(day, {"spx5": float(index + 1), "europe": 0.0, "dxy": 0.0, "brent": 0.0})
    store.save_snapshots("2026-10-16", {"europe_px": 10.0, "dxy_px": 80.0, "brent_px": 70.0})
    store.set_meta("ahead_edit_day", "2026-10-19")
    stamp = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    bars = _bars("2026-10-19", [100, 120, 140, 160, 180, 200, 240])
    run_ahead(
        store,
        stamp,
        bars,
        lambda: {"europe": 10.0, "dxy": 80.0, "brent": 70.0},
    )
    call = store.ahead_call("2026-10-19")
    parts = json.loads(call["parts"])
    assert call["predicted_move"] == pytest.approx(call["model_move"])
    assert call["predicted_move"] == pytest.approx(sum(value for _label, value in parts))
    assert abs(call["predicted_move"]) > 1
    store.close()


def test_driver_edits_use_days_outside_the_latest_window():
    samples = []
    for index in range(40):
        day = (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        samples.append(
            {
                "day": day,
                "values": {"signal": float(index + 1), "noise": 0.0},
                "actual": 80.0 * (index + 1),
            }
        )
    kept, probation, changed = consider_edit(samples, ["signal", "noise"], [])
    assert changed == "noise"
    assert kept == ["signal"]
    assert probation == ["noise"]
    promoted, _probation, added = consider_edit(samples, ["noise"], ["signal"])
    assert added == "signal"
    assert promoted == ["noise", "signal"]
    hidden = []
    for index in range(40):
        day = (date(2026, 3, 1) + timedelta(days=index)).isoformat()
        if index < 20:
            values = {"noise": 1.0, "signal": 0.0}
            actual = 5.0
        else:
            values = {"noise": 1.0, "signal": float(index)}
            actual = 80.0 * index
        hidden.append({"day": day, "values": values, "actual": actual})
    kept_hidden, _probation, added = consider_edit(hidden, ["noise"], ["signal"])
    assert "signal" not in kept_hidden
    assert added != "signal"
    _kept, _probation, changed = consider_edit(samples, ["signal", "noise"], [], samples[-5]["day"])
    assert changed is None


def test_every_query_keeps_eight_stories():
    def fetch(query):
        return [{"title": f"{query} {index}"} for index in range(9)]

    titles = gather_stories(fetch)
    assert len(titles) == 8 * len(QUERIES)
    assert titles[0] == f"{QUERIES[0]} 0"
    assert f"{QUERIES[0]} 8" not in titles


def test_a_missing_prior_price_still_stores_the_extra_readings(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    store.add_pcr("2026-10-19T15:20:00-04:00", "index", 1.1, 0.0, "2026-10-19", "day")
    stamp = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    run_ahead(
        store,
        stamp,
        _bars("2026-10-19", [100, 110, 120, 130, 140, 150, 160, 170]),
        lambda: {"europe": 5100.0, "dxy": 104.5, "brent": 72.0, "gold": 2000.0, "vix": 14.0, "es": 172.0},
        lambda: ["Crude supply rises sharply"],
    )
    saved = store.snapshots("2026-10-19")
    assert store.ahead_call("2026-10-19") is None
    assert saved["gold_px"] == 2000.0
    assert saved["vix_px"] == 14.0
    assert saved["es"] == 2.0
    assert saved["pcr_px"] == 1.1
    assert store.topic("crude supply rises")["direction"] == 1
    store.close()


def test_a_new_phrase_has_direction_and_no_points(tmp_path):
    store = _ready(tmp_path)
    stamp = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    run_ahead(
        store,
        stamp,
        _bars("2026-10-19", [100, 110, 120, 130, 140, 150, 160, 170]),
        lambda: {"europe": 5100.0, "dxy": 104.5, "brent": 72.0},
        lambda: ["Crude supply rises sharply", "Cabinet meets on Tuesday"],
    )
    assert store.topic("crude supply rises")["direction"] == 1
    assert "topic:crude supply rises" not in store.snapshots("2026-10-19")
    assert store.ahead_call("2026-10-19")["predicted_move"] == 0
    assert json.loads(store.ahead_call("2026-10-19")["parts"]) == [
        ["Five sessions", 0.0],
        ["Europe", 0.0],
        ["Dollar", 0.0],
        ["Brent", 0.0],
        ["Base", 0.0],
    ]
    store.close()


def test_the_same_phrase_updates_the_same_topic(tmp_path):
    store = _ready(tmp_path)
    titles = lambda: ["Crude supply rises sharply"]
    bare = lambda: {"europe": 5100.0, "dxy": 104.5, "brent": 72.0}
    with_gold = lambda: {"europe": 5100.0, "dxy": 85.0, "brent": 72.0, "gold": 2000.0}
    run_ahead(store, datetime(2026, 10, 19, 16, 35, tzinfo=ET), _bars("2026-10-19", [100, 110, 120, 130, 140, 150, 160, 170]), bare, titles)
    assert store.ahead_call("2026-10-19") is not None
    assert "gold" not in store.snapshots("2026-10-19")
    run_ahead(store, datetime(2026, 10, 20, 16, 35, tzinfo=ET), _bars("2026-10-20", [100, 110, 120, 130, 140, 150, 160, 180]), with_gold, titles)
    assert store.conn.execute("SELECT COUNT(*) AS n FROM ahead_topics").fetchone()["n"] == 1
    assert store.snapshots("2026-10-20")["topic:crude supply rises"] == 1
    assert store.conn.execute(
        "SELECT status FROM ahead_drivers WHERE name = ?",
        ("topic:crude supply rises",),
    ).fetchone()["status"] == "probation"
    assert "gold" in store.snapshots("2026-10-20")
    store.close()


def test_three_misses_in_five_fade_the_phrase(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    phrase = "crude supply rises"
    store.ensure_topic(phrase, 1)
    for index, actual in enumerate((10, -10, -10, 10, -10)):
        day = (date(2026, 10, 1) + timedelta(days=index)).isoformat()
        store.save_topic_day(day, phrase, 1)
        settle_phrases(store, day, actual)
    topic = store.topic(phrase)
    assert topic["alive"] == 0
    assert topic["direction"] == 0
    assert topic["misses"] == 3
    store.close()


def test_a_repeated_phrase_joins_only_through_the_error_test(tmp_path):
    store = _ready(tmp_path)
    name = "topic:crude supply rises"
    for index in range(40):
        day = (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        signal = 1.0 if index % 2 == 0 else -1.0
        store.add_ahead_call(day, 1000, day, 0, 80.0 * signal, 30, "[]")
        store.finish_ahead(day, 80.0 * signal, "scored")
        store.save_snapshots(
            day,
            {"spx5": 0.0, "europe": 0.0, "dxy": 0.0, "brent": 0.0, name: signal},
        )
        store.save_topic_day(day, "crude supply rises", int(signal))
    store.ensure_topic("crude supply rises", 1)
    stamp = datetime(2026, 10, 19, 16, 35, tzinfo=ET)
    run_ahead(
        store,
        stamp,
        _bars("2026-10-19", [100, 110, 120, 130, 140, 150, 160, 170]),
        lambda: {"europe": 5100.0, "dxy": 104.5, "brent": 72.0},
        lambda: ["Crude supply rises sharply"],
    )
    call = store.ahead_call("2026-10-19")
    assert call["predicted_move"] == 0
    assert store.conn.execute(
        "SELECT status FROM ahead_drivers WHERE name = ?",
        (name,),
    ).fetchone()["status"] == "kept"
    store.close()


def test_the_page_names_a_worse_window(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    for index in range(40):
        day = (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        model = 0 if index < 20 else 9
        store.add_ahead_call(day, 1000, day, 0, model, 30, "[]")
        store.finish_ahead(day, 10, "scored")
    note = improvement_note(store)
    assert note == "Recent miss 1. Earlier miss 10."
    assert "Better than" not in note
    page = render_page(None, [], [], None, 0, ahead={"move": 0, "parts": [], "due": "2026-10-20", "note": note})
    assert "Better than the previous 20" not in page
    assert "<h2>Next close" not in page
    assert "Recent miss 1" in page
    store.close()


def _ready(tmp_path):
    store = Store(tmp_path / "ahead.sqlite")
    store.save_snapshots("2026-10-16", {"europe_px": 5000.0, "dxy_px": 104.0, "brent_px": 70.0, "gold_px": 1900.0})
    return store


def _scored(index: int, train_n: int, actual: float, model: float, day: str | None = None) -> dict:
    return {
        "day": day or (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
        "actual": actual,
        "model_move": model,
        "train_n": train_n,
    }


def _bars(end: str, closes: list[float]) -> list[tuple[str, float]]:
    day = date.fromisoformat(end)
    found = []
    for close in reversed(closes):
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        found.append((day.isoformat(), float(close)))
        day -= timedelta(days=1)
    return list(reversed(found))
