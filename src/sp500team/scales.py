"""Index-point sizes. The freeze step overwrites RATIO and the measured priors.

Nifty's signed flat miss was 133 points. spx_flat is the mean absolute
next-close change on the last 80 fitted replay rows. Every index-point
constant is that Nifty number times spx_flat/133.
"""

from __future__ import annotations

NIFTY_FLAT = 133.0
# Frozen 2026-10-30. Mean absolute next-close change on the last 80 fitted
# rows, 2026-07-08 through 2026-10-28. Yahoo ^GSPC replay.
SPX_FLAT = 41.892279052734374
RATIO = SPX_FLAT / NIFTY_FLAT
# DX-Y.NYB daily bar clock was 00:00 New York, which is at or before 16:00.
DXY_SAME_DAY = True

# spx priors scale with RATIO. The rest are mean absolute daily changes on
# those same 80 rows.
PRIOR = {
    "spx1": 180.0 * RATIO,
    "spx5": 450.0 * RATIO,
    "europe": 38.5479,
    "dxy": 0.2321,
    "brent": 2.1858,
    "vix": 0.8461,
    "gold": 43.2125,
}


def pts(points: float) -> float:
    return points * RATIO


LIVE_GAP = pts(15)
EVOLVE_SIGMA_FLOOR = pts(40)
RESID_PRIOR = (200.0 * RATIO) ** 2
CAP = pts(400)
LEAN = pts(15)
QUIETER = pts(5)
MAE_STEP = pts(30)
NEWS_SIGMA_FLOOR = pts(30)
QUIET_SKIP = pts(40)
GIFT_IGNORE = pts(8)
GIFT_CLAMP = pts(150)
GIFT_HALF = pts(10)
LEVEL_CLAMP = pts(80)
LEVEL_MIN = pts(4)
LEVEL_HALF = pts(8)
CLAIM_CLAMP = pts(150)
FAMILY_CAP = pts(30)
AGREE = pts(8)
SCORE_FLOOR = pts(15)
SIGMA_CAP = pts(200)
SIGMA_FLOOR_MIN = pts(8)
CUSHION = pts(8)


def gap_text() -> str:
    value = LIVE_GAP
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


_OLD_TYPICAL = {"brent": 1.8, "dxy": 0.25, "gold": 8.0, "europe": 40.0, "vix": 1.2}
_LEVEL_BASE = {
    "brent": (-8.0, 0.4),
    "dxy": (-40.0, 0.08),
    "gold": (-0.12, 8.0),
    "europe": (0.35, 8.0),
    "vix": (-10.0, 0.3),
}


def level_rule(name: str) -> dict[str, float]:
    """Index-point impact scale, plus a threshold scaled only by the typical-change ratio."""
    scale, threshold = _LEVEL_BASE[name]
    old = _OLD_TYPICAL[name]
    new = PRIOR[name] if PRIOR[name] > 1e-8 else old
    return {
        "scale": scale * RATIO * (old / new),
        "threshold": threshold * (new / old),
    }


def center_sentence() -> str:
    return (
        "The middle stays on the last close until 60 new closes beat a no-change guess by "
        f"{gap_text()} points."
    )
