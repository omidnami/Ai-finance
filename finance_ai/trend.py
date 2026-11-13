"""Five causal, end-of-window trend features for Finance AI v0.2."""
from __future__ import annotations

import numpy as np

TREND_FEATURE_NAMES = (
    "trend_5", "trend_10", "trend_20", "trend_strength", "atr14_ratio",
)


def summary_at(closes: np.ndarray, atr14_ratio: np.ndarray, end: int) -> np.ndarray:
    """Summarize the 20 completed candles ending at `end` (inclusive).

    5/10 bar returns compare against the close five/ten bars earlier;
    the 20-candle return compares the first and last of these 20 candles.
    Trend strength is the unsigned efficiency ratio; direction is in trend_20.
    ATR14 is Wilder's ATR divided by the last observed close.
    """
    if end < 19 or end >= len(closes) or len(atr14_ratio) != len(closes):
        raise ValueError("Expected at least 20 completed candles and aligned ATR values")
    window = closes[end - 19 : end + 1]
    moves = np.diff(window)
    gross = np.abs(moves).sum()
    strength = (abs(window[-1] - window[0]) / gross) if gross > 0 else 0.0
    result = np.array([
        window[-1] / window[-6] - 1.0,
        window[-1] / window[-11] - 1.0,
        window[-1] / window[0] - 1.0,
        strength,
        atr14_ratio[end],
    ], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite trend feature")
    return result
