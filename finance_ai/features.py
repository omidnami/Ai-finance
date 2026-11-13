"""Shared CSV/Excel -> causal features pipeline for training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TIMEFRAME_MINUTES = 15
LOOKBACK = 20
FEATURE_NAMES = (
    "open_return", "high_return", "low_return", "close_return", "log_volume",
    "macd_ratio", "macd_signal_ratio", "macd_hist_ratio", "rsi14",
    "sma20_ratio", "sma50_ratio", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "month_early", "month_middle", "month_late",
    "asia_session", "europe_session", "us_session", "europe_us_overlap",
    "log_gap", "month_day",
)
assert len(FEATURE_NAMES) == 24


@dataclass(frozen=True)
class PreparedData:
    timestamps: np.ndarray
    features: np.ndarray
    closes: np.ndarray


def _sessions(timestamp: pd.Timestamp) -> tuple[float, float, float]:
    # These are configurable *proxies* for regional trading hours, not exchange calendars.
    # ZoneInfo correctly accounts for regional DST transitions.
    def between(zone: str, start: int, end: int) -> float:
        local = timestamp.tz_convert(ZoneInfo(zone))
        return float(start <= local.hour < end and local.weekday() < 5)

    return (between("Asia/Tokyo", 9, 18),
            between("Europe/London", 8, 17),
            between("America/New_York", 8, 17))


def read_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # MetaTrader exports can be comma- or tab-separated despite the .csv suffix.
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Expected a MetaTrader .csv, .xlsx or .xls file")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"time", "open", "high", "low", "close", "volume", "timeframe"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[list(("time", "open", "high", "low", "close", "volume", "timeframe"))].copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError("Input contains missing or non-finite values")
    if not (frame["timeframe"] == TIMEFRAME_MINUTES).all():
        raise ValueError("v0.1 requires timeframe=15 on every row")
    if (frame["time"] % 1 != 0).any() or (frame["time"] <= 0).any():
        raise ValueError("time must contain positive Unix timestamps in seconds")
    if (frame["time"] % (TIMEFRAME_MINUTES * 60) != 0).any():
        raise ValueError("Timestamps must align with 15-minute UTC boundaries")
    if not frame["time"].is_monotonic_increasing or frame["time"].duplicated().any():
        raise ValueError("Candles must be strictly time-ordered with unique timestamps")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
            (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC values")
    return frame.reset_index(drop=True)


def prepare(path: str | Path) -> PreparedData:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, SMAIndicator

    df = read_candles(path)
    if len(df) < 100:
        raise ValueError("At least 100 candles needed for warmup and evaluation")
    close = df["close"].astype(float)
    prev = close.shift(1)
    macd = MACD(close, window_fast=12, window_slow=26, window_sign=9, fillna=False)
    rsi = RSIIndicator(close, window=14, fillna=False).rsi()
    sma20 = SMAIndicator(close, window=20, fillna=False).sma_indicator()
    sma50 = SMAIndicator(close, window=50, fillna=False).sma_indicator()
    time = pd.to_datetime(df["time"], unit="s", utc=True)
    hours = time.dt.hour + time.dt.minute / 60.0
    weekdays = time.dt.weekday
    days = time.dt.day
    sessions = np.array([_sessions(t) for t in time], dtype=np.float64)
    gap = df["time"].diff() / (TIMEFRAME_MINUTES * 60)
    f = pd.DataFrame(index=df.index)
    for c in ("open", "high", "low", "close"):
        f[c + "_return"] = df[c] / prev - 1.0
    f["log_volume"] = np.log1p(df["volume"])
    f["macd_ratio"] = macd.macd() / prev
    f["macd_signal_ratio"] = macd.macd_signal() / prev
    f["macd_hist_ratio"] = macd.macd_diff() / prev
    f["rsi14"] = rsi / 100.0
    f["sma20_ratio"] = sma20 / prev - 1.0
    f["sma50_ratio"] = sma50 / prev - 1.0
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    f["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    f["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
    f["month_early"] = (days <= 10).astype(float)
    f["month_middle"] = ((days >= 11) & (days <= 20)).astype(float)
    f["month_late"] = (days >= 21).astype(float)
    f["asia_session"] = sessions[:, 0]
    f["europe_session"] = sessions[:, 1]
    f["us_session"] = sessions[:, 2]
    f["europe_us_overlap"] = sessions[:, 1] * sessions[:, 2]
    f["log_gap"] = np.log1p(gap)
    f["month_day"] = (days - 1) / 30.0
    f = f[list(FEATURE_NAMES)]
    # Drop only the warmup at the start; never remove a gap or fabricate candles.
    valid = np.isfinite(f.to_numpy(dtype=float)).all(axis=1)
    first = int(np.argmax(valid))
    if not valid.any() or not valid[first:].all():
        raise ValueError("Indicator data contains unexpected missing values")
    return PreparedData(
        timestamps=df["time"].to_numpy(dtype=np.int64)[first:],
        features=f.to_numpy(dtype=np.float64)[first:],
        closes=close.to_numpy(dtype=np.float64)[first:],
    )


def labeled_windows(data: PreparedData, lookback: int = LOOKBACK):
    """Each sample ends at t and predicts close(t+1) > close(t)."""
    if len(data.closes) < lookback + 2:
        raise ValueError("Not enough usable candles to generate labeled windows")
    x, y, target_times = [], [], []
    for end in range(lookback - 1, len(data.closes) - 1):
        x.append(data.features[end - lookback + 1:end + 1])
        y.append(float(data.closes[end + 1] > data.closes[end]))
        target_times.append(data.timestamps[end + 1])
    return np.asarray(x), np.asarray(y), np.asarray(target_times)


def latest_window(data: PreparedData, lookback: int = LOOKBACK) -> np.ndarray:
    if len(data.features) < lookback:
        raise ValueError("Not enough completed candles for inference")
    return data.features[-lookback:]
