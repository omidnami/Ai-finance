"""Fixed-distance, wick-triggered paper TP/SL review of existing AI signals.

Entry is the signal candle CLOSE (hypothetical fill after confirmation).
Only future candle HIGH/LOW may hit TP/SL. Every signal is evaluated
independently; overlapping outcomes are not a realizable portfolio.
"""
from __future__ import annotations

import csv as csv_module
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import read_candles


DEFAULT_TP = 2.5  # XAUUSD quote-price USD per troy ounce
DEFAULT_SL = 2.0


def _validated_report(path: str, raw: pd.DataFrame) -> pd.DataFrame:
    report = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if report.empty:
        raise ValueError("Signal report is empty")
    if "candle_utc" in report:
        time_col, source = "candle_utc", "v0.9"
        required = {"candidate", "quality_reason", "decision"}
        if not required.issubset(report):
            raise ValueError(f"v0.9 signal report requires {sorted(required)}")
        allowed = {"none", "BUY", "SELL"}
        if not set(report["candidate"].dropna()).issubset(allowed):
            raise ValueError("Unexpected candidate direction")
    elif "window_end_utc" in report:
        time_col, source = "window_end_utc", "v0.8"
        required = {"new_signal", "signal_candle_close"}
        if not required.issubset(report):
            raise ValueError(f"v0.8 signal report requires {sorted(required)}")
        allowed = {"none", "BUY", "SELL"}
        if not set(report["new_signal"].dropna()).issubset(allowed):
            raise ValueError("Unexpected v0.8 new_signal direction")
    else:
        raise ValueError("Expected v0.9 candle_utc or v0.8 window_end_utc report")
    timestamps = pd.to_datetime(report[time_col], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Signal report timestamps must be unique and chronologically ordered")
    ns = timestamps.astype("int64").to_numpy()
    if np.any(ns % 1_000_000_000):
        raise ValueError("Signal timestamps must have whole-second precision")
    raw_times = raw["time"].to_numpy(dtype=np.int64)
    times = ns // 1_000_000_000
    idx = np.searchsorted(raw_times, times)
    if np.any(idx >= len(raw_times)) or not np.array_equal(raw_times[idx], times):
        raise ValueError("Signal timestamps do not match OHLC candle file; use the same dataset")
    if len(idx) > 1 and not np.all(np.diff(idx) == 1):
        raise ValueError("Signal report must have consecutive one-candle steps (no missing bars)")
    if source == "v0.8":
        src_close = pd.to_numeric(report["signal_candle_close"], errors="coerce").to_numpy()
        raw_close = raw["close"].to_numpy(dtype=float)[idx]
        if not np.isfinite(src_close).all() or not np.allclose(src_close, raw_close, atol=1e-7, rtol=0):
            raise ValueError("Signal candle closes differ from OHLC data: reports are not from the same dataset")
    report = report.copy()
    report["_raw_index"] = idx
    report["_source"] = source
    return report


def _simulate(raw: pd.DataFrame, i: int, side: str, tp: float, sl: float,
              max_holding_bars: int | None) -> dict:
    d = 1 if side == "BUY" else -1
    entry = float(raw.iloc[i]["close"])
    take = entry + d * tp
    stop = entry - d * sl
    last = len(raw) - 1
    if max_holding_bars is not None:
        last = min(last, i + max_holding_bars)
    checked = max(0, last - i)
    result = "PENDING_END" if max_holding_bars is None or i + max_holding_bars > len(raw) - 1 else "NO_HIT"
    exit_index = None
    mfe = 0.0
    mae = 0.0
    highs = raw["high"].to_numpy(dtype=float)
    lows = raw["low"].to_numpy(dtype=float)
    for j in range(i + 1, last + 1):
        favorable = (highs[j] - entry) if d == 1 else (entry - lows[j])
        adverse = (entry - lows[j]) if d == 1 else (highs[j] - entry)
        mfe = max(mfe, float(favorable))
        mae = max(mae, float(adverse))
        hit_take = highs[j] >= take if d == 1 else lows[j] <= take
        hit_stop = lows[j] <= stop if d == 1 else highs[j] >= stop
        if hit_take or hit_stop:
            exit_index = j
            checked = j - i
            result = ("AMBIGUOUS_SL_FIRST" if hit_take and hit_stop else
                      "TP" if hit_take else "SL")
            break
    if exit_index is None and checked == 0:
        result = "PENDING_END"
    signed_move = (tp if result == "TP" else -sl if result in {"SL", "AMBIGUOUS_SL_FIRST"} else None)
    return {"entry_close_price": entry, "tp_price": take, "sl_price": stop,
            "exit_index": exit_index, "exit_candle_utc": (
                pd.Timestamp(int(raw.iloc[exit_index]["time"]), unit="s", tz="UTC").isoformat()
                if exit_index is not None else ""),
            "bars_checked": checked, "result": result, "gross_price_move_usd_per_oz": signed_move,
            "mfe_usd_per_oz_until_exit": mfe, "mae_usd_per_oz_until_exit": mae,
            "last_checked_candle_utc": (
                pd.Timestamp(int(raw.iloc[i + checked]["time"]), unit="s", tz="UTC").isoformat()
                if checked else "")}



def _read_m1_candles(path: str) -> pd.DataFrame:
    """Load same-broker, UTC minute candles; never reuse the M15-only feature loader."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        frame = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(p)
    else:
        raise ValueError("M1 candles must be .csv, .xlsx or .xls")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"time", "open", "high", "low", "close", "timeframe"}
    if not required.issubset(frame):
        raise ValueError(f"M1 candles require columns {sorted(required)}")
    frame = frame[list(("time", "open", "high", "low", "close", "timeframe"))].copy()
    for col in frame:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame.empty or frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("M1 data cannot be empty or contain invalid numbers")
    if not (frame["timeframe"] == 1).all():
        raise ValueError("M1 candle timeframe must equal 1 minute")
    times = frame["time"].to_numpy(dtype=float)
    if (times <= 0).any() or (times % 60 != 0).any() or not np.all(np.diff(times) > 0):
        raise ValueError("M1 timestamps must be unique, strictly ascending UTC epoch seconds on minute boundaries")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("M1 prices must be positive")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
            (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("M1 OHLC prices are inconsistent")
    return frame.reset_index(drop=True)


def _refine_ambiguous_with_m1(raw: pd.DataFrame, m1: pd.DataFrame, outcome: dict,
                              side: str, tolerance: float) -> dict:
    """Use the 15 minute bars inside the FIRST ambiguous M15 candle to find first hit.

    A nonambiguous M15 candle already proves the winner and is left alone.
    Require all 15 M1 bars *and* matching aggregate M15 OHLC before resolving
    an ambiguous candle; otherwise exclude it from the measured win rate.
    """
    if outcome["result"] != "AMBIGUOUS_SL_FIRST":
        return {"m15_result_before_m1": outcome["result"],
                "m1_resolution": "M15_UNAMBIGUOUS_OR_NOT_HIT",
                "m1_first_touch_utc": "", "m1_bars_in_ambiguous_candle": ""}
    j = outcome["exit_index"]
    m15_bar = raw.iloc[j]
    begin = int(m15_bar["time"])
    times = m1["time"].to_numpy(dtype=np.int64)
    start = int(np.searchsorted(times, begin))
    stop = int(np.searchsorted(times, begin + 900))
    sub = m1.iloc[start:stop]
    original = outcome["result"]
    info = {"m15_result_before_m1": original, "m1_first_touch_utc": "",
            "m1_bars_in_ambiguous_candle": int(len(sub))}
    if len(sub) != 15 or not np.array_equal(sub["time"].to_numpy(dtype=np.int64),
                                             np.arange(begin, begin + 900, 60, dtype=np.int64)):
        outcome["result"] = "M1_DATA_INCOMPLETE"
        outcome["gross_price_move_usd_per_oz"] = None
        info["m1_resolution"] = "M1_DATA_INCOMPLETE"
        return info
    aggregate = np.array([float(sub.iloc[0]["open"]), float(sub["high"].max()),
                          float(sub["low"].min()), float(sub.iloc[-1]["close"])])
    m15_ohlc = m15_bar[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.allclose(aggregate, m15_ohlc, atol=tolerance, rtol=0):
        outcome["result"] = "M1_PRICE_MISMATCH"
        outcome["gross_price_move_usd_per_oz"] = None
        info["m1_resolution"] = "M1_PRICE_MISMATCH"
        return info
    d = 1 if side == "BUY" else -1
    tp_price = outcome["tp_price"]
    sl_price = outcome["sl_price"]
    for _, minute in sub.iterrows():
        hit_tp = (minute["high"] >= tp_price) if d == 1 else (minute["low"] <= tp_price)
        hit_sl = (minute["low"] <= sl_price) if d == 1 else (minute["high"] >= sl_price)
        if not (hit_tp or hit_sl):
            continue
        result = "AMBIGUOUS_SL_FIRST" if hit_tp and hit_sl else "TP" if hit_tp else "SL"
        outcome["result"] = result
        outcome["gross_price_move_usd_per_oz"] = (float(abs(tp_price - outcome["entry_close_price"]))
                                                  if result == "TP" else
                                                  -float(abs(sl_price - outcome["entry_close_price"])))
        info["m1_first_touch_utc"] = pd.Timestamp(int(minute["time"]), unit="s", tz="UTC").isoformat()
        info["m1_resolution"] = ("M1_STILL_AMBIGUOUS" if hit_tp and hit_sl else "M1_RESOLVED")
        return info
    # Matching M15 ranges guarantee at least one level was hit; this indicates inconsistency.
    outcome["result"] = "M1_PRICE_MISMATCH"
    outcome["gross_price_move_usd_per_oz"] = None
    info["m1_resolution"] = "M1_PRICE_MISMATCH"
    return info

def _statistics(rows: list[dict]) -> dict:
    def stats(parts: list[dict]) -> dict:
        wins = sum(r["result"] == "TP" for r in parts)
        stops = sum(r["result"] == "SL" for r in parts)
        ambiguous = sum(r["result"] == "AMBIGUOUS_SL_FIRST" for r in parts)
        pending = sum(r["result"] == "PENDING_END" for r in parts)
        no_hit = sum(r["result"] == "NO_HIT" for r in parts)
        undecidable_m1 = sum(r["result"] in {"M1_DATA_INCOMPLETE", "M1_PRICE_MISMATCH"} for r in parts)
        decided = wins + stops + ambiguous
        return {"signals": len(parts), "tp_hits": wins, "sl_hits": stops,
                "ambiguous_both_hit_same_bar_counted_as_loss": ambiguous,
                "no_hit_within_horizon": no_hit, "pending_at_file_end": pending,
                "m1_unresolved_missing_or_mismatch": undecidable_m1,
                "m15_originally_ambiguous": sum(r.get("m15_result_before_m1") == "AMBIGUOUS_SL_FIRST" for r in parts),
                "m1_resolved": sum(r.get("m1_resolution") == "M1_RESOLVED" for r in parts),
                "resolved": decided,
                "conservative_win_rate_resolved": wins / decided if decided else None,
                "resolution_rate": decided / len(parts) if parts else None}
    def group(parts):
        out = stats(parts)
        out["gross_sum_usd_per_oz_resolved"] = round(sum(
            r["gross_price_move_usd_per_oz"] for r in parts
            if r["gross_price_move_usd_per_oz"] is not None), 8)
        return out
    return {"overall": group(rows), "BUY": group([r for r in rows if r["direction"] == "BUY"]),
            "SELL": group([r for r in rows if r["direction"] == "SELL"])}


def run_fixed_tpsl(csv: str, signals: str, out: str, *, tp: float = DEFAULT_TP,
                   sl: float = DEFAULT_SL, max_holding_bars: int | None = None,
                   m1_csv: str | None = None, m1_price_tolerance: float = 0.02,
                   echo: bool = True) -> dict:
    """Replay signals from a saved per-candle v0.8/v0.9 report on matched OHLC data.

    - Entry = close of confirmation candle t (paper fill, no next-open slip).
    - Only future bars t+1.. use HIGH/LOW (never closing prices).
    - If both levels touched in one bar, ambiguous; counted LOSS in conservative rate.
    - Each signal independently tracked even if positions overlap.
    """
    if any(not np.isfinite(x) or x <= 0 for x in (tp, sl)):
        raise ValueError("TP and SL must be finite positive quote-price distances")
    if max_holding_bars is not None and (
            isinstance(max_holding_bars, bool) or not isinstance(max_holding_bars, int)
            or max_holding_bars <= 0):
        raise ValueError("max_holding_bars must be a positive integer or unset")
    if not np.isfinite(m1_price_tolerance) or m1_price_tolerance < 0:
        raise ValueError("M1 price tolerance must be finite and non-negative")
    outpath = Path(out)
    if outpath.suffix.lower() != ".csv":
        raise ValueError("Output must be a CSV file")
    raw = read_candles(csv)
    report = _validated_report(signals, raw)
    m1 = _read_m1_candles(m1_csv) if m1_csv else None
    source = report.iloc[0]["_source"]
    rows = []
    for _, item in report.iterrows():
        side = str(item["candidate"] if source == "v0.9" else item["new_signal"])
        if side not in {"BUY", "SELL"}:
            continue
        i = int(item["_raw_index"])
        quality_reason = str(item["quality_reason"]) if source == "v0.9" else "n/a"
        decision = str(item["decision"]) if source == "v0.9" else "n/a"
        qualified = quality_reason == "accepted" if source == "v0.9" else True
        queued = decision == "QUEUE_" + side if source == "v0.9" else False
        outcome = _simulate(raw, i, side, float(tp), float(sl), max_holding_bars)
        m1_info = (_refine_ambiguous_with_m1(raw, m1, outcome, side, m1_price_tolerance)
                   if m1 is not None else {})
        row = {"signal_candle_utc": pd.Timestamp(int(raw.iloc[i]["time"]), unit="s", tz="UTC").isoformat(),
               "direction": side, "raw_candle_index": i,
               "market_cycle": str(item["market_cycle"]) if source == "v0.9" else "",
               "quality_score": (item["quality_score"] if source == "v0.9" else ""),
               "quality_reason": quality_reason, "v09_decision": decision,
               "qualified_setup": qualified, "queued_in_prior_v09": queued,
               "tp_distance_usd_per_oz": float(tp), "sl_distance_usd_per_oz": float(sl),
               **outcome, **m1_info}
        rows.append(row)
    if not rows:
        raise ValueError("Signal report contains no BUY/SELL candidates")
    subsets = {"all_matched_signals": rows,
               "qualified_setups": [r for r in rows if r["qualified_setup"]],
               "queued_v09_setups": [r for r in rows if r["queued_in_prior_v09"]]}
    summary = {"task": "fixed_tp_sl_at_signal_close_wick_first_touch",
               "ohlc_data": str(csv), "signal_report": str(signals), "source": source,
               "entry_price": "CLOSE of completed signal candle t (hypothetical close fill)",
               "first_exitable_bar": "t+1; no future data used to create signals",
               "tp_distance_usd_per_oz": float(tp), "sl_distance_usd_per_oz": float(sl),
               "max_holding_bars": max_holding_bars,
               "m1_candles_file": str(m1_csv) if m1_csv else None,
               "m1_refines_only_m15_ambiguous_first_hit": m1 is not None,
               "m1_price_tolerance_usd_per_oz": m1_price_tolerance if m1 is not None else None,
               "summary_by_selection": {k: _statistics(v) for k, v in subsets.items()},
               "important": [
                   "Price level hit determined only by subsequent OHLC HIGH/LOW wicks; never signal-bar wicks or subsequent closes.",
                   "If TP and SL were both within one candle range, path is unobservable in OHLC; counted as SL for conservative win rate and flagged separately.",
                   "With M1 provided, only original M15 ambiguous cases are refined; original unambiguous outcomes stay unchanged. M1 itself can remain ambiguous.",
                   "To trust M1 resolution, all 15 minute bars and M1-aggregated OHLC matching the M15 candle are required; incomplete or mismatched cases are EXCLUDED from resolved win rate.",
                   "All matched signals are evaluated independently; overlap is allowed, so gross sums are not realized portfolio P&L.",
                   "NO_HIT (if a horizon is specified) and PENDING_END are not wins or losses and are excluded from the resolved win rate.",
                   "No spread, transaction fees, slippage, gap execution adjustment, broker fill, lot sizing or actual dollar P&L.",
                   "This uses already-generated past-only model signals; the prior paper stop-entry and dynamic TP/SL rules are NOT used.",
                   "The v0.8 reference is also its training target, not independent confirmation; final_test.csv already used in research."]}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_file = outpath.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if echo:
        for label in ("all_matched_signals", "qualified_setups", "queued_v09_setups"):
            s = summary["summary_by_selection"][label]["overall"]
            print(f"{label}: {s['signals']} signals | TP={s['tp_hits']} SL={s['sl_hits']} "
                  f"AMBIGUOUS={s['ambiguous_both_hit_same_bar_counted_as_loss']} "
                  f"M1_RESOLVED={s['m1_resolved']} M1_UNRESOLVED={s['m1_unresolved_missing_or_mismatch']} "
                  f"NO_HIT={s['no_hit_within_horizon']} PENDING={s['pending_at_file_end']} "
                  f"conservative resolved win rate={s['conservative_win_rate_resolved']}")
        print(f"Excel-ready CSV: {outpath} | Summary: {summary_file}")
    return {"rows": rows, "summary": summary, "csv": str(outpath), "summary_json": str(summary_file)}
