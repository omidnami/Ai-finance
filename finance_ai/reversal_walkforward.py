"""Batch-by-batch, chronological v0.8 signal review without model retraining.

A batch's last *closed* candle is the only decision timestamp.  When both
v0.8 and its causal price-only reference agree on a non-none reversal, a
candidate is logged. The next batch of N observed candles evaluates that
candidate from its first open to its last close.  No future candle can affect
signal generation; evaluation happens one batch later.

This is a direction-only, fixed-horizon *simulation*, not live execution or
proof of profitability. Price-only reference is a training proxy, not an
independent oracle. No stops, slippage, commission or leverage are assumed;
optional cost is a synthetic price-unit deduction only.
"""
from __future__ import annotations

import csv as csv_module
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import transform
from .features import read_candles
from .reversal_detection import (
    VERSION, TIMEFRAME, LOOKBACK, FEATURES, CLASSES, RSI_PERIOD,
    BB_PERIOD, BB_STD, MACD, ATR_PERIOD, PRETREND_PERIOD, SWING_PERIOD,
    PRETREND_ATR, CONFIRM_ATR, BalancedSoftmax,
    prepare_reversal, reversal_reference,
)


def _utc(timestamp: int) -> str:
    return pd.Timestamp(int(timestamp), unit="s", tz="UTC").isoformat()


def _load_model(path: str):
    root = Path(path)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    pre = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    sizes = cfg.get("architecture")
    if (cfg.get("model_version") != VERSION or
            cfg.get("task") != "past_only_confirmed_swing_reversal_reference_not_forecasting" or
            cfg.get("timeframe") != TIMEFRAME or cfg.get("lookback") != LOOKBACK or
            cfg.get("features") != list(FEATURES) or cfg.get("classes") != list(CLASSES) or
            cfg.get("indicators") != {"rsi": RSI_PERIOD, "bollinger": [BB_PERIOD, BB_STD],
                                      "macd": list(MACD), "atr": ATR_PERIOD} or
            cfg.get("reference") != {"pretrend_period": PRETREND_PERIOD,
                                     "swing_period": SWING_PERIOD,
                                     "pretrend_atr": PRETREND_ATR, "confirm_atr": CONFIRM_ATR} or
            pre.get("version") != 8 or not isinstance(sizes, list) or
            len(sizes) < 2 or sizes[0] != LOOKBACK * len(FEATURES) or
            sizes[-1] != len(CLASSES) or
            any(not isinstance(s, int) or s < 1 for s in sizes)):
        raise ValueError("Incompatible v0.8 reversal checkpoint")
    mu, sigma = np.asarray(pre.get("mean"), dtype=float), np.asarray(pre.get("std"), dtype=float)
    if (mu.shape != (len(FEATURES),) or sigma.shape != mu.shape or
            not np.isfinite(mu).all() or not np.isfinite(sigma).all() or
            (sigma <= 0).any()):
        raise ValueError("Invalid v0.8 preprocessing scaler")
    model = BalancedSoftmax(tuple(sizes), np.ones(len(CLASSES), dtype=float))
    with np.load(root / "model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    return model, mu, sigma


def _assess(signal: dict, raw: pd.DataFrame, start: int, stop: int, cost: float) -> dict:
    """Score an *existing* signal using exactly the next full batch of bars."""
    side = signal["signal"]
    if side not in ("BUY", "SELL"):
        raise ValueError("Only BUY/SELL signals can be assessed")
    first, last = raw.iloc[start], raw.iloc[stop - 1]
    entry, exit_ = float(first["open"]), float(last["close"])
    direction = 1 if side == "BUY" else -1
    gross_move = direction * (exit_ - entry)
    net_move = gross_move - cost
    highs = raw.iloc[start:stop]["high"].to_numpy(dtype=float)
    lows = raw.iloc[start:stop]["low"].to_numpy(dtype=float)
    mfe = float((highs.max() - entry) if direction == 1 else (entry - lows.min()))
    mae = float((entry - lows.min()) if direction == 1 else (highs.max() - entry))
    return {
        "previous_signal_batch": signal["batch"],
        "previous_signal": side,
        "entry_candle_utc": _utc(int(first["time"])),
        "entry_open": entry,
        "exit_candle_utc": _utc(int(last["time"])),
        "exit_close": exit_,
        "gross_directional_move": float(gross_move),
        "net_directional_move": float(net_move),
        "net_return_pct": float(100.0 * net_move / entry),
        "max_favorable_excursion": mfe,
        "max_adverse_excursion": mae,
        "result": "WIN" if net_move > 0 else "LOSS" if net_move < 0 else "FLAT",
    }


def run_reversal_walkforward(csv: str, model_dir: str, out: str,
                             batch_candles: int = 20, max_batches: int | None = None,
                             cost_per_unit: float = 0.0,
                             echo: bool = True) -> dict:
    """Walk 20 (configurable) closed candles at a time, emitting auditable rows.

    Uses the existing trained checkpoint, without touching holdouts or fitting
    anything. Signals are issued only when AI==price-reference!=none, at the
    batch-ending candle close. An open signal settles at the next complete
    batch's closing price, assuming next batch's first open is the entry.
    """
    if isinstance(batch_candles, bool) or not isinstance(batch_candles, int) or batch_candles < 2:
        raise ValueError("batch_candles must be >= 2")
    if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("max_batches must be >= 1")
    if not np.isfinite(cost_per_unit) or cost_per_unit < 0:
        raise ValueError("cost_per_unit must be nonnegative and finite")
    output = Path(out)
    if output.suffix.lower() != ".csv":
        raise ValueError("--out must be a .csv file")
    raw = read_candles(csv)
    data = prepare_reversal(csv)
    model, mu, sigma = _load_model(model_dir)
    # Feature calculation is causal: restricting to <= ending timestamp gives
    # identical vectors even if the rest of the file changes in the future.
    first_feature_raw = int(np.searchsorted(raw["time"].to_numpy(), data.timestamps[0]))
    min_seen = max(110, first_feature_raw + LOOKBACK)
    first_end = ((min_seen + batch_candles - 1) // batch_candles) * batch_candles
    complete_end = (len(raw) // batch_candles) * batch_candles
    if complete_end < first_end:
        raise ValueError("Not enough complete candles for initial warmup and first batch")
    # By construction each decision uses one fully completed batch, and every
    # settlement uses exactly the following completed batch (never a partial).
    end_points = range(first_end, complete_end + 1, batch_candles)
    if max_batches is not None:
        end_points = list(end_points)[:max_batches]
    rows, settled, outstanding = [], [], None
    times = data.timestamps
    for batch, end in enumerate(end_points, start=1):
        start = end - batch_candles
        prev = _assess(outstanding, raw, start, end, cost_per_unit) if outstanding else None
        if prev:
            settled.append(prev)
        timestamp = int(raw.iloc[end - 1]["time"])
        k = int(np.searchsorted(times, timestamp))
        if k >= len(times) or times[k] != timestamp or k < LOOKBACK - 1:
            raise ValueError("Insufficient past-only indicators at batch end")
        x = transform(data.features[k - LOOKBACK + 1:k + 1][None, ...], mu, sigma)
        scores = model.forward(x)[0]
        ai = CLASSES[int(scores.argmax())]
        ref = CLASSES[int(reversal_reference(data, np.asarray([k]))[0][0])]
        signal = "BUY" if ai == ref == "bullish_reversal" else (
                 "SELL" if ai == ref == "bearish_reversal" else "none")
        row = {
            "batch": batch,
            "batch_start_utc": _utc(int(raw.iloc[start]["time"])),
            "batch_end_utc": _utc(timestamp),
            "candle_count": batch_candles,
            "signal_candle_close": float(raw.iloc[end - 1]["close"]),
            "model_class": ai,
            "price_reference_class": ref,
            "model_score": float(scores.max()),
            "agreement": bool(ai == ref and ai != "none"),
            "new_signal": signal,
            "previous_signal_batch": prev["previous_signal_batch"] if prev else None,
            "previous_signal": prev["previous_signal"] if prev else None,
            "previous_result": prev["result"] if prev else None,
            "entry_candle_utc": prev["entry_candle_utc"] if prev else None,
            "entry_open": prev["entry_open"] if prev else None,
            "exit_candle_utc": prev["exit_candle_utc"] if prev else None,
            "exit_close": prev["exit_close"] if prev else None,
            "gross_directional_move": prev["gross_directional_move"] if prev else None,
            "net_directional_move": prev["net_directional_move"] if prev else None,
            "net_return_pct": prev["net_return_pct"] if prev else None,
            "max_favorable_excursion": prev["max_favorable_excursion"] if prev else None,
            "max_adverse_excursion": prev["max_adverse_excursion"] if prev else None,
        }
        rows.append(row)
        outstanding = {"batch": batch, "signal": signal} if signal != "none" else None
        if echo:
            outcome = (f"{prev['previous_signal']} {prev['result']} "
                       f"(gross={prev['gross_directional_move']:+.3f}; "
                       f"net={prev['net_directional_move']:+.3f}; "
                       f"return={prev['net_return_pct']:+.3f}%)") if prev else "-"
            print(f"[batch {batch:04d}] {_utc(timestamp)} | previous: {outcome} | "
                  f"AI={ai} ref={ref} | new={signal}", flush=True)

    wins = sum(x["result"] == "WIN" for x in settled)
    losses = sum(x["result"] == "LOSS" for x in settled)
    flats = sum(x["result"] == "FLAT" for x in settled)
    outcomes_by_side = {}
    for side in ("BUY", "SELL"):
        side_results = [x for x in settled if x["previous_signal"] == side]
        side_wins = sum(x["result"] == "WIN" for x in side_results)
        outcomes_by_side[side] = {
            "signals": sum(r["new_signal"] == side for r in rows),
            "resolved": len(side_results),
            "wins": side_wins,
            "losses": sum(x["result"] == "LOSS" for x in side_results),
            "flat": sum(x["result"] == "FLAT" for x in side_results),
            "win_rate_resolved": side_wins / len(side_results) if side_results else None,
            "mean_net_return_pct": (sum(x["net_return_pct"] for x in side_results) / len(side_results)
                                    if side_results else None),
        }
    summary = {
        "model_version": VERSION,
        "task": "causal_batch_reversal_signal_review_not_live_trading",
        "source": str(csv), "model": str(model_dir),
        "batch_candles": batch_candles,
        "signal_rule": "AI reversal class == price-only reference class != none at batch last closed candle",
        "entry_rule": "following batch first candle OPEN",
        "exit_rule": "following batch final candle CLOSE",
        "cost_per_unit": float(cost_per_unit),
        "outcome_rule": "signed(exit_close - entry_open) - cost_per_unit; positive WIN, negative LOSS, zero FLAT",
        "batches": len(rows),
        "total_signals": sum(r["new_signal"] != "none" for r in rows),
        "resolved_signals": len(settled), "wins": wins, "losses": losses, "flat": flats,
        "win_rate_resolved": float(wins / len(settled)) if settled else None,
        "mean_net_return_pct": (float(np.mean([x["net_return_pct"] for x in settled]))
                                if settled else None),
        "outcomes_by_side": outcomes_by_side,
        "ai_reversals_at_batch_end": sum(r["model_class"] != "none" for r in rows),
        "price_reference_reversals_at_batch_end": sum(r["price_reference_class"] != "none" for r in rows),
        "pending_signal": outstanding["signal"] if outstanding else None,
        "note": ("Gross directional walk-forward proxy unless cost_per_unit specified. "
                 "No stops, slippage, market execution or portfolio sizing. "
                 "Price reference was used for model training, so agreement is not independent confirmation. "
                 "Do not tune on final_test.csv if reserving it as untouched final test."),
    }
    # A max_batches run intentionally leaves more unprocessed candles than a
    # complete run; distinguish it from a trailing incomplete batch.
    summary["unprocessed_candles"] = len(raw) - (first_end + (len(rows) - 1) * batch_candles)
    summary["incomplete_trailing_candles_ignored"] = len(raw) % batch_candles if max_batches is None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file = output.with_suffix(".summary.json")
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if echo:
        rate = f"{100 * summary['win_rate_resolved']:.2f}%" if settled else "n/a"
        print(f"\nSUMMARY | batches={len(rows)} signals={summary['total_signals']} "
              f"resolved={len(settled)} wins={wins} losses={losses} flat={flats} "
              f"win_rate={rate} pending={summary['pending_signal'] or '-'}", flush=True)
        print(f"Reports: {output} | {json_file}", flush=True)
    return {"summary": summary, "batches": rows,
            "report_csv": str(output), "summary_json": str(json_file)}
