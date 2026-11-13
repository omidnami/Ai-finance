"""One-candle-at-a-time causal v0.8 reversal signal review.

At each completed candle, evaluate the model's trailing 40-bar window and
its price-only reference. An agreed candidate at t is paper-entered at
OPEN[t+1] and scored only once CLOSE[t+horizon] is available. Several
signals can be pending at once; each is evaluated independently.

This is a fixed-horizon direction-only research simulation, not live trading.
Price-only confirmation is the same label proxy used during training and is
not an independent oracle. An optional flat price-unit cost is illustrative.
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
    """Score one past signal only after all horizon candles have closed."""
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
    """Slide a 40-candle model window by ONE candle; score after `batch_candles`.

    `batch_candles` is retained for backward CLI compatibility but now means
    the EVALUATION HORIZON, not a 20-candle step. No future candles are read
    when generating a signal. Outcomes from t are available only at t+horizon.
    """
    if isinstance(batch_candles, bool) or not isinstance(batch_candles, int) or batch_candles < 2:
        raise ValueError("batch_candles (evaluation horizon) must be >= 2")
    if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("max_batches (maximum one-candle steps) must be >= 1")
    if not np.isfinite(cost_per_unit) or cost_per_unit < 0:
        raise ValueError("cost_per_unit must be nonnegative and finite")
    output = Path(out)
    if output.suffix.lower() != ".csv":
        raise ValueError("--out must be a .csv file")

    raw = read_candles(csv)
    data = prepare_reversal(csv)
    model, mu, sigma = _load_model(model_dir)
    timestamps = raw["time"].to_numpy(dtype=np.int64)
    times = data.timestamps
    # Retain the same >=110-bar indicator warmup as the original code, and
    # ensure 40 fully prepared feature rows are available at the first step.
    first_feature_raw = int(np.searchsorted(timestamps, times[0]))
    first_end = max(110, first_feature_raw + LOOKBACK)  # exclusive raw index
    if len(raw) < first_end:
        raise ValueError("Not enough completed candles for feature warmup")
    last_end = len(raw)  # every closed candle is eligible, even near EOF
    if max_batches is not None:
        last_end = min(last_end, first_end + max_batches - 1)

    # Due results are keyed by the raw *confirmation candle* index plus the
    # full forward horizon. At most one can become due per one-candle step.
    pending: dict[int, dict] = {}
    rows: list[dict] = []
    settled: list[dict] = []
    for step, end in enumerate(range(first_end, last_end + 1), start=1):
        t = end - 1
        window_start = end - LOOKBACK
        signal_close = float(raw.iloc[t]["close"])
        timestamp = int(timestamps[t])

        # Today's close can settle a signal issued 20 (or N) candles ago.
        # No outcome is computed before its full horizon has elapsed.
        due = pending.pop(t, None)
        previous = None
        if due is not None:
            previous = _assess(due, raw, due["confirmation_index"] + 1,
                               t + 1, cost_per_unit)
            settled.append(previous)

        k = int(np.searchsorted(times, timestamp))
        if k >= len(times) or times[k] != timestamp or k < LOOKBACK - 1:
            raise ValueError("Insufficient past-only indicators at window end")
        x = transform(data.features[k - LOOKBACK + 1:k + 1][None, ...], mu, sigma)
        scores = model.forward(x)[0]
        ai = CLASSES[int(scores.argmax())]
        ref = CLASSES[int(reversal_reference(data, np.asarray([k]))[0][0])]
        side = "BUY" if ai == ref == "bullish_reversal" else (
               "SELL" if ai == ref == "bearish_reversal" else "none")
        if side != "none":
            pending[t + batch_candles] = {
                "batch": step, "signal": side, "confirmation_index": t,
            }
        row = {
            "step": step,
            "window_start_utc": _utc(int(timestamps[window_start])),
            "window_end_utc": _utc(timestamp),
            "history_candles": LOOKBACK,
            "evaluation_horizon_candles": batch_candles,
            "signal_candle_close": signal_close,
            "model_class": ai,
            "price_reference_class": ref,
            "model_score": float(scores.max()),
            "agreement": bool(ai == ref and ai != "none"),
            "new_signal": side,
            "signal_evaluation_after_candles": batch_candles if side != "none" else None,
            "previous_signal_step": previous["previous_signal_batch"] if previous else None,
            "previous_signal": previous["previous_signal"] if previous else None,
            "previous_result": previous["result"] if previous else None,
            "entry_candle_utc": previous["entry_candle_utc"] if previous else None,
            "entry_open": previous["entry_open"] if previous else None,
            "exit_candle_utc": previous["exit_candle_utc"] if previous else None,
            "exit_close": previous["exit_close"] if previous else None,
            "gross_directional_move": previous["gross_directional_move"] if previous else None,
            "net_directional_move": previous["net_directional_move"] if previous else None,
            "net_return_pct": previous["net_return_pct"] if previous else None,
            "max_favorable_excursion": previous["max_favorable_excursion"] if previous else None,
            "max_adverse_excursion": previous["max_adverse_excursion"] if previous else None,
            "pending_signals": len(pending),
        }
        rows.append(row)
        if echo:
            outcome = (f"{previous['previous_signal']} {previous['result']} "
                       f"(gross={previous['gross_directional_move']:+.3f}; "
                       f"net={previous['net_directional_move']:+.3f}; "
                       f"return={previous['net_return_pct']:+.3f}%)") if previous else "-"
            print(f"[step {step:04d}] {_utc(timestamp)} | previous: {outcome} | "
                  f"AI={ai} ref={ref} | new={side} | pending={len(pending)}", flush=True)

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
            "pending": sum(x["signal"] == side for x in pending.values()),
            "wins": side_wins,
            "losses": sum(x["result"] == "LOSS" for x in side_results),
            "flat": sum(x["result"] == "FLAT" for x in side_results),
            "win_rate_resolved": side_wins / len(side_results) if side_results else None,
            "mean_net_return_pct": (sum(x["net_return_pct"] for x in side_results) / len(side_results)
                                    if side_results else None),
        }
    summary = {
        "model_version": VERSION,
        "task": "causal_sliding_window_reversal_signal_review_not_live_trading",
        "source": str(csv), "model": str(model_dir),
        "step_candles": 1,
        "history_candles": LOOKBACK,
        "evaluation_horizon_candles": batch_candles,
        "signal_rule": "AI reversal class == price-only reference class != none on latest closed candle",
        "entry_rule": "next candle OPEN following confirmation close",
        "exit_rule": f"CLOSE of {batch_candles}th subsequent candle",
        "cost_per_unit": float(cost_per_unit),
        "outcome_rule": "signed(exit_close - entry_open) - cost_per_unit; positive WIN, negative LOSS, zero FLAT",
        "steps": len(rows),
        "total_signals": sum(r["new_signal"] != "none" for r in rows),
        "resolved_signals": len(settled), "wins": wins, "losses": losses, "flat": flats,
        "win_rate_resolved": float(wins / len(settled)) if settled else None,
        "mean_net_return_pct": (float(np.mean([x["net_return_pct"] for x in settled]))
                                if settled else None),
        "outcomes_by_side": outcomes_by_side,
        "ai_reversals_at_window_end": sum(r["model_class"] != "none" for r in rows),
        "price_reference_reversals_at_window_end": sum(r["price_reference_class"] != "none" for r in rows),
        "pending_signals": len(pending),
        "pending_by_side": {side: sum(x["signal"] == side for x in pending.values())
                            for side in ("BUY", "SELL")},
        "unprocessed_candles": len(raw) - last_end,
        "note": (f"Sliding {LOOKBACK}-candle history with one-candle steps. Independent "
                 f"{batch_candles}-candle forward evaluation per signal; pending signals may overlap. "
                 "No stops, slippage, market execution or portfolio sizing. "
                 "Price reference was used for model training; agreement is not independent confirmation. "
                 "Do not tune on final_test.csv if reserving it as untouched final test."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file = output.with_suffix(".summary.json")
    json_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if echo:
        rate = f"{100 * summary['win_rate_resolved']:.2f}%" if settled else "n/a"
        print(f"\nSUMMARY | steps={len(rows)} signals={summary['total_signals']} "
              f"resolved={len(settled)} wins={wins} losses={losses} flat={flats} "
              f"win_rate={rate} pending={len(pending)}", flush=True)
        print(f"Reports: {output} | {json_file}", flush=True)
    return {"summary": summary, "steps": rows, "batches": rows,
            "report_csv": str(output), "summary_json": str(json_file)}
