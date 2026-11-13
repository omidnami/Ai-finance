"""v0.9 causal price-action context, setup quality and paper stop-entry replay.

Research-only implementation inspired by Al Brooks concepts (not his rules):
regime -> setup/signal bar -> triggered entry -> protective stop/target.
No model retraining, no hindsight pivots, no automatic live orders.
"""
from __future__ import annotations

import csv as csv_module
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import transform
from .features import read_candles
from .network import MLP
from .reversal_detection import CLASSES as REV_CLASSES, LOOKBACK as REV_LOOKBACK, prepare_reversal, reversal_reference
from .reversal_walkforward import _load_model as load_reversal_model
from .trend_detection import (
    MODEL_VERSION as TREND_VERSION, LOOKBACK as TREND_LOOKBACK,
    FEATURE_NAMES as TREND_FEATURES, CLASS_NAMES as TREND_CLASSES,
    BB_PERIOD, BB_STD, EMA_PERIOD, AROON_PERIOD, ATR_PERIOD,
    LABEL_PERIOD, WEAK_SLOPE_ATR, TIMEFRAME_MINUTES,
    prepare_trend,
)

VERSION = "0.9.0"


@dataclass(frozen=True)
class EntryRules:
    """Heuristic choices, explicitly NOT calibrated trading probabilities."""
    min_quality: int = 65
    order_valid_bars: int = 3
    max_holding_bars: int = 20
    target_rr: float = 2.0
    tick_size: float = 0.01
    stop_buffer_atr: float = 0.10
    min_risk_atr: float = 0.15
    max_risk_atr: float = 3.0
    roundtrip_cost_units: float = 0.0
    risk_budget_fraction: float = 0.005
    initial_equity: float = 10_000.0

    def check(self):
        if (not 0 <= self.min_quality <= 100 or self.order_valid_bars < 1 or
                self.max_holding_bars < 1):
            raise ValueError("Invalid score threshold or positive-bar settings")
        values = (self.target_rr, self.tick_size, self.stop_buffer_atr, self.min_risk_atr,
                  self.max_risk_atr, self.roundtrip_cost_units, self.risk_budget_fraction,
                  self.initial_equity)
        if any(not np.isfinite(v) for v in values):
            raise ValueError("All numeric rules must be finite")
        if (self.target_rr <= 0 or self.tick_size <= 0 or self.stop_buffer_atr < 0 or
                self.min_risk_atr < 0 or self.max_risk_atr <= self.min_risk_atr or
                self.roundtrip_cost_units < 0 or not 0 < self.risk_budget_fraction <= .02 or
                self.initial_equity <= 0):
            raise ValueError("Invalid risk settings")


def utc(value: int) -> str:
    return pd.Timestamp(int(value), unit="s", tz="UTC").isoformat()


def _load_trend_model(root_path: str):
    root = Path(root_path)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    pre = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    sizes = cfg.get("architecture")
    expected = {"bollinger": [BB_PERIOD, BB_STD], "ema": EMA_PERIOD,
                "aroon": AROON_PERIOD, "atr": ATR_PERIOD}
    if (cfg.get("model_version") != TREND_VERSION or
            cfg.get("task") != "current_trend_classification_not_forecasting" or
            cfg.get("lookback") != TREND_LOOKBACK or cfg.get("timeframe") != TIMEFRAME_MINUTES or
            cfg.get("label_period") != LABEL_PERIOD or cfg.get("features") != list(TREND_FEATURES) or
            cfg.get("classes") != list(TREND_CLASSES) or cfg.get("indicators") != expected or
            cfg.get("weak_trend_threshold_atr") != WEAK_SLOPE_ATR or
            pre.get("version") != 7 or not isinstance(sizes, list) or
            len(sizes) < 2 or sizes[0] != TREND_LOOKBACK*len(TREND_FEATURES) or
            sizes[-1] != 1 or any(not isinstance(s, int) or s < 1 for s in sizes)):
        raise ValueError("Incompatible v0.7 trend checkpoint")
    mu, std = np.asarray(pre.get("mean"), dtype=float), np.asarray(pre.get("std"), dtype=float)
    if (mu.shape != (len(TREND_FEATURES),) or std.shape != mu.shape or
            not np.isfinite(mu).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Invalid v0.7 scaler")
    model = MLP(sizes=tuple(sizes))
    with np.load(root / "model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    return model, mu, std


def market_context(raw: pd.DataFrame, t: int, trend_up_score: float, atr: float) -> dict:
    """Analyze only fully observed candles <= t: cycle, strength, location."""
    if t < 40 or not np.isfinite(atr) or atr <= 0:
        raise ValueError("Need 41 observed bars and a positive ATR")
    part = raw.iloc[t - 31:t + 1]
    close = part["close"].to_numpy(dtype=float)
    high = part["high"].to_numpy(dtype=float)
    low = part["low"].to_numpy(dtype=float)
    trend = "uptrend" if trend_up_score >= .5 else "downtrend"
    x = np.arange(len(close), dtype=float)
    x -= x.mean()
    slope = float((np.log(close) @ x) / (x @ x))
    slope_atr = float(slope * 31 * close[-1] / atr)
    travel = float(np.abs(np.diff(close)).sum())
    efficiency = float(abs(close[-1] - close[0]) / travel) if travel else 0.0
    hi, lo = float(high.max()), float(low.min())
    position = float(np.clip((close[-1] - lo) / max(hi - lo, 1e-12), 0, 1))
    previous = raw.iloc[t - 20:t]
    breakout_up = bool(close[-1] > float(previous["high"].max()))
    breakout_down = bool(close[-1] < float(previous["low"].min()))
    if (breakout_up or breakout_down) and abs(close[-1]-float(raw.iloc[t]["open"])) >= .6*atr:
        cycle = "breakout"
    elif abs(slope_atr) >= 1.4 and efficiency >= .28:
        cycle = "trend_channel"
    else:
        cycle = "trading_range"
    return {"market_cycle": cycle, "trend_v07": trend, "trend_up_score_uncalibrated": float(trend_up_score),
            "price_slope_atr": slope_atr, "directional_efficiency": efficiency,
            "range_position_32": position, "range_high": hi, "range_low": lo,
            "breakout_direction": ("BUY" if breakout_up else "SELL" if breakout_down else "none")}


def grade_setup(raw: pd.DataFrame, t: int, side: str, ctx: dict,
                reversal_score: float, atr: float, rules: EntryRules) -> dict:
    """Transparent heuristic quality score; independently check acceptance."""
    bar, pivot = raw.iloc[t], raw.iloc[t-1]
    high, low = float(bar["high"]), float(bar["low"])
    opn, close = float(bar["open"]), float(bar["close"])
    span = max(high-low, 1e-12)
    bull = side == "BUY"
    direction = 1 if bull else -1
    body = direction * (close - opn)
    close_location = ((close-low)/span) if bull else ((high-close)/span)
    wick = ((min(opn, close)-low)/span) if bull else ((high-max(opn, close))/span)
    score = 20 # matching model/reference is a prerequisite, not evidence of independence
    score += min(15, int(max(0, reversal_score) * 15))
    score += int(15 * max(0, close_location))
    score += 12 if body >= .2*atr else 6 if body > 0 else 0
    score += min(10, int(max(0, wick) * 25))
    in_direction = ctx["trend_v07"] == ("uptrend" if bull else "downtrend")
    if ctx["market_cycle"] == "trading_range":
        position = ctx["range_position_32"]
        edge = (position <= .33) if bull else (position >= .67)
        score += 15 if edge else 0
    elif ctx["market_cycle"] == "trend_channel":
        score += 8 if in_direction else 2
    else:  # breakout continuation has a risk of fading reversals
        score += 12 if in_direction else 0
    prior_move = ctx["price_slope_atr"]
    # A swing reversal is more credible after a meaningful move opposite its direction.
    score += 10 if direction * prior_move <= -1.0 else 0
    score = int(min(100, score))
    buffer = rules.stop_buffer_atr*atr
    trigger = (high + rules.tick_size) if bull else (low - rules.tick_size)
    stop = (min(low, float(pivot["low"]))-buffer) if bull else (
        max(high, float(pivot["high"]))+buffer)
    initial_risk = direction*(trigger-stop)
    is_middle = (ctx["market_cycle"] == "trading_range" and .33 < ctx["range_position_32"] < .67)
    reason = "accepted"
    if is_middle:
        reason = "range_middle"
    elif initial_risk <= 0 or initial_risk < rules.min_risk_atr*atr:
        reason = "risk_too_small"
    elif initial_risk > rules.max_risk_atr*atr:
        reason = "risk_too_large"
    elif score < rules.min_quality:
        reason = "low_quality"
    # Not pretending model scores are calibrated probabilities or independent confirmation.
    return {"score_0_to_100_heuristic": score, "reason": reason,
            "accepted": reason == "accepted", "signal_bar_close_location": float(close_location),
            "signal_bar_body_atr": float(body/atr), "signal_wick_fraction": float(wick),
            "trigger_price": float(trigger), "protective_stop": float(stop),
            "planned_risk_price_units": float(initial_risk),
            "planned_target_price": float(trigger+direction*rules.target_rr*initial_risk),
            "planned_reward_risk": float(rules.target_rr)}


def _probs(data, model, mu, std, k: int, n: int) -> np.ndarray:
    return model.forward(transform(data.features[k - n + 1:k + 1][None, ...], mu, std))[0]


def _snapshot(t: int, raw: pd.DataFrame, rdata, tdata, rlookup, tlookup,
              rmodel, rmu, rstd, tmodel, tmu, tstd, rules: EntryRules) -> dict:
    stamp = int(raw.iloc[t]["time"])
    rk, tk = rlookup[stamp], tlookup[stamp]
    rp = _probs(rdata, rmodel, rmu, rstd, rk, REV_LOOKBACK)
    tp = float(_probs(tdata, tmodel, tmu, tstd, tk, TREND_LOOKBACK))
    ref = int(reversal_reference(rdata, np.asarray([rk]))[0][0])
    predicted = int(np.argmax(rp))
    match = predicted == ref and ref in (1, 2)
    side = "BUY" if match and ref == 1 else "SELL" if match and ref == 2 else "none"
    atr = float(rdata.atr[rk])
    ctx = market_context(raw, t, tp, atr)
    quality = (grade_setup(raw, t, side, ctx, float(rp[predicted]), atr, rules)
               if side != "none" else None)
    return {"timestamp_utc": utc(stamp), "t": t, "ai_reversal": REV_CLASSES[predicted],
            "price_reference": REV_CLASSES[ref], "reversal_score_uncalibrated": float(rp[predicted]),
            "candidate": side, "context": ctx, "quality": quality,
            "action": ("PLACE_STOP_ENTRY" if quality and quality["accepted"] else "PASS")}


def _advance_order(order: dict, bar: pd.Series, i: int, rules: EntryRules) -> tuple[dict | None, dict | None]:
    """Pending orders fill only on touch; conservative fill on ambiguous OHLC."""
    direction = 1 if order["side"] == "BUY" else -1
    trigger, stop = order["trigger"], order["stop"]
    high, low, opn = (float(bar[x]) for x in ("high", "low", "open"))
    hit = high >= trigger if direction == 1 else low <= trigger
    if hit:
        # Stop market entry: opening gaps in the triggered direction slip adversely.
        fill = max(trigger, opn) if direction == 1 else min(trigger, opn)
        risk = direction*(fill-stop)
        if risk <= 0:
            # Gapped beyond stop: reject an impossible stop placement.
            return None, {"event": "ORDER_CANCELLED", "reason": "invalid_gap_stop", "i": i}
        target = fill + direction*rules.target_rr*risk
        order = dict(order, status="open", entry_i=i, entry_price=fill,
                     target=target, risk=risk, units=order["risk_dollars"] /
                     (risk + rules.roundtrip_cost_units))
        return order, {"event": "ENTRY_FILLED", "i": i, "fill": fill, "target": target}
    if i - order["created_i"] >= rules.order_valid_bars:
        return None, {"event": "ORDER_EXPIRED", "i": i}
    return order, None


def _advance_position(position: dict, bar: pd.Series, i: int, rules: EntryRules) -> tuple[dict | None, dict | None]:
    d = 1 if position["side"] == "BUY" else -1
    opn, high, low, close = (float(bar[x]) for x in ("open", "high", "low", "close"))
    stop, target = position["stop"], position["target"]
    if d == 1:
        gap_stop, gap_target = opn <= stop, opn >= target
        hit_stop, hit_target = low <= stop, high >= target
    else:
        gap_stop, gap_target = opn >= stop, opn <= target
        hit_stop, hit_target = high >= stop, low <= target
    # Conservative priority: adverse gap > favorable gap > intrabar stop > target.
    if gap_stop:
        price, exit_reason = opn, "stop_gap"
    elif gap_target:
        price, exit_reason = target, "target_gap_capped"
    elif hit_stop:
        price, exit_reason = stop, "stop"  # worst case if both reached intrabar
    elif hit_target:
        price, exit_reason = target, "target"
    elif i - position["entry_i"] + 1 >= rules.max_holding_bars:
        price, exit_reason = close, "time_exit"
    else:
        return position, None
    gross = d*(price-position["entry_price"])
    net = gross-rules.roundtrip_cost_units
    result = dict(position, status="closed", exit_i=i, exit_price=float(price),
                  exit_reason=exit_reason, net_price_units=float(net),
                  outcome="WIN" if net > 0 else "LOSS" if net < 0 else "FLAT",
                  realized_R=float(net/position["risk"]),
                  hypothetical_pnl=float(net*position["units"]))
    return None, result


def run_entries(csv: str, trend_model: str, reversal_model: str, out: str,
                rules: EntryRules | None = None, max_steps: int | None = None,
                echo: bool = True) -> dict:
    """Chronological replay: each signal uses only completed current/past bars."""
    cfg = rules or EntryRules()
    cfg.check()
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be positive")
    output = Path(out)
    if output.suffix.lower() != ".csv":
        raise ValueError("Output path must be a .csv")
    raw = read_candles(csv)
    rdata = prepare_reversal(csv)
    tdata = prepare_trend(csv)
    rmodel, rmu, rstd = load_reversal_model(reversal_model)
    tmodel, tmu, tstd = _load_trend_model(trend_model)
    rlookup = {int(t): i for i, t in enumerate(rdata.timestamps)}
    tlookup = {int(t): i for i, t in enumerate(tdata.timestamps)}
    all_stamps = raw["time"].to_numpy(dtype=np.int64)
    first_r = int(np.searchsorted(all_stamps, rdata.timestamps[REV_LOOKBACK-1]))
    first_t = int(np.searchsorted(all_stamps, tdata.timestamps[TREND_LOOKBACK-1]))
    first = max(110, first_r, first_t, 41)
    last = min(len(raw), first + max_steps) if max_steps else len(raw)
    if first >= last:
        raise ValueError("Insufficient candles for any entry review")
    state = None  # exactly one pending stop-entry or open paper position
    rows, trades = [], []
    issued, skipped, expired, filled, canceled = 0, 0, 0, 0, 0
    for t in range(first, last):
        bar = raw.iloc[t]
        change = ""
        if state is not None:
            if state["status"] == "pending":
                state, event = _advance_order(state, bar, t, cfg)
                if event is not None:
                    change = event["event"]
                    if change == "ORDER_EXPIRED": expired += 1
                    elif change == "ORDER_CANCELLED": canceled += 1
                    elif change == "ENTRY_FILLED": filled += 1
                    if change == "ENTRY_FILLED":
                        # OHLC cannot reveal if the subsequent stop was earlier;
                        # pessimistic same-bar stop-vs-target priority applies.
                        state, result = _advance_position(state, bar, t, cfg)
                        if result is not None:
                            trades.append(result)
                            change += "+" + result["exit_reason"].upper()
            else:
                state, result = _advance_position(state, bar, t, cfg)
                if result is not None:
                    trades.append(result)
                    change = "TRADE_"+result["exit_reason"].upper()

        snapshot = _snapshot(t, raw, rdata, tdata, rlookup, tlookup,
                             rmodel, rmu, rstd, tmodel, tmu, tstd, cfg)
        quality = snapshot["quality"]
        decision = "none"
        if snapshot["candidate"] != "none":
            if state is not None:
                decision = "SKIP_BUSY"
                skipped += 1
            elif quality is None or not quality["accepted"]:
                decision = "SKIP_"+(quality["reason"].upper() if quality else "UNQUALIFIED")
                skipped += 1
            else:
                side = snapshot["candidate"]
                state = {"status": "pending", "side": side,
                         "created_i": t, "confirmation_utc": snapshot["timestamp_utc"],
                         "trigger": quality["trigger_price"],
                         "stop": quality["protective_stop"],
                         "score": quality["score_0_to_100_heuristic"],
                         "context": snapshot["context"]["market_cycle"],
                         "risk_dollars": cfg.initial_equity*cfg.risk_budget_fraction}
                decision = "QUEUE_"+side
                issued += 1
        row = {"step": t-first+1, "candle_utc": snapshot["timestamp_utc"],
               "market_cycle": snapshot["context"]["market_cycle"],
               "v07_trend": snapshot["context"]["trend_v07"],
               "slope_atr": snapshot["context"]["price_slope_atr"],
               "efficiency": snapshot["context"]["directional_efficiency"],
               "range_position": snapshot["context"]["range_position_32"],
               "ai_reversal": snapshot["ai_reversal"], "price_reference": snapshot["price_reference"],
               "candidate": snapshot["candidate"],
               "quality_score": quality["score_0_to_100_heuristic"] if quality else None,
               "quality_reason": quality["reason"] if quality else "none",
               "planned_entry_trigger": quality["trigger_price"] if quality else None,
               "planned_stop": quality["protective_stop"] if quality else None,
               "planned_target": quality["planned_target_price"] if quality else None,
               "decision": decision, "order_trade_event": change,
               "state_after_close": state["status"] if state else "flat",
               "last_trade_outcome": (trades[-1]["outcome"] if change and trades and
                                      trades[-1]["exit_i"] == t else None)}
        rows.append(row)
        if echo and (decision != "none" or change):
            print(f"[{row['step']:>5}] {row['candle_utc']} context={row['market_cycle']} "
                  f"candidate={row['candidate']} quality={row['quality_score']} "
                  f"decision={decision} event={change or '-'}", flush=True)
    complete = [t for t in trades if t["status"] == "closed"]
    def group_stats(side: str | None):
        subset = [t for t in complete if side is None or t["side"] == side]
        n = len(subset)
        wins = sum(t["outcome"] == "WIN" for t in subset)
        return {"trades": n, "wins": wins, "losses": sum(t["outcome"] == "LOSS" for t in subset),
                "flat": sum(t["outcome"] == "FLAT" for t in subset),
                "win_rate": float(wins/n) if n else None,
                "mean_R": float(np.mean([t["realized_R"] for t in subset])) if n else None,
                "sum_paper_pnl": float(sum(t["hypothetical_pnl"] for t in subset))}
    summary = {"version": VERSION, "task": "causal_brooks_inspired_context_setup_entry_research",
               "data": str(csv), "trend_checkpoint": str(trend_model),
               "reversal_checkpoint": str(reversal_model), "rules": asdict(cfg),
               "steps": len(rows), "matched_candidates": sum(r["candidate"] != "none" for r in rows),
               "queued_orders": issued, "skipped_candidates": skipped, "expired_orders": expired,
               "canceled_orders": canceled, "filled_orders": filled, "open_or_pending_at_end": state,
               "overall": group_stats(None), "by_side": {s: group_stats(s) for s in ("BUY", "SELL")},
               "assumptions": ["No trading API; research-only paper orders",
                 "Heuristic setup score is not calibrated confidence/probability",
                 "One open or pending position at a time; rejected overlapping candidates",
                 "Stop entry after confirmation; no entry at signal close",
                 "Intrabar stop/target conflict assumes stop first",
                 "Stop gaps may exit at unfavorable open; target gaps capped at target",
                 "No variable spread, funding, swap or real broker fills",
                 "Price-reference confirmation reuses v0.8 training target; not independent",
                 "Do not optimize on the consumed final_test.csv; use fresh unseen data"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trades_path = output.with_name(output.stem + ".trades.csv")
    with trades_path.open("w", newline="", encoding="utf-8-sig") as f:
        keys = (list(complete[0]) if complete else [
            "side", "confirmation_utc", "created_i", "entry_i", "entry_price",
            "stop", "target", "risk", "units", "exit_i", "exit_price",
            "exit_reason", "net_price_units", "outcome", "realized_R",
            "hypothetical_pnl"])
        writer = csv_module.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(complete)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if echo:
        print(f"SUMMARY v0.9 steps={len(rows)} candidates={summary['matched_candidates']} "
              f"queued={issued} filled={filled} closed={summary['overall']['trades']} "
              f"win_rate={summary['overall']['win_rate']} mean_R={summary['overall']['mean_R']}")
        print(f"Reports: {output} {trades_path} {summary_path}")
    return {"summary": summary, "steps": rows, "trades": complete,
            "report_csv": str(output), "trades_csv": str(trades_path),
            "summary_json": str(summary_path)}


def inspect_last(csv: str, trend_model: str, reversal_model: str,
                 rules: EntryRules | None = None) -> dict:
    cfg = rules or EntryRules()
    cfg.check()
    raw = read_candles(csv)
    rdata, tdata = prepare_reversal(csv), prepare_trend(csv)
    rmodel, rmu, rstd = load_reversal_model(reversal_model)
    tmodel, tmu, tstd = _load_trend_model(trend_model)
    t = len(raw)-1
    if t < 110:
        raise ValueError("At least 111 closed candles required")
    return _snapshot(t, raw, rdata, tdata,
                     {int(v): i for i, v in enumerate(rdata.timestamps)},
                     {int(v): i for i, v in enumerate(tdata.timestamps)},
                     rmodel, rmu, rstd, tmodel, tmu, tstd, cfg)
