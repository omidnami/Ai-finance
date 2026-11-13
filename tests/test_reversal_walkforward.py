"""Test v0.8 sliding 40-candle window, one-candle steps, 20-candle outcomes."""
import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.reversal_detection import CLASSES, detect_reversal
from finance_ai.reversal_walkforward import _assess, run_reversal_walkforward


@pytest.fixture
def test_candles(tmp_path):
    n = 360
    times = 1_745_362_800 + np.arange(n) * 900
    rng = np.random.default_rng(73)
    close = 3400 + np.cumsum(rng.normal(0, 2.3, n)) + 7 * np.sin(np.arange(n)/8)
    opn = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "time": times, "open": opn,
        "high": np.maximum(opn, close) + .8,
        "low": np.minimum(opn, close) - .8,
        "close": close, "volume": 150,
        "timeframe": 15,
    })
    src = tmp_path / "new_candles.csv"
    frame.to_csv(src, index=False)
    return src


@pytest.fixture
def small_checkpoint(tmp_path):
    from finance_ai.reversal_detection import (
        BalancedSoftmax, FEATURES, LOOKBACK, VERSION, RSI_PERIOD, BB_PERIOD,
        BB_STD, MACD, ATR_PERIOD, PRETREND_PERIOD, SWING_PERIOD,
        PRETREND_ATR, CONFIRM_ATR,
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    network = BalancedSoftmax((LOOKBACK*len(FEATURES), 12, 3), np.ones(3), seed=22)
    np.savez_compressed(root / "model.npz", **network.state_dict())
    (root / "config.json").write_text(json.dumps({
        "model_version": VERSION,
        "task": "past_only_confirmed_swing_reversal_reference_not_forecasting",
        "architecture": [LOOKBACK*len(FEATURES), 12, 3],
        "timeframe": 15, "lookback": LOOKBACK,
        "features": list(FEATURES), "classes": list(CLASSES),
        "indicators": {"rsi": RSI_PERIOD, "bollinger": [BB_PERIOD, BB_STD],
                       "macd": list(MACD), "atr": ATR_PERIOD},
        "reference": {"pretrend_period": PRETREND_PERIOD, "swing_period": SWING_PERIOD,
                      "pretrend_atr": PRETREND_ATR, "confirm_atr": CONFIRM_ATR},
    }))
    (root / "preprocessing.json").write_text(json.dumps({
        "version": 8, "mean": [0.0]*len(FEATURES), "std": [1.0]*len(FEATURES),
    }))
    return root


def test_next_horizon_settlement_uses_open_and_last_close():
    opens = np.arange(100, 120, dtype=float)
    closes = opens + 0.5
    frame = pd.DataFrame({
        "time": 1_745_362_800 + np.arange(20)*900,
        "open": opens,
        "high": opens + 4,
        "low": opens - 3,
        "close": closes,
    })
    bull = _assess({"batch": 4, "signal": "BUY"}, frame, 0, 20, 1.0)
    assert bull["previous_signal_batch"] == 4
    assert bull["entry_open"] == 100.0
    assert bull["exit_close"] == 119.5
    assert bull["gross_directional_move"] == pytest.approx(19.5)
    assert bull["net_directional_move"] == pytest.approx(18.5)
    assert bull["result"] == "WIN"
    assert bull["max_favorable_excursion"] == pytest.approx(23.0)
    assert bull["max_adverse_excursion"] == pytest.approx(3.0)
    bear = _assess({"batch": 3, "signal": "SELL"}, frame, 0, 20, 0)
    assert bear["gross_directional_move"] == pytest.approx(-19.5)
    assert bear["result"] == "LOSS"


def test_window_moves_one_bar_and_overlapping_signals_are_all_settled(test_candles, small_checkpoint,
                                                                         monkeypatch, tmp_path):
    from finance_ai import reversal_walkforward as module

    class ThreeEarlySignals:
        calls = 0
        def forward(self, x):
            assert x.shape == (1, 800)  # 40 historical candles, 20 channels each
            self.calls += 1
            if self.calls in (1, 3):
                return np.array([[.05, .90, .05]])
            if self.calls == 2:
                return np.array([[.05, .05, .90]])
            return np.array([[.90, .05, .05]])

    model = ThreeEarlySignals()
    monkeypatch.setattr(module, "_load_model", lambda _: (model, np.zeros(20), np.ones(20)))
    ref_calls = 0
    def fake_reference(_data, ends):
        nonlocal ref_calls
        ref_calls += 1
        return np.array([[1, 2, 1][ref_calls-1] if ref_calls <= 3 else 0]), np.array([0.0])
    monkeypatch.setattr(module, "reversal_reference", fake_reference)

    result = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "report.csv"), max_batches=23, echo=False)
    rows = result["batches"]
    assert len(rows) == 23
    assert [r["new_signal"] for r in rows[:4]] == ["BUY", "SELL", "BUY", "none"]
    assert [r["previous_result"] for r in rows[:20]] == [None] * 20
    assert rows[20]["previous_signal_step"] == 1
    assert rows[21]["previous_signal_step"] == 2
    assert rows[22]["previous_signal_step"] == 3
    assert [r["previous_signal"] for r in rows[20:23]] == ["BUY", "SELL", "BUY"]
    assert all(r["previous_result"] in {"WIN", "LOSS", "FLAT"} for r in rows[20:23])
    raw = pd.read_csv(test_candles)
    assert rows[20]["entry_open"] == pytest.approx(float(raw.iloc[110]["open"]))
    assert rows[20]["exit_close"] == pytest.approx(float(raw.iloc[129]["close"]))
    for n, row in enumerate(rows):
        assert row["history_candles"] == 40
        assert row["evaluation_horizon_candles"] == 20
        assert row["window_end_utc"] == pd.Timestamp(int(raw.iloc[109+n]["time"]), unit="s", tz="UTC").isoformat()
        assert row["window_start_utc"] == pd.Timestamp(int(raw.iloc[70+n]["time"]), unit="s", tz="UTC").isoformat()
    summary = result["summary"]
    assert summary["step_candles"] == 1
    assert summary["total_signals"] == 3
    assert summary["resolved_signals"] == 3
    assert summary["pending_signals"] == 0
    assert len(pd.read_csv(result["report_csv"])) == 23
    assert json.loads((tmp_path / "report.summary.json").read_text())["pending_signals"] == 0


def test_partial_steps_leave_multiple_pending_signals(test_candles, small_checkpoint, monkeypatch, tmp_path):
    from finance_ai import reversal_walkforward as module

    class AlwaysBull:
        def forward(self, x):
            return np.array([[.01, .98, .01]])
    monkeypatch.setattr(module, "_load_model", lambda _: (AlwaysBull(), np.zeros(20), np.ones(20)))
    monkeypatch.setattr(module, "reversal_reference", lambda _, __: (np.array([1]), np.array([0.])))
    result = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "pending.csv"), max_batches=3, echo=False)
    assert result["summary"]["pending_signals"] == 3
    assert result["summary"]["resolved_signals"] == 0
    assert [x["pending_signals"] for x in result["steps"]] == [1, 2, 3]


def test_real_checkpoint_matches_each_prefix_inference_and_future_mutation(test_candles,
                                                                               small_checkpoint, tmp_path):
    report = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "original.csv"), max_batches=4, echo=False)
    raw = pd.read_csv(test_candles)
    for row in report["steps"]:
        stop = 109 + row["step"]  # first step observes raw indices 0..109
        prefix = tmp_path / "prefix.csv"
        raw.iloc[:stop].to_csv(prefix, index=False)
        truth = detect_reversal(str(prefix), str(small_checkpoint))
        assert row["model_class"] == truth["detected_class"]
        assert row["price_reference_class"] == truth["price_only_confirmed_reference"]
        assert row["model_score"] == pytest.approx(
            max(truth["scores_uncalibrated"].values()), abs=1e-12)
    # Mutating future prices cannot alter the four past-only predictions.
    changed = raw.copy()
    for c in ("open", "high", "low", "close"):
        changed.loc[180:, c] += 400
    altered_csv = tmp_path / "altered.csv"
    changed.to_csv(altered_csv, index=False)
    modified = run_reversal_walkforward(str(altered_csv), str(small_checkpoint),
                                        str(tmp_path / "modified.csv"), max_batches=4, echo=False)
    for left, right in zip(report["steps"], modified["steps"]):
        for key in left:
            if key == "model_score":
                assert left[key] == pytest.approx(right[key], abs=1e-12)
            else:
                assert left[key] == right[key]


def test_ref_mismatch_prevents_signal(test_candles, small_checkpoint, monkeypatch, tmp_path):
    from finance_ai import reversal_walkforward as module

    class AlwaysBull:
        def forward(self, x):
            return np.array([[.01, .98, .01]])
    monkeypatch.setattr(module, "_load_model", lambda _: (AlwaysBull(), np.zeros(20), np.ones(20)))
    monkeypatch.setattr(module, "reversal_reference", lambda _, __: (np.array([2]), np.array([0.])))
    result = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "mismatch.csv"), max_batches=3, echo=False)
    assert all(x["new_signal"] == "none" for x in result["steps"])
    assert result["summary"]["resolved_signals"] == 0
    assert result["summary"]["win_rate_resolved"] is None


def test_invalid_arguments_and_schema(test_candles, small_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="batch_candles"):
        run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                 str(tmp_path / "x.csv"), batch_candles=1, echo=False)
    with pytest.raises(ValueError, match="cost_per_unit"):
        run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                 str(tmp_path / "x.csv"), cost_per_unit=-1, echo=False)
    with pytest.raises(ValueError, match="--out"):
        run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                 str(tmp_path / "x.txt"), echo=False)
    original = json.loads((small_checkpoint / "config.json").read_text())
    original["features"] = ["bad"]
    (small_checkpoint / "config.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Incompatible"):
        run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                 str(tmp_path / "x.csv"), max_batches=1, echo=False)
