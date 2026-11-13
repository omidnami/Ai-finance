"""Verify 20-candle chronological v0.8 review: past-only signals, next-batch outcomes."""
import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.reversal_detection import CLASSES, detect_reversal
from finance_ai.reversal_walkforward import _assess, run_reversal_walkforward


@pytest.fixture
def test_candles(tmp_path):
    # 360 closed M15 candles, sufficient for indicator warmup and 12+ steps.
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


def test_next_batch_settlement_uses_open_and_last_close():
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


def test_twenty_at_a_time_both_markers_then_next_batch_outcome(test_candles, small_checkpoint,
                                                                monkeypatch, tmp_path):
    from finance_ai import reversal_walkforward as module
    class ThreeBatchModel:
        calls = 0
        def forward(self, x):
            assert x.shape == (1, 800)
            self.calls += 1
            ix = min(self.calls, 3)  # bull, bear, bull, then none
            if ix == 1 or ix == 3:
                return np.array([[.05, .90, .05]])
            if ix == 2:
                return np.array([[.05, .05, .90]])
            return np.array([[.90, .05, .05]])
    model = ThreeBatchModel()
    monkeypatch.setattr(module, "_load_model", lambda _: (model, np.zeros(20), np.ones(20)))
    reference_calls = [1, 2, 1]
    def fake_reference(_d, ends):
        return np.array([reference_calls.pop(0)]), np.array([0.0])
    monkeypatch.setattr(module, "reversal_reference", fake_reference)
    result = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "report.csv"), max_batches=3, echo=False)
    rows = result["batches"]
    assert [r["new_signal"] for r in rows] == ["BUY", "SELL", "BUY"]
    assert rows[0]["previous_result"] is None
    assert rows[1]["previous_signal_batch"] == 1
    assert rows[1]["previous_signal"] == "BUY"
    assert rows[2]["previous_signal_batch"] == 2
    assert rows[2]["previous_signal"] == "SELL"
    # A candidate is never scored in the same batch it is issued.
    assert rows[0]["entry_open"] is None
    assert rows[1]["entry_open"] == pytest.approx(float(pd.read_csv(test_candles).iloc[120]["open"]))
    assert rows[1]["exit_close"] == pytest.approx(float(pd.read_csv(test_candles).iloc[139]["close"]))
    assert result["summary"]["total_signals"] == 3
    assert result["summary"]["resolved_signals"] == 2
    assert result["summary"]["pending_signal"] == "BUY"
    saved = pd.read_csv(result["report_csv"])
    assert len(saved) == 3
    assert json.loads((tmp_path / "report.summary.json").read_text())["pending_signal"] == "BUY"


def test_real_checkpoint_matches_prefix_inference_and_future_mutation(test_candles,
                                                                       small_checkpoint, tmp_path):
    report = run_reversal_walkforward(str(test_candles), str(small_checkpoint),
                                     str(tmp_path / "original.csv"), max_batches=4, echo=False)
    raw = pd.read_csv(test_candles)
    for row in report["batches"]:
        stop = 100 + row["batch"] * 20
        prefix = tmp_path / "prefix.csv"
        raw.iloc[:stop].to_csv(prefix, index=False)
        truth = detect_reversal(str(prefix), str(small_checkpoint))
        assert row["model_class"] == truth["detected_class"]
        assert row["price_reference_class"] == truth["price_only_confirmed_reference"]
        assert row["model_score"] == pytest.approx(
            max(truth["scores_uncalibrated"].values()), abs=1e-12)
    # Prices strictly after the first 4 batches cannot change those decisions
    # even though the whole dataset is preprocessed once for efficiency.
    changed = raw.copy()
    for c in ("open", "high", "low", "close"):
        changed.loc[180:, c] += 400
    altered_csv = tmp_path / "altered.csv"
    changed.to_csv(altered_csv, index=False)
    modified = run_reversal_walkforward(str(altered_csv), str(small_checkpoint),
                                        str(tmp_path / "modified.csv"), max_batches=4, echo=False)
    for left, right in zip(report["batches"], modified["batches"]):
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
    assert all(x["new_signal"] == "none" for x in result["batches"])
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
