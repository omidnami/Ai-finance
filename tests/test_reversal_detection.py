"""v0.8 retrospective reversal candidates are independent from v0.7 and causal."""
import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.experiment import Settings, fit_scaler
from finance_ai.reversal_detection import (
    VERSION, FEATURES, CLASSES, LOOKBACK, PURGE, BalancedSoftmax,
    _selections, prepare_reversal, reversal_reference, reversal_samples,
    train_reversal, detect_reversal,
)
from finance_ai.reversal_chart import reversal_validation_data, plot_validation_reversals


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    root = tmp_path_factory.mktemp("swing")
    path = root / "development.csv"
    rng = np.random.default_rng(8)
    n = 1600
    t = np.arange(n)
    close = 3300 + 25*np.sin(2*np.pi*t/45) + 2.2*np.sin(2*np.pi*t/8) + rng.normal(0, .5, n)
    opened = np.r_[close[0], close[:-1]]+rng.normal(0, .2, n)
    high = np.maximum(opened, close)+rng.uniform(.3, 1.4, n)
    low = np.minimum(opened, close)-rng.uniform(.3, 1.4, n)
    pd.DataFrame({
        "time": 1_745_362_800+t*900, "open": opened,
        "high": high, "low": low, "close": close,
        "volume": rng.integers(100, 1000, n), "timeframe": 15,
    }).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def trained(synthetic_csv, tmp_path_factory):
    root = tmp_path_factory.mktemp("swing_model")
    report = train_reversal(str(synthetic_csv), str(root),
                            Settings(epochs=1, seeds=(42,), batch_size=64),
                            hidden=(16, 8))
    return root, report


def test_feature_schema_and_causal_label_outcomes(synthetic_csv):
    data = prepare_reversal(synthetic_csv)
    x, y, strength, ends = reversal_samples(data)
    assert VERSION == "0.8.0"
    assert len(FEATURES) == 20 and x.shape[1:] == (40, 20)
    assert set(np.unique(y)) == {0, 1, 2}
    assert len(x) == len(ends) == len(y) == len(strength)
    assert len(CLASSES) == 3
    # Labels NEVER mean future direction; bullish signals require previous decline.
    assert np.all(strength[y == 1] < 0)
    assert np.all(strength[y == 2] > 0)
    assert not np.isnan(x).any()
    assert "macd_hist_ratio" in FEATURES and "rsi14" in FEATURES
    assert "bb_upper_touch" in FEATURES and "bb_lower_touch" in FEATURES


def test_training_roundtrip_saved_schema_reserved_splits(trained, synthetic_csv):
    root, report = trained
    assert report["task"].startswith("past_only_confirmed_swing_reversal")
    assert report["reference_is_proxy_not_ground_truth"]
    assert report["reference_causal"]
    assert report["internal_holdout_evaluated"] is False
    assert report["external_final_test_evaluated"] is False
    assert report["split_windows"]["purge_each_boundary"] == PURGE
    assert report["architecture"] == [LOOKBACK*len(FEATURES), 16, 8, 3]
    assert report["validation"]["samples"] > 100
    assert sum(report["validation"]["reference_counts"].values()) == report["validation"]["samples"]
    pre = json.loads((root/"preprocessing.json").read_text())
    cfg = json.loads((root/"config.json").read_text())
    assert cfg["model_version"] == VERSION
    assert cfg["features"] == list(FEATURES)
    assert cfg["indicators"]["macd"] == [12, 26, 9]
    assert pre["version"] == 8
    assert len(pre["mean"]) == len(FEATURES)
    assert len(pre["class_weights"]) == 3
    assert pre["class_weights"][1] > pre["class_weights"][0]
    data = prepare_reversal(synthetic_csv)
    x, _, _, _ = reversal_samples(data)
    train_sel, val_sel, hold_sel = _selections(len(x))
    np.testing.assert_allclose(pre["mean"], fit_scaler(x[train_sel])[0])
    assert val_sel.start-train_sel.stop == PURGE
    assert hold_sel.start-val_sel.stop == PURGE
    out = detect_reversal(str(synthetic_csv), str(root))
    assert out["model_version"] == VERSION
    assert out["detected_class"] in CLASSES
    assert out["price_only_confirmed_reference"] in CLASSES
    assert sum(out["scores_uncalibrated"].values()) == pytest.approx(1.0)
    assert out["candidate_pivot_candle_utc"] < out["last_closed_candle_utc"]
    assert isinstance(out["indicators"]["bollinger_upper_touch"], bool)


def test_past_point_unchanged_after_future_data_changes(synthetic_csv, tmp_path):
    full = pd.read_csv(synthetic_csv)
    # 1) historical data up through time t cannot change when later candles arrive.
    prefix = full.iloc[:750].copy()
    another = full.copy()
    another.loc[750:, "close"] *= 1.25
    another.loc[750:, "open"] *= 1.25
    another.loc[750:, "high"] *= 1.30
    another.loc[750:, "low"] *= 1.20
    partial_path = tmp_path/"early.csv"
    modified_path = tmp_path/"modified.csv"
    prefix.to_csv(partial_path, index=False)
    another.to_csv(modified_path, index=False)
    a = prepare_reversal(partial_path)
    b = prepare_reversal(modified_path)
    assert len(b.features) > len(a.features)
    np.testing.assert_array_equal(a.timestamps, b.timestamps[:len(a.timestamps)])
    np.testing.assert_allclose(a.features, b.features[:len(a.features)], atol=0, rtol=0)
    end = np.array([len(a.close)-1])
    ay, as_ = reversal_reference(a, end)
    by, bs_ = reversal_reference(b, end)
    np.testing.assert_array_equal(ay, by)
    np.testing.assert_array_equal(as_, bs_)


def test_training_labels_depend_on_price_not_on_rsi_bollinger_or_macd(synthetic_csv):
    d = prepare_reversal(synthetic_csv)
    x, y, strength, ends = reversal_samples(d)
    rng = np.random.default_rng(123)
    new_feats = d.features.copy()
    new_feats[:, [FEATURES.index(s) for s in ("rsi14", "bb_percent_b", "macd_hist_ratio")]] = rng.normal(
        size=(len(new_feats), 3))
    from finance_ai.reversal_detection import ReversalData
    modified = ReversalData(d.timestamps, new_feats, d.close, d.high, d.low, d.atr)
    y2, strength2 = reversal_reference(modified, ends)
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(strength, strength2)


def test_weighted_cross_entropy_gradient_matches_numeric_difference():
    rng = np.random.default_rng(5)
    m = BalancedSoftmax((4, 6, 3), np.array([1., 4., 3.]), seed=5)
    x = rng.normal(size=(9, 4))
    y = np.array([0, 1, 2, 0, 0, 1, 0, 2, 1])
    loss, gw, gb = m.gradients(x, y)
    assert np.isfinite(loss)
    before = m.weights[0][0, 0]
    h = 1e-6
    m.weights[0][0, 0] = before+h
    plus = m.gradients(x, y)[0]
    m.weights[0][0, 0] = before-h
    minus = m.gradients(x, y)[0]
    m.weights[0][0, 0] = before
    assert gw[0][0, 0] == pytest.approx((plus-minus)/(2*h), rel=1e-4, abs=1e-5)


def test_validation_chart_has_only_validation_prices_and_markers(trained, synthetic_csv, tmp_path):
    pytest.importorskip("matplotlib")
    model, _ = trained
    c = reversal_validation_data(str(synthetic_csv), str(model), 100)
    assert len(c["predicted"]) == len(c["reference"]) == 100
    assert len(c["timestamps"]) == 100
    assert c["validation_sample_range"][1] == c["validation_end_sample"]
    assert c["validation_end_sample"] < c["holdout_start_sample"]
    assert c["validation_sample_range"][0] >= c["train_end_sample"]
    path = tmp_path/"chart.png"
    report = plot_validation_reversals(str(synthetic_csv), str(model), str(path), 100)
    assert path.exists() and path.stat().st_size > 50_000
    assert report["validation_only"] is True
    assert report["internal_holdout_evaluated"] is False
    assert report["external_final_test_evaluated"] is False
    with pytest.raises(ValueError, match="validation candles"):
        reversal_validation_data(str(synthetic_csv), str(model), 999_999)


def test_rejects_corrupt_schema_inference(trained, synthetic_csv, tmp_path):
    root, _ = trained
    checkpoint = tmp_path/"broken"
    checkpoint.mkdir()
    for name in ("model.npz", "preprocessing.json", "config.json"):
        (checkpoint/name).write_bytes((root/name).read_bytes())
    cfg = json.loads((checkpoint/"config.json").read_text())
    cfg["features"] = cfg["features"][:-1]
    (checkpoint/"config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Incompatible v0.8"):
        detect_reversal(str(synthetic_csv), str(checkpoint))
