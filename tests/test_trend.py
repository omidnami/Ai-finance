import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.trend import TREND_FEATURE_NAMES, summary_at
from finance_ai.experiment import fit_scaler, fit_trend_scaler, transform_v2, train, predict, Settings
from finance_ai.features import prepare, labeled_windows_v2, latest_window_v2
from finance_ai.network import LAYERS


def _sample_csv(tmp_path, n=260):
    rng = np.random.default_rng(4)
    time = 1745362800 + np.arange(n) * 900
    close = 3330 + np.cumsum(rng.normal(0, 1.5, n))
    opening = close + rng.normal(0, .15, n)
    high = np.maximum(opening, close) + 1
    low = np.minimum(opening, close) - 1
    frame = pd.DataFrame({"time": time, "open": opening, "high": high, "low": low,
                          "close": close, "volume": rng.integers(50, 200, n),
                          "timeframe": 15})
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_summary_calculation_and_flat_market():
    closes = np.arange(100., 120.)
    atr = np.full(20, 0.012)
    out = summary_at(closes, atr, end=19)
    assert len(TREND_FEATURE_NAMES) == 5
    np.testing.assert_allclose(out, [119/114-1, 119/109-1, 119/100-1, 1.0, 0.012])
    flat = summary_at(np.ones(20)*100, atr, 19)
    np.testing.assert_allclose(flat[:4], 0.0)
    with pytest.raises(ValueError, match="at least 20"):
        summary_at(closes, atr, end=18)


def test_strength_range_and_direction():
    zigzag = np.array([100, 104, 98, 105, 97, 106, 99, 103, 95, 108,
                       100, 110, 101, 111, 99, 112, 105, 106, 104, 102.], dtype=float)
    trend = summary_at(zigzag, np.ones(20)*.01, 19)
    assert 0 <= trend[3] <= 1
    assert trend[3] < 1
    assert trend[2] > 0


def test_scaling_uses_only_training_windows():
    x = np.ones((4, 20, 24))
    extra = np.ones((4, 5))
    mu, std = fit_scaler(x)
    trend_mu, trend_std = fit_trend_scaler(extra)
    assert np.all(mu == 1) and np.all(trend_mu == 1)
    future = transform_v2(np.ones((1, 20, 24))*1000, np.ones((1, 5))*1000,
                          mu, std, trend_mu, trend_std)
    assert future.shape == (1, 485)
    assert np.max(future) == 10


def test_v2_windows_are_causal_and_training_inference_match(tmp_path):
    pytest.importorskip("ta")
    whole_path = _sample_csv(tmp_path, n=260)
    whole = prepare(whole_path)
    x, extra, y, t = labeled_windows_v2(whole)
    assert x.shape[1:] == (20, 24)
    assert extra.shape == (len(x), 5)
    assert len(x) == len(y) == len(t)
    assert np.isfinite(extra).all()
    last_x, last_extra = latest_window_v2(whole)
    np.testing.assert_allclose(last_x, whole.features[-20:])
    assert last_extra.shape == (5,)
    # Alter all future bars: old past-derived summaries MUST remain unchanged.
    df = pd.read_csv(whole_path)
    short_path = tmp_path / "prefix.csv"
    df.iloc[:180].to_csv(short_path, index=False)
    prefix = prepare(short_path)
    px, pe = latest_window_v2(prefix)
    # For whole, same 20 completed candles end at index 179.
    relative_end = int(np.searchsorted(whole.timestamps, prefix.timestamps[-1]))
    np.testing.assert_allclose(px, whole.features[relative_end-19:relative_end+1])
    from finance_ai.trend import summary_at
    np.testing.assert_allclose(pe, summary_at(whole.closes, whole.atr14_ratio, relative_end))


def test_v2_train_save_predict_and_legacy_v1_compatibility(tmp_path):
    pytest.importorskip("ta")
    source = _sample_csv(tmp_path, n=260)
    new_dir = tmp_path / "finance-v0.2"
    result = train(source, str(new_dir), Settings(epochs=1, seeds=(42,), batch_size=64))
    assert result["input_features"] == LAYERS[0] == 485
    config = json.loads((new_dir / "config.json").read_text())
    prep = json.loads((new_dir / "preprocessing.json").read_text())
    assert config["trend_features"] == list(TREND_FEATURE_NAMES)
    assert prep["version"] == 2 and len(prep["trend_mean"]) == 5
    assert 0.0 <= predict(source, str(new_dir)) <= 1.0


def test_old_v01_checkpoint_still_loads(tmp_path):
    pytest.importorskip("ta")
    from finance_ai.features import FEATURE_NAMES
    from finance_ai.network import MLP

    source = _sample_csv(tmp_path, n=260)
    old_dir = tmp_path / "finance-v0.1"
    old_dir.mkdir()
    model = MLP(sizes=(480, 128, 64, 32, 1), seed=23)
    np.savez_compressed(old_dir / "model.npz", **model.state_dict())
    (old_dir / "config.json").write_text(json.dumps({
        "architecture": [480, 128, 64, 32, 1], "lookback": 20,
        "timeframe": 15, "features": list(FEATURE_NAMES)}))
    (old_dir / "preprocessing.json").write_text(json.dumps({
        "version": 1, "mean": [0.0]*24, "std": [1.0]*24}))
    p = predict(source, str(old_dir))
    assert np.isfinite(p) and 0.0 <= p <= 1.0
