import numpy as np
import pandas as pd
import pytest

from finance_ai.features import FEATURE_NAMES, read_candles, prepare, labeled_windows


def _excel(tmp_path, n=260, gap=False):
    rng = np.random.default_rng(4)
    t = 1745362800 + np.arange(n) * 900
    if gap:
        t[100:] += 7200
    prices = 3300 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame({"time": t, "open": prices, "high": prices+3,
                       "low": prices-3, "close": prices+rng.uniform(-1, 1, n),
                       "volume": rng.integers(50, 200, n), "timeframe": 15})
    path = tmp_path / "candles.xlsx"
    df.to_excel(path, index=False)
    return path


def test_excel_feature_pipeline_and_gaps(tmp_path):
    pytest.importorskip("ta")
    source = _excel(tmp_path, gap=True)
    data = prepare(source)
    assert data.features.shape[1] == len(FEATURE_NAMES) == 24
    assert np.isfinite(data.features).all()
    assert np.max(np.diff(data.timestamps)) > 900
    x, y, times = labeled_windows(data)
    assert x.shape[1:] == (20, 24)
    assert len(y) == len(times) == len(x)


def test_invalid_timeframe_rejected(tmp_path):
    path = _excel(tmp_path)
    frame = pd.read_excel(path)
    frame.loc[0, "timeframe"] = 5
    frame.to_excel(path, index=False)
    with pytest.raises(ValueError, match="timeframe=15"):
        read_candles(path)


def test_csv_import_matches_excel(tmp_path):
    excel_path = _excel(tmp_path, gap=True)
    csv_path = tmp_path / "candles.csv"
    pd.read_excel(excel_path).to_csv(csv_path, index=False)
    excel = read_candles(excel_path)
    csv = read_candles(csv_path)
    pd.testing.assert_frame_equal(excel, csv)


def test_csv_feature_pipeline(tmp_path):
    pytest.importorskip("ta")
    excel_path = _excel(tmp_path, gap=True)
    csv_path = tmp_path / "candles.csv"
    pd.read_excel(excel_path).to_csv(csv_path, index=False)
    data = prepare(csv_path)
    assert data.features.shape[1] == 24
    assert np.isfinite(data.features).all()
    assert np.max(np.diff(data.timestamps)) > 900
