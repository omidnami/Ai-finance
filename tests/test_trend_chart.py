"""Validation-only visual review for v0.7: no reserved holdout or forecast."""

import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.experiment import Settings, split_positions
from finance_ai.trend_chart import plot_validation_trend, validation_chart_data
from finance_ai.trend_detection import LOOKBACK, prepare_trend, train_trend
from test_trend_detection import candles


def test_validation_plot_and_sample_boundaries(tmp_path):
    pytest.importorskip('matplotlib')
    source = candles(tmp_path, n=260)
    model = tmp_path / 'model'
    report = train_trend(str(source), str(model), Settings(epochs=1, seeds=(42,)), hidden=(16, 8))
    assert report['internal_holdout_evaluated'] is False
    chart = validation_chart_data(str(source), str(model), candles=20)
    data = prepare_trend(source)
    a, b = split_positions(len(data.closes) - LOOKBACK + 1)
    assert chart['validation_sample_range'] == [b - 20, b]
    assert chart['train_end_sample'] == a
    assert chart['reserved_holdout_start_sample'] == b
    np.testing.assert_array_equal(chart['timestamps'], data.timestamps[np.arange(b-20, b)+LOOKBACK-1])
    assert len(chart['detected']) == 20
    assert set(np.unique(chart['detected'])) <= {0, 1}
    assert set(np.unique(chart['reference'])) <= {0, 1}
    assert np.isfinite(chart['scores_uncalibrated']).all()
    output = tmp_path/'chart.png'
    metadata = plot_validation_trend(str(source), str(model), str(output), candles=20)
    assert output.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert output.stat().st_size > 10000
    assert metadata['candles'] == 20
    assert metadata['segment'] == 'development_validation_only'
    assert metadata['internal_holdout_evaluated'] is False
    assert metadata['external_final_test_evaluated'] is False
    assert metadata['ai_vs_historical_slope_agreement'] == pytest.approx(
        np.mean(chart['detected'] == chart['reference']))


def test_reject_invalid_count_and_model_schema(tmp_path):
    source = candles(tmp_path, n=260)
    path = tmp_path/'checkpoint'
    train_trend(str(source), str(path), Settings(epochs=1, seeds=(13,)), hidden=(10,))
    with pytest.raises(ValueError, match='positive integer'):
        validation_chart_data(str(source), str(path), 0)
    with pytest.raises(ValueError, match='validation has only'):
        validation_chart_data(str(source), str(path), 100)
    cfg_path = path/'config.json'
    cfg = json.loads(cfg_path.read_text())
    cfg['lookback'] = 20
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Incompatible v0.7'):
        validation_chart_data(str(source), str(path), 10)


def test_mutating_reserved_holdout_does_not_change_validation_chart(tmp_path):
    source = candles(tmp_path, n=260)
    model = tmp_path/'model'
    train_trend(str(source), str(model), Settings(epochs=1, seeds=(42,)), hidden=(10,))
    previous = validation_chart_data(str(source), str(model), candles=20)
    raw = pd.read_csv(source)
    holdout_boundary = previous['timestamps'][-1]
    # Change every candle strictly AFTER the validation window (preserve OHLC).
    raw.loc[raw['time'] > holdout_boundary, ['open','high','low','close']] += 500
    changed = tmp_path/'holdout_modified.csv'
    raw.to_csv(changed, index=False)
    after = validation_chart_data(str(changed), str(model), candles=20)
    for key in ('timestamps','open','high','low','close','ema10','bb_middle',
                'bb_upper','bb_lower','detected','reference','scores_uncalibrated','weak'):
        np.testing.assert_allclose(previous[key], after[key], rtol=1e-12, atol=1e-10)
