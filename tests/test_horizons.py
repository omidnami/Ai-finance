"""Horizon comparisons keep features fixed and future labels partitioned."""
import json
import numpy as np

from finance_ai.features import PreparedData, LOOKBACK
from finance_ai.horizons import build_comparison_samples, comparison_slices, compare_horizons
from finance_ai.experiment import Settings, predict


def synthetic_candles(n=260):
    rng = np.random.default_rng(2026)
    closes = 3300 + np.cumsum(rng.normal(0, 2, n))
    return PreparedData(
        timestamps=1_745_362_800 + np.arange(n) * 900,
        features=rng.normal(size=(n, 24)),
        closes=closes,
        atr14_ratio=np.full(n, .002),
    )


def test_horizon_labels_use_exact_future_closes_and_equal_inputs():
    data = synthetic_candles(400)
    x, extra, ends, labels = build_comparison_samples(data, (1, 4, 8))
    assert x.shape == (400 - LOOKBACK - 8 + 1, 20, 24)
    assert extra.shape == (len(x), 5)
    for h in (1, 4, 8):
        expected = (data.closes[ends+h] > data.closes[ends]).astype(float)
        np.testing.assert_array_equal(labels[h], expected)
    # Future prices do not affect the last observed model input or trend features.
    changed = synthetic_candles(400)
    changed.closes[ends[17]+1:] += 500
    x2, extra2, _, _ = build_comparison_samples(changed, (1, 4, 8))
    np.testing.assert_array_equal(x[:18], x2[:18])
    np.testing.assert_allclose(extra[:18], extra2[:18])


def test_purged_boundary_blocks_cross_split_targets():
    n = 400
    train, val, hold = comparison_slices(n, 8)
    assert train.stop + 8 < val.start + 1
    assert val.stop + 8 < hold.start + 1
    assert val.start > train.stop
    assert hold.start > val.stop


def test_compare_only_validation_and_checkpoints(monkeypatch, tmp_path):
    fake = synthetic_candles(260)
    monkeypatch.setattr('finance_ai.horizons.prepare', lambda _: fake)
    output = tmp_path / 'v03'
    result = compare_horizons('ignored.csv', str(output),
                             Settings(epochs=1, seeds=(42,), batch_size=32))
    assert result['selected_horizon'] in (1, 4, 8)
    assert result['internal_holdout_evaluated'] is False
    assert result['external_final_test_evaluated'] is False
    assert {r['horizon_candles'] for r in result['comparisons']} == {1, 4, 8}
    best = min(result['comparisons'], key=lambda d:d['validation']['loss'])
    assert result['selected_model'] == best['checkpoint']
    for r in result['comparisons']:
        directory = output / f"h{r['horizon_candles']}"
        config = json.loads((directory / 'config.json').read_text())
        assert config['horizon_candles'] == r['horizon_candles']
        assert config['architecture'][0] == 485
        assert not any('test' in key for key in r)
        assert (directory / 'model.npz').exists()
        assert (directory / 'preprocessing.json').exists()
    monkeypatch.setattr('finance_ai.experiment.prepare', lambda _: fake)
    assert 0 <= predict('ignored.csv', best['checkpoint']) <= 1
