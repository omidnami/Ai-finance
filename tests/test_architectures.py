"""v0.4 holds labels/features/boundaries fixed while shrinking network size."""
import json

import numpy as np
import pytest

from finance_ai.architectures import ARCHITECTURES, compare_architectures
from finance_ai.experiment import Settings, predict
from finance_ai.features import PreparedData
from finance_ai.horizons import build_comparison_samples, comparison_slices, compare_horizons
from finance_ai.network import MLP


def synthetic_candles(n=260):
    rng = np.random.default_rng(2026)
    return PreparedData(
        timestamps=1_745_362_800 + np.arange(n) * 900,
        features=rng.normal(size=(n, 24)),
        closes=3300 + np.cumsum(rng.normal(0, 2, n)),
        atr14_ratio=np.full(n, .002),
    )


def test_all_architectures_match_expected_sizes_and_can_train():
    assert ARCHITECTURES == {
        "large": (485, 128, 64, 32, 1),
        "medium": (485, 64, 32, 1),
        "small": (485, 32, 16, 1),
        "linear": (485, 1),
    }
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 485))
    y = rng.integers(0, 2, 8).astype(float)
    for sizes in ARCHITECTURES.values():
        model = MLP(sizes=sizes)
        assert model.forward(x).shape == (8,)
        loss = model.train_batch(x, y)
        assert np.isfinite(loss)
        assert model.forward(x).shape == (8,)


def test_v04_compares_only_validation_and_predicts_all_sizes(monkeypatch, tmp_path):
    fake = synthetic_candles()
    monkeypatch.setattr('finance_ai.architectures.prepare', lambda _: fake)
    out = tmp_path / 'finance-v0.4'
    report = compare_architectures('ignored.csv', str(out),
                                   Settings(epochs=1, seeds=(42,), batch_size=32))
    assert report['model_version'] == '0.4.0'
    assert report['horizon_candles'] == 1
    assert report['selection_metric'] == 'validation_loss'
    assert not report['internal_holdout_evaluated']
    assert not report['external_final_test_evaluated']
    assert {r['architecture_name'] for r in report['comparisons']} == set(ARCHITECTURES)
    assert report['selected_model'] == min(report['comparisons'],
                                           key=lambda r:r['validation']['loss'])['checkpoint']
    x, extra, ends, target = build_comparison_samples(fake, (1, 4, 8))
    t, v, h = comparison_slices(len(x), 8)
    assert report['split_windows'] == {
        'train': len(ends[t]), 'validation': len(ends[v]),
        'internal_holdout_reserved': len(ends[h]),
    }
    monkeypatch.setattr('finance_ai.experiment.prepare', lambda _: fake)
    for item in report['comparisons']:
        directory = out / item['architecture_name']
        config = json.loads((directory / 'config.json').read_text())
        pre = json.loads((directory / 'preprocessing.json').read_text())
        assert config['architecture'] == item['architecture']
        assert config['horizon_candles'] == 1
        assert pre['version'] == 4
        assert len(pre['mean']) == 24 and len(pre['trend_mean']) == 5
        assert (directory / 'model.npz').exists()
        assert 'test' not in item
        assert 0 <= predict('ignored.csv', str(directory)) <= 1
    assert json.loads((out / 'comparison.json').read_text()) == report


def test_large_network_is_identical_to_v03_h1_control(monkeypatch, tmp_path):
    data = synthetic_candles()
    monkeypatch.setattr('finance_ai.horizons.prepare', lambda _: data)
    monkeypatch.setattr('finance_ai.architectures.prepare', lambda _: data)
    cfg = Settings(epochs=1, seeds=(13,), batch_size=32)
    old = compare_horizons('ignored.csv', str(tmp_path / 'v03'), cfg)
    new = compare_architectures('ignored.csv', str(tmp_path / 'v04'), cfg)
    h1 = next(x for x in old['comparisons'] if x['horizon_candles'] == 1)
    large = next(x for x in new['comparisons'] if x['architecture_name'] == 'large')
    assert large['selected_epoch'] == h1['selected_epoch']
    assert large['validation'] == h1['validation']
    assert large['train'] == h1['train']
    assert new['split_windows'] == old['split_windows']


def test_holdout_prices_cannot_change_comparison_selection(monkeypatch, tmp_path):
    data = synthetic_candles()
    x, extra, ends, targets = build_comparison_samples(data, (1, 4, 8))
    _, _, hold = comparison_slices(len(x), 8)
    first_holdout_end = ends[hold.start]
    mutated = synthetic_candles()
    mutated.closes[first_holdout_end:] += 100000
    cfg = Settings(epochs=1, seeds=(97,), batch_size=64)
    monkeypatch.setattr('finance_ai.architectures.prepare', lambda _: data)
    first = compare_architectures('ignored.csv', str(tmp_path / 'a'), cfg)
    monkeypatch.setattr('finance_ai.architectures.prepare', lambda _: mutated)
    second = compare_architectures('ignored.csv', str(tmp_path / 'b'), cfg)
    assert first['selected_architecture'] == second['selected_architecture']
    for a, b in zip(first['comparisons'], second['comparisons']):
        assert a['validation'] == b['validation']
        assert a['train'] == b['train']


def test_invalid_settings_rejected_before_data_read(monkeypatch, tmp_path):
    monkeypatch.setattr('finance_ai.architectures.prepare', lambda _: pytest.fail('read data'))
    with pytest.raises(ValueError, match='Invalid training'):
        compare_architectures('x.csv', str(tmp_path), Settings(epochs=0))
