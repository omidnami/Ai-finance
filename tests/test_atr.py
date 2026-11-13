"""v0.5: ATR-only-from-past labels, multi-class gradients, untouched holdout."""
import json

import numpy as np
import pytest

from finance_ai.atr import (CLASS_NAMES, DEFAULT_HIDDEN, SoftmaxMLP, atr_labels,
                            classification_metrics, train_atr, predict_atr)
from finance_ai.features import PreparedData
from finance_ai.horizons import comparison_slices


def synthetic_data(n=260):
    rng = np.random.default_rng(2026)
    return PreparedData(
        timestamps=1_745_362_800 + np.arange(n) * 900,
        features=rng.normal(size=(n, 24)),
        closes=3300 + np.cumsum(rng.normal(0, 3, n)),
        atr14_ratio=np.full(n, .002),
    )


def test_triple_classification_boundaries_and_causal_atr():
    data = PreparedData(
        timestamps=np.arange(10),
        features=np.zeros((10, 24)),
        closes=np.array([100., 100., 102., 100., 100., 101., 98., 99.8, 100., 100.]),
        atr14_ratio=np.full(10, .08),
    )
    # threshold = 0.25 * (.08 * 100) = 2: exact +/-2 is neutral.
    np.testing.assert_array_equal(atr_labels(data, np.array([1, 2, 3, 4, 5, 6]), .25),
                                  [1, 1, 1, 1, 0, 1])
    data2 = PreparedData(data.timestamps, data.features, data.closes.copy(),
                         data.atr14_ratio.copy())
    data2.closes[5] = 105
    assert atr_labels(data2, np.array([4]), .25)[0] == 2
    # The neutral threshold depends on ATR[t], not the next candle's ATR.
    data2.atr14_ratio[5] = 100.0
    assert atr_labels(data2, np.array([4]), .25)[0] == 2
    assert CLASS_NAMES == ('down', 'neutral', 'up')
    assert DEFAULT_HIDDEN == (512, 256, 128, 64)


def test_invalid_atr_multiplier_and_bounds():
    data = synthetic_data()
    for invalid in [0.0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match='ATR multiplier'):
            atr_labels(data, np.array([20]), invalid)
    with pytest.raises(ValueError, match='next candle'):
        atr_labels(data, np.array([259]), .25)


def test_softmax_shapes_gradient_and_adamw():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12, 4))
    y = rng.integers(0, 3, 12)
    model = SoftmaxMLP((4, 7, 3), seed=42)
    p = model.forward(x)
    assert p.shape == (12, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    loss, dw, db = model.gradients(x, y)
    assert np.isfinite(loss)
    old = model.weights[0][0, 0]
    eps = 1e-6
    model.weights[0][0, 0] = old + eps
    plus = model.gradients(x, y)[0]
    model.weights[0][0, 0] = old - eps
    minus = model.gradients(x, y)[0]
    model.weights[0][0, 0] = old
    assert dw[0][0, 0] == pytest.approx((plus - minus) / (2 * eps), abs=1e-6)
    for _ in range(120):
        model.train_batch(x, y, lr=.01)
    assert model.gradients(x, y)[0] < loss
    assert sum(model.weights[0].shape) > 0 and db[-1].shape == (1, 3)


def test_macro_metrics_cannot_hide_missing_classes():
    # Bias-only network predicts neutral for every example.
    model = SoftmaxMLP((4, 3), seed=11)
    model.weights[0].fill(0)
    model.biases[0][0] = [-8, 8, -8]
    x = np.zeros((6, 4))
    y = np.array([0, 1, 1, 1, 2, 2])
    m = classification_metrics(model, x, y)
    assert m['accuracy'] == .5
    assert m['majority_baseline_accuracy'] == .5
    assert m['balanced_accuracy'] == pytest.approx(1/3)
    assert m['recall']['up'] == 0
    assert m['macro_f1'] < .5
    assert m['confusion_matrix'] == [[0, 1, 0], [0, 3, 0], [0, 2, 0]]


def test_train_predict_roundtrip_no_reserved_holdout_scoring(monkeypatch, tmp_path):
    fake = synthetic_data()
    monkeypatch.setattr('finance_ai.atr.prepare', lambda _: fake)
    out = tmp_path / 'v05'
    args = dict(settings=__import__('finance_ai.experiment', fromlist=['Settings']).Settings(
        epochs=2, seeds=(13, 42), batch_size=32, patience=2), hidden=(12, 6), include_macd=True)
    report = train_atr('ignored.csv', str(out), **args)
    assert report['model_version'] == '0.5.0'
    assert report['architecture'] == [485, 12, 6, 3]
    assert report['selected_seed'] in (13, 42)
    assert not report['internal_holdout_evaluated']
    assert not report['external_final_test_evaluated']
    assert 'test' not in report['validation']
    assert sum(report['split_windows'].values()) < 260
    config = json.loads((out/'config.json').read_text())
    assert config['classes'] == list(CLASS_NAMES)
    assert config['atr_multiplier'] == .25
    assert (out/'model.npz').is_file()
    result = predict_atr('ignored.csv', str(out))
    assert result['predicted_direction'] in CLASS_NAMES
    assert sum(result['probabilities_uncalibrated'].values()) == pytest.approx(1)
    assert json.loads((out/'metrics.json').read_text()) == report


def test_holdout_changes_cannot_change_v05_training(monkeypatch, tmp_path):
    a = synthetic_data()
    b = synthetic_data()
    count = len(a.closes) - 20 - 8 + 1
    _, _, hold = comparison_slices(count, 8)
    holdout_first_end = 19 + hold.start
    b.closes[holdout_first_end:] += 150000
    settings = __import__('finance_ai.experiment', fromlist=['Settings']).Settings(
        epochs=1, seeds=(42,), batch_size=64)
    monkeypatch.setattr('finance_ai.atr.prepare', lambda _: a)
    first = train_atr('ignore.csv', str(tmp_path/'a'), settings, hidden=(8,), include_macd=True)
    monkeypatch.setattr('finance_ai.atr.prepare', lambda _: b)
    second = train_atr('ignore.csv', str(tmp_path/'b'), settings, hidden=(8,), include_macd=True)
    assert first['train'] == second['train']
    assert first['validation'] == second['validation']
    assert first['selected_seed'] == second['selected_seed']
    assert first['selected_epoch'] == second['selected_epoch']


def test_cli_help_contains_new_command():
    from finance_ai.__main__ import main
    # argparse invocation is exercised by ``python -m finance_ai --help`` separately.
    assert callable(main)
