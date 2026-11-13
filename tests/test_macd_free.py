"""v0.6 removes MACD from model inputs without changing legacy v0.5 checkpoints."""
import json
import numpy as np
import pytest

from finance_ai.atr import train_atr, predict_atr
from finance_ai.experiment import Settings
from finance_ai.features import PreparedData, FEATURE_NAMES, FEATURE_NAMES_NO_MACD, MACD_FEATURE_NAMES


def fake_data(n, count):
    rng = np.random.default_rng(14)
    return PreparedData(
        timestamps=1_745_362_800 + np.arange(n) * 900,
        features=rng.normal(size=(n, count)),
        closes=3300 + np.cumsum(rng.normal(0, 2, n)),
        atr14_ratio=np.full(n, 0.002),
    )


def test_macd_removed_from_schema_and_input_shape():
    assert len(FEATURE_NAMES) == 24
    assert len(FEATURE_NAMES_NO_MACD) == 21
    assert not set(MACD_FEATURE_NAMES) & set(FEATURE_NAMES_NO_MACD)
    assert len(FEATURE_NAMES_NO_MACD) * 20 + 5 == 425


def test_v06_trains_and_infers_without_macd_and_v05_still_loads(monkeypatch, tmp_path):
    newer = fake_data(260, 21)
    older = fake_data(260, 24)
    called = []

    def prepare(_path, *, include_macd=True):
        called.append(include_macd)
        return older if include_macd else newer

    monkeypatch.setattr('finance_ai.atr.prepare', prepare)
    settings = Settings(epochs=1, batch_size=64, seeds=(42,))
    v6_path = tmp_path / 'v06'
    v6 = train_atr('not-used.csv', str(v6_path), settings, hidden=(12, 6))
    assert v6['model_version'] == '0.6.0'
    assert v6['architecture'] == [425, 12, 6, 3]
    assert v6['internal_holdout_evaluated'] is False
    assert v6['external_final_test_evaluated'] is False
    assert v6['split_windows'] == {'train': 155, 'validation': 27, 'internal_holdout_reserved': 35}
    config = json.loads((v6_path / 'config.json').read_text())
    prep = json.loads((v6_path / 'preprocessing.json').read_text())
    assert config['features'] == list(FEATURE_NAMES_NO_MACD)
    assert 'macd' not in prep['indicator_settings']
    assert prep['version'] == 6 and len(prep['mean']) == 21
    assert len(prep['trend_mean']) == 5
    with np.load(v6_path / 'model.npz', allow_pickle=False) as ckpt:
        assert ckpt['w0'].shape == (425, 12)
    out = predict_atr('not-used.csv', str(v6_path))
    assert out['model_version'] == '0.6.0'
    assert sum(out['probabilities_uncalibrated'].values()) == pytest.approx(1)
    assert called == [False, False]

    v5_path = tmp_path / 'v05'
    v5 = train_atr('not-used.csv', str(v5_path), settings, hidden=(12, 6), include_macd=True)
    assert v5['model_version'] == '0.5.0'
    assert v5['architecture'] == [485, 12, 6, 3]
    old = predict_atr('not-used.csv', str(v5_path))
    assert old['model_version'] == '0.5.0'
    assert called == [False, False, True, True]


def test_v06_rejects_mismatched_checkpoint_schema(monkeypatch, tmp_path):
    monkeypatch.setattr('finance_ai.atr.prepare', lambda _, **kwargs: fake_data(260, 21))
    root = tmp_path / 'v06'
    train_atr('ignored.csv', str(root), Settings(epochs=1, seeds=(42,)), hidden=(6,))
    config_file = root/'config.json'
    cfg = json.loads(config_file.read_text())
    cfg['features'] = list(FEATURE_NAMES)
    config_file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Incompatible ATR'):
        predict_atr('ignored.csv', str(root))


def test_feature_extraction_without_macd_real_ta(tmp_path):
    pytest.importorskip('ta')
    import pandas as pd
    from finance_ai.features import prepare
    rng = np.random.default_rng(18)
    close = 3300 + np.cumsum(rng.normal(size=150))
    frame = pd.DataFrame({
        'time': 1_745_362_800 + np.arange(150)*900, 'open': close,
        'high': close+3, 'low': close-3, 'close': close,
        'volume': 100, 'timeframe': 15,
    })
    file = tmp_path/'input.csv'
    frame.to_csv(file, index=False)
    old, new = prepare(file), prepare(file, include_macd=False)
    assert old.features.shape[1] == 24
    assert new.features.shape[1] == 21
    np.testing.assert_array_equal(old.timestamps, new.timestamps)
    keep_indices = [i for i, name in enumerate(FEATURE_NAMES) if name not in MACD_FEATURE_NAMES]
    np.testing.assert_allclose(old.features[:, keep_indices], new.features)
    np.testing.assert_allclose(old.atr14_ratio, new.atr14_ratio)
