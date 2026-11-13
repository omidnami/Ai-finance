"""v0.7 current-regime classifier: historical-only, binary, independent of v0.6."""
import json

import numpy as np
import pandas as pd
import pytest

from finance_ai.experiment import Settings
from finance_ai.trend_detection import (
    MODEL_VERSION, FEATURE_NAMES, LOOKBACK, LABEL_PERIOD, WEAK_SLOPE_ATR,
    aroon_channels, wilder_atr, prepare_trend, price_trend_reference,
    trend_samples, train_trend, detect_trend,
)


def candles(tmp_path, n=260, mode='mixed', filename='candles.csv'):
    t = 1745362800 + np.arange(n) * 900
    if mode == 'up':
        close = 3000 + np.arange(n) * 0.8
    elif mode == 'down':
        close = 3200 - np.arange(n) * 0.6
    elif mode == 'flat':
        close = np.full(n, 3200.)
    else:
        rng = np.random.default_rng(8)
        close = 3300 + np.cumsum(rng.normal(0.03, 1.8, n))
    close = close.astype(float)
    frame = pd.DataFrame({
        'time': t, 'open': close, 'high': close + 1.2,
        'low': close - 1.2, 'close': close,
        'volume': 100 + np.arange(n) % 100, 'timeframe': 15,
    })
    path = tmp_path / filename
    frame.to_csv(path, index=False)
    return path


def test_aroon_25_uses_current_and_previous_25_bars():
    hi, lo = np.arange(40.) + 200, np.arange(40.) + 199
    up, down = aroon_channels(hi, lo)
    assert np.isnan(up[:25]).all() and np.isnan(down[:25]).all()
    assert up[25] == 1.0 and down[25] == 0.0
    up2, down2 = aroon_channels(hi[::-1], lo[::-1])
    assert up2[25] == 0.0 and down2[25] == 1.0
    # Repeated extremes: np.argmax returns oldest occurrence, matching ta.
    flat_up, flat_down = aroon_channels(np.ones(27), np.ones(27))
    assert flat_up[25] == 0 and flat_down[25] == 0


def test_wilder_atr_seed_and_recursive_smoothing():
    close = np.ones(30) * 100
    high, low = close + 1, close - 1
    atr = wilder_atr(high, low, close)
    assert np.isnan(atr[:13]).all()
    np.testing.assert_allclose(atr[13:], 2.0)
    high[14] = 104
    atr2 = wilder_atr(high, low, close)
    assert atr2[14] == pytest.approx((2*13 + 5)/14)
    assert atr2[15] == pytest.approx((atr2[14]*13 + 2)/14)


def test_indicator_features_no_macd_and_flat_band(tmp_path):
    data = prepare_trend(candles(tmp_path, mode='flat'))
    assert data.features.shape[1] == len(FEATURE_NAMES) == 15
    assert np.isfinite(data.features).all()
    assert not any('macd' in key or 'rsi' in key for key in FEATURE_NAMES)
    np.testing.assert_allclose(data.features[:, FEATURE_NAMES.index('bb_percent_b')], .5)
    np.testing.assert_allclose(data.features[:, FEATURE_NAMES.index('bb_bandwidth')], 0)
    np.testing.assert_allclose(data.features[:, FEATURE_NAMES.index('ema10_vs_bb_middle')], 0)
    assert len(data.features) >= 200


def test_up_down_indicators_and_current_reference(tmp_path):
    up = prepare_trend(candles(tmp_path, mode='up', filename='up.csv'))
    down = prepare_trend(candles(tmp_path, mode='down', filename='down.csv'))
    au = FEATURE_NAMES.index('aroon_up')
    ad = FEATURE_NAMES.index('aroon_down')
    cross = FEATURE_NAMES.index('ema10_vs_bb_middle')
    assert up.features[-1, au] == 1 and up.features[-1, ad] == 0
    assert down.features[-1, au] == 0 and down.features[-1, ad] == 1
    assert up.features[-1, cross] > 0
    assert down.features[-1, cross] < 0
    y_up, strength_up = price_trend_reference(up, np.array([len(up.closes)-1]))
    y_down, strength_down = price_trend_reference(down, np.array([len(down.closes)-1]))
    assert y_up[0] == 1 and strength_up[0] > WEAK_SLOPE_ATR
    assert y_down[0] == 0 and strength_down[0] < -WEAK_SLOPE_ATR
    x, y, strength, ends = trend_samples(up)
    assert x.shape == (len(up.closes)-LOOKBACK+1, LOOKBACK, len(FEATURE_NAMES))
    assert np.all(y == 1) and np.all(strength > 0)
    assert ends[-1] == len(up.closes)-1


def test_future_candles_do_not_affect_past_features_or_targets(tmp_path):
    whole = candles(tmp_path, n=260)
    frame = pd.read_csv(whole)
    prefix = tmp_path / 'prefix.csv'
    frame.iloc[:180].to_csv(prefix, index=False)
    a, b = prepare_trend(whole), prepare_trend(prefix)
    np.testing.assert_array_equal(a.timestamps[:len(b.timestamps)], b.timestamps)
    np.testing.assert_allclose(a.features[:len(b.features)], b.features, atol=1e-12)
    # The newest historical label for the prefix must not use bar 181.
    ia = len(b.closes)-1
    np.testing.assert_allclose(price_trend_reference(a, np.array([ia]))[1],
                               price_trend_reference(b, np.array([ia]))[1])
    # Deliberately rewrite every future OHLCV; older samples remain identical.
    frame.loc[180:, ['open', 'high', 'low', 'close']] += 800
    changed = tmp_path / 'changed.csv'
    frame.to_csv(changed, index=False)
    c = prepare_trend(changed)
    np.testing.assert_allclose(a.features[:len(b.features)], c.features[:len(b.features)], atol=1e-12)
    np.testing.assert_allclose(price_trend_reference(a, np.array([ia]))[1],
                               price_trend_reference(c, np.array([ia]))[1])


def test_binary_train_save_inference_and_holdout(monkeypatch, tmp_path):
    source = candles(tmp_path)
    out = tmp_path / 'v07'
    config = Settings(epochs=2, seeds=(13, 42), patience=2, batch_size=64)
    result = train_trend(str(source), str(out), config, hidden=(12, 6))
    assert result['model_version'] == MODEL_VERSION
    assert result['task'] == 'current_trend_classification_not_forecasting'
    assert result['architecture'] == [600, 12, 6, 1]
    assert result['reference_is_proxy_not_ground_truth'] is True
    assert result['selected_seed'] in (13,42)
    assert result['internal_holdout_evaluated'] is False
    assert result['external_final_test_evaluated'] is False
    assert result['split_windows']['internal_holdout_reserved'] > 0
    assert 0 <= result['validation']['persistence_baseline_accuracy'] <= 1
    assert 0 <= result['validation']['ema10_vs_bb_middle_baseline_accuracy'] <= 1
    prep = json.loads((out/'preprocessing.json').read_text())
    assert prep['version'] == 7 and len(prep['mean']) == 15
    saved = json.loads((out/'config.json').read_text())
    assert saved['features'] == list(FEATURE_NAMES)
    assert saved['indicators'] == {'bollinger':[20, 2.0], 'ema':10,'aroon':25,'atr':14}
    with np.load(out/'model.npz', allow_pickle=False) as ckpt:
        assert ckpt['w0'].shape == (600, 12)
    observation = detect_trend(str(source), str(out))
    assert observation['trend'] in ('downtrend', 'uptrend')
    assert abs(observation['up_score_uncalibrated'] + observation['down_score_uncalibrated']-1)<1e-12
    assert observation['last_closed_candle_utc'].endswith('+00:00')
    assert observation['indicators']['ema10_bb_middle_cross'] in ('none','bullish','bearish')
    assert not any(word in observation for word in ('reversal','entry','future_prediction'))
    assert json.loads((out/'metrics.json').read_text()) == result
    saved['indicators']['ema'] = 9
    (out/'config.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Incompatible v0.7'):
        detect_trend(str(source), str(out))


def test_does_not_touch_external_test_file(tmp_path):
    path = candles(tmp_path)
    test = tmp_path/'final_test.csv'
    test.write_text('DO NOT TOUCH THIS FILE')
    train_trend(str(path), str(tmp_path/'trained'), Settings(epochs=1, seeds=(13,)), hidden=(8,))
    assert test.read_text() == 'DO NOT TOUCH THIS FILE'


def test_current_trend_cli_is_available():
    from finance_ai.__main__ import main
    assert callable(main)
