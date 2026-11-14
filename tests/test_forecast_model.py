import numpy as np
import pandas as pd
import pytest
pytest.importorskip('torch')
from finance_ai.forecast_model import (TrendForecastSettings, inputs, trend_strength,
    categorise, candidates, splits, train_forecast, predict_forecast,
    test_forecast as replay_forecast, CLASSES)
from test_decision_model import candles


def test_label_and_causal_features():
    raw,_=candles(400)
    cfg=TrendForecastSettings()
    f,atr=inputs(raw)
    i=200
    for sign,y in ((1,2),(-1,0),(0,1)):
        changed=raw.copy()
        changed.loc[i+1:i+cfg.horizon,'close']=raw.close.iloc[i]+sign*atr[i]*np.arange(1,cfg.horizon+1)
        z=trend_strength(changed,[i],atr,cfg)
        assert categorise(z,cfg.neutral_atr)[0]==y
        np.testing.assert_array_equal(inputs(changed)[0][60:i+1],f[60:i+1])


def test_purge_and_gap():
    raw,_=candles(1400)
    cfg=TrendForecastSettings()
    raw.loc[700:,'time']+=900
    ix=candidates(raw,cfg)
    assert 699 not in ix and 692 not in ix and 691 in ix
    tr,va,hold=splits(raw,cfg)
    assert tr[-1]+cfg.horizon<va[0]-cfg.lookback+1
    assert va[-1]+cfg.horizon<hold[0]-cfg.lookback+1


def test_train_predict_replay(tmp_path):
    raw,_=candles(1800)
    dev=tmp_path/'dev.csv'; raw.to_csv(dev,index=False)
    root=tmp_path/'model'
    cfg=TrendForecastSettings(epochs=1,channels=4,threads=1)
    result=train_forecast(dev,root,cfg)
    assert result['split_counts']['reserved_holdout_not_evaluated']>0
    assert not result['external_test_evaluated']
    r,_=candles(400,seed=4,start=int(raw.time.iloc[-1])+900)
    p=tmp_path/'test.csv'; r.to_csv(p,index=False)
    pred=predict_forecast(p,root)
    assert pred['predicted_trend'] in CLASSES
    assert sum(pred['p_'+k.lower()] for k in CLASSES)==pytest.approx(1.)
    out=tmp_path/'report.csv'
    report=replay_forecast(p,root,out)
    assert report['test']['network']['windows']==len(pd.read_csv(out))
    with pytest.raises(ValueError,match='chronologically after'): replay_forecast(dev,root,out)
