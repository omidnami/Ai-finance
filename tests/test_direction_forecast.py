import numpy as np
import pandas as pd
import pytest
pytest.importorskip('torch')
from finance_ai.forecast_model import TrendForecastSettings, inputs
from finance_ai.direction_forecast import (labels, common_splits, metrics, train_direction,
    predict_direction, replay_direction, HORIZONS)
from test_decision_model import candles


def test_sign_and_metrics():
    np.testing.assert_array_equal(labels([-2,-.0001,0,.0001,2]),[0,0,-1,1,1])
    r=metrics([0,0,1,1],[0,1,1,1],np.array([[.7,.3],[.3,.7],[.4,.6],[.2,.8]]))
    assert r['accuracy']==.75 and r['balanced_accuracy']==.75
    assert r['confusion_matrix']==[[1,1],[0,2]]
    assert metrics([],[]) is None


def test_common_windows():
    raw,_=candles(1800); cfg=TrendForecastSettings()
    raw.loc[700:,'time']+=900
    _,atr=inputs(raw)
    tr,va,hold=common_splits(raw,atr,cfg,HORIZONS)
    assert not any(i in tr for i in range(668,700))
    assert tr[-1]+max(HORIZONS)<va[0]-cfg.lookback+1
    assert va[-1]+max(HORIZONS)<hold[0]-cfg.lookback+1


def test_train_predict_replay(tmp_path):
    raw,_=candles(1800); dev=tmp_path/'dev.csv'; raw.to_csv(dev,index=False)
    cfg=TrendForecastSettings(epochs=1,channels=4,threads=1)
    root=tmp_path/'models'
    r=train_direction(dev,root,cfg,HORIZONS)
    assert len(r['ranking_by_log_loss_improvement'])==4
    assert len({v['network']['windows'] for v in r['ranking_by_log_loss_improvement']})==1
    assert not r['external_test_evaluated']
    r,_=candles(500,seed=4,start=int(raw.time.iloc[-1])+900)
    test=tmp_path/'test.csv'; r.to_csv(test,index=False)
    model=root/'h4'
    p=predict_direction(test,model)
    assert p['predicted_trend'] in ('UP','DOWN')
    assert p['p_up']+p['p_down']==pytest.approx(1.)
    out=tmp_path/'test-report.csv'
    report=replay_direction(test,model,out)
    assert report['test']['network']['windows']==len(pd.read_csv(out))
    with pytest.raises(ValueError,match='chronologically after'): replay_direction(dev,model,out)
    with pytest.raises(FileExistsError): train_direction(dev,root,cfg,HORIZONS)
