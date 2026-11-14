import numpy as np
import pandas as pd
import pytest
from finance_ai.value_model import (ValueSettings, ValueOutcomes, bounds, choices, decide,
    targets_from_moves, utility, values, train_value, predict_value, backtest_value)
from finance_ai.decision_model import WARMUP
from test_decision_model import candles


def test_utility_wait_no_daily_cap():
    cfg=ValueSettings(); cfg.validate()
    moves=np.array([[2.5,-2,3.5,-2,5,-2],[-2,-2,-2,-2,-2,-2]])
    p=targets_from_moves(moves,cfg)
    np.testing.assert_allclose(values(p,cfg),utility(moves,cfg),atol=1e-6)
    a,_=decide(p,cfg); assert a.tolist()==[4,-1]
    a,_=decide(np.repeat(p[:1],100,axis=0),cfg); assert (a==4).all()
    lo,hi=bounds(cfg); zero=np.broadcast_to(-lo/(hi-lo),(1,6))
    a,_=decide(zero-.001,cfg); assert a.tolist()==[-1]


def test_loss_and_cost():
    cfg=ValueSettings(cost=.2,loss_aversion=2)
    assert utility(np.array([-2.2,2.3]),cfg).tolist()==[-4.4,2.3]
    with pytest.raises(ValueError): ValueSettings(loss_aversion=.9).validate()


def test_future_labels_ambiguity_timeout_and_missing():
    t=1728000000
    raw=pd.DataFrame({'time':[t,t+900],'open':[100,100],'high':[110.,103.],'low':[90.,97.],'close':[100.,100.]})
    m=pd.DataFrame({'time':t+900+60*np.arange(15),'open':100.,'high':100.,'low':100.,'close':100.})
    m.loc[0,'high']=103; m.loc[1,'low']=97
    cfg=ValueSettings(horizon=1)
    e=ValueOutcomes(raw,m,cfg); assert e.one(0,0)[:2]==('TP',2.5)
    assert e.one(0,2)[:2]==('SL',-2.)
    m.loc[0,'low']=97
    assert ValueOutcomes(raw,m,cfg).one(0,0)[:2]==('AMBIGUOUS',-2.)
    assert np.isnan(ValueOutcomes(raw,m.iloc[1:],cfg).one(0,0)[1])
    raw.loc[1,['high','low','close']]=[101.5,99.5,101.]
    assert ValueOutcomes(raw,m,cfg).one(0,0)[:2]==('TIMEOUT',1.)
    assert ValueOutcomes(raw,m,cfg).one(0,1)[:2]==('TIMEOUT',-1.)
    assert np.isnan(ValueOutcomes(raw,m,cfg).one(1,0)[1])


def test_full_train_predict_backtest(tmp_path):
    r,m=candles(1400); dev=tmp_path/'dev.csv'; minute=tmp_path/'min.csv'; root=tmp_path/'model'
    r.to_csv(dev,index=False); m.to_csv(minute,index=False)
    cfg=ValueSettings(epochs=2,hidden=(8,),batch_size=128)
    report=train_value(dev,minute,root,cfg)
    assert report['architecture'][-1]==6 and report['split_counts']['reserved_holdout_not_evaluated']>0
    assert predict_value(dev,root)['decision'] in ('BUY','SELL','WAIT')
    with pytest.raises(FileExistsError): train_value(dev,minute,root,cfg)
    r,m=candles(230,seed=4,start=int(r.time.iloc[-1])+900); test=tmp_path/'test.csv'; tm=tmp_path/'tm.csv'; out=tmp_path/'out.csv'
    r.to_csv(test,index=False); m.to_csv(tm,index=False)
    result=backtest_value(test,tm,root,out)
    assert result['overall']['windows']==len(r)-WARMUP-cfg.lookback+1
    df=pd.read_csv(out); assert len(df)==result['overall']['windows']
    assert set(df.decision).issubset({'BUY','SELL','WAIT'})
    with pytest.raises(ValueError,match='chronologically after'): backtest_value(dev,minute,root,out)
