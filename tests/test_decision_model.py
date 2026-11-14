import numpy as np
import pandas as pd
import pytest
from finance_ai.decision_model import (DecisionSettings, Outcomes, Net, FEATURES, features,
    splits, actions, windows, train_decision, predict_decision, backtest_decision, WARMUP)


def candles(n,seed=3,start=1728000000):
    rng=np.random.default_rng(seed)
    # Align timestamps with M15 boundaries and construct matching M1 bars.
    start=start-start%900
    c=2700+np.cumsum(rng.normal(0,1.2,n*15))
    o=np.r_[2700,c[:-1]]
    m=pd.DataFrame({'time':start+60*np.arange(n*15),'open':o,'close':c,
                    'high':np.maximum(o,c)+.15,'low':np.minimum(o,c)-.15,'timeframe':1})
    groups=m.groupby(np.arange(len(m))//15)
    r=groups.agg({'time':'first','open':'first','high':'max','low':'min','close':'last'}).assign(volume=123,timeframe=15)
    return r,m


def test_features_no_future_leak():
    r,_=candles(230)
    f=features(r); g=features(r.iloc[:190])
    np.testing.assert_allclose(f[WARMUP:190],g[WARMUP:],rtol=1e-6,atol=1e-6)


def test_split_no_label_or_input_overlap():
    cfg=DecisionSettings(lookback=500)
    tr,va,ho=splits(10000,cfg)
    assert tr[-1]+cfg.horizon<va[0]-499
    assert va[-1]+cfg.horizon<ho[0]-499


def test_decision_wait_and_direction():
    a,_=actions(np.array([[.1,.3],[.8,.1],[.6,.8]]),DecisionSettings())
    np.testing.assert_array_equal(a,[-1,0,1])


def test_m1_order_and_signal_bar_excluded():
    t=1728000000
    raw=pd.DataFrame({'time':[t,t+900],'open':[100,100],'high':[110,103],'low':[90,97],'close':[100,100]})
    # First minute reaches TP, second reaches SL, all remaining minutes stay flat.
    m=pd.DataFrame({'time':t+900+60*np.arange(15),'open':100.,'high':100.,'low':100.,'close':100.})
    m.loc[0,'high']=103; m.loc[1,'low']=97
    e=Outcomes(raw,m,DecisionSettings())
    assert e.one(0,1)[0]=='TP'
    assert e.one(0,-1)[0]=='SL'
    m.loc[0,'low']=97
    assert Outcomes(raw,m,DecisionSettings()).one(0,1)[0]=='AMBIGUOUS'
    assert Outcomes(raw,m.iloc[1:],DecisionSettings()).one(0,1)[0]=='M1_MISSING'


def test_masked_training():
    rng=np.random.default_rng(6); x=rng.normal(size=(64,3)).astype('float32')
    y=np.column_stack([(x[:,0]>0).astype(float),np.full(len(x),np.nan)]).astype('float32')
    net=Net((3,12,2)); before=net.forward(x)[1][:,0]
    for _ in range(100): net.train(x,y,.01)
    after=net.forward(x)[1][:,0]
    assert np.mean((after-y[:,0])**2)<np.mean((before-y[:,0])**2)
    assert np.isfinite(net.forward(x)[1]).all()


def test_full_synthetic_train_predict_replay(tmp_path):
    r,m=candles(1400); dev=tmp_path/'dev.csv'; minute=tmp_path/'m1.csv'; root=tmp_path/'model'
    r.to_csv(dev,index=False); m.to_csv(minute,index=False)
    cfg=DecisionSettings(epochs=2,hidden=(8,),batch_size=128)
    report=train_decision(dev,minute,root,cfg)
    assert report['split_counts']['reserved_holdout_windows_not_evaluated']>0
    prediction=predict_decision(dev,root); assert prediction['decision'] in ('BUY','SELL','WAIT')
    test,mt=candles(230,seed=4,start=int(r.time.iloc[-1])+900)
    testpath=tmp_path/'test.csv'; mtpath=tmp_path/'testm1.csv'; out=tmp_path/'result.csv'
    test.to_csv(testpath,index=False); mt.to_csv(mtpath,index=False)
    summary=backtest_decision(testpath,mtpath,root,out)
    assert summary['overall']['windows']==len(test)-WARMUP-cfg.lookback+1
    assert len(pd.read_csv(out))==summary['overall']['windows']
    with pytest.raises(ValueError,match='chronologically after'): backtest_decision(dev,minute,root,out)


def test_window_500():
    r,_=candles(600); f=features(r); mu=np.zeros(len(FEATURES)); sd=np.ones(len(FEATURES))
    assert windows(f,[599],500,mu,sd).shape==(1,500*len(FEATURES))
