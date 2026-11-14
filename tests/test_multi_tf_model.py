import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
from finance_ai.multi_tf_model import (Dataset, FRAMES, MultiSettings, build_model,
    train_multi, predict_multi, backtest_multi, excursion_labels)
from finance_ai.decision_model import FEATURES
from test_decision_model import candles


def write_frames(path,prefix,n,start=1728000000,seed=3):
    _,m=candles(n,seed=seed,start=start-start%3600)
    m['volume']=1
    for k,sec in FRAMES.items():
        r=m.groupby(m.time//sec).agg({'time':'first','open':'first','high':'max',
                                    'low':'min','close':'last','volume':'sum'})
        r['timeframe']=sec//60
        r.to_csv(path/f'{prefix}_{k}.csv',index=False)


def test_alignment_and_no_future(tmp_path):
    write_frames(tmp_path,'development',800)
    d=Dataset(tmp_path,'development',100)
    i=650
    assert d.valid[i]
    now=d.closes['m15'][i]
    for k,sec in FRAMES.items():
        j=d.indices[k][i]
        assert d.closes[k][j]<=now<d.closes[k][j]+sec
    scale=d.scaler(np.array([i]))
    before=d.batch(np.array([i]),scale)
    for k in FRAMES:
        p=tmp_path/f'development_{k}.csv'
        r=pd.read_csv(p)
        future=r.time+FRAMES[k]>now
        r.loc[future,['open','high','low','close']]+=100
        r.to_csv(p,index=False)
    changed=Dataset(tmp_path,'development',100)
    after=changed.batch(np.array([i]),scale)
    for k in FRAMES:
        torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        assert before[k].shape==(1,100,len(FEATURES))


def test_gradient_all_branches():
    cfg=MultiSettings(channels=4,threads=1)
    model=build_model(cfg)
    x={k:torch.randn(2,100,len(FEATURES)) for k in FRAMES}
    z,e=model(x)
    assert z.shape==(2,6) and e.shape==(2,2)
    (z.square().mean()+e.square().mean()).backward()
    for branch in model.branches.values():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())


def test_end_to_end(tmp_path):
    write_frames(tmp_path,'development',4000)
    cfg=MultiSettings(channels=4,threads=1,epochs=1,batch_size=128)
    d=Dataset(tmp_path,'development',100)
    tr,va,hold=d.block_splits(cfg.horizon)
    for ends,boundary in ((va,int(len(d.signal)*.7)),(hold,int(len(d.signal)*.85))):
        for k in FRAMES:
            assert d.opens[k][d.indices[k][ends[0]]-99]>=d.opens['m15'][boundary]
    root=tmp_path/'model'
    report=train_multi(tmp_path,root,cfg)
    assert report['split_counts']['reserved_holdout_not_evaluated']>0
    start=int(d.closes['m15'][-1])
    write_frames(tmp_path,'final_test',800,start=start,seed=9)
    pred=predict_multi(tmp_path,root)
    assert pred['decision'] in ('BUY','SELL','WAIT')
    out=tmp_path/'replay.csv'
    replay=backtest_multi(tmp_path,root,out)
    assert replay['overall']['windows']==len(pd.read_csv(out))
    with pytest.raises(FileExistsError): train_multi(tmp_path,root,cfg)
    write_frames(tmp_path,'final_test',800)
    with pytest.raises(ValueError,match='chronologically after'): backtest_multi(tmp_path,root,out)
