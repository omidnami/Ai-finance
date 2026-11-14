import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
from finance_ai.decision_model import FEATURES, WARMUP, features
from finance_ai.temporal_model import (TemporalSettings, ARCHITECTURES, build_model,
    sequence_batch, masked_loss, state_arrays, restore, train_temporal, predict_temporal,
    backtest_temporal, checkpoint)
from test_decision_model import candles


@pytest.mark.parametrize('arch',ARCHITECTURES)
def test_sequence_model_gradients_and_checkpoint(arch):
    cfg=TemporalSettings(channels=4,threads=1,lookback=100)
    prior=np.full(6,.4,np.float32); model=build_model(arch,cfg,prior)
    x=torch.randn(3,100,len(FEATURES)); y=torch.rand(3,6); y[0,0]=float('nan')
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        opt.zero_grad(); z=model(x); assert z.shape==(3,6)
        loss=masked_loss(z,y); assert torch.isfinite(loss); loss.backward(); opt.step()
    model.eval(); before=model(x).detach(); other=build_model(arch,cfg); restore(other,state_arrays(model)); other.eval()
    torch.testing.assert_close(before,other(x)); assert masked_loss(z,torch.full_like(y,float('nan'))) is None


def test_causal_input_sequence():
    raw,_=candles(240); f=features(raw); i=179; cfg=TemporalSettings(threads=1)
    x=sequence_batch(f,[i],cfg,np.zeros(len(FEATURES)),np.ones(len(FEATURES)))
    assert x.shape==(1,100,len(FEATURES)); np.testing.assert_array_equal(x[0],f[i-99:i+1])
    raw.loc[i+1:,'close']+=100
    np.testing.assert_array_equal(features(raw)[:i+1],f[:i+1])


def test_end_to_end_all_encoders(tmp_path):
    raw,m=candles(1400); dev=tmp_path/'dev.csv'; minute=tmp_path/'min.csv'; root=tmp_path/'models'
    raw.to_csv(dev,index=False); m.to_csv(minute,index=False)
    cfg=TemporalSettings(epochs=1,channels=4,threads=1,batch_size=128)
    report=train_temporal(dev,minute,root,cfg)
    assert {r['encoder'] for r in report['ranking_by_validation_loss']}==set(ARCHITECTURES)
    assert not report['holdout_evaluated'] and not report['external_test_evaluated']
    r,m=candles(230,seed=4,start=int(raw.time.iloc[-1])+900); test=tmp_path/'test.csv'; tm=tmp_path/'tm.csv'
    r.to_csv(test,index=False); m.to_csv(tm,index=False)
    for arch in ARCHITECTURES:
        path=root/arch; out=tmp_path/(arch+'.csv')
        pred=predict_temporal(test,path); assert pred['decision'] in ('BUY','SELL','WAIT')
        result=backtest_temporal(test,tm,path,out); df=pd.read_csv(out)
        assert len(df)==len(r)-WARMUP-cfg.lookback+1==result['overall']['windows']
        assert result['encoder']==arch
        assert df.iloc[-1].decision==pred['decision']
        with pytest.raises(ValueError,match='chronologically after'): backtest_temporal(dev,minute,path,out)
    with pytest.raises(FileExistsError): train_temporal(dev,minute,root,cfg)
