"""v2.0: forecast future M15 price trend, not entries or trading outcomes."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_model import load_candles, save_json
from .temporal_model import torch_cpu, state_arrays, restore

VERSION = '2.0.0'
CLASSES = ('DOWN', 'SIDEWAYS', 'UP')
FEATURES = ('open_relative','high_relative','low_relative','close_relative',
            'body','upper_wick','lower_wick','range','log_atr_relative',
            'relative_volume','log_gap')
WARMUP = 60


@dataclass(frozen=True)
class TrendForecastSettings:
    lookback: int = 100
    horizon: int = 8
    neutral_atr: float = 1.0
    epochs: int = 25
    batch_size: int = 128
    lr: float = .0003
    patience: int = 5
    seed: int = 42
    threads: int = 4
    channels: int = 16
    dropout: float = .2

    def validate(self):
        if not 32 <= self.lookback <= 500 or not 2 <= self.horizon <= 96:
            raise ValueError('lookback must be 32..500; horizon 2..96')
        if not np.isfinite(self.neutral_atr) or self.neutral_atr <= 0:
            raise ValueError('neutral-atr must be positive')
        if min(self.epochs,self.batch_size,self.patience) < 1 or not 1 <= self.threads <= 64:
            raise ValueError('Invalid training settings')
        if not 4 <= self.channels <= 128 or not 0 <= self.dropout < .5 or not np.isfinite(self.lr) or self.lr<=0:
            raise ValueError('Invalid model settings')


def inputs(raw):
    o,h,l,c=(raw[k].astype(float) for k in ('open','high','low','close'))
    prev=c.shift(1)
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean().clip(lower=1e-6)
    vol=np.log1p(raw.volume.astype(float))
    gap=np.log1p(raw.time.diff()/900)
    f=np.column_stack([(o-prev)/atr,(h-prev)/atr,(l-prev)/atr,(c-prev)/atr,
        (c-o)/atr,(h-np.maximum(o,c))/atr,(np.minimum(o,c)-l)/atr,
        (h-l)/atr,np.log(atr/c),vol-vol.rolling(30).mean(),gap]).astype(np.float32)
    if len(f)<=WARMUP or not np.isfinite(f[WARMUP:]).all():
        raise ValueError('Invalid features or insufficient history')
    return f,atr.to_numpy(float)


def trend_strength(raw,ends,atr,cfg,forward=True):
    # OLS slope fitted across H+1 closes, multiplied by H and normalized
    # by ATR at the decision close. Forward values are labels, never inputs.
    h=cfg.horizon
    offsets=np.arange(0,h+1) if forward else np.arange(-h,1)
    x=np.arange(h+1,dtype=float); x-=x.mean()
    w=x/(x@x)*h
    close=raw.close.to_numpy(float)
    ix=np.asarray(ends,dtype=np.int64)
    return ((close[ix[:,None]+offsets[None,:]]-close[ix,None])@w)/atr[ix]


def categorise(z,threshold):
    return np.where(z>threshold,2,np.where(z < -threshold,0,1)).astype(np.int64)


def candidates(raw,cfg,start=0,stop=None):
    # Full raw input window and future label in one block. A future horizon
    # interrupted by a market closure/data gap is excluded, not called sideways.
    stop=len(raw) if stop is None else stop
    first=max(start+cfg.lookback-1,WARMUP+cfg.lookback-1,cfg.horizon)
    ix=np.arange(first,max(first,stop-cfg.horizon),dtype=np.int64)
    t=raw.time.to_numpy(np.int64)
    actual=t[ix+cfg.horizon]-t[ix]
    return ix[actual==900*cfg.horizon]


def splits(raw,cfg):
    n=len(raw); a=int(.7*n); b=int(.85*n)
    groups=[candidates(raw,cfg,s,e) for s,e in ((0,a),(a,b),(b,n))]
    if min(map(len,groups))<30:
        raise ValueError('Insufficient data for chronological train/validation/holdout')
    return groups


def batch(f,ends,cfg,mu,sd):
    ix=np.asarray(ends)[:,None]+np.arange(1-cfg.lookback,1)[None,:]
    return np.ascontiguousarray((f[ix]-mu)/sd,dtype=np.float32)


def build_model(cfg,prior=None):
    torch=torch_cpu(cfg.threads)
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(cfg.seed)
    c=cfg.channels

    class Causal(nn.Module):
        def __init__(self,a,b,k):
            super().__init__()
            self.pad=k-1; self.conv=nn.Conv1d(a,b,k)
        def forward(self,x):
            return self.conv(F.pad(x,(self.pad,0)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(Causal(len(FEATURES),c,5),nn.GELU(),
                Causal(c,c,5),nn.GELU(),nn.AvgPool1d(2),
                Causal(c,c,5),nn.GELU())
            self.head=nn.Sequential(nn.LayerNorm(c*6),nn.Linear(c*6,24),
                nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(24,3))
            if prior is not None:
                with torch.no_grad():
                    self.head[-1].bias.copy_(torch.log(torch.as_tensor(prior,dtype=torch.float32).clamp_min(1e-6)))
        def forward(self,x):
            z=self.encoder(x.transpose(1,2))
            return self.head(torch.cat((z[:,:,-1],z.mean(2),
                         F.adaptive_avg_pool1d(z,4).flatten(1)),dim=1))
    return Model().cpu()


def probabilities(model,f,ends,cfg,mu,sd):
    torch=torch_cpu(cfg.threads); model.eval(); parts=[]
    with torch.inference_mode():
        for i in range(0,len(ends),cfg.batch_size):
            x=batch(f,ends[i:i+cfg.batch_size],cfg,mu,sd)
            parts.append(model(torch.from_numpy(x)).softmax(dim=1).numpy())
    if not parts: raise ValueError('No eligible forecast windows')
    return np.concatenate(parts)


def metrics(y,pred,p=None):
    cm=np.zeros((3,3),np.int64)
    np.add.at(cm,(y,pred),1)
    support=cm.sum(1); predicted=cm.sum(0); tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(3,float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(3,float),where=predicted>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(3,float),where=(support+predicted)>0)
    out={'windows':len(y),'accuracy':float((y==pred).mean()),
         'balanced_accuracy':float(recall[support>0].mean()),'macro_f1':float(f1.mean()),
         'confusion_matrix':cm.tolist(),
         'per_class':{k:{'support':int(support[i]),'predicted':int(predicted[i]),
                         'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])}
                      for i,k in enumerate(CLASSES)}}
    if p is not None:
        out['log_loss']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1)).mean())
        out['brier_score']=float(np.square(p-np.eye(3)[y]).sum(axis=1).mean())
    return out


def evaluation(raw,ends,atr,cfg,p,prior):
    y=categorise(trend_strength(raw,ends,atr,cfg),cfg.neutral_atr)
    prev=categorise(trend_strength(raw,ends,atr,cfg,False),cfg.neutral_atr)
    return {'network':metrics(y,p.argmax(1),p),
            'constant_training_prior':metrics(y,np.full(len(y),prior.argmax()),
                                               np.broadcast_to(prior,p.shape)),
            'past_trend_persistence':metrics(y,prev)}


def train_forecast(path,out,cfg=None):
    cfg=cfg or TrendForecastSettings(); cfg.validate(); root=Path(out)
    if root.exists(): raise FileExistsError('Use a new model directory')
    torch=torch_cpu(cfg.threads); raw=load_candles(path,15); f,atr=inputs(raw)
    tr,va,hold=splits(raw,cfg)
    yt=categorise(trend_strength(raw,tr,atr,cfg),cfg.neutral_atr)
    yv=categorise(trend_strength(raw,va,atr,cfg),cfg.neutral_atr)
    prior=(np.bincount(yt,minlength=3)+1)/(len(yt)+3)
    if min(np.bincount(yt,minlength=3))<20: raise ValueError('Too few examples of at least one training class')
    used=f[tr[0]-cfg.lookback+1:tr[-1]+1]
    mu=used.mean(axis=0); sd=np.maximum(used.std(axis=0),1e-6)
    model=build_model(cfg,prior)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    rng=np.random.default_rng(cfg.seed)
    best=np.inf; state=None; selected=0; misses=0; history=[]
    print(f'[v2.0] train={len(tr)} validation={len(va)} reserved={len(hold)} classes={np.bincount(yt,minlength=3).tolist()}',file=sys.stderr,flush=True)
    for epoch in range(1,cfg.epochs+1):
        model.train(); order=rng.permutation(len(tr))
        for j in range(0,len(order),cfg.batch_size):
            k=order[j:j+cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            z=model(torch.from_numpy(batch(f,tr[k],cfg,mu,sd)))
            loss=torch.nn.functional.cross_entropy(z,torch.from_numpy(yt[k]))
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        pv=probabilities(model,f,va,cfg,mu,sd)
        m=metrics(yv,pv.argmax(1),pv); loss=m['log_loss']
        history.append({'epoch':epoch,'log_loss':loss,'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
        print(f'[v2.0] epoch={epoch}/{cfg.epochs} val_loss={loss:.6f} accuracy={m["accuracy"]:.2%} macro_f1={m["macro_f1"]:.4f}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; state=state_arrays(model); selected=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if state is None: raise ValueError('No finite validation checkpoint')
    restore(model,state)
    pt=probabilities(model,f,tr,cfg,mu,sd); pv=probabilities(model,f,va,cfg,mu,sd)
    report_val=evaluation(raw,va,atr,cfg,pv,prior)
    meta={'model_version':VERSION,'settings':asdict(cfg),'features':list(FEATURES),
          'classes':list(CLASSES),'warmup':WARMUP,'training_prior':prior.tolist(),
          'development_range':[int(raw.time.iloc[0]),int(raw.time.iloc[-1])],
          'parameters':sum(p.numel() for p in model.parameters()),
          'label_definition':'OLS slope of current and next H closes times H, divided by current ATR14; +/- neutral_atr threshold; future gaps excluded'}
    report={**meta,'selected_epoch':selected,'epochs':history,
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
            'train':evaluation(raw,tr,atr,cfg,pt,prior),'validation':report_val,
            'beats_prior_log_loss':report_val['network']['log_loss']<report_val['constant_training_prior']['log_loss'],
            'beats_persistence_macro_f1':report_val['network']['macro_f1']>report_val['past_trend_persistence']['macro_f1'],
            'external_test_evaluated':False}
    root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'model.npz',**state,mean=mu,std=sd)
    save_json(root/'config.json',meta); save_json(root/'training.json',report)
    return report


def checkpoint(path,threads=None):
    root=Path(path); meta=json.loads((root/'config.json').read_text())
    if (meta['model_version']!=VERSION or meta['features']!=list(FEATURES)
            or meta['classes']!=list(CLASSES) or meta['warmup']!=WARMUP):
        raise ValueError('Trend forecast checkpoint schema mismatch')
    settings=meta['settings']
    cfg=TrendForecastSettings(**({**settings,'threads':threads} if threads is not None else settings))
    cfg.validate(); model=build_model(cfg)
    with np.load(root/'model.npz',allow_pickle=False) as data:
        mu=data['mean']; sd=data['std']
        restore(model,{k:data[k] for k in data.files if k not in ('mean','std')})
    if mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise ValueError('Invalid scaler')
    return model,cfg,mu,sd,meta


def forecast_row(raw,i,p,cfg):
    return {'candle_close_utc':pd.Timestamp(int(raw.time.iloc[i])+900,unit='s',tz='UTC').isoformat(),
            'horizon_candles':cfg.horizon,'predicted_trend':CLASSES[int(p.argmax())],
            **{f'p_{k.lower()}':float(p[j]) for j,k in enumerate(CLASSES)}}


def predict_forecast(path,model_path,threads=None):
    model,cfg,mu,sd,_=checkpoint(model_path,threads)
    raw=load_candles(path,15); f,_=inputs(raw)
    if len(raw)<WARMUP+cfg.lookback: raise ValueError('Insufficient completed M15 history')
    i=len(raw)-1; p=probabilities(model,f,np.array([i]),cfg,mu,sd)[0]
    return {'version':VERSION,**forecast_row(raw,i,p,cfg)}


def test_forecast(path,model_path,out,threads=None):
    model,cfg,mu,sd,meta=checkpoint(model_path,threads)
    raw=load_candles(path,15)
    if int(raw.time.iloc[0])<=meta['development_range'][1]:
        raise ValueError('External test must be chronologically after development')
    f,atr=inputs(raw); ix=candidates(raw,cfg)
    p=probabilities(model,f,ix,cfg,mu,sd)
    z=trend_strength(raw,ix,atr,cfg); y=categorise(z,cfg.neutral_atr)
    rows=[{**forecast_row(raw,int(i),p[j],cfg),'actual_trend':CLASSES[int(y[j])],
           'actual_strength_atr':float(z[j])} for j,i in enumerate(ix)]
    dst=Path(out)
    if dst.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    dst.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dst,index=False)
    result={'version':VERSION,'settings':asdict(cfg),'test':evaluation(raw,ix,atr,cfg,p,np.asarray(meta['training_prior'])),
            'note':'Forecast classification only; not a trading strategy or calibrated probability guarantee.'}
    save_json(dst.with_suffix('.summary.json'),result); return result


def register(actions):
    tr=actions.add_parser('train-forecast',help='v2.0: forecast future three-class M15 trend')
    tr.add_argument('--csv',default='data/multi_tf/development_m15.csv')
    tr.add_argument('--out',default='models/finance-v2.0')
    cfg=TrendForecastSettings()
    for name,kind in [('lookback',int),('horizon',int),('neutral-atr',float),('epochs',int),
                      ('batch-size',int),('lr',float),('patience',int),('seed',int),
                      ('threads',int),('channels',int),('dropout',float)]:
        tr.add_argument('--'+name,type=kind,default=getattr(cfg,name.replace('-','_')))
    for cmd in ('predict-forecast','test-forecast'):
        a=actions.add_parser(cmd)
        a.add_argument('--csv',default='data/multi_tf/final_test_m15.csv')
        a.add_argument('--model',default='models/finance-v2.0')
        a.add_argument('--threads',type=int)
        if cmd=='test-forecast': a.add_argument('--out',default='reports/v2.0-forecast.csv')


def dispatch(args):
    if args.action=='train-forecast':
        cfg=TrendForecastSettings(**{k:v for k,v in vars(args).items() if k not in ('action','csv','out')})
        return train_forecast(args.csv,args.out,cfg)
    if args.action=='predict-forecast': return predict_forecast(args.csv,args.model,args.threads)
    if args.action=='test-forecast': return test_forecast(args.csv,args.model,args.out,args.threads)
    raise ValueError('Unknown trend forecast command')
