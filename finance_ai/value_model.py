"""v1.1 — learn risk-adjusted entry value directly from subsequent price outcomes.

Six independent value heads (BUY/SELL at each configured TP); WAIT has zero
utility. No technical-indicator gates, trade quotas, or old model signals.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_model import (DecisionSettings, FEATURES, WARMUP, Net, Outcomes,
    counts, features, load_candles, probabilities, save_json, splits, windows)

VERSION = '1.1.0'


@dataclass(frozen=True)
class ValueSettings(DecisionSettings):
    targets: tuple[float, ...] = (2.5, 3.5, 5.0)
    loss_aversion: float = 1.5
    hidden: tuple[int, ...] = (64, 32)
    lr: float = .0003
    input_dropout: float = .10

    def validate(self):
        super().validate()
        if (not self.targets or len(self.targets)>10 or
                any(not np.isfinite(t) or t <= self.cost or t < 2 for t in self.targets) or
                tuple(sorted(set(self.targets))) != tuple(self.targets)):
            raise ValueError('Targets must be unique, increasing, >=2 and greater than cost')
        if not np.isfinite(self.loss_aversion) or self.loss_aversion < 1:
            raise ValueError('loss-aversion must be finite and >=1')
        if not np.isfinite(self.input_dropout) or not 0 <= self.input_dropout < .5:
            raise ValueError('input-dropout must be in [0,0.5)')


def choices(cfg):
    return [(side,float(tp)) for tp in cfg.targets for side in ('BUY','SELL')]


def utility(net,cfg):
    return np.where(net < 0, net*cfg.loss_aversion, net)


def bounds(cfg):
    low = -cfg.loss_aversion*(cfg.sl+cfg.cost)
    high = np.array([tp-cfg.cost for _,tp in choices(cfg)],dtype=np.float32)
    return low,high


def targets_from_moves(moves,cfg):
    lo,hi=bounds(cfg)
    return np.clip((utility(moves,cfg)-lo)/(hi-lo),0,1).astype(np.float32)


def values(p,cfg):
    lo,hi=bounds(cfg)
    return lo+np.asarray(p)*(hi-lo)


def decide(p,cfg):
    v=values(p,cfg)
    a=v.argmax(axis=1)
    return np.where(v[np.arange(len(v)),a]>0,a,-1),v


class ValueOutcomes:
    def __init__(self,raw,m1,cfg):
        self.raw=raw; self.cfg=cfg; self.engines=[]
        for tp in cfg.targets:
            e=Outcomes(raw,m1,replace(cfg,tp=float(tp)))
            if self.engines:
                # Each target uses the same minute validation/cache.
                e.cache=self.engines[0].cache
            self.engines.append(e)

    def one(self,i,k):
        side,tp=choices(self.cfg)[k]; d=1 if side=='BUY' else -1
        status,j,minute=self.engines[k//2].one(i,d)
        if status=='TP': move=tp
        elif status in ('SL','AMBIGUOUS'): move=-self.cfg.sl
        elif status=='NO_HIT':
            j=i+self.cfg.horizon
            # Full horizon elapsed: hypothetical exit at its completed close.
            move=d*(self.engines[0].p[j,3]-self.engines[0].p[i,3]); status='TIMEOUT'
        else: return status,np.nan,j,minute
        return status,float(move-self.cfg.cost),j,minute

    def labels(self,ends):
        k=len(choices(self.cfg)); moves=np.full((len(ends),k),np.nan,np.float32)
        st=np.empty(moves.shape,object)
        for n,i in enumerate(ends):
            for a in range(k):
                st[n,a],moves[n,a],_,_=self.one(int(i),a)
        return moves,st


def stats(a,moves,st,cfg):
    trade=np.flatnonzero(a>=0); m=moves[trade,a[trade]]; s=st[trade,a[trade]]
    known=np.isfinite(m); realized=m[known].astype(float); n=len(realized)
    return {'windows':len(a),'signals':len(trade),'wait':int((a<0).sum()),
            'signal_rate':len(trade)/len(a) if len(a) else None,
            'BUY':int(((a>=0)&(a%2==0)).sum()),'SELL':int(((a>=0)&(a%2==1)).sum()),
            'by_target':{str(t):int(((a==2*k)|(a==2*k+1)).sum()) for k,t in enumerate(cfg.targets)},
            'outcomes':counts(s),'known':n,'unknown':int((~known).sum()),
            'positive_net_rate':float((realized>0).mean()) if n else None,
            'net_usd_per_oz_sum':float(realized.sum()),
            'mean_net_per_resolved_signal':float(realized.mean()) if n else None,
            'risk_adjusted_utility_sum':float(utility(realized,cfg).sum()),
            'utility_per_candidate':float(utility(realized,cfg).sum()/len(a)) if len(a) else None,
            'independent_overlapping_paper_signals':True}


def evaluate(p,moves,st,cfg):
    y=targets_from_moves(moves,cfg); mask=np.isfinite(y)
    q=np.clip(p[mask].astype(float),1e-7,1-1e-7); t=y[mask]
    a,v=decide(p,cfg)
    if not mask.any(): raise ValueError('No known outcomes in evaluation')
    return {'soft_label_log_loss':float(np.mean(-t*np.log(q)-(1-t)*np.log1p(-q))),
            'utility_mse':float(np.mean((v[mask]-utility(moves,cfg)[mask])**2)),
            'known_labels':int(mask.sum()),'possible_labels':int(mask.size),
            'strategy':stats(a,moves,st,cfg),
            'per_action':{f'{side}_{tp:g}':{'outcomes':counts(st[:,k]),
                 'mean_predicted_utility':float(v[:,k].mean()),
                 'mean_observed_utility':float(np.nanmean(utility(moves[:,k],cfg))) if mask[:,k].any() else None}
                 for k,(side,tp) in enumerate(choices(cfg))}}


def baselines(m,st,cfg):
    n=len(m)
    return {'WAIT':stats(np.full(n,-1),m,st,cfg),**{
        f'{s}_{t:g}':stats(np.full(n,k),m,st,cfg) for k,(s,t) in enumerate(choices(cfg))}}


def train_value(csv,m1_csv,out,cfg=None):
    cfg=cfg or ValueSettings(); cfg.validate(); root=Path(out)
    if any((root/name).exists() for name in ('model.npz','config.json','training.json')):
        raise FileExistsError('Use a new output directory to preserve existing checkpoints')
    raw=load_candles(csv,15); m1=load_candles(m1_csv,1); f=features(raw)
    tr,va,hold=splits(len(raw),cfg)
    if int(m1.time.iloc[0])>int(raw.time.iloc[tr[0]+1]) or int(m1.time.iloc[-1])<int(raw.time.iloc[va[-1]+cfg.horizon])+840:
        raise ValueError('M1 must cover the training AND validation future-label period')
    print('[v1.1] Preparing multi-target price outcome labels...',file=sys.stderr,flush=True)
    engine=ValueOutcomes(raw,m1,cfg); mt,st=engine.labels(tr); mv,sv=engine.labels(va)
    yt=targets_from_moves(mt,cfg); yv=targets_from_moves(mv,cfg)
    if min(np.isfinite(yt).sum(axis=0))<20 or min(np.isfinite(yv).sum(axis=0))<20:
        raise ValueError('Too few known outcomes for at least one action')
    used=f[tr[0]-cfg.lookback+1:tr[-1]+1]; mu=used.mean(axis=0); sd=np.maximum(used.std(axis=0),1e-6)
    size=(cfg.lookback*len(FEATURES),*cfg.hidden,len(choices(cfg)))
    net=Net(size,cfg.seed); rng=np.random.default_rng(cfg.seed)
    # Start with training-only unconditional mean utility, rather than optimistic 0.5 outputs.
    prior=np.clip(np.nanmean(yt,axis=0),1e-5,1-1e-5)
    net.w[-1][:]=0; net.b[-1][:]=np.log(prior/(1-prior))
    best=np.inf; state=None; epoch_best=0; misses=0; history=[]
    for epoch in range(1,cfg.epochs+1):
        order=rng.permutation(len(tr))
        for j in range(0,len(order),cfg.batch_size):
            k=order[j:j+cfg.batch_size]; x=windows(f,tr[k],cfg.lookback,mu,sd)
            if cfg.input_dropout:
                x*=((rng.random(x.shape)>=cfg.input_dropout)/(1-cfg.input_dropout)).astype(np.float32)
            # BCE with bounded soft targets is minimized by their conditional mean.
            net.train(x,yt[k],cfg.lr)
        pv=probabilities(net,f,va,cfg,mu,sd); report=evaluate(pv,mv,sv,cfg); loss=report['soft_label_log_loss']
        history.append({'epoch':epoch,'validation_loss':loss,'strategy':report['strategy']})
        print(f'[v1.1] epoch={epoch}/{cfg.epochs} val_loss={loss:.6f} signals={report["strategy"]["signals"]} utility={report["strategy"]["risk_adjusted_utility_sum"]:.2f}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; state=net.state(); epoch_best=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if state is None: raise ValueError('No finite validation loss')
    net.restore(state)
    pv=probabilities(net,f,va,cfg,mu,sd); pt=probabilities(net,f,tr,cfg,mu,sd)
    meta={'model_version':VERSION,'architecture':list(size),'settings':asdict(cfg),'features':list(FEATURES),
          'warmup':WARMUP,'actions':choices(cfg),'development_time_range':[int(raw.time.iloc[0]),int(raw.time.iloc[-1])]}
    report={**meta,'selected_epoch':epoch_best,'epochs':history,'train':evaluate(pt,mt,st,cfg),
            'validation':evaluate(pv,mv,sv,cfg),
            'validation_constant_training_prior':evaluate(np.broadcast_to(prior,pv.shape),mv,sv,cfg),
            'validation_baselines':baselines(mv,sv,cfg),
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
            'assumptions':'Risk-adjusted expected utility, not a calibrated win probability. WAIT=0; no daily quota. Missing data masked; ambiguous minute conservative SL; timeout exit at horizon close. Hypothetical overlapping close-fill signals, fixed cost only.'}
    root.mkdir(parents=True,exist_ok=True); np.savez_compressed(root/'model.npz',**state,mean=mu,std=sd)
    save_json(root/'config.json',meta); save_json(root/'training.json',report)
    return report


def checkpoint(path):
    root=Path(path); meta=json.loads((root/'config.json').read_text())
    if meta['model_version']!=VERSION or meta['features']!=list(FEATURES) or meta['warmup']!=WARMUP:
        raise ValueError('v1.1 checkpoint schema mismatch')
    cfg=ValueSettings(**meta['settings']); cfg.validate()
    size=(cfg.lookback*len(FEATURES),*cfg.hidden,len(choices(cfg)))
    if tuple(meta['architecture'])!=size or meta['actions']!=[list(x) for x in choices(cfg)]: raise ValueError('Invalid architecture/action schema')
    net=Net(size,cfg.seed)
    with np.load(root/'model.npz',allow_pickle=False) as data:
        mu=data['mean']; sd=data['std']; net.restore({k:data[k] for k in data.files if k not in ('mean','std')})
    if mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any(): raise ValueError('Invalid scaler')
    return net,cfg,mu,sd,meta


def stamp(t): return pd.Timestamp(int(t),unit='s',tz='UTC').isoformat()


def prediction(raw,i,a,v,cfg):
    side,tp=choices(cfg)[a] if a>=0 else ('WAIT',None); e=float(raw.close.iloc[i]); d=1 if side=='BUY' else -1
    return {'signal_candle_open_utc':stamp(raw.time.iloc[i]),'signal_close_utc':stamp(raw.time.iloc[i]+900),
            'decision':side,'tp_distance':tp,'entry_close':e if a>=0 else None,
            'tp_price':e+d*tp if a>=0 else None,'sl_price':e-d*cfg.sl if a>=0 else None,
            **{f'{s.lower()}_{t:g}_utility':float(v[k]) for k,(s,t) in enumerate(choices(cfg))}}


def predict_value(csv,model):
    net,cfg,mu,sd,_=checkpoint(model); raw=load_candles(csv,15); f=features(raw)
    if len(raw)<WARMUP+cfg.lookback: raise ValueError('Insufficient completed M15 candles')
    p=probabilities(net,f,np.array([len(raw)-1]),cfg,mu,sd); a,v=decide(p,cfg)
    return {'version':VERSION,**prediction(raw,len(raw)-1,int(a[0]),v[0],cfg)}


def backtest_value(csv,m1_csv,model,out):
    net,cfg,mu,sd,meta=checkpoint(model); raw=load_candles(csv,15); f=features(raw)
    if int(raw.time.iloc[0])<=meta['development_time_range'][1]: raise ValueError('External replay must be chronologically after development')
    ends=np.arange(WARMUP+cfg.lookback-1,len(raw),dtype=np.int64)
    if not len(ends): raise ValueError('Not enough history')
    p=probabilities(net,f,ends,cfg,mu,sd); a,v=decide(p,cfg)
    # All decisions already fixed before future-price evaluation.
    engine=ValueOutcomes(raw,load_candles(m1_csv,1),cfg); moves,st=engine.labels(ends)
    rows=[]
    for k,i in enumerate(ends):
        chosen=int(a[k]); status,m,j,minute=engine.one(int(i),chosen) if chosen>=0 else ('WAIT',np.nan,None,None)
        rows.append({**prediction(raw,i,chosen,v[k],cfg),'result':status,
                     'exit_m15_open_utc':stamp(raw.time.iloc[j]) if j is not None else None,
                     'exit_m1_open_utc':stamp(minute) if minute is not None else None,
                     'net_usd_per_oz':float(m) if np.isfinite(m) else None,
                     'risk_adjusted_utility':float(utility(m,cfg)) if np.isfinite(m) else None})
    path=Path(out)
    if path.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
    report={'version':VERSION,'settings':asdict(cfg),'overall':stats(a,moves,st,cfg),
            'baselines':baselines(moves,st,cfg),'evaluation':evaluate(p,moves,st,cfg),
            'assumptions':'Independent overlapping hypothetical close fills; conservative ambiguous SL; full-horizon timeout close; missing/pending excluded; fixed costs only; no portfolio P&L.'}
    save_json(path.with_suffix('.summary.json'),report); return report


def register(actions):
    train=actions.add_parser('train-value',help='v1.1: learn risk-adjusted expected value of BUY/SELL at multiple targets')
    train.add_argument('--csv','--data','--excel',dest='data',required=True); train.add_argument('--m1-csv',required=True)
    train.add_argument('--out',default='models/finance-v1.1')
    for name,kind,default in [('lookback',int,100),('sl',float,2.),('horizon',int,20),('cost',float,0.),('loss-aversion',float,1.5),('tolerance',float,.02),('epochs',int,30),('batch-size',int,128),('lr',float,.0003),('patience',int,5),('seed',int,42),('input-dropout',float,.10)]:
        train.add_argument('--'+name,type=kind,default=default)
    train.add_argument('--targets',type=float,nargs='+',default=[2.5,3.5,5.]); train.add_argument('--hidden',type=int,nargs='+',default=[64,32])
    for cmd in ('predict-value','backtest-value'):
        sub=actions.add_parser(cmd); sub.add_argument('--csv','--data','--excel',dest='data',required=True); sub.add_argument('--model',default='models/finance-v1.1')
        if cmd=='backtest-value': sub.add_argument('--m1-csv',required=True); sub.add_argument('--out',default='reports/v1.1-decisions.csv')


def dispatch(args):
    if args.action=='train-value':
        cfg=ValueSettings(**{k:v for k,v in vars(args).items() if k not in ('action','data','m1_csv','out')})
        return train_value(args.data,args.m1_csv,args.out,cfg)
    if args.action=='predict-value': return predict_value(args.data,args.model)
    if args.action=='backtest-value': return backtest_value(args.data,args.m1_csv,args.model,args.out)
    raise ValueError('Unknown value command')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); register(parser.add_subparsers(dest='action',required=True))
    print(json.dumps(dispatch(parser.parse_args()),indent=2,allow_nan=False))
