"""v2.1: binary future trend forecasts at multiple horizons, no trading."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_model import load_candles, save_json
from .temporal_model import torch_cpu, state_arrays, restore
from .forecast_model import (TrendForecastSettings, FEATURES, WARMUP, inputs,
    trend_strength, candidates, splits, batch, build_model as build_three,
    probabilities)

VERSION = '2.1.0'
CLASSES = ('DOWN', 'UP')
HORIZONS = (4, 8, 16, 32)
EPS = 1e-8


def labels(strength):
    z = np.asarray(strength)
    return np.where(np.abs(z)<=EPS,-1,(z>0).astype(np.int64)).astype(np.int64)


def common_splits(raw,atr,cfg,horizons):
    # Use the longest future horizon to purge split boundaries for all models.
    # Drop mathematical ties at any requested horizon to keep identical rows.
    groups = splits(raw,replace(cfg,horizon=max(horizons)))
    result=[]
    for ix in groups:
        valid=np.ones(len(ix),dtype=bool)
        for h in horizons:
            valid &= labels(trend_strength(raw,ix,atr,replace(cfg,horizon=h)))>=0
        result.append(ix[valid])
    if min(map(len,result))<30:
        raise ValueError('Too few shared non-tie windows')
    return result


def build_model(cfg,prior=None):
    torch=torch_cpu(cfg.threads)
    model=build_three(cfg)
    # Preserve the v2.0 encoder and dense head except for the two-class output.
    model.head[-1]=torch.nn.Linear(24,2)
    if prior is not None:
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.log(torch.as_tensor(prior,dtype=torch.float32).clamp_min(1e-6)))
    return model


def metrics(y,pred,p=None):
    y=np.asarray(y,dtype=np.int64); pred=np.asarray(pred,dtype=np.int64)
    if not len(y): return None
    cm=np.zeros((2,2),np.int64); np.add.at(cm,(y,pred),1)
    support=cm.sum(1); predicted=cm.sum(0); tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(2,float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(2,float),where=predicted>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(2,float),where=(support+predicted)>0)
    out={'windows':len(y),'accuracy':float((y==pred).mean()),
        'balanced_accuracy':float(recall.mean()) if (support>0).all() else None,
        'macro_f1':float(f1.mean()),'confusion_matrix':cm.tolist(),
        'per_class':{k:{'support':int(support[i]),'predicted':int(predicted[i]),
            'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])}
            for i,k in enumerate(CLASSES)}}
    if p is not None:
        out['log_loss']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1)).mean())
        out['brier_score_up']=float(np.square(p[:,1]-y).mean())
    return out


def evaluation(raw,ends,atr,cfg,p,prior):
    z=trend_strength(raw,ends,atr,cfg)
    y=labels(z)
    if (y<0).any(): raise ValueError('Ties must be excluded before evaluation')
    prev=labels(trend_strength(raw,ends,atr,cfg,False))
    prev=np.where(prev<0,int(prior.argmax()),prev)
    up=p.argmax(1)
    strong=np.abs(z)>=cfg.neutral_atr
    return {'network':metrics(y,up,p),
        'constant_training_prior':metrics(y,np.full(len(y),prior.argmax()),np.broadcast_to(prior,p.shape)),
        'past_trend_persistence':metrics(y,prev),
        'strong_realized_trend_subgroup':metrics(y[strong],up[strong],p[strong]),
        'weak_realized_trend_subgroup':metrics(y[~strong],up[~strong],p[~strong]),
        'subgroup_note':'Realized strength is known only after the forecast; never an inference filter.',
        'confidence_audit':{str(t):metrics(y[p.max(1)>=t],up[p.max(1)>=t],p[p.max(1)>=t])
                            for t in (.55,.60,.65,.70)}}


def fit_one(raw,f,atr,tr,va,hold,cfg,root,horizons):
    torch=torch_cpu(cfg.threads)
    yt=labels(trend_strength(raw,tr,atr,cfg))
    yv=labels(trend_strength(raw,va,atr,cfg))
    counts=np.bincount(yt,minlength=2)
    if min(counts)<20 or min(np.bincount(yv,minlength=2))<20:
        raise ValueError('Too few examples of either direction')
    prior=(counts+1)/(len(yt)+2)
    used=f[tr[0]-cfg.lookback+1:tr[-1]+1]
    mu=used.mean(axis=0); sd=np.maximum(used.std(axis=0),1e-6)
    model=build_model(cfg,prior)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    rng=np.random.default_rng(cfg.seed); best=np.inf; state=None; selected=0; misses=0; history=[]
    print(f'[v2.1 H{cfg.horizon}] train={len(tr)} validation={len(va)} reserved={len(hold)} classes={counts.tolist()}',file=sys.stderr,flush=True)
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
        history.append({'epoch':epoch,'log_loss':loss,'accuracy':m['accuracy'],
                        'balanced_accuracy':m['balanced_accuracy'],'macro_f1':m['macro_f1']})
        print(f'[v2.1 H{cfg.horizon}] epoch={epoch}/{cfg.epochs} val_loss={loss:.6f} accuracy={m["accuracy"]:.2%} balanced={m["balanced_accuracy"]:.2%}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; state=state_arrays(model); selected=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if state is None: raise ValueError('No finite validation checkpoint')
    restore(model,state)
    pt=probabilities(model,f,tr,cfg,mu,sd); pv=probabilities(model,f,va,cfg,mu,sd)
    validation=evaluation(raw,va,atr,cfg,pv,prior)
    meta={'model_version':VERSION,'settings':asdict(cfg),'features':list(FEATURES),
          'classes':list(CLASSES),'warmup':WARMUP,'training_prior':prior.tolist(),
          'comparison_horizons':list(horizons),
          'development_range':[int(raw.time.iloc[0]),int(raw.time.iloc[-1])],
          'parameters':sum(p.numel() for p in model.parameters()),
          'label_definition':'Sign of least squares slope over current and next H closes, normalized by current ATR. Near-exact zero slope excluded; no sideways class.'}
    report={**meta,'selected_epoch':selected,'epochs':history,
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
            'train':evaluation(raw,tr,atr,cfg,pt,prior),'validation':validation,
            'external_test_evaluated':False}
    root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'model.npz',**state,mean=mu,std=sd)
    save_json(root/'config.json',meta); save_json(root/'training.json',report)
    return report


def train_direction(path,out,cfg=None,horizons=HORIZONS):
    cfg=cfg or TrendForecastSettings(); cfg.validate(); horizons=tuple(horizons)
    if not horizons or len(set(horizons))!=len(horizons) or any(not 2<=h<=96 for h in horizons):
        raise ValueError('Select unique horizons from 2..96')
    root=Path(out)
    if root.exists(): raise FileExistsError('Use a new model directory')
    raw=load_candles(path,15); f,atr=inputs(raw)
    tr,va,hold=common_splits(raw,atr,cfg,horizons)
    rows=[]
    root.mkdir(parents=True,exist_ok=False)
    for h in horizons:
        child=root/f'h{h}'
        r=fit_one(raw,f,atr,tr,va,hold,replace(cfg,horizon=h),child,horizons)
        v=r['validation']; m=v['network']; base=v['constant_training_prior']; past=v['past_trend_persistence']
        lift=base['log_loss']-m['log_loss']
        rows.append({'horizon':h,'checkpoint':str(child),'selected_epoch':r['selected_epoch'],
                     'network':m,'constant_training_prior':base,'past_trend_persistence':past,
                     'log_loss_improvement':lift,
                     'candidate':lift>0 and m['balanced_accuracy']>.5 and
                         m['macro_f1']>max(base['macro_f1'],past['macro_f1']),
                     'confidence_audit':v['confidence_audit'],
                     'strong_realized_trend_subgroup':v['strong_realized_trend_subgroup']})
    ranking=sorted(rows,key=lambda r:r['log_loss_improvement'],reverse=True)
    eligible=[r for r in ranking if r['candidate']]
    summary={'version':VERSION,'horizons':list(horizons),
             'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
             'ranking_by_log_loss_improvement':ranking,
             'candidate_for_further_validation':eligible[0]['horizon'] if eligible else None,
             'external_test_evaluated':False,
             'note':'Single-seed exploratory comparison on identical timestamps; different horizons predict different targets. No trading edge established.'}
    save_json(root/'comparison.json',summary)
    return summary


def checkpoint(path,threads=None):
    root=Path(path); meta=json.loads((root/'config.json').read_text())
    if (meta['model_version']!=VERSION or meta['features']!=list(FEATURES)
            or meta['classes']!=list(CLASSES) or meta['warmup']!=WARMUP):
        raise ValueError('Binary forecast checkpoint schema mismatch')
    settings=meta['settings']
    cfg=TrendForecastSettings(**({**settings,'threads':threads} if threads is not None else settings))
    cfg.validate(); model=build_model(cfg)
    with np.load(root/'model.npz',allow_pickle=False) as data:
        mu=data['mean']; sd=data['std']
        restore(model,{k:data[k] for k in data.files if k not in ('mean','std')})
    if mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise ValueError('Invalid scaler')
    return model,cfg,mu,sd,meta


def row(raw,i,p,cfg):
    return {'candle_close_utc':pd.Timestamp(int(raw.time.iloc[i])+900,unit='s',tz='UTC').isoformat(),
            'horizon_candles':cfg.horizon,'predicted_trend':CLASSES[int(p.argmax())],
            'p_down':float(p[0]),'p_up':float(p[1])}


def predict_direction(path,model_path,threads=None):
    model,cfg,mu,sd,_=checkpoint(model_path,threads)
    raw=load_candles(path,15); f,_=inputs(raw)
    if len(raw)<WARMUP+cfg.lookback: raise ValueError('Insufficient completed M15 history')
    i=len(raw)-1; p=probabilities(model,f,np.array([i]),cfg,mu,sd)[0]
    return {'version':VERSION,**row(raw,i,p,cfg)}


def replay_direction(path,model_path,out,threads=None):
    model,cfg,mu,sd,meta=checkpoint(model_path,threads)
    raw=load_candles(path,15)
    if int(raw.time.iloc[0])<=meta['development_range'][1]:
        raise ValueError('External test must be chronologically after development')
    f,atr=inputs(raw); hs=meta['comparison_horizons']
    ix=candidates(raw,replace(cfg,horizon=max(hs)))
    before=len(ix)
    for h in hs:
        ix=ix[labels(trend_strength(raw,ix,atr,replace(cfg,horizon=h)))>=0]
    if not len(ix): raise ValueError('No eligible test windows')
    p=probabilities(model,f,ix,cfg,mu,sd)
    z=trend_strength(raw,ix,atr,cfg); y=labels(z)
    rows=[{**row(raw,int(i),p[j],cfg),'actual_trend':CLASSES[int(y[j])],
           'actual_strength_atr':float(z[j])} for j,i in enumerate(ix)]
    dst=Path(out)
    if dst.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    dst.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(dst,index=False)
    result={'version':VERSION,'settings':asdict(cfg),'shared_tie_exclusions':before-len(ix),
            'test':evaluation(raw,ix,atr,cfg,p,np.asarray(meta['training_prior'])),
            'note':'Forecast only. Confidence scores not guaranteed calibrated; overlapping windows are correlated.'}
    save_json(dst.with_suffix('.summary.json'),result)
    return result


def register(actions):
    tr=actions.add_parser('train-direction',help='v2.1: compare binary future trend horizons')
    tr.add_argument('--csv',default='data/multi_tf/development_m15.csv')
    tr.add_argument('--out',default='models/finance-v2.1')
    tr.add_argument('--horizons',type=int,nargs='+',default=list(HORIZONS))
    cfg=TrendForecastSettings()
    for name,kind in [('lookback',int),('epochs',int),('batch-size',int),('lr',float),
                      ('patience',int),('seed',int),('threads',int),('channels',int),('dropout',float)]:
        tr.add_argument('--'+name,type=kind,default=getattr(cfg,name.replace('-','_')))
    for cmd in ('predict-direction','test-direction'):
        a=actions.add_parser(cmd)
        a.add_argument('--csv',default='data/multi_tf/final_test_m15.csv')
        a.add_argument('--model',required=True); a.add_argument('--threads',type=int)
        if cmd=='test-direction': a.add_argument('--out',default='reports/v2.1-direction.csv')


def dispatch(args):
    if args.action=='train-direction':
        cfg=TrendForecastSettings(**{k:v for k,v in vars(args).items() if k not in ('action','csv','out','horizons')})
        return train_direction(args.csv,args.out,cfg,args.horizons)
    if args.action=='predict-direction': return predict_direction(args.csv,args.model,args.threads)
    if args.action=='test-direction': return replay_direction(args.csv,args.model,args.out,args.threads)
    raise ValueError('Unknown binary forecast command')
