"""v1.2: compare temporal encoders on the same v1.1 risk-adjusted targets.

CPU PyTorch is optional; old NumPy commands do not import it. No pretrained
weights, chart-rule gates, daily quota, or future features.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_model import FEATURES, WARMUP, features, load_candles, save_json, splits
from .value_model import (ValueSettings, ValueOutcomes, baselines, choices, decide,
                          evaluate, prediction, stamp, stats, targets_from_moves, utility)

VERSION = '1.2.0'
ARCHITECTURES = ('cnn', 'gru', 'tcn')


@dataclass(frozen=True)
class TemporalSettings(ValueSettings):
    loss_aversion: float = 1.1  # same as last v1.1 experiment
    epochs: int = 15
    patience: int = 4
    channels: int = 24
    dropout: float = .20
    weight_decay: float = .01
    threads: int = 4

    def validate(self):
        super().validate()
        if not 4 <= self.channels <= 128 or not 1 <= self.threads <= 64:
            raise ValueError('channels must be 4..128, threads 1..64')
        if not 0 <= self.dropout < .5 or not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid temporal regularization')


def torch_cpu(threads):
    try:
        import torch
    except ImportError as exc:
        raise ImportError('v1.2 requires PyTorch; install CPU torch in your venv') from exc
    torch.set_num_threads(threads)
    return torch


def build_model(arch, cfg, prior=None):
    if arch not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture {arch}')
    torch = torch_cpu(cfg.threads)
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(cfg.seed)
    c = cfg.channels

    class CausalConv(nn.Module):
        def __init__(self, a, b, kernel, dilation=1):
            super().__init__(); self.pad=(kernel-1)*dilation
            self.conv=nn.Conv1d(a,b,kernel,dilation=dilation)
        def forward(self,x):
            return self.conv(F.pad(x,(self.pad,0)))

    class Residual(nn.Module):
        def __init__(self,d):
            super().__init__()
            self.layers=nn.Sequential(CausalConv(c,c,3,d),nn.GELU(),nn.Dropout(cfg.dropout),
                                      CausalConv(c,c,3,d),nn.GELU())
        def forward(self,x): return x+self.layers(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dropout=nn.Dropout(cfg.input_dropout)
            if arch=='cnn':
                self.encoder=nn.Sequential(CausalConv(len(FEATURES),c,5),nn.GELU(),
                                           CausalConv(c,c,5),nn.GELU(),
                                           nn.AvgPool1d(2),CausalConv(c,c,5),nn.GELU())
            elif arch=='gru':
                self.encoder=nn.GRU(len(FEATURES),c,batch_first=True)
            else:
                dilations=[]; receptive=1; d=1
                while receptive<cfg.lookback:
                    dilations.append(d); receptive+=4*d; d*=2
                self.encoder=nn.Sequential(nn.Conv1d(len(FEATURES),c,1),
                                           *(Residual(d) for d in dilations))
            # Recent state, long-context mean, and coarse ordered summaries.
            self.head=nn.Sequential(nn.LayerNorm(c*6),nn.Linear(c*6,32),nn.GELU(),
                                    nn.Dropout(cfg.dropout),nn.Linear(32,len(choices(cfg))))
            if prior is not None:
                with torch.no_grad():
                    pr=torch.as_tensor(prior,dtype=torch.float32).clamp(1e-5,1-1e-5)
                    self.head[-1].weight.zero_(); self.head[-1].bias.copy_(torch.logit(pr))
        def forward(self,x):
            x=self.input_dropout(x)
            if arch=='gru':
                x=self.encoder(x)[0].transpose(1,2)
            else:
                x=self.encoder(x.transpose(1,2))
            summary=torch.cat((x[:,:,-1],x.mean(dim=2),F.adaptive_avg_pool1d(x,4).flatten(1)),dim=1)
            return self.head(summary)
    return Model().cpu()


def sequence_batch(f,ends,cfg,mu,sd):
    ix=np.asarray(ends)[:,None]+np.arange(1-cfg.lookback,1)[None,:]
    return np.ascontiguousarray((f[ix]-mu)/sd,dtype=np.float32)


def masked_loss(logits,y):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits
    mask=torch.isfinite(y)
    if not bool(mask.any()): return None
    return binary_cross_entropy_with_logits(logits[mask],y[mask])


def scores(model,f,ends,cfg,mu,sd):
    import torch
    model.eval(); parts=[]
    with torch.inference_mode():
        for j in range(0,len(ends),cfg.batch_size):
            x=sequence_batch(f,ends[j:j+cfg.batch_size],cfg,mu,sd)
            parts.append(model(torch.from_numpy(x)).sigmoid().cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0,len(choices(cfg))),np.float32)


def counter_context(raw,ends,a,moves,st,cfg):
    """After-the-fact audit only; opposite trailing price move is NOT a confirmed reversal."""
    change=raw.close.to_numpy(float)[ends]-raw.close.to_numpy(float)[ends-20]
    counter=((a>=0)&(((a%2==0)&(change<0))|((a%2==1)&(change>0))))
    aligned=(a>=0)&~counter&(change!=0)
    return {'definition':'chosen direction opposite/aligned with past 20-bar close change; not a reversal label or filter',
            'counter_move':stats(np.where(counter,a,-1),moves,st,cfg),
            'aligned_move':stats(np.where(aligned,a,-1),moves,st,cfg)}


def state_arrays(model):
    return {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}


def restore(model,state):
    import torch
    expected=model.state_dict()
    if set(expected)!=set(state): raise ValueError('Checkpoint tensor keys mismatch')
    params={}
    for k,v in expected.items():
        a=np.asarray(state[k]);
        if tuple(a.shape)!=tuple(v.shape) or not np.isfinite(a).all(): raise ValueError('Invalid checkpoint tensor '+k)
        params[k]=torch.as_tensor(a,dtype=v.dtype)
    model.load_state_dict(params,strict=True)


def fit_one(arch,root,raw,f,tr,va,hold,mt,st,mv,sv,cfg,mu,sd):
    torch=torch_cpu(cfg.threads)
    yt=targets_from_moves(mt,cfg); prior=np.nanmean(yt,axis=0)
    model=build_model(arch,cfg,prior); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    rng=np.random.default_rng(cfg.seed); best=np.inf; state=None; history=[]; misses=0; chosen_epoch=0
    for epoch in range(1,cfg.epochs+1):
        model.train(); order=rng.permutation(len(tr))
        for j in range(0,len(order),cfg.batch_size):
            k=order[j:j+cfg.batch_size]; x=sequence_batch(f,tr[k],cfg,mu,sd)
            opt.zero_grad(set_to_none=True); loss=masked_loss(model(torch.from_numpy(x)),torch.from_numpy(yt[k]))
            if loss is None: continue
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        pv=scores(model,f,va,cfg,mu,sd); v=evaluate(pv,mv,sv,cfg); loss=v['soft_label_log_loss']
        history.append({'epoch':epoch,'validation_loss':loss,'strategy':v['strategy']})
        print(f'[v1.2 {arch}] epoch={epoch}/{cfg.epochs} val_loss={loss:.6f} signals={v["strategy"]["signals"]} utility={v["strategy"]["risk_adjusted_utility_sum"]:.2f}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; state=state_arrays(model); chosen_epoch=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if state is None: raise ValueError('No finite validation checkpoint')
    restore(model,state); pv=scores(model,f,va,cfg,mu,sd); pt=scores(model,f,tr,cfg,mu,sd)
    v=evaluate(pv,mv,sv,cfg); base=evaluate(np.broadcast_to(prior,pv.shape),mv,sv,cfg); a,_=decide(pv,cfg)
    meta={'model_version':VERSION,'encoder':arch,'settings':asdict(cfg),'features':list(FEATURES),
          'warmup':WARMUP,'actions':choices(cfg),'parameters':sum(p.numel() for p in model.parameters()),
          'development_time_range':[int(raw.time.iloc[0]),int(raw.time.iloc[-1])]}
    report={**meta,'selected_epoch':chosen_epoch,'epochs':history,'train':evaluate(pt,mt,st,cfg),
            'validation':v,'validation_constant_training_prior':base,'validation_baselines':baselines(mv,sv,cfg),
            'validation_direction_context':counter_context(raw,va,a,mv,sv,cfg),
            'beats_constant_prior_loss':v['soft_label_log_loss']<base['soft_label_log_loss'],
            'beats_wait_utility':v['strategy']['risk_adjusted_utility_sum']>0,
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
            'assumptions':'Same v1.1 outcome utility, no indicator gates or quota; hypothetical independent close fills; no real execution. Counter-move context is descriptive only.'}
    root.mkdir(parents=True,exist_ok=False); np.savez_compressed(root/'model.npz',**state,mean=mu,std=sd)
    save_json(root/'config.json',meta); save_json(root/'training.json',report); return report


def train_temporal(csv,m1_csv,out,cfg=None,architectures=ARCHITECTURES):
    cfg=cfg or TemporalSettings(); cfg.validate(); architectures=tuple(architectures)
    if not architectures or len(set(architectures))!=len(architectures) or any(a not in ARCHITECTURES for a in architectures):
        raise ValueError('Select unique cnn/gru/tcn architectures')
    torch_cpu(cfg.threads); root=Path(out)
    if root.exists(): raise FileExistsError('Use a new comparison directory to preserve checkpoints')
    raw=load_candles(csv,15); m1=load_candles(m1_csv,1); f=features(raw); tr,va,hold=splits(len(raw),cfg)
    if int(m1.time.iloc[0])>int(raw.time.iloc[tr[0]+1]) or int(m1.time.iloc[-1])<int(raw.time.iloc[va[-1]+cfg.horizon])+840:
        raise ValueError('M1 must cover the training AND validation future-label period')
    print('[v1.2] Preparing shared v1.1 outcome labels...',file=sys.stderr,flush=True)
    engine=ValueOutcomes(raw,m1,cfg); mt,st=engine.labels(tr); mv,sv=engine.labels(va)
    if min(np.isfinite(mt).sum(axis=0))<20 or min(np.isfinite(mv).sum(axis=0))<20:
        raise ValueError('Too few known outcomes')
    used=f[tr[0]-cfg.lookback+1:tr[-1]+1]; mu=used.mean(axis=0); sd=np.maximum(used.std(axis=0),1e-6)
    root.mkdir(parents=True,exist_ok=False); ranking=[]
    for arch in architectures:
        r=fit_one(arch,root/arch,raw,f,tr,va,hold,mt,st,mv,sv,cfg,mu,sd); v=r['validation']
        ranking.append({'encoder':arch,'checkpoint':str(root/arch),'parameters':r['parameters'],'selected_epoch':r['selected_epoch'],
                        'validation_loss':v['soft_label_log_loss'],'constant_prior_loss':r['validation_constant_training_prior']['soft_label_log_loss'],
                        'beats_constant_prior_loss':r['beats_constant_prior_loss'],'beats_wait_utility':r['beats_wait_utility'],
                        'strategy':v['strategy'],'direction_context':r['validation_direction_context']})
    ranking.sort(key=lambda r:r['validation_loss']); eligible=[r for r in ranking if r['beats_constant_prior_loss'] and r['beats_wait_utility']]
    summary={'version':VERSION,'settings':asdict(cfg),'ranking_by_validation_loss':ranking,
             'best_validation_loss_encoder':ranking[0]['encoder'],
             'candidate_for_further_validation':eligible[0]['encoder'] if eligible else None,
             'holdout_evaluated':False,'external_test_evaluated':False,
             'note':'Candidate requires both lower validation loss than prior and positive risk-adjusted utility; not evidence of profitability; compare on untouched data.'}
    save_json(root/'comparison.json',summary); return summary


def checkpoint(path,threads=None):
    root=Path(path); meta=json.loads((root/'config.json').read_text())
    if meta['model_version']!=VERSION or meta['features']!=list(FEATURES) or meta['warmup']!=WARMUP:
        raise ValueError('v1.2 checkpoint schema mismatch')
    settings=meta['settings']; cfg=TemporalSettings(**({**settings,'threads':threads} if threads is not None else settings)); cfg.validate()
    if meta['actions']!=[list(x) for x in choices(cfg)]: raise ValueError('Action schema mismatch')
    model=build_model(meta['encoder'],cfg)
    with np.load(root/'model.npz',allow_pickle=False) as data:
        mu=data['mean']; sd=data['std']; restore(model,{k:data[k] for k in data.files if k not in ('mean','std')})
    if mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any(): raise ValueError('Invalid scaler')
    return model,cfg,mu,sd,meta


def predict_temporal(csv,path,threads=None):
    model,cfg,mu,sd,meta=checkpoint(path,threads); raw=load_candles(csv,15); f=features(raw)
    if len(raw)<WARMUP+cfg.lookback: raise ValueError('Insufficient completed M15 candles')
    p=scores(model,f,np.array([len(raw)-1]),cfg,mu,sd); a,v=decide(p,cfg)
    return {'version':VERSION,'encoder':meta['encoder'],**prediction(raw,len(raw)-1,int(a[0]),v[0],cfg)}


def backtest_temporal(csv,m1_csv,path,out,threads=None):
    model,cfg,mu,sd,meta=checkpoint(path,threads); raw=load_candles(csv,15); f=features(raw)
    if int(raw.time.iloc[0])<=meta['development_time_range'][1]: raise ValueError('External replay must be chronologically after development')
    ends=np.arange(WARMUP+cfg.lookback-1,len(raw),dtype=np.int64)
    if not len(ends): raise ValueError('Insufficient history')
    p=scores(model,f,ends,cfg,mu,sd); a,v=decide(p,cfg)
    engine=ValueOutcomes(raw,load_candles(m1_csv,1),cfg); moves,st=engine.labels(ends); rows=[]
    for k,i in enumerate(ends):
        chosen=int(a[k]); status,m,j,minute=engine.one(int(i),chosen) if chosen>=0 else ('WAIT',np.nan,None,None)
        rows.append({**prediction(raw,i,chosen,v[k],cfg),'prior_20_bar_close_change':float(raw.close.iloc[i]-raw.close.iloc[i-20]),
                     'result':status,'exit_m15_open_utc':stamp(raw.time.iloc[j]) if j is not None else None,
                     'exit_m1_open_utc':stamp(minute) if minute is not None else None,
                     'net_usd_per_oz':float(m) if np.isfinite(m) else None,
                     'risk_adjusted_utility':float(utility(m,cfg)) if np.isfinite(m) else None})
    dst=Path(out)
    if dst.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    dst.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(dst,index=False,encoding='utf-8-sig')
    report={'version':VERSION,'encoder':meta['encoder'],'settings':asdict(cfg),'overall':stats(a,moves,st,cfg),
            'baselines':baselines(moves,st,cfg),'evaluation':evaluate(p,moves,st,cfg),
            'direction_context':counter_context(raw,ends,a,moves,st,cfg),
            'assumptions':'Independent overlapping hypothetical close fills; conservative ambiguous SL; timeout close; missing/pending excluded; fixed cost only; no portfolio P&L.'}
    save_json(dst.with_suffix('.summary.json'),report); return report


def register(actions):
    tr=actions.add_parser('train-temporal',help='v1.2: train and compare CPU CNN/GRU/TCN on identical price-outcome data')
    tr.add_argument('--csv','--data','--excel',dest='data',required=True); tr.add_argument('--m1-csv',required=True)
    tr.add_argument('--out',default='models/finance-v1.2'); tr.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES))
    cfg=TemporalSettings()
    for name,kind in [('lookback',int),('sl',float),('horizon',int),('cost',float),('loss-aversion',float),('tolerance',float),('epochs',int),('batch-size',int),('lr',float),('patience',int),('seed',int),('input-dropout',float),('channels',int),('dropout',float),('weight-decay',float),('threads',int)]:
        tr.add_argument('--'+name,type=kind,default=getattr(cfg,name.replace('-','_')))
    tr.add_argument('--targets',type=float,nargs='+',default=list(cfg.targets))
    for cmd in ('predict-temporal','backtest-temporal'):
        sub=actions.add_parser(cmd); sub.add_argument('--csv','--data','--excel',dest='data',required=True)
        sub.add_argument('--model',required=True); sub.add_argument('--threads',type=int)
        if cmd=='backtest-temporal': sub.add_argument('--m1-csv',required=True); sub.add_argument('--out',default='reports/v1.2-decisions.csv')


def dispatch(args):
    if args.action=='train-temporal':
        cfg=TemporalSettings(**{k:v for k,v in vars(args).items() if k not in ('action','data','m1_csv','out','architectures')})
        return train_temporal(args.data,args.m1_csv,args.out,cfg,args.architectures)
    if args.action=='predict-temporal': return predict_temporal(args.data,args.model,args.threads)
    if args.action=='backtest-temporal': return backtest_temporal(args.data,args.m1_csv,args.model,args.out,args.threads)
    raise ValueError('Unknown temporal command')
