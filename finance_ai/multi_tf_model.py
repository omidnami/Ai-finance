"""v1.3: independently encode four native candle sequences, fuse at M15 close."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_model import FEATURES, WARMUP, features, save_json
from .value_model import (ValueOutcomes, baselines, choices, decide, evaluate,
                          prediction, stamp, stats, targets_from_moves, utility)
from .temporal_model import (TemporalSettings, torch_cpu, masked_loss,
                             state_arrays, restore, counter_context)

VERSION = '1.3.0'
FRAMES = {'m1': 60, 'm5': 300, 'm15': 900, 'h1': 3600}


@dataclass(frozen=True)
class MultiSettings(TemporalSettings):
    channels: int = 16
    excursion_weight: float = .15

    def validate(self):
        super().validate()
        if not np.isfinite(self.excursion_weight) or self.excursion_weight < 0:
            raise ValueError('Invalid excursion weight')


def load_frame(path, seconds):
    # The older loader discards non-M15 volume. Preserve actual OHLCV in every branch.
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        sep = csv.Sniffer().sniff(f.readline(), delimiters=',;\t').delimiter
    df = pd.read_csv(path, sep=sep, encoding='utf-8-sig')
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = ['time','open','high','low','close','volume','timeframe']
    if not set(cols).issubset(df):
        raise ValueError(f'{path}: missing {sorted(set(cols)-set(df))}')
    df = df[cols].apply(pd.to_numeric, errors='raise')
    tf = seconds//60
    accepted = [tf, 16385] if tf == 60 else [tf]
    t = df.time.to_numpy(float)
    if (not len(df) or not np.isfinite(df.to_numpy(float)).all()
            or not df.timeframe.isin(accepted).all()
            or (t <= 0).any() or (t % seconds != 0).any()
            or (np.diff(t) <= 0).any()):
        raise ValueError(f'{path}: invalid timeframe or ascending unique UTC opening timestamps')
    p = df[['open','high','low','close']]
    if ((p <= 0).any().any() or (df.high < p.max(axis=1)).any()
            or (df.low > p.min(axis=1)).any() or (df.volume < 0).any()):
        raise ValueError(f'{path}: invalid OHLCV')
    df['time'] = df.time.astype(np.int64)
    df['timeframe'] = tf
    return df.reset_index(drop=True)


class Dataset:
    def __init__(self, directory, prefix, lookback):
        self.lookback = lookback
        self.raw = {k:load_frame(Path(directory)/f'{prefix}_{k}.csv', sec)
                    for k,sec in FRAMES.items()}
        self.signal = self.raw['m15']
        self.opens = {k:r.time.to_numpy(np.int64) for k,r in self.raw.items()}
        self.closes = {k:self.opens[k]+sec for k,sec in FRAMES.items()}
        self.range = {k:[int(t[0]),int(self.closes[k][-1])] for k,t in self.opens.items()}
        self.features = {k:features(r) for k,r in self.raw.items()}
        now = self.closes['m15']
        self.indices = {}
        self.valid = np.ones(len(now), dtype=bool)
        for k,sec in FRAMES.items():
            # Only completed candles at the exact decision time; never use open H1.
            ix = np.searchsorted(self.closes[k], now, side='right')-1
            self.indices[k] = ix
            age = now-self.closes[k][np.maximum(ix,0)]
            self.valid &= (ix >= WARMUP+lookback-1) & (age >= 0) & (age < sec)

    def eligible(self, start=0, stop=None, horizon=0, isolated=False):
        stop = len(self.signal) if stop is None else stop
        ix = np.arange(start, max(start,stop-horizon), dtype=np.int64)
        ix = ix[self.valid[ix]]
        if isolated and len(ix):
            # All four raw input windows remain in their respective M15 block.
            # Causal indicators may retain earlier history, just as in v1.2.
            boundary = self.opens['m15'][start]
            for k in FRAMES:
                first = self.indices[k][ix]-self.lookback+1
                ix = ix[self.opens[k][first] >= boundary]
        return ix

    def scaler(self, ends):
        result = {}
        for k in FRAMES:
            ix = self.indices[k][ends]
            f = self.features[k][ix.min()-self.lookback+1:ix.max()+1]
            result[k] = (f.mean(axis=0),np.maximum(f.std(axis=0),1e-6))
        return result

    def batch(self, ends, scaler):
        import torch
        out = {}
        offsets = np.arange(1-self.lookback,1)
        for k in FRAMES:
            ix = self.indices[k][ends,None]+offsets[None,:]
            mu,sd = scaler[k]
            x = np.ascontiguousarray((self.features[k][ix]-mu)/sd,dtype=np.float32)
            out[k] = torch.from_numpy(x)
        return out

    def block_splits(self, horizon):
        n=len(self.signal); a,b=int(n*.7),int(n*.85)
        items=[self.eligible(s,e,horizon,True) for s,e in ((0,a),(a,b),(b,n))]
        if min(map(len,items))<20:
            raise ValueError('Insufficient shared history for isolated four-timeframe splits; export a longer M15/H1 range or reduce lookback')
        return items


def build_model(cfg, prior=None):
    torch=torch_cpu(cfg.threads)
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(cfg.seed)
    c=cfg.channels

    class Causal(nn.Module):
        def __init__(self,a,b,k,d=1):
            super().__init__()
            self.pad=(k-1)*d
            self.conv=nn.Conv1d(a,b,k,dilation=d)
        def forward(self,x):
            return self.conv(F.pad(x,(self.pad,0)))

    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(Causal(len(FEATURES),c,5),nn.GELU(),
                Causal(c,c,5),nn.GELU(),nn.AvgPool1d(2),
                Causal(c,c,5,2),nn.GELU())
            self.drop=nn.Dropout(cfg.input_dropout)
        def forward(self,x):
            z=self.encoder(self.drop(x).transpose(1,2))
            return torch.cat((z[:,:,-1],z.mean(2),
                              F.adaptive_avg_pool1d(z,4).flatten(1)),dim=1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches=nn.ModuleDict({k:Branch() for k in FRAMES})
            self.fusion=nn.Sequential(nn.LayerNorm(c*6*len(FRAMES)),
                nn.Linear(c*6*len(FRAMES),64),nn.GELU(),nn.Dropout(cfg.dropout),
                nn.Linear(64,32),nn.GELU(),nn.Dropout(cfg.dropout))
            self.value=nn.Linear(32,len(choices(cfg)))
            self.excursion=nn.Linear(32,2)
            if prior is not None:
                with torch.no_grad():
                    self.value.weight.zero_()
                    self.value.bias.copy_(torch.logit(torch.as_tensor(prior,dtype=torch.float32).clamp(1e-5,1-1e-5)))
        def forward(self,x):
            z=self.fusion(torch.cat([self.branches[k](x[k]) for k in FRAMES],dim=1))
            return self.value(z),F.softplus(self.excursion(z))
    return Model().cpu()


def excursion_labels(raw,ends,horizon):
    # Auxiliary future maximum upward/downward movement (not a first-touch label).
    # Only used for supervised targets, never available as input or entry gate.
    p=raw[['high','low','close']].to_numpy(float)
    y=np.empty((len(ends),2),np.float32)
    for k,i in enumerate(ends):
        later=p[i+1:i+1+horizon]
        if len(later)!=horizon: raise ValueError('Incomplete excursion label horizon')
        y[k]=np.log1p([max(0.,later[:,0].max()-p[i,2]),
                       max(0.,p[i,2]-later[:,1].min())])
    return y


def scores(model,data,ends,cfg,scaler):
    torch=torch_cpu(cfg.threads); model.eval(); pp=[]; ee=[]
    with torch.inference_mode():
        for j in range(0,len(ends),cfg.batch_size):
            z,e=model(data.batch(ends[j:j+cfg.batch_size],scaler))
            pp.append(z.sigmoid().numpy())
            ee.append(e.numpy())
    if not pp: raise ValueError('No eligible candles')
    return np.concatenate(pp),np.concatenate(ee)


def evaluate_multi(p,e,moves,status,raw,ends,cfg):
    result=evaluate(p,moves,status,cfg)
    result['direction_context']=counter_context(raw,ends,decide(p,cfg)[0],moves,status,cfg)
    complete=ends+cfg.horizon < len(raw)
    if complete.any():
        truth=excursion_labels(raw,ends[complete],cfg.horizon)
        result['excursion_log_mae']=dict(zip(('up','down'),map(float,np.abs(e[complete]-truth).mean(axis=0))))
    return result


def train_multi(directory,out,cfg=None):
    cfg=cfg or MultiSettings(); cfg.validate(); root=Path(out)
    if root.exists(): raise FileExistsError('Use a new output directory')
    torch=torch_cpu(cfg.threads)
    data=Dataset(directory,'development',cfg.lookback)
    raw=data.signal; tr,va,hold=data.block_splits(cfg.horizon)
    print(f'[v1.3] candles={ {k:len(r) for k,r in data.raw.items()} } splits={len(tr)}/{len(va)}/{len(hold)}',file=sys.stderr,flush=True)
    if data.range['m1'][1]<int(raw.time.iloc[va[-1]+cfg.horizon])+900:
        raise ValueError('M1 does not cover validation label horizon')
    engine=ValueOutcomes(raw,data.raw['m1'],cfg)
    mt,st=engine.labels(tr); mv,sv=engine.labels(va)
    yt=targets_from_moves(mt,cfg)
    if min(np.isfinite(mt).sum(axis=0))<20 or min(np.isfinite(mv).sum(axis=0))<20:
        raise ValueError('Too few known outcomes')
    aux=excursion_labels(raw,tr,cfg.horizon)
    prior=np.nanmean(yt,axis=0); scaler=data.scaler(tr)
    model=build_model(cfg,prior); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    rng=np.random.default_rng(cfg.seed)
    best=np.inf; state=None; selected=0; misses=0; history=[]
    for epoch in range(1,cfg.epochs+1):
        model.train(); order=rng.permutation(len(tr))
        for j in range(0,len(tr),cfg.batch_size):
            ix=order[j:j+cfg.batch_size]; opt.zero_grad(set_to_none=True)
            z,e=model(data.batch(tr[ix],scaler))
            primary=masked_loss(z,torch.from_numpy(yt[ix]))
            if primary is None: continue
            loss=primary+cfg.excursion_weight*torch.nn.functional.smooth_l1_loss(e,torch.from_numpy(aux[ix]))
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        p,e=scores(model,data,va,cfg,scaler); v=evaluate(p,mv,sv,cfg)
        loss=v['soft_label_log_loss']
        history.append({'epoch':epoch,'validation_loss':loss,'strategy':v['strategy']})
        print(f'[v1.3] epoch={epoch}/{cfg.epochs} val_loss={loss:.6f} signals={v["strategy"]["signals"]} utility={v["strategy"]["risk_adjusted_utility_sum"]:.2f}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; state=state_arrays(model); selected=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if state is None: raise ValueError('No finite validation checkpoint')
    restore(model,state)
    pt,et=scores(model,data,tr,cfg,scaler); pv,ev=scores(model,data,va,cfg,scaler)
    validation=evaluate_multi(pv,ev,mv,sv,raw,va,cfg)
    base=evaluate(np.broadcast_to(prior,pv.shape),mv,sv,cfg)
    meta={'model_version':VERSION,'encoder':'four_branch_cnn','settings':asdict(cfg),
          'frames':FRAMES,'features':list(FEATURES),'warmup':WARMUP,'actions':choices(cfg),
          'development_ranges':data.range,'parameters':sum(p.numel() for p in model.parameters())}
    report={**meta,'selected_epoch':selected,'epochs':history,
            'input_rows':{k:len(r) for k,r in data.raw.items()},
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_not_evaluated':len(hold)},
            'm15_candidates_with_sufficient_synced_history':int(data.valid.sum()),
            'train':evaluate_multi(pt,et,mt,st,raw,tr,cfg),
            'validation':validation,'validation_constant_training_prior':base,
            'validation_baselines':baselines(mv,sv,cfg),
            'candidate_for_further_validation':validation['soft_label_log_loss']<base['soft_label_log_loss'] and validation['strategy']['risk_adjusted_utility_sum']>0,
            'assumptions':'Completed candles, no daily quota; auxiliary excursion is not TP/SL probability or a reversal gate. No holdout/test evaluated. Hypothetical independent overlapping fills; fixed cost only.'}
    root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'model.npz',**state,**{f'{k}_{name}':a for k,(mu,sd) in scaler.items() for name,a in [('mean',mu),('std',sd)]})
    save_json(root/'config.json',meta); save_json(root/'training.json',report)
    return report


def checkpoint(path,threads=None):
    root=Path(path); meta=json.loads((root/'config.json').read_text())
    if (meta['model_version']!=VERSION or meta['frames']!=FRAMES
            or meta['features']!=list(FEATURES) or meta['warmup']!=WARMUP
            or meta['encoder']!='four_branch_cnn'):
        raise ValueError('v1.3 checkpoint schema mismatch')
    settings=meta['settings']
    cfg=MultiSettings(**({**settings,'threads':threads} if threads is not None else settings))
    cfg.validate()
    if meta['actions']!=[list(x) for x in choices(cfg)]: raise ValueError('Action schema mismatch')
    model=build_model(cfg); scaler={}
    with np.load(root/'model.npz',allow_pickle=False) as archive:
        keys=set()
        for k in FRAMES:
            names=(f'{k}_mean',f'{k}_std'); keys.update(names)
            mu,sd=(archive[name] for name in names)
            if (mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all()
                    or not np.isfinite(sd).all() or (sd<=0).any()):
                raise ValueError(f'Invalid {k} scaler')
            scaler[k]=(mu,sd)
        restore(model,{k:archive[k] for k in archive.files if k not in keys})
    return model,cfg,scaler,meta


def row_prediction(data,i,a,v,e,cfg):
    up,down=np.expm1(np.clip(e.astype(float),0,15))
    return {**prediction(data.signal,i,a,v,cfg),
            'predicted_up_excursion_logspace_estimate_usd':float(up),
            'predicted_down_excursion_logspace_estimate_usd':float(down),
            **{f'{k}_last_close_utc':stamp(data.closes[k][data.indices[k][i]]) for k in FRAMES}}


def predict_multi(directory,path,prefix='final_test',threads=None):
    model,cfg,scaler,_=checkpoint(path,threads)
    data=Dataset(directory,prefix,cfg.lookback); i=len(data.signal)-1
    if not data.valid[i]: raise ValueError('Latest M15 lacks synchronized completed history')
    p,e=scores(model,data,np.array([i]),cfg,scaler); a,v=decide(p,cfg)
    return {'version':VERSION,**row_prediction(data,i,int(a[0]),v[0],e[0],cfg)}


def backtest_multi(directory,path,out,threads=None):
    model,cfg,scaler,meta=checkpoint(path,threads)
    data=Dataset(directory,'final_test',cfg.lookback); raw=data.signal
    for k in FRAMES:
        if data.range[k][0]<meta['development_ranges'][k][1]:
            raise ValueError(f'{k} external replay must be chronologically after development')
    ends=data.eligible(horizon=cfg.horizon)
    p,e=scores(model,data,ends,cfg,scaler); actions,v=decide(p,cfg)
    engine=ValueOutcomes(raw,data.raw['m1'],cfg); moves,st=engine.labels(ends)
    rows=[]
    for k,i in enumerate(ends):
        a=int(actions[k])
        status,m,j,minute=engine.one(int(i),a) if a>=0 else ('WAIT',np.nan,None,None)
        rows.append({**row_prediction(data,int(i),a,v[k],e[k],cfg),'result':status,
                     'prior_20_bar_close_change':float(raw.close.iloc[i]-raw.close.iloc[i-20]),
                     'exit_m15_open_utc':stamp(raw.time.iloc[j]) if j is not None else None,
                     'exit_m1_open_utc':stamp(minute) if minute is not None else None,
                     'net_usd_per_oz':float(m) if np.isfinite(m) else None,
                     'risk_adjusted_utility':float(utility(m,cfg)) if np.isfinite(m) else None})
    dst=Path(out)
    if dst.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    dst.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dst,index=False,encoding='utf-8-sig')
    report={'version':VERSION,'settings':asdict(cfg),'test_ranges':data.range,
            'input_rows':{k:len(r) for k,r in data.raw.items()},
            'overall':stats(actions,moves,st,cfg),'baselines':baselines(moves,st,cfg),
            'evaluation':evaluate_multi(p,e,moves,st,raw,ends,cfg),
            'assumptions':'Only complete future-horizon candidates evaluated; hypothetical overlapping close fills, conservative ambiguous SL, fixed cost only; no portfolio P&L.'}
    save_json(dst.with_suffix('.summary.json'),report)
    return report


def register(actions):
    tr=actions.add_parser('train-multi',help='v1.3: four-timeframe CNN fusion')
    tr.add_argument('--data-dir',default='data/multi_tf'); tr.add_argument('--out',default='models/finance-v1.3')
    cfg=MultiSettings()
    for name,kind in [('lookback',int),('sl',float),('horizon',int),('cost',float),
                      ('loss-aversion',float),('tolerance',float),('epochs',int),
                      ('batch-size',int),('lr',float),('patience',int),('seed',int),
                      ('input-dropout',float),('channels',int),('dropout',float),
                      ('weight-decay',float),('threads',int),('excursion-weight',float)]:
        tr.add_argument('--'+name,type=kind,default=getattr(cfg,name.replace('-','_')))
    tr.add_argument('--targets',type=float,nargs='+',default=list(cfg.targets))
    for cmd in ('predict-multi','backtest-multi'):
        sub=actions.add_parser(cmd)
        sub.add_argument('--data-dir',default='data/multi_tf')
        sub.add_argument('--model',default='models/finance-v1.3')
        sub.add_argument('--threads',type=int)
        if cmd=='predict-multi': sub.add_argument('--prefix',default='final_test')
        else: sub.add_argument('--out',default='reports/v1.3-decisions.csv')


def dispatch(args):
    if args.action=='train-multi':
        cfg=MultiSettings(**{k:v for k,v in vars(args).items() if k not in ('action','data_dir','out')})
        return train_multi(args.data_dir,args.out,cfg)
    if args.action=='predict-multi': return predict_multi(args.data_dir,args.model,args.prefix,args.threads)
    if args.action=='backtest-multi': return backtest_multi(args.data_dir,args.model,args.out,args.threads)
    raise ValueError('Unknown multi-timeframe command')
