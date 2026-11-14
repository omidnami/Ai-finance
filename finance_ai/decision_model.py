"""v1.0: learn entries from future price outcomes, never indicator-generated labels."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '1.0.0'
SIDES = ('BUY', 'SELL')
WARMUP = 60
FEATURES = ('open_delta', 'high_delta', 'low_delta', 'close_delta', 'log_return',
            'body', 'upper_wick', 'lower_wick', 'range', 'log_volume', 'atr14',
            'rsi14', 'bb_position', 'bb_width', 'ema10_distance', 'ema20_distance',
            'ema50_distance', 'macd', 'macd_signal', 'range20_position',
            'range50_position','hour_sin','hour_cos','weekday_sin','weekday_cos','log_gap')


@dataclass(frozen=True)
class DecisionSettings:
    lookback: int = 100
    tp: float = 2.5
    sl: float = 2.0
    horizon: int = 20
    cost: float = 0.0
    tolerance: float = 0.02
    epochs: int = 30
    batch_size: int = 128
    hidden: tuple[int, ...] = (128, 64, 32)
    lr: float = 0.001
    patience: int = 5
    seed: int = 42

    def validate(self):
        if not 100 <= self.lookback <= 500 or self.horizon < 1 or self.horizon > 500:
            raise ValueError('lookback must be 100..500; horizon must be 1..500')
        if not all(np.isfinite(x) for x in (self.tp,self.sl,self.cost,self.tolerance,self.lr)) or min(self.tp,self.sl,self.lr) <= 0 or not 0 <= self.cost < self.tp or self.tolerance < 0:
            raise ValueError('Invalid TP, SL, cost, tolerance or learning rate')
        if min(self.epochs,self.batch_size,self.patience) < 1 or not self.hidden or any(x<1 for x in self.hidden):
            raise ValueError('Invalid training settings')


def load_candles(path, timeframe):
    p = Path(path)
    if p.suffix.lower() == '.csv':
        # Detect separator from the header only; use fast C parser for 850-day M1 exports.
        import csv
        with p.open(encoding='utf-8-sig',newline='') as f:
            sep = csv.Sniffer().sniff(f.readline(), delimiters=',;\t').delimiter
        df = pd.read_csv(p,sep=sep,encoding='utf-8-sig')
    elif p.suffix.lower() in ('.xlsx','.xls'):
        df = pd.read_excel(p)
    else:
        raise ValueError('Expected CSV or Excel candle data')
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = ['time','open','high','low','close','timeframe'] + (['volume'] if timeframe == 15 else [])
    if not set(cols).issubset(df):
        raise ValueError(f'Missing columns: {sorted(set(cols)-set(df))}')
    df = df[cols].apply(pd.to_numeric,errors='raise')
    a = df.to_numpy(float)
    if not len(a) or not np.isfinite(a).all() or (df.timeframe != timeframe).any():
        raise ValueError('Invalid or empty candle data/timeframe')
    t = df.time.to_numpy(float)
    if (t <= 0).any() or (t % (60*timeframe) != 0).any() or (np.diff(t)<=0).any():
        raise ValueError('Expected strictly ascending unique UTC Unix candle-opening timestamps')
    if (df[['open','high','low','close']]<=0).any().any() or (df.high<df[['open','low','close']].max(axis=1)).any() or (df.low>df[['open','high','close']].min(axis=1)).any():
        raise ValueError('Invalid OHLC ranges')
    if timeframe==15 and (df.volume<0).any():
        raise ValueError('Negative volume')
    return df.reset_index(drop=True)


def features(raw):
    """Past-only numerical inputs; none constitutes an entry gate."""
    o,h,l,c = (raw[n].astype(float) for n in ('open','high','low','close'))
    prev = c.shift(1)
    change = c-prev
    tr = pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    up = change.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    dn = (-change.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi = (up/(up+dn).replace(0,np.nan)).fillna(.5)
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    ema = lambda n: c.ewm(span=n,adjust=False,min_periods=n).mean()
    macd = ema(12)-ema(26)
    t = pd.to_datetime(raw.time,unit='s',utc=True)
    hour = t.dt.hour+t.dt.minute/60
    f = pd.DataFrame(index=raw.index)
    for k,s in zip(('open','high','low','close'),(o,h,l,c)):
        f[k+'_delta'] = s-prev
    f['log_return'] = np.log(c/prev)
    f['body']=c-o; f['upper_wick']=h-np.maximum(o,c); f['lower_wick']=np.minimum(o,c)-l
    f['range']=h-l; f['log_volume']=np.log1p(raw.volume); f['atr14']=atr
    f['rsi14']=rsi; f['bb_position']=((c-mid)/(2*sd.replace(0,np.nan))).fillna(0)
    f['bb_width']=4*sd
    for n in (10,20,50): f[f'ema{n}_distance']=c-ema(n)
    f['macd']=macd; f['macd_signal']=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    for n in (20,50):
        lo=l.rolling(n).min(); hi=h.rolling(n).max()
        f[f'range{n}_position']=((c-lo)/(hi-lo).replace(0,np.nan)).fillna(.5)
    f['hour_sin']=np.sin(2*np.pi*hour/24); f['hour_cos']=np.cos(2*np.pi*hour/24)
    f['weekday_sin']=np.sin(2*np.pi*t.dt.weekday/7); f['weekday_cos']=np.cos(2*np.pi*t.dt.weekday/7)
    f['log_gap']=np.log1p(raw.time.diff()/900)
    a=f[list(FEATURES)].to_numpy(np.float32)
    if len(a)<=WARMUP or not np.isfinite(a[WARMUP:]).all():
        raise ValueError('Insufficient candles or invalid features after warmup')
    return a


class Outcomes:
    """First future wick touch. M1 disambiguates M15; gaps/missing data not invented."""
    def __init__(self, raw, m1, cfg):
        self.cfg=cfg
        self.t=raw.time.to_numpy(np.int64); self.p=raw[['open','high','low','close']].to_numpy(float)
        self.mt=m1.time.to_numpy(np.int64); self.mp=m1[['open','high','low','close']].to_numpy(float)
        self.cache={}

    def minutes(self,j):
        if j not in self.cache:
            t=self.t[j]; k=np.searchsorted(self.mt,t); sub=self.mp[k:k+15]
            if len(sub)!=15 or not np.array_equal(self.mt[k:k+15],t+60*np.arange(15)):
                self.cache[j]=('M1_MISSING',None,k)
            elif not np.allclose([sub[0,0],sub[:,1].max(),sub[:,2].min(),sub[-1,3]],self.p[j],rtol=0,atol=self.cfg.tolerance):
                self.cache[j]=('M1_MISMATCH',None,k)
            else: self.cache[j]=('OK',sub,k)
        return self.cache[j]

    def one(self,i,side):
        e=self.p[i,3]; tp=e+side*self.cfg.tp; sl=e-side*self.cfg.sl
        last=min(i+self.cfg.horizon,len(self.p)-1)
        def hits(p):
            return (p[1]>=tp,p[2]<=sl) if side==1 else (p[2]<=tp,p[1]>=sl)
        for j in range(i+1,last+1):
            win,loss=hits(self.p[j])
            if not (win or loss): continue
            if not (win and loss): return ('TP' if win else 'SL',j,None)
            status,sub,k=self.minutes(j)
            if status!='OK': return status,j,None
            for n,row in enumerate(sub):
                w,s=hits(row)
                if w or s: return ('AMBIGUOUS' if w and s else 'TP' if w else 'SL',j,int(self.mt[k+n]))
            return 'M1_MISMATCH',j,None
        return ('NO_HIT' if last==i+self.cfg.horizon else 'PENDING_END',None,None)

    def labels(self,ends):
        # Exact TP/SL only: do not teach the model that unknown outcomes are known losses.
        y=np.full((len(ends),2),np.nan,np.float32); statuses=np.empty((len(ends),2),object)
        for n,i in enumerate(ends):
            for k,d in enumerate((1,-1)):
                status,_,_=self.one(int(i),d); statuses[n,k]=status
                if status in ('TP','SL'): y[n,k]=status=='TP'
        return y,statuses


class Net:
    """Independent sigmoid heads with masked BCE and Adam, NumPy float32 on CPU."""
    def __init__(self,sizes,seed=42):
        rng=np.random.default_rng(seed); self.sizes=tuple(sizes)
        self.w=[(rng.standard_normal((a,b))*np.sqrt(2/a)).astype(np.float32) for a,b in zip(sizes,sizes[1:])]
        self.b=[np.zeros((1,b),np.float32) for b in sizes[1:]]
        self.params=[v for pair in zip(self.w,self.b) for v in pair]
        self.m=[np.zeros_like(v) for v in self.params]; self.v=[np.zeros_like(v) for v in self.params]; self.step=0

    def forward(self,x):
        acts=[x]
        for w,b in zip(self.w[:-1],self.b[:-1]): acts.append(np.maximum(acts[-1]@w+b,0))
        z=acts[-1]@self.w[-1]+self.b[-1]
        p=1/(1+np.exp(-np.clip(z,-40,40)))
        return z,p,acts

    def train(self,x,y,lr):
        mask=np.isfinite(y); n=mask.sum()
        if not n: return
        targets=np.nan_to_num(y); z,p,acts=self.forward(x)
        delta=np.where(mask,p-targets,0)/n
        gradients=[]
        for i in reversed(range(len(self.w))):
            gradients[0:0]=[acts[i].T@delta,delta.sum(axis=0,keepdims=True)]
            if i: delta=(delta@self.w[i].T)*(acts[i]>0)
        self.step+=1
        for i,(param,g) in enumerate(zip(self.params,gradients)):
            self.m[i]*=.9; self.m[i]+=.1*g
            self.v[i]*=.999; self.v[i]+=.001*g*g
            mh=self.m[i]/(1-.9**self.step); vh=self.v[i]/(1-.999**self.step)
            param-=lr*(mh/(np.sqrt(vh)+1e-8)+(1e-4*param if i%2==0 else 0))

    def state(self): return {f'p{i}':p.copy() for i,p in enumerate(self.params)}
    def restore(self,state):
        if set(state)!=set(f'p{i}' for i in range(len(self.params))): raise ValueError('Checkpoint parameter mismatch')
        for i,p in enumerate(self.params):
            v=state[f'p{i}']
            if p.shape!=v.shape or not np.isfinite(v).all(): raise ValueError('Invalid checkpoint weights')
            p[:]=v


def windows(f,ends,lookback,mu,sd):
    indices=np.asarray(ends)[:,None]+np.arange(-lookback+1,1)[None,:]
    return ((f[indices]-mu)/sd).reshape(len(indices),-1).astype(np.float32)


def probabilities(net,f,ends,cfg,mu,sd):
    parts=[]
    for i in range(0,len(ends),cfg.batch_size):
        parts.append(net.forward(windows(f,ends[i:i+cfg.batch_size],cfg.lookback,mu,sd))[1])
    return np.concatenate(parts) if parts else np.empty((0,2),np.float32)


def actions(p,cfg):
    ev=p*(cfg.tp+cfg.sl)-cfg.sl-cfg.cost
    best=ev.argmax(axis=1)
    # WAIT is an economic abstention, not an indicator rule or imposed chart pattern.
    chosen=np.where(ev[np.arange(len(ev)),best]>0,best,-1)
    return chosen,ev


def counts(values):
    keys,n=np.unique(values,return_counts=True)
    return {str(k):int(v) for k,v in zip(keys,n)}


def trading_stats(chosen,statuses,cfg):
    trade=np.flatnonzero(chosen>=0); st=statuses[trade,chosen[trade]]
    win=int((st=='TP').sum()); loss=int((st=='SL').sum()); amb=int((st=='AMBIGUOUS').sum()); n=win+loss+amb
    return {'windows':len(chosen),'signals':len(trade),'wait':int((chosen<0).sum()),'BUY':int((chosen==0).sum()),'SELL':int((chosen==1).sum()),
            'outcomes':counts(st),'resolved_conservative':n,'win_rate_conservative':win/n if n else None,
            'gross_usd_per_oz_sum':win*cfg.tp-(loss+amb)*cfg.sl,
            'net_usd_per_oz_sum_after_fixed_cost':win*cfg.tp-(loss+amb)*cfg.sl-n*cfg.cost,
            'excluded_unresolved':len(trade)-n,'independent_overlapping_paper_signals':True}


def metrics(p,y,st,cfg):
    mask=np.isfinite(y); q=np.clip(p[mask].astype(float),1e-7,1-1e-7); target=y[mask]
    loss=float(np.mean(-target*np.log(q)-(1-target)*np.log1p(-q)))
    return {'masked_log_loss':loss,'known_labels':int(mask.sum()),'possible_labels':int(mask.size),
            'calibration_brier':float(np.mean((q-target)**2)),
            'per_side':{side:{'label_counts':counts(st[:,i]),'observed_tp_fraction':float(np.nanmean(y[:,i])) if mask[:,i].any() else None,
                              'mean_model_score':float(p[:,i].mean())} for i,side in enumerate(SIDES)},
            'strategy':trading_stats(actions(p,cfg)[0],st,cfg)}


def splits(n,cfg):
    # Boundaries in raw candles, not shuffled samples: BOTH lookback and future label remain within block.
    a,b=int(n*.70),int(n*.85)
    t=np.arange(WARMUP+cfg.lookback-1,a-cfg.horizon,dtype=np.int64)
    v=np.arange(a+cfg.lookback-1,b-cfg.horizon,dtype=np.int64)
    hold=np.arange(b+cfg.lookback-1,n-cfg.horizon,dtype=np.int64)
    if min(len(t),len(v),len(hold))<20: raise ValueError('Not enough M15 candles for chronological splits and requested history/horizon')
    return t,v,hold


def save_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def train_decision(csv,m1_csv,out,cfg=None):
    cfg=cfg or DecisionSettings(); cfg.validate()
    raw=load_candles(csv,15); m1=load_candles(m1_csv,1); f=features(raw)
    tr,va,hold=splits(len(raw),cfg)
    # Require M1 chronological coverage for training/validation; local missing minutes remain flagged.
    if int(m1.time.iloc[0])>int(raw.time.iloc[tr[0]+1]) or int(m1.time.iloc[-1])<int(raw.time.iloc[va[-1]+cfg.horizon])+840:
        raise ValueError('M1 must cover the training AND validation future-label period; export the full matching history')
    engine=Outcomes(raw,m1,cfg); yt,st=engine.labels(tr); yv,sv=engine.labels(va)
    for label,y in [('train',yt),('validation',yv)]:
        for k in range(2):
            if min(np.sum(y[:,k]==1),np.sum(y[:,k]==0))<10: raise ValueError(f'Insufficient known TP and SL labels in {label} for {SIDES[k]}')
    used=f[tr[0]-cfg.lookback+1:tr[-1]+1]; mu=used.mean(axis=0); sd=used.std(axis=0); sd=np.maximum(sd,1e-6)
    size=(cfg.lookback*len(FEATURES),*cfg.hidden,2); net=Net(size,cfg.seed); rng=np.random.default_rng(cfg.seed)
    best=float('inf'); best_state=None; best_epoch=0; misses=0; trace=[]
    for epoch in range(1,cfg.epochs+1):
        order=rng.permutation(len(tr))
        for i in range(0,len(order),cfg.batch_size):
            idx=order[i:i+cfg.batch_size]; net.train(windows(f,tr[idx],cfg.lookback,mu,sd),yt[idx],cfg.lr)
        pv=probabilities(net,f,va,cfg,mu,sd); m=metrics(pv,yv,sv,cfg); loss=m['masked_log_loss']
        trace.append({'epoch':epoch,'validation_log_loss':loss,'validation_strategy':m['strategy']})
        print(f'[v1.0] epoch={epoch}/{cfg.epochs} val_log_loss={loss:.6f} signals={m["strategy"]["signals"]} win_rate={m["strategy"]["win_rate_conservative"]}',file=sys.stderr,flush=True)
        if loss<best-1e-6: best=loss; best_state=net.state(); best_epoch=epoch; misses=0
        else: misses+=1
        if misses>=cfg.patience: break
    if best_state is None: raise ValueError('Training did not produce finite validation loss')
    net.restore(best_state)
    pv=probabilities(net,f,va,cfg,mu,sd); pt=probabilities(net,f,tr,cfg,mu,sd)
    base=np.broadcast_to(np.nanmean(yt,axis=0),pv.shape)
    report={'version':VERSION,'task':'independent_outcome_supervised_entry_decision','settings':asdict(cfg),'architecture':list(size),
            'selected_epoch':best_epoch,'epochs':trace,'train':metrics(pt,yt,st,cfg),'validation':metrics(pv,yv,sv,cfg),
            'validation_constant_training_prior_baseline':metrics(base,yv,sv,cfg),
            'validation_always_buy':trading_stats(np.zeros(len(va),int),sv,cfg),'validation_always_sell':trading_stats(np.ones(len(va),int),sv,cfg),
            'split_counts':{'train':len(tr),'validation':len(va),'reserved_holdout_windows_not_evaluated':len(hold)},
            'development_time_range':[int(raw.time.iloc[0]),int(raw.time.iloc[-1])],
            'note':'Model scores are uncalibrated conditional on known TP/SL labels; censoring can bias results. No holdout evaluated. Paper signal-close fills, fixed costs only.'}
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ('model.npz','config.json','training.json')): raise FileExistsError('Use a new output directory to preserve existing checkpoints')
    np.savez_compressed(root/'model.npz',**best_state,mean=mu,std=sd)
    save_json(root/'config.json',{'model_version':VERSION,'architecture':list(size),'settings':asdict(cfg),'features':list(FEATURES),'warmup':WARMUP,'development_time_range':report['development_time_range']})
    save_json(root/'training.json',report)
    return report


def checkpoint(path):
    root=Path(path); obj=json.loads((root/'config.json').read_text())
    if obj['model_version']!=VERSION or obj['features']!=list(FEATURES) or obj['warmup']!=WARMUP: raise ValueError('v1.0 checkpoint/schema mismatch')
    cfg=DecisionSettings(**obj['settings']); cfg.validate()
    size=(cfg.lookback*len(FEATURES),*cfg.hidden,2)
    if tuple(obj['architecture'])!=size: raise ValueError('Checkpoint dimensions mismatch')
    net=Net(size,cfg.seed)
    with np.load(root/'model.npz',allow_pickle=False) as data:
        mu=data['mean']; sd=data['std']; net.restore({k:data[k] for k in data.files if k not in ('mean','std')})
    if mu.shape!=(len(FEATURES),) or sd.shape!=mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any(): raise ValueError('Invalid scaler')
    return net,cfg,mu,sd,obj


def predict_decision(csv,model):
    net,cfg,mu,sd,_=checkpoint(model); raw=load_candles(csv,15); f=features(raw)
    if len(raw)<WARMUP+cfg.lookback: raise ValueError(f'Need at least {WARMUP+cfg.lookback} completed M15 candles')
    p=probabilities(net,f,np.array([len(raw)-1]),cfg,mu,sd); act,ev=actions(p,cfg); a=int(act[0]); entry=float(raw.close.iloc[-1])
    d=1 if a==0 else -1
    return {'version':VERSION,'decision':'WAIT' if a<0 else SIDES[a],'signal_candle_open_utc':pd.Timestamp(int(raw.time.iloc[-1]),unit='s',tz='UTC').isoformat(),
            'lookback':cfg.lookback,'scores_uncalibrated':dict(zip(SIDES,map(float,p[0]))),'expected_net_conditional':dict(zip(SIDES,map(float,ev[0]))),
            'entry_close':entry if a>=0 else None,'tp_price':entry+d*cfg.tp if a>=0 else None,'sl_price':entry-d*cfg.sl if a>=0 else None}


def backtest_decision(csv,m1_csv,model,out):
    net,cfg,mu,sd,meta=checkpoint(model); raw=load_candles(csv,15); f=features(raw)
    # Original development includes the reserved internal holdout. External test must be later.
    if int(raw.time.iloc[0])<=meta['development_time_range'][1]: raise ValueError('External replay must be chronologically after development; use a separate later M15 file')
    ends=np.arange(WARMUP+cfg.lookback-1,len(raw),dtype=np.int64)
    if not len(ends): raise ValueError('Not enough candles for requested lookback')
    p=probabilities(net,f,ends,cfg,mu,sd); chosen,ev=actions(p,cfg)
    m1=load_candles(m1_csv,1); engine=Outcomes(raw,m1,cfg)
    # Labels evaluated only AFTER predictions have been computed; never used to change a decision.
    _,st=engine.labels(ends)
    rows=[]
    for k,i in enumerate(ends):
        a=int(chosen[k]); e=float(raw.close.iloc[i]); d=1 if a==0 else -1
        result,j,minute=engine.one(int(i),d) if a>=0 else ('WAIT',None,None)
        move=cfg.tp-cfg.cost if result=='TP' else -cfg.sl-cfg.cost if result in ('SL','AMBIGUOUS') else None
        stamp=lambda t: pd.Timestamp(int(t),unit='s',tz='UTC').isoformat()
        rows.append({'signal_candle_open_utc':stamp(raw.time.iloc[i]),'signal_close_utc':stamp(raw.time.iloc[i]+900),
                     'decision':'WAIT' if a<0 else SIDES[a],'buy_score':float(p[k,0]),'sell_score':float(p[k,1]),
                     'buy_expected_net_conditional':float(ev[k,0]),'sell_expected_net_conditional':float(ev[k,1]),
                     'entry_close':e if a>=0 else None,'tp_price':e+d*cfg.tp if a>=0 else None,'sl_price':e-d*cfg.sl if a>=0 else None,
                     'result':result,'exit_m15_open_utc':stamp(raw.time.iloc[j]) if j is not None else None,
                     'exit_m1_open_utc':stamp(minute) if minute is not None else None,'bars_to_touch':j-i if j is not None else None,
                     'net_usd_per_oz_conservative':move})
    path=Path(out)
    if path.suffix.lower()!='.csv': raise ValueError('Output must end in .csv')
    path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
    summary={'version':VERSION,'settings':asdict(cfg),'overall':trading_stats(chosen,st,cfg),
             'by_side':{side:trading_stats(np.where(chosen==k,k,-1),st,cfg) for k,side in enumerate(SIDES)},
             'always_buy':trading_stats(np.zeros(len(ends),int),st,cfg),'always_sell':trading_stats(np.ones(len(ends),int),st,cfg),
             'assumptions':'Independent paper signals; hypothetical close fill; future High/Low first touch; ambiguous minute counted as loss; missing, timeout and pending excluded; no bid/ask, gap fill, lot sizing or portfolio P&L.'}
    save_json(path.with_suffix('.summary.json'),summary); return summary
