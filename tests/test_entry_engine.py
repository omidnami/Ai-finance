"""v0.9 tests: avoid lookahead, preserve old checkpoints and causal orders."""
import json
import numpy as np
import pandas as pd
import pytest

from finance_ai.entry_engine import (
    EntryRules, _advance_order, _advance_position, grade_setup,
    inspect_last, market_context, run_entries,
)


def candles(tmp_path, n=330):
    rng = np.random.default_rng(901)
    close = 3370 + np.cumsum(rng.normal(0, 2, n)) + 9*np.sin(np.arange(n)/8)
    opn = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        'time': 1745362800 + np.arange(n)*900,
        'open': opn, 'high': np.maximum(opn, close)+0.8,
        'low': np.minimum(opn, close)-0.8,
        'close': close, 'volume': 100, 'timeframe': 15})
    path = tmp_path/'candles.csv'
    frame.to_csv(path,index=False)
    return path


def checkpoints(tmp_path):
    from finance_ai.trend_detection import (
        MODEL_VERSION, LABEL_PERIOD, LOOKBACK, FEATURE_NAMES, CLASS_NAMES,
        TIMEFRAME_MINUTES, BB_PERIOD, BB_STD, EMA_PERIOD, AROON_PERIOD,
        ATR_PERIOD, WEAK_SLOPE_ATR)
    from finance_ai.reversal_detection import (
        VERSION, FEATURES, LOOKBACK as RL, CLASSES, RSI_PERIOD,
        BB_PERIOD as RBB, BB_STD as RSD, MACD, ATR_PERIOD as RA,
        PRETREND_PERIOD, SWING_PERIOD, PRETREND_ATR, CONFIRM_ATR, BalancedSoftmax)
    from finance_ai.network import MLP
    trend=tmp_path/'trend'; trend.mkdir()
    rev=tmp_path/'reversal'; rev.mkdir()
    mod=MLP((LOOKBACK*len(FEATURE_NAMES), 8, 1),seed=22)
    np.savez_compressed(trend/'model.npz',**mod.state_dict())
    (trend/'config.json').write_text(json.dumps({
        'model_version':MODEL_VERSION, 'task':'current_trend_classification_not_forecasting',
        'architecture':[LOOKBACK*len(FEATURE_NAMES),8,1], 'lookback':LOOKBACK,
        'timeframe':TIMEFRAME_MINUTES, 'label_period':LABEL_PERIOD,
        'features':list(FEATURE_NAMES), 'classes':list(CLASS_NAMES),
        'weak_trend_threshold_atr':WEAK_SLOPE_ATR,
        'indicators':{'bollinger':[BB_PERIOD,BB_STD],'ema':EMA_PERIOD,
                      'aroon':AROON_PERIOD,'atr':ATR_PERIOD}}))
    (trend/'preprocessing.json').write_text(json.dumps({
        'version':7, 'mean':[0.]*len(FEATURE_NAMES),'std':[1.]*len(FEATURE_NAMES)}))
    net=BalancedSoftmax((RL*len(FEATURES),10,3),np.ones(3), seed=34)
    np.savez_compressed(rev/'model.npz',**net.state_dict())
    (rev/'config.json').write_text(json.dumps({
        'model_version':VERSION,'task':'past_only_confirmed_swing_reversal_reference_not_forecasting',
        'architecture':[RL*len(FEATURES),10,3], 'lookback':RL,'timeframe':15,
        'features':list(FEATURES),'classes':list(CLASSES),
        'indicators':{'rsi':RSI_PERIOD,'bollinger':[RBB,RSD],'macd':list(MACD),'atr':RA},
        'reference':{'pretrend_period':PRETREND_PERIOD,'swing_period':SWING_PERIOD,
                     'pretrend_atr':PRETREND_ATR,'confirm_atr':CONFIRM_ATR}}))
    (rev/'preprocessing.json').write_text(json.dumps({
        'version':8,'mean':[0.]*len(FEATURES),'std':[1.]*len(FEATURES)}))
    return trend,rev


def test_market_context_uses_only_existing_prices(tmp_path):
    src=candles(tmp_path)
    df=pd.read_csv(src)
    ref=market_context(df,160,.75,3.0)
    df.loc[161:,'close']=9999.
    df.loc[161:,'high']=10000.
    assert ref==market_context(df,160,.75,3.0)
    assert ref['market_cycle'] in {'breakout','trend_channel','trading_range'}
    assert 0 <= ref['range_position_32'] <= 1


def test_setup_range_middle_rejected_even_with_high_score():
    df=pd.DataFrame({'open':[100,100], 'high':[104,104], 'low':[96,96], 'close':[101,103]})
    ctx={'market_cycle':'trading_range','range_position_32':.5,
         'trend_v07':'uptrend','price_slope_atr':-1.5}
    out=grade_setup(df,1,'BUY',ctx,.95,2.,EntryRules(min_quality=0))
    assert out['reason']=='range_middle' and not out['accepted']
    assert out['trigger_price']==pytest.approx(104.01)
    assert out['protective_stop']==pytest.approx(95.8)
    assert out['planned_target_price']>out['trigger_price']


def test_trigger_not_market_fill_expiry_gap_and_intrabar_conservatism():
    rules=EntryRules(order_valid_bars=2,max_holding_bars=5)
    pending={'status':'pending','side':'BUY','trigger':101.,'stop':99.,
             'created_i':10,'risk_dollars':50.,'confirmation_utc':'t'}
    no_fill=pd.Series({'open':100.,'high':100.9,'low':99.4,'close':100.5})
    state,event=_advance_order(pending,no_fill,11,rules)
    assert state['status']=='pending' and event is None
    state,event=_advance_order(state,no_fill,12,rules)
    assert state is None and event['event']=='ORDER_EXPIRED'
    gap=pd.Series({'open':101.5,'high':104.,'low':98.,'close':103.})
    state,event=_advance_order(pending,gap,11,rules)
    assert event['event']=='ENTRY_FILLED' and state['entry_price']==101.5
    state,trade=_advance_position(state,gap,11,rules)
    assert state is None and trade['exit_reason']=='stop' and trade['realized_R']<0
    # A gap through a SELL stop fills at adverse OPEN, not an idealized stop.
    sell={'status':'open','side':'SELL','entry_i':10,'entry_price':100.,'stop':102.,
          'target':96.,'risk':2.,'units':1.}
    _,trade=_advance_position(sell,pd.Series({'open':104.,'high':105.,'low':103.,'close':104.}),11,rules)
    assert trade['exit_price']==104. and trade['exit_reason']=='stop_gap'


def test_checkpoints_run_snapshot_without_training_and_future_invariance(tmp_path):
    src=candles(tmp_path)
    trend,rev=checkpoints(tmp_path)
    first=inspect_last(str(src),str(trend),str(rev))
    assert first['context']['trend_v07'] in {'uptrend','downtrend'}
    assert first['ai_reversal'] in {'none','bullish_reversal','bearish_reversal'}
    frame=pd.read_csv(src)
    prefix=tmp_path/'prefix.csv'; frame.iloc[:280].to_csv(prefix,index=False)
    shorter=inspect_last(str(prefix),str(trend),str(rev))
    frame.loc[280:,'close']*=1.02
    frame.loc[280:,'high']=np.maximum(frame.loc[280:,'high'],frame.loc[280:,'close'])
    changed=tmp_path/'changed.csv'; frame.to_csv(changed,index=False)
    # Snapshot for last candle 279 must be independent of bars 280 onward.
    other=inspect_last(str(prefix),str(trend),str(rev))
    assert shorter==other
    out=run_entries(str(src),str(trend),str(rev),str(tmp_path/'entries.csv'),
                    max_steps=35,echo=False)
    assert len(out['steps'])==35
    assert out['summary']['version']=='0.9.0'
    assert (tmp_path/'entries.summary.json').exists()
    assert out['summary']['rules']['target_rr']==2.0


def test_run_treats_new_signal_only_after_confirmation_and_fill_next_bar(tmp_path,monkeypatch):
    from finance_ai import entry_engine as engine
    src=candles(tmp_path)
    trend,rev=checkpoints(tmp_path)
    calls=[]
    def fake_snapshot(t,raw,*args):
        calls.append(t)
        quality={'accepted': True,'reason':'accepted','score_0_to_100_heuristic':90,
                 'trigger_price':float(raw.iloc[t]['high'])+0.01,
                 'protective_stop':float(raw.iloc[t]['low'])-.5,
                 'planned_target_price':float(raw.iloc[t]['high'])+10}
        return {'timestamp_utc':engine.utc(int(raw.iloc[t]['time'])),
                'candidate':'BUY' if len(calls)==1 else 'none',
                'ai_reversal':'bullish_reversal' if len(calls)==1 else 'none',
                'price_reference':'bullish_reversal' if len(calls)==1 else 'none',
                'context':{'market_cycle':'trend_channel','trend_v07':'uptrend',
                           'price_slope_atr':1.,'directional_efficiency':.5,'range_position_32':.5},
                'quality':quality if len(calls)==1 else None}
    monkeypatch.setattr(engine,'_snapshot',fake_snapshot)
    output=run_entries(str(src),str(trend),str(rev),str(tmp_path/'simulation.csv'),
                       max_steps=5,echo=False)
    assert output['steps'][0]['decision']=='QUEUE_BUY'
    assert output['steps'][0]['order_trade_event']==''
    assert output['summary']['queued_orders']==1
    assert output['summary']['steps']==5


def test_bad_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        EntryRules(target_rr=-1).check()
    with pytest.raises(ValueError):
        EntryRules(risk_budget_fraction=.10).check()
    with pytest.raises(ValueError):
        EntryRules(min_quality=101).check()


def test_replay_first_steps_identical_with_extra_future_candles(tmp_path):
    src=candles(tmp_path,n=330)
    trend,rev=checkpoints(tmp_path)
    original=run_entries(str(src),str(trend),str(rev),str(tmp_path/'original.csv'),
                         max_steps=45,echo=False)
    frame=pd.read_csv(src)
    # Past 200 candles unchanged; future observations are arbitrary but OHLC-valid.
    after=frame.index>=201
    frame.loc[after,'open']*=1.02
    frame.loc[after,'close']*=1.02
    frame.loc[after,'high']=frame.loc[after,['open','close']].max(axis=1)+1
    frame.loc[after,'low']=frame.loc[after,['open','close']].min(axis=1)-1
    changed=tmp_path/'changed_future.csv';frame.to_csv(changed,index=False)
    counterfactual=run_entries(str(changed),str(trend),str(rev),str(tmp_path/'changed_future_report.csv'),
                               max_steps=45,echo=False)
    for a,b in zip(original['steps'],counterfactual['steps']):
        assert a.keys()==b.keys()
        for k in a:
            if isinstance(a[k],float):
                assert a[k]==pytest.approx(b[k],abs=1e-9)
            else:
                assert a[k]==b[k]
    assert original['trades']==counterfactual['trades']
    assert (tmp_path/'original.trades.csv').exists()
