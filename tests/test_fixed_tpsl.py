"""Fixed TP/SL regression: confirmed candle CLOSE entry, subsequent HIGH/LOW hits."""
import json
from pathlib import Path

import pandas as pd
import pytest

from finance_ai.fixed_tpsl import run_fixed_tpsl


def _inputs(tmp_path, entries, overrides=None):
    n = 126
    epoch = 1_700_006_400  # aligned to 900 seconds
    epoch -= epoch % 900
    candles = pd.DataFrame({
        'time': [epoch + 900*i for i in range(n)],
        'open': [100.] * n, 'high': [100.5] * n, 'low': [99.5] * n,
        'close': [100.] * n, 'volume': [100.] * n, 'timeframe': [15] * n,
    })
    # Large shadows on the confirmation candle must never trigger exits.
    candles.loc[109, ['high','low']] = [110.,90.]
    for i, change in (overrides or {}).items():
        for key, value in change.items():
            candles.loc[i, key] = value
    raw = tmp_path / 'ohlc.csv'
    candles.to_csv(raw,index=False)
    rows=[]
    for index, side, reason, decision in entries:
        rows.append({'step':index-109,'candle_utc':pd.Timestamp(candles.loc[index,'time'],unit='s',tz='UTC').isoformat(),
                     'candidate': side, 'quality_reason':reason, 'quality_score':85,
                     'market_cycle':'trading_range','decision':decision})
    signal=tmp_path/'v09.csv';pd.DataFrame(rows).to_csv(signal,index=False)
    return raw, signal


def test_buy_tp_uses_future_wick_not_candle_close(tmp_path):
    raw,sig=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY')],
                    {110:{'high':102.51,'low':99.7,'close':100.0}})
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'results.csv'),echo=False)
    r=out['rows'][0]
    assert r['entry_close_price']==100.0
    assert r['tp_price']==102.5 and r['sl_price']==98.
    assert r['result']=='TP' and r['bars_checked']==1
    assert r['exit_index']==110 and r['gross_price_move_usd_per_oz']==2.5
    assert out['summary']['summary_by_selection']['all_matched_signals']['overall']['conservative_win_rate_resolved']==1.


def test_sell_tp_and_sl_use_opposite_wicks(tmp_path):
    raw,sig=_inputs(tmp_path,[(109,'SELL','accepted','QUEUE_SELL')],
                    {110:{'low':97.49,'high':101.0,'close':100.0}})
    result=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'x.csv'),echo=False)['rows'][0]
    assert result['result']=='TP' and result['tp_price']==97.5 and result['sl_price']==102.0
    raw,sig=_inputs(tmp_path,[(109,'SELL','accepted','QUEUE_SELL')],
                    {110:{'low':99.5,'high':102.01,'close':100.0}})
    result=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'x.csv'),echo=False)['rows'][0]
    assert result['result']=='SL' and result['gross_price_move_usd_per_oz']==-2.


def test_both_wicks_ambiguous_marked_loss_separately(tmp_path):
    raw,sig=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY')],
                    {110:{'low':97.5,'high':103.0}})
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'ambig.csv'),echo=False)
    assert out['rows'][0]['result']=='AMBIGUOUS_SL_FIRST'
    stats=out['summary']['summary_by_selection']['all_matched_signals']['overall']
    assert stats['tp_hits']==0 and stats['sl_hits']==0
    assert stats['ambiguous_both_hit_same_bar_counted_as_loss']==1
    assert stats['conservative_win_rate_resolved']==0.


def test_no_hit_and_pending_cannot_be_falsely_won(tmp_path):
    raw,sig=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY')])
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'limited.csv'),max_holding_bars=2,echo=False)
    assert out['rows'][0]['result']=='NO_HIT' and out['rows'][0]['bars_checked']==2
    assert out['summary']['summary_by_selection']['all_matched_signals']['overall']['conservative_win_rate_resolved'] is None
    # No arbitrary horizon: if TP/SL unhit through end, remains pending.
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'unlimited.csv'),echo=False)
    assert out['rows'][0]['result']=='PENDING_END'


def test_qualified_queued_breakdown_and_overlap(tmp_path):
    # Contiguous report rows; two adjacent BUYs with independent futures.
    raw,sig=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY'),(110,'SELL','range_middle','SKIP_RANGE_MIDDLE')],
                    {111:{'low':97.0,'high':100.5}})
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'stat.csv'),echo=False)
    assert len(out['rows'])==2
    assert out['rows'][0]['result']=='SL'
    assert out['rows'][1]['result']=='TP'
    summaries=out['summary']['summary_by_selection']
    assert summaries['all_matched_signals']['overall']['signals']==2
    assert summaries['qualified_setups']['overall']['signals']==1
    assert summaries['queued_v09_setups']['overall']['signals']==1
    assert out['rows'][1]['entry_close_price']==100.0
    assert (tmp_path/'stat.summary.json').exists()
    assert list(pd.read_csv(tmp_path/'stat.csv')['result'])==['SL','TP']


def test_bad_time_close_v08_or_inputs_rejected(tmp_path):
    raw,sig=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY')])
    with pytest.raises(ValueError,match='positive'):
        run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'invalid.csv'),tp=-2.,echo=False)
    with pytest.raises(ValueError,match='max_holding_bars'):
        run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'invalid.csv'),max_holding_bars=0,echo=False)
    df=pd.read_csv(sig)
    df['candle_utc']='2020-01-01T00:00:00+00:00'
    df.to_csv(sig,index=False)
    with pytest.raises(ValueError,match='same dataset'):
        run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'invalid.csv'),echo=False)


def test_v08_close_verification_and_horizon_lookahead(tmp_path):
    raw,_=_inputs(tmp_path,[(109,'BUY','accepted','QUEUE_BUY')],
                  {110:{'high':103.0}})
    epoch=int(pd.read_csv(raw).loc[109,'time'])
    rows=[{'window_end_utc':pd.Timestamp(epoch,unit='s',tz='UTC').isoformat(),
           'signal_candle_close':100.,'new_signal':'BUY'}]
    sig=tmp_path/'v08.csv';pd.DataFrame(rows).to_csv(sig,index=False)
    out=run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'v08-res.csv'),echo=False)
    assert out['rows'][0]['result']=='TP' and out['summary']['source']=='v0.8'
    rows[0]['signal_candle_close']=101.
    pd.DataFrame(rows).to_csv(sig,index=False)
    with pytest.raises(ValueError,match='same dataset'):
        run_fixed_tpsl(str(raw),str(sig),str(tmp_path/'v08-res.csv'),echo=False)
