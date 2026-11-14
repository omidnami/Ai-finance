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


def _m1_for_ambiguous_candle(tmp_path, raw, *, m15_index=110, sequence='tp_first', missing_minute=None, discrepancy=False):
    """One complete 15-minute segment whose aggregate matches M15; 15 records."""
    frame = pd.read_csv(raw)
    start = int(frame.loc[m15_index, 'time'])
    rows = [{'time': start + 60*j, 'open': 100.0, 'high': 100.1,
             'low': 99.9, 'close': 100.0, 'volume': 10, 'timeframe': 1}
            for j in range(15)]
    if sequence == 'tp_first':
        rows[1]['high'] = 102.6
        rows[3]['low'] = 97.0
        rows[5]['high'] = 103.0
    elif sequence == 'sl_first':
        rows[1]['low'] = 97.0
        rows[3]['high'] = 103.0
    elif sequence == 'both_first':
        rows[1]['high'] = 103.0
        rows[1]['low'] = 97.0
    else:
        raise AssertionError(sequence)
    if discrepancy:
        rows[8]['high'] = 104.0
    if missing_minute is not None:
        del rows[missing_minute]
    p = tmp_path/'m1.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    return p


def test_m1_refines_same_m15_bar_tp_before_sl_using_minute_wicks(tmp_path):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 97.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw, sequence='tp_first')
    res = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    r = res['rows'][0]
    assert r['entry_close_price'] == 100.
    assert r['tp_price'] == 102.5 and r['sl_price'] == 98.
    assert r['m15_result_before_m1'] == 'AMBIGUOUS_SL_FIRST'
    assert r['m1_resolution'] == 'M1_RESOLVED'
    assert r['result'] == 'TP'
    assert r['gross_price_move_usd_per_oz'] == 2.5
    assert r['m1_bars_in_ambiguous_candle'] == 15
    assert r['m1_first_touch_utc'] == pd.Timestamp(int(pd.read_csv(raw).loc[110,'time'])+60,unit='s',tz='UTC').isoformat()
    s = res['summary']['summary_by_selection']['all_matched_signals']['overall']
    assert s['m15_originally_ambiguous'] == 1 and s['m1_resolved'] == 1
    assert s['conservative_win_rate_resolved'] == 1.


def test_m1_refines_sl_first_for_sell_and_buy(tmp_path):
    for side, sequence, result in [('BUY','sl_first','SL'), ('SELL','tp_first','SL'),
                                   ('SELL','sl_first','TP')]:
        raw, sig = _inputs(tmp_path, [(109, side, 'accepted', 'QUEUE_'+side)],
                           {110: {'high': 103., 'low': 97.}})
        m1 = _m1_for_ambiguous_candle(tmp_path, raw, sequence=sequence)
        row = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)['rows'][0]
        assert row['result'] == result


def test_m1_still_ambiguous_if_both_levels_hit_within_one_minute(tmp_path):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 97.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw, sequence='both_first')
    out = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    assert out['rows'][0]['result'] == 'AMBIGUOUS_SL_FIRST'
    assert out['rows'][0]['m1_resolution'] == 'M1_STILL_AMBIGUOUS'
    s = out['summary']['summary_by_selection']['all_matched_signals']['overall']
    assert s['conservative_win_rate_resolved'] == 0.0
    assert s['m1_resolved'] == 0


@pytest.mark.parametrize('missing,discrepancy,expected', [(5,False,'M1_DATA_INCOMPLETE'),
                                                         (None,True,'M1_PRICE_MISMATCH')])
def test_m1_missing_or_price_mismatch_excluded_from_win_rate(tmp_path, missing, discrepancy, expected):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 97.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw, missing_minute=missing, discrepancy=discrepancy)
    out = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    assert out['rows'][0]['result'] == expected
    assert out['rows'][0]['gross_price_move_usd_per_oz'] is None
    s = out['summary']['summary_by_selection']['all_matched_signals']['overall']
    assert s['m1_unresolved_missing_or_mismatch'] == 1
    assert s['resolved'] == 0 and s['conservative_win_rate_resolved'] is None


def test_m15_unambiguous_does_not_require_minute_resolution(tmp_path):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 99.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw)  # aggregated lows differ from M15; unused
    out = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    assert out['rows'][0]['result'] == 'TP'
    assert out['rows'][0]['m1_resolution'] == 'M15_UNAMBIGUOUS_OR_NOT_HIT'
    assert out['summary']['summary_by_selection']['all_matched_signals']['overall']['m15_originally_ambiguous'] == 0


def test_invalid_m1_timeframe_and_duplicate_time_rejected(tmp_path):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 97.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw)
    data = pd.read_csv(m1)
    data.loc[0,'timeframe'] = 15
    data.to_csv(m1,index=False)
    with pytest.raises(ValueError, match='timeframe'):
        run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    data.loc[0,'timeframe'] = 1
    data.loc[1,'time'] = data.loc[0,'time']
    data.to_csv(m1,index=False)
    with pytest.raises(ValueError, match='unique'):
        run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)


def test_m1_confirmation_candle_wicks_not_used(tmp_path):
    raw, sig = _inputs(tmp_path, [(109, 'BUY', 'accepted', 'QUEUE_BUY')],
                       {110: {'high': 103., 'low': 97.}})
    m1 = _m1_for_ambiguous_candle(tmp_path, raw, sequence='sl_first')
    # Include a huge wick DURING the signal M15 candle; must not count as hit.
    before = int(pd.read_csv(raw).loc[109,'time'])
    extraneous = {'time':before+14*60, 'open':100, 'high':110, 'low':90,
                  'close':100, 'volume':10, 'timeframe':1}
    data = pd.concat([pd.DataFrame([extraneous]), pd.read_csv(m1)],ignore_index=True)
    data.to_csv(m1,index=False)
    out = run_fixed_tpsl(str(raw), str(sig), str(tmp_path/'res.csv'), m1_csv=str(m1), echo=False)
    assert out['rows'][0]['result'] == 'SL'
