"""v0.9 full-history chart: visual-only, exact timestamps, no TP/SL assumptions."""
import numpy as np
import pandas as pd
import pytest

from finance_ai.history_chart import plot_history


def sample(tmp_path, n=230):
    t = 1745362800 + np.arange(n) * 900
    cl = 3400 + np.cumsum(np.sin(np.arange(n) / 7) * 2 + 0.05)
    op = np.r_[cl[0], cl[:-1]]
    frame = pd.DataFrame({"time":t,"open":op,"high":np.maximum(op,cl)+1,
                          "low":np.minimum(op,cl)-1,"close":cl,
                          "volume":np.full(n,100),"timeframe":np.full(n,15)})
    raw = tmp_path / 'candles.csv'
    frame.to_csv(raw,index=False)
    datetimes = pd.to_datetime(t,unit='s',utc=True).map(lambda x:x.isoformat())
    rev = pd.DataFrame({"window_end_utc":datetimes[100:],
        "model_class":["bullish_reversal" if j == 20 else "bearish_reversal" if j == 50 else "none" for j in range(n-100)],
        "price_reference_class":["bullish_reversal" if j == 20 else "bearish_reversal" if j == 60 else "none" for j in range(n-100)]})
    rev_file = tmp_path/'rev.csv';rev.to_csv(rev_file,index=False)
    setup = pd.DataFrame({"candle_utc":datetimes[110:],
        "decision":["QUEUE_BUY" if j == 10 else "QUEUE_SELL" if j == 40 else "none" for j in range(n-110)]})
    setup_file = tmp_path/'setup.csv';setup.to_csv(setup_file,index=False)
    return raw, rev_file, setup_file


def test_full_html_renders_every_candle_and_separate_overlays(tmp_path):
    pytest.importorskip('plotly')
    raw,rev,setup = sample(tmp_path)
    out=tmp_path/'full.html'
    result=plot_history(str(raw),str(rev),str(out),setups=str(setup),initial_candles=80)
    assert result['candles']==230
    assert result['reversal_report_rows']==130
    assert result['ai_bullish']==1 and result['ai_bearish']==1
    assert result['reference_bullish']==1 and result['reference_bearish']==1
    assert result['agreed_bullish']==1 and result['agreed_bearish']==0
    assert result['proposed_v09_setups']==2
    html=out.read_text()
    assert len(html)>100_000
    assert 'Plotly.newPlot' in html
    assert 'XAUUSD M15' in html
    assert 'Agreed BUY candidate' in html
    assert 'v0.9 proposed SELL setup' in html
    assert 'Not a filled trade' in html
    assert 'No TP' in html and 'SL' in html
    assert '2025-04-25' in html


def test_report_cannot_be_mapped_to_another_dataset(tmp_path):
    pytest.importorskip('plotly')
    raw,rev,setup = sample(tmp_path)
    df=pd.read_csv(rev)
    df.loc[0,'window_end_utc']='2020-01-01T00:00:00+00:00'
    df.to_csv(rev,index=False)
    with pytest.raises(ValueError,match='same dataset'):
        plot_history(str(raw),str(rev),str(tmp_path/'bad.html'),setups=str(setup))


def test_optional_v09_no_trade_assumptions_and_input_validations(tmp_path):
    pytest.importorskip('plotly')
    raw,rev,setup = sample(tmp_path)
    chart=tmp_path/'only.html'
    result=plot_history(str(raw),str(rev),str(chart))
    assert result['candles']==230 and result['proposed_v09_setups']==0
    with pytest.raises(ValueError, match='Output must be'):
        plot_history(str(raw),str(rev),str(tmp_path/'wrong.png'))
    with pytest.raises(ValueError, match='initial_candles'):
        plot_history(str(raw),str(rev),str(chart),initial_candles=0)
