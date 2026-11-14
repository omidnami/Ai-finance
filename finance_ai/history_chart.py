"""Offline interactive OHLC chart using existing v0.8/v0.9 research reports.

This is a visualization, not a backtest. No weights, thresholds, signals,
stop losses, profit targets, or trade results are calculated here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import read_candles

BULLISH = "bullish_reversal"
BEARISH = "bearish_reversal"


def _load_report(path: str | Path, timestamp_column: str, expected: pd.DatetimeIndex) -> pd.DataFrame:
    report = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if timestamp_column not in report:
        raise ValueError(f"{path} is missing {timestamp_column!r}")
    if report.empty:
        raise ValueError(f"{path} contains no report rows")
    time = pd.to_datetime(report[timestamp_column], utc=True, errors="coerce")
    if time.isna().any() or time.duplicated().any():
        raise ValueError(f"{path} has invalid or duplicated candle timestamps")
    if not time.isin(expected).all():
        raise ValueError(f"{path} has timestamps absent from the OHLC file; check that both use the same dataset")
    report.index = pd.DatetimeIndex(time)
    return report.sort_index()



def plot_history(csv: str, reversals: str, out: str,
                 setups: str | None = None, initial_candles: int = 280) -> dict:
    """Plot ALL raw candles with optional v0.9 overlay, initial viewport zoomed.

    v0.8 markers are *the recorded inference outputs* from an existing
    walk-forward report; v0.9 markers are only accepted research setups.
    No reference labels are recomputed from future candles.
    """
    if not isinstance(initial_candles, int) or initial_candles < 40:
        raise ValueError("initial_candles must be >= 40")
    target = Path(out)
    if target.suffix.lower() != ".html":
        raise ValueError("Output must be a .html file")
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install interactive charts: python -m pip install -e '.[interactive]'") from exc

    raw = read_candles(csv)
    x = pd.DatetimeIndex(pd.to_datetime(raw["time"], unit="s", utc=True))
    n = len(raw)
    highs = raw["high"].to_numpy(dtype=float)
    lows = raw["low"].to_numpy(dtype=float)
    closes = raw["close"].to_numpy(dtype=float)
    # Consistent visual spacing only; not used by any classifier.
    ranges = highs - lows
    pad = max(float(np.median(ranges) * 0.32), float(np.median(closes) * 0.00008))

    rev = _load_report(reversals, "window_end_utc", x)
    required = {"model_class", "price_reference_class"}
    if not required.issubset(rev):
        raise ValueError(f"v0.8 report must include {sorted(required)}")
    valid_classes = {"none", BULLISH, BEARISH}
    for col in required:
        if not set(rev[col].dropna()).issubset(valid_classes):
            raise ValueError(f"Unknown reversal class in {col}")

    mapped = rev.reindex(x)
    ai = mapped["model_class"].fillna("none")
    reference = mapped["price_reference_class"].fillna("none")
    agreement = (ai == reference) & (ai != "none")
    index_array = np.arange(n)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=raw["open"], high=raw["high"], low=raw["low"], close=raw["close"],
        increasing_line_color="#16a34a", decreasing_line_color="#ef4444",
        increasing_fillcolor="#16a34a", decreasing_fillcolor="#ef4444",
        name="XAUUSD M15", showlegend=True,
    ))

    def mark(mask: np.ndarray, name: str, symbol: str, color: str, above: bool, size: int,
             showlegend: bool = True, text_extra: str = ""):
        idx = index_array[mask]
        if len(idx) == 0:
            return
        position = (highs[idx] + pad) if above else (lows[idx] - pad)
        fig.add_trace(go.Scatter(
            x=x[idx], y=position, mode="markers", name=f"{name} ({len(idx)})",
            showlegend=showlegend,
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.2, color="#f8fafc")),
            customdata=np.array([x[k].strftime("%Y-%m-%d %H:%M UTC") for k in idx]),
            hovertemplate="<b>" + name + "</b><br>%{customdata}<br>Price: %{y:.2f}" + text_extra + "<extra></extra>",
        ))

    bull_ai = (ai == BULLISH).to_numpy()
    bear_ai = (ai == BEARISH).to_numpy()
    bull_ref = (reference == BULLISH).to_numpy()
    bear_ref = (reference == BEARISH).to_numpy()
    mark(bull_ai, "AI bullish", "triangle-up", "#22c55e", False, 10)
    mark(bear_ai, "AI bearish", "triangle-down", "#f43f5e", True, 10)
    # Open circle is positioned a little further outside the wick than the AI triangle.
    # It is NOT proof of an independent confirmation: this is v0.8's own training proxy.
    def ref_mark(mask, name, color, above):
        idx = index_array[mask]
        if not len(idx):
            return
        pos = highs[idx] + pad * 2.05 if above else lows[idx] - pad * 2.05
        fig.add_trace(go.Scatter(x=x[idx], y=pos, mode="markers", name=f"{name} ({len(idx)})",
            marker=dict(symbol="circle-open", size=11, color=color, line=dict(width=2.2)),
            customdata=np.array([x[k].strftime("%Y-%m-%d %H:%M UTC") for k in idx]),
            hovertemplate="<b>" + name + "</b><br>%{customdata}<extra></extra>"))
    ref_mark(bull_ref, "Price reference bullish", "#4ade80", False)
    ref_mark(bear_ref, "Price reference bearish", "#fb7185", True)
    mark((agreement & (ai == BULLISH)).to_numpy(), "Agreed BUY candidate", "star", "#facc15", False, 14)
    mark((agreement & (ai == BEARISH)).to_numpy(), "Agreed SELL candidate", "star", "#facc15", True, 14)

    setup_count = 0
    if setups is not None:
        setup = _load_report(setups, "candle_utc", x)
        if "decision" not in setup:
            raise ValueError("v0.9 report must include 'decision'")
        allowed = {"QUEUE_BUY", "QUEUE_SELL"}
        selected = setup[setup["decision"].isin(allowed)]
        setup_count = len(selected)
        # Setting up a stop-entry order is NOT a filled trade or validated entry.
        for side, color, above in (("BUY", "#60a5fa", False), ("SELL", "#c084fc", True)):
            stamps = selected.index[selected["decision"] == f"QUEUE_{side}"]
            idx = x.get_indexer(stamps)
            if not len(idx):
                continue
            pos = highs[idx] + 3.0 * pad if above else lows[idx] - 3.0 * pad
            fig.add_trace(go.Scatter(x=x[idx], y=pos, mode="markers", name=f"v0.9 proposed {side} setup ({len(idx)})",
                marker=dict(symbol="diamond", size=13, color=color, line=dict(width=1, color="#0f172a")),
                customdata=np.array([t.strftime("%Y-%m-%d %H:%M UTC") for t in stamps]),
                hovertemplate=f"v0.9 proposed {side} setup<br>%{{customdata}}<br>Not a filled trade<extra></extra>"))

    # Retain all candles in the figure; zoom only the initial viewport.
    visible = min(initial_candles, n)
    start = x[n - visible]
    finish = x[-1] + pd.Timedelta(minutes=30)
    fig.update_layout(
        template="plotly_dark", height=790, autosize=True,
        title=(f"AI Finance | XAUUSD M15 | {n:,} candles — v0.8 signals & v0.9 setups"),
        xaxis=dict(title="UTC time (zoom / pan / range slider)", type="date",
                   rangeslider=dict(visible=True, thickness=0.085), range=[start, finish],
                   showspikes=True),
        yaxis=dict(title="XAUUSD price", fixedrange=False, showspikes=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                    groupclick="toggleitem"),
        margin=dict(l=65, r=32, t=130, b=75),
        hovermode="closest", dragmode="pan",
        annotations=[dict(x=0, y=1.15, xref="paper", yref="paper", showarrow=False,
                          text="Triangles: AI · hollow circles: price-only reference · stars: agreement · diamonds: v0.9 proposed setups (NOT filled entries). No TP/SL.",
                          font=dict(size=11, color="#cbd5e1"), xanchor="left")],
    )
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1d", step="day", stepmode="backward"),
         dict(count=7, label="7d", step="day", stepmode="backward"),
         dict(count=1, label="1m", step="month", stepmode="backward"),
         dict(step="all", label="All")], bgcolor="#334155", y=1.0))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(target), include_plotlyjs=True, full_html=True, auto_open=False,
                   config={"scrollZoom": True, "displaylogo": False, "responsive": True})
    return {"chart": str(target), "candles": n, "reversal_report_rows": len(rev),
            "ai_bullish": int(bull_ai.sum()), "ai_bearish": int(bear_ai.sum()),
            "reference_bullish": int(bull_ref.sum()), "reference_bearish": int(bear_ref.sum()),
            "agreed_bullish": int((agreement & (ai == BULLISH)).sum()),
            "agreed_bearish": int((agreement & (ai == BEARISH)).sum()),
            "proposed_v09_setups": setup_count,
            "note": "Visual-only. No TP/SL, execution, outcomes or future predictions. Reports must match the raw candle dataset."}
