import streamlit as st
st.write("### 🚀 System Booting...")
""" 
╔══════════════════════════════════════════════════════════════╗
║        AI TRADING BOT  —  STREAMLIT DASHBOARD               ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
    pip install streamlit plotly yfinance pandas numpy scikit-learn xgboost ta

RUN:
    streamlit run dashboard.py

DEPLOY (free, public URL):
    1. Push this file + trading_bot_final.py to GitHub
    2. Go to share.streamlit.io → connect repo → deploy
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

# ── page config (must be first streamlit call) ───────────────
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── import bot functions ─────────────────────────────────────
# ── import bot functions ─────────────────────────────────────
import trading_bot_final as bot

# Connecting the dashboard to your Signal Edition logic
fetch = bot.fetch
add_indicators = bot.add_indicators
score_signals = bot.score_signals
dedup_signals = bot.dedup_signals
pair_trades = bot.pair_trades
run_ml = bot.run_ml


# ══════════════════════════════════════════════════════════════
#  DARK THEME CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #2a2a2a; }
    .metric-card {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
    .metric-value { font-size: 26px; font-weight: 600; white-space: nowrap; }
    .signal-buy  { color: #00e676; }
    .signal-sell { color: #ff1744; }
    .signal-hold { color: #ffeb3b; }
    .win  { color: #00e676; }
    .loss { color: #ff5252; }
    div[data-testid="stMetricValue"] { color: #e0e0e0; }
    .stSelectbox > div, .stSlider > div { color: #e0e0e0; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stDataFrame { background: #161616; }
    hr { border-color: #2a2a2a; }
    .stAlert { background: #1a1a1a; border-color: #2a2a2a; }
    [data-testid="stMetricLabel"] { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — user controls
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    symbol = st.text_input(
        "Symbol",
        value="BTC-USD",
        help="Stocks: AAPL, RELIANCE.NS, TSLA  |  Crypto: BTC-USD, ETH-USD, SOL-USD"
    ).upper().strip()

    period = st.selectbox(
        "Data Period",
        ["6mo", "1y", "2y", "5y"],
        index=2,
        help="How far back to fetch historical data"
    )

    interval = st.selectbox(
        "Candle Interval",
        ["1d", "1h"],
        index=0,
        help="1d = daily (recommended), 1h = hourly"
    )

    signal_strength = st.slider(
        "Signal Strength",
        min_value=1, max_value=7, value=4,
        help="Higher = fewer but stronger signals. Recommended: 3–5"
    )

    initial_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
    )

    ml_enabled = st.toggle("XGBoost ML Filter", value=True)

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#555;line-height:1.7'>
    <b style='color:#888'>Quick symbols:</b><br>
    BTC-USD · ETH-USD · SOL-USD<br>
    AAPL · TSLA · NVDA · MSFT<br>
    RELIANCE.NS · TCS.NS · INFY.NS<br>
    NIFTY50=F · ^GSPC · GC=F
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:8px'>
  <div style='font-size:32px'>📈</div>
  <div>
    <div style='font-size:24px;font-weight:600;color:#e0e0e0'>AI Trading Bot</div>
    <div style='font-size:13px;color:#555'>Multi-indicator signals · XGBoost ML · Paired trade analysis</div>
  </div>
</div>
<hr style='border-color:#1e1e1e;margin:0 0 24px'>
""", unsafe_allow_html=True)

# ── placeholder when not yet run ────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;color:#444'>
      <div style='font-size:48px;margin-bottom:16px'>📊</div>
      <div style='font-size:18px;margin-bottom:8px;color:#666'>Configure your settings and click Run Analysis</div>
      <div style='font-size:13px'>Supports stocks, crypto, indices, and commodities</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RUN THE BOT
# ══════════════════════════════════════════════════════════════
cfg = {
    "SYMBOL"          : symbol,
    "PERIOD"          : period,
    "INTERVAL"        : interval,
    "SIGNAL_STRENGTH" : signal_strength,
    "RSI_PERIOD"      : 14,
    "RSI_OB"          : 65,
    "RSI_OS"          : 35,
    "EMA_FAST"        : 20,
    "EMA_SLOW"        : 50,
    "EMA_TREND"       : 200,
    "MACD_FAST"       : 12,
    "MACD_SLOW"       : 26,
    "MACD_SIGNAL"     : 9,
    "BB_PERIOD"       : 20,
    "BB_STD"          : 2.0,
    "ATR_PERIOD"      : 14,
    "STOCH_K"         : 14,
    "STOCH_D"         : 3,
    "ADX_PERIOD"      : 14,
    "ML_ENABLED"      : ml_enabled,
    "ML_FORWARD_BARS" : 5,
    "ML_MIN_MOVE"     : 0.015,
    "INITIAL_CAPITAL" : initial_capital,
    "SHOW_CHART"      : False,
}

with st.spinner(f"Fetching {symbol} data and running analysis..."):
    try:
        # data
        import yfinance as yf
        raw = yf.download(symbol, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            st.error(f"No data found for **{symbol}**. Check the symbol and try again.")
            st.stop()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df_raw = raw[["Open","High","Low","Close","Volume"]].copy().dropna()
        for c in df_raw.columns: df_raw[c] = df_raw[c].astype(float)

        df  = add_indicators(df_raw, cfg)
        df  = score_signals(df, cfg)
        df  = dedup_signals(df)
        trades = pair_trades(df, initial_capital)
        imp_series, ml_pred, ml_conf = run_ml(df, cfg)

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

last      = df.iloc[-1]
sig       = last["Signal"]
price     = last["Close"]
sig_color = "#00e676" if sig=="BUY" else ("#ff1744" if sig=="SELL" else "#ffeb3b")
sig_class = "signal-buy" if sig=="BUY" else ("signal-sell" if sig=="SELL" else "signal-hold")

n_buy     = (df["Signal"] == "BUY").sum()
n_sell    = (df["Signal"] == "SELL").sum()


# ══════════════════════════════════════════════════════════════
#  TOP METRICS ROW
# ══════════════════════════════════════════════════════════════
def metric_html(label, value, color="#e0e0e0", sub=None):
    sub_html = f"<div style='font-size:11px;color:#555;margin-top:4px'>{sub}</div>" if sub else ""
    return f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{value}</div>
      {sub_html}
    </div>"""

wins     = trades[trades["Result"]=="WIN"]  if not trades.empty else pd.DataFrame()
losses   = trades[trades["Result"]=="LOSS"] if not trades.empty else pd.DataFrame()
win_rate = len(wins)/len(trades)*100 if len(trades) > 0 else 0
total_pnl= trades["PnL $"].sum()    if not trades.empty else 0
pf       = (wins["PnL $"].sum()/abs(losses["PnL $"].sum())
            if not losses.empty and losses["PnL $"].sum() != 0 else 0)

cols = st.columns(7)
cards = [
    ("Current Signal",   sig,                            sig_color),
    ("Price",            f"{price:,.2f}",                "#e0e0e0"),
    ("Score",            f"{last['Score']:.1f}",         "#00bcd4"),
    ("Total Trades",     str(len(trades)),                "#e0e0e0"),
    ("Win Rate",         f"{win_rate:.1f}%",
     "#00e676" if win_rate >= 50 else "#ff5252"),
    ("Total PnL",        f"${total_pnl:,.0f}",
     "#00e676" if total_pnl >= 0 else "#ff5252"),
    ("Profit Factor",    f"{pf:.2f}" if pf else "—",
     "#00e676" if pf >= 1 else "#ff9800"),
]
for col, (label, val, color) in zip(cols, cards):
    col.markdown(metric_html(label, val, color), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Price Chart",
    "💰  Trade Analysis",
    "🤖  ML Insights",
    "📊  Indicators",
    "📋  Trade Log",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — PRICE CHART
# ══════════════════════════════════════════════════════════════
with tab1:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=["", "RSI", "MACD", "Volume"],
    )

    # ── Candlestick ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff5252",
        name="Price", line=dict(width=0.8),
    ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────
    for col_name, color, name in [
        ("EMA_fast", "#00bcd4", f"EMA{cfg['EMA_FAST']}"),
        ("EMA_slow", "#ff9800", f"EMA{cfg['EMA_SLOW']}"),
        ("EMA_trend","#9c27b0", f"EMA{cfg['EMA_TREND']}"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name], name=name,
            line=dict(color=color, width=1.2),
            opacity=0.85,
        ), row=1, col=1)

    # ── Bollinger Bands ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"], name="BB Upper",
        line=dict(color="#4caf50", width=0.6, dash="dot"),
        opacity=0.5, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"], name="BB Lower",
        line=dict(color="#4caf50", width=0.6, dash="dot"),
        fill="tonexty", fillcolor="rgba(76,175,80,0.04)",
        opacity=0.5, showlegend=False,
    ), row=1, col=1)

    # ── BUY signals ──────────────────────────────────────────
    buys  = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="#00e676",
                    line=dict(color="#ffffff", width=1)),
        name=f"BUY ({len(buys)})",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="#ff1744",
                    line=dict(color="#ffffff", width=1)),
        name=f"SELL ({len(sells)})",
    ), row=1, col=1)

    # ── Trade lines ──────────────────────────────────────────
    if not trades.empty:
        for _, tr in trades.iterrows():
            try:
                color = "rgba(0,230,118,0.25)" if tr["Result"]=="WIN" else "rgba(255,82,82,0.25)"
                fig.add_shape(type="line",
                    x0=str(tr["Entry Date"]), x1=str(tr["Exit Date"]),
                    y0=tr["Entry Price"], y1=tr["Exit Price"],
                    line=dict(color=color, width=1),
                    row=1, col=1)
            except Exception:
                pass

    # ── RSI ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#e91e63", width=1.2),
        showlegend=False,
    ), row=2, col=1)
    for level, color in [(cfg["RSI_OB"], "rgba(255,87,34,0.4)"),
                          (cfg["RSI_OS"], "rgba(33,150,243,0.4)"),
                          (50, "rgba(100,100,100,0.3)")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      line_width=0.8, row=2, col=1)

    # ── MACD ─────────────────────────────────────────────────
    hist_colors = ["#00e676" if v >= 0 else "#ff5252"
                   for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_hist"],
        marker_color=hist_colors, name="MACD Hist",
        opacity=0.6, showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        line=dict(color="#00bcd4", width=1), name="MACD",
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_sig"],
        line=dict(color="#ff9800", width=1), name="Signal",
        showlegend=False,
    ), row=3, col=1)

    # ── Volume ────────────────────────────────────────────────
    vol_colors = ["#00e676" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else "#ff5252" for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="Volume",
        opacity=0.6, showlegend=False,
    ), row=4, col=1)

    fig.update_layout(
        height=780,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111111",
        font=dict(color="#888", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=20, t=30, b=20),
        title=dict(
            text=f"{symbol}  ·  {period}  ·  {interval}  ·  "
                 f"{df.index[0].date()} → {df.index[-1].date()}",
            font=dict(color="#e0e0e0", size=14),
            x=0.01,
        ),
    )
    for row in range(1, 5):
        fig.update_xaxes(gridcolor="#1e1e1e", showgrid=True,
                         zeroline=False, row=row, col=1)
        fig.update_yaxes(gridcolor="#1e1e1e", showgrid=True,
                         zeroline=False, row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # current signal box
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card' style='border-color:{sig_color};margin-top:8px'>
          <div class='metric-label'>Latest Signal — {df.index[-1].date()}</div>
          <div class='metric-value {sig_class}'>{sig}</div>
          <div style='font-size:12px;color:#555;margin-top:6px'>Score: {last["Score"]:.1f}  ·  RSI: {last["RSI"]:.1f}  ·  ADX: {last["ADX"]:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sl = price - last["ATR"] * 1.5
        tp = price + last["ATR"] * 3.0
        if sig == "BUY":
            st.markdown(f"""
            <div class='metric-card' style='margin-top:8px'>
              <div class='metric-label'>Suggested Levels</div>
              <div style='font-size:13px;margin-top:8px'>
                Entry &nbsp;<span style='color:#e0e0e0;font-weight:500'>{price:,.2f}</span><br>
                Stop Loss &nbsp;<span style='color:#ff5252;font-weight:500'>{sl:,.2f}</span><br>
                Take Profit &nbsp;<span style='color:#00e676;font-weight:500'>{tp:,.2f}</span><br>
                Risk:Reward &nbsp;<span style='color:#00bcd4;font-weight:500'>1 : 2.0</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card' style='margin-top:8px'>
              <div class='metric-label'>Suggested Levels</div>
              <div style='font-size:13px;color:#555;margin-top:16px'>No active BUY signal</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        regime_color = "#00e676" if last.get("Score",0) > 2 else ("#ff5252" if last.get("Score",0) < -2 else "#ffeb3b")
        st.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
          <div class='metric-label'>Indicator Snapshot</div>
          <div style='font-size:12px;margin-top:8px;line-height:2;color:#888'>
            MACD Hist &nbsp;<span style='color:#e0e0e0'>{last["MACD_hist"]:.2f}</span><br>
            Stoch K &nbsp;<span style='color:#e0e0e0'>{last["Stoch_K"]:.1f}</span><br>
            CCI &nbsp;<span style='color:#e0e0e0'>{last["CCI"]:.1f}</span><br>
            HV 20d &nbsp;<span style='color:#e0e0e0'>{last["HV_20"]:.1f}%</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — TRADE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if trades.empty:
        st.info("No completed trades found. Try lowering Signal Strength or increasing the period.")
    else:
        # ── summary stats row ─────────────────────────────────
        avg_win  = wins["PnL $"].mean()   if len(wins)   > 0 else 0
        avg_loss = losses["PnL $"].mean() if len(losses) > 0 else 0
        best     = trades["PnL %"].max()
        worst    = trades["PnL %"].min()
        avg_pnl  = trades["PnL %"].mean()

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        for col, label, val, color in [
            (r1, "Total Trades",   str(len(trades)),           "#e0e0e0"),
            (r2, "Win Rate",       f"{win_rate:.1f}%",
             "#00e676" if win_rate >= 50 else "#ff5252"),
            (r3, "Avg Win",        f"${avg_win:,.0f}",         "#00e676"),
            (r4, "Avg Loss",       f"${avg_loss:,.0f}",        "#ff5252"),
            (r5, "Best Trade",     f"+{best:.1f}%",            "#00e676"),
            (r6, "Worst Trade",    f"{worst:.1f}%",            "#ff5252"),
        ]:
            col.markdown(metric_html(label, val, color), unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── equity curve ─────────────────────────────────────
        cum_pnl   = trades["PnL $"].cumsum()
        equity    = initial_capital + cum_pnl

        fig2 = make_subplots(rows=2, cols=1,
                              shared_xaxes=False,
                              vertical_spacing=0.08,
                              subplot_titles=["Equity Curve", "PnL per Trade"])
        fig2.add_trace(go.Scatter(
            x=list(range(len(trades))),
            y=equity,
            fill="tozeroy",
            fillcolor="rgba(0,188,212,0.08)",
            line=dict(color="#00bcd4", width=2),
            name="Equity",
        ), row=1, col=1)
        fig2.add_hline(y=initial_capital, line_dash="dot",
                       line_color="rgba(255,255,255,0.2)",
                       line_width=1, row=1, col=1)

        # dots coloured by win/loss
        fig2.add_trace(go.Scatter(
            x=[i for i, r in enumerate(trades["Result"]) if r=="WIN"],
            y=[equity.iloc[i] for i, r in enumerate(trades["Result"]) if r=="WIN"],
            mode="markers",
            marker=dict(color="#00e676", size=7),
            name="Win", showlegend=True,
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=[i for i, r in enumerate(trades["Result"]) if r=="LOSS"],
            y=[equity.iloc[i] for i, r in enumerate(trades["Result"]) if r=="LOSS"],
            mode="markers",
            marker=dict(color="#ff5252", size=7),
            name="Loss", showlegend=True,
        ), row=1, col=1)

        bar_cols2 = ["#00e676" if r=="WIN" else "#ff5252"
                     for r in trades["Result"]]
        fig2.add_trace(go.Bar(
            x=list(range(len(trades))),
            y=trades["PnL $"],
            marker_color=bar_cols2,
            opacity=0.75,
            name="PnL",
            showlegend=False,
        ), row=2, col=1)
        fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)",
                       line_width=1, row=2, col=1)

        fig2.update_layout(
            height=520,
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#111111",
            font=dict(color="#888", size=11),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a", borderwidth=1),
            margin=dict(l=60, r=20, t=40, b=20),
        )
        for row in [1, 2]:
            fig2.update_xaxes(gridcolor="#1e1e1e", row=row, col=1)
            fig2.update_yaxes(gridcolor="#1e1e1e", row=row, col=1)

        st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
        #  🔥 UPGRADE 1: THE UNDERWATER CHART (DRAWDOWN)
        # =====================================================================
        running_max = np.maximum.accumulate(equity)
        drawdown = ((equity - running_max) / running_max) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=list(range(len(drawdown))),
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(255,82,82,0.15)",
            line=dict(color="#ff5252", width=1.5),
            name="Drawdown %"
        ))
        fig_dd.update_layout(
            height=180,
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#111111",
            font=dict(color="#888", size=11),
            margin=dict(l=60, r=20, t=30, b=10),
            title=dict(text="Underwater Chart (Drawdown from Peak)", font=dict(color="#e0e0e0", size=13)),
            yaxis=dict(title="Drawdown %", gridcolor="#1e1e1e", zeroline=True, zerolinecolor="#333333"),
            xaxis=dict(gridcolor="#1e1e1e", showticklabels=False)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # =====================================================================
        #  🔥 UPGRADE 2 & 3: HEATMAP & DURATION ANALYSIS (SIDE-BY-SIDE)
        # =====================================================================
        hm_col, dur_col = st.columns(2)

        with hm_col:
            # ── Monthly Returns Heatmap ──
            if "Exit Date" in trades.columns and not trades.empty:
                hm_df = trades.copy()
                hm_df["Exit Date"] = pd.to_datetime(hm_df["Exit Date"])
                hm_df["Year"] = hm_df["Exit Date"].dt.year
                hm_df["Month"] = hm_df["Exit Date"].dt.strftime('%b')
                
                # Pivot and order months
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pivot = hm_df.pivot_table(values='PnL %', index='Year', columns='Month', aggfunc='sum')
                pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns]).fillna(0)

                # Custom colorscale: Red for negative, Black for 0, Green for positive
                fig_hm = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale=[[0.0, "#ff5252"], [0.5, "#111111"], [1.0, "#00e676"]],
                    zmid=0,
                    text=[[f"{val:+.1f}%" if val != 0 else "" for val in row] for row in pivot.values],
                    texttemplate="%{text}",
                    textfont={"color": "#ffffff", "size": 11},
                    showscale=False
                ))
                fig_hm.update_layout(
                    height=300,
                    paper_bgcolor="#0d0d0d",
                    plot_bgcolor="#111111",
                    font=dict(color="#888", size=11),
                    margin=dict(l=40, r=20, t=40, b=20),
                    title=dict(text="Monthly Net Returns (%)", font=dict(color="#e0e0e0", size=13)),
                    yaxis=dict(autorange="reversed", type="category")
                )
                st.plotly_chart(fig_hm, use_container_width=True)

        with dur_col:
            # ── Duration vs PnL Scatter ──
            if "Entry Date" in trades.columns and "Exit Date" in trades.columns:
                dur_df = trades.copy()
                dur_df["Entry Date"] = pd.to_datetime(dur_df["Entry Date"])
                dur_df["Exit Date"] = pd.to_datetime(dur_df["Exit Date"])
                dur_df["Duration"] = (dur_df["Exit Date"] - dur_df["Entry Date"]).dt.days
                
                dur_colors = ["#00e676" if pnl >= 0 else "#ff5252" for pnl in dur_df["PnL %"]]
                
                fig_dur = go.Figure()
                fig_dur.add_trace(go.Scatter(
                    x=dur_df["Duration"],
                    y=dur_df["PnL %"],
                    mode="markers",
                    marker=dict(color=dur_colors, size=9, line=dict(color="#ffffff", width=0.5)),
                    text=[f"Trade #{i+1}" for i in range(len(dur_df))],
                    hovertemplate="%{text}<br>Days Held: %{x}<br>PnL: %{y:.2f}%<extra></extra>"
                ))
                fig_dur.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_dur.update_layout(
                    height=300,
                    paper_bgcolor="#0d0d0d",
                    plot_bgcolor="#111111",
                    font=dict(color="#888", size=11),
                    margin=dict(l=40, r=20, t=40, b=30),
                    title=dict(text="Holding Period vs. Profitability", font=dict(color="#e0e0e0", size=13)),
                    xaxis=dict(title="Days Held", gridcolor="#1e1e1e"),
                    yaxis=dict(title="PnL %", gridcolor="#1e1e1e")
                )
                st.plotly_chart(fig_dur, use_container_width=True)
                
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        # =====================================================================

        # ── win/loss pie ──────────────────────────────────────
        pc1, pc2 = st.columns(2)
        with pc1:
            pie = go.Figure(go.Pie(
                labels=["Wins", "Losses"],
                values=[len(wins), len(losses)],
                hole=0.55,
                marker=dict(colors=["#00e676", "#ff5252"],
                            line=dict(color="#0d0d0d", width=2)),
                textfont=dict(size=13),
            ))
            pie.update_layout(
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#0d0d0d",
                font=dict(color="#888"),
                showlegend=True,
                legend=dict(bgcolor="#1a1a1a"),
                margin=dict(l=10, r=10, t=40, b=10),
                title=dict(text="Win / Loss Breakdown",
                           font=dict(color="#e0e0e0", size=13)),
                annotations=[dict(
                    text=f"{win_rate:.0f}%<br>win rate",
                    x=0.5, y=0.5, font_size=15,
                    font_color="#e0e0e0",
                    showarrow=False,
                )],
            )
            st.plotly_chart(pie, use_container_width=True)

        with pc2:
            # PnL distribution
            hist_fig = px.histogram(
                trades, x="PnL %",
                nbins=20,
                color_discrete_sequence=["#00bcd4"],
                title="PnL % Distribution",
            )
            hist_fig.add_vline(x=0, line_dash="dot",
                               line_color="rgba(255,255,255,0.3)")
            hist_fig.add_vline(x=avg_pnl, line_dash="dash",
                               line_color="#ff9800",
                               annotation_text=f"avg {avg_pnl:.1f}%",
                               annotation_font_color="#ff9800")
            hist_fig.update_layout(
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#111111",
                font=dict(color="#888", size=11),
                title=dict(font=dict(color="#e0e0e0", size=13)),
                showlegend=False,
                margin=dict(l=40, r=20, t=40, b=20),
            )
            hist_fig.update_xaxes(gridcolor="#1e1e1e")
            hist_fig.update_yaxes(gridcolor="#1e1e1e")
            st.plotly_chart(hist_fig, use_container_width=True)

# =====================================================================
        #  🔥 UPGRADE 4 & 5: MONTE CARLO & DAY-OF-WEEK ANALYSIS
        # =====================================================================
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        mc_col, dow_col = st.columns(2)

        with dow_col:
            # ── Day of Week Edge ──
            if "Entry Date" in trades.columns and not trades.empty:
                dow_df = trades.copy()
                dow_df["Entry Date"] = pd.to_datetime(dow_df["Entry Date"])
                dow_df["DOW"] = dow_df["Entry Date"].dt.day_name()
                
                # Order days properly
                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                
                # Group and aggregate
                dow_stats = dow_df.groupby("DOW").agg(
                    Trades=("PnL %", "count"),
                    Win_Rate=("Result", lambda x: (x == "WIN").mean() * 100)
                ).reindex(dow_order).fillna(0)

                fig_dow = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Volume Bars
                fig_dow.add_trace(go.Bar(
                    x=dow_stats.index, y=dow_stats["Trades"],
                    name="Trades Taken", marker_color="rgba(100,100,100,0.2)",
                ), secondary_y=False)
                
                # Win Rate Line
                fig_dow.add_trace(go.Scatter(
                    x=dow_stats.index, y=dow_stats["Win_Rate"],
                    name="Win Rate %", mode="lines+markers",
                    line=dict(color="#9c27b0", width=2.5),
                    marker=dict(size=8, color="#ffffff", line=dict(color="#9c27b0", width=2))
                ), secondary_y=True)

                fig_dow.update_layout(
                    height=320, paper_bgcolor="#0d0d0d", plot_bgcolor="#111111",
                    font=dict(color="#888", size=11), margin=dict(l=40, r=40, t=40, b=20),
                    title=dict(text="Temporal Edge: Performance by Day of Week", font=dict(color="#e0e0e0", size=13)),
                    showlegend=False, hovermode="x unified"
                )
                fig_dow.update_xaxes(gridcolor="#1e1e1e")
                fig_dow.update_yaxes(title_text="Number of Trades", gridcolor="#1e1e1e", secondary_y=False, showgrid=False)
                fig_dow.update_yaxes(title_text="Win Rate (%)", secondary_y=True, range=[0, 105], showgrid=True, gridcolor="#1e1e1e")
                st.plotly_chart(fig_dow, use_container_width=True)

        with mc_col:
            # ── Monte Carlo Simulation (100 alternate paths) ──
            if not trades.empty and len(trades) > 3:
                simulations = 100
                steps = len(trades)
                # Convert % PnL to decimal multipliers (e.g. +5% -> 1.05)
                returns = trades["PnL %"].values / 100.0
                
                mc_paths = np.zeros((simulations, steps))
                for i in range(simulations):
                    # Randomly sample our historical trades with replacement
                    random_returns = np.random.choice(returns, size=steps, replace=True)
                    mc_paths[i] = initial_capital * np.cumprod(1 + random_returns)

                fig_mc = go.Figure()
                
                # Plot all faint alternate realities
                for i in range(simulations):
                    fig_mc.add_trace(go.Scatter(
                        y=mc_paths[i], mode="lines",
                        line=dict(color="rgba(0,188,212,0.04)", width=1),
                        showlegend=False, hoverinfo="skip"
                    ))
                
                # Plot Median Path
                median_path = np.median(mc_paths, axis=0)
                fig_mc.add_trace(go.Scatter(
                    y=median_path, mode="lines",
                    line=dict(color="#00e676", width=2),
                    name="Expected (Median)"
                ))
                
                # Plot 5th Percentile (Stress Test / Worst Case)
                p5_path = np.percentile(mc_paths, 5, axis=0)
                fig_mc.add_trace(go.Scatter(
                    y=p5_path, mode="lines",
                    line=dict(color="#ff5252", width=2, dash="dot"),
                    name="Stress Test (5th Pct)"
                ))

                fig_mc.update_layout(
                    height=320, paper_bgcolor="#0d0d0d", plot_bgcolor="#111111",
                    font=dict(color="#888", size=11), margin=dict(l=40, r=20, t=40, b=20),
                    title=dict(text="Monte Carlo Simulation (100 Alternate Realities)", font=dict(color="#e0e0e0", size=13)),
                    xaxis=dict(title="Trades Taken", gridcolor="#1e1e1e"),
                    yaxis=dict(title="Simulated Equity ($)", gridcolor="#1e1e1e"),
                    legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig_mc, use_container_width=True)
        # =====================================================================


# ══════════════════════════════════════════════════════════════
#  TAB 3 — ML INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    if not ml_enabled:
        st.info("Enable XGBoost ML Filter in the sidebar to see ML insights.")
    elif imp_series is None:
        st.warning("Not enough data to train ML model. Try a longer period.")
    else:
        # prediction card
        m1, m2, m3 = st.columns(3)
        pred_color = "#00e676" if ml_pred=="UP" else ("#ff5252" if ml_pred=="DOWN" else "#ffeb3b")
        with m1:
            st.markdown(metric_html(
                "ML Prediction (next 5 bars)",
                ml_pred,
                pred_color,
                sub=f"{ml_conf:.1%} confidence"
            ), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_html(
                "Model Type", "XGBoost", "#00bcd4",
                sub="Gradient boosting classifier"
            ), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_html(
                "Features Used", str(len(imp_series)), "#9c27b0",
                sub="Technical indicator features"
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("#### Feature Importance")
        st.markdown(
            "<div style='font-size:13px;color:#555;margin-bottom:16px'>"
            "How much each indicator contributed to XGBoost's predictions, "
            "measured by <b style='color:#888'>gain</b> — the improvement in accuracy "
            "when that feature is used to split the data."
            "</div>",
            unsafe_allow_html=True
        )

        top15 = imp_series.sort_values(ascending=False).head(15)
        rng   = top15.max() - top15.min()
        norm  = (top15 - top15.min()) / rng if rng > 0 else top15

        bar_colors_ml = []
        for v in norm.values:
            if   v >= 0.75: bar_colors_ml.append("#00e676")
            elif v >= 0.50: bar_colors_ml.append("#00bcd4")
            elif v >= 0.25: bar_colors_ml.append("#ff9800")
            else:           bar_colors_ml.append("#9c27b0")

        fig_ml = go.Figure(go.Bar(
            x=top15.values,
            y=top15.index,
            orientation="h",
            marker_color=bar_colors_ml,
            opacity=0.85,
            text=[f"{v:.2f}" for v in top15.values],
            textposition="outside",
            textfont=dict(color="#888", size=11),
        ))
        fig_ml.add_vline(
            x=float(top15.median()),
            line_dash="dash", line_color="#ff5722",
            line_width=1.2,
            annotation_text="median",
            annotation_font_color="#ff5722",
            annotation_position="top right",
        )
        fig_ml.update_layout(
            height=500,
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#111111",
            font=dict(color="#888", size=12),
            margin=dict(l=120, r=80, t=20, b=40),
            xaxis=dict(title="Importance (gain)", gridcolor="#1e1e1e", zeroline=False),
            yaxis=dict(autorange="reversed", gridcolor="#1e1e1e"),
            showlegend=False,
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # quartile legend
        lc1, lc2, lc3, lc4 = st.columns(4)
        for col, color, label in [
            (lc1, "#00e676", "Top 25% — strongest predictors"),
            (lc2, "#00bcd4", "2nd quartile"),
            (lc3, "#ff9800", "3rd quartile"),
            (lc4, "#9c27b0", "Bottom 25% — weakest"),
        ]:
            col.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;font-size:12px;color:#666'>"
                f"<div style='width:12px;height:12px;background:{color};border-radius:2px'></div>"
                f"{label}</div>",
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:12px;color:#444;padding:12px;background:#111;border-radius:8px;"
            "border:1px solid #1e1e1e'>"
            "<b style='color:#666'>Why XGBoost?</b> Your inputs are engineered tabular features "
            "(RSI, MACD, ATR...) not raw sequences. XGBoost consistently outperforms neural networks "
            "on tabular data, trains in seconds, and — unlike LSTM — shows exactly which features "
            "drive predictions through this importance chart."
            "</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
#  TAB 4 — INDICATORS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Live Indicator Dashboard")
    st.markdown(
        f"<div style='font-size:13px;color:#555;margin-bottom:20px'>"
        f"All values as of {df.index[-1].date()} · {symbol} @ {price:,.2f}"
        f"</div>", unsafe_allow_html=True
    )

    ind_data = {
        "RSI (14)"          : (last["RSI"],      0,   100,  35,   65,   "Momentum"),
        "Stochastic K"      : (last["Stoch_K"],  0,   100,  20,   80,   "Momentum"),
        "Williams %R"       : (last["WilliamsR"],-100, 0,  -80,  -20,  "Momentum"),
        "CCI (20)"          : (last["CCI"],      -200, 200,-100,  100,  "Trend"),
        "MFI (14)"          : (last["MFI"],       0,   100,  20,   80,  "Volume"),
        "ADX (14)"          : (last["ADX"],       0,    60,   0,   25,  "Trend"),
    }

    cols_ind = st.columns(3)
    for idx_i, (name, (val, lo, hi, warn_lo, warn_hi, cat)) in enumerate(ind_data.items()):
        col = cols_ind[idx_i % 3]
        pct = (val - lo) / (hi - lo) * 100 if (hi - lo) != 0 else 50
        pct = max(0, min(100, pct))
        if   val > warn_hi: bar_c = "#ff5252"
        elif val < warn_lo: bar_c = "#2196f3"
        else:               bar_c = "#00e676"
        col.markdown(f"""
        <div class='metric-card' style='margin-bottom:12px;text-align:left'>
          <div style='display:flex;justify-content:space-between;margin-bottom:8px'>
            <span style='font-size:12px;color:#666'>{name}</span>
            <span style='font-size:11px;color:#444;background:#1a1a1a;
                   padding:2px 6px;border-radius:4px'>{cat}</span>
          </div>
          <div style='font-size:22px;font-weight:600;color:{bar_c};margin-bottom:8px'>
            {val:.1f}
          </div>
          <div style='background:#1a1a1a;border-radius:4px;height:5px;overflow:hidden'>
            <div style='width:{pct:.0f}%;height:100%;background:{bar_c};
                   border-radius:4px;transition:width 0.3s'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Score Over Time")

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=df.index, y=df["Score"],
        fill="tozeroy",
        fillcolor="rgba(0,188,212,0.06)",
        line=dict(color="#00bcd4", width=1.2),
        name="Score",
    ))
    fig_score.add_hline(y=cfg["SIGNAL_STRENGTH"], line_dash="dash",
                        line_color="rgba(0,230,118,0.5)",
                        annotation_text=f"BUY threshold ({cfg['SIGNAL_STRENGTH']})",
                        annotation_font_color="#00e676")
    fig_score.add_hline(y=-cfg["SIGNAL_STRENGTH"], line_dash="dash",
                        line_color="rgba(255,82,82,0.5)",
                        annotation_text=f"SELL threshold (-{cfg['SIGNAL_STRENGTH']})",
                        annotation_font_color="#ff5252")
    fig_score.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=0.8)
    fig_score.update_layout(
        height=300,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111111",
        font=dict(color="#888", size=11),
        margin=dict(l=60, r=20, t=20, b=20),
        showlegend=False,
    )
    fig_score.update_xaxes(gridcolor="#1e1e1e")
    fig_score.update_yaxes(gridcolor="#1e1e1e")
    st.plotly_chart(fig_score, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — TRADE LOG
# ══════════════════════════════════════════════════════════════
with tab5:
    if trades.empty:
        st.info("No completed trades yet.")
    else:
        st.markdown(
            f"<div style='font-size:13px;color:#555;margin-bottom:16px'>"
            f"{len(trades)} completed trades  ·  "
            f"{len(wins)} wins  ·  {len(losses)} losses  ·  "
            f"Win rate {win_rate:.1f}%"
            f"</div>", unsafe_allow_html=True
        )

        def style_trades(df_t):
            def row_color(row):
                c = "color: #00e676" if row["Result"]=="WIN" else "color: #ff5252"
                return [c] * len(row)
            return df_t.style.apply(row_color, axis=1).format({
                "Entry Price": "{:,.2f}",
                "Exit Price":  "{:,.2f}",
                "PnL %":       "{:+.2f}%",
                "PnL $":       "${:,.0f}",
            })

        display_trades = trades.copy()
        display_trades.insert(0, "#", range(1, len(trades)+1))
        st.dataframe(
            style_trades(display_trades.drop(columns=["#"])),
            use_container_width=True,
            height=500,
        )

        # download button
        csv = trades.to_csv(index=False)
        st.download_button(
            label="⬇  Download Trade Log (CSV)",
            data=csv,
            file_name=f"trades_{symbol.replace('-','_')}_{period}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:11px;color:#333;padding:8px'>"
    f"AI Trading Bot  ·  {symbol}  ·  Data via Yahoo Finance  ·  "
    f"Not financial advice  ·  For educational purposes only"
    f"</div>",
    unsafe_allow_html=True
)
