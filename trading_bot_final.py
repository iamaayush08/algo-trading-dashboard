"""
╔══════════════════════════════════════════════════════════════╗
║         AI TRADING BOT  —  SIGNAL EDITION                   ║
║  Clean Signals · Paired Trades · XGBoost ML · Full Stats    ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
    pip install yfinance pandas numpy matplotlib scikit-learn xgboost ta

RUN:
    python trading_bot_final.py
"""

import warnings
warnings.filterwarnings("ignore")
import sys

# ── dependency check ──────────────────────────────────────────
for mod, pkg in [("yfinance","yfinance"),("pandas","pandas"),("numpy","numpy"),
                 ("matplotlib","matplotlib"),("sklearn","scikit-learn"),
                 ("xgboost","xgboost"),("ta","ta")]:
    try: __import__(mod)
    except ImportError:
        print(f"Missing: pip install {pkg}"); sys.exit(1)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates

import ta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ══════════════════════════════════════════════════════════════
#  CONFIG  ← edit here
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # Asset — stocks: "AAPL" "RELIANCE.NS"  crypto: "BTC-USD" "ETH-USD"
    "SYMBOL"          : "BTC-USD",
    "PERIOD"          : "2y",        # how far back to look
    "INTERVAL"        : "1d",        # 1d = daily candles

    # Signal sensitivity — higher = fewer signals (recommended: 3-5)
    "SIGNAL_STRENGTH" : 1,           # min score to trigger BUY or SELL

    # Indicators
    "RSI_PERIOD"      : 14,
    "RSI_OB"          : 65,          # overbought
    "RSI_OS"          : 35,          # oversold
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

    # ML
    "ML_ENABLED"      : True,
    "ML_FORWARD_BARS" : 5,           # how many bars ahead to predict
    "ML_MIN_MOVE"     : 0.015,       # 1.5% move = meaningful signal

    # Display
    "INITIAL_CAPITAL" : 100_000,     # starting cash for PnL calc
    "SHOW_CHART"      : True,
}


# ══════════════════════════════════════════════════════════════
#  1. FETCH DATA
# ══════════════════════════════════════════════════════════════
def fetch(cfg):
    print(f"\n  Downloading {cfg['SYMBOL']} ({cfg['PERIOD']}, {cfg['INTERVAL']})...")
    raw = yf.download(cfg["SYMBOL"], period=cfg["PERIOD"],
                      interval=cfg["INTERVAL"], auto_adjust=True, progress=False)
    if raw.empty:
        print("  ERROR: No data. Check symbol / internet."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open","High","Low","Close","Volume"]].copy().dropna()
    for c in df.columns: df[c] = df[c].astype(float)
    print(f"  {len(df)} candles  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════
#  2. INDICATORS
# ══════════════════════════════════════════════════════════════
def add_indicators(df, cfg):
    d = df.copy()

    # Trend
    d["EMA_fast"]  = ta.trend.EMAIndicator(d["Close"], cfg["EMA_FAST"]).ema_indicator()
    d["EMA_slow"]  = ta.trend.EMAIndicator(d["Close"], cfg["EMA_SLOW"]).ema_indicator()
    d["EMA_trend"] = ta.trend.EMAIndicator(d["Close"], cfg["EMA_TREND"]).ema_indicator()

    # MACD
    macd           = ta.trend.MACD(d["Close"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])
    d["MACD"]      = macd.macd()
    d["MACD_sig"]  = macd.macd_signal()
    d["MACD_hist"] = macd.macd_diff()

    # RSI
    d["RSI"]       = ta.momentum.RSIIndicator(d["Close"], cfg["RSI_PERIOD"]).rsi()

    # Bollinger Bands
    bb             = ta.volatility.BollingerBands(d["Close"], cfg["BB_PERIOD"], cfg["BB_STD"])
    d["BB_upper"]  = bb.bollinger_hband()
    d["BB_lower"]  = bb.bollinger_lband()
    d["BB_mid"]    = bb.bollinger_mavg()
    d["BB_pct"]    = bb.bollinger_pband()

    # ATR
    d["ATR"]       = ta.volatility.AverageTrueRange(
                         d["High"], d["Low"], d["Close"], cfg["ATR_PERIOD"]
                     ).average_true_range()

    # Stochastic
    stoch          = ta.momentum.StochasticOscillator(
                         d["High"], d["Low"], d["Close"], cfg["STOCH_K"], cfg["STOCH_D"])
    d["Stoch_K"]   = stoch.stoch()
    d["Stoch_D"]   = stoch.stoch_signal()

    # ADX
    adx            = ta.trend.ADXIndicator(d["High"], d["Low"], d["Close"], cfg["ADX_PERIOD"])
    d["ADX"]       = adx.adx()
    d["DI_pos"]    = adx.adx_pos()
    d["DI_neg"]    = adx.adx_neg()

    # Williams %R
    d["WilliamsR"] = ta.momentum.WilliamsRIndicator(
                         d["High"], d["Low"], d["Close"], 14).williams_r()

    # CCI
    d["CCI"]       = ta.trend.CCIIndicator(d["High"], d["Low"], d["Close"], 20).cci()

    # MFI
    d["MFI"]       = ta.volume.MFIIndicator(
                         d["High"], d["Low"], d["Close"], d["Volume"], 14).money_flow_index()

    # OBV
    d["OBV"]       = ta.volume.OnBalanceVolumeIndicator(d["Close"], d["Volume"]).on_balance_volume()

    # Historical volatility
    d["HV_20"]     = d["Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Price vs EMA distances (normalised by ATR)
    d["Dist_fast"] = (d["Close"] - d["EMA_fast"]) / d["ATR"].replace(0, np.nan)
    d["Dist_slow"] = (d["Close"] - d["EMA_slow"]) / d["ATR"].replace(0, np.nan)

    # Returns
    for n in [1, 3, 5]:
        d[f"Ret_{n}d"] = d["Close"].pct_change(n) * 100

    d.ffill().bfill()
    d.dropna(inplace=True)
    return d


# ══════════════════════════════════════════════════════════════
#  3. SIGNAL SCORING  (each condition = +1 bull / -1 bear)
# ══════════════════════════════════════════════════════════════
def score_signals(df, cfg):
    d  = df.copy()
    sc = pd.Series(0.0, index=d.index)

    # Trend (3 conditions)
    sc += np.where(d["Close"]    > d["EMA_trend"], 1, -1)
    sc += np.where(d["EMA_fast"] > d["EMA_slow"],  1, -1)
    sc += np.where(d["DI_pos"]   > d["DI_neg"],    1, -1)

    # MACD (2 conditions)
    sc += np.where(d["MACD_hist"] > 0, 1, -1)
    sc += np.where(d["MACD"]      > d["MACD_sig"], 1, -1)

    # RSI (2 conditions)
    sc += np.where(d["RSI"] > 50, 1, -1)
    sc += np.where(d["RSI"] < cfg["RSI_OB"], 1, -1)
    sc += np.where(d["RSI"] > cfg["RSI_OS"], 1, -1)

    # Bollinger (1 condition)
    sc += np.where(d["BB_pct"] < 0.4, 1, 0)
    sc += np.where(d["BB_pct"] > 0.6, -1, 0)

    # Stochastic (1 condition)
    sc += np.where(d["Stoch_K"] > d["Stoch_D"], 0.5, -0.5)
    sc += np.where(d["Stoch_K"] < 25, 1, 0)
    sc += np.where(d["Stoch_K"] > 75, -1, 0)

    # CCI
    sc += np.where(d["CCI"] > 0,  0.5, -0.5)

    # Williams %R
    sc += np.where(d["WilliamsR"] > -50, 0.5, -0.5)

    # MFI
    sc += np.where(d["MFI"] > 50, 0.5, -0.5)

    # ADX strength multiplier — only boost when trend is strong
    strong = d["ADX"] > 25
    sc     = sc * np.where(strong, 1.2, 0.85)

    d["Score"] = sc.round(2)

    thresh = cfg["SIGNAL_STRENGTH"]
    d["Raw_Signal"] = "HOLD"
    d.loc[sc >=  thresh, "Raw_Signal"] = "BUY"
    d.loc[sc <= -thresh, "Raw_Signal"] = "SELL"

    return d


# ══════════════════════════════════════════════════════════════
#  4. DEDUP SIGNALS  — no back-to-back same signals
#     BUY must be followed by SELL to form a trade.
#     This is what keeps the count sensible.
# ══════════════════════════════════════════════════════════════
def dedup_signals(df):
    d      = df.copy()
    result = ["HOLD"] * len(d)
    last   = "HOLD"

    for i, raw in enumerate(d["Raw_Signal"]):
        if raw == "BUY"  and last != "BUY":
            result[i] = "BUY";  last = "BUY"
        elif raw == "SELL" and last == "BUY":
            result[i] = "SELL"; last = "SELL"

    d["Signal"] = result
    return d


# ══════════════════════════════════════════════════════════════
#  5. TRADE PAIRING  — BUY in, next SELL out = one complete trade
# ══════════════════════════════════════════════════════════════
def pair_trades(df, initial_capital=100_000):
    buys  = df[df["Signal"] == "BUY"].copy()
    sells = df[df["Signal"] == "SELL"].copy()

    trades    = []
    sell_iter = iter(sells.iterrows())
    next_sell = next(sell_iter, None)

    for buy_date, buy_row in buys.iterrows():
        # advance sells until we find one after this buy
        while next_sell is not None and next_sell[0] <= buy_date:
            next_sell = next(sell_iter, None)
        if next_sell is None:
            break

        sell_date, sell_row = next_sell
        entry = buy_row["Close"]
        exit_ = sell_row["Close"]
        pnl_pct = (exit_ - entry) / entry * 100
        pnl_usd = (exit_ - entry) / entry * initial_capital

        trades.append({
            "Entry Date"  : buy_date.date(),
            "Exit Date"   : sell_date.date(),
            "Entry Price" : round(entry, 2),
            "Exit Price"  : round(exit_, 2),
            "PnL %"       : round(pnl_pct, 2),
            "PnL $"       : round(pnl_usd, 2),
            "Result"      : "WIN" if pnl_pct > 0 else "LOSS",
        })
        next_sell = next(sell_iter, None)

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════
#  6. XGBOOST ML — predicts next-5-bar direction
#     Uses NORMALISED feature importance so bars look different
# ══════════════════════════════════════════════════════════════
def run_ml(df, cfg):
    if not cfg["ML_ENABLED"]:
        return None, None, None

    print("\n  Training XGBoost ML model...")

    feat_cols = ["RSI", "MACD_hist", "BB_pct", "ATR", "Stoch_K",
                 "ADX", "WilliamsR", "CCI", "MFI", "HV_20",
                 "Dist_fast", "Dist_slow", "Ret_1d", "Ret_3d", "Score"]
    available = [c for c in feat_cols if c in df.columns]
    X_df      = df[available].copy()

    # Lag features so model sees recent history
    for col in ["RSI", "MACD_hist", "Score"]:
        if col in X_df.columns:
            X_df[f"{col}_lag1"] = X_df[col].shift(1)
            X_df[f"{col}_lag2"] = X_df[col].shift(2)

    X_df.dropna(inplace=True)

    # Labels: 0=DOWN, 1=FLAT, 2=UP  (XGBoost needs 0-indexed integers)
    fwd  = cfg["ML_FORWARD_BARS"]
    move = cfg["ML_MIN_MOVE"]
    fut  = df["Close"].pct_change(fwd).shift(-fwd).reindex(X_df.index)
    y    = np.where(fut >  move, 2,
           np.where(fut < -move, 0, 1))

    # Remove last fwd rows (no label)
    valid    = ~np.isnan(fut.values)
    X_clean  = X_df.values[valid]
    y_clean  = y[valid]
    feat_names = list(X_df.columns)

    if len(X_clean) < 100:
        print("  Not enough data for ML. Skipping.")
        return None, None, None

    # Time-series CV
    tscv   = TimeSeriesSplit(n_splits=4)
    cv_acc = []
    model  = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                           subsample=0.75, colsample_bytree=0.75,
                           eval_metric="mlogloss", random_state=42,
                           verbosity=0, n_jobs=-1)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", model)])

    for tr, te in tscv.split(X_clean):
        pipe.fit(X_clean[tr], y_clean[tr])
        cv_acc.append(accuracy_score(y_clean[te], pipe.predict(X_clean[te])))

    pipe.fit(X_clean, y_clean)
    mean_acc = np.mean(cv_acc)
    print(f"  XGBoost CV accuracy: {mean_acc:.1%}  ({len(X_clean)} samples)")

    # Feature importance — using gain (shows real differences)
    booster = pipe.named_steps["clf"].get_booster()
    scores  = booster.get_score(importance_type="gain")
    # Map back to feature names
    imp = {}
    for k, v in scores.items():
        # XGBoost uses f0, f1... internally; map by index
        try:
            idx = int(k[1:])
            if idx < len(feat_names):
                imp[feat_names[idx]] = v
        except Exception:
            pass

    if not imp:
        # fallback to sklearn importances
        raw_imp = pipe.named_steps["clf"].feature_importances_
        imp = dict(zip(feat_names, raw_imp))

    imp_series = pd.Series(imp).sort_values(ascending=True)

    # Current bar prediction
    last_feat = X_df.iloc[[-1]].values
    proba     = pipe.predict_proba(last_feat)[0]
    classes   = pipe.classes_
    pred_cls  = classes[np.argmax(proba)]
    pred_label= {0: "DOWN", 1: "FLAT", 2: "UP"}.get(pred_cls, "?")
    pred_conf = proba.max()

    return imp_series, pred_label, pred_conf


# ══════════════════════════════════════════════════════════════
#  7. STATS SUMMARY
# ══════════════════════════════════════════════════════════════
def print_stats(df, trades, cfg, ml_pred, ml_conf):
    last = df.iloc[-1]
    sig  = last["Signal"]
    sig_c = "\033[92m" if sig=="BUY" else ("\033[91m" if sig=="SELL" else "\033[93m")
    rst  = "\033[0m"

    print(f"\n{'═'*58}")
    print(f"  AI TRADING BOT  —  {cfg['SYMBOL']}")
    print(f"{'═'*58}")
    print(f"  Latest date    : {df.index[-1].date()}")
    print(f"  Current price  : {last['Close']:,.2f}")
    print(f"  Signal         : {sig_c}{sig}{rst}  (score: {last['Score']:.1f})")
    print(f"  RSI            : {last['RSI']:.1f}")
    print(f"  MACD hist      : {last['MACD_hist']:.2f}")
    print(f"  ADX            : {last['ADX']:.1f}  ({'Trending' if last['ADX']>25 else 'Ranging'})")
    if ml_pred:
        conf_c = "\033[92m" if ml_conf>0.6 else ("\033[93m" if ml_conf>0.5 else "\033[91m")
        print(f"  ML prediction  : {conf_c}{ml_pred}  ({ml_conf:.1%} confidence){rst}")
    print(f"{'─'*58}")

    if trades.empty:
        print("  No completed trades in this period."); return

    wins       = trades[trades["Result"] == "WIN"]
    losses     = trades[trades["Result"] == "LOSS"]
    win_rate   = len(wins) / len(trades) * 100
    total_pnl  = trades["PnL $"].sum()
    avg_win    = wins["PnL $"].mean()   if len(wins)   > 0 else 0
    avg_loss   = losses["PnL $"].mean() if len(losses) > 0 else 0
    best       = trades["PnL %"].max()
    worst      = trades["PnL %"].min()
    pf         = (wins["PnL $"].sum() / abs(losses["PnL $"].sum())
                  if losses["PnL $"].sum() != 0 else 999)

    print(f"  {'TRADE STATISTICS':}")
    print(f"{'─'*58}")
    print(f"  Total trades   : {len(trades)}")
    print(f"  Wins / Losses  : {len(wins)} / {len(losses)}")
    pnl_c = "\033[92m" if win_rate >= 50 else "\033[91m"
    print(f"  Win rate       : {pnl_c}{win_rate:.1f}%{rst}")
    tot_c = "\033[92m" if total_pnl >= 0 else "\033[91m"
    print(f"  Total PnL      : {tot_c}${total_pnl:,.0f}{rst}  "
          f"({total_pnl/cfg['INITIAL_CAPITAL']*100:.1f}%)")
    print(f"  Avg win        : \033[92m${avg_win:,.0f}\033[0m")
    print(f"  Avg loss       : \033[91m${avg_loss:,.0f}\033[0m")
    print(f"  Profit factor  : {pf:.2f}")
    print(f"  Best trade     : +{best:.1f}%")
    print(f"  Worst trade    : {worst:.1f}%")
    print(f"{'─'*58}")
    print(f"  LAST 10 TRADES")
    print(f"{'─'*58}")
    for _, r in trades.tail(10).iterrows():
        c = "\033[92m" if r["Result"]=="WIN" else "\033[91m"
        print(f"  {str(r['Entry Date']):<12} → {str(r['Exit Date']):<12}  "
              f"{c}{r['PnL %']:+6.2f}%   ${r['PnL $']:>9,.0f}{rst}")
    print(f"{'═'*58}\n")


# ══════════════════════════════════════════════════════════════
#  8. CHARTS
# ══════════════════════════════════════════════════════════════
def plot_all(df, trades, imp_series, cfg):
    sym  = cfg["SYMBOL"]
    safe = sym.replace("-","_").replace(".","_")

    # ── CHART 1: Main trading chart ─────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(4, 1, height_ratios=[3.5, 1, 1, 0.5], hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    dark = {"facecolor":"#111111","grid":"#1e1e1e","tick":"#888","spine":"#2a2a2a"}
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(dark["facecolor"])
        ax.tick_params(colors=dark["tick"], labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(dark["spine"])
        ax.grid(color=dark["grid"], lw=0.4, alpha=0.8)

    idx = df.index

    # ── Panel 1: Price ──────────────────────────────────────
    ax1.plot(idx, df["Close"],    color="#e0e0e0", lw=1.4, label="Price", zorder=4)
    ax1.plot(idx, df["EMA_fast"], color="#00bcd4", lw=0.9,
             label=f"EMA{cfg['EMA_FAST']}", alpha=0.85)
    ax1.plot(idx, df["EMA_slow"], color="#ff9800", lw=0.9,
             label=f"EMA{cfg['EMA_SLOW']}", alpha=0.85)
    ax1.plot(idx, df["EMA_trend"],color="#9c27b0", lw=0.9,
             label=f"EMA{cfg['EMA_TREND']}", alpha=0.75)
    ax1.fill_between(idx, df["BB_upper"], df["BB_lower"],
                     color="#4caf50", alpha=0.06)
    ax1.plot(idx, df["BB_upper"], color="#4caf50", lw=0.5, alpha=0.5, label="BB")
    ax1.plot(idx, df["BB_lower"], color="#4caf50", lw=0.5, alpha=0.5)

    # Trade entry/exit lines
    if not trades.empty:
        for _, tr in trades.iterrows():
            try:
                e_idx = pd.Timestamp(tr["Entry Date"])
                x_idx = pd.Timestamp(tr["Exit Date"])
                e_p   = tr["Entry Price"]
                x_p   = tr["Exit Price"]
                col   = "#00e676" if tr["Result"]=="WIN" else "#ff5252"
                ax1.plot([e_idx, x_idx], [e_p, x_p],
                         color=col, lw=0.6, alpha=0.4, zorder=3)
            except Exception:
                pass

    # BUY / SELL markers — larger and clearer
    buys  = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    ax1.scatter(buys.index,  buys["Close"],
                color="#00e676", marker="^", s=150, zorder=6,
                label=f"BUY ({len(buys)})", edgecolors="#ffffff", linewidths=0.5)
    ax1.scatter(sells.index, sells["Close"],
                color="#ff1744", marker="v", s=150, zorder=6,
                label=f"SELL ({len(sells)})", edgecolors="#ffffff", linewidths=0.5)

    ax1.set_title(f"{sym}  ·  Signal Strategy  ·  "
                  f"{df.index[0].date()} → {df.index[-1].date()}",
                  color="#e0e0e0", fontsize=13, pad=10)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
               labelcolor="#cccccc", framealpha=0.85, ncol=4)
    ax1.set_ylabel("Price", color="#888", fontsize=9)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: RSI ────────────────────────────────────────
    ax2.plot(idx, df["RSI"], color="#e91e63", lw=1.0, label="RSI")
    ax2.axhline(cfg["RSI_OB"], color="#ff5722", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(cfg["RSI_OS"], color="#2196f3", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(50, color="#555", lw=0.6, ls=":")
    ax2.fill_between(idx, df["RSI"], cfg["RSI_OB"],
                     where=df["RSI"]>cfg["RSI_OB"], color="#ff5722", alpha=0.15)
    ax2.fill_between(idx, df["RSI"], cfg["RSI_OS"],
                     where=df["RSI"]<cfg["RSI_OS"], color="#2196f3", alpha=0.15)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="#888", fontsize=9)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: MACD ───────────────────────────────────────
    ax3.plot(idx, df["MACD"],     color="#00bcd4", lw=0.9, label="MACD")
    ax3.plot(idx, df["MACD_sig"], color="#ff9800", lw=0.9, label="Signal")
    hcols = ["#00e676" if v >= 0 else "#ff1744" for v in df["MACD_hist"]]
    ax3.bar(idx, df["MACD_hist"], color=hcols, alpha=0.55, width=0.8)
    ax3.axhline(0, color="#444", lw=0.6)
    ax3.set_ylabel("MACD", color="#888", fontsize=9)
    ax3.legend(loc="upper left", fontsize=7, facecolor="#1a1a1a", labelcolor="#ccc")
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── Panel 4: Score heatmap ──────────────────────────────
    cmap = LinearSegmentedColormap.from_list("score", ["#ff1744","#111111","#00e676"])
    scores = df["Score"].values.reshape(1, -1)
    ax4.imshow(scores, aspect="auto", cmap=cmap,
               extent=[mdates.date2num(idx[0]),
                       mdates.date2num(idx[-1]), 0, 1],
               vmin=df["Score"].quantile(0.05),
               vmax=df["Score"].quantile(0.95))
    ax4.set_ylabel("Score", color="#888", fontsize=8)
    ax4.set_yticks([])
    ax4.xaxis_date()
    ax4.tick_params(axis="x", labelsize=7, colors="#888", rotation=15)

    plt.tight_layout()
    f1 = f"chart_{safe}.png"
    plt.savefig(f1, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  Chart saved → {f1}")
    if cfg["SHOW_CHART"]: plt.show()
    plt.close()

    # ── CHART 2: Cumulative PnL ─────────────────────────────
    if not trades.empty:
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={"height_ratios":[2,1]})
        fig2.patch.set_facecolor("#0d0d0d")

        for ax in axes2:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#888", labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2a2a")
            ax.grid(color="#1e1e1e", lw=0.4, alpha=0.8)

        cum_pnl  = trades["PnL $"].cumsum()
        t_idx    = range(len(trades))

        # Cumulative PnL line
        colors_pnl = ["#00e676" if v >= 0 else "#ff1744" for v in cum_pnl]
        axes2[0].fill_between(t_idx, cum_pnl, 0,
                              where=cum_pnl >= 0, color="#00e676", alpha=0.15)
        axes2[0].fill_between(t_idx, cum_pnl, 0,
                              where=cum_pnl < 0,  color="#ff1744", alpha=0.15)
        axes2[0].plot(t_idx, cum_pnl, color="#00bcd4", lw=1.8, zorder=4)
        axes2[0].axhline(0, color="#555", lw=0.8, ls="--")
        axes2[0].scatter(t_idx,
                         [p if t=="WIN" else np.nan
                          for p,t in zip(cum_pnl, trades["Result"])],
                         color="#00e676", s=40, zorder=5)
        axes2[0].scatter(t_idx,
                         [p if t=="LOSS" else np.nan
                          for p,t in zip(cum_pnl, trades["Result"])],
                         color="#ff1744", s=40, zorder=5)
        axes2[0].set_title(f"{sym}  ·  Cumulative PnL  ·  "
                           f"{len(trades)} trades  ·  "
                           f"Win rate {len(trades[trades['Result']=='WIN'])/len(trades)*100:.1f}%",
                           color="#e0e0e0", fontsize=12, pad=10)
        axes2[0].set_ylabel("Cumulative PnL ($)", color="#888", fontsize=9)

        # Per-trade bar chart
        bar_cols = ["#00e676" if r=="WIN" else "#ff1744"
                    for r in trades["Result"]]
        axes2[1].bar(t_idx, trades["PnL $"], color=bar_cols, alpha=0.75, width=0.7)
        axes2[1].axhline(0, color="#555", lw=0.8)
        axes2[1].set_xlabel("Trade #", color="#888", fontsize=9)
        axes2[1].set_ylabel("PnL per trade ($)", color="#888", fontsize=9)

        plt.tight_layout()
        f2 = f"pnl_{safe}.png"
        plt.savefig(f2, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  PnL chart saved → {f2}")
        if cfg["SHOW_CHART"]: plt.show()
        plt.close()

    # ── CHART 3: XGBoost Feature Importance ─────────────────
    if imp_series is not None and len(imp_series) > 0:
        top = imp_series.sort_values(ascending=True).tail(15)

        # Normalize so the spread is visible
        rng  = top.max() - top.min()
        norm = (top - top.min()) / rng if rng > 0 else top

        fig3, ax = plt.subplots(figsize=(10, 7))
        fig3.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#111111")

        # Color by quartile so bars look visually different
        bar_colors = []
        for v in norm.values:
            if   v >= 0.75: bar_colors.append("#00e676")   # top quartile — green
            elif v >= 0.50: bar_colors.append("#00bcd4")   # 2nd — cyan
            elif v >= 0.25: bar_colors.append("#ff9800")   # 3rd — amber
            else:           bar_colors.append("#9c27b0")   # bottom — purple

        bars = ax.barh(top.index, top.values, color=bar_colors,
                       alpha=0.85, height=0.65)

        # Value labels on bars
        for bar, val in zip(bars, top.values):
            ax.text(bar.get_width() + top.max()*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", ha="left",
                    color="#cccccc", fontsize=8)

        ax.axvline(top.median(), color="#ff5722", lw=1.2,
                   ls="--", alpha=0.7, label="median")

        ax.set_xlabel("Importance (gain)", color="#aaaaaa", fontsize=10)
        ax.set_title(f"XGBoost — Feature Importance  ({sym})\n"
                     f"Green = highest predictive power",
                     color="#ffffff", fontsize=12, pad=12)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a2a")
        ax.grid(axis="x", color="#1e1e1e", lw=0.5, alpha=0.8)
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="#ccc")

        # Quartile legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(color="#00e676", label="Top 25% — strongest predictors"),
            Patch(color="#00bcd4", label="2nd quartile"),
            Patch(color="#ff9800", label="3rd quartile"),
            Patch(color="#9c27b0", label="Bottom 25%"),
        ]
        ax.legend(handles=legend_els, fontsize=8,
                  facecolor="#1a1a1a", labelcolor="#ccc",
                  loc="lower right")

        plt.tight_layout()
        f3 = f"feature_importance_{safe}.png"
        plt.savefig(f3, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Feature importance chart saved → {f3}")
        if cfg["SHOW_CHART"]: plt.show()
        plt.close()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    print("\n" + "═"*58)
    print(f"  AI TRADING BOT  —  {cfg['SYMBOL']}")
    print("═"*58)

    # 1. Data
    df = fetch(cfg)

    # 2. Indicators
    print("  Computing indicators...")
    df = add_indicators(df, cfg)

    # 3. Score → raw signals
    print("  Scoring signals...")
    df = score_signals(df, cfg)

    # 4. Dedup — enforce BUY → SELL pairing
    df = dedup_signals(df)

    n_buy  = (df["Signal"] == "BUY").sum()
    n_sell = (df["Signal"] == "SELL").sum()
    print(f"  Signals: {n_buy} BUY  |  {n_sell} SELL  "
          f"(strength threshold = {cfg['SIGNAL_STRENGTH']})")

    # 5. Pair trades
    trades = pair_trades(df, cfg["INITIAL_CAPITAL"])
    print(f"  Completed trades: {len(trades)}")

    # 6. ML
    imp_series, ml_pred, ml_conf = run_ml(df, cfg)

    # 7. Print stats
    print_stats(df, trades, cfg, ml_pred, ml_conf)

    # 8. Charts
    if cfg["SHOW_CHART"]:
        print("  Rendering charts...")
        plot_all(df, trades, imp_series, cfg)

    print("  Done.\n")


if __name__ == "__main__":
    main()
