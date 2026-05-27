# ============================================================
#  MULTI-STOCK v24 — HONEST QUANTITATIVE BACKTESTER
# ============================================================
#  A leak-free, walk-forward evaluated directional model with
#  statistically honest reporting. No fabricated confidence,
#  no "production ready" theatre — it tells you if there is a
#  real, significant edge, and refuses to pretend when there isn't.
#
#  Methodology highlights (vs naive single-split scripts):
#    * Walk-forward (expanding window) out-of-sample evaluation
#    * Purged training to remove overlapping-horizon leakage
#    * Evaluation horizon == training horizon (consistent)
#    * Non-overlapping holding-period equity curve
#    * Transaction cost + slippage modeled
#    * Binomial significance test + Spearman information coefficient
#    * Adjusted close (dividend/split aware)
#
#  DISCLAIMER: Educational/research tool. NOT investment advice.
#  Past backtested performance does not guarantee future results.
# ============================================================

from __future__ import annotations   # allow "X | None" type hints on Python 3.7–3.9

import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import yfinance as yf

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TICKERS = {
    'VOO':  {'name': 'Vanguard S&P 500 ETF',     'benchmark': 'SOXX'},
    'INTC': {'name': 'Intel Corporation',     'benchmark': 'SOXX'},
    'AMD':  {'name': 'Advanced Micro Devices', 'benchmark': 'SOXX'},
    'NVDA': {'name': 'NVIDIA Corporation',     'benchmark': 'SOXX'},
}

YEARS_HISTORY      = 6          # more data -> more walk-forward folds (was 3)
PREDICTION_HORIZON = 21         # trading days ahead
LOOKBACK           = 20         # feature window length
RIDGE_ALPHA        = 5.0
N_WALK_FOLDS       = 6
MIN_TRAIN          = 250
COST_BPS           = 5.0        # round-trip cost (commission + slippage), basis points
SEED               = 42

FEATURE_COLS = ['mom5', 'mom20', 'vol20', 'vol60', 'sma_ratio', 'rsi14', 'dist_high', 'vix']


# ============================================================
# DATA
# ============================================================
def download_close(ticker: str) -> pd.Series | None:
    """Download dividend/split-adjusted close. Returns None on failure."""
    end = datetime.now()
    start = end - timedelta(days=int(YEARS_HISTORY * 365.25))
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"   download error: {e}")
        return None
    if df is None or df.empty:
        return None
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    s = df['Close'].dropna()
    s.name = ticker
    return s


def download_vix() -> pd.Series | None:
    end = datetime.now()
    start = end - timedelta(days=int(YEARS_HISTORY * 365.25))
    try:
        df = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if hasattr(df.columns, 'get_level_values'):
            df.columns = df.columns.get_level_values(0)
        return df['Close'].dropna()
    except Exception:
        return None


# ============================================================
# FEATURES (strictly backward-looking)
# ============================================================
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(close: pd.Series, vix: pd.Series | None) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df['Close']     = close
    df['LogRet']    = np.log(close / close.shift(1))
    df['mom5']      = close.pct_change(5)
    df['mom20']     = close.pct_change(20)
    df['vol20']     = df['LogRet'].rolling(20).std()
    df['vol60']     = df['LogRet'].rolling(60).std()
    df['sma_ratio'] = close.rolling(20).mean() / close.rolling(50).mean()
    df['rsi14']     = _rsi(close, 14)
    df['dist_high'] = close / close.rolling(60).max() - 1.0

    if vix is not None and len(vix):
        df['vix'] = vix.reindex(df.index).ffill()
    else:
        df['vix'] = 20.0

    # Forward target. Last PREDICTION_HORIZON rows are unknown (NaN) -> dropped later.
    df['target'] = np.log(close.shift(-PREDICTION_HORIZON) / close)
    return df


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
@dataclass
class FoldResult:
    dates: list
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class BacktestResult:
    ok: bool = False
    n_obs: int = 0
    dir_acc: float = 0.0
    naive: float = 0.0
    edge: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    ic: float = 0.0
    ic_p: float = 1.0
    strat_return: float = 0.0
    bh_return: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    n_trades: int = 0
    folds: list = field(default_factory=list)
    all_dates: list = field(default_factory=list)
    all_true: np.ndarray = None
    all_pred: np.ndarray = None
    strat_curve: np.ndarray = None
    bh_curve: np.ndarray = None
    latest_pred: float = 0.0       # model's view on the most recent (unrealized) point


def _make_xy(feat: pd.DataFrame):
    fvals = feat[FEATURE_COLS].values
    yvals = feat['target'].values
    idx = feat.index
    X, y, dates = [], [], []
    for i in range(LOOKBACK, len(feat)):
        if np.isnan(yvals[i]):
            continue
        window = fvals[i - LOOKBACK:i]
        if np.isnan(window).any():
            continue
        X.append(window.flatten()); y.append(yvals[i]); dates.append(idx[i])
    return np.array(X), np.array(y), dates


def _latest_live_window(feat: pd.DataFrame):
    """Most recent fully-formed feature window (label not yet known)."""
    fvals = feat[FEATURE_COLS].values
    for i in range(len(feat) - 1, LOOKBACK - 1, -1):
        window = fvals[i - LOOKBACK:i]
        if not np.isnan(window).any():
            return window.flatten()
    return None


def walk_forward(feat: pd.DataFrame) -> BacktestResult:
    np.random.seed(SEED)
    res = BacktestResult()
    X, y, dates = _make_xy(feat)
    if len(X) < MIN_TRAIN + N_WALK_FOLDS * 10:
        return res

    n = len(X)
    test_block = (n - MIN_TRAIN) // N_WALK_FOLDS
    if test_block < 5:
        return res

    folds = []
    last_model = last_xs = last_ys = None
    for k in range(N_WALK_FOLDS):
        test_lo = MIN_TRAIN + k * test_block
        test_hi = (MIN_TRAIN + (k + 1) * test_block) if k < N_WALK_FOLDS - 1 else n
        if test_lo >= n:
            break
        train_hi = test_lo - PREDICTION_HORIZON          # PURGE overlap
        if train_hi < MIN_TRAIN // 2:
            continue

        Xtr, ytr = X[:train_hi], y[:train_hi]
        Xte, yte = X[test_lo:test_hi], y[test_lo:test_hi]
        dte = dates[test_lo:test_hi]
        if len(Xte) == 0:
            continue

        xs, ysc = StandardScaler(), StandardScaler()
        Xtr_s = xs.fit_transform(Xtr)
        ytr_s = ysc.fit_transform(ytr.reshape(-1, 1)).ravel()
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(Xtr_s, ytr_s)
        pred = ysc.inverse_transform(model.predict(xs.transform(Xte)).reshape(-1, 1)).ravel()
        folds.append(FoldResult(dte, yte, pred))
        last_model, last_xs, last_ys = model, xs, ysc

    if not folds:
        return res

    all_true = np.concatenate([f.y_true for f in folds])
    all_pred = np.concatenate([f.y_pred for f in folds])
    all_dates = [d for f in folds for d in f.dates]

    res.ok = True
    res.folds = folds
    res.all_true, res.all_pred, res.all_dates = all_true, all_pred, all_dates
    res.n_obs = len(all_true)

    nz = all_true != 0
    correct = int((np.sign(all_pred) == np.sign(all_true))[nz].sum())
    res.dir_acc = correct / nz.sum() * 100
    pos_rate = (all_true > 0).mean()
    res.naive = max(pos_rate, 1 - pos_rate) * 100
    res.edge = res.dir_acc - res.naive

    res.p_value = stats.binomtest(correct, int(nz.sum()), res.naive / 100,
                                  alternative='greater').pvalue
    res.significant = res.p_value < 0.05

    if len(all_true) > 5:
        ic, icp = stats.spearmanr(all_pred, all_true)
        res.ic, res.ic_p = float(ic), float(icp)

    _equity(res, all_dates, all_true, all_pred)

    # Live prediction for the most recent point (forward-looking, unrealized)
    live = _latest_live_window(feat)
    if live is not None and last_model is not None:
        live_s = last_xs.transform(live.reshape(1, -1))
        res.latest_pred = float(last_ys.inverse_transform(
            last_model.predict(live_s).reshape(-1, 1)).ravel()[0])
    return res


def _equity(res: BacktestResult, dates, y_true, y_pred):
    order = np.argsort(dates)
    yt = y_true[order]; yp = y_pred[order]
    cost = COST_BPS / 1e4
    strat, bh = [], []
    n_trades = 0
    i = 0
    while i < len(yt):
        sig = np.sign(yp[i])
        g = sig * (np.exp(yt[i]) - 1)
        if sig != 0:
            n_trades += 1
            g -= cost
        strat.append(g)
        bh.append(np.exp(yt[i]) - 1)
        i += PREDICTION_HORIZON                  # non-overlapping holds
    strat = np.array(strat); bh = np.array(bh)
    res.n_trades = n_trades
    res.strat_return = (np.prod(1 + strat) - 1) * 100
    res.bh_return = (np.prod(1 + bh) - 1) * 100
    if len(strat) > 1 and strat.std() > 0:
        res.sharpe = strat.mean() / strat.std() * np.sqrt(252 / PREDICTION_HORIZON)
    eq = np.cumprod(1 + strat)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    res.max_dd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
    res.strat_curve = eq
    res.bh_curve = np.cumprod(1 + bh)


# ============================================================
# VERDICT (honest, rule-based)
# ============================================================
def verdict(res: BacktestResult) -> tuple[str, str, str]:
    """Returns (label, color, explanation). Based on statistical significance,
    not on cosmetics. Refuses to claim an edge that isn't significant."""
    if not res.ok:
        return ("INSUFFICIENT DATA", "#6b7280",
                "Not enough history to evaluate this model reliably.")
    if res.significant and res.ic > 0.05 and res.strat_return > res.bh_return:
        return ("STATISTICALLY SIGNIFICANT EDGE", "#15803d",
                "The model beats the naive baseline at p < 0.05, shows positive "
                "rank correlation with future returns, and outperforms buy & hold "
                "net of costs in this backtest. Treat with caution: still no guarantee.")
    if res.significant and res.ic > 0:
        return ("WEAK / BORDERLINE SIGNAL", "#a16207",
                "Statistically better than the baseline, but the economic edge is "
                "thin and may not survive live trading costs and regime change.")
    return ("NO RELIABLE EDGE", "#b91c1c",
            "The model does NOT beat a naive majority-class baseline at a "
            "statistically significant level. On this data it provides no "
            "dependable directional edge. Do not trade it.")


def direction_text(latest_pred: float) -> str:
    if latest_pred > 0.005:
        return f"Model leans UP over next {PREDICTION_HORIZON}d (est. {latest_pred*100:+.1f}%)"
    if latest_pred < -0.005:
        return f"Model leans DOWN over next {PREDICTION_HORIZON}d (est. {latest_pred*100:+.1f}%)"
    return f"Model is roughly NEUTRAL over next {PREDICTION_HORIZON}d ({latest_pred*100:+.1f}%)"


def _score_to_action(score: float, band: float) -> tuple[str, str]:
    """Map a normalized score to a BUY/SELL/HOLD label + color, with STRONG tiers."""
    band = max(band, 1e-9)
    if score > band:
        strong = score > 2 * band
        return ("STRONG BUY" if strong else "BUY",
                "#15803d" if strong else "#16a34a")
    if score < -band:
        strong = score < -2 * band
        return ("STRONG SELL" if strong else "SELL",
                "#b91c1c" if strong else "#dc2626")
    return ("HOLD / NEUTRAL", "#6b7280")


def trade_signals(res: BacktestResult, feat: pd.DataFrame) -> dict:
    """
    Produce THREE complementary signals, each shown raw (always visible):

      1) NOW · Technical  -> today's technical posture (momentum + RSI + trend).
                             Short-term read, independent of the ML model.
      2) NOW · Model      -> the model's single most-recent prediction (nearest day).
      3) 21-DAY · Forecast-> the model's forward {H}-day directional call.

    A separate reliability tag (from the walk-forward backtest) is returned once,
    describing how much trust the MODEL-based signals have earned on this data.
    """
    H = PREDICTION_HORIZON

    # ---- 2 & 3: model-based band, scaled to the model's own prediction spread ----
    if res.ok and res.all_pred is not None and len(res.all_pred) > 5:
        m_band = max(0.5 * float(np.std(res.all_pred)), 0.004)
    else:
        m_band = 0.005

    # (3) 21-day forecast = the live forward prediction
    fc_pred = res.latest_pred
    fc_act, fc_col = _score_to_action(fc_pred, m_band)

    # (2) "now / model" = most-recent realized out-of-sample prediction if available,
    #     else the live forecast. Represents the model's nearest-day stance.
    if res.ok and res.all_pred is not None and len(res.all_pred):
        now_model_pred = float(res.all_pred[-1])
    else:
        now_model_pred = fc_pred
    nm_act, nm_col = _score_to_action(now_model_pred, m_band)

    # ---- 1: NOW / technical, from today's feature row (model-independent) ----
    last = feat.iloc[-1]
    mom20 = float(last.get('mom20', 0.0))
    mom5  = float(last.get('mom5', 0.0))
    rsi   = float(last.get('rsi14', 50.0))
    sma_r = float(last.get('sma_ratio', 1.0))

    # composite technical score in [-1, 1]: trend + momentum + RSI tilt
    trend_term = np.tanh((sma_r - 1.0) * 25)        # >0 if 20d MA above 50d MA
    mom_term   = np.tanh((mom20 + mom5) * 8)        # recent momentum
    rsi_term   = np.tanh((rsi - 50) / 20)           # overbought/oversold tilt
    tech_score = float(np.clip(0.45 * trend_term + 0.40 * mom_term + 0.15 * rsi_term, -1, 1))
    tech_act, tech_col = _score_to_action(tech_score, 0.15)

    # ---- reliability of the model-based signals ----
    if not res.ok:
        rel, rel_col = "Model unverified (insufficient data)", "#6b7280"
    elif res.significant and res.ic > 0.05 and res.strat_return > res.bh_return:
        rel, rel_col = "Model backtest: statistically reliable", "#15803d"
    elif res.significant and res.ic > 0:
        rel, rel_col = "Model backtest: weak / borderline", "#a16207"
    else:
        rel, rel_col = "Model backtest: no reliable edge — informational only", "#b91c1c"

    return dict(
        # 1) now technical
        tech_action=tech_act, tech_color=tech_col, tech_score=tech_score,
        rsi=rsi, mom20=mom20,
        # 2) now model
        now_action=nm_act, now_color=nm_col, now_pred=now_model_pred,
        # 3) 21-day forecast
        fc_action=fc_act, fc_color=fc_col, fc_pred=fc_pred, band=m_band,
        # reliability (applies to model signals)
        reliability=rel, reliability_color=rel_col,
    )


# ============================================================
# PLOTS
# ============================================================
def _fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    uri = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{uri}"


def plot_price(close: pd.Series, res: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(15, 5.5))

    # --- Actual price line ---
    ax.plot(close.index, close.values, lw=1.7, color='#1e3a5f',
            label='Actual price (adj close)', zorder=3)

    # --- Out-of-sample test region + per-point directional calls ---
    if res.ok and res.all_dates:
        t0 = min(res.all_dates)
        ax.axvspan(t0, close.index[-1], color='#fde68a', alpha=0.22,
                   label='Walk-forward test region')
        up = [(d, close.loc[d]) for d, p in zip(res.all_dates, res.all_pred) if p > 0]
        dn = [(d, close.loc[d]) for d, p in zip(res.all_dates, res.all_pred) if p <= 0]
        if up:
            ax.scatter(*zip(*up), s=16, c='#16a34a', marker='^', alpha=0.45,
                       label='Past pred: up', zorder=4)
        if dn:
            ax.scatter(*zip(*dn), s=16, c='#dc2626', marker='v', alpha=0.45,
                       label='Past pred: down', zorder=4)

    # --- FORWARD 21-DAY PROJECTION (the live, unrealized forecast) ---
    last_date = close.index[-1]
    last_price = float(close.iloc[-1])
    pred = res.latest_pred                                   # predicted H-day log return
    target = last_price * np.exp(pred)                       # projected price level

    future_dates = pd.bdate_range(last_date, periods=PREDICTION_HORIZON + 1)
    path = last_price * np.exp(np.linspace(0, pred, len(future_dates)))

    up_fc = pred > 0
    fc_color = '#16a34a' if up_fc else ('#dc2626' if pred < 0 else '#6b7280')
    arrow = '^' if up_fc else ('v' if pred < 0 else '>')

    ax.plot(future_dates, path, lw=2.4, color=fc_color, ls='--',
            label=f'Forecast next {PREDICTION_HORIZON}d ({pred*100:+.1f}%)', zorder=5)

    # uncertainty band scaled by historical prediction error
    if res.ok and res.all_true is not None and len(res.all_true) > 5:
        resid_std = float(np.std(res.all_true - res.all_pred))
    else:
        resid_std = abs(pred) + 0.02
    upper = last_price * np.exp(np.linspace(0, pred + resid_std, len(future_dates)))
    lower = last_price * np.exp(np.linspace(0, pred - resid_std, len(future_dates)))
    ax.fill_between(future_dates, lower, upper, color=fc_color, alpha=0.12,
                    label='Forecast uncertainty (1 sigma)', zorder=2)

    # mark today's price and the projected target
    ax.scatter([last_date], [last_price], s=70, color='#1e3a5f', zorder=6)
    ax.scatter([future_dates[-1]], [target], s=110, color=fc_color,
               marker='*', zorder=6, edgecolor='white', linewidth=0.8)
    ax.annotate(f'{arrow} ${target:,.2f}',
                xy=(future_dates[-1], target),
                xytext=(8, 10 if up_fc else -16), textcoords='offset points',
                fontsize=11, fontweight='bold', color=fc_color)
    ax.annotate(f'Today ${last_price:,.2f}',
                xy=(last_date, last_price),
                xytext=(-98, -18), textcoords='offset points',
                fontsize=9.5, color='#1e3a5f')

    ax.set_title(f'Price History + {PREDICTION_HORIZON}-Day Forward Forecast',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8.5, ncol=2)
    # zoom to recent ~400 days so the forecast at the right edge is clearly visible
    lo = max(0, len(close) - 400)
    ax.set_xlim(close.index[lo], future_dates[-1])
    # tighten y-axis to the visible window (with headroom) instead of starting at 0
    vis = close.values[lo:]
    ymin = min(vis.min(), lower.min()); ymax = max(vis.max(), upper.max())
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad)
    return _fig_to_data_uri(fig)


def plot_equity(res: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(15, 5))
    if res.ok and res.strat_curve is not None:
        ax.plot(res.strat_curve, lw=2, color='#ea580c', marker='o', ms=3,
                label=f'Strategy (net, {res.n_trades} trades)')
        ax.plot(res.bh_curve, lw=2, color='#1e3a5f', marker='o', ms=3, label='Buy & Hold')
        ax.axhline(1.0, color='#666', ls='--', alpha=0.5)
    ax.set_title('Equity Curve — Non-Overlapping Holds, Net of Costs',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Growth of $1'); ax.set_xlabel('Trade #')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=10)
    return _fig_to_data_uri(fig)


def plot_diagnostics(res: BacktestResult) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].bar(['Model', 'Naive\nbaseline'], [res.dir_acc, res.naive],
                color=['#1e3a5f', '#9ca3af'], alpha=0.85)
    axes[0].axhline(50, color='#dc2626', ls='--', alpha=0.5, label='Coin flip (50%)')
    for i, v in enumerate([res.dir_acc, res.naive]):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[0].set_ylim(0, 100); axes[0].set_ylabel('Directional accuracy (%)')
    axes[0].set_title('Accuracy vs Baseline', fontweight='bold'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25, axis='y')

    if res.ok:
        axes[1].scatter(res.all_pred * 100, res.all_true * 100, s=14, alpha=0.4, color='#1e3a5f')
        lim = max(np.abs(res.all_pred).max(), np.abs(res.all_true).max()) * 100 * 1.1
        axes[1].plot([-lim, lim], [-lim, lim], color='#dc2626', ls='--', alpha=0.6)
        axes[1].axhline(0, color='#999', lw=0.6); axes[1].axvline(0, color='#999', lw=0.6)
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel('Predicted return (%)'); axes[1].set_ylabel('Actual return (%)')
    axes[1].set_title(f'Predicted vs Actual  (IC={res.ic:+.3f})', fontweight='bold')
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    return _fig_to_data_uri(fig)


# ---------- NEW CHARTS (added; existing charts untouched) ----------
def plot_drawdown(close: pd.Series) -> str:
    """Underwater (drawdown-from-peak) curve — the key risk picture."""
    c = close.values
    peak = np.maximum.accumulate(c)
    dd = (c - peak) / peak * 100.0
    max_dd = dd.min()
    max_dd_date = close.index[int(np.argmin(dd))]

    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.fill_between(close.index, dd, 0, color='#dc2626', alpha=0.25)
    ax.plot(close.index, dd, color='#b91c1c', lw=1.3)
    ax.axhline(0, color='#444', lw=0.8)
    ax.scatter([max_dd_date], [max_dd], color='#7f1d1d', s=55, zorder=5)
    ax.annotate(f'Max drawdown {max_dd:.1f}%',
                xy=(max_dd_date, max_dd), xytext=(10, 14),
                textcoords='offset points', fontsize=10.5, fontweight='bold',
                color='#7f1d1d')
    ax.set_title('Drawdown from Peak (Underwater Curve)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Drawdown (%)'); ax.grid(True, alpha=0.25)
    return _fig_to_data_uri(fig)


def plot_bollinger(close: pd.Series) -> str:
    """Price with 20/50-day moving averages and 20-day Bollinger Bands."""
    s = close
    ma20 = s.rolling(20).mean()
    ma50 = s.rolling(50).mean()
    std20 = s.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    # show recent ~300 sessions for clarity
    lo = max(0, len(s) - 300)
    idx = s.index[lo:]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(idx, lower.values[lo:], upper.values[lo:],
                    color='#1e3a5f', alpha=0.10, label='Bollinger (20, 2σ)')
    ax.plot(idx, s.values[lo:], color='#1e3a5f', lw=1.7, label='Price')
    ax.plot(idx, ma20.values[lo:], color='#ea580c', lw=1.4, label='MA 20')
    ax.plot(idx, ma50.values[lo:], color='#15803d', lw=1.4, ls='--', label='MA 50')
    # mark last price
    ax.scatter([idx[-1]], [s.values[-1]], color='#1e3a5f', s=55, zorder=5)
    ax.annotate(f'${s.values[-1]:,.2f}', xy=(idx[-1], s.values[-1]),
                xytext=(8, 0), textcoords='offset points', fontsize=10,
                fontweight='bold', color='#1e3a5f', va='center')
    ax.set_title('Moving Averages & Bollinger Bands (last ~300 sessions)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Price ($)'); ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    return _fig_to_data_uri(fig)


def plot_monthly_heatmap(close: pd.Series) -> str:
    """Calendar heatmap of monthly returns (rows = year, cols = month)."""
    monthly = close.resample('ME').last().pct_change() * 100
    if monthly.dropna().empty:
        fig, ax = plt.subplots(figsize=(15, 2))
        ax.text(0.5, 0.5, 'Not enough data for seasonality', ha='center')
        ax.axis('off')
        return _fig_to_data_uri(fig)

    df = pd.DataFrame({'ret': monthly.dropna()})
    df['year'] = df.index.year
    df['month'] = df.index.month
    grid = df.pivot_table(index='year', columns='month', values='ret', aggfunc='first')
    grid = grid.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(15, 0.7 * len(grid) + 1.6))
    vmax = np.nanmax(np.abs(grid.values)) if np.isfinite(grid.values).any() else 5
    im = ax.imshow(grid.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(12)); ax.set_xticklabels(months)
    ax.set_yticks(range(len(grid))); ax.set_yticklabels(grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=8.5, color='#222')
    ax.set_title('Monthly Returns Heatmap (%) — Seasonality',
                 fontweight='bold', fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('Return (%)', fontsize=9)
    plt.tight_layout()
    return _fig_to_data_uri(fig)
# ---------- end new charts ----------


# ============================================================
# ANALYZE ONE TICKER
# ============================================================
def analyze(ticker: str, name: str, vix: pd.Series | None):
    print(f"\n{'='*62}\n  {ticker} — {name}\n{'='*62}")
    close = download_close(ticker)
    if close is None or len(close) < 300:
        print("   No / insufficient data")
        return None
    print(f"   {len(close)} trading days (~{YEARS_HISTORY}y)")

    feat = build_features(close, vix)
    cur = float(close.iloc[-1])
    chg = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    cur_vix = float(vix.dropna().iloc[-1]) if (vix is not None and len(vix.dropna())) else 20.0
    print(f"   Last close: ${cur:.2f} ({chg:+.2f}%)  VIX={cur_vix:.1f}")

    res = walk_forward(feat)
    if not res.ok:
        print("   Backtest could not run (insufficient folds)")
        return None

    lbl, color, expl = verdict(res)
    print(f"\n   ── WALK-FORWARD RESULTS ──")
    print(f"   Test obs: {res.n_obs} (across {len(res.folds)} folds)")
    print(f"   Dir Acc: {res.dir_acc:.1f}% | Naive: {res.naive:.1f}% | Edge: {res.edge:+.2f}%")
    print(f"   Significance p-value: {res.p_value:.4f}  -> {'SIGNIFICANT' if res.significant else 'not significant'}")
    print(f"   Info Coefficient (Spearman): {res.ic:+.3f} (p={res.ic_p:.3f})")
    print(f"   Strategy net: {res.strat_return:+.1f}% | Buy&Hold: {res.bh_return:+.1f}% | "
          f"Sharpe: {res.sharpe:.2f} | MaxDD: {res.max_dd:.1f}%")
    print(f"   VERDICT: {lbl}")
    print(f"   {direction_text(res.latest_pred)}")

    sig = trade_signals(res, feat)
    print(f"   SIGNALS  ->  NOW/Technical: {sig['tech_action']:14s} | "
          f"NOW/Model: {sig['now_action']:14s} | "
          f"{PREDICTION_HORIZON}d/Forecast: {sig['fc_action']}")
    print(f"   ({sig['reliability']})")

    # 52-week range + position (for the price card)
    win52 = close.iloc[-252:] if len(close) >= 252 else close
    hi52 = float(win52.max())
    lo52 = float(win52.min())
    pos52 = (cur - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else 50.0

    return dict(
        ticker=ticker, name=name, cur=cur, chg=chg, cur_vix=cur_vix,
        hi52=hi52, lo52=lo52, pos52=pos52,
        res=res, label=lbl, color=color, expl=expl, sig=sig,
        img_price=plot_price(close, res),
        img_equity=plot_equity(res),
        img_diag=plot_diagnostics(res),
        img_drawdown=plot_drawdown(close),
        img_bollinger=plot_bollinger(close),
        img_heatmap=plot_monthly_heatmap(close),
    )


# ============================================================
# HTML REPORT
# ============================================================
CSS = """<style>
:root{--ink:#0f1c2e;--paper:#f7f5f0;--card:#ffffff;--line:#e2ddd3;--muted:#6b6256;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif;
  background:var(--paper);color:var(--ink);line-height:1.55;padding:32px 18px}
.wrap{max-width:1180px;margin:0 auto}
.masthead{border-bottom:3px double var(--ink);padding-bottom:18px;margin-bottom:8px}
.kicker{font-family:'Helvetica Neue',Arial,sans-serif;letter-spacing:.32em;
  text-transform:uppercase;font-size:11px;color:var(--muted)}
h1{font-size:46px;line-height:1.02;letter-spacing:-.5px;margin:6px 0 4px;font-weight:700}
.sub{font-family:'Helvetica Neue',Arial,sans-serif;font-size:12.5px;color:var(--muted)}
.disclaimer{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11.5px;color:#8a1f1f;
  background:#fbeeee;border:1px solid #e9c9c9;padding:10px 14px;border-radius:6px;margin:18px 0 30px}
.ticker-block{margin-bottom:54px}
.tk-head{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);
  padding-bottom:8px;margin-bottom:18px}
.tk-sym{font-size:30px;font-weight:700}
.tk-name{font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;color:var(--muted)}
.price-card{display:grid;grid-template-columns:1.1fr 1fr 1.2fr;gap:0;align-items:center;
  background:var(--ink);color:#fff;border-radius:12px;padding:24px 28px;margin-bottom:22px;
  box-shadow:0 4px 18px rgba(15,28,46,.18)}
.pc-main{border-right:1px solid rgba(255,255,255,.15);padding-right:24px}
.pc-label{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:rgba(255,255,255,.6);margin-bottom:6px}
.pc-price{font-family:'Helvetica Neue',Arial,sans-serif;font-size:42px;font-weight:800;
  line-height:1}
.pc-change{font-family:'Helvetica Neue',Arial,sans-serif;font-size:15px;font-weight:600;
  margin-top:6px}
.pc-change.up{color:#4ade80}.pc-change.down{color:#f87171}
.pc-stats{display:flex;flex-direction:column;gap:10px;padding:0 24px;
  border-right:1px solid rgba(255,255,255,.15)}
.pc-stat{display:flex;justify-content:space-between;font-family:'Helvetica Neue',Arial,sans-serif}
.pc-stat span{font-size:12px;color:rgba(255,255,255,.6)}
.pc-stat b{font-size:14px;font-weight:700}
.pc-range{padding-left:24px;font-family:'Helvetica Neue',Arial,sans-serif}
.pc-range-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:rgba(255,255,255,.6);margin-bottom:12px}
.pc-bar{position:relative;height:6px;background:rgba(255,255,255,.18);border-radius:3px;
  margin-bottom:8px}
.pc-bar-fill{position:absolute;left:0;top:0;height:6px;background:#4ade80;border-radius:3px}
.pc-bar-dot{position:absolute;top:-4px;width:14px;height:14px;background:#fff;border-radius:50%;
  transform:translateX(-50%);box-shadow:0 1px 4px rgba(0,0,0,.4)}
.pc-range-ends{display:flex;justify-content:space-between;font-size:12px;
  color:rgba(255,255,255,.7)}
.verdict{padding:20px 22px;border-radius:8px;color:#fff;margin-bottom:22px}
.verdict .v-label{font-family:'Helvetica Neue',Arial,sans-serif;font-weight:700;
  font-size:21px;letter-spacing:.02em;margin-bottom:6px}
.verdict .v-expl{font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;opacity:.95}
.verdict .v-dir{font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;
  margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.3);font-weight:600}
.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;
  overflow:hidden;margin-bottom:24px}
.stat{background:var(--card);padding:15px 16px}
.stat .l{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:5px}
.stat .v{font-size:25px;font-weight:700}
.stat .v small{font-size:13px;font-weight:400;color:var(--muted)}
.fig{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:14px;margin:16px 0}
.fig h3{font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:10px}
.fig img{width:100%;height:auto;display:block;border-radius:4px}
.method{font-family:'Helvetica Neue',Arial,sans-serif;font-size:12.5px;color:var(--muted);
  background:var(--card);border:1px solid var(--line);border-left:4px solid var(--ink);
  border-radius:6px;padding:16px 18px;margin-top:22px}
.method b{color:var(--ink)}
.sig-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:12px}
.sig-card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:18px 16px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,.05)}
.sig-tier{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}
.sig-pill{display:inline-block;color:#fff;font-family:'Helvetica Neue',Arial,sans-serif;
  font-weight:800;font-size:20px;letter-spacing:.01em;padding:10px 18px;border-radius:8px;
  min-width:160px}
.sig-meta{font-family:'Helvetica Neue',Arial,sans-serif;font-size:12px;color:var(--muted);
  margin-top:12px}
.sig-foot{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11.5px;color:var(--muted);
  background:var(--card);border:1px solid var(--line);border-left:4px solid var(--ink);
  border-radius:6px;padding:13px 16px;margin-bottom:24px;line-height:1.55}
.sig-foot i{color:var(--ink);font-style:normal;font-weight:600}
.foot{border-top:3px double var(--ink);margin-top:40px;padding-top:14px;
  font-family:'Helvetica Neue',Arial,sans-serif;font-size:11px;color:var(--muted);text-align:center}
.pos{color:#15803d}.neg{color:#b91c1c}
</style>"""


def fmt_pct(x):
    cls = 'pos' if x > 0 else ('neg' if x < 0 else '')
    return f'<span class="{cls}">{x:+.2f}%</span>'


def build_html(results):
    now = datetime.now().strftime('%B %d, %Y · %H:%M')
    blocks = ""
    for r in results:
        if r is None:
            continue
        res = r['res']
        sig_txt = f"p = {res.p_value:.4f}"
        blocks += f"""
  <section class="ticker-block">
    <div class="tk-head">
      <span class="tk-sym">{r['ticker']}</span>
      <span class="tk-name">{r['name']}</span>
    </div>

    <div class="price-card">
      <div class="pc-main">
        <div class="pc-label">Current Price</div>
        <div class="pc-price">${r['cur']:,.2f}</div>
        <div class="pc-change {('up' if r['chg'] >= 0 else 'down')}">{r['chg']:+.2f}% today</div>
      </div>
      <div class="pc-stats">
        <div class="pc-stat"><span>52-Week High</span><b>${r['hi52']:,.2f}</b></div>
        <div class="pc-stat"><span>52-Week Low</span><b>${r['lo52']:,.2f}</b></div>
        <div class="pc-stat"><span>VIX</span><b>{r['cur_vix']:.1f}</b></div>
      </div>
      <div class="pc-range">
        <div class="pc-range-label">52-week range</div>
        <div class="pc-bar">
          <div class="pc-bar-fill" style="width:{r['pos52']:.0f}%"></div>
          <div class="pc-bar-dot" style="left:{r['pos52']:.0f}%"></div>
        </div>
        <div class="pc-range-ends"><span>${r['lo52']:,.0f}</span><span>${r['hi52']:,.0f}</span></div>
      </div>
    </div>

    <div class="verdict" style="background:{r['color']}">
      <div class="v-label">{r['label']}</div>
      <div class="v-expl">{r['expl']}</div>
      <div class="v-dir">{direction_text(res.latest_pred)}</div>
    </div>

    <div class="sig-grid">
      <div class="sig-card">
        <div class="sig-tier">NOW · Technical</div>
        <div class="sig-pill" style="background:{r['sig']['tech_color']}">{r['sig']['tech_action']}</div>
        <div class="sig-meta">RSI {r['sig']['rsi']:.0f} · 20d mom {r['sig']['mom20']*100:+.1f}%</div>
      </div>
      <div class="sig-card">
        <div class="sig-tier">NOW · Model</div>
        <div class="sig-pill" style="background:{r['sig']['now_color']}">{r['sig']['now_action']}</div>
        <div class="sig-meta">latest model call: {r['sig']['now_pred']*100:+.1f}%</div>
      </div>
      <div class="sig-card">
        <div class="sig-tier">{PREDICTION_HORIZON}-DAY · Forecast</div>
        <div class="sig-pill" style="background:{r['sig']['fc_color']}">{r['sig']['fc_action']}</div>
        <div class="sig-meta">forward est: {r['sig']['fc_pred']*100:+.1f}%</div>
      </div>
    </div>
    <div class="sig-foot" style="border-left-color:{r['sig']['reliability_color']}">
      <b style="color:{r['sig']['reliability_color']}">{r['sig']['reliability']}.</b>
      &nbsp;<i>NOW · Technical</i> reads today's trend/momentum/RSI (model-independent);
      <i>NOW · Model</i> is the model's nearest-day call; <i>{PREDICTION_HORIZON}-day · Forecast</i>
      is its forward directional estimate. Signals are shown raw — the reliability note
      states whether the model earned a significant edge in backtesting. Educational only, not advice.
    </div>

    <div class="stat-row">
      <div class="stat"><div class="l">Directional Acc.</div><div class="v">{res.dir_acc:.1f}<small>%</small></div></div>
      <div class="stat"><div class="l">Naive Baseline</div><div class="v">{res.naive:.1f}<small>%</small></div></div>
      <div class="stat"><div class="l">Edge</div><div class="v">{fmt_pct(res.edge)}</div></div>
      <div class="stat"><div class="l">Significance</div><div class="v" style="font-size:18px">{sig_txt}</div></div>
      <div class="stat"><div class="l">Info Coefficient</div><div class="v">{res.ic:+.3f}</div></div>
    </div>
    <div class="stat-row">
      <div class="stat"><div class="l">Strategy (net)</div><div class="v">{fmt_pct(res.strat_return)}</div></div>
      <div class="stat"><div class="l">Buy &amp; Hold</div><div class="v">{fmt_pct(res.bh_return)}</div></div>
      <div class="stat"><div class="l">Sharpe</div><div class="v">{res.sharpe:.2f}</div></div>
      <div class="stat"><div class="l">Max Drawdown</div><div class="v">{res.max_dd:.1f}<small>%</small></div></div>
      <div class="stat"><div class="l">Test Obs / Trades</div><div class="v">{res.n_obs}<small>/{res.n_trades}</small></div></div>
    </div>

    <div class="fig"><h3>Price History &amp; 21-Day Forward Forecast</h3><img src="{r['img_price']}"></div>
    <div class="fig"><h3>Moving Averages &amp; Bollinger Bands</h3><img src="{r['img_bollinger']}"></div>
    <div class="fig"><h3>Drawdown from Peak (Risk View)</h3><img src="{r['img_drawdown']}"></div>
    <div class="fig"><h3>Equity Curve (net of {COST_BPS:.0f}bps costs)</h3><img src="{r['img_equity']}"></div>
    <div class="fig"><h3>Diagnostics</h3><img src="{r['img_diag']}"></div>
    <div class="fig"><h3>Monthly Returns Heatmap (Seasonality)</h3><img src="{r['img_heatmap']}"></div>
  </section>"""

    if not blocks:
        blocks = '<p style="text-align:center;color:#b91c1c">No tickers could be analyzed.</p>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MultiStock v24 — Honest Backtester</title>{CSS}</head>
<body><div class="wrap">
  <header class="masthead">
    <div class="kicker">Quantitative Research · Walk-Forward Evaluation</div>
    <h1>MultiStock v24</h1>
    <div class="sub">Ridge regression · {YEARS_HISTORY}-year history · {PREDICTION_HORIZON}-day horizon ·
      {N_WALK_FOLDS}-fold walk-forward · purged for overlap leakage · generated {now}</div>
  </header>

  <div class="disclaimer">
    ⚠ Educational / research tool — NOT investment advice. Backtested results are
    hypothetical, exclude taxes and real fill dynamics, and never guarantee future
    performance. The "verdict" reflects statistical evidence in historical data only.
  </div>

  {blocks}

  <div class="method">
    <b>Methodology.</b> Features use only past data. The model is evaluated with
    expanding-window walk-forward folds; each fold trains only on observations whose
    {PREDICTION_HORIZON}-day forward label was realized before the test window begins
    (purging removes overlapping-horizon leakage). Directional accuracy is compared to a
    naive majority-class baseline via a one-sided binomial test; the information coefficient
    is the Spearman rank correlation between predicted and realized returns. The equity curve
    uses <b>non-overlapping</b> holding periods and subtracts {COST_BPS:.0f} bps round-trip
    costs, so it reflects tradable performance rather than inflated overlapping returns.
  </div>

  <div class="foot">MultiStock v24 · Honest Quantitative Backtester · {now}</div>
</div></body></html>"""


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 62)
    print("  MULTI-STOCK v24 — HONEST QUANTITATIVE BACKTESTER")
    print("=" * 62)
    vix = download_vix()
    print(f"  VIX history: {0 if vix is None else len(vix)} days")

    results = []
    for tk, nm in TICKERS.items():
        try:
            results.append(analyze(tk, nm, vix))
        except Exception as e:
            print(f"   error on {tk}: {e}")
            results.append(None)

    html = build_html(results)
    out = "MultiStock_v24.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report -> {out}")
    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
