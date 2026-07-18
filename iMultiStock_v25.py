# ============================================================
#  MULTI-STOCK v25 — HONEST QUANTITATIVE BACKTESTER
# ============================================================
#  v24 -> v25 CHANGELOG (four statistical corrections)
#
#  [FIX 1] EFFECTIVE SAMPLE SIZE IN SIGNIFICANCE TESTS
#      The H-day target overlaps: consecutive observations share ~(H-1)/H
#      of the same price move, so they are NOT independent. v24 fed the raw
#      count into binomtest / spearmanr, making p-values optimistic by one
#      to two orders of magnitude. v25 deflates the sample to
#      n_eff = n / PREDICTION_HORIZON before every test.
#
#  [FIX 2] NAIVE BASELINE COMES FROM TRAIN, NOT TEST
#      v24 computed the majority class from the TEST labels — a look-ahead
#      that made the model compete against a threshold it could not know.
#      v25 derives, per fold, the majority direction and its expected hit
#      rate from TRAIN ONLY, then measures how that rule actually performed
#      out-of-sample. The binomial null probability (naive_p0) is now
#      strictly ex-ante.
#
#  [FIX 3] MULTIPLE-COMPARISON CONTROL (BENJAMINI-HOCHBERG FDR)
#      Screening 50+ tickers at p<0.05 yields 2-3 "significant" names by
#      pure chance, and the leaderboard sorts them straight to the top.
#      v25 collects every ticker's p-value, applies BH-FDR at q = FDR_Q,
#      and only then sets `significant`. A per-ticker q-value is reported.
#
#  [FIX 4] BENCHMARK-RELATIVE (DEMEANED) TARGET
#      v24 predicted raw returns over a 6-year semiconductor bull market,
#      so a long-biased model looked skilful while only harvesting beta.
#      v25 predicts EXCESS log return vs each ticker's benchmark. The
#      equity curve is then a dollar-neutral long/short pair (costs charged
#      on both legs), and Buy & Hold becomes the stock's relative
#      performance — so what is measured is alpha, not beta.
#      Set BENCHMARK_NEUTRAL = False to recover v24's absolute-return mode.
#
#  Also added: Sharpe standard error (a 6-year sample cannot distinguish
#  Sharpe 0.5 from 0.0), n_eff and q-value surfaced in the report.
#
#  Retained from v24: walk-forward expanding windows, purged training,
#  non-overlapping holding periods, transaction costs, leakage-safe news
#  features, adjusted close.
#
#  DISCLAIMER: Educational/research tool. NOT investment advice.
#  Past backtested performance does not guarantee future results.
# ============================================================

from __future__ import annotations   # allow "X | None" type hints on Python 3.7–3.9

import warnings
import os
import time
import hashlib
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

# ============================================================
#  NEWS FEATURES  —  leakage-safe sentiment/buzz inputs
# ============================================================
#  Turns a daily news stream (sentiment + article volume) into two
#  model features aligned to trading days:
#     news_sent : exponentially-decayed average sentiment
#     news_buzz : log of exponentially-decayed article count
#
#  LEAKAGE SAFETY (the whole point): for any trading day t, the value
#  uses ONLY news from days STRICTLY BEFORE t. Enforced by a .shift(1)
#  on a continuous daily calendar before mapping onto trading days, so
#  even intraday/after-hours timing can never bleed forward.
#
#  DATA SOURCE: GDELT (free, timestamped, 2017-present). Gives a daily
#  average "tone" (sentiment proxy) and a daily article volume for a
#  query. Cached to disk so backtests are repeatable. GDELT matches a
#  query STRING (company name), not a ticker, so matching is approximate
#  — prefer a specific phrase query like '"NVIDIA Corporation"'.
#  Requires the `requests` package for the fetch step only.
# ============================================================
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def build_news_features(trading_dates, news_daily, halflife: float = 3.0):
    """Map a daily news stream onto trading days WITHOUT look-ahead.

    news_daily: DataFrame indexed by calendar day with columns
    ['sent','buzz'] (or None). Returns DataFrame indexed by
    trading_dates with columns ['news_sent','news_buzz'], never NaN
    (no-prior-news days are 0.0 so _make_xy keeps the row).
    """
    out = pd.DataFrame(index=trading_dates,
                       data={'news_sent': 0.0, 'news_buzz': 0.0})
    if news_daily is None or len(news_daily) == 0:
        return out

    nd = news_daily.copy()
    nd.index = pd.to_datetime(nd.index).normalize()
    nd = nd[~nd.index.duplicated(keep='last')].sort_index()
    for col in ('sent', 'buzz'):
        if col not in nd.columns:
            nd[col] = 0.0

    start = min(nd.index.min(), trading_dates.min())
    end = max(nd.index.max(), trading_dates.max())
    cal = pd.date_range(start, end, freq='D')

    sent = nd['sent'].reindex(cal).fillna(0.0)   # no-news day -> neutral
    buzz = nd['buzz'].reindex(cal).fillna(0.0)   # no-news day -> 0 articles

    # Exponential decay, then SHIFT(1): trading day t sees news through
    # day t-1 only. This single shift is what guarantees no look-ahead.
    sent_dec = sent.ewm(halflife=halflife, adjust=False).mean().shift(1)
    buzz_dec = buzz.ewm(halflife=halflife, adjust=False).mean().shift(1)

    out['news_sent'] = (sent_dec.reindex(trading_dates, method='ffill')
                        .fillna(0.0).values)
    out['news_buzz'] = (np.log1p(buzz_dec.reindex(trading_dates, method='ffill')
                        .fillna(0.0)).values)
    return out


def _news_cache_path(cache_dir: str, query: str, start: str, end: str) -> str:
    key = hashlib.md5(f"{query}|{start}|{end}".encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in query)[:24]
    return os.path.join(cache_dir, f"gdelt_{safe}_{key}.parquet")


def _gdelt_timeline(query: str, mode: str, start: str, end: str) -> pd.DataFrame:
    """One GDELT timeline call -> DataFrame[date,value]. Needs `requests`."""
    import requests
    params = dict(query=query, mode=mode, format="json",
                  startdatetime=start, enddatetime=end, timelinesmooth="0")
    r = requests.get(GDELT_DOC, params=params, timeout=30)
    r.raise_for_status()
    series = r.json().get("timeline", [])
    if not series:
        return pd.DataFrame(columns=["value"])
    recs = [(pd.to_datetime(row["date"]).normalize(), float(row.get("value", 0.0)))
            for row in series[0].get("data", [])]
    return pd.DataFrame(recs, columns=["date", "value"]).set_index("date")


def fetch_news_gdelt_timeline(query: str, start: str, end: str,
                              cache_dir: str = "news_cache",
                              pause: float = 1.0) -> pd.DataFrame:
    """Daily news sentiment + volume for `query` from GDELT, cached to disk.

    query : GDELT search string, e.g. '"NVIDIA Corporation"'.
    start, end : 'YYYYMMDD'. Returns DataFrame[day] with ['sent','buzz'].
    A repeat call with the same args reads the cache (no network).
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _news_cache_path(cache_dir, query, start, end)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    sd, ed = f"{start}000000", f"{end}235959"
    try:
        tone = _gdelt_timeline(query, "TimelineTone", sd, ed)
        time.sleep(pause)
        vol = _gdelt_timeline(query, "TimelineVolRaw", sd, ed)
    except Exception as e:
        print(f"   [news] GDELT fetch failed for {query!r}: {e}")
        return pd.DataFrame(columns=["sent", "buzz"])

    df = pd.DataFrame(index=tone.index.union(vol.index))
    df["sent"] = tone["value"].reindex(df.index).fillna(0.0) if len(tone) else 0.0
    df["buzz"] = vol["value"].reindex(df.index).fillna(0.0) if len(vol) else 0.0
    df = df.sort_index()
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"))
    return df


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TICKERS = {
    # Bellek / depolama
    'SNDK':  {'name': 'SanDisk Corporation',        'benchmark': 'SOXX'},
    'MU':    {'name': 'Micron Technology',          'benchmark': 'SOXX'},
    'WDC':   {'name': 'Western Digital',            'benchmark': 'SOXX'},
    'STX':   {'name': 'Seagate Technology',         'benchmark': 'SOXX'},
    # CPU
    'INTC':  {'name': 'Intel Corporation',          'benchmark': 'SOXX'},
    'AMD':   {'name': 'Advanced Micro Devices',     'benchmark': 'SOXX'},
    # Küçük-cap yarı iletken tedarikçileri
    'AXTI':  {'name': 'AXT Inc.',                   'benchmark': 'SOXX'},
    'AEHR':  {'name': 'Aehr Test Systems',          'benchmark': 'SOXX'},
    'ALAB':  {'name': 'Astera Labs',                'benchmark': 'SOXX'},
    'CRDO':  {'name': 'Credo Technology',           'benchmark': 'SOXX'},
    # Çip / ekipman / IP
    'AVGO':  {'name': 'Broadcom Inc.',              'benchmark': 'SOXX'},
    'MRVL':  {'name': 'Marvell Technology',         'benchmark': 'SOXX'},
    'AMAT':  {'name': 'Applied Materials',          'benchmark': 'SOXX'},
    'ARM':   {'name': 'Arm Holdings',               'benchmark': 'SOXX'},
    'MPWR':  {'name': 'Monolithic Power Systems',   'benchmark': 'SOXX'},
    'NVDA':  {'name': 'NVIDIA Corporation',         'benchmark': 'SOXX'},
    # Döküm / litografi / ekipman
    'TSM':   {'name': 'Taiwan Semiconductor',       'benchmark': 'SOXX'},
    'ASML':  {'name': 'ASML Holding',               'benchmark': 'SOXX'},
    'LRCX':  {'name': 'Lam Research',               'benchmark': 'SOXX'},
    'KLAC':  {'name': 'KLA Corporation',            'benchmark': 'SOXX'},
    # EDA
    'SNPS':  {'name': 'Synopsys Inc.',              'benchmark': 'SOXX'},
    'CDNS':  {'name': 'Cadence Design Systems',     'benchmark': 'SOXX'},
    # Diğer çip üreticileri
    'QCOM':  {'name': 'Qualcomm Inc.',              'benchmark': 'SOXX'},
    'NXPI':  {'name': 'NXP Semiconductors',         'benchmark': 'SOXX'},
    # Optik / ağ
    'COHR':  {'name': 'Coherent Corp.',             'benchmark': 'SOXX'},
    'LITE':  {'name': 'Lumentum Holdings',          'benchmark': 'SOXX'},
    'ANET':  {'name': 'Arista Networks',            'benchmark': 'SOXX'},
    # Veri merkezi donanımı / güç
    'VRT':   {'name': 'Vertiv Holdings',            'benchmark': 'SOXX'},
    'SMCI':  {'name': 'Super Micro Computer',       'benchmark': 'SOXX'},
    'DELL':  {'name': 'Dell Technologies',          'benchmark': 'SOXX'},
    'CLS':   {'name': 'Celestica Inc.',             'benchmark': 'SOXX'},
    # AI bulut / hyperscale
    'CRWV':  {'name': 'CoreWeave Inc.',             'benchmark': 'SOXX'},
    'NBIS':  {'name': 'Nebius Group',               'benchmark': 'SOXX'},
    'ORCL':  {'name': 'Oracle Corporation',         'benchmark': 'SOXX'},
    'RXT':   {'name': 'Rackspace Technology',       'benchmark': 'SOXX'},
    'MSFT':  {'name': 'Microsoft Corporation',      'benchmark': 'SOXX'},
    'META':  {'name': 'Meta Platforms',             'benchmark': 'SOXX'},
    'AMZN':  {'name': 'Amazon.com Inc.',            'benchmark': 'SOXX'},
    # Yazılım / uygulama / güvenlik
    'PLTR':  {'name': 'Palantir Technologies',      'benchmark': 'SOXX'},
    'APP':   {'name': 'AppLovin Corporation',       'benchmark': 'SOXX'},
    'CRWD':  {'name': 'CrowdStrike Holdings',       'benchmark': 'SOXX'},
    'PANW':  {'name': 'Palo Alto Networks',         'benchmark': 'SOXX'},
    'SNOW':  {'name': 'Snowflake Inc.',             'benchmark': 'SOXX'},
    'NET':   {'name': 'Cloudflare Inc.',            'benchmark': 'SOXX'},
    # Büyük teknoloji
    'GOOGL': {'name': 'Alphabet Inc.',              'benchmark': 'SOXX'},
    'AAPL':  {'name': 'Apple Inc.',                 'benchmark': 'SOXX'},
    # Üretim ekipmanı / test / paketleme
    'TER':   {'name': 'Teradyne, Inc.',             'benchmark': 'SOXX'},
    'AMKR':  {'name': 'Amkor Technology',           'benchmark': 'SOXX'},
    'TXN':   {'name': 'Texas Instruments',          'benchmark': 'SOXX'},
    'ADI':   {'name': 'Analog Devices',             'benchmark': 'SOXX'},
    # Veri merkezi fiziksel altyapı / güç
    'ETN':   {'name': 'Eaton Corporation',          'benchmark': 'SPY'},
    'PWR':   {'name': 'Quanta Services',            'benchmark': 'SPY'},
    'GEV':   {'name': 'GE Vernova',                 'benchmark': 'SPY'},
    # Ağ / bağlantı donanımları
    'APH':   {'name': 'Amphenol Corporation',       'benchmark': 'SPY'},
    'GLW':   {'name': 'Corning Inc.',               'benchmark': 'SPY'},
    # Kurumsal AI yazılımı / danışmanlık
    'NOW':   {'name': 'ServiceNow, Inc.',           'benchmark': 'QQQ'},
    'ACN':   {'name': 'Accenture plc',              'benchmark': 'SPY'},
    'CRM':   {'name': 'Salesforce, Inc.',           'benchmark': 'QQQ'},
}

# !! SELECTION BIAS WARNING !!
# This list was assembled with hindsight ("2026'nın liderleri"). Every
# cross-sectional statistic below — especially the leaderboard — inherits
# that bias. The four fixes in v25 repair the STATISTICS, not the sample.
# For a bias-free study, reconstruct the universe as it was known at the
# START of the backtest window (e.g. SOXX constituents as of 2020) and
# include names that later died or were delisted.

DEFAULT_BENCHMARK = 'SPY'


def _ticker_meta(cfg):
    """Accept either a plain name string or a {'name':..., 'benchmark':...} dict."""
    if isinstance(cfg, dict):
        return cfg.get('name', ''), cfg.get('benchmark', DEFAULT_BENCHMARK)
    return str(cfg), DEFAULT_BENCHMARK


YEARS_HISTORY      = 6          # more data -> more walk-forward folds
PREDICTION_HORIZON = 7          # trading days ahead
LOOKBACK           = 20         # feature window length
RIDGE_ALPHA        = 5.0
N_WALK_FOLDS       = 6
MIN_TRAIN          = 250
COST_BPS           = 5.0        # round-trip cost per leg (commission + slippage), bps
SEED               = 42

# --- [FIX 4] BENCHMARK-RELATIVE TARGET -----------------------------------
# True  : model predicts EXCESS return vs the ticker's benchmark, and the
#         equity curve is a dollar-neutral long/short pair (stock vs index).
#         "Buy & Hold" becomes the stock's RELATIVE performance. This strips
#         out beta, so a rising sector no longer flatters the model.
#         Costs are charged on BOTH legs (see COST_LEGS below).
# False : v24 behaviour — absolute returns, long-only/short outright.
BENCHMARK_NEUTRAL = True

# --- [FIX 3] FALSE DISCOVERY RATE ----------------------------------------
# Screening N tickers means N chances to get lucky. Benjamini-Hochberg
# controls the expected proportion of false "edges" among the names we
# call significant. 0.10 is a common research setting; 0.05 is stricter.
FDR_Q = 0.10

# --- SELECTIVE TRADE FILTER ---
# Trade only when BOTH hold:
#   (1) High conviction: |prediction| above the CONVICTION_PCTL percentile of
#       the model's own TRAINING predictions (learned per-fold, no look-ahead).
#   (2) Agreement: model direction matches the technical posture.
# NOTE (v25): with BENCHMARK_NEUTRAL the model predicts RELATIVE return while
# _technical_score reads ABSOLUTE price posture, so condition (2) is a weaker
# fit than it was in v24. Treat filt_* metrics as exploratory.
USE_SELECTIVE_FILTER = True
CONVICTION_PCTL      = 60

# --- NEWS FEATURES (model input, leakage-safe) ---
USE_NEWS_FEATURES = True
NEWS_HALFLIFE     = 3.0

# --- ABSOLUTE PATHS FOR CRON JOB & WORDPRESS INTEGRATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_CACHE_DIR = os.path.join(BASE_DIR, "news_cache")

# WORDPRESS PUBLIC_HTML YOLUNU BURAYA YAZIN
# Örnek: /home/kullanici_adiniz/public_html
WP_OUTPUT_DIR = "https://ozgursaygi.com/MultiStock_v25.html" 

FEATURE_COLS = ['mom5', 'mom20', 'vol20', 'vol60', 'sma_ratio', 'rsi14', 'dist_high', 'vix']
if USE_NEWS_FEATURES:
    FEATURE_COLS = FEATURE_COLS + ['news_sent', 'news_buzz']

# Costs: neutral mode trades two legs (stock + index hedge).
COST_LEGS = 2 if BENCHMARK_NEUTRAL else 1


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


def download_benchmarks(tickers: dict) -> dict:
    """[FIX 4] Fetch every distinct benchmark once, up front."""
    wanted = sorted({_ticker_meta(cfg)[1] for cfg in tickers.values()
                     if _ticker_meta(cfg)[1]})
    out = {}
    for b in wanted:
        s = download_close(b)
        if s is None or len(s) < 300:
            print(f"  benchmark {b}: FAILED (relative mode will fall back to absolute)")
            out[b] = None
        else:
            print(f"  benchmark {b}: {len(s)} days")
            out[b] = s
    return out


# ============================================================
# FEATURES (strictly backward-looking)
# ============================================================
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(close: pd.Series, vix: pd.Series | None,
                   news_daily: pd.DataFrame | None = None,
                   bench: pd.Series | None = None) -> tuple[pd.DataFrame, bool]:
    """Returns (features_df, relative_flag).

    relative_flag is True when the target is benchmark-demeaned [FIX 4];
    it is False when relative mode is off or the benchmark was unavailable.
    """
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

    # News inputs (leakage-safe; neutral 0.0 when news is off/unavailable).
    if USE_NEWS_FEATURES:
        news_feat = build_news_features(df.index, news_daily,
                                        halflife=NEWS_HALFLIFE)
        df['news_sent'] = news_feat['news_sent']
        df['news_buzz'] = news_feat['news_buzz']

    # ---- [FIX 4] TARGET: excess (benchmark-demeaned) forward log return ----
    # v24: target = log(P[t+H]/P[t])                      -> contains beta
    # v25: target = log(P[t+H]/P[t]) - log(B[t+H]/B[t])   -> alpha only
    fwd_stock = np.log(close.shift(-PREDICTION_HORIZON) / close)
    relative = False
    if BENCHMARK_NEUTRAL and bench is not None and len(bench):
        b = bench.reindex(df.index).ffill()
        fwd_bench = np.log(b.shift(-PREDICTION_HORIZON) / b)
        # Only demean where the benchmark is actually observed.
        if b.notna().sum() > 0.9 * len(b):
            df['target'] = fwd_stock - fwd_bench
            df['bench_fwd'] = fwd_bench
            relative = True
        else:
            df['target'] = fwd_stock
    else:
        df['target'] = fwd_stock

    # Last PREDICTION_HORIZON rows are unknown (NaN) -> dropped in _make_xy.
    return df, relative


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
@dataclass
class FoldResult:
    dates: list
    y_true: np.ndarray
    y_pred: np.ndarray
    conv_thr: float = 0.0          # per-fold conviction threshold (from TRAIN preds)
    naive_dir: float = 1.0         # [FIX 2] majority direction learned from TRAIN
    naive_p0: float = 0.5          # [FIX 2] that rule's expected hit rate, from TRAIN


@dataclass
class BacktestResult:
    ok: bool = False
    n_obs: int = 0
    n_eff: int = 0                 # [FIX 1] independent-equivalent observations
    relative: bool = False         # [FIX 4] target is benchmark-demeaned
    benchmark: str = ''
    dir_acc: float = 0.0
    naive: float = 0.0             # [FIX 2] realized OOS accuracy of the train-derived rule
    naive_p0: float = 0.5          # [FIX 2] ex-ante null probability from TRAIN
    edge: float = 0.0
    p_value: float = 1.0
    q_value: float = 1.0           # [FIX 3] BH-adjusted p-value across tickers
    significant: bool = False      # [FIX 3] set only after FDR control
    ic: float = 0.0
    ic_p: float = 1.0
    strat_return: float = 0.0
    bh_return: float = 0.0
    sharpe: float = 0.0
    sharpe_se: float = 0.0         # honesty: Sharpe is very imprecise on 6y
    max_dd: float = 0.0
    n_trades: int = 0
    folds: list = field(default_factory=list)
    all_dates: list = field(default_factory=list)
    all_true: np.ndarray = None
    all_pred: np.ndarray = None
    all_conv: np.ndarray = None
    all_tech: np.ndarray = None
    all_naive: np.ndarray = None   # [FIX 2] per-point naive direction from train
    strat_curve: np.ndarray = None
    bh_curve: np.ndarray = None
    latest_pred: float = 0.0
    # --- selective-filter results (filled when USE_SELECTIVE_FILTER) ---
    filt_used: bool = False
    filt_return: float = 0.0
    filt_sharpe: float = 0.0
    filt_max_dd: float = 0.0
    filt_trades: int = 0
    filt_dir_acc: float = 0.0
    filt_curve: np.ndarray = None


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


def walk_forward(feat: pd.DataFrame, relative: bool = False,
                 benchmark: str = '') -> BacktestResult:
    np.random.seed(SEED)
    res = BacktestResult()
    res.relative = relative
    res.benchmark = benchmark
    X, y, dates = _make_xy(feat)
    if len(X) < MIN_TRAIN + N_WALK_FOLDS * 10:
        return res

    n = len(X)
    test_block = (n - MIN_TRAIN) // N_WALK_FOLDS
    if test_block < 5:
        return res

    folds = []
    last_model = last_xs = last_ys = None
    last_clip = None
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

        # ---- [FIX 2] NAIVE BASELINE FROM TRAIN ONLY --------------------
        # v24 read the majority class off the TEST labels (look-ahead).
        # The honest baseline is the rule a trader could have written at
        # the fold boundary: "always bet the direction that dominated the
        # training data". Its expected hit rate (naive_p0) is the null we
        # test against; how it ACTUALLY did on test is measured later.
        tr_pos_rate = float((ytr > 0).mean())
        naive_dir = 1.0 if tr_pos_rate >= 0.5 else -1.0
        naive_p0 = float(max(tr_pos_rate, 1.0 - tr_pos_rate))
        # ----------------------------------------------------------------

        # WINSORIZE training target: clip extreme outliers (1st/99th pct).
        lo_w, hi_w = np.percentile(ytr, [1, 99])
        ytr_w = np.clip(ytr, lo_w, hi_w)

        # Plausible output range = a bit beyond the realized training spread.
        clip_lo, clip_hi = np.percentile(ytr, [2, 98])
        span = clip_hi - clip_lo
        clip_lo -= 0.25 * span
        clip_hi += 0.25 * span

        xs, ysc = StandardScaler(), StandardScaler()
        Xtr_s = xs.fit_transform(Xtr)
        ytr_s = ysc.fit_transform(ytr_w.reshape(-1, 1)).ravel()
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(Xtr_s, ytr_s)
        pred = ysc.inverse_transform(model.predict(xs.transform(Xte)).reshape(-1, 1)).ravel()
        pred = np.clip(pred, clip_lo, clip_hi)

        # Conviction threshold learned ONLY from training predictions.
        tr_pred = ysc.inverse_transform(model.predict(Xtr_s).reshape(-1, 1)).ravel()
        conv_thr = float(np.percentile(np.abs(tr_pred), CONVICTION_PCTL))

        folds.append(FoldResult(dte, yte, pred, conv_thr=conv_thr,
                                naive_dir=naive_dir, naive_p0=naive_p0))
        last_model, last_xs, last_ys = model, xs, ysc
        last_clip = (clip_lo, clip_hi)

    if not folds:
        return res

    all_true = np.concatenate([f.y_true for f in folds])
    all_pred = np.concatenate([f.y_pred for f in folds])
    all_dates = [d for f in folds for d in f.dates]
    all_conv = np.concatenate([np.full(len(f.y_pred), f.conv_thr) for f in folds])
    all_naive = np.concatenate([np.full(len(f.y_pred), f.naive_dir) for f in folds])

    # technical score per test date (model-independent), aligned to all_dates
    all_tech = np.array([_technical_score(feat.loc[d]) if d in feat.index else 0.0
                         for d in all_dates])

    res.ok = True
    res.folds = folds
    res.all_true, res.all_pred, res.all_dates = all_true, all_pred, all_dates
    res.all_conv, res.all_tech, res.all_naive = all_conv, all_tech, all_naive
    res.n_obs = len(all_true)

    nz = all_true != 0
    n_nz = int(nz.sum())
    correct = int((np.sign(all_pred) == np.sign(all_true))[nz].sum())
    res.dir_acc = correct / n_nz * 100

    # [FIX 2] realized accuracy of the TRAIN-derived naive rule, out-of-sample
    naive_correct = int((np.sign(all_true) == all_naive)[nz].sum())
    res.naive = naive_correct / n_nz * 100
    # ex-ante null probability: fold-size-weighted average of train hit rates
    w = np.concatenate([np.full(len(f.y_pred), f.naive_p0) for f in folds])
    res.naive_p0 = float(w[nz].mean())
    res.edge = res.dir_acc - res.naive

    # ---- [FIX 1] EFFECTIVE SAMPLE SIZE ---------------------------------
    # Targets overlap by H-1 days, so n_nz observations carry roughly
    # n_nz / H independent pieces of information. Testing on the raw count
    # is the single biggest source of false confidence in v24.
    res.n_eff = max(int(n_nz // PREDICTION_HORIZON), 1)
    correct_eff = int(round(res.dir_acc / 100.0 * res.n_eff))
    correct_eff = min(max(correct_eff, 0), res.n_eff)
    p0 = min(max(res.naive_p0, 1e-6), 1 - 1e-6)
    res.p_value = float(stats.binomtest(correct_eff, res.n_eff, p0,
                                        alternative='greater').pvalue)
    # ---------------------------------------------------------------------

    if n_nz > 5:
        ic, _ = stats.spearmanr(all_pred, all_true)
        res.ic = float(ic) if np.isfinite(ic) else 0.0
        # [FIX 1] re-derive the IC p-value on n_eff, not n
        if res.n_eff > 3 and abs(res.ic) < 1.0:
            t_stat = res.ic * np.sqrt((res.n_eff - 2) / (1 - res.ic ** 2))
            res.ic_p = float(2 * stats.t.sf(abs(t_stat), res.n_eff - 2))
        else:
            res.ic_p = 1.0

    _equity(res, all_dates, all_true, all_pred)
    if USE_SELECTIVE_FILTER:
        _equity_filtered(res, all_dates, all_true, all_pred, all_conv, all_tech)

    # Live prediction for the most recent point (forward-looking, unrealized)
    live = _latest_live_window(feat)
    if live is not None and last_model is not None:
        live_s = last_xs.transform(live.reshape(1, -1))
        lp = float(last_ys.inverse_transform(
            last_model.predict(live_s).reshape(-1, 1)).ravel()[0])
        if last_clip is not None:
            lp = float(np.clip(lp, last_clip[0], last_clip[1]))
        res.latest_pred = lp
    return res


def _equity(res: BacktestResult, dates, y_true, y_pred):
    """Non-overlapping holds, net of costs.

    [FIX 4] When res.relative is True, y_true is EXCESS return, so each
    'trade' is a dollar-neutral pair (long stock / short benchmark, or the
    reverse) and costs are charged on both legs via COST_LEGS. Buy & Hold
    then means "hold the stock, short the benchmark" = relative performance.
    """
    order = np.argsort(dates)
    yt = y_true[order]; yp = y_pred[order]
    cost = COST_BPS / 1e4 * COST_LEGS
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
    n_per = len(strat)
    if n_per > 1 and strat.std() > 0:
        sp = strat.mean() / strat.std()          # per-period Sharpe
        ann = np.sqrt(252 / PREDICTION_HORIZON)
        res.sharpe = sp * ann
        # Lo (2002) approximation for the standard error of the Sharpe ratio.
        # With ~170 independent periods this is large — that is the point.
        res.sharpe_se = float(np.sqrt((1 + 0.5 * sp ** 2) / n_per) * ann)
    eq = np.cumprod(1 + strat)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    res.max_dd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
    res.strat_curve = eq
    res.bh_curve = np.cumprod(1 + bh)


def _equity_filtered(res, dates, y_true, y_pred, conv, tech):
    """Selective strategy: trade only on high conviction AND technical agreement."""
    order = np.argsort(dates)
    yt = y_true[order]; yp = y_pred[order]
    cv = conv[order]; tc = tech[order]
    cost = COST_BPS / 1e4 * COST_LEGS

    rets = []
    n_trades = 0
    traded_correct = 0
    i = 0
    while i < len(yt):
        pred = yp[i]
        high_conv = abs(pred) >= cv[i]
        agree = (np.sign(pred) == np.sign(tc[i])) and np.sign(pred) != 0
        if high_conv and agree:
            sig = np.sign(pred)
            g = sig * (np.exp(yt[i]) - 1) - cost
            n_trades += 1
            if np.sign(pred) == np.sign(yt[i]):
                traded_correct += 1
        else:
            g = 0.0                       # stay flat / in cash this period
        rets.append(g)
        i += PREDICTION_HORIZON

    rets = np.array(rets)
    res.filt_used = True
    res.filt_trades = n_trades
    res.filt_return = (np.prod(1 + rets) - 1) * 100
    res.filt_dir_acc = (traded_correct / n_trades * 100) if n_trades else 0.0
    if len(rets) > 1 and rets.std() > 0:
        res.filt_sharpe = rets.mean() / rets.std() * np.sqrt(252 / PREDICTION_HORIZON)
    eqf = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eqf) if len(eqf) else np.array([1.0])
    res.filt_max_dd = ((eqf - peak) / peak).min() * 100 if len(eqf) else 0.0
    res.filt_curve = eqf


# ============================================================
# [FIX 3] MULTIPLE COMPARISONS — BENJAMINI-HOCHBERG FDR
# ============================================================
def benjamini_hochberg(pvals, q: float = FDR_Q):
    """Return (reject_flags, adjusted_pvalues) controlling FDR at level q.

    Why this matters here: with 57 tickers and a 0.05 threshold you expect
    ~3 "significant" names even if every model is worthless — and the
    leaderboard promotes exactly those to the top. BH bounds the expected
    share of false discoveries among the names we DO call significant.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    # adjusted p (step-up, enforce monotonicity)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_adj = np.empty(n)
    out_adj[order] = adj
    return out_adj <= q, out_adj


# ============================================================
# VERDICT (honest, rule-based)
# ============================================================
def verdict(res: BacktestResult) -> tuple[str, str, str]:
    """Based on FDR-controlled significance [FIX 3], an ex-ante baseline
    [FIX 2], overlap-corrected tests [FIX 1] and benchmark-relative
    returns [FIX 4]. Refuses to claim an edge that isn't there."""
    if not res.ok:
        return ("INSUFFICIENT DATA", "#6b7280",
                "Not enough history to evaluate this model reliably.")
    bench_txt = (f"net of costs, measured against {res.benchmark}"
                 if res.relative else "net of costs")
    if res.significant and res.ic > 0.05 and res.strat_return > res.bh_return:
        return ("STATISTICALLY SIGNIFICANT EDGE", "#15803d",
                f"Survives FDR control at q={FDR_Q:g} across all screened tickers, "
                f"shows positive rank correlation with future returns, and beats "
                f"buy & hold {bench_txt} — using overlap-corrected tests and an "
                f"ex-ante baseline. Still no guarantee: one universe, one period.")
    if res.significant and res.ic > 0:
        return ("WEAK / BORDERLINE SIGNAL", "#a16207",
                "Survives FDR control, but the economic edge is thin and may not "
                "outlive live costs and regime change.")
    return ("NO RELIABLE EDGE", "#b91c1c",
            "After correcting for overlapping targets and for testing many "
            "tickers at once, this model does NOT beat an ex-ante naive baseline "
            "at a defensible level. On this data it provides no dependable "
            "directional edge. Do not trade it.")


def direction_text(res: BacktestResult) -> str:
    lp = res.latest_pred
    suffix = f" vs {res.benchmark}" if res.relative else ""
    if lp > 0.005:
        return f"Model leans UP{suffix} over next {PREDICTION_HORIZON}d (est. {lp*100:+.1f}%)"
    if lp < -0.005:
        return f"Model leans DOWN{suffix} over next {PREDICTION_HORIZON}d (est. {lp*100:+.1f}%)"
    return f"Model is roughly NEUTRAL{suffix} over next {PREDICTION_HORIZON}d ({lp*100:+.1f}%)"


def _technical_score(row) -> float:
    """Composite technical posture in [-1,1] from one feature row:
    trend (SMA ratio) + momentum (5/20d) + RSI tilt. Model-independent.
    NOTE: absolute posture — see the BENCHMARK_NEUTRAL caveat above."""
    mom20 = float(row.get('mom20', 0.0)); mom5 = float(row.get('mom5', 0.0))
    rsi = float(row.get('rsi14', 50.0)); sma_r = float(row.get('sma_ratio', 1.0))
    if not np.isfinite(mom20): mom20 = 0.0
    if not np.isfinite(mom5): mom5 = 0.0
    if not np.isfinite(rsi): rsi = 50.0
    if not np.isfinite(sma_r): sma_r = 1.0
    trend_term = np.tanh((sma_r - 1.0) * 25)
    mom_term   = np.tanh((mom20 + mom5) * 8)
    rsi_term   = np.tanh((rsi - 50) / 20)
    return float(np.clip(0.45 * trend_term + 0.40 * mom_term + 0.15 * rsi_term, -1, 1))


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
    """Three complementary signals, shown raw, plus one reliability tag.

      1) NOW · Technical  -> today's absolute technical posture (model-free)
      2) NOW · Model      -> model's most recent out-of-sample call
      3) H-DAY · Forecast -> model's forward directional call
                             (RELATIVE to benchmark when BENCHMARK_NEUTRAL)
    """
    if res.ok and res.all_pred is not None and len(res.all_pred) > 5:
        m_band = max(0.5 * float(np.std(res.all_pred)), 0.004)
    else:
        m_band = 0.005

    fc_pred = res.latest_pred
    fc_act, fc_col = _score_to_action(fc_pred, m_band)

    if res.ok and res.all_pred is not None and len(res.all_pred):
        now_model_pred = float(res.all_pred[-1])
    else:
        now_model_pred = fc_pred
    nm_act, nm_col = _score_to_action(now_model_pred, m_band)

    last = feat.iloc[-1]
    mom20 = float(last.get('mom20', 0.0))
    mom5  = float(last.get('mom5', 0.0))
    rsi   = float(last.get('rsi14', 50.0))
    sma_r = float(last.get('sma_ratio', 1.0))

    trend_term = np.tanh((sma_r - 1.0) * 25)
    mom_term   = np.tanh((mom20 + mom5) * 8)
    rsi_term   = np.tanh((rsi - 50) / 20)
    tech_score = float(np.clip(0.45 * trend_term + 0.40 * mom_term + 0.15 * rsi_term, -1, 1))
    tech_act, tech_col = _score_to_action(tech_score, 0.15)

    # reliability of the model-based signals — now FDR-aware [FIX 3]
    if not res.ok:
        rel, rel_col = "Model unverified (insufficient data)", "#6b7280"
    elif res.significant and res.ic > 0.05 and res.strat_return > res.bh_return:
        rel, rel_col = (f"Model backtest: survives FDR control (q={res.q_value:.3f})",
                        "#15803d")
    elif res.significant and res.ic > 0:
        rel, rel_col = (f"Model backtest: weak / borderline (q={res.q_value:.3f})",
                        "#a16207")
    else:
        rel, rel_col = (f"Model backtest: no reliable edge (q={res.q_value:.3f}) — "
                        f"informational only", "#b91c1c")

    return dict(
        tech_action=tech_act, tech_color=tech_col, tech_score=tech_score,
        rsi=rsi, mom20=mom20,
        now_action=nm_act, now_color=nm_col, now_pred=now_model_pred,
        fc_action=fc_act, fc_color=fc_col, fc_pred=fc_pred, band=m_band,
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
    rel = res.relative
    rel_tag = f" vs {res.benchmark}" if rel else ""

    ax.plot(close.index, close.values, lw=1.7, color='#1e3a5f',
            label='Actual price (adj close)', zorder=3)

    if res.ok and res.all_dates:
        t0 = min(res.all_dates)
        ax.axvspan(t0, close.index[-1], color='#fde68a', alpha=0.22,
                   label='Walk-forward test region')
        lab_up = f'Past pred: outperform{rel_tag}' if rel else 'Past pred: up'
        lab_dn = f'Past pred: underperform{rel_tag}' if rel else 'Past pred: down'
        up = [(d, close.loc[d]) for d, p in zip(res.all_dates, res.all_pred) if p > 0]
        dn = [(d, close.loc[d]) for d, p in zip(res.all_dates, res.all_pred) if p <= 0]
        if up:
            ax.scatter(*zip(*up), s=16, c='#16a34a', marker='^', alpha=0.45,
                       label=lab_up, zorder=4)
        if dn:
            ax.scatter(*zip(*dn), s=16, c='#dc2626', marker='v', alpha=0.45,
                       label=lab_dn, zorder=4)

    last_date = close.index[-1]
    last_price = float(close.iloc[-1])
    pred = res.latest_pred
    target = last_price * np.exp(pred)

    future_dates = pd.bdate_range(last_date, periods=PREDICTION_HORIZON + 1)
    path = last_price * np.exp(np.linspace(0, pred, len(future_dates)))

    up_fc = pred > 0
    fc_color = '#16a34a' if up_fc else ('#dc2626' if pred < 0 else '#6b7280')
    arrow = '^' if up_fc else ('v' if pred < 0 else '>')

    # [FIX 4] In relative mode the forecast is EXCESS return, so this dashed
    # path is NOT a price target — it is the implied path if the benchmark
    # were flat. Labelled explicitly rather than silently mis-drawn.
    fc_label = (f'Forecast next {PREDICTION_HORIZON}d: {pred*100:+.1f}%{rel_tag} '
                f'(relative — path assumes flat {res.benchmark})'
                if rel else
                f'Forecast next {PREDICTION_HORIZON}d ({pred*100:+.1f}%)')
    ax.plot(future_dates, path, lw=2.4, color=fc_color, ls='--',
            label=fc_label, zorder=5)

    if res.ok and res.all_true is not None and len(res.all_true) > 5:
        resid_std = float(np.std(res.all_true - res.all_pred))
    else:
        resid_std = abs(pred) + 0.02
    upper = last_price * np.exp(np.linspace(0, pred + resid_std, len(future_dates)))
    lower = last_price * np.exp(np.linspace(0, pred - resid_std, len(future_dates)))
    ax.fill_between(future_dates, lower, upper, color=fc_color, alpha=0.12,
                    label='Forecast uncertainty (1 sigma)', zorder=2)

    ax.scatter([last_date], [last_price], s=70, color='#1e3a5f', zorder=6)
    ax.scatter([future_dates[-1]], [target], s=110, color=fc_color,
               marker='*', zorder=6, edgecolor='white', linewidth=0.8)
    star_txt = (f'{arrow} {pred*100:+.1f}%{rel_tag}' if rel
                else f'{arrow} ${target:,.2f}')
    ax.annotate(star_txt,
                xy=(future_dates[-1], target),
                xytext=(8, 10 if up_fc else -16), textcoords='offset points',
                fontsize=11, fontweight='bold', color=fc_color)
    ax.annotate(f'Today ${last_price:,.2f}',
                xy=(last_date, last_price),
                xytext=(-98, -18), textcoords='offset points',
                fontsize=9.5, color='#1e3a5f')

    title = (f'Price History + {PREDICTION_HORIZON}-Day Forward Forecast'
             f'{" (relative to " + res.benchmark + ")" if rel else ""}')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8.5, ncol=2)
    lo = max(0, len(close) - 400)
    ax.set_xlim(close.index[lo], future_dates[-1])
    vis = close.values[lo:]
    ymin = min(vis.min(), lower.min()); ymax = max(vis.max(), upper.max())
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad)
    return _fig_to_data_uri(fig)


def plot_equity(res: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(15, 5))
    bh_label = (f'Buy & Hold (relative to {res.benchmark})'
                if res.relative else 'Buy & Hold')
    if res.ok and res.strat_curve is not None:
        ax.plot(res.strat_curve, lw=2, color='#ea580c', marker='o', ms=3,
                label=f'Strategy — all trades ({res.n_trades})')
        if res.filt_used and res.filt_curve is not None:
            ax.plot(res.filt_curve, lw=2, color='#7c3aed', marker='s', ms=3,
                    label=f'Selective filter ({res.filt_trades} trades)')
        ax.plot(res.bh_curve, lw=2, color='#1e3a5f', marker='o', ms=3, label=bh_label)
        ax.axhline(1.0, color='#666', ls='--', alpha=0.5)
    mode = (f'Benchmark-Neutral vs {res.benchmark}' if res.relative
            else 'Absolute Returns')
    ax.set_title(f'Equity Curve — Non-Overlapping Holds, Net of Costs ({mode})',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Growth of $1'); ax.set_xlabel('Period #')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=10)
    return _fig_to_data_uri(fig)


def plot_diagnostics(res: BacktestResult) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].bar(['Model', 'Naive baseline\n(from TRAIN)'], [res.dir_acc, res.naive],
                color=['#1e3a5f', '#9ca3af'], alpha=0.85)
    axes[0].axhline(50, color='#dc2626', ls='--', alpha=0.5, label='Coin flip (50%)')
    for i, v in enumerate([res.dir_acc, res.naive]):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[0].set_ylim(0, 100); axes[0].set_ylabel('Directional accuracy (%)')
    axes[0].set_title(f'Accuracy vs Ex-Ante Baseline  (n_eff={res.n_eff})',
                      fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25, axis='y')

    if res.ok:
        axes[1].scatter(res.all_pred * 100, res.all_true * 100, s=14, alpha=0.4,
                        color='#1e3a5f')
        lim = max(np.abs(res.all_pred).max(), np.abs(res.all_true).max()) * 100 * 1.1
        axes[1].plot([-lim, lim], [-lim, lim], color='#dc2626', ls='--', alpha=0.6)
        axes[1].axhline(0, color='#999', lw=0.6); axes[1].axvline(0, color='#999', lw=0.6)
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    unit = f' excess vs {res.benchmark}' if res.relative else ''
    axes[1].set_xlabel(f'Predicted return{unit} (%)')
    axes[1].set_ylabel(f'Actual return{unit} (%)')
    axes[1].set_title(f'Predicted vs Actual  (IC={res.ic:+.3f}, p={res.ic_p:.3f} on n_eff)',
                      fontweight='bold')
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    return _fig_to_data_uri(fig)


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

    lo = max(0, len(s) - 300)
    idx = s.index[lo:]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(idx, lower.values[lo:], upper.values[lo:],
                    color='#1e3a5f', alpha=0.10, label='Bollinger (20, 2σ)')
    ax.plot(idx, s.values[lo:], color='#1e3a5f', lw=1.7, label='Price')
    ax.plot(idx, ma20.values[lo:], color='#ea580c', lw=1.4, label='MA 20')
    ax.plot(idx, ma50.values[lo:], color='#15803d', lw=1.4, ls='--', label='MA 50')
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


# ============================================================
# ANALYZE ONE TICKER
# ============================================================
def analyze(ticker: str, name: str, vix: pd.Series | None,
            bench_name: str = '', bench: pd.Series | None = None):
    """Backtest + plots. NOTE: verdict/signals are NOT decided here — they
    depend on the FDR correction across ALL tickers [FIX 3], so they are
    assigned later in finalize()."""
    print(f"\n{'='*62}\n  {ticker} — {name}\n{'='*62}")
    close = download_close(ticker)
    if close is None or len(close) < 300:
        print("   No / insufficient data")
        return None
    print(f"   {len(close)} trading days (~{YEARS_HISTORY}y)")

    news_daily = None
    if USE_NEWS_FEATURES:
        try:
            q = f'"{name}"' if name else ticker
            s = close.index.min().strftime("%Y%m%d")
            e = close.index.max().strftime("%Y%m%d")
            news_daily = fetch_news_gdelt_timeline(q, s, e, NEWS_CACHE_DIR)
            ndays = 0 if news_daily is None else int((news_daily['buzz'] > 0).sum())
            print(f"   News: {ndays} days with coverage (query {q})")
        except Exception as ex:
            print(f"   News fetch skipped ({ex}); using neutral features")
            news_daily = None

    feat, relative = build_features(close, vix, news_daily, bench)
    if BENCHMARK_NEUTRAL and not relative:
        print(f"   WARNING: benchmark {bench_name} unavailable -> absolute target")
    print(f"   Target: {'EXCESS vs ' + bench_name if relative else 'ABSOLUTE return'}")

    cur = float(close.iloc[-1])
    chg = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    cur_vix = float(vix.dropna().iloc[-1]) if (vix is not None and len(vix.dropna())) else 20.0
    print(f"   Last close: ${cur:.2f} ({chg:+.2f}%)  VIX={cur_vix:.1f}")

    res = walk_forward(feat, relative=relative,
                       benchmark=bench_name if relative else '')
    if not res.ok:
        print("   Backtest could not run (insufficient folds)")
        return None

    print(f"\n   ── WALK-FORWARD RESULTS (pre-FDR) ──")
    print(f"   Test obs: {res.n_obs} across {len(res.folds)} folds "
          f"-> n_eff = {res.n_eff}  [FIX 1: overlap-corrected]")
    print(f"   Dir Acc: {res.dir_acc:.1f}% | Naive(train-derived): {res.naive:.1f}% "
          f"| null p0: {res.naive_p0*100:.1f}% | Edge: {res.edge:+.2f}%")
    print(f"   Raw p-value (on n_eff): {res.p_value:.4f}   <- FDR applied later")
    print(f"   Info Coefficient (Spearman): {res.ic:+.3f} (p={res.ic_p:.3f} on n_eff)")
    print(f"   Strategy net: {res.strat_return:+.1f}% | "
          f"{'Relative B&H' if relative else 'Buy&Hold'}: {res.bh_return:+.1f}% | "
          f"Sharpe: {res.sharpe:.2f} ± {res.sharpe_se:.2f} | MaxDD: {res.max_dd:.1f}%")
    if res.filt_used:
        print(f"   FILTERED (selective): {res.filt_return:+.1f}% | "
              f"Sharpe: {res.filt_sharpe:.2f} | MaxDD: {res.filt_max_dd:.1f}% | "
              f"Trades: {res.filt_trades} (acc {res.filt_dir_acc:.1f}%)")

    win52 = close.iloc[-252:] if len(close) >= 252 else close
    hi52 = float(win52.max())
    lo52 = float(win52.min())
    pos52 = (cur - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else 50.0

    return dict(
        ticker=ticker, name=name, cur=cur, chg=chg, cur_vix=cur_vix,
        hi52=hi52, lo52=lo52, pos52=pos52,
        res=res, feat=feat,
        img_price=plot_price(close, res),
        img_equity=plot_equity(res),
        img_diag=plot_diagnostics(res),
        img_drawdown=plot_drawdown(close),
        img_bollinger=plot_bollinger(close),
        img_heatmap=plot_monthly_heatmap(close),
    )


# ============================================================
# [FIX 3] FINALIZE — apply FDR across tickers, then assign verdicts
# ============================================================
def finalize(results, q: float = FDR_Q):
    """Collect every ticker's p-value, control the false discovery rate, and
    ONLY THEN decide who gets to be called significant. This must happen
    after all tickers are analyzed — significance is a property of the
    screen, not of a single ticker in isolation."""
    live = [r for r in results if r is not None and r.get('res') is not None
            and r['res'].ok]
    if not live:
        return results

    pvals = [r['res'].p_value for r in live]
    reject, qvals = benjamini_hochberg(pvals, q)

    n_naive = sum(1 for p in pvals if p < 0.05)
    print(f"\n{'='*62}")
    print(f"  [FIX 3] MULTIPLE-COMPARISON CONTROL — {len(live)} tickers screened")
    print(f"{'='*62}")
    print(f"  Uncorrected p < 0.05:            {n_naive} ticker(s)")
    print(f"  Expected by chance alone:        ~{0.05*len(live):.1f}")
    print(f"  Survive BH-FDR at q = {q:g}:        {int(reject.sum())} ticker(s)")

    for r, rej, qv in zip(live, reject, qvals):
        res = r['res']
        res.significant = bool(rej)
        res.q_value = float(qv)
        lbl, color, expl = verdict(res)
        r['label'], r['color'], r['expl'] = lbl, color, expl
        r['sig'] = trade_signals(res, r['feat'])

    print(f"\n  {'TICKER':<8}{'ACC%':>7}{'NAIVE%':>8}{'n_eff':>7}"
          f"{'p':>9}{'q':>9}{'IC':>8}  VERDICT")
    print("  " + "-" * 76)
    for r in sorted(live, key=lambda x: x['res'].q_value):
        res = r['res']
        print(f"  {r['ticker']:<8}{res.dir_acc:>7.1f}{res.naive:>8.1f}{res.n_eff:>7d}"
              f"{res.p_value:>9.4f}{res.q_value:>9.4f}{res.ic:>+8.3f}  {r['label']}")
    return results


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
  background:#fbeeee;border:1px solid #e9c9c9;padding:10px 14px;border-radius:6px;margin:18px 0 12px}
.fixnote{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11.5px;color:#1e3a5f;
  background:#eef4fb;border:1px solid #c9d9e9;padding:12px 14px;border-radius:6px;margin:0 0 30px}
.fixnote b{color:#0f1c2e}
.fixnote ul{margin:6px 0 0 18px}
.ticker-block{margin-bottom:54px}
.tk-head{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);
  padding-bottom:8px;margin-bottom:18px}
.tk-sym{font-size:30px;font-weight:700}
.tk-name{font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;color:var(--muted)}
.tk-bench{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11px;color:#fff;
  background:#1e3a5f;padding:3px 8px;border-radius:4px;letter-spacing:.06em}
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
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:2px}
.stat .tr{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;
  color:var(--ink);font-weight:600;margin-bottom:6px}
.stat .v{font-size:25px;font-weight:700}
.stat .v small{font-size:13px;font-weight:400;color:var(--muted)}
.stat .desc{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10.5px;
  color:var(--muted);line-height:1.45;margin-top:7px;border-top:1px solid var(--line);
  padding-top:7px}
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
.leaderboard{margin:44px 0 10px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:26px 28px;box-shadow:0 3px 14px rgba(15,28,46,.07)}
.lb-head{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:16px}
.lb-head h2{font-size:30px;font-weight:700;letter-spacing:-.3px;margin:4px 0}
.lb-warn{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11.5px;color:#8a1f1f;
  background:#fbeeee;border:1px solid #e9c9c9;padding:9px 12px;border-radius:5px;margin:10px 0 4px}
.lb-table{width:100%;border-collapse:collapse;font-family:'Helvetica Neue',Arial,sans-serif}
.lb-table th{text-align:left;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line)}
.lb-table td{padding:12px 10px;border-bottom:1px solid var(--line);font-size:14px;vertical-align:middle}
.lb-table tr:last-child td{border-bottom:none}
.lb-rank{font-size:22px;width:42px;text-align:center;color:var(--ink)}
.lb-sym{font-weight:800;font-size:16px}
.lb-name{color:var(--muted);font-size:12.5px}
.lb-main{font-size:18px;font-weight:800}
.lb-cell{font-size:13.5px}
.lb-verdict{display:inline-block;color:#fff;font-size:10px;font-weight:700;letter-spacing:.03em;
  padding:4px 9px;border-radius:5px;line-height:1.3}
.strongbuy{margin:36px 0 10px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:26px 28px;box-shadow:0 3px 14px rgba(15,28,46,.07)}
.sb-item{margin-top:20px}
.sb-item:first-of-type{margin-top:6px}
.sb-head{display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--line);
  padding-bottom:6px;margin-bottom:12px}
.sb-sym{font-weight:800;font-size:18px}
.sb-name{font-family:'Helvetica Neue',Arial,sans-serif;font-size:12.5px;color:var(--muted)}
.sb-badge{font-family:'Helvetica Neue',Arial,sans-serif;font-size:10px;font-weight:800;
  letter-spacing:.06em;color:#fff;padding:3px 9px;border-radius:5px;white-space:nowrap}
.sb-rel{font-family:'Helvetica Neue',Arial,sans-serif;font-size:11px;font-weight:600;
  margin-left:auto;text-align:right}
.strongsell{margin:36px 0 10px;background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:26px 28px;box-shadow:0 3px 14px rgba(15,28,46,.07)}
</style>"""


def fmt_pct(x):
    cls = 'pos' if x > 0 else ('neg' if x < 0 else '')
    return f'<span class="{cls}">{x:+.2f}%</span>'


def _filter_row_html(res) -> str:
    """Optional row showing the selective-filter comparison vs trading everything."""
    if not getattr(res, 'filt_used', False):
        return ""
    verdict_txt = ("Filtre Sharpe'ı iyileştirdiyse riski azaltmış olabilir; getiri "
                   "düşse de daha az/seçili işlem yapar. Yön doğruluğu %50 altındaysa "
                   "filtre gerçek edge yaratmıyor, sadece işlem sayısını kısıyor demektir.")
    return f"""
    <div class="stat-row">
      <div class="stat"><div class="l">Selective · Return</div><div class="tr">Seçici Filtre Getirisi</div>
        <div class="v">{fmt_pct(res.filt_return)}</div>
        <div class="desc">Sadece yüksek güven + sinyal uzlaşması olan dönemlerde işlem; gerisinde nakit.</div></div>
      <div class="stat"><div class="l">Selective · Sharpe</div><div class="tr">Seçici Sharpe</div>
        <div class="v">{res.filt_sharpe:.2f}</div>
        <div class="desc">Filtreli stratejinin riske göre getirisi. Tüm-işlem Sharpe'ından yüksekse risk azalmış.</div></div>
      <div class="stat"><div class="l">Selective · MaxDD</div><div class="tr">Seçici Maks. Düşüş</div>
        <div class="v">{res.filt_max_dd:.1f}<small>%</small></div>
        <div class="desc">Filtreli stratejinin en büyük tepe-dip kaybı.</div></div>
      <div class="stat"><div class="l">Selective · Trades</div><div class="tr">Seçici İşlem Sayısı</div>
        <div class="v">{res.filt_trades}</div>
        <div class="desc">Filtreyi geçen işlem adedi. Tüm-işlemden ({res.n_trades}) ne kadar az olduğu seçiciliği gösterir.</div></div>
      <div class="stat"><div class="l">Selective · Hit Rate</div><div class="tr">Seçici İsabet</div>
        <div class="v">{res.filt_dir_acc:.1f}<small>%</small></div>
        <div class="desc">{verdict_txt}</div></div>
    </div>"""


# ============================================================
# LEADERBOARD
# ============================================================
LEADERBOARD_KEY = 'strat_return'
LEADERBOARD_N   = 20
_LEADERBOARD_LABELS = {
    'strat_return': 'Strateji Getirisi (net, benchmark-nötr)' if BENCHMARK_NEUTRAL
                    else 'Strateji Getirisi (net)',
    'bh_return':    'Al & Tut (göreli)' if BENCHMARK_NEUTRAL else 'Al & Tut Getirisi',
    'filt_return':  'Secici Filtre Getirisi',
}


def build_leaderboard_html(results, key: str = LEADERBOARD_KEY, n: int = LEADERBOARD_N) -> str:
    rows = [r for r in results
            if r is not None and r.get('res') is not None and r['res'].ok]
    if not rows:
        return ""
    metric_label = _LEADERBOARD_LABELS.get(key, key)

    def _val(r):
        return float(getattr(r['res'], key, 0.0) or 0.0)

    rows.sort(key=_val, reverse=True)
    top = rows[:n]

    medals = ['\u2460', '\u2461', '\u2462', '\u2463', '\u2464',
              '\u2465', '\u2466', '\u2467', '\u2468', '\u2469']
    items = ""
    for i, r in enumerate(top):
        res = r['res']
        val = _val(r)
        rank = medals[i] if i < len(medals) else str(i + 1)
        items += f"""
      <tr>
        <td class="lb-rank">{rank}</td>
        <td class="lb-sym">{r['ticker']}</td>
        <td class="lb-name">{r['name']}</td>
        <td class="lb-main">{fmt_pct(val)}</td>
        <td class="lb-cell">{fmt_pct(res.bh_return)}</td>
        <td class="lb-cell">{res.sharpe:.2f} <small>&plusmn;{res.sharpe_se:.2f}</small></td>
        <td class="lb-cell">{res.q_value:.3f}</td>
        <td class="lb-cell"><span class="lb-verdict" style="background:{r['color']}">{r['label']}</span></td>
      </tr>"""

    return f"""
  <section class="leaderboard">
    <div class="lb-head">
      <div class="kicker">Ranking &middot; Top {len(top)}</div>
      <h2>En &Ccedil;ok Kazand&#305;ran {len(top)} Hisse</h2>
      <div class="sub">S&#305;ralama &ouml;l&ccedil;&uuml;t&uuml;: <b>{metric_label}</b> &middot; masraflar d&uuml;&#351;&uuml;lm&uuml;&#351; &middot;
        walk-forward test d&ouml;nemi.</div>
      <div class="lb-warn"><b>Se&ccedil;im yanl&#305;l&#305;&#287;&#305; uyar&#305;s&#305;:</b> Bu evren geriye d&ouml;n&uuml;k bilgiyle
        (&quot;2026'n&#305;n liderleri&quot;) se&ccedil;ildi ve &ouml;lm&uuml;&#351;/borsadan &ccedil;&#305;km&#305;&#351; isimleri i&ccedil;ermiyor.
        v25'in d&uuml;zeltmeleri <i>istatisti&#287;i</i> onar&#305;r, <i>&ouml;rneklemi</i> de&#287;il. Bu tablodaki
        s&#305;ralama bir tarama sonucudur, yat&#305;r&#305;m tavsiyesi de&#287;ildir. <b>q</b> s&uuml;tunu FDR
        d&uuml;zeltmesinden sonraki anlaml&#305;l&#305;kt&#305;r &mdash; getiri s&#305;ralamas&#305;na de&#287;il, ona bak&#305;n.</div>
    </div>
    <table class="lb-table">
      <thead>
        <tr>
          <th></th><th>Sembol</th><th>&#350;irket</th>
          <th>{metric_label}</th><th>Al &amp; Tut</th><th>Sharpe</th><th>q (FDR)</th><th>Karar</th>
        </tr>
      </thead>
      <tbody>{items}
      </tbody>
    </table>
  </section>"""


# ============================================================
# BUY / SELL BOARDS  (STRONG BUY + BUY, STRONG SELL + SELL)
# ============================================================
# Iki AYRI eksen var; karistirmayin:
#
#   BUY_ACTIONS / SELL_ACTIONS  ->  HANGI KADEME listeye girer
#   *_REQUIRE                   ->  KAC SINYAL ayni yonde olmali
#
# BUY_ACTIONS / SELL_ACTIONS, kabul edilen kademeleri EN GUCLU ONCE
# siralar; bu sira hem rozeti hem siralamayi belirler.
#   ('STRONG BUY', 'BUY')  -> iki kademe birden      (varsayilan)
#   ('STRONG BUY',)        -> sadece guclu sinyaller (v24 davranisi)
#
# *_REQUIRE, uc sinyalin (Technical / Model / Forecast) nasil dizilmesi
# gerektigini soyler:
#   'any'           -> herhangi biri kademeye girsin. En genis; tek basina
#                      BUY | HOLD | HOLD bile listeye girer.
#   'tech_gated'    -> NOW-Technical kademeye GIRMEK ZORUNDA, ayrica 3
#                      sinyalden en az CONSENSUS_MIN tanesi ayni yonde
#                      olsun; ters sinyal bulunmasin.  <-- VARSAYILAN
#                      Mantik: teknik durus zorunlu bir kapidir — model ne
#                      derse desin, fiyat momentumu onaylamiyorsa hisse
#                      listeye girmez. Model+Forecast al deyip Technical
#                      HOLD kaldiginda hisse ELENIR.
#                      Tam olarak 16 desen gecer:
#                        ucu de al olan 8 desen (SB/B'nin tum kombinasyonlari)
#                        + tech al, biri daha al, biri HOLD olan 8 desen
#                      Ornek gecenler:  SB|SB|SB   SB|B|SB   SB|B|B   SB|SB|B
#                                       B|B|B      B|SB|SB   B|SB|B   B|B|SB
#                                       SB|SB|HOLD SB|HOLD|SB SB|B|HOLD
#                                       SB|HOLD|B  B|SB|HOLD  B|HOLD|SB
#                                       B|B|HOLD   B|HOLD|B
#                      Ornek elenenler: HOLD|SB|SB  (Technical notr -> kapi kapali)
#                                       SB|HOLD|HOLD (tek al sinyali)
#                                       SB|SB|SELL   (ters sinyal)
#   'majority'      -> KADEMELER TOPLANARAK en az CONSENSUS_MIN tanesi ayni
#                      yonde olsun; ters sinyal bulunmasin. 'tech_gated'den
#                      farki: Technical'i zorunlu tutmaz, yani HOLD|SB|SB de
#                      gecer (16 yerine 20 desen).
#   'tier_majority' -> AYNI KADEMEDEN en az CONSENSUS_MIN tane olsun.
#                      Kademeler AYRI AYRI sayilir: "2 STRONG BUY" VEYA
#                      "2 BUY". SB|BUY|HOLD'u ELER (1+1, hicbiri 2 degil).
#   'all'           -> ucu de kademeye girsin. En dar; tek bir HOLD bile eler.
#   'model'         -> yalnizca NOW-Model sinyali sayilir.
#
# ROZET KURALI: 'tech_gated' ve 'majority'de rozet, sinyaller arasinda
# BULUNAN en guclu kademedir — B|SB|HOLD hissesi tek bir SB tasisa da SB
# rozeti alir. ('tier_majority'de ise rozet esigi FIILEN TUTTURAN kademedir.)
# Rozet siralamayi da belirledigi icin bu fark tahtanin ust siralarini
# degistirir. Technical kademesini rozet yapmak isterseniz _matched_tier
# icindeki ilgili dala bakin.
#
# CONSENSUS_BLOCK_OPPOSING ucunde de ('tech_gated', 'majority',
# 'tier_majority') gecerlidir.
#
# MAX = 0 -> hepsini goster; >0 -> listeyi kirp.
BUY_ACTIONS        = ('STRONG BUY', 'BUY')
SELL_ACTIONS       = ('STRONG SELL', 'SELL')
STRONGBUY_REQUIRE  = 'tech_gated'
STRONGBUY_MAX      = 0
STRONGSELL_REQUIRE = 'tech_gated'
STRONGSELL_MAX     = 0

# 'majority' ve 'tier_majority' modlarinin ayarlari.
CONSENSUS_MIN            = 2      # kac sinyal gerekli
CONSENSUS_BLOCK_OPPOSING = True   # ters yonde sinyal varsa hisseyi ele

# Tier ranking used for sorting and for the badge (lower = stronger).
_TIER_RANK = {'STRONG BUY': 0, 'BUY': 1, 'STRONG SELL': 0, 'SELL': 1}
_TIER_COLOR = {'STRONG BUY': '#15803d', 'BUY': '#16a34a',
               'STRONG SELL': '#b91c1c', 'SELL': '#dc2626'}
_ALL_BUY  = ('STRONG BUY', 'BUY')
_ALL_SELL = ('STRONG SELL', 'SELL')


def _opposing(actions) -> tuple:
    """Bu tahtanin ters yonu — 'majority' modunda celiskiyi yakalamak icin."""
    if any(a in _ALL_BUY for a in actions):
        return _ALL_SELL
    if any(a in _ALL_SELL for a in actions):
        return _ALL_BUY
    return ()


def _require_text(mode: str) -> str:
    """Rapordaki insan-okur aciklama."""
    opp = " ve ters sinyal yok" if CONSENSUS_BLOCK_OPPOSING else ""
    if mode == 'tech_gated':
        return (f"NOW-Technical ayni yonde OLMALI + 3 sinyalden en az "
                f"{CONSENSUS_MIN}'i ayni yonde{opp}")
    if mode == 'tier_majority':
        return f"ayni kademeden en az {CONSENSUS_MIN} sinyal{opp}"
    if mode == 'majority':
        return f"3 sinyalden en az {CONSENSUS_MIN}'i ayni yonde{opp}"
    if mode == 'all':
        return "3 sinyalin de ayni yonde olmasi sart"
    if mode == 'model':
        return "yalnizca NOW-Model sinyali dikkate alinir"
    return "3 sinyalden herhangi biri yeterli"


def _sig_is_action(sig: dict, action, mode: str) -> bool:
    """True if the ticker qualifies for `action` under `mode`.

    `action` may be a single label ('BUY') or a tuple of accepted labels
    (('STRONG BUY','BUY')) — the tuple form is what lets one board hold
    both tiers.
    """
    wanted = (action,) if isinstance(action, str) else tuple(action)
    acts = [sig.get('tech_action', ''), sig.get('now_action', ''), sig.get('fc_action', '')]
    hit = [a in wanted for a in acts]
    if mode == 'all':
        return all(hit)
    if mode == 'model':
        return sig.get('now_action', '') in wanted
    if mode in ('majority', 'tier_majority', 'tech_gated'):
        if CONSENSUS_BLOCK_OPPOSING:
            opp = _opposing(wanted)
            if any(a in opp for a in acts):
                return False
        if mode == 'tech_gated':
            # NOW-Technical zorunlu kapi: model ne derse desin, teknik durus
            # onaylamiyorsa hisse girmez. HOLD|SB|SB burada elenir.
            if not hit[0]:
                return False
            return sum(hit) >= CONSENSUS_MIN
        if mode == 'tier_majority':
            # Kademeler AYRI sayilir: "2 STRONG BUY" VEYA "2 BUY".
            return any(acts.count(a) >= CONSENSUS_MIN for a in wanted)
        # 'majority': kademeler toplanir -> SB|BUY|HOLD gecer.
        return sum(hit) >= CONSENSUS_MIN
    return any(hit)   # 'any' (varsayilan)


def _matched_tier(sig: dict, actions, mode: str) -> str:
    """Tier badge for a ticker that qualifies for this board, else ''.

    Two steps, and the order matters:
      1. QUALIFY against the whole accepted set. In 'all' mode a ticker
         showing STRONG BUY / BUY / BUY must pass — every signal IS a buy,
         just at mixed tiers. Testing each tier separately would reject it
         (not all three are STRONG BUY, not all three are BUY) and the
         ticker would silently disappear from the board.
      2. LABEL with the strongest tier actually present among the signals
         (`actions` is strongest-first). In 'model' mode only the model's
         own call decides the badge.
    """
    actions = (actions,) if isinstance(actions, str) else tuple(actions)
    if not _sig_is_action(sig, actions, mode):
        return ''
    if mode == 'model':
        nm = sig.get('now_action', '')
        return nm if nm in actions else ''
    acts = [sig.get('tech_action', ''), sig.get('now_action', ''), sig.get('fc_action', '')]
    if mode == 'tier_majority':
        # Rozet, esigi FIILEN TUTTURAN kademe olmali; sadece ekranda gorunen
        # en gurultulu kademe degil. BUY|STRONG BUY|BUY ornegi: nitelenmesini
        # saglayan sey 2 adet BUY'dir (STRONG BUY tek basina 1 tane), yani
        # rozet BUY. `actions` en guclu once sirali oldugu icin ilk tutan kazanir.
        for a in actions:
            if acts.count(a) >= CONSENSUS_MIN:
                return a
        return ''
    present = [a for a in actions if a in acts]
    return present[0] if present else ''


def _signal_cards_html(r, tier: str = '') -> str:
    """Bir hisse icin uc sinyal kartini uretir.
    `tier` verilirse basliga hangi kademeden girdigini gosteren rozet eklenir."""
    s = r['sig']
    rel_tag = f" vs {r['res'].benchmark}" if r['res'].relative else ""
    badge = ""
    if tier:
        badge = (f'<span class="sb-badge" style="background:'
                 f'{_TIER_COLOR.get(tier, "#6b7280")}">{tier}</span>')
    # Kart uzerinde ayrica FDR sonrasi guvenilirlik notu — bir hissenin
    # listede olmasi modelin ona guvenildigi anlamina gelmez.
    rel_note = (f'<span class="sb-rel" style="color:{s["reliability_color"]}">'
                f'{s["reliability"]}</span>')
    return f"""
    <div class="sb-item">
      <div class="sb-head"><span class="sb-sym">{r['ticker']}</span>
        <span class="sb-name">{r['name']}</span>{badge}{rel_note}</div>
      <div class="sig-grid">
        <div class="sig-card">
          <div class="sig-tier">NOW &middot; Technical</div>
          <div class="sig-pill" style="background:{s['tech_color']}">{s['tech_action']}</div>
          <div class="sig-meta">RSI {s['rsi']:.0f} &middot; 20d mom {s['mom20']*100:+.1f}%</div>
        </div>
        <div class="sig-card">
          <div class="sig-tier">NOW &middot; Model</div>
          <div class="sig-pill" style="background:{s['now_color']}">{s['now_action']}</div>
          <div class="sig-meta">latest model call: {s['now_pred']*100:+.1f}%{rel_tag}</div>
        </div>
        <div class="sig-card">
          <div class="sig-tier">{PREDICTION_HORIZON}-DAY &middot; Forecast</div>
          <div class="sig-pill" style="background:{s['fc_color']}">{s['fc_action']}</div>
          <div class="sig-meta">forward est: {s['fc_pred']*100:+.1f}%{rel_tag}</div>
        </div>
      </div>
    </div>"""


def _board_html(results, actions, css_class: str, title_word: str,
                require: str, max_n: int, reverse: bool) -> str:
    """One board holding every accepted tier in `actions` (strongest first)."""
    actions = (actions,) if isinstance(actions, str) else tuple(actions)
    title = " / ".join(actions)

    rows = []
    for r in results:
        if r is None or r.get('res') is None or not r['res'].ok or not r.get('sig'):
            continue
        tier = _matched_tier(r['sig'], actions, require)
        if tier:
            rows.append((r, tier))

    if not rows:
        return f"""
  <section class="{css_class}">
    <div class="lb-head">
      <div class="kicker">Signals &middot; {title}</div>
      <h2>{title} Sinyali Olan Hisse Yok</h2>
      <div class="sub">Bu ko&#351;ulda ('{require}') hi&ccedil;bir hisse {title} vermedi.</div>
    </div>
  </section>"""

    # Once kademe (STRONG ustte), sonra kademe icinde model cagrisinin gucune gore.
    rows.sort(key=lambda rt: (_TIER_RANK.get(rt[1], 9),
                              -float(rt[0]['sig'].get('now_pred', 0.0)) if reverse
                              else float(rt[0]['sig'].get('now_pred', 0.0))))
    if max_n and max_n > 0:
        rows = rows[:max_n]

    counts = {a: sum(1 for _, t in rows if t == a) for a in actions}
    breakdown = " &middot; ".join(f"{counts[a]} {a}" for a in actions if counts[a])

    cards = "".join(_signal_cards_html(r, tier) for r, tier in rows)
    rel_note = (" Model &ccedil;a&#287;r&#305;lar&#305; <b>benchmark'a g&ouml;re g&ouml;reli</b> getiridir "
                "(mutlak fiyat y&ouml;n&uuml; de&#287;il)." if BENCHMARK_NEUTRAL else "")
    return f"""
  <section class="{css_class}">
    <div class="lb-head">
      <div class="kicker">Signals &middot; {title}</div>
      <h2>{title} Sinyali Olan {len(rows)} Hisse</h2>
      <div class="sub">Da&#287;&#305;l&#305;m: <b>{breakdown}</b> &middot; kural:
        <b>{_require_text(require)}</b> ('{require}') &middot;
        &ouml;nce kademe, sonra en g&uuml;&ccedil;l&uuml; {title_word} &uuml;stte.
        Sinyaller ham g&ouml;sterilir; her kart&#305;n ba&#351;l&#305;&#287;&#305;nda o hissenin FDR sonras&#305;
        g&uuml;venilirlik notu vard&#305;r &mdash; listede olmak modelin o hissede kan&#305;tlanm&#305;&#351; bir
        &uuml;st&uuml;nl&uuml;&#287;&uuml; oldu&#287;u anlam&#305;na <b>gelmez</b>.{rel_note}
        Yat&#305;r&#305;m tavsiyesi de&#287;ildir.</div>
    </div>{cards}
  </section>"""


def build_strong_buy_html(results, require: str = STRONGBUY_REQUIRE,
                          max_n: int = STRONGBUY_MAX, actions=BUY_ACTIONS) -> str:
    return _board_html(results, actions, 'strongbuy', 'model &ccedil;a&#287;r&#305;s&#305;',
                       require, max_n, reverse=True)


def build_strong_sell_html(results, require: str = STRONGSELL_REQUIRE,
                           max_n: int = STRONGSELL_MAX, actions=SELL_ACTIONS) -> str:
    return _board_html(results, actions, 'strongsell', 'sat&#305;&#351; &ccedil;a&#287;r&#305;s&#305;',
                       require, max_n, reverse=False)


def build_html(results):
    now = datetime.now().strftime('%B %d, %Y · %H:%M')
    n_live = len([r for r in results if r is not None and r.get('res') and r['res'].ok])
    n_sig = len([r for r in results if r is not None and r.get('res')
                 and r['res'].ok and r['res'].significant])
    blocks = ""
    for r in results:
        if r is None or 'label' not in r:
            continue
        res = r['res']
        rel_tag = f" vs {res.benchmark}" if res.relative else ""
        bench_pill = (f'<span class="tk-bench">TARGET: EXCESS vs {res.benchmark}</span>'
                      if res.relative else
                      '<span class="tk-bench">TARGET: ABSOLUTE</span>')
        bh_label = 'Al &amp; Tut (göreli)' if res.relative else 'Al ve Tut'
        bh_desc = (f"Hisseyi al&#305;p {res.benchmark}'i short'layman&#305;n getirisi — yani hissenin "
                   f"endekse g&ouml;re g&ouml;reli performans&#305;. Stratejinin ge&ccedil;mesi gereken k&#305;yas."
                   if res.relative else
                   "Hisseyi al&#305;p hi&ccedil; i&#351;lem yapmadan tutman&#305;n getirisi. Stratejinin ge&ccedil;mesi gereken k&#305;yas.")
        strat_desc = (f"Modelin sinyallerine uyulsayd&#305;, {res.benchmark}'e kar&#351;&#305; n&ouml;tr (long/short &ccedil;ift) "
                      f"kurulup her iki bacak i&ccedil;in masraf d&uuml;&#351;&uuml;l&uuml;nce elde edilecek toplam getiri. Beta ar&#305;nd&#305;r&#305;lm&#305;&#351;."
                      if res.relative else
                      "Modelin sinyallerine uyulsayd&#305;, masraflar d&uuml;&#351;&uuml;l&uuml;nce elde edilecek toplam getiri.")
        blocks += f"""
  <section class="ticker-block">
    <div class="tk-head">
      <span class="tk-sym">{r['ticker']}</span>
      <span class="tk-name">{r['name']}</span>
      {bench_pill}
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
      <div class="v-dir">{direction_text(res)}</div>
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
        <div class="sig-meta">latest model call: {r['sig']['now_pred']*100:+.1f}%{rel_tag}</div>
      </div>
      <div class="sig-card">
        <div class="sig-tier">{PREDICTION_HORIZON}-DAY · Forecast</div>
        <div class="sig-pill" style="background:{r['sig']['fc_color']}">{r['sig']['fc_action']}</div>
        <div class="sig-meta">forward est: {r['sig']['fc_pred']*100:+.1f}%{rel_tag}</div>
      </div>
    </div>
    <div class="sig-foot" style="border-left-color:{r['sig']['reliability_color']}">
      <b style="color:{r['sig']['reliability_color']}">{r['sig']['reliability']}.</b>
      &nbsp;<i>NOW · Technical</i> reads today's absolute trend/momentum/RSI (model-independent);
      <i>NOW · Model</i> is the model's nearest-day call; <i>{PREDICTION_HORIZON}-day · Forecast</i>
      is its forward directional estimate{rel_tag}. Signals are shown raw — the reliability note
      states whether the model earned an edge that survives overlap correction AND
      FDR control across all {n_live} screened tickers. Educational only, not advice.
    </div>

    <div class="stat-row">
      <div class="stat"><div class="l">Directional Acc.</div><div class="tr">Yön Doğruluğu</div>
        <div class="v">{res.dir_acc:.1f}<small>%</small></div>
        <div class="desc">Modelin {'göreli ' if res.relative else ''}yönü kaç kez doğru bildiği. %50 yazı-turadır; naif referansı geçmesi gerekir.</div></div>
      <div class="stat"><div class="l">Naive (ex-ante)</div><div class="tr">Naif Referans (train'den)</div>
        <div class="v">{res.naive:.1f}<small>%</small></div>
        <div class="desc">FIX 2: Çoğunluk yönü <b>yalnızca eğitim verisinden</b> öğrenildi (v24 test setine bakıyordu = look-ahead), sonra test setinde ölçüldü. Null olasılık p₀={res.naive_p0*100:.1f}%.</div></div>
      <div class="stat"><div class="l">Edge</div><div class="tr">Üstünlük (Fark)</div>
        <div class="v">{fmt_pct(res.edge)}</div>
        <div class="desc">Yön doğruluğu eksi naif referans. Pozitifse model referansı yener; negatifse referans daha iyi.</div></div>
      <div class="stat"><div class="l">n / n_eff</div><div class="tr">Gözlem / Etkin Gözlem</div>
        <div class="v">{res.n_obs}<small>/{res.n_eff}</small></div>
        <div class="desc">FIX 1: {PREDICTION_HORIZON} günlük hedefler örtüştüğü için ardışık gözlemler bağımsız değil. Tüm testler <b>n_eff = n/{PREDICTION_HORIZON}</b> üzerinden yapıldı.</div></div>
      <div class="stat"><div class="l">p → q (FDR)</div><div class="tr">Anlamlılık → FDR sonrası</div>
        <div class="v" style="font-size:17px">{res.p_value:.4f} → <b>{res.q_value:.3f}</b></div>
        <div class="desc">FIX 3: {n_live} hisse tarandı, şans eseri ~{0.05*n_live:.1f} tanesi p&lt;0.05 verirdi. Benjamini-Hochberg (q={FDR_Q:g}) sonrası {n_sig} hisse hayatta kaldı. <b>q'ya bakın, p'ye değil.</b></div></div>
      <div class="stat"><div class="l">Info Coefficient</div><div class="tr">Bilgi Katsayısı</div>
        <div class="v">{res.ic:+.3f}</div>
        <div class="desc">Tahmin ile gerçek getiri arasındaki sıralama korelasyonu (-1..+1). p={res.ic_p:.3f} (n_eff üzerinden). İyi bir quant fonda 0.02–0.05 normaldir.</div></div>
    </div>
    <div class="stat-row">
      <div class="stat"><div class="l">Strategy (net)</div><div class="tr">Strateji Getirisi (net)</div>
        <div class="v">{fmt_pct(res.strat_return)}</div>
        <div class="desc">{strat_desc}</div></div>
      <div class="stat"><div class="l">{'Relative B&amp;H' if res.relative else 'Buy &amp; Hold'}</div><div class="tr">{bh_label}</div>
        <div class="v">{fmt_pct(res.bh_return)}</div>
        <div class="desc">{bh_desc}</div></div>
      <div class="stat"><div class="l">Sharpe &plusmn; SE</div><div class="tr">Sharpe Oranı ± Std. Hata</div>
        <div class="v" style="font-size:20px">{res.sharpe:.2f} <small>&plusmn; {res.sharpe_se:.2f}</small></div>
        <div class="desc">Risk başına getiri. <b>Std. hataya dikkat:</b> {YEARS_HISTORY} yılda Sharpe çok belirsiz ölçülür — hata payı bandı 0'ı içeriyorsa "iyi Sharpe" istatistiksel olarak sıfırdan ayırt edilemez.</div></div>
      <div class="stat"><div class="l">Max Drawdown</div><div class="tr">Maks. Düşüş</div>
        <div class="v">{res.max_dd:.1f}<small>%</small></div>
        <div class="desc">Zirveden en dip noktaya yaşanan en büyük kayıp. Riskin ölçüsü; sıfıra yakın iyidir.</div></div>
      <div class="stat"><div class="l">Trades</div><div class="tr">İşlem Sayısı</div>
        <div class="v">{res.n_trades}</div>
        <div class="desc">Örtüşmeyen {PREDICTION_HORIZON} günlük tutuş dönemi sayısı. Her işlemden {COST_BPS:.0f}bps × {COST_LEGS} bacak masraf düşüldü.</div></div>
    </div>
    {_filter_row_html(res)}

    <div class="fig"><h3>Price History &amp; {PREDICTION_HORIZON}-Day Forward Forecast</h3><img src="{r['img_price']}"></div>
    <div class="fig"><h3>Moving Averages &amp; Bollinger Bands</h3><img src="{r['img_bollinger']}"></div>
    <div class="fig"><h3>Drawdown from Peak (Risk View)</h3><img src="{r['img_drawdown']}"></div>
    <div class="fig"><h3>Equity Curve (net of {COST_BPS:.0f}bps &times; {COST_LEGS} costs)</h3><img src="{r['img_equity']}"></div>
    <div class="fig"><h3>Diagnostics</h3><img src="{r['img_diag']}"></div>
    <div class="fig"><h3>Monthly Returns Heatmap (Seasonality)</h3><img src="{r['img_heatmap']}"></div>
  </section>"""

    if not blocks:
        blocks = '<p style="text-align:center;color:#b91c1c">No tickers could be analyzed.</p>'

    mode_txt = ('benchmark-neutral (excess return) targets' if BENCHMARK_NEUTRAL
                else 'absolute return targets')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MultiStock v25 — Honest Backtester</title>{CSS}</head>
<body><div class="wrap">
  <header class="masthead">
    <div class="kicker">Quantitative Research · Walk-Forward Evaluation</div>
    <h1>MultiStock v25</h1>
    <div class="sub">Ridge regression · {YEARS_HISTORY}-year history · {PREDICTION_HORIZON}-day horizon ·
      {N_WALK_FOLDS}-fold walk-forward · purged for overlap leakage · {mode_txt} ·
      overlap-corrected tests · BH-FDR at q={FDR_Q:g} across {n_live} tickers · generated {now}</div>
  </header>

  <div class="disclaimer">
    ⚠ Educational / research tool — NOT investment advice. Backtested results are
    hypothetical, exclude taxes, borrow costs and real fill dynamics, and never guarantee
    future performance. The "verdict" reflects statistical evidence in historical data only.
    The ticker universe was chosen with hindsight — see the leaderboard warning.
  </div>

  <div class="fixnote">
    <b>v24 → v25: dört istatistiksel düzeltme.</b> Bu raporun sayıları v24'ünkilerle
    kıyaslanamaz — v24 sistematik olarak iyimserdi.
    <ul>
      <li><b>FIX 1 · Etkin örneklem:</b> {PREDICTION_HORIZON} günlük hedefler örtüşüyor;
        tüm p-değerleri artık n yerine <b>n_eff = n/{PREDICTION_HORIZON}</b> üzerinden.</li>
      <li><b>FIX 2 · Ex-ante referans:</b> naif çoğunluk yönü test setinden değil,
        <b>her fold'un eğitim verisinden</b> öğreniliyor (look-ahead giderildi).</li>
      <li><b>FIX 3 · Çoklu test:</b> {n_live} hisse tarandığı için Benjamini-Hochberg
        FDR (q={FDR_Q:g}) uygulanıyor. <b>{n_sig}</b> hisse hayatta kaldı.</li>
      <li><b>FIX 4 · Benchmark-nötr hedef:</b> model artık ham getiriyi değil,
        <b>endekse göre fazla getiriyi</b> tahmin ediyor — 6 yıllık boğa piyasasının
        beta'sı alfa gibi görünmüyor.</li>
    </ul>
  </div>

  {blocks}

  {build_leaderboard_html(results)}

  {build_strong_buy_html(results)}

  {build_strong_sell_html(results)}

  <div class="method">
    <b>Methodology.</b> Features use only past data (the feature window ends at t-1 while
    entry uses the close at t). The model is evaluated with expanding-window walk-forward
    folds; each fold trains only on observations whose {PREDICTION_HORIZON}-day forward label
    was realized before the test window begins (purging removes overlapping-horizon leakage).
    <b>[FIX 4]</b> The label is the {'excess log return over the ticker&#39;s benchmark' if BENCHMARK_NEUTRAL else 'absolute log return'},
    so the equity curve represents a {'dollar-neutral long/short pair and isolates alpha rather than beta' if BENCHMARK_NEUTRAL else 'directional outright position'}.
    <b>[FIX 2]</b> Directional accuracy is compared to a naive majority-class rule whose
    direction and expected hit rate are estimated on TRAIN data only.
    <b>[FIX 1]</b> Because H-day labels overlap, the binomial and Spearman tests are run on
    n_eff = n/{PREDICTION_HORIZON} independent-equivalent observations.
    <b>[FIX 3]</b> Across all screened tickers, p-values are adjusted with Benjamini-Hochberg
    to control the false discovery rate at q={FDR_Q:g}; only survivors are labelled significant.
    The equity curve uses <b>non-overlapping</b> holding periods and subtracts
    {COST_BPS:.0f} bps &times; {COST_LEGS} leg(s) of round-trip costs.
    <br><br>
    <b>Known remaining limitations</b> (not fixed in v25): the ticker universe is
    hindsight-selected and survivorship-biased; the equity curve samples one of
    {PREDICTION_HORIZON} possible non-overlapping offsets; short legs ignore borrow cost and
    locate risk; 5 bps may be optimistic for small caps; the selective filter compares a
    relative prediction to an absolute technical posture; and the news query matches a company
    name string, which fails for thinly-covered small caps.
  </div>

  <div class="foot">MultiStock v25 · Honest Quantitative Backtester · {now}</div>
</div></body></html>"""


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 62)
    print("  MULTI-STOCK v25 — HONEST QUANTITATIVE BACKTESTER")
    print("=" * 62)
    print(f"  Mode: {'BENCHMARK-NEUTRAL (excess returns)' if BENCHMARK_NEUTRAL else 'ABSOLUTE returns'}")
    print(f"  FDR:  Benjamini-Hochberg at q = {FDR_Q:g} across {len(TICKERS)} tickers")
    print(f"  Tests run on n_eff = n / {PREDICTION_HORIZON} (overlap-corrected)")
    print("=" * 62)

    vix = download_vix()
    print(f"  VIX history: {0 if vix is None else len(vix)} days")

    benches = download_benchmarks(TICKERS) if BENCHMARK_NEUTRAL else {}

    results = []
    for tk, cfg in TICKERS.items():
        name, bname = _ticker_meta(cfg)
        bseries = benches.get(bname) if BENCHMARK_NEUTRAL else None
        try:
            results.append(analyze(tk, name, vix, bname, bseries))
        except Exception as e:
            print(f"   error on {tk}: {e}")
            results.append(None)

    # [FIX 3] significance can only be decided once every ticker is in.
    results = finalize(results, FDR_Q)

    html = build_html(results)
    
    # Dosyayı doğrudan web üzerinden erişilebilir WordPress dizinine kaydet
    out_filename = "MultiStock_v25.html"
    out = os.path.join(WP_OUTPUT_DIR, out_filename)
    
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  HTML report -> {out}")
    except Exception as e:
        print(f"\n  HATA: {out} dizinine yazılamadı! Lütfen WP_OUTPUT_DIR yolunu kontrol edin. Detay: {e}")
        # Hata durumunda yedeği projenin kendi klasörüne kaydet
        backup_out = os.path.join(BASE_DIR, out_filename)
        with open(backup_out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Yedek olarak buraya kaydedildi -> {backup_out}")
        
    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()