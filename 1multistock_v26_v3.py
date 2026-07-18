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
    trading_dates with columns ['news_sent','news_buzz], never NaN
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


def _articles_cache_path(cache_dir: str, query: str, start: str, end: str) -> str:
    key = hashlib.md5(f"ARTLIST|{query}|{start}|{end}".encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in query)[:24]
    return os.path.join(cache_dir, f"gdelt_art_{safe}_{key}.parquet")


def fetch_news_gdelt_articles(query: str, start: str, end: str,
                              cache_dir: str = "news_cache",
                              max_records: int = 50) -> pd.DataFrame:
    """Sample of actual headlines for `query` from GDELT (ArtList mode).

    Returns DataFrame[date, title, url, domain], most-recent first.
    These are the underlying articles GDELT aggregated into the daily
    tone/volume that the model consumes; shown for human context only —
    the model itself never reads individual headlines, only the daily
    sentiment + volume aggregate. Cached to disk like the timeline.
    Failure (no network) -> empty DataFrame (report just omits the list).
    """
    cols = ["date", "title", "url", "domain"]
    os.makedirs(cache_dir, exist_ok=True)
    path = _articles_cache_path(cache_dir, query, start, end)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    try:
        import requests
        params = dict(query=query, mode="ArtList", format="json",
                      startdatetime=f"{start}000000", enddatetime=f"{end}235959",
                      maxrecords=int(max_records), sort="DateDesc")
        r = requests.get(GDELT_DOC, params=params, timeout=30)
        r.raise_for_status()
        arts = r.json().get("articles", []) or []
    except Exception as e:
        print(f"   [news] GDELT headline fetch failed for {query!r}: {e}")
        return pd.DataFrame(columns=cols)

    recs = []
    for a in arts:
        raw = str(a.get("seendate", ""))
        try:
            day = pd.to_datetime(raw).normalize()
        except Exception:
            day = pd.NaT
        recs.append((day, str(a.get("title", "")).strip(),
                     str(a.get("url", "")).strip(),
                     str(a.get("domain", "")).strip()))
    df = pd.DataFrame(recs, columns=cols)
    df = df[df["title"] != ""].drop_duplicates(subset=["title"]).reset_index(drop=True)
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"))
    return df


# ============================================================
#  GOOGLE NEWS  (RSS) — alternative news source
# ------------------------------------------------------------
#  Google News has no official API, but it publishes a free RSS search
#  feed (no key) that returns recent articles for a query:
#    https://news.google.com/rss/search?q=QUERY&hl=en-US&gl=US&ceid=US:en
#  We parse title / link / pubDate / source with the stdlib XML parser
#  (no extra dependency). The feed only covers RECENT news (~last month,
#  up to ~100 items), so — like any free news source — the model's daily
#  sentiment/volume features only populate for the recent window; older
#  backtest days stay neutral (0.0). Sentiment is derived from headline
#  text with a small finance lexicon (transparent, offline, no deps).
# ============================================================
GNEWS_RSS = "https://news.google.com/rss/search"

_POS_WORDS = {
    'beat', 'beats', 'surge', 'surges', 'soar', 'soars', 'jump', 'jumps',
    'rise', 'rises', 'rally', 'gain', 'gains', 'record', 'high', 'profit',
    'growth', 'strong', 'upgrade', 'upgraded', 'outperform', 'bullish',
    'boost', 'boosts', 'win', 'wins', 'success', 'breakthrough', 'top',
    'tops', 'raise', 'raised', 'optimistic', 'expand', 'expands', 'deal',
    'partnership', 'approve', 'approved', 'positive', 'rebound', 'climb',
}
_NEG_WORDS = {
    'miss', 'misses', 'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges',
    'slump', 'slumps', 'slide', 'slides', 'tumble', 'tumbles', 'loss',
    'losses', 'weak', 'downgrade', 'downgraded', 'underperform', 'bearish',
    'cut', 'cuts', 'layoff', 'layoffs', 'lawsuit', 'probe', 'fine', 'fined',
    'recall', 'warn', 'warns', 'warning', 'concern', 'concerns', 'risk',
    'fraud', 'decline', 'declines', 'crash', 'sink', 'sinks', 'fear',
    'fears', 'negative', 'delay', 'delays', 'halt', 'ban', 'banned', 'sue',
}


def _headline_sentiment(text: str) -> float:
    """Lightweight headline polarity in [-1, 1] from a finance word list.

    A transparent, dependency-free proxy. Swap in VADER/FinBERT for more
    accuracy if those libraries are available.
    """
    if not text:
        return 0.0
    words = ''.join(c.lower() if (c.isalnum() or c.isspace()) else ' '
                    for c in text).split()
    pos = sum(w in _POS_WORDS for w in words)
    neg = sum(w in _NEG_WORDS for w in words)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def fetch_news_google_rss(query: str, cache_dir: str = "news_cache",
                          max_records: int = 100,
                          hl: str = "en-US", gl: str = "US",
                          ceid: str = "US:en") -> pd.DataFrame:
    """Recent articles for `query` from the Google News RSS feed.

    Returns DataFrame[date, title, url, domain], most-recent first.
    Cached to disk per day (repeatable within a day). No API key needed.
    """
    cols = ["date", "title", "url", "domain"]
    os.makedirs(cache_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    key = hashlib.md5(f"GNEWS|{query}|{hl}{gl}|{today}".encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in query)[:24]
    path = os.path.join(cache_dir, f"gnews_{safe}_{key}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    try:
        import requests
        import urllib.parse
        import xml.etree.ElementTree as ET
        url = (f"{GNEWS_RSS}?q={urllib.parse.quote(query)}"
               f"&hl={hl}&gl={gl}&ceid={ceid}")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"   [news] Google News fetch failed for {query!r}: {e}")
        return pd.DataFrame(columns=cols)

    recs = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        if not source and " - " in title:        # title is "Headline - Source"
            source = title.rsplit(" - ", 1)[-1].strip()
        try:
            day = pd.to_datetime(pub).tz_localize(None).normalize()
        except Exception:
            try:
                day = pd.to_datetime(pub, utc=True).tz_localize(None).normalize()
            except Exception:
                day = pd.NaT
        # strip the trailing " - Source" GoogleNews appends to titles
        clean_title = title
        if source and clean_title.endswith(" - " + source):
            clean_title = clean_title[: -(len(source) + 3)].strip()
        recs.append((day, clean_title, link, source))

    df = pd.DataFrame(recs, columns=cols)
    df = df[df["title"] != ""].drop_duplicates(subset=["title"])
    df = df.sort_values("date", ascending=False, na_position="last").head(max_records)
    df = df.reset_index(drop=True)
    if df.empty:
        print(f"   [news] Google News returned 0 articles for {query!r}.")
        return df                      # don't cache empties -> allow retry/fallback
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"))
    return df


def _articles_to_daily(articles: pd.DataFrame,
                       use_sentiment: bool = True) -> pd.DataFrame:
    """Turn an article list into a daily ['sent','buzz'] stream for the model.

    buzz = number of articles that day; sent = mean headline polarity that
    day (0.0 if sentiment disabled). Days with no articles are simply absent
    (build_news_features fills them neutral).
    """
    if articles is None or len(articles) == 0:
        return pd.DataFrame(columns=["sent", "buzz"])
    a = articles.dropna(subset=["date"]).copy()
    if a.empty:
        return pd.DataFrame(columns=["sent", "buzz"])
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    if use_sentiment:
        a["pol"] = a["title"].map(_headline_sentiment)
    else:
        a["pol"] = 0.0
    g = a.groupby("date")
    out = pd.DataFrame({"sent": g["pol"].mean(), "buzz": g.size()}).sort_index()
    return out


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TICKERS = {
    # Either form works:
    #   'TICKER': 'Display Name'
    #   'TICKER': {'name': 'Display Name', 'benchmark': 'SOXX'}
    # Optional 'news' key: a custom GDELT query to pin the search to THIS
    # company (overrides the auto query built from the name). Use it when the
    # plain name is ambiguous or misses coverage, e.g.
    #   'GOOGL': {'name': 'Alphabet Inc.', 'news': '"Alphabet Inc" OR "Google"'}
    'INTC': {'name': 'Intel Corporation',     'benchmark': 'SOXX'},
    'AAPL':  {'name': 'Apple Inc.', 'benchmark': 'SOXX'},
    'NVDA': {'name': 'NVIDIA Corporation',     'benchmark': 'SOXX'},    
    'GOOGL': {'name': 'Alphabet Inc.',     'benchmark': 'SOXX'},
    'SNDK': {'name': 'SanDisk Corporation', 'benchmark': 'SOXX'},
}


def _ticker_meta(cfg):
    """Accept either a plain name string or a {'name':..., 'benchmark':...} dict."""
    if isinstance(cfg, dict):
        return cfg.get('name', ''), cfg.get('benchmark', None)
    return str(cfg), None


def _news_query(ticker: str, name: str, cfg=None) -> str:
    """Build a GDELT query pinned to THIS specific stock/company.

    Priority:
      1. Explicit 'news' override in the ticker's config dict.
      2. The company name as an exact phrase (precise, stock-specific) —
         plus the name with common corporate suffixes stripped as an OR
         term so we still catch coverage that drops 'Inc.'/'Corporation',
         e.g. '"Apple Inc." OR "Apple"'. Both are quoted phrases so the
         match stays tied to the company, not stray keywords.
      3. Fall back to the ticker symbol if no name is available.
    """
    if isinstance(cfg, dict) and cfg.get('news'):
        return str(cfg['news'])
    if not name:
        return ticker
    full = name.strip()
    short = full
    for suf in (' Inc.', ' Inc', ' Corporation', ' Corp.', ' Corp',
                ' Company', ' Co.', ' Co', ' Ltd.', ' Ltd', ' plc',
                ' Holdings', ' Group', ' Technologies', ' Limited', ','):
        if short.endswith(suf):
            short = short[: -len(suf)].strip()
    if short and short.lower() != full.lower() and len(short) >= 4:
        return f'"{full}" OR "{short}"'
    return f'"{full}"'


def _short_name(name: str) -> str:
    """Company name with common corporate suffixes stripped."""
    short = (name or "").strip()
    for suf in (' Inc.', ' Inc', ' Corporation', ' Corp.', ' Corp',
                ' Company', ' Co.', ' Co', ' Ltd.', ' Ltd', ' plc',
                ' Holdings', ' Group', ' Technologies', ' Limited', ','):
        if short.endswith(suf):
            short = short[: -len(suf)].strip()
    return short


def _gnews_queries(ticker: str, name: str, cfg=None) -> list:
    """Candidate Google News queries to try IN ORDER (first non-empty wins).

    Google News works best with a plain, recognizable name — NOT GDELT's
    quoted-boolean syntax, which can return nothing. So we try the short
    name first ('Apple'), then the full name ('Apple Inc.'), then the
    ticker. An explicit 'news' override (cfg['news']) wins outright.
    """
    if isinstance(cfg, dict) and cfg.get('news'):
        return [str(cfg['news'])]
    cands = []
    short = _short_name(name)
    if short:
        cands.append(short)
    if name and name.strip() and name.strip() != short:
        cands.append(name.strip())
    if ticker:
        cands.append(ticker)
    # de-dupe, keep order
    seen, out = set(), []
    for c in cands:
        if c and c.lower() not in seen:
            seen.add(c.lower()); out.append(c)
    return out or [ticker]


YEARS_HISTORY      = 6          # more data -> more walk-forward folds (was 3)
PREDICTION_HORIZON = 10         # trading days ahead (~2 trading weeks)
LOOKBACK           = 20         # feature window length
RIDGE_ALPHA        = 5.0
N_WALK_FOLDS       = 6
MIN_TRAIN          = 250
COST_BPS           = 5.0        # round-trip cost (commission + slippage), basis points
SEED               = 42

# --- SELECTIVE TRADE FILTER ---
# Instead of trading every period, only trade when BOTH conditions hold:
#   (1) High conviction: |prediction| is in the top tier of the model's own
#       TRAINING predictions (threshold learned per-fold from train data only,
#       so there is no look-ahead).
#   (2) Agreement: the model's direction matches the technical posture
#       (trend/momentum/RSI) on that date.
# The idea: the model's strongest, confirmed calls may be its least noisy.
# This is a hypothesis, not a guarantee — the honest metrics will judge it.
USE_SELECTIVE_FILTER = True
CONVICTION_PCTL      = 60       # trade only if |pred| above this pct of train |pred|

# --- NEWS FEATURES (model input, leakage-safe) ---
# Set USE_NEWS_FEATURES = False to get the baseline price-only model.
# To judge whether news actually helps, run BOTH and compare the honest
# metrics (binomial p-value, Spearman IC, net-of-cost Sharpe).
# Requires `news_features.py` in the same folder and the `requests`
# package. If GDELT can't be reached, the features fall back to neutral
# (0.0) and the model behaves exactly like the baseline — no crash.
USE_NEWS_FEATURES = True
NEWS_HALFLIFE     = 3.0          # headline weight halves every N days
NEWS_CACHE_DIR    = "news_cache" # results cached here (repeatable)

# --- NEWS SOURCE ---
# "google" -> Google News RSS (recent headlines, no key, sentiment from a
#             small finance lexicon). "gdelt" -> original GDELT timeline.
NEWS_SOURCE       = "google"
GNEWS_HL          = "en-US"      # Google News UI language
GNEWS_GL          = "US"         # Google News country edition
GNEWS_CEID        = "US:en"      # country:language code
GNEWS_SENTIMENT   = True         # derive headline sentiment for the model
# For Turkish-language coverage instead, use: hl="tr", gl="TR", ceid="TR:tr"

# Source-aware text used in the HTML news block.
if NEWS_SOURCE == "google":
    _NEWS_SRC_LABEL = "Google News"
    _NEWS_NOTE = (f"Modele, başlıklardan çıkarılan <i>günlük ortalama duygu</i> ve "
                  f"<i>haber sayısı</i> girer (üstel sönümleme · yarı-ömür "
                  f"{NEWS_HALFLIFE:.0f} gün · 1 gün gecikmeli, sızıntısız).")
else:
    _NEWS_SRC_LABEL = "GDELT"
    _NEWS_NOTE = (f"Modele <i>tek tek başlıklar değil</i>, GDELT'in günlük ortalama "
                  f"tonu ve haber hacmi girer (üstel sönümleme · yarı-ömür "
                  f"{NEWS_HALFLIFE:.0f} gün · 1 gün gecikmeli, sızıntısız).")

# --- NEWS LIST IN THE HTML REPORT ---
# Show, under each stock's info, the news that fed the model: the current
# decayed sentiment/volume values that actually enter the live signal, the
# recent days that had coverage, and a sample of real headlines from GDELT.
NEWS_SHOW_IN_REPORT = True       # render the per-stock news list in the HTML
NEWS_MAX_HEADLINES  = 15         # how many recent headlines to list per stock
NEWS_MAX_DAYS_SHOWN = 15         # how many recent coverage days to list per stock

FEATURE_COLS = ['mom5', 'mom20', 'vol20', 'vol60', 'sma_ratio', 'rsi14', 'dist_high', 'vix']
if USE_NEWS_FEATURES:
    FEATURE_COLS = FEATURE_COLS + ['news_sent', 'news_buzz']


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


def build_features(close: pd.Series, vix: pd.Series | None,
                   news_daily: pd.DataFrame | None = None) -> pd.DataFrame:
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
    conv_thr: float = 0.0          # per-fold conviction threshold (from TRAIN preds)


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
    all_conv: np.ndarray = None    # per-point conviction threshold
    all_tech: np.ndarray = None    # per-point technical score (for agreement)
    strat_curve: np.ndarray = None
    bh_curve: np.ndarray = None
    latest_pred: float = 0.0       # model's view on the most recent (unrealized) point
    # --- selective-filter results (filled when USE_SELECTIVE_FILTER) ---
    filt_used: bool = False
    filt_return: float = 0.0
    filt_sharpe: float = 0.0
    filt_max_dd: float = 0.0
    filt_trades: int = 0
    filt_dir_acc: float = 0.0      # accuracy on traded subset only
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

        # WINSORIZE training target: clip extreme outliers (1st/99th pct) so the
        # model is not dragged by rare crash/melt-up moves and won't extrapolate
        # to absurd values like +33% / -28% over 21 days.
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
        pred = np.clip(pred, clip_lo, clip_hi)           # keep predictions realistic

        # Conviction threshold learned ONLY from training predictions (no look-ahead):
        # the CONVICTION_PCTL percentile of |train preds|. Test trades only fire
        # when |test pred| exceeds this.
        tr_pred = ysc.inverse_transform(model.predict(Xtr_s).reshape(-1, 1)).ravel()
        conv_thr = float(np.percentile(np.abs(tr_pred), CONVICTION_PCTL))

        folds.append(FoldResult(dte, yte, pred, conv_thr=conv_thr))
        last_model, last_xs, last_ys = model, xs, ysc
        last_clip = (clip_lo, clip_hi)

    if not folds:
        return res

    all_true = np.concatenate([f.y_true for f in folds])
    all_pred = np.concatenate([f.y_pred for f in folds])
    all_dates = [d for f in folds for d in f.dates]
    all_conv = np.concatenate([np.full(len(f.y_pred), f.conv_thr) for f in folds])

    # technical score per test date (model-independent), aligned to all_dates
    all_tech = np.array([_technical_score(feat.loc[d]) if d in feat.index else 0.0
                         for d in all_dates])

    res.ok = True
    res.folds = folds
    res.all_true, res.all_pred, res.all_dates = all_true, all_pred, all_dates
    res.all_conv, res.all_tech = all_conv, all_tech
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
    if USE_SELECTIVE_FILTER:
        _equity_filtered(res, all_dates, all_true, all_pred, all_conv, all_tech)

    # Live prediction for the most recent point (forward-looking, unrealized)
    live = _latest_live_window(feat)
    if live is not None and last_model is not None:
        live_s = last_xs.transform(live.reshape(1, -1))
        lp = float(last_ys.inverse_transform(
            last_model.predict(live_s).reshape(-1, 1)).ravel()[0])
        if last_clip is not None:                        # keep live forecast realistic too
            lp = float(np.clip(lp, last_clip[0], last_clip[1]))
        res.latest_pred = lp
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


def _equity_filtered(res, dates, y_true, y_pred, conv, tech):
    """Selective strategy: trade a non-overlapping period ONLY when BOTH
       (1) |pred| >= that fold's conviction threshold (high conviction), AND
       (2) sign(pred) == sign(tech score) (model agrees with technical posture).
    Otherwise stay flat (0 return) for that period. Costs charged only on trades."""
    order = np.argsort(dates)
    yt = y_true[order]; yp = y_pred[order]
    cv = conv[order]; tc = tech[order]
    cost = COST_BPS / 1e4

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


def _technical_score(row) -> float:
    """Composite technical posture in [-1,1] from one feature row:
    trend (SMA ratio) + momentum (5/20d) + RSI tilt. Model-independent."""
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
                label=f'Strategy — all trades ({res.n_trades})')
        if res.filt_used and res.filt_curve is not None:
            ax.plot(res.filt_curve, lw=2, color='#7c3aed', marker='s', ms=3,
                    label=f'Selective filter ({res.filt_trades} trades)')
        ax.plot(res.bh_curve, lw=2, color='#1e3a5f', marker='o', ms=3, label='Buy & Hold')
        ax.axhline(1.0, color='#666', ls='--', alpha=0.5)
    ax.set_title('Equity Curve — Non-Overlapping Holds, Net of Costs',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Growth of $1'); ax.set_xlabel('Period #')
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


# ---------- end new charts ----------


# ============================================================
# ANALYZE ONE TICKER
# ============================================================
def analyze(ticker: str, name: str, vix: pd.Series | None, news_q: str | None = None,
            cfg=None):
    print(f"\n{'='*62}\n  {ticker} — {name}\n{'='*62}")
    close = download_close(ticker)
    if close is None or len(close) < 300:
        n = 0 if close is None else len(close)
        print("   No / insufficient data")
        if close is None:
            reason = "Fiyat verisi indirilemedi (ticker bulunamadı veya ağ hatası)."
            last = None
        else:
            reason = (f"Yetersiz geçmiş: yalnızca {n} işlem günü mevcut "
                      f"(en az 300 gerekli). Muhtemelen yeni halka açılan / "
                      f"yeni ayrılan bir şirket.")
            last = float(close.iloc[-1]) if len(close) else None
        return dict(ticker=ticker, name=name, skipped=True,
                    reason=reason, days=n, cur=last)
    print(f"   {len(close)} trading days (~{YEARS_HISTORY}y)")

    # News input: fetch recent coverage for THIS company. Failure (no network,
    # bad query) -> neutral features. Source set by NEWS_SOURCE.
    news_daily = None
    news_articles = None
    news_query = None
    if USE_NEWS_FEATURES:
        try:
            q = news_q or (f'"{name}"' if name else ticker)
            news_query = q
            if NEWS_SOURCE == "google":
                cands = _gnews_queries(ticker, name, cfg)
                news_articles = None
                used_q = cands[0] if cands else q
                for cand in cands:
                    news_articles = fetch_news_google_rss(
                        cand, NEWS_CACHE_DIR, max_records=max(NEWS_MAX_HEADLINES * 4, 60),
                        hl=GNEWS_HL, gl=GNEWS_GL, ceid=GNEWS_CEID)
                    if news_articles is not None and len(news_articles):
                        used_q = cand
                        break        # first query that returns something wins
                news_query = used_q
                news_daily = _articles_to_daily(news_articles,
                                                use_sentiment=GNEWS_SENTIMENT)
                ndays = 0 if news_daily is None else int((news_daily['buzz'] > 0).sum())
                print(f"   News [Google]: {0 if news_articles is None else len(news_articles)} "
                      f"articles, {ndays} days with coverage (query {used_q!r})")
            else:
                s = close.index.min().strftime("%Y%m%d")
                e = close.index.max().strftime("%Y%m%d")
                news_daily = fetch_news_gdelt_timeline(q, s, e, NEWS_CACHE_DIR)
                ndays = 0 if news_daily is None else int((news_daily['buzz'] > 0).sum())
                print(f"   News [GDELT]: {ndays} days with coverage (query {q})")
                if NEWS_SHOW_IN_REPORT:
                    news_articles = fetch_news_gdelt_articles(
                        q, s, e, NEWS_CACHE_DIR, max_records=max(NEWS_MAX_HEADLINES * 3, 45))
                    print(f"   News headlines fetched: "
                          f"{0 if news_articles is None else len(news_articles)}")
        except Exception as ex:
            print(f"   News fetch skipped ({ex}); using neutral features")
            news_daily = None

    feat = build_features(close, vix, news_daily)
    cur = float(close.iloc[-1])
    chg = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    cur_vix = float(vix.dropna().iloc[-1]) if (vix is not None and len(vix.dropna())) else 20.0
    print(f"   Last close: ${cur:.2f} ({chg:+.2f}%)  VIX={cur_vix:.1f}")

    res = walk_forward(feat)
    if not res.ok:
        print("   Backtest could not run (insufficient folds)")
        need = MIN_TRAIN + N_WALK_FOLDS * 10
        reason = (f"Backtest çalıştırılamadı: {len(close)} işlem günü "
                  f"walk-forward için yeterli örnek üretmiyor. Özellik ısınması "
                  f"(~80 gün) ve {PREDICTION_HORIZON} günlük hedef ufku düşülünce "
                  f"gereken ~{need} örneğin altında kalıyor (≈400 temiz işlem günü "
                  f"/ ~1,6 yıl gerekir).")
        return dict(ticker=ticker, name=name, skipped=True,
                    reason=reason, days=len(close), cur=cur)

    lbl, color, expl = verdict(res)
    print(f"\n   ── WALK-FORWARD RESULTS ──")
    print(f"   Test obs: {res.n_obs} (across {len(res.folds)} folds)")
    print(f"   Dir Acc: {res.dir_acc:.1f}% | Naive: {res.naive:.1f}% | Edge: {res.edge:+.2f}%")
    print(f"   Significance p-value: {res.p_value:.4f}  -> {'SIGNIFICANT' if res.significant else 'not significant'}")
    print(f"   Info Coefficient (Spearman): {res.ic:+.3f} (p={res.ic_p:.3f})")
    print(f"   Strategy net: {res.strat_return:+.1f}% | Buy&Hold: {res.bh_return:+.1f}% | "
          f"Sharpe: {res.sharpe:.2f} | MaxDD: {res.max_dd:.1f}%")
    if res.filt_used:
        print(f"   FILTERED (selective): {res.filt_return:+.1f}% | "
              f"Sharpe: {res.filt_sharpe:.2f} | MaxDD: {res.filt_max_dd:.1f}% | "
              f"Trades: {res.filt_trades} (acc {res.filt_dir_acc:.1f}%)")
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

    # --- News summary for the report (what actually fed the model) ---
    news_info = _build_news_info(feat, news_daily, news_articles, news_query)

    return dict(
        ticker=ticker, name=name, cur=cur, chg=chg, cur_vix=cur_vix,
        hi52=hi52, lo52=lo52, pos52=pos52,
        res=res, label=lbl, color=color, expl=expl, sig=sig,
        news=news_info,
        img_price=plot_price(close, res),
        img_equity=plot_equity(res),
        img_diag=plot_diagnostics(res),
        img_drawdown=plot_drawdown(close),
        img_bollinger=plot_bollinger(close),
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
.skip-note{font-family:'Helvetica Neue',Arial,sans-serif;background:#fbf6ea;
  border:1px solid #e8dcc0;border-left:4px solid #c9a227;border-radius:8px;
  padding:18px 22px;margin-bottom:10px}
.skip-badge{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  font-weight:700;color:#9a7b12;margin-bottom:6px}
.skip-reason{font-size:14px;color:var(--ink)}
.skip-meta{font-size:12.5px;color:var(--muted);margin-top:8px}
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
.news-block{background:var(--card);border:1px solid var(--line);border-left:4px solid #2563eb;
  border-radius:8px;margin:0 0 22px;padding:0;overflow:hidden}
.news-block>summary{font-family:'Helvetica Neue',Arial,sans-serif;font-size:14px;
  font-weight:700;color:var(--ink);padding:14px 18px;cursor:pointer;list-style:none;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.news-block>summary::-webkit-details-marker{display:none}
.news-block>summary::before{content:"▸";color:#2563eb;font-weight:700;margin-right:2px}
.news-block[open]>summary::before{content:"▾"}
.news-block .muted{color:var(--muted);font-weight:500;font-size:12.5px}
.news-body{padding:4px 18px 18px;font-family:'Helvetica Neue',Arial,sans-serif}
.news-model{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:10px;margin:6px 0 12px}
.nm-stat{background:var(--bg,#fafafa);border:1px solid var(--line);border-radius:7px;
  padding:10px 12px}
.nm-stat span{display:block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;
  color:var(--muted);margin-bottom:4px}
.nm-stat b{font-size:18px}
.news-note{font-size:12px;color:var(--muted);line-height:1.5;margin:8px 0 14px}
.news-note b,.news-note i{color:var(--ink);font-style:normal}
.news-cols{display:grid;grid-template-columns:1.4fr 1fr;gap:20px}
@media(max-width:720px){.news-cols{grid-template-columns:1fr}}
.news-cols h4{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  margin:0 0 8px;border-bottom:1px solid var(--line);padding-bottom:5px}
.news-list{list-style:none;margin:0;padding:0}
.news-list li{font-size:13px;line-height:1.5;padding:7px 0;border-bottom:1px solid var(--line)}
.news-list li:last-child{border-bottom:none}
.news-list a{color:#1d4ed8;text-decoration:none}
.news-list a:hover{text-decoration:underline}
.news-list .n-date{display:inline-block;font-size:11px;color:var(--muted);
  font-variant-numeric:tabular-nums;margin-right:6px}
.news-list .n-dom{display:inline-block;font-size:11px;color:var(--muted);margin-left:4px}
.news-days{width:100%;border-collapse:collapse;font-size:12.5px}
.news-days th{text-align:left;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--line);padding:5px 6px}
.news-days td{padding:5px 6px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
</style>"""


def fmt_pct(x):
    cls = 'pos' if x > 0 else ('neg' if x < 0 else '')
    return f'<span class="{cls}">{x:+.2f}%</span>'


def _esc(s) -> str:
    """Minimal HTML escaping for headline text/URLs."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _build_news_info(feat, news_daily, news_articles, query) -> dict:
    """Assemble the per-stock news payload shown in the HTML report.

    Keys: used, query, cov_days, latest_sent, latest_buzz,
          daily [{date,sent,buzz}], articles [{date,title,url,domain}].
    latest_sent/buzz are the CURRENT decayed values that literally enter
    the live model signal; daily/articles are context for the reader.
    """
    info = {'used': False, 'query': query, 'cov_days': 0,
            'latest_sent': 0.0, 'latest_buzz': 0.0, 'daily': [], 'articles': []}
    if not USE_NEWS_FEATURES:
        return info

    if 'news_sent' in feat.columns and len(feat):
        info['latest_sent'] = float(feat['news_sent'].iloc[-1])
    if 'news_buzz' in feat.columns and len(feat):
        info['latest_buzz'] = float(feat['news_buzz'].iloc[-1])

    if news_daily is not None and len(news_daily):
        nd = news_daily.copy()
        nd.index = pd.to_datetime(nd.index).normalize()
        if 'buzz' not in nd.columns:
            nd['buzz'] = 0.0
        if 'sent' not in nd.columns:
            nd['sent'] = 0.0
        cov = nd[nd['buzz'] > 0].sort_index()
        info['cov_days'] = int(len(cov))
        info['used'] = info['cov_days'] > 0
        for day, row in cov.tail(NEWS_MAX_DAYS_SHOWN)[::-1].iterrows():
            info['daily'].append({'date': day.strftime('%Y-%m-%d'),
                                  'sent': float(row['sent']),
                                  'buzz': float(row['buzz'])})

    if news_articles is not None and len(news_articles):
        arts = news_articles.copy()
        if 'date' in arts.columns:
            arts = arts.sort_values('date', ascending=False, na_position='last')
        for _, a in arts.head(NEWS_MAX_HEADLINES).iterrows():
            d = a.get('date', None)
            try:
                ds = pd.to_datetime(d).strftime('%Y-%m-%d') if pd.notna(d) else ''
            except Exception:
                ds = ''
            info['articles'].append({'date': ds,
                                     'title': str(a.get('title', '')),
                                     'url': str(a.get('url', '')),
                                     'domain': str(a.get('domain', ''))})
        if info['articles']:
            info['used'] = True
    return info


def _sent_word_color(s: float):
    if s > 0.15:
        return 'Pozitif', '#16a34a'
    if s < -0.15:
        return 'Negatif', '#dc2626'
    return 'Nötr', '#64748b'


def _news_block_html(r) -> str:
    """Per-stock news list shown directly under the stock's price info."""
    if not NEWS_SHOW_IN_REPORT:
        return ""
    info = r.get('news') or {}
    s = float(info.get('latest_sent', 0.0))
    s_word, s_col = _sent_word_color(s)

    if not info.get('used'):
        return f"""
    <details class="news-block">
      <summary>📰 Hesaplamaya Katılan Haberler <span class="muted">· veri yok / nötr (0.0)</span></summary>
      <div class="news-body">
        <p class="news-note">Bu hisse için {_NEWS_SRC_LABEL}'tan haber verisi gelmedi (ağ yok ya da
        sorgu boş döndü); modele haber katkısı <b>nötr (0.0)</b> olarak girdi — bu hissede
        sonuç saf fiyat-tabanlı baseline ile aynıdır.</p>
      </div>
    </details>"""

    arts_html = ""
    for a in info.get('articles', []):
        title, url = _esc(a['title']), _esc(a['url'])
        dom, dt = _esc(a['domain']), _esc(a['date'])
        link = (f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
                if url else title)
        arts_html += (f'<li><span class="n-date">{dt}</span>{link}'
                      f'<span class="n-dom">{dom}</span></li>')
    if not arts_html:
        arts_html = ('<li class="muted">Başlık örneği alınamadı — modele yalnızca '
                     'toplu günlük ton/hacim verisi girdi.</li>')

    days_html = ""
    for d in info.get('daily', []):
        sv = d['sent']
        _, scol = _sent_word_color(sv)
        days_html += (f'<tr><td>{_esc(d["date"])}</td>'
                      f'<td style="color:{scol};font-weight:600">{sv:+.2f}</td>'
                      f'<td>{int(round(d["buzz"]))}</td></tr>')
    if not days_html:
        days_html = '<tr><td colspan="3" class="muted">—</td></tr>'

    return f"""
    <details class="news-block" open>
      <summary>📰 Hesaplamaya Katılan Haberler · <b style="color:{s_col}">{s_word}</b>
        <span class="muted">· {info.get('cov_days', 0)} gün haber kapsamı</span></summary>
      <div class="news-body">
        <div class="news-model">
          <div class="nm-stat"><span>Modele giren güncel ton</span>
            <b style="color:{s_col}">{s:+.3f}</b></div>
          <div class="nm-stat"><span>Haber yoğunluğu (log, modele giren)</span>
            <b>{info.get('latest_buzz', 0.0):+.3f}</b></div>
          <div class="nm-stat"><span>Haber kapsamı olan gün</span>
            <b>{info.get('cov_days', 0)}</b></div>
        </div>
        <p class="news-note">{_NEWS_NOTE} Aşağıdaki başlıklar
          {_NEWS_SRC_LABEL} üzerinden bu hisseye ait son haberlerdir; sayısal etki
          yukarıdaki "modele giren" değerlerdedir.</p>
        <div class="news-cols">
          <div>
            <h4>Bu hisseye ait son {len(info.get('articles', []))} haber</h4>
            <ul class="news-list">{arts_html}</ul>
          </div>
          <div>
            <h4>Haber kapsamı olan son günler</h4>
            <table class="news-days">
              <thead><tr><th>Tarih</th><th>Ort. ton</th><th>Makale</th></tr></thead>
              <tbody>{days_html}</tbody>
            </table>
          </div>
        </div>
      </div>
    </details>"""


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


def build_html(results):
    now = datetime.now().strftime('%B %d, %Y · %H:%M')
    blocks = ""
    for r in results:
        if r is None:
            continue
        if r.get('skipped'):
            price_line = (f'<div class="skip-meta">Son fiyat: ${r["cur"]:,.2f}'
                          f' · {r["days"]} işlem günü</div>'
                          if r.get('cur') is not None else '')
            blocks += f"""
  <section class="ticker-block">
    <div class="tk-head">
      <span class="tk-sym">{r['ticker']}</span>
      <span class="tk-name">{r['name']}</span>
    </div>
    <div class="skip-note">
      <div class="skip-badge">Analiz Edilemedi</div>
      <div class="skip-reason">{r['reason']}</div>
      {price_line}
    </div>
  </section>"""
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
      <div class="stat"><div class="l">Directional Acc.</div><div class="tr">Yön Doğruluğu</div>
        <div class="v">{res.dir_acc:.1f}<small>%</small></div>
        <div class="desc">Modelin yukarı/aşağı yönü kaç kez doğru bildiği. %50 yazı-turadır; yüksek olması iyidir.</div></div>
      <div class="stat"><div class="l">Naive Baseline</div><div class="tr">Naif Referans</div>
        <div class="v">{res.naive:.1f}<small>%</small></div>
        <div class="desc">"Her zaman çoğunluk yönü" deseydik elde edilecek doğruluk. Modelin geçmesi gereken eşik.</div></div>
      <div class="stat"><div class="l">Edge</div><div class="tr">Üstünlük (Fark)</div>
        <div class="v">{fmt_pct(res.edge)}</div>
        <div class="desc">Yön doğruluğu eksi naif referans. Pozitifse model referansı yener; negatifse referans daha iyi.</div></div>
      <div class="stat"><div class="l">Significance</div><div class="tr">İstatistiksel Anlamlılık</div>
        <div class="v" style="font-size:18px">{sig_txt}</div>
        <div class="desc">Üstünlüğün şansa bağlı olma olasılığı (p). 0.05 altı = anlamlı. Yüksekse sonuç tesadüfi olabilir.</div></div>
      <div class="stat"><div class="l">Info Coefficient</div><div class="tr">Bilgi Katsayısı</div>
        <div class="v">{res.ic:+.3f}</div>
        <div class="desc">Tahmin ile gerçek getiri arasındaki sıralama korelasyonu (-1..+1). Pozitif ve yüksek olması istenir.</div></div>
    </div>
    <div class="stat-row">
      <div class="stat"><div class="l">Strategy (net)</div><div class="tr">Strateji Getirisi (net)</div>
        <div class="v">{fmt_pct(res.strat_return)}</div>
        <div class="desc">Modelin sinyallerine uyulsaydı, masraflar düşülünce elde edilecek toplam getiri.</div></div>
      <div class="stat"><div class="l">Buy &amp; Hold</div><div class="tr">Al ve Tut</div>
        <div class="v">{fmt_pct(res.bh_return)}</div>
        <div class="desc">Hisseyi alıp hiç işlem yapmadan tutmanın getirisi. Stratejinin geçmesi gereken kıyas.</div></div>
      <div class="stat"><div class="l">Sharpe</div><div class="tr">Sharpe Oranı</div>
        <div class="v">{res.sharpe:.2f}</div>
        <div class="desc">Alınan risk başına getiri. 1 üstü iyi, 0 altı kötü. Riske göre düzeltilmiş performans.</div></div>
      <div class="stat"><div class="l">Max Drawdown</div><div class="tr">Maks. Düşüş</div>
        <div class="v">{res.max_dd:.1f}<small>%</small></div>
        <div class="desc">Zirveden en dip noktaya yaşanan en büyük kayıp. Riskin/acının ölçüsü; sıfıra yakın iyidir.</div></div>
      <div class="stat"><div class="l">Test Obs / Trades</div><div class="tr">Test Gözlemi / İşlem</div>
        <div class="v">{res.n_obs}<small>/{res.n_trades}</small></div>
        <div class="desc">Değerlendirmede kullanılan örnek sayısı ve yapılan işlem adedi. Daha fazla gözlem daha güvenilir.</div></div>
    </div>
    {_filter_row_html(res)}

    <div class="fig"><h3>Price History &amp; {PREDICTION_HORIZON}-Day Forward Forecast</h3><img src="{r['img_price']}"></div>
    <div class="fig"><h3>Moving Averages &amp; Bollinger Bands</h3><img src="{r['img_bollinger']}"></div>
    <div class="fig"><h3>Drawdown from Peak (Risk View)</h3><img src="{r['img_drawdown']}"></div>
    <div class="fig"><h3>Equity Curve (net of {COST_BPS:.0f}bps costs)</h3><img src="{r['img_equity']}"></div>
    <div class="fig"><h3>Diagnostics</h3><img src="{r['img_diag']}"></div>

    {_news_block_html(r)}
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
    for tk, cfg in TICKERS.items():
        name, _bench = _ticker_meta(cfg)
        nq = _news_query(tk, name, cfg)
        try:
            results.append(analyze(tk, name, vix, news_q=nq, cfg=cfg))
        except Exception as e:
            print(f"   error on {tk}: {e}")
            results.append(None)

    html = build_html(results)
    out = "MultiStock_v26.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report -> {out}")
    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
