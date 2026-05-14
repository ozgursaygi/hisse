# ============================================================
# MULTI-STOCK v15.3 — EWT (Elliott Wave Theory) FEATURES EKLENDİ
#
# v15.2 üzerine değişiklikler:
#
# [EWT-1] FİBONACCİ SEVİYELERİ (eSignal Bölüm 10 istatistikleri):
#   Hissenin son 60g range içindeki Fib geri çekilme seviyelerine
#   göre konumu. eSignal: Wave2 %73'ü 0.50-0.62 bandında bitirir.
#   → fib_dist_382, fib_dist_500, fib_dist_618, fib_dist_786
#   → range_pos (0=dip, 1=tavan)
#
# [EWT-2] HURST EXPONENT (ritmik trend kalıcılığı):
#   H > 0.55 → trend persistent (EWT'nin impuls dalga sezgisi)
#   H < 0.45 → mean-reverting (EWT'nin düzeltme sezgisi)
#   H ≈ 0.50 → random walk
#   → hurst_60 (60-günlük rolling, causal)
#
# [EWT-3] SWING STRUCTURE (pivot analizi):
#   Son 60g içindeki swing high/low sayısı ve amplitüdü.
#   EWT'nin "5 dalga impuls + 3 dalga düzeltme" yapısının
#   operasyonel karşılığı.
#   → pivot_count_60, swing_amp_60
#
# [EWT-4] WAVE MOMENTUM (dalga ivmesi):
#   Kısa vade momentum / orta vade momentum oranı.
#   EWT Wave 3'ün en güçlü dalga olduğu kuralının testi.
#   → wave_momentum (Mom_5 / Mom_20 oranı)
#
# [EWT-5] CHANNEL POSITION (Elliott Channel):
#   Fiyatın son 60g lineer trend kanalındaki konumu.
#   EWT kanalları Wave 4 desteği ve Wave 5 projeksiyonu için kullanır.
#   → chan_pos (0=kanal tabanı, 1=kanal tavanı)
#   → chan_slope (kanalın eğimi, normalize)
#   → chan_width (kanalın genişliği, volatilite proxy)
#
# [EWT-6] BENCHMARKa GÖRE EWT (sektör rölatif):
#   Aynı EWT features'lar benchmark (SOXX) üzerinde de hesaplanır.
#   Hisse−benchmark farkı, rölatif momentum ve dalga pozisyonu verir.
#   → bench_hurst_60, bench_chan_pos
#
# Tüm v15.2 fix'leri korundu:
#   FIX-1: StandardScaler (y_sc bias düzeltmesi)
#   FIX-2: Non-overlapping backtest (kümülatif alfa)
#   FIX-3: Regime prob boyut hizalaması
#   FIX-4: Seed izolasyonu
# ============================================================

import os, sqlite3, warnings, random, base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal as sp_signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow import random as tf_random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# ============================================================
# 🎯 KONFİGÜRASYON
# ============================================================
TICKERS_CONFIG = {
    'INTC': {'name': 'Intel Corporation',     'benchmark': 'SOXX'},
    'AMD':  {'name': 'Advanced Micro Devices', 'benchmark': 'SOXX'},
    'NVDA': {'name': 'NVIDIA Corporation',     'benchmark': 'SOXX'},
    # 'AAPL': {'name': 'Apple Inc.',       'benchmark': 'SPY'},
    # 'TSLA': {'name': 'Tesla Inc.',       'benchmark': 'SPY'},
    # 'JPM':  {'name': 'JPMorgan Chase',   'benchmark': 'XLF'},
}

PREDICTION_HORIZON   = 7
TRANSACTION_COST_BPS = 5
LOOKBACK             = 60
TRAIN_FRAC           = 0.80
VAL_FRAC             = 0.10
ENSEMBLE_SEEDS       = [42, 52, 62]
EPOCHS               = 60
DB_FOLDER            = r"C:\Projects\ML"
OUTPUT_HTML          = "MultiStock_Analiz_v15.3.html"

# EWT parametreleri
EWT_WINDOW    = 60     # Fibonacci ve swing analizi penceresi
EWT_HURST_W   = 60     # Hurst hesaplama penceresi
EWT_CHAN_W    = 60     # Elliott Channel penceresi

# ============================================================
# YARDIMCI
# ============================================================

def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf_random.set_seed(seed)

# ============================================================
# VERİ
# ============================================================

def download_pair(ticker, benchmark):
    end   = datetime.now()
    start = end - timedelta(days=12 * 365)
    try:
        ds = yf.download(ticker,    start=start, end=end, progress=False, auto_adjust=False)
        db = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"   HATA {ticker}: {e}"); return None, None
    if ds is None or ds.empty or db is None or db.empty: return None, None
    for df in [ds, db]:
        if hasattr(df.columns, 'get_level_values'):
            df.columns = df.columns.get_level_values(0)
    ds = ds[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    db = db[['Close']].rename(columns={'Close': 'Bench_Close'}).dropna()
    idx = ds.index.intersection(db.index)
    return ds.loc[idx], db.loc[idx]

def get_macro():
    end  = datetime.now(); start = end - timedelta(days=12 * 365)
    tmap = {"^VIX":"VIX","^TNX":"TNX","DX-Y.NYB":"DXY","^GSPC":"SP500","^IXIC":"NASDAQ"}
    try:
        df = yf.download(list(tmap.keys()), start=start, end=end, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.rename(columns=tmap, inplace=True)
        return df.ffill()
    except:
        return pd.DataFrame()

# ============================================================
# REJİM TESPİTİ
# ============================================================

def detect_regime(bench_close):
    log_ret   = np.log(bench_close / bench_close.shift(1))
    sma50     = bench_close.rolling(50).mean()
    sma200    = bench_close.rolling(200).mean()
    trend_str = 1.0 / (1.0 + np.exp(-((sma50 / sma200) - 1.0) * 50))
    vol       = log_ret.rolling(20).std()
    vol_q     = vol.expanding(min_periods=252).quantile(0.70)
    p_chaos   = (1.0 / (1.0 + np.exp(-(vol / vol_q - 1.0) * 5))).clip(0.05, 0.95)
    rem       = 1.0 - p_chaos
    p_bull    = rem * trend_str
    p_bear    = rem * (1.0 - trend_str)
    df_r = pd.DataFrame({'p_bull': p_bull, 'p_bear': p_bear,
                         'p_chaos': p_chaos, 'vol_20': vol},
                        index=bench_close.index)
    probs = df_r[['p_bull','p_bear','p_chaos']].values
    df_r['regime'] = np.array(['bull','bear','chaos'])[np.argmax(probs, axis=1)]
    return df_r

def rolling_beta(sr, br, window=60):
    beta = np.full(len(sr), np.nan)
    sv, bv = sr.values, br.values
    for i in range(window, len(sv)):
        sw, bw = sv[i-window:i], bv[i-window:i]
        if np.std(bw) == 0: continue
        cov = np.cov(sw, bw)[0, 1]; vb = np.var(bw)
        beta[i] = cov / vb if vb > 0 else 1.0
    return pd.Series(beta, index=sr.index)

# ============================================================
# EWT FEATURES (tümü causal — geleceğe bakmaz)
# ============================================================

def ewt_hurst(series, max_lag=20):
    """Hurst exponent via R/S. H>0.55 trend, H<0.45 mean-revert, H≈0.5 random."""
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < max_lag + 4: return 0.5
    lags = range(2, min(max_lag, len(arr)//2))
    tau  = []
    for lag in lags:
        diff = arr[lag:] - arr[:-lag]
        sd   = np.std(diff)
        tau.append(sd if sd > 0 else 1e-10)
    try:
        ll = np.log(list(lags)); lt = np.log(tau)
        if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(lt))): return 0.5
        return float(np.polyfit(ll, lt, 1)[0] * 2.0)
    except: return 0.5

def ewt_fibonacci_features(close_series, high_series, low_series, window=EWT_WINDOW):
    """
    [EWT-1] Fibonacci seviyeleri.
    Son `window` günün max/min'e göre Fibonacci geri çekilme seviyelerine mesafe.
    eSignal: Wave2 %73'ü 0.50–0.62 bandında. Bu, fiyatın hangi Fib bölgesinde
    olduğunu ölçer.
    Tümü causal (rolling window).
    """
    n     = len(close_series)
    h_w   = high_series.rolling(window, min_periods=window//2).max()
    l_w   = low_series.rolling(window, min_periods=window//2).min()
    rng   = (h_w - l_w).replace(0, np.nan)
    # Fib seviyeleri (yükselişten düşüşe)
    f382  = l_w + 0.382 * rng
    f500  = l_w + 0.500 * rng
    f618  = l_w + 0.618 * rng
    f786  = l_w + 0.786 * rng
    out   = pd.DataFrame(index=close_series.index)
    out['fib_dist_382'] = (close_series - f382) / rng   # + = üstünde, - = altında
    out['fib_dist_500'] = (close_series - f500) / rng
    out['fib_dist_618'] = (close_series - f618) / rng
    out['fib_dist_786'] = (close_series - f786) / rng
    out['range_pos']    = (close_series - l_w)  / rng   # 0=dip, 1=tavan
    return out

def ewt_hurst_rolling(log_returns, window=EWT_HURST_W):
    """
    [EWT-2] Rolling Hurst exponent.
    Her t anında sadece [t-window, t) kullanılır — geleceğe bakmaz.
    """
    arr = log_returns.fillna(0).values
    n   = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = ewt_hurst(arr[i-window:i])
    return pd.Series(out, index=log_returns.index)

def ewt_swing_features(close_series, window=EWT_WINDOW):
    """
    [EWT-3] Swing structure — pivot sayısı ve amplitüdü.
    EWT'nin "dalga sayımı" kavramının operasyonel karşılığı.
    Adaptive prominence kullanır; causal.
    """
    arr         = close_series.values
    n           = len(arr)
    pivot_count = np.full(n, np.nan)
    swing_amp   = np.full(n, np.nan)
    for i in range(window, n):
        seg  = arr[i-window:i]
        prom = np.std(seg) * 1.0   # adaptive
        if prom <= 0: continue
        ph, _ = sp_signal.find_peaks(seg,  prominence=prom)
        pl, _ = sp_signal.find_peaks(-seg, prominence=prom)
        pivot_count[i] = len(ph) + len(pl)
        if len(ph) > 0 and len(pl) > 0:
            swing_amp[i] = (np.max(seg[ph]) - np.min(seg[pl])) / np.mean(seg)
    return (pd.Series(pivot_count, index=close_series.index),
            pd.Series(swing_amp,   index=close_series.index))

def ewt_channel_features(close_series, window=EWT_CHAN_W):
    """
    [EWT-5] Elliott Channel: lineer regresyon kanalı.
    Wave 4 desteği ve Wave 5 projeksiyonunda kullanılan kanal tekniğinin
    matematiksel operasyonelleştirilmesi.
    """
    arr = close_series.values
    n   = len(arr)
    pos = np.full(n, np.nan)
    slp = np.full(n, np.nan)
    wid = np.full(n, np.nan)
    x   = np.arange(window)
    for i in range(window, n):
        seg = arr[i-window:i]
        try:
            s, b = np.polyfit(x, seg, 1)
            fit  = s * x + b
            res  = seg - fit
            w    = np.max(res) - np.min(res)
            if w <= 0 or np.mean(seg) == 0: continue
            last_res = arr[i] - (s * window + b)
            pos[i]   = (last_res - np.min(res)) / w      # 0=taban, 1=tavan
            slp[i]   = s / (np.mean(seg) + 1e-9)         # normalize eğim
            wid[i]   = w / np.mean(seg)                  # normalize genişlik
        except: pass
    return (pd.Series(pos, index=close_series.index),
            pd.Series(slp, index=close_series.index),
            pd.Series(wid, index=close_series.index))

def add_ewt_features(df, close_col='Close', high_col='High',
                     low_col='Low', log_ret_col='Log_Ret', prefix=''):
    """
    Tüm EWT features'larını DataFrame'e ekler.
    prefix='bench_' → benchmark için de çağrılabilir [EWT-6].
    """
    p = prefix
    # [EWT-1] Fibonacci
    fib_df = ewt_fibonacci_features(df[close_col], df[high_col], df[low_col])
    for col in fib_df.columns:
        df[f'{p}{col}'] = fib_df[col].values

    # [EWT-2] Hurst
    df[f'{p}hurst_60'] = ewt_hurst_rolling(df[log_ret_col]).values

    # [EWT-3] Swing
    pc, sa = ewt_swing_features(df[close_col])
    df[f'{p}pivot_count_60'] = pc.values
    df[f'{p}swing_amp_60']   = sa.values

    # [EWT-4] Wave momentum (Wave 3 en güçlü dalga kuralı testi)
    mom5  = df[close_col].pct_change(5)
    mom20 = df[close_col].pct_change(20)
    denom = mom20.replace(0, np.nan)
    df[f'{p}wave_momentum'] = (mom5 / denom).clip(-5, 5)

    # [EWT-5] Channel
    cp, cs, cw = ewt_channel_features(df[close_col])
    df[f'{p}chan_pos']   = cp.values
    df[f'{p}chan_slope'] = cs.values
    df[f'{p}chan_width'] = cw.values

    return df

# ============================================================
# FEATURE ENGINEERING (klasik + EWT)
# ============================================================

def build_features(ds, db, macro, df_regime):
    df = ds.copy()
    df['Bench_Close'] = db['Bench_Close']
    try:
        import ta
        # ── Klasik teknik indikatörler ──
        df['RSI']         = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD']        = ta.trend.MACD(df['Close']).macd()
        df['MACD_Sig']    = ta.trend.MACD(df['Close']).macd_signal()
        df['ATR']         = ta.volatility.AverageTrueRange(df['High'],df['Low'],df['Close']).average_true_range()
        df['CCI']         = ta.trend.CCIIndicator(df['High'],df['Low'],df['Close']).cci()
        df['SMA20']       = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        df['SMA50']       = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
        df['SMA200']      = ta.trend.SMAIndicator(df['Close'],200).sma_indicator()
        bb                = ta.volatility.BollingerBands(df['Close'])
        df['BB_pct']      = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        df['Log_Ret']     = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_5']       = df['Log_Ret'].rolling(5).std()
        df['Vol_20']      = df['Log_Ret'].rolling(20).std()
        df['Mom_5']       = df['Close'].pct_change(5)
        df['Mom_20']      = df['Close'].pct_change(20)
        df['Vol_ratio']   = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Px_SMA50']    = df['Close'] / df['SMA50']
        df['Px_SMA200']   = df['Close'] / df['SMA200']

        # ── Sektör / benchmark features ──
        df['B_LogRet']    = np.log(df['Bench_Close'] / df['Bench_Close'].shift(1))
        df['B_Mom5']      = df['Bench_Close'].pct_change(5)
        df['B_Mom20']     = df['Bench_Close'].pct_change(20)
        df['B_Vol20']     = df['B_LogRet'].rolling(20).std()
        df['Rel_Mom5']    = df['Mom_5']  - df['B_Mom5']
        df['Rel_Mom20']   = df['Mom_20'] - df['B_Mom20']
        df['Rel_Vol20']   = df['Vol_20'] - df['B_Vol20']
        df['Rel_Str']     = df['Close']  / df['Bench_Close']
        df['Rel_Str_SMA'] = df['Rel_Str'].rolling(20).mean()
        df['Rel_Str_Dev'] = (df['Rel_Str'] - df['Rel_Str_SMA']) / df['Rel_Str_SMA']
        df['Beta_60']     = rolling_beta(df['Log_Ret'].fillna(0), df['B_LogRet'].fillna(0))
        df['Beta_60']     = df['Beta_60'].clip(0.0, 3.0)
        df['Corr_60']     = df['Log_Ret'].rolling(60).corr(df['B_LogRet'])

        # ── Rejim olasılıkları ──
        df = df.join(df_regime[['p_bull','p_bear','p_chaos']], how='left')

        # ── [EWT-1..5] Hisse üzerinde EWT features ──
        print(f"   EWT features hesaplanıyor (hisse)...")
        df = add_ewt_features(df, close_col='Close', high_col='High',
                              low_col='Low', log_ret_col='Log_Ret', prefix='')

        # ── [EWT-6] Benchmark üzerinde EWT features ──
        # Benchmark High/Low olmadığı için Close ile yaklaşık hesap yapılır
        print(f"   EWT features hesaplanıyor (benchmark)...")
        df['Bench_High'] = df['Bench_Close']   # SOXX için High=Low=Close yaklaşımı
        df['Bench_Low']  = df['Bench_Close']
        df['B_LogRet_col'] = df['B_LogRet']
        df = add_ewt_features(df, close_col='Bench_Close', high_col='Bench_High',
                              low_col='Bench_Low', log_ret_col='B_LogRet_col',
                              prefix='bench_')
        # Rölatif EWT features
        df['rel_hurst']    = df['hurst_60']    - df['bench_hurst_60']
        df['rel_chan_pos'] = df['chan_pos']     - df['bench_chan_pos']
        df['rel_pivots']   = df['pivot_count_60'] - df['bench_pivot_count_60']

        # ── Hedef: residual return ──
        fut_s = np.log(df['Close'].shift(-PREDICTION_HORIZON)       / df['Close'])
        fut_b = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON) / df['Bench_Close'])
        df['Fut_Stock'] = fut_s
        df['Fut_Bench'] = fut_b
        df['Fut_Resid'] = fut_s - df['Beta_60'] * fut_b

        if not macro.empty:
            df = df.join(macro, how='left').ffill()

        # Yardımcı kolonları çıkar
        df.drop(columns=['Bench_High','Bench_Low','B_LogRet_col'], errors='ignore', inplace=True)

        df = df.dropna(subset=['Fut_Resid','Beta_60','p_bull','p_bear','p_chaos'])
        df = df.dropna()
        return df
    except Exception as e:
        import traceback; print(f"   Feature Hata: {e}"); traceback.print_exc()
        return None

# ============================================================
# İSTATİSTİK
# ============================================================

def binom_p(c, t):
    if t < 10: return 1.0
    return stats.binomtest(c, t, p=0.5, alternative='greater').pvalue

def info_ratio(rets, ppy):
    arr = np.asarray(rets); arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    sd = np.std(arr, ddof=1)
    return float(np.sqrt(ppy) * np.mean(arr) / sd) if sd > 0 else 0.0

# ============================================================
# MODEL
# ============================================================

def make_dataset(X, y, lb):
    Xs, ys = [], []
    for i in range(lb, len(X)):
        Xs.append(X[i-lb:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm(shape):
    m = Sequential([
        Input(shape=shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def train_model(Xt, yt, Xv, yv, seed, epochs, sw=None):
    set_seeds(seed)
    m  = build_lstm((Xt.shape[1], Xt.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    kw = dict(epochs=epochs, batch_size=32, validation_data=(Xv,yv),
              callbacks=[es,rl], verbose=0)
    if sw is not None: kw['sample_weight'] = sw
    m.fit(Xt, yt, **kw)
    return m

def train_experts(Xt, yt, Xv, yv, tr_probs):
    experts = {}
    for i, name in enumerate(['bull','bear','chaos']):
        print(f"      🎓 {name.upper()}...")
        sw    = np.clip(tr_probs[:,i], 0.05, 1.0)
        eff_n = sw.sum()
        if eff_n < 100:
            print(f"         ⚠ yetersiz ({eff_n:.0f})"); experts[name] = None; continue
        experts[name] = [train_model(Xt, yt, Xv, yv, seed=sd, epochs=EPOCHS, sw=sw)
                         for sd in ENSEMBLE_SEEDS]
        print(f"         ✓ eff_n={eff_n:.0f}")
    return experts

def predict_soft(experts, X, probs):
    preds = {}
    for i, name in enumerate(['bull','bear','chaos']):
        if experts.get(name) is None:
            preds[name] = np.zeros(len(X)); continue
        preds[name] = np.mean([m.predict(X, verbose=0).flatten()
                               for m in experts[name]], axis=0)
    return (probs[:,0]*preds['bull']
          + probs[:,1]*preds['bear']
          + probs[:,2]*preds['chaos'])

# ============================================================
# BACKTEST
# ============================================================

def backtest(pred, actual):
    pos   = np.sign(pred)
    raw   = pos * actual
    chg   = np.abs(np.diff(np.concatenate([[0], pos])))
    cost  = chg * (2 * TRANSACTION_COST_BPS / 10000.0)
    daily = raw - cost
    idx   = np.arange(0, len(pred), PREDICTION_HORIZON)
    pos_n = pos[idx]; raw_n = pos_n * actual[idx]
    chg_n = np.abs(np.diff(np.concatenate([[0], pos_n])))
    no    = raw_n - chg_n * (2 * TRANSACTION_COST_BPS / 10000.0)
    return daily, no

# ============================================================
# ANA ANALİZ
# ============================================================

def analyze(ticker, config, macro):
    set_seeds(42)
    name, bench = config['name'], config['benchmark']
    print(f"\n{'='*60}")
    print(f"📊 {ticker} — {name}")
    print(f"{'='*60}")

    ds, db = download_pair(ticker, bench)
    if ds is None: return None
    print(f"   {len(ds)} gün")

    df_regime = detect_regime(db['Bench_Close'])
    cur_reg   = df_regime['regime'].iloc[-1]
    cur_probs = df_regime[['p_bull','p_bear','p_chaos']].iloc[-1].values
    print(f"   Güncel rejim: {cur_reg.upper()} "
          f"(bull={cur_probs[0]:.2f} bear={cur_probs[1]:.2f} chaos={cur_probs[2]:.2f})")

    df = build_features(ds, db, macro, df_regime)
    if df is None or len(df) < 500: return None
    print(f"   {len(df)} kullanılabilir gün")

    # EWT diagnostik (son değerler)
    h_val  = df['hurst_60'].iloc[-1]  if 'hurst_60' in df.columns else np.nan
    cp_val = df['chan_pos'].iloc[-1]   if 'chan_pos' in df.columns else np.nan
    pc_val = df['pivot_count_60'].iloc[-1] if 'pivot_count_60' in df.columns else np.nan
    if np.isfinite(h_val):
        hurst_interp = "trend" if h_val>0.55 else "mean-rev" if h_val<0.45 else "random"
        print(f"   EWT diagnostik: Hurst={h_val:.3f}({hurst_interp}), "
              f"Chan_pos={cp_val:.2f}, Pivots(60g)={pc_val:.0f}")

    excl  = ['Open','High','Low','Volume','Close','Bench_Close','Adj Close',
             'Fut_Stock','Fut_Bench','Fut_Resid']
    fcols = [c for c in df.columns if c not in excl]
    X_arr = df[fcols].values
    y_arr = df['Fut_Resid'].values
    rp    = df[['p_bull','p_bear','p_chaos']].values

    # EWT features sayısı
    ewt_fcols = [c for c in fcols if any(k in c for k in
                 ['fib_','hurst','pivot','swing','wave_mom','chan_','rel_hurst','rel_chan','rel_piv'])]
    print(f"   Toplam feature: {len(fcols)}  (EWT: {len(ewt_fcols)})")

    tr = int(len(df) * TRAIN_FRAC)
    vl = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    print(f"   Split: train={tr} val={vl-tr} test={len(df)-vl}")

    xs = StandardScaler(); xs.fit(X_arr[:tr])
    ys = StandardScaler(); ys.fit(y_arr[:tr].reshape(-1,1))

    Xtr_s = xs.transform(X_arr[:tr])
    ytr_s = ys.transform(y_arr[:tr].reshape(-1,1)).flatten()
    Xv_s  = xs.transform(X_arr[tr-LOOKBACK:vl])
    yv_s  = ys.transform(y_arr[tr-LOOKBACK:vl].reshape(-1,1)).flatten()
    Xte_s = xs.transform(X_arr[vl-LOOKBACK:])
    yte_s = ys.transform(y_arr[vl-LOOKBACK:].reshape(-1,1)).flatten()

    Xt,yt   = make_dataset(Xtr_s, ytr_s, LOOKBACK)
    Xv,yv   = make_dataset(Xv_s,  yv_s,  LOOKBACK)
    Xte,yte = make_dataset(Xte_s, yte_s, LOOKBACK)

    tr_probs = rp[LOOKBACK:tr]      # ✓ len=len(Xt)
    xv_probs = rp[tr:vl]            # ✓ len=len(Xv)
    te_probs = rp[vl:vl+len(yte)]   # ✓ len=len(Xte)

    print(f"\n   Uzman eğitimi ({len(ENSEMBLE_SEEDS)*3} LSTM, {len(fcols)} feature):")
    experts = train_experts(Xt, yt, Xv, yv, tr_probs)

    pred_s = predict_soft(experts, Xte, te_probs)
    pred_r = ys.inverse_transform(pred_s.reshape(-1,1)).flatten()
    act_r  = ys.inverse_transform(yte.reshape(-1,1)).flatten()

    r2      = float(np.corrcoef(act_r, pred_r)[0,1]**2) if np.std(pred_r)>0 else 0.0
    correct = int((np.sign(pred_r)==np.sign(act_r)).sum())
    total   = len(pred_r)
    dir_acc = correct / total * 100
    p_val   = binom_p(correct, total)
    act_pos = (act_r  > 0).mean() * 100
    prd_pos = (pred_r > 0).mean() * 100
    naive   = max(act_pos, 100-act_pos)
    edge    = dir_acc - naive

    print(f"   R²={r2:+.4f}  Yön=%{dir_acc:.1f} ({correct}/{total} p={p_val:.4f})")
    print(f"   Pos oran: gerçek=%{act_pos:.1f} tahmin=%{prd_pos:.1f}  edge=%{edge:+.2f}")

    reg_lab = np.array(['bull','bear','chaos'])[np.argmax(te_probs, axis=1)]
    reg_res = {}
    for reg in ['bull','bear','chaos']:
        mask = reg_lab==reg; n=int(mask.sum())
        if n < 10: reg_res[reg]=None; continue
        rc = int((np.sign(pred_r[mask])==np.sign(act_r[mask])).sum())
        reg_res[reg] = {'n':n,'dir':rc/n*100,'p':binom_p(rc,n)}
    for reg, v in reg_res.items():
        if v: print(f"      {reg.upper():>6}: n={v['n']} yön=%{v['dir']:.1f} p={v['p']:.3f}")

    daily, no = backtest(pred_r, act_r)
    ir_no  = info_ratio(no, 252/PREDICTION_HORIZON)
    bh_alp = (df['Fut_Stock'].iloc[vl:vl+total].values
              - df['Beta_60'].iloc[vl:vl+total].values
              * df['Fut_Bench'].iloc[vl:vl+total].values)
    bh_ir  = info_ratio(bh_alp, 252)
    cum_no = float((np.exp(no.sum())-1)*100)
    print(f"   IR(NoOver)={ir_no:+.2f}  B&H IR={bh_ir:+.2f}  Cum α(NoOver)=%{cum_no:+.1f}")

    last_X    = xs.transform(X_arr)[-LOOKBACK:].reshape(1,LOOKBACK,len(fcols))
    last_prob = rp[-1:].reshape(1,3)
    fut_s     = predict_soft(experts, last_X, last_prob)
    fut_ret   = float(ys.inverse_transform(fut_s.reshape(-1,1))[0,0])
    resid_std = float(np.std(act_r - pred_r))
    cur_price = float(df['Close'].iloc[-1])
    cur_bprice= float(db['Bench_Close'].iloc[-1])
    last_beta = float(df['Beta_60'].iloc[-1])

    if   abs(fut_ret)<0.5*resid_std: sig,sc = "NÖTR","gray"
    elif fut_ret>0: sig,sc = ("GÜÇLÜ AL","green") if fut_ret>1.5*resid_std else ("AL","blue")
    else:           sig,sc = ("SAT","red")         if fut_ret<-1.5*resid_std else ("ZAYIF SAT","orange")
    print(f"   Tahmin: {fut_ret*100:+.2f}%  Sinyal: {sig}  Rejim: {cur_reg.upper()}")

    return dict(
        ticker=ticker, name=name, bench=bench,
        df=df, df_regime=df_regime,
        pred_r=pred_r, act_r=act_r,
        r2=r2, dir_acc=dir_acc, p_val=p_val,
        act_pos=act_pos, prd_pos=prd_pos, edge=edge,
        reg_res=reg_res,
        ir_no=ir_no, bh_ir=bh_ir, cum_no=cum_no,
        daily=daily, no=no,
        cur_price=cur_price, cur_bench=cur_bprice, last_beta=last_beta,
        fut_ret=fut_ret, resid_std=resid_std,
        signal=sig, sig_color=sc,
        cur_reg=cur_reg, cur_probs=cur_probs.tolist(),
        val_split=vl, total=total, n_feat=len(fcols), n_ewt=len(ewt_fcols),
        hurst=float(h_val) if np.isfinite(h_val) else None,
        chan_pos=float(cp_val) if np.isfinite(cp_val) else None,
        pivot_count=float(pc_val) if np.isfinite(pc_val) else None,
    )

# ============================================================
# ANA GRAFİK — EWT Fibonacci seviyeleri eklendi
# ============================================================

def make_chart(res):
    ticker = res['ticker']
    df     = res['df']
    vl     = res['val_split']
    pred_r = res['pred_r']

    fig, ax = plt.subplots(figsize=(14, 7))
    show_n  = min(600, len(df))
    df_show = df.iloc[-show_n:]

    # Rejim renk bantları
    dom  = np.argmax(df_show[['p_bull','p_bear','p_chaos']].values, axis=1)
    rcol = {0:'#86efac', 1:'#fca5a5', 2:'#fde047'}
    s, lr = 0, dom[0]
    for i in range(1, len(dom)):
        if dom[i] != lr:
            ax.axvspan(df_show.index[s], df_show.index[i],
                       alpha=0.12, color=rcol[lr], zorder=0)
            s, lr = i, dom[i]
    ax.axvspan(df_show.index[s], df_show.index[-1],
               alpha=0.12, color=rcol[lr], zorder=0)

    # Gerçek fiyat
    ax.plot(df_show.index, df_show['Close'],
            color='#1f2937', linewidth=1.8, label=f'{ticker} Gerçek', zorder=3)

    # EWT Fibonacci seviyeleri (son EWT_WINDOW günün range'i)
    if 'range_pos' in df.columns:
        last_n  = df_show.iloc[-EWT_WINDOW:]
        h_max   = last_n['High'].max() if 'High' in last_n.columns else last_n['Close'].max()
        l_min   = last_n['Low'].min()  if 'Low'  in last_n.columns else last_n['Close'].min()
        rng     = h_max - l_min
        if rng > 0:
            fib_levels = {
                '0.786': l_min + 0.786*rng,
                '0.618': l_min + 0.618*rng,
                '0.500': l_min + 0.500*rng,
                '0.382': l_min + 0.382*rng,
            }
            fib_colors = {'0.786':'#7c3aed','0.618':'#2563eb',
                          '0.500':'#059669','0.382':'#d97706'}
            x_start = df_show.index[-EWT_WINDOW]
            x_end   = df.index[-1] + timedelta(days=PREDICTION_HORIZON+2)
            for level, price in fib_levels.items():
                ax.hlines(price, x_start, x_end,
                          colors=fib_colors[level], linewidths=0.9,
                          linestyles='-.', alpha=0.7,
                          label=f'Fib {level}: ${price:.2f}', zorder=2)

    # OOS test tahmin çizgisi
    n_te   = len(pred_r)
    tc     = df['Close'].iloc[vl:vl+n_te].values
    tb     = df['Beta_60'].iloc[vl:vl+n_te].values
    tbr    = df['Fut_Bench'].iloc[vl:vl+n_te].values
    pp     = tc * np.exp(pred_r + tb * tbr)
    pd_idx = df.index[vl:vl+n_te] + pd.Timedelta(days=PREDICTION_HORIZON)
    ax.plot(pd_idx, pp, linestyle='--', color='#f59e0b',
            linewidth=1.5, alpha=0.85, label=f'Model OOS ({PREDICTION_HORIZON}g)', zorder=4)

    # Test başlangıç çizgisi
    ax.axvline(df.index[vl], color='red', linestyle=':', alpha=0.5,
               linewidth=1.5, label='Test başlangıcı', zorder=5)

    # Gelecek tahmin noktası
    fut_date  = df.index[-1] + timedelta(days=PREDICTION_HORIZON)
    fut_price = res['cur_price'] * np.exp(res['fut_ret'])
    ax.scatter([fut_date], [fut_price], color=res['sig_color'],
               s=180, zorder=10, edgecolors='black', linewidth=1.5,
               label=f'{res["signal"]} (${fut_price:.2f})')

    # EWT Hurst bilgisi başlıkta
    h_str = f"Hurst={res['hurst']:.3f}" if res.get('hurst') else ""
    cp_str= f"Chan={res['chan_pos']:.2f}" if res.get('chan_pos') else ""

    ax.set_title(
        f"{res['name']} ({ticker})  |  Rejim: {res['cur_reg'].upper()}  |  "
        f"Sinyal: {res['signal']}  |  IR(NoOver): {res['ir_no']:+.2f}  |  "
        f"Yön: %{res['dir_acc']:.1f}  |  Edge: %{res['edge']:+.2f}\n"
        f"EWT → {h_str}  {cp_str}  |  "
        f"Noktalı yatay çizgiler = son {EWT_WINDOW}g Fibonacci seviyeleri  |  "
        f"🟢 Bull  🔴 Bear  🟡 Chaos",
        fontsize=10, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.2); ax.set_ylabel(f'{ticker} ($)')
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================
# TEK HTML
# ============================================================

CSS = """
<style>
*{box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#f1f5f9;padding:24px;margin:0}
.page{max-width:1280px;margin:0 auto}
h1{text-align:center;color:#0f172a;font-size:1.6em;margin-bottom:6px}
.subtitle{text-align:center;color:#64748b;font-size:.88em;margin-bottom:28px}
.card{background:#fff;border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);
      margin-bottom:36px;overflow:hidden}
.card-header{padding:13px 22px;display:flex;align-items:center;
             justify-content:space-between;background:#0f172a;color:#fff}
.card-header .ticker{font-size:1.35em;font-weight:700}
.rbadge{font-size:.78em;padding:3px 11px;border-radius:18px;font-weight:600}
.bull-b{background:#16a34a}.bear-b{background:#dc2626}.chaos-b{background:#ca8a04}
.signal-bar{padding:13px 22px;text-align:center;font-weight:700;font-size:1.1em}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#e2e8f0;
         border-top:1px solid #e2e8f0}
.m{background:#fff;padding:13px;text-align:center}
.m-l{font-size:.71em;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.m-v{font-size:1.22em;font-weight:800;margin:4px 0;color:#0f172a}
.m-s{font-size:.67em;color:#94a3b8}
.ewt-bar{padding:10px 22px;background:#f0f9ff;border-top:1px solid #bae6fd;
         font-size:.83em;color:#0369a1;display:flex;gap:24px;flex-wrap:wrap}
.ewt-item{display:flex;flex-direction:column;align-items:center}
.ewt-label{font-size:.75em;color:#0891b2;font-weight:600;text-transform:uppercase}
.ewt-val{font-size:1.05em;font-weight:700;color:#0f172a}
.chart-wrap{padding:20px 22px;background:#f8fafc;border-top:1px solid #e2e8f0;text-align:center}
.chart-wrap img{max-width:100%;border-radius:10px;border:1px solid #e2e8f0}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
.reg-table{padding:14px 22px}
.reg-table b{font-size:.83em;color:#475569}
.reg-table table{width:100%;border-collapse:collapse;font-size:.86em;margin-top:7px}
.reg-table th{background:#f1f5f9;padding:6px 9px;border:1px solid #e2e8f0;font-weight:700;color:#374151}
.reg-table td{padding:6px 9px;border:1px solid #e2e8f0;text-align:center}
.bull-row{background:#f0fdf4}.bear-row{background:#fef2f2}.chaos-row{background:#fefce8}
.scen{padding:14px 22px;border-left:1px solid #e2e8f0}
.scen h4{margin:0 0 8px;font-size:.85em;color:#475569}
.scen table{width:100%;border-collapse:collapse;font-size:.86em}
.scen th{background:#f1f5f9;padding:5px 9px;border:1px solid #e2e8f0}
.scen td{padding:5px 9px;border:1px solid #e2e8f0;text-align:center}
.pos{background:#f0fdf4;color:#15803d;font-weight:600}
.neg{background:#fef2f2;color:#b91c1c;font-weight:600}
.sum-wrap{background:#fff;border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);
          margin-bottom:36px;overflow:hidden}
.sum-h{background:#0f172a;color:#fff;padding:13px 22px;font-size:1.05em;font-weight:700}
.sum-wrap table{width:100%;border-collapse:collapse;font-size:.87em}
.sum-wrap th{background:#f1f5f9;padding:8px 11px;border:1px solid #e2e8f0;
             font-weight:700;color:#374151}
.sum-wrap td{padding:8px 11px;border:1px solid #e2e8f0;text-align:center}
.green{color:#15803d;font-weight:700}.orange{color:#ca8a04;font-weight:700}
.red{color:#b91c1c;font-weight:700}
.note{margin:0 0 24px;padding:10px 14px;border-radius:8px;font-size:.81em;
      background:#fefce8;border-left:4px solid #ca8a04;color:#713f12}
</style>
"""

def sbg(sc):
    return {'green':'#dcfce7','blue':'#dbeafe','gray':'#f1f5f9',
            'orange':'#ffedd5','red':'#fee2e2'}.get(sc,'#f1f5f9')

def ecls(e):
    return 'green' if e>3 else 'orange' if e>0 else 'red'

def dcls(d):
    return 'green' if d>55 else 'orange' if d>50 else 'red'

def hurst_interp(h):
    if h is None or not np.isfinite(h): return "—"
    return "Trend↗" if h>0.55 else "MeanRev↩" if h<0.45 else "Random~"

def build_html(results):
    valid = [r for r in results if r is not None]

    # ── Özet tablosu ──
    sum_rows = ""
    for r in valid:
        ec = ecls(r['edge']); dc = dcls(r['dir_acc'])
        rb = f"{r['cur_reg']}-b"
        h_raw   = r.get('hurst')
        h_str   = f"{h_raw:.3f}" if (h_raw is not None and np.isfinite(h_raw)) else ""
        h_interp = hurst_interp(h_raw)
        sum_rows += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b><br>"
            f"<span style='font-size:.78em;color:#64748b'>{r['name']}</span></td>"
            f"<td><span class='rbadge {rb}' style='padding:2px 7px;border-radius:8px;"
            f"font-size:.76em'>{r['cur_reg'].upper()}</span></td>"
            f"<td class='{dc}'>%{r['dir_acc']:.1f}<br>"
            f"<span style='font-size:.73em;font-weight:400'>p={r['p_val']:.3f}</span></td>"
            f"<td>{r['r2']:+.4f}</td>"
            f"<td class='{ec}'>%{r['edge']:+.2f}</td>"
            f"<td>{r['ir_no']:+.2f}</td>"
            f"<td>{r['bh_ir']:+.2f}</td>"
            f"<td>%{r['cum_no']:+.1f}</td>"
            f"<td style='font-size:.8em'>{h_interp}<br>"
            f"<span style='color:#94a3b8'>{h_str}</span></td>"
            f"<td style='color:{r['sig_color']};font-weight:700'>{r['signal']}</td>"
            f"</tr>"
        )

    summary = f"""
<div class="sum-wrap">
  <div class="sum-h">📊 Özet Karşılaştırma — v15.3 (EWT Dahil)</div>
  <div style="padding:14px 22px;overflow-x:auto">
  <table>
  <tr><th>Ticker</th><th>Rejim</th><th>Yön</th><th>R²</th>
      <th>Alfa Edge</th><th>IR(NoOver)</th><th>Passive IR</th>
      <th>Cum α</th><th>Hurst (EWT)</th><th>Sinyal</th></tr>
  {sum_rows}
  </table>
  </div>
</div>"""

    # ── Hisse kartları ──
    cards = ""
    for r in valid:
        chart_b64 = make_chart(r)

        # Rejim tablosu
        reg_rows = ""
        for reg in ['bull','bear','chaos']:
            rv  = r['reg_res'].get(reg); cls = f"{reg}-row"
            if rv is None:
                reg_rows += f"<tr class='{cls}'><td><b>{reg.upper()}</b></td><td colspan=3 style='color:#94a3b8'>Yetersiz veri</td></tr>"
            else:
                dc = dcls(rv['dir'])
                reg_rows += (f"<tr class='{cls}'>"
                             f"<td><b>{reg.upper()}</b></td><td>{rv['n']}</td>"
                             f"<td class='{dc}'>%{rv['dir']:.1f}</td>"
                             f"<td>{rv['p']:.3f}</td></tr>")

        # Senaryo tablosu
        scen_rows = ""
        for sm in [-0.04,-0.02,0.0,0.02,0.04]:
            im = r['fut_ret'] + r['last_beta'] * sm
            ip = r['cur_price'] * np.exp(im)
            cls = 'pos' if im>0 else 'neg'
            scen_rows += (f"<tr><td>{sm*100:+.1f}%</td>"
                          f"<td class='{cls}'>{im*100:+.2f}%</td>"
                          f"<td class='{cls}'>${ip:.2f}</td></tr>")

        # EWT bar
        h    = r.get('hurst')
        cp   = r.get('chan_pos')
        pc   = r.get('pivot_count')
        hi   = hurst_interp(h)
        h_s  = f"{h:.3f}" if h is not None else "—"
        cp_s = f"{cp:.2f}" if cp is not None else "—"
        pc_s = f"{pc:.0f}" if pc is not None else "—"

        # Renk kodu: Hurst
        hc = '#15803d' if (h and h>0.55) else '#b91c1c' if (h and h<0.45) else '#ca8a04'
        # Renk kodu: Chan pos (0=taban, 1=tavan)
        cpc= '#b91c1c' if (cp and cp>0.7) else '#15803d' if (cp and cp<0.3) else '#ca8a04'

        avc = ecls(r['edge'])
        avt = 'GERÇEK ALFA' if r['edge']>3 else 'ZAYIF' if r['edge']>0 else 'ALFA YOK'
        sig_bar_bg = sbg(r['sig_color'])
        rb  = f"{r['cur_reg']}-b"

        cards += f"""
<div class="card">
  <div class="card-header">
    <span class="ticker">{r['ticker']} — {r['name']}</span>
    <span class="rbadge {rb}">{r['cur_reg'].upper()}
      &nbsp;|&nbsp; bull={r['cur_probs'][0]:.2f}
      bear={r['cur_probs'][1]:.2f}
      chaos={r['cur_probs'][2]:.2f}</span>
  </div>

  <div class="signal-bar" style="background:{sig_bar_bg};color:{r['sig_color']}">
    {r['signal']}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Residual tahmin: {r['fut_ret']*100:+.2f}%
    &nbsp;|&nbsp; β={r['last_beta']:.2f} ({r['bench']})
  </div>

  <!-- EWT diagnostik bar -->
  <div class="ewt-bar">
    <div class="ewt-item">
      <span class="ewt-label">📐 Hurst (60g)</span>
      <span class="ewt-val" style="color:{hc}">{h_s} — {hi}</span>
    </div>
    <div class="ewt-item">
      <span class="ewt-label">📏 Elliott Channel</span>
      <span class="ewt-val" style="color:{cpc}">{cp_s} {"(tavana yakın)" if cp and cp>0.7 else "(tabana yakın)" if cp and cp<0.3 else "(orta)"}</span>
    </div>
    <div class="ewt-item">
      <span class="ewt-label">🔢 Pivot Sayısı (60g)</span>
      <span class="ewt-val">{pc_s}</span>
    </div>
    <div class="ewt-item">
      <span class="ewt-label">🔢 EWT Feature Sayısı</span>
      <span class="ewt-val">{r.get('n_ewt',0)} / {r.get('n_feat',0)}</span>
    </div>
    <div class="ewt-item" style="flex:1;text-align:left;font-size:.79em;color:#0369a1;padding-left:10px">
      Grafik üzerinde <b>noktalı yatay çizgiler</b> = son {EWT_WINDOW}g Fibonacci seviyeleri (0.382/0.500/0.618/0.786).
      eSignal istatistikleri: Wave2'nin %73'ü 0.50–0.62 bandında bitirir.
    </div>
  </div>

  <!-- Metrik grid -->
  <div class="metrics">
    <div class="m"><div class="m-l">Mevcut</div>
      <div class="m-v">${r['cur_price']:.2f}</div>
      <div class="m-s">{r['bench']}: ${r['cur_bench']:.2f}</div></div>
    <div class="m"><div class="m-l">Yön Doğruluğu</div>
      <div class="m-v" style="color:{'#15803d' if r['dir_acc']>55 else '#ca8a04' if r['dir_acc']>50 else '#b91c1c'}">
        %{r['dir_acc']:.1f}</div>
      <div class="m-s">{r['total']} OOS · p={r['p_val']:.3f}</div></div>
    <div class="m"><div class="m-l">Alfa Edge</div>
      <div class="m-v" style="color:{'#15803d' if r['edge']>3 else '#ca8a04' if r['edge']>0 else '#b91c1c'}">
        %{r['edge']:+.2f}</div>
      <div class="m-s">{avt}</div></div>
    <div class="m"><div class="m-l">R² (Residual)</div>
      <div class="m-v">{r['r2']:+.4f}</div>
      <div class="m-s">OOS test seti</div></div>
    <div class="m"><div class="m-l">IR (NoOver) ✓</div>
      <div class="m-v" style="color:{'#15803d' if r['ir_no']>0.3 else '#ca8a04' if r['ir_no']>0 else '#b91c1c'}">
        {r['ir_no']:+.2f}</div>
      <div class="m-s">Gerçekçi bağımsız</div></div>
    <div class="m"><div class="m-l">Passive IR</div>
      <div class="m-v">{r['bh_ir']:+.2f}</div>
      <div class="m-s">B&H modelsiz</div></div>
    <div class="m"><div class="m-l">Cum α (NoOver)</div>
      <div class="m-v" style="color:{'#15803d' if r['cum_no']>0 else '#b91c1c'}">
        %{r['cum_no']:+.1f}</div>
      <div class="m-s">Bağımsız periyot</div></div>
    <div class="m"><div class="m-l">Pos Oran Δ</div>
      <div class="m-v">%{abs(r['prd_pos']-r['act_pos']):.1f}</div>
      <div class="m-s">tahmin−gerçek</div></div>
  </div>

  <!-- Grafik -->
  <div class="chart-wrap">
    <img src="data:image/png;base64,{chart_b64}" alt="{r['ticker']} grafik">
  </div>

  <!-- Rejim + Senaryo -->
  <div class="bottom-grid">
    <div class="reg-table">
      <b>🎭 Rejim-Bazlı OOS Performans</b>
      <table><tr><th>Rejim</th><th>n</th><th>Yön %</th><th>p</th></tr>
      {reg_rows}</table>
    </div>
    <div class="scen">
      <h4>📐 Senaryo ({r['bench']} hareketine göre)</h4>
      <table><tr><th>{r['bench']} 7g</th><th>{r['ticker']} Tahmin</th><th>Fiyat</th></tr>
      {scen_rows}</table>
    </div>
  </div>
</div>"""

    now    = datetime.now().strftime('%d.%m.%Y %H:%M')
    tickers= ', '.join(r['ticker'] for r in valid)

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Multi-Stock Analiz v15.3 — {tickers}</title>
  {CSS}
</head>
<body>
<div class="page">
  <h1>🔬 Multi-Stock Regime-Switching + EWT — v15.3</h1>
  <div class="subtitle">
    {tickers} &nbsp;·&nbsp; Market-neutral residual return &nbsp;·&nbsp;
    Soft ensemble 3 rejim &nbsp;·&nbsp;
    EWT: Fibonacci · Hurst · Swing · Channel &nbsp;·&nbsp; {now}
  </div>

  {summary}
  {cards}

  <div class="note">
    <b>v15.3 EWT entegrasyonu:</b>
    [EWT-1] Fibonacci seviyeleri (0.382/0.500/0.618/0.786) — eSignal istatistikleriyle kalibre edildi.
    [EWT-2] Hurst exponent (60g rolling) — H&gt;0.55: trend persistent, H&lt;0.45: mean-reverting.
    [EWT-3] Swing structure — pivot sayısı ve amplitüdü (EWT dalga sayımı proxy'si).
    [EWT-4] Wave momentum — 5g/20g momentum oranı (Wave 3 güç testi).
    [EWT-5] Elliott Channel — lineer kanal pozisyonu, eğimi, genişliği.
    [EWT-6] Benchmark rölatif EWT — SOXX'a göre rölatif Hurst, kanal, pivot farkı.
    Tüm EWT features causal (geleceğe bakmaz). v15.2 fix'leri korundu.
  </div>
</div>
</body>
</html>"""

# ============================================================
# ANA AKIŞ
# ============================================================

def main():
    set_seeds()
    init_db()

    tickers = list(TICKERS_CONFIG.keys())
    n_lstm  = len(tickers) * len(ENSEMBLE_SEEDS) * 3

    print("=" * 60)
    print(f"MULTI-STOCK v15.3 — EWT ENTEGRE  |  {', '.join(tickers)}")
    print("=" * 60)
    print(f"Hisseler : {tickers}")
    print(f"Ufuk     : {PREDICTION_HORIZON} gün")
    print(f"Toplam LSTM: {n_lstm}")
    print(f"EWT features: Fibonacci, Hurst, Swing, WaveMom, Channel, BenchRel")

    macro = get_macro()

    results = []
    for ticker, cfg in TICKERS_CONFIG.items():
        res = analyze(ticker, cfg, macro)
        results.append(res)

    print(f"\n📄 HTML oluşturuluyor → {OUTPUT_HTML}")
    html = build_html(results)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✓ {OUTPUT_HTML}")

    print("\n" + "=" * 60)
    print(f"{'TICKER':<6} {'YÖN%':<7} {'EDGE%':<8} {'IR':<7} {'CUM%':<9} {'HURST':<8} {'SİNYAL'}")
    print("-" * 60)
    for r in results:
        if r is None: continue
        h_s = f"{r['hurst']:.3f}" if r.get('hurst') else "—"
        print(f"{r['ticker']:<6} %{r['dir_acc']:<5.1f} %{r['edge']:<+6.2f} "
              f"{r['ir_no']:<+5.2f}  %{r['cum_no']:<+7.1f} {h_s:<8} {r['signal']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
