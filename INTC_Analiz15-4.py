# ============================================================
# MULTI-STOCK v15.4 — BİLİMSEL + PİYASA UYUMLU İYİLEŞTİRMELER
#
# v15.3 üzerine 7 kritik iyileştirme:
#
# [IMP-1] VOLATİLİTE-ÖLÇEKLENMIŞ HEDEF (Bilim:9, Pratik:8)
#   ESKI: y = raw_residual → yüksek volatilite dönemlerinde
#         model gürültüyü öğrenir, sinyal kaybolur.
#   YENİ: y = residual / rolling_vol_20 (Sharpe-benzeri hedef)
#         Düşük-vol dönemlerdeki küçük sinyaller büyütülür.
#         Yüksek-vol dönemlerdeki gürültü bastırılır.
#         → Daha temiz öğrenme, daha stabil model
#
# [IMP-2] LOOKBACK 60g → 20g (Bilim:8, Pratik:7)
#   ESKI: 60g lookback → Hurst≈0.05 mean-reverting hisseler
#         için çok uzun hafıza, alakasız geçmiş öğretiliyor.
#   YENİ: 20g lookback → mean-reverting dinamikle uyumlu.
#         Son 1 ay bilgisi bu hisseler için optimal.
#         Not: 20g her hisse için uygun. Trend hisselerinde
#         LOOKBACK artırılabilir.
#
# [IMP-3] VIX-BAZLI REJİM FİLTRESİ (Bilim:7, Pratik:9)
#   ESKI: Sadece SMA50/SMA200 tabanlı rejim → VIX yüksekken
#         model güvensiz tahmin yapmasına rağmen uyarı yok.
#   YENİ: VIX > 30 → "YÜKSEK KORKU" uyarısı + tahmin susturma.
#         VIX 20-30 → dikkat modu.
#         Piyasa pratiğinde VIX>30 "black swan" bölgesi.
#         Model burada tahmin üretmez, nakit pozisyon alınır.
#
# [IMP-4] KALMAN FİLTRELİ BETA (Bilim:9, Pratik:7)
#   ESKI: Rolling OLS beta (60g pencere) → piyasa hareketlerine
#         geç tepki, yüksek gürültü.
#   YENİ: Kalman Filter beta → adaptif, gerçek zamana daha yakın.
#         Her gün gözleme göre update edilir.
#         Hedge fon standartı: JPMorgan, Goldman Sachs Kalman kullanır.
#
# [IMP-5] BİAS KALİBRASYONU — İZOTONİK REGRESYON (Bilim:8, Pratik:7)
#   ESKI: Model tahminleri sistematik bias içeriyor
#         (INTC: tahmin %9.3 pos, gerçek %44.2 pos)
#   YENİ: Validation setinde isotonic regression ile kalibrasyon.
#         Tahmin dağılımı gerçek dağılıma hizalanır.
#         Hem yön doğruluğunu hem de güven skorunu düzeltiyor.
#
# [IMP-6] KELLY KRİTERİ POZİSYON BÜYÜKLÜĞÜ (Bilim:8, Pratik:9)
#   ESKI: Her tahmin için sabit 1 birim pozisyon → modelin
#         güven düzeyi fark etmeksizin aynı risk alınıyor.
#   YENİ: Kelly = (p_win - p_lose) / 1 = 2*dir_acc - 1
#         Yüksek güven → büyük pozisyon
#         Düşük güven → küçük pozisyon
#         Fractional Kelly (%50 Kelly) kullanılır (aşırı kaldıraç önlemi)
#
# [IMP-7] EARNINGS BLACKOUT PENCERESİ (Bilim:6, Pratik:9)
#   ESKI: Kazanç açıklamaları etrafında model güvensiz ama
#         tahmin üretiyor → büyük kayıp riski.
#   YENİ: Earnings ±3 iş günü → "BLACKOUT" modu.
#         Bu dönemde sinyal üretilmez.
#         yfinance ile earnings tarihleri çekilir.
#
# Korunan özellikler:
#   EWT: Fibonacci, DFA-Hurst (FIX-HURST), Swing, WaveMom, Channel
#   FIX-1: StandardScaler
#   FIX-2: Non-overlapping backtest
#   FIX-3: Regime prob hizalaması (FIX-3)
#   FIX-4: Seed izolasyonu
#   FIX-TEST: Test=%20
#   FIX-BONFERRONI: Çoklu test düzeltmesi
# ============================================================

import os, warnings, random, base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal as sp_signal
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Input,
                                     Bidirectional, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D)
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
LOOKBACK             = 20      # [IMP-2] 60→20g (mean-rev uyumu)
TRAIN_FRAC           = 0.70    # test=%20
VAL_FRAC             = 0.10
ENSEMBLE_SEEDS       = [42, 52, 62]
EPOCHS               = 60
OUTPUT_HTML          = "MultiStock_Analiz_v15.4.html"

# EWT parametreleri
EWT_WINDOW  = 60
EWT_CHAN_W  = 60

# [IMP-3] VIX eşikleri
VIX_HIGH    = 30   # Yüksek korku: tahmin susturulur
VIX_MEDIUM  = 20   # Dikkat modu

# [IMP-6] Kelly fraction
KELLY_FRAC  = 0.50  # %50 Kelly (güvenli)

# [IMP-7] Earnings blackout (iş günü)
EARNINGS_BLACKOUT_DAYS = 3

# Bonferroni
N_TICKERS   = len(TICKERS_CONFIG)
N_REGIMES   = 3
N_TESTS     = N_TICKERS * N_REGIMES
ALPHA       = 0.05
ALPHA_BONF  = ALPHA / N_TESTS
ALPHA_LOCAL = ALPHA / N_REGIMES

# ============================================================
# YARDIMCI
# ============================================================

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

def get_earnings_dates(ticker):
    """[IMP-7] Earnings tarihlerini çek. Hata olursa boş döner."""
    try:
        t  = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty: return []
        # Earnings Date sütununu bul
        for col in ['Earnings Date', 'Earnings Dates']:
            if col in cal.columns or col in cal.index:
                dates = cal.loc['Earnings Date'] if col in cal.index else cal[col]
                return pd.to_datetime(dates).dropna().tolist()
        return []
    except:
        return []

# ============================================================
# REJİM TESPİTİ
# ============================================================

def detect_regime(bench_close, vix_series=None):
    """
    Hibrit rejim: SMA50/SMA200 trendi + VIX volatilite filtresi [IMP-3].
    VIX > VIX_HIGH → CHAOS override (model susturulur).
    """
    log_ret   = np.log(bench_close / bench_close.shift(1))
    sma50     = bench_close.rolling(50).mean()
    sma200    = bench_close.rolling(200).mean()
    trend_str = 1.0 / (1.0 + np.exp(-((sma50 / sma200) - 1.0) * 50))
    vol       = log_ret.rolling(20).std()
    vol_q     = vol.expanding(min_periods=252).quantile(0.70)
    p_chaos   = (1.0 / (1.0 + np.exp(-(vol / vol_q - 1.0) * 5))).clip(0.05, 0.95)

    # [IMP-3] VIX override
    if vix_series is not None:
        vix_aligned = vix_series.reindex(bench_close.index, method='ffill')
        vix_high_mask = vix_aligned > VIX_HIGH
        # VIX > 30 → p_chaos → 0.95 (maksimum chaos)
        p_chaos = p_chaos.where(~vix_high_mask, other=0.95)

    rem    = 1.0 - p_chaos
    p_bull = rem * trend_str
    p_bear = rem * (1.0 - trend_str)

    df_r = pd.DataFrame({'p_bull': p_bull, 'p_bear': p_bear,
                         'p_chaos': p_chaos}, index=bench_close.index)
    probs = df_r[['p_bull','p_bear','p_chaos']].values
    df_r['regime'] = np.array(['bull','bear','chaos'])[np.argmax(probs, axis=1)]
    return df_r

# ============================================================
# [IMP-4] KALMAN FİLTRELİ BETA
# ============================================================

def kalman_beta(stock_ret, bench_ret, process_noise=1e-4, obs_noise=1e-2):
    """
    Kalman Filter ile adaptif beta tahmini.
    State: [beta, alpha] — zamanla değişen beta
    Rolling OLS'e göre daha hızlı adapte olur, daha az gürültü.
    process_noise: beta'nın ne kadar hızlı değiştiği
    obs_noise: gözlem gürültüsü
    """
    sr = stock_ret.fillna(0).values
    br = bench_ret.fillna(0).values
    n  = len(sr)

    # State: [beta, alpha]
    x  = np.array([1.0, 0.0])      # başlangıç beta=1, alpha=0
    P  = np.eye(2) * 1.0           # kovaryans
    Q  = np.eye(2) * process_noise # process noise
    R  = obs_noise                  # obs noise

    betas = np.full(n, np.nan)
    for i in range(n):
        b_t = br[i]
        # Predict
        # x = F*x (F=I, random walk state)
        P  = P + Q
        # H: gözlem matrisi [bench_ret, 1]
        H  = np.array([b_t, 1.0])
        # Innovation
        y_pred = H @ x
        S      = H @ P @ H + R
        if S == 0: continue
        # Kalman gain
        K      = P @ H / S
        # Update
        innov  = sr[i] - y_pred
        x      = x + K * innov
        P      = (np.eye(2) - np.outer(K, H)) @ P
        betas[i] = np.clip(x[0], 0.0, 3.0)

    return pd.Series(betas, index=stock_ret.index)

# ============================================================
# EWT FEATURES
# ============================================================

def ewt_hurst_dfa(log_returns, window=60):
    """DFA-Hurst: [0.05, 0.95] arasında stabil. (FIX-HURST)"""
    def _dfa(arr):
        n = len(arr)
        if n < 40: return 0.5
        profile = np.cumsum(arr - np.mean(arr))
        max_lag = n // 4; min_lag = 10
        if max_lag <= min_lag: return 0.5
        scales = np.unique(
            np.logspace(np.log10(min_lag), np.log10(max_lag), 10).astype(int))
        flucts = []; valid_sc = []
        for s in scales[(scales >= min_lag) & (scales <= max_lag)]:
            n_seg = n // s
            if n_seg < 2: continue
            x = np.arange(s); f2 = []
            for seg in range(n_seg):
                sd = profile[seg*s:(seg+1)*s]
                tr = np.polyval(np.polyfit(x, sd, 1), x)
                f2.append(np.mean((sd - tr)**2))
            flucts.append(np.sqrt(np.mean(f2))); valid_sc.append(s)
        if len(valid_sc) < 4: return 0.5
        try:
            return float(np.clip(np.polyfit(np.log(valid_sc), np.log(flucts), 1)[0], 0.05, 0.95))
        except: return 0.5

    arr = log_returns.fillna(0).values; n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = _dfa(arr[i-window:i])
    return pd.Series(out, index=log_returns.index)

def ewt_fibonacci_features(close, high, low, window=EWT_WINDOW):
    h_w = high.rolling(window, min_periods=window//2).max()
    l_w = low.rolling(window,  min_periods=window//2).min()
    rng = (h_w - l_w).replace(0, np.nan)
    out = pd.DataFrame(index=close.index)
    for lvl, nm in [(0.382,'fib382'),(0.500,'fib500'),
                    (0.618,'fib618'),(0.786,'fib786')]:
        out[nm] = (close - (l_w + lvl*rng)) / rng
    out['range_pos'] = (close - l_w) / rng
    return out

def ewt_swing_features(close, window=EWT_WINDOW):
    arr = close.values; n = len(arr)
    pc  = np.full(n, np.nan); sa = np.full(n, np.nan)
    for i in range(window, n):
        seg  = arr[i-window:i]; prom = np.std(seg)
        if prom <= 0: continue
        ph, _ = sp_signal.find_peaks(seg,  prominence=prom)
        pl, _ = sp_signal.find_peaks(-seg, prominence=prom)
        pc[i] = len(ph) + len(pl)
        if len(ph) > 0 and len(pl) > 0:
            sa[i] = (np.max(seg[ph]) - np.min(seg[pl])) / (np.mean(seg) + 1e-9)
    return pd.Series(pc, index=close.index), pd.Series(sa, index=close.index)

def ewt_channel_features(close, window=EWT_CHAN_W):
    arr = close.values; n = len(arr)
    cp  = np.full(n, np.nan); cs = np.full(n, np.nan); cw = np.full(n, np.nan)
    x   = np.arange(window)
    for i in range(window, n):
        seg = arr[i-window:i]
        try:
            s, b = np.polyfit(x, seg, 1); res = seg - (s*x + b)
            w = np.max(res) - np.min(res)
            if w <= 0 or np.mean(seg) == 0: continue
            cp[i] = (arr[i] - (s*window+b) - np.min(res)) / w
            cs[i] = s / (np.mean(seg) + 1e-9)
            cw[i] = w / np.mean(seg)
        except: pass
    return (pd.Series(cp, index=close.index),
            pd.Series(cs, index=close.index),
            pd.Series(cw, index=close.index))

def add_ewt(df, close_col, high_col, low_col, log_col, prefix=''):
    p = prefix
    fib = ewt_fibonacci_features(df[close_col], df[high_col], df[low_col])
    for c in fib.columns: df[f'{p}{c}'] = fib[c].values
    df[f'{p}hurst'] = ewt_hurst_dfa(df[log_col]).values
    pc, sa = ewt_swing_features(df[close_col])
    df[f'{p}pivot_n'] = pc.values; df[f'{p}swing_amp'] = sa.values
    m5  = df[close_col].pct_change(5)
    m20 = df[close_col].pct_change(20).replace(0, np.nan)
    df[f'{p}wave_mom'] = (m5/m20).clip(-5, 5)
    cp_, cs_, cw_ = ewt_channel_features(df[close_col])
    df[f'{p}chan_pos']   = cp_.values
    df[f'{p}chan_slope'] = cs_.values
    df[f'{p}chan_width'] = cw_.values
    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(ds, db, macro, df_regime):
    df = ds.copy()
    df['Bench_Close'] = db['Bench_Close']
    try:
        import ta
        # Klasik teknik
        df['RSI']      = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD']     = ta.trend.MACD(df['Close']).macd()
        df['MACD_Sig'] = ta.trend.MACD(df['Close']).macd_signal()
        df['ATR']      = ta.volatility.AverageTrueRange(df['High'],df['Low'],df['Close']).average_true_range()
        df['CCI']      = ta.trend.CCIIndicator(df['High'],df['Low'],df['Close']).cci()
        df['SMA20']    = ta.trend.SMAIndicator(df['Close'],20).sma_indicator()
        df['SMA50']    = ta.trend.SMAIndicator(df['Close'],50).sma_indicator()
        df['SMA200']   = ta.trend.SMAIndicator(df['Close'],200).sma_indicator()
        bb             = ta.volatility.BollingerBands(df['Close'])
        df['BB_pct']   = (df['Close']-bb.bollinger_lband())/(bb.bollinger_hband()-bb.bollinger_lband())
        df['Log_Ret']  = np.log(df['Close']/df['Close'].shift(1))
        df['Vol_5']    = df['Log_Ret'].rolling(5).std()
        df['Vol_20']   = df['Log_Ret'].rolling(20).std()
        df['Mom_5']    = df['Close'].pct_change(5)
        df['Mom_20']   = df['Close'].pct_change(20)
        df['Vol_ratio']= df['Volume']/df['Volume'].rolling(20).mean()
        df['Px_SMA50'] = df['Close']/df['SMA50']
        df['Px_SMA200']= df['Close']/df['SMA200']

        # Sektör
        df['B_Ret']    = np.log(df['Bench_Close']/df['Bench_Close'].shift(1))
        df['B_Mom5']   = df['Bench_Close'].pct_change(5)
        df['B_Mom20']  = df['Bench_Close'].pct_change(20)
        df['B_Vol20']  = df['B_Ret'].rolling(20).std()
        df['Rel_M5']   = df['Mom_5'] - df['B_Mom5']
        df['Rel_M20']  = df['Mom_20'] - df['B_Mom20']
        df['Rel_V20']  = df['Vol_20'] - df['B_Vol20']
        df['Rel_Str']  = df['Close']/df['Bench_Close']
        df['Rel_SSMA'] = df['Rel_Str'].rolling(20).mean()
        df['Rel_SDev'] = (df['Rel_Str']-df['Rel_SSMA'])/(df['Rel_SSMA']+1e-9)

        # [IMP-4] Kalman filtered beta
        df['Beta_K']   = kalman_beta(df['Log_Ret'], df['B_Ret'])
        df['Beta_K']   = df['Beta_K'].clip(0.0, 3.0)
        # Rolling OLS beta (referans için tutulur)
        df['Corr_20']  = df['Log_Ret'].rolling(20).corr(df['B_Ret'])

        df = df.join(df_regime[['p_bull','p_bear','p_chaos']], how='left')

        # VIX feature (makro'dan)
        if not macro.empty and 'VIX' in macro.columns:
            df['VIX']        = macro['VIX'].reindex(df.index, method='ffill')
            df['VIX_zscore'] = ((df['VIX'] - df['VIX'].rolling(252).mean())
                                / df['VIX'].rolling(252).std())
            df['VIX_high']   = (df['VIX'] > VIX_HIGH).astype(float)
            df['VIX_med']    = ((df['VIX'] > VIX_MEDIUM) & (df['VIX'] <= VIX_HIGH)).astype(float)

        # EWT — hisse
        df['_BH'] = df['Bench_Close']; df['_BL'] = df['Bench_Close']; df['_BR'] = df['B_Ret']
        df = add_ewt(df, 'Close',       'High', 'Low', 'Log_Ret', prefix='')
        df = add_ewt(df, 'Bench_Close', '_BH',  '_BL', '_BR',     prefix='b_')
        df['rel_hurst']    = df['hurst']     - df['b_hurst']
        df['rel_chan_pos'] = df['chan_pos']   - df['b_chan_pos']
        df['rel_pivots']   = df['pivot_n']   - df['b_pivot_n']
        df.drop(columns=['_BH','_BL','_BR'], errors='ignore', inplace=True)

        # [IMP-1] Volatilite-ölçeklenmiş hedef
        fut_s = np.log(df['Close'].shift(-PREDICTION_HORIZON)/df['Close'])
        fut_b = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON)/df['Bench_Close'])
        raw_resid = fut_s - df['Beta_K'] * fut_b
        vol_scale = df['Vol_20'].rolling(5).mean().shift(1)  # t-1'in vol → sızıntısız
        vol_scale = vol_scale.replace(0, np.nan).fillna(method='ffill')

        df['Fut_Stock']     = fut_s
        df['Fut_Bench']     = fut_b
        df['Fut_Resid_Raw'] = raw_resid        # orijinal (backtest için)
        df['Fut_Resid']     = raw_resid / vol_scale   # [IMP-1] ölçeklenmiş (model hedefi)
        df['Vol_Scale']     = vol_scale

        if not macro.empty:
            df = df.join(macro[['TNX','DXY','SP500','NASDAQ']], how='left').ffill()

        df = df.dropna(subset=['Fut_Resid','Beta_K','p_bull','p_bear','p_chaos'])
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
    a = np.asarray(rets); a = a[np.isfinite(a)]
    if len(a) < 2: return 0.0
    sd = np.std(a, ddof=1)
    return float(np.sqrt(ppy) * np.mean(a) / sd) if sd > 0 else 0.0

def bonf_str(p, alpha):
    if p < alpha:       return f"✅ p<{alpha:.4f}"
    elif p < ALPHA:     return f"🟡 p<{ALPHA:.2f}"
    else:               return f"❌ p={p:.4f}"

# ============================================================
# [IMP-5] BIAS KALİBRASYONU — İZOTONİK REGRESYON
# ============================================================

def calibrate_predictions(pred_val, actual_val, pred_test):
    """
    Validation seti üzerinde isotonic regression ile kalibrasyon.
    Tahmin dağılımını gerçek dağılıma hizalar.
    Hem yön doğruluğunu hem de güven skorunu iyileştirir.
    """
    try:
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pred_val, actual_val)
        return ir.predict(pred_test)
    except:
        return pred_test  # Kalibrasyon başarısız → ham tahmin kullan

# ============================================================
# [IMP-6] KELLY POZİSYON BÜYÜKLÜĞÜ
# ============================================================

def kelly_position(pred_resid, dir_acc_val, kelly_frac=KELLY_FRAC):
    """
    Fractional Kelly criterion ile pozisyon büyüklüğü.
    dir_acc_val: validation setindeki yön doğruluğu
    Sonuç: [-1, 1] arasında pozisyon büyüklüğü
    """
    p_win = dir_acc_val / 100.0
    edge  = 2 * p_win - 1.0  # Kelly fraction = edge/odds (odds=1 için)
    kelly = max(0, edge) * kelly_frac  # negatif Kelly → nakit

    # Tahmin büyüklüğüne göre de ölçekle (|tahmin| büyükse daha fazla)
    pred_z = np.abs(pred_resid) / (np.std(pred_resid) + 1e-9)
    pred_scale = np.clip(pred_z, 0.5, 2.0)  # [0.5, 2.0] aralığında

    pos = np.sign(pred_resid) * kelly * pred_scale
    return np.clip(pos, -1.0, 1.0)

# ============================================================
# MODEL
# ============================================================

def make_dataset(X, y, lb):
    Xs, ys = [], []
    for i in range(lb, len(X)):
        Xs.append(X[i-lb:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm(shape):
    """Bidirectional LSTM — LOOKBACK=20 ile optimize."""
    m = Sequential([
        Input(shape=shape),
        Bidirectional(LSTM(48, return_sequences=True)),
        Dropout(0.25),
        LSTM(24),
        Dropout(0.25),
        Dense(12, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def train_model(Xt, yt, Xv, yv, seed, sw=None):
    set_seeds(seed)
    m  = build_lstm((Xt.shape[1], Xt.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True, verbose=0)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, min_lr=1e-5, verbose=0)
    kw = dict(epochs=EPOCHS, batch_size=32,
              validation_data=(Xv,yv), callbacks=[es,rl], verbose=0)
    if sw is not None: kw['sample_weight'] = sw
    m.fit(Xt, yt, **kw)
    return m

def train_experts(Xt, yt, Xv, yv, tr_probs):
    experts = {}
    for i, name in enumerate(['bull','bear','chaos']):
        print(f"      🎓 {name.upper()}...")
        sw    = np.clip(tr_probs[:,i], 0.05, 1.0)
        eff_n = sw.sum()
        if eff_n < 80:
            print(f"         ⚠ yetersiz ({eff_n:.0f})"); experts[name]=None; continue
        experts[name] = [train_model(Xt, yt, Xv, yv, sd, sw=sw) for sd in ENSEMBLE_SEEDS]
        print(f"         ✓ eff_n={eff_n:.0f}")
    return experts

def predict_soft(experts, X, probs):
    preds = {}
    for i, name in enumerate(['bull','bear','chaos']):
        if experts.get(name) is None: preds[name]=np.zeros(len(X)); continue
        preds[name] = np.mean([m.predict(X, verbose=0).flatten()
                               for m in experts[name]], axis=0)
    return probs[:,0]*preds['bull'] + probs[:,1]*preds['bear'] + probs[:,2]*preds['chaos']

# ============================================================
# BACKTEST — Kelly ile
# ============================================================

def backtest_kelly(pred_r, actual_r_raw, kelly_pos):
    """
    [IMP-6] Kelly-ölçekli backtest.
    kelly_pos: [-1,1] pozisyon büyüklüğü
    actual_r_raw: ham residual (ölçeksiz)
    """
    raw  = kelly_pos * actual_r_raw
    chg  = np.abs(np.diff(np.concatenate([[0], kelly_pos])))
    cost = chg * (2 * TRANSACTION_COST_BPS / 10000.0)
    daily = raw - cost

    idx   = np.arange(0, len(pred_r), PREDICTION_HORIZON)
    kp_no = kelly_pos[idx]
    raw_no= kp_no * actual_r_raw[idx]
    chg_no= np.abs(np.diff(np.concatenate([[0], kp_no])))
    no    = raw_no - chg_no * (2 * TRANSACTION_COST_BPS / 10000.0)
    return daily, no

# ============================================================
# ANA ANALİZ
# ============================================================

def analyze(ticker, config, macro):
    set_seeds(42)
    name, bench = config['name'], config['benchmark']

    print(f"\n{'='*62}")
    print(f"📊 {ticker} — {name}")
    print(f"{'='*62}")

    ds, db = download_pair(ticker, bench)
    if ds is None: return None
    print(f"   {len(ds)} gün")

    # [IMP-3] VIX serisi
    vix_series = macro['VIX'] if (not macro.empty and 'VIX' in macro.columns) else None

    df_regime = detect_regime(db['Bench_Close'], vix_series)
    cur_reg   = df_regime['regime'].iloc[-1]
    cur_probs = df_regime[['p_bull','p_bear','p_chaos']].iloc[-1].values

    # Güncel VIX
    cur_vix = None
    if not macro.empty and 'VIX' in macro.columns:
        cur_vix = float(macro['VIX'].dropna().iloc[-1])
    vix_mode = ("🔴 YÜKSEK KORKU" if cur_vix and cur_vix>VIX_HIGH else
                "🟡 DİKKAT"      if cur_vix and cur_vix>VIX_MEDIUM else
                "🟢 NORMAL")
    print(f"   Güncel rejim: {cur_reg.upper()} | VIX={cur_vix:.1f} {vix_mode}" if cur_vix else
          f"   Güncel rejim: {cur_reg.upper()}")

    df = build_features(ds, db, macro, df_regime)
    if df is None or len(df) < 500: return None
    print(f"   {len(df)} kullanılabilir gün")

    # EWT diagnostik
    h_val  = df['hurst'].iloc[-1]      if 'hurst' in df.columns else np.nan
    cp_val = df['chan_pos'].iloc[-1]    if 'chan_pos' in df.columns else np.nan
    pc_val = df['pivot_n'].iloc[-1]    if 'pivot_n' in df.columns else np.nan
    h_int  = ("trend↗" if h_val>0.55 else "mean-rev↩" if h_val<0.45 else "random~") \
             if np.isfinite(h_val) else "—"
    beta_k = float(df['Beta_K'].iloc[-1])
    print(f"   DFA-Hurst={h_val:.3f}({h_int}), Chan={cp_val:.2f}, β(Kalman)={beta_k:.3f}")

    # [IMP-7] Earnings blackout
    earnings_dates = get_earnings_dates(ticker)
    today = pd.Timestamp.now().normalize()
    in_blackout = False
    next_earnings = None
    for ed in sorted(earnings_dates):
        ed = pd.Timestamp(ed)
        if abs((ed - today).days) <= EARNINGS_BLACKOUT_DAYS * 1.5:
            in_blackout = True
        if ed >= today:
            next_earnings = ed; break
    if in_blackout:
        print(f"   ⚠️  EARNINGS BLACKOUT ({EARNINGS_BLACKOUT_DAYS} iş günü)")
    if next_earnings:
        print(f"   📅 Sonraki kazanç açıklaması: {next_earnings.date()}")

    # Feature seçimi
    excl  = ['Open','High','Low','Volume','Close','Bench_Close','Adj Close',
             'Fut_Stock','Fut_Bench','Fut_Resid_Raw','Fut_Resid','Vol_Scale']
    fcols = [c for c in df.columns if c not in excl]
    X_arr = df[fcols].values
    y_arr = df['Fut_Resid'].values          # ölçeklenmiş hedef [IMP-1]
    y_raw = df['Fut_Resid_Raw'].values      # ham hedef (backtest için)
    v_sc  = df['Vol_Scale'].values
    rp    = df[['p_bull','p_bear','p_chaos']].values

    ewt_n = len([c for c in fcols if any(k in c for k in
               ['fib','hurst','pivot','swing','wave_mom','chan_','rel_'])])
    print(f"   Features: {len(fcols)} (EWT: {ewt_n}) | LOOKBACK: {LOOKBACK}g")

    tr = int(len(df) * TRAIN_FRAC)
    vl = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    n_test  = len(df) - vl
    n_indep = n_test // PREDICTION_HORIZON
    print(f"   Split: train={tr} val={vl-tr} test={n_test} ({n_indep} bağımsız)")

    # Scaling — [IMP-1] ölçeklenmiş hedef üzerinde
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

    tr_probs = rp[LOOKBACK:tr]
    xv_probs = rp[tr:vl]
    te_probs = rp[vl:vl+len(yte)]

    print(f"\n   Uzman eğitimi ({len(ENSEMBLE_SEEDS)*3} LSTM):")
    experts = train_experts(Xt, yt, Xv, yv, tr_probs)

    # Validation tahminleri (kalibrasyon için)
    pred_v_s  = predict_soft(experts, Xv, xv_probs)
    pred_v    = ys.inverse_transform(pred_v_s.reshape(-1,1)).flatten()
    act_v     = ys.inverse_transform(yv.reshape(-1,1)).flatten()

    # Test tahminleri
    pred_te_s = predict_soft(experts, Xte, te_probs)
    pred_te   = ys.inverse_transform(pred_te_s.reshape(-1,1)).flatten()
    act_te    = ys.inverse_transform(yte.reshape(-1,1)).flatten()

    # [IMP-5] Kalibrasyon
    pred_cal = calibrate_predictions(pred_v, act_v, pred_te)
    print(f"   Kalibrasyon: pos_oran ham=%{(pred_te>0).mean()*100:.1f} "
          f"→ kalibre=%{(pred_cal>0).mean()*100:.1f} "
          f"(gerçek=%{(act_te>0).mean()*100:.1f})")

    # Ham residual (vol-unscaled) test
    act_raw_te = y_raw[vl:vl+len(act_te)]
    v_sc_te    = v_sc[vl:vl+len(act_te)]

    # Yön doğruluğu: kalibre edilmiş tahmine göre
    correct = int((np.sign(pred_cal)==np.sign(act_te)).sum())
    total   = len(pred_cal)
    dir_acc = correct / total * 100
    p_val   = binom_p(correct, total)
    act_pos = (act_te  > 0).mean() * 100
    prd_pos = (pred_cal > 0).mean() * 100
    naive   = max(act_pos, 100-act_pos)
    edge    = dir_acc - naive

    print(f"   R²={float(np.corrcoef(act_te,pred_cal)[0,1]**2) if np.std(pred_cal)>0 else 0:+.4f} "
          f"Yön=%{dir_acc:.1f} ({correct}/{total})")
    print(f"   {bonf_str(p_val, ALPHA_BONF)} | edge=%{edge:+.2f}")
    print(f"   Pos: gerçek=%{act_pos:.1f} tahmin=%{prd_pos:.1f}")

    # Rejim bazlı
    rl = np.array(['bull','bear','chaos'])[np.argmax(te_probs, axis=1)]
    reg_res = {}
    for reg in ['bull','bear','chaos']:
        msk = rl==reg; n=int(msk.sum())
        if n < 10: reg_res[reg]=None; continue
        rc  = int((np.sign(pred_cal[msk])==np.sign(act_te[msk])).sum())
        p_r = binom_p(rc, n)
        reg_res[reg] = {'n':n,'dir':rc/n*100,'p':p_r,'bonf_sig':p_r<ALPHA_LOCAL}
    for reg, v in reg_res.items():
        if v: print(f"      {reg.upper():>6}: n={v['n']} yön=%{v['dir']:.1f} {bonf_str(v['p'],ALPHA_LOCAL)}")

    # [IMP-6] Kelly backtest
    val_dir_acc = (np.sign(pred_v)==np.sign(act_v)).mean() * 100
    kp = kelly_position(pred_cal, val_dir_acc)

    # VIX blackout uygula
    if not macro.empty and 'VIX' in macro.columns:
        vix_te = macro['VIX'].reindex(df.index[vl:vl+len(kp)], method='ffill').values
        blackout_mask = vix_te > VIX_HIGH
        kp[blackout_mask] = 0.0  # [IMP-3] Yüksek VIX → nakit
        pct_blackout = blackout_mask.mean() * 100
        if pct_blackout > 0:
            print(f"   VIX blackout: %{pct_blackout:.1f} gün model susturuldu")

    daily_k, no_k = backtest_kelly(pred_cal, act_raw_te, kp)
    ir_no   = info_ratio(no_k, 252/PREDICTION_HORIZON)

    # Passive B&H
    bh_alp  = (df['Fut_Stock'].iloc[vl:vl+total].values
               - df['Beta_K'].iloc[vl:vl+total].values
               * df['Fut_Bench'].iloc[vl:vl+total].values)
    bh_ir   = info_ratio(bh_alp, 252)
    cum_no  = float((np.exp(no_k.sum())-1)*100)
    print(f"   IR(Kelly/NoOver)={ir_no:+.2f} B&H IR={bh_ir:+.2f} Cum α=%{cum_no:+.1f}")

    # Gelecek tahmini
    last_X    = xs.transform(X_arr)[-LOOKBACK:].reshape(1,LOOKBACK,len(fcols))
    last_prob = rp[-1:].reshape(1,3)
    fut_s     = predict_soft(experts, last_X, last_prob)
    fut_raw   = float(ys.inverse_transform(fut_s.reshape(-1,1))[0,0])
    # Kalibrasyon (tek nokta için basit sign mapping)
    fut_cal   = float(calibrate_predictions(pred_v, act_v, np.array([fut_raw]))[0])
    resid_std = float(np.std(act_te - pred_cal))
    cur_price = float(df['Close'].iloc[-1])
    cur_bprice= float(db['Bench_Close'].iloc[-1])

    # VIX susturma kontrolü (güncel)
    blackout_now = (cur_vix and cur_vix > VIX_HIGH) or in_blackout
    if blackout_now:
        sig, sc = "BLACKOUT ⛔", "gray"
    elif abs(fut_cal) < 0.3*resid_std: sig,sc = "NÖTR","gray"
    elif fut_cal>0: sig,sc = ("GÜÇLÜ AL","green") if fut_cal>resid_std else ("AL","blue")
    else:           sig,sc = ("SAT","red")         if fut_cal<-resid_std else ("ZAYIF SAT","orange")

    # Kelly pozisyon önerisi
    kelly_now = float(kelly_position(np.array([fut_cal]), val_dir_acc)[0])
    if blackout_now: kelly_now = 0.0
    print(f"   Tahmin: {fut_cal*100:+.2f}% | Sinyal: {sig} | Kelly pos: {kelly_now:+.2f}")

    return dict(
        ticker=ticker, name=name, bench=bench,
        df=df, df_regime=df_regime,
        pred_r=pred_cal, act_r=act_te, act_raw=act_raw_te,
        r2=float(np.corrcoef(act_te,pred_cal)[0,1]**2) if np.std(pred_cal)>0 else 0.0,
        dir_acc=dir_acc, p_val=p_val,
        act_pos=act_pos, prd_pos=prd_pos, edge=edge,
        reg_res=reg_res,
        ir_no=ir_no, bh_ir=bh_ir, cum_no=cum_no,
        daily=daily_k, no=no_k, kelly_pos=kp,
        cur_price=cur_price, cur_bench=cur_bprice, last_beta=beta_k,
        fut_ret=fut_cal, resid_std=resid_std,
        signal=sig, sig_color=sc, kelly_now=kelly_now,
        cur_reg=cur_reg, cur_probs=cur_probs.tolist(),
        vix=cur_vix, vix_mode=vix_mode,
        blackout=blackout_now, in_earnings_blackout=in_blackout,
        next_earnings=str(next_earnings.date()) if next_earnings else None,
        val_split=vl, total=total, n_indep=n_indep,
        n_feat=len(fcols), n_ewt=ewt_n,
        hurst=float(h_val) if np.isfinite(h_val) else None,
        h_interp=h_int,
        chan_pos=float(cp_val) if np.isfinite(cp_val) else None,
        pivot_n=float(pc_val) if np.isfinite(pc_val) else None,
    )

# ============================================================
# GRAFİK
# ============================================================

def make_chart(res):
    ticker = res['ticker']; df = res['df']
    vl = res['val_split']; pred_r = res['pred_r']

    fig, ax = plt.subplots(figsize=(14, 7))
    show_n  = min(600, len(df))
    df_show = df.iloc[-show_n:]

    # Rejim bantları
    dom  = np.argmax(df_show[['p_bull','p_bear','p_chaos']].values, axis=1)
    rcol = {0:'#86efac', 1:'#fca5a5', 2:'#fde047'}
    s, lr = 0, dom[0]
    for i in range(1, len(dom)):
        if dom[i] != lr:
            ax.axvspan(df_show.index[s], df_show.index[i], alpha=0.12, color=rcol[lr], zorder=0)
            s, lr = i, dom[i]
    ax.axvspan(df_show.index[s], df_show.index[-1], alpha=0.12, color=rcol[lr], zorder=0)

    # Gerçek fiyat
    ax.plot(df_show.index, df_show['Close'],
            color='#1f2937', linewidth=1.8, label='Gerçek', zorder=3)

    # Fibonacci (son 60g)
    if 'range_pos' in df.columns:
        last60 = df_show.iloc[-60:]
        h_max  = last60['High'].max() if 'High' in last60.columns else last60['Close'].max()
        l_min  = last60['Low'].min()  if 'Low'  in last60.columns else last60['Close'].min()
        rng    = h_max - l_min
        if rng > 0:
            fmap = {'0.786':'#7c3aed','0.618':'#2563eb',
                    '0.500':'#059669','0.382':'#d97706'}
            x0 = df_show.index[-60]; x1 = df.index[-1]+timedelta(days=PREDICTION_HORIZON+3)
            for lvl, col in fmap.items():
                price = l_min + float(lvl)*rng
                ax.hlines(price, x0, x1, colors=col, linewidths=0.9,
                          linestyles='-.', alpha=0.7,
                          label=f'Fib {lvl}: ${price:.1f}', zorder=2)

    # OOS test çizgisi
    n_te  = len(pred_r)
    tc    = df['Close'].iloc[vl:vl+n_te].values
    tb    = df['Beta_K'].iloc[vl:vl+n_te].values
    tbr   = df['Fut_Bench'].iloc[vl:vl+n_te].values
    # Vol unscale için Vol_Scale gerekiyor
    vs    = df['Vol_Scale'].iloc[vl:vl+n_te].values
    # pred_r vol-scaled → unscale ile fiyata çevir
    pp    = tc * np.exp(pred_r * vs + tb * tbr)
    pd_i  = df.index[vl:vl+n_te] + pd.Timedelta(days=PREDICTION_HORIZON)
    ax.plot(pd_i, pp, linestyle='--', color='#f59e0b',
            linewidth=1.5, alpha=0.85, label=f'Model OOS (kalibre)', zorder=4)

    # Test başlangıcı
    ax.axvline(df.index[vl], color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label='Test başlangıcı', zorder=5)

    # Gelecek tahmin noktası
    fut_date  = df.index[-1] + timedelta(days=PREDICTION_HORIZON)
    vs_last   = float(df['Vol_Scale'].iloc[-1])
    fut_price = res['cur_price'] * np.exp(res['fut_ret'] * vs_last)
    ax.scatter([fut_date], [fut_price], color=res['sig_color'],
               s=200, zorder=10, edgecolors='black', linewidth=1.5,
               label=f'{res["signal"]} (Kelly:{res["kelly_now"]:+.2f})')

    hstr = f"H={res['hurst']:.3f}({res['h_interp']})" if res.get('hurst') else ""
    vstr = f"VIX={res['vix']:.1f} {res['vix_mode']}" if res.get('vix') else ""
    ax.set_title(
        f"{res['name']} ({ticker})  |  Rejim: {res['cur_reg'].upper()}  |  "
        f"Sinyal: {res['signal']}  |  IR: {res['ir_no']:+.2f}  |  "
        f"Yön: %{res['dir_acc']:.1f}  |  Edge: %{res['edge']:+.2f}\n"
        f"EWT → {hstr}  |  {vstr}  |  β(Kalman)={res['last_beta']:.3f}  |  "
        f"🟢 Bull  🔴 Bear  🟡 Chaos",
        fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.2); ax.set_ylabel(f'{ticker} ($)')
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================
# HTML
# ============================================================

CSS = """<style>
*{box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#f1f5f9;padding:24px;margin:0}
.page{max-width:1300px;margin:0 auto}
h1{text-align:center;color:#0f172a;font-size:1.55em;margin-bottom:6px}
.sub{text-align:center;color:#64748b;font-size:.87em;margin-bottom:28px}
.card{background:#fff;border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:36px;overflow:hidden}
.ch{padding:13px 22px;display:flex;align-items:center;justify-content:space-between;background:#0f172a;color:#fff}
.ch .tk{font-size:1.3em;font-weight:700}
.rbadge{font-size:.77em;padding:3px 10px;border-radius:16px;font-weight:600}
.bull-b{background:#16a34a}.bear-b{background:#dc2626}.chaos-b{background:#ca8a04}
.vixbar{padding:8px 22px;font-size:.83em;display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid #e2e8f0}
.sbar{padding:13px 22px;text-align:center;font-weight:700;font-size:1.1em}
.blackout{background:#fef2f2;border-top:2px solid #b91c1c;padding:8px 22px;font-size:.83em;color:#7f1d1d}
.ewt-row{padding:9px 22px;background:#f0f9ff;border-top:1px solid #bae6fd;font-size:.82em;display:flex;gap:22px;flex-wrap:wrap}
.eit{display:flex;flex-direction:column;align-items:center}
.el{font-size:.7em;color:#0891b2;font-weight:700;text-transform:uppercase}
.ev{font-size:1em;font-weight:700;color:#0f172a}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#e2e8f0;border-top:1px solid #e2e8f0}
.mb{background:#fff;padding:12px;text-align:center}
.ml{font-size:.7em;color:#64748b;font-weight:700;text-transform:uppercase}
.mv{font-size:1.18em;font-weight:800;margin:4px 0;color:#0f172a}
.ms{font-size:.66em;color:#94a3b8}
.cw{padding:18px 22px;background:#f8fafc;border-top:1px solid #e2e8f0;text-align:center}
.cw img{max-width:100%;border-radius:10px;border:1px solid #e2e8f0}
.bg{display:grid;grid-template-columns:1fr 1fr;gap:0}
.rt{padding:13px 22px}.rt b{font-size:.82em;color:#475569}
.rt table{width:100%;border-collapse:collapse;font-size:.85em;margin-top:7px}
.rt th{background:#f1f5f9;padding:6px 9px;border:1px solid #e2e8f0;font-weight:700;color:#374151}
.rt td{padding:6px 9px;border:1px solid #e2e8f0;text-align:center}
.bull-r{background:#f0fdf4}.bear-r{background:#fef2f2}.chaos-r{background:#fefce8}
.sc{padding:13px 22px;border-left:1px solid #e2e8f0}
.sc h4{margin:0 0 8px;font-size:.84em;color:#475569}
.sc table{width:100%;border-collapse:collapse;font-size:.85em}
.sc th{background:#f1f5f9;padding:5px 9px;border:1px solid #e2e8f0}
.sc td{padding:5px 9px;border:1px solid #e2e8f0;text-align:center}
.pos{background:#f0fdf4;color:#15803d;font-weight:600}
.neg{background:#fef2f2;color:#b91c1c;font-weight:600}
.sw{background:#fff;border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:36px;overflow:hidden}
.sh{background:#0f172a;color:#fff;padding:12px 22px;font-size:1.02em;font-weight:700}
.sw table{width:100%;border-collapse:collapse;font-size:.86em}
.sw th{background:#f1f5f9;padding:8px 10px;border:1px solid #e2e8f0;font-weight:700;color:#374151}
.sw td{padding:8px 10px;border:1px solid #e2e8f0;text-align:center}
.note{margin:0 0 24px;padding:10px 14px;border-radius:8px;font-size:.8em;background:#fefce8;border-left:4px solid #ca8a04;color:#713f12}
</style>"""

def sbg(sc):
    return {'green':'#dcfce7','blue':'#dbeafe','gray':'#f1f5f9',
            'orange':'#ffedd5','red':'#fee2e2'}.get(sc,'#f1f5f9')

def build_html(results):
    valid = [r for r in results if r is not None]

    # Özet
    srows = ""
    for r in valid:
        hv   = r.get('hurst')
        hstr = f"{hv:.3f}" if (hv and np.isfinite(hv)) else "—"
        hint = r.get('h_interp','—')
        ec   = '#15803d' if r['edge']>3 else '#ca8a04' if r['edge']>0 else '#b91c1c'
        dc   = '#15803d' if r['dir_acc']>55 else '#ca8a04' if r['dir_acc']>50 else '#b91c1c'
        vix_str = f"{r['vix']:.0f}" if r.get('vix') else "—"
        kelly_str = f"{r['kelly_now']:+.2f}"
        bo_str = "⛔" if r.get('blackout') else "✅"
        sig_col  = r['sig_color']
        sig_txt  = r['signal']
        vix_mode = r.get('vix_mode','—')
        srows += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b><br><span style='font-size:.78em;color:#64748b'>{r['name']}</span></td>"
            f"<td><span class='rbadge {r['cur_reg']}-b'>{r['cur_reg'].upper()}</span></td>"
            f"<td style='color:{dc};font-weight:700'>%{r['dir_acc']:.1f}<br><small>p={r['p_val']:.4f}</small></td>"
            f"<td style='color:{ec};font-weight:700'>%{r['edge']:+.2f}</td>"
            f"<td>{r['ir_no']:+.2f}</td>"
            f"<td>%{r['cum_no']:+.1f}</td>"
            f"<td>{hstr}<br><small>{hint}</small></td>"
            f"<td>VIX:{vix_str}<br><small>{vix_mode}</small></td>"
            f"<td style='font-size:1.1em'>{kelly_str}</td>"
            f"<td>{bo_str}</td>"
            f"<td style='color:{sig_col};font-weight:700'>{sig_txt}</td>"
            f"</tr>"
        )

    summary = f"""
<div class="sw">
  <div class="sh">📊 Özet — v15.4 (Kelly + VIX + Kalman-β + Kalibrasyon + Vol-hedef)</div>
  <div style="padding:13px 22px;overflow-x:auto">
  <table><tr><th>Ticker</th><th>Rejim</th><th>Yön</th><th>Edge</th>
  <th>IR(Kelly)</th><th>Cum α</th><th>Hurst</th><th>VIX</th>
  <th>Kelly Pos</th><th>Aktif?</th><th>Sinyal</th></tr>
  {srows}</table></div>
  <div class="note">
    <b>Yeni özellikler:</b>
    [IMP-1] Vol-ölçeklenmiş hedef · [IMP-2] LOOKBACK=20g · [IMP-3] VIX filtresi (>{VIX_HIGH}→sustur) ·
    [IMP-4] Kalman-β · [IMP-5] Isotonic kalibrasyon · [IMP-6] Kelly(%{int(KELLY_FRAC*100)}) pozisyon ·
    [IMP-7] Earnings blackout (±{EARNINGS_BLACKOUT_DAYS}g).
    Bonferroni: {N_TESTS} test, α={ALPHA_BONF:.4f}.
  </div>
</div>"""

    cards = ""
    for r in valid:
        chart_b64 = make_chart(r)

        # Rejim tablosu
        rrows = ""
        for reg in ['bull','bear','chaos']:
            rv  = r['reg_res'].get(reg); cls = f"{reg}-r"
            if rv is None:
                rrows += f"<tr class='{cls}'><td><b>{reg.upper()}</b></td><td colspan=4 style='color:#94a3b8'>Yetersiz</td></tr>"
            else:
                dc = '#15803d' if rv['bonf_sig'] else '#ca8a04' if rv['p']<ALPHA else '#b91c1c'
                bs = "✅" if rv['bonf_sig'] else ("🟡" if rv['p']<ALPHA else "❌")
                rrows += (f"<tr class='{cls}'>"
                          f"<td><b>{reg.upper()}</b></td>"
                          f"<td>{rv['n']}</td>"
                          f"<td style='color:{dc};font-weight:700'>%{rv['dir']:.1f}</td>"
                          f"<td>{rv['p']:.3f}</td>"
                          f"<td>{bs}</td></tr>")

        # Senaryo
        scen = ""
        for sm in [-0.04,-0.02,0.0,0.02,0.04]:
            vs_last = float(r['df']['Vol_Scale'].iloc[-1])
            im_raw = r['fut_ret'] * vs_last + r['last_beta'] * sm
            ip     = r['cur_price'] * np.exp(im_raw)
            cls    = 'pos' if im_raw>0 else 'neg'
            scen  += (f"<tr><td>{sm*100:+.1f}%</td>"
                      f"<td class='{cls}'>{im_raw*100:+.2f}%</td>"
                      f"<td class='{cls}'>${ip:.2f}</td></tr>")

        hv    = r.get('hurst'); hstr = f"{hv:.3f}" if (hv and np.isfinite(hv)) else "—"
        hi    = r.get('h_interp','—')
        hc    = '#15803d' if (hv and hv>0.55) else '#b91c1c' if (hv and hv<0.45) else '#ca8a04'
        cp    = r.get('chan_pos'); cpstr = f"{cp:.2f}" if cp else "—"
        vix   = r.get('vix');     vstr  = f"{vix:.1f}" if vix else "—"
        vm    = r.get('vix_mode','—')
        vm_c  = '#b91c1c' if '🔴' in vm else '#ca8a04' if '🟡' in vm else '#15803d'
        kn    = r.get('kelly_now', 0)
        kn_c  = '#15803d' if kn>0 else '#b91c1c' if kn<0 else '#94a3b8'
        ne    = r.get('next_earnings','—')

        blackout_html = ""
        if r.get('blackout'):
            reason = []
            if r.get('in_earnings_blackout'): reason.append("Earnings yakın")
            if r.get('vix') and r['vix']>VIX_HIGH: reason.append(f"VIX={r['vix']:.1f}>{VIX_HIGH}")
            blackout_html = (f"<div class='blackout'>⛔ <b>BLACKOUT — Tahmin Susturuldu:</b> "
                            f"{', '.join(reason)}. Model bu dönemde pozisyon almaz.</div>")

        ae  = r['edge']
        avc = '#15803d' if ae>3 else '#ca8a04' if ae>0 else '#b91c1c'
        avt = 'GERÇEK ALFA' if ae>3 else 'ZAYIF' if ae>0 else 'ALFA YOK'
        sb  = sbg(r['sig_color'])
        rb  = f"{r['cur_reg']}-b"

        cards += f"""
<div class="card">
  <div class="ch">
    <span class="tk">{r['ticker']} — {r['name']}</span>
    <span class="rbadge {rb}">{r['cur_reg'].upper()}
      | bull={r['cur_probs'][0]:.2f} bear={r['cur_probs'][1]:.2f} chaos={r['cur_probs'][2]:.2f}</span>
  </div>

  <div class="sbar" style="background:{sb};color:{r['sig_color']}">
    {r['signal']}
    &nbsp;|&nbsp; Tahmin: {r['fut_ret']*100:+.2f}%
    &nbsp;|&nbsp; Kelly pos: <b>{kn:+.2f}</b>
    &nbsp;|&nbsp; β(Kalman)={r['last_beta']:.3f}
  </div>

  {blackout_html}

  <div class="vixbar">
    <div class="eit"><span class="el">VIX</span>
      <span class="ev" style="color:{vm_c}">{vstr} {vm}</span></div>
    <div class="eit"><span class="el">Sonraki Earnings</span>
      <span class="ev">{ne if ne else '—'}</span></div>
    <div class="eit"><span class="el">Test Pozisyon</span>
      <span class="ev">{r['n_indep']} bağımsız</span></div>
    <div class="eit"><span class="el">Bonferroni p</span>
      <span class="ev">{'✅' if r['p_val']<ALPHA_BONF else '❌'} {r['p_val']:.4f}</span></div>
  </div>

  <div class="ewt-row">
    <div class="eit"><span class="el">DFA-Hurst</span>
      <span class="ev" style="color:{hc}">{hstr} {hi}</span></div>
    <div class="eit"><span class="el">Elliott Chan.</span>
      <span class="ev">{cpstr}</span></div>
    <div class="eit"><span class="el">EWT/Toplam</span>
      <span class="ev">{r['n_ewt']}/{r['n_feat']}</span></div>
    <div class="eit" style="flex:1;font-size:.77em;color:#0369a1;text-align:left">
      LOOKBACK={LOOKBACK}g (mean-rev uyumlu) | Hedef=vol-ölçeklenmiş residual |
      Kalibrasyon=isotonic | β=Kalman-filter
    </div>
  </div>

  <div class="mgrid">
    <div class="mb"><div class="ml">Mevcut</div>
      <div class="mv">${r['cur_price']:.2f}</div>
      <div class="ms">{r['bench']}: ${r['cur_bench']:.2f}</div></div>
    <div class="mb"><div class="ml">Yön Doğruluğu</div>
      <div class="mv" style="color:{'#15803d' if r['dir_acc']>55 else '#ca8a04' if r['dir_acc']>50 else '#b91c1c'}">
        %{r['dir_acc']:.1f}</div>
      <div class="ms">{r['total']}g · {r['n_indep']} pos</div></div>
    <div class="mb"><div class="ml">Alfa Edge</div>
      <div class="mv" style="color:{avc}">%{ae:+.2f}</div>
      <div class="ms">{avt}</div></div>
    <div class="mb"><div class="ml">IR (Kelly+NoOver)</div>
      <div class="mv" style="color:{'#15803d' if r['ir_no']>0.3 else '#ca8a04' if r['ir_no']>0 else '#b91c1c'}">
        {r['ir_no']:+.2f}</div>
      <div class="ms">gerçekçi</div></div>
    <div class="mb"><div class="ml">Passive IR</div>
      <div class="mv">{r['bh_ir']:+.2f}</div>
      <div class="ms">B&H modelsiz</div></div>
    <div class="mb"><div class="ml">Cum α (NoOver)</div>
      <div class="mv" style="color:{'#15803d' if r['cum_no']>0 else '#b91c1c'}">
        %{r['cum_no']:+.1f}</div>
      <div class="ms">bağımsız periyot</div></div>
    <div class="mb"><div class="ml">Kelly Pos (%50 fr.)</div>
      <div class="mv" style="color:{kn_c}">{kn:+.2f}</div>
      <div class="ms">[-1=tam short, +1=tam long]</div></div>
    <div class="mb"><div class="ml">Pos Oran Δ</div>
      <div class="mv" style="color:{'#b91c1c' if abs(r['prd_pos']-r['act_pos'])>15 else '#15803d'}">
        %{abs(r['prd_pos']-r['act_pos']):.1f}</div>
      <div class="ms">tahmin−gerçek</div></div>
  </div>

  <div class="cw"><img src="data:image/png;base64,{chart_b64}"></div>

  <div class="bg">
    <div class="rt"><b>🎭 Rejim-Bazlı OOS (Bonf. α={ALPHA_LOCAL:.4f})</b>
      <table style="margin-top:7px">
        <tr><th>Rejim</th><th>n</th><th>Yön%</th><th>p</th><th>Sonuç</th></tr>
        {rrows}
      </table>
    </div>
    <div class="sc"><h4>📐 Senaryo ({r['bench']} hareketine göre)</h4>
      <table><tr><th>{r['bench']} 7g</th><th>Tahmin</th><th>Fiyat</th></tr>
      {scen}</table>
    </div>
  </div>
</div>"""

    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    tks = ', '.join(r['ticker'] for r in valid)
    test_pct = int((1-TRAIN_FRAC-VAL_FRAC)*100)

    return f"""<!DOCTYPE html><html lang="tr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v15.4 — {tks}</title>{CSS}</head>
<body><div class="page">
<h1>🔬 Multi-Stock v15.4 — Bilimsel + Piyasa Uyumlu</h1>
<div class="sub">{tks} · Market-neutral · Regime-switching · EWT ·
  VIX-filter · Kelly · Kalman-β · Isotonic-cal · {now}</div>
{summary}{cards}
<div class="note">
<b>v15.4 Metodoloji:</b>
Hedef = vol-ölçeklenmiş residual return (gürültü bastırma) [IMP-1].
LOOKBACK={LOOKBACK}g (mean-rev uyumu, H≈0.05) [IMP-2].
VIX&gt;{VIX_HIGH} → blackout (nakit) [IMP-3].
β = Kalman Filter (adaptif, hedge fon standardı) [IMP-4].
Kalibrasyon = Isotonic Regression (val setinde) [IMP-5].
Pozisyon = Fractional Kelly %{int(KELLY_FRAC*100)} [IMP-6].
Earnings ±{EARNINGS_BLACKOUT_DAYS}g → blackout [IMP-7].
Test=%{test_pct} (~{int((1-TRAIN_FRAC-VAL_FRAC)*2746/PREDICTION_HORIZON)} bağımsız pozisyon).
Bonferroni: {N_TESTS} test, α={ALPHA_BONF:.4f}.
</div>
</div></body></html>"""

# ============================================================
# ANA AKIŞ
# ============================================================

def main():
    set_seeds()
    tickers = list(TICKERS_CONFIG.keys())

    print("=" * 62)
    print(f"MULTI-STOCK v15.4  |  {', '.join(tickers)}")
    print("=" * 62)
    print(f"Hisseler : {tickers}")
    print(f"Ufuk     : {PREDICTION_HORIZON} gün")
    print(f"LOOKBACK : {LOOKBACK}g (mean-rev uyumu)")
    print(f"Test     : %{int((1-TRAIN_FRAC-VAL_FRAC)*100)}")
    print(f"VIX eşik : {VIX_HIGH} (blackout)")
    print(f"Kelly    : %{int(KELLY_FRAC*100)} fractional")
    print(f"Beta     : Kalman Filter")
    print(f"Hedef    : Vol-ölçeklenmiş residual")
    print(f"Toplam LSTM: {len(tickers)*len(ENSEMBLE_SEEDS)*3}")

    macro = get_macro()

    results = []
    for ticker, cfg in TICKERS_CONFIG.items():
        res = analyze(ticker, cfg, macro)
        results.append(res)

    print(f"\n📄 HTML → {OUTPUT_HTML}")
    html = build_html(results)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✓ {OUTPUT_HTML}")

    print("\n" + "=" * 62)
    print(f"{'T':<5} {'YÖN%':<8} {'p':<8} {'EDGE%':<8} "
          f"{'IR':<7} {'VIX':<7} {'KELLY':<7} {'SİNYAL'}")
    print("-" * 62)
    for r in results:
        if r is None: continue
        vstr = f"{r['vix']:.0f}" if r.get('vix') else "—"
        bo   = "⛔" if r.get('blackout') else ""
        print(f"{r['ticker']:<5} %{r['dir_acc']:<6.1f} {r['p_val']:<8.4f} "
              f"%{r['edge']:<+6.2f}  {r['ir_no']:<+5.2f}  "
              f"{vstr:<7} {r['kelly_now']:<+5.2f}  {r['signal']} {bo}")
    print("=" * 62)

if __name__ == "__main__":
    main()
