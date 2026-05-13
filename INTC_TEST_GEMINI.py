# ============================================================
# MULTI-STOCK v15.1 — REGIME-SWITCHING MARKET-NEUTRAL FRAMEWORK
# (INTC, AMD, NVDA Entegrasyonu + SQLite İyileştirmeleri)
# ============================================================

import sys, os, sqlite3, warnings, random, base64, json
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
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
    'INTC': {
        'name': 'Intel Corporation',
        'benchmark': 'SOXX',
        'sector': 'semiconductors',
    },
    'AMD': {
        'name': 'Advanced Micro Devices',
        'benchmark': 'SOXX',
        'sector': 'semiconductors',
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'benchmark': 'SOXX',
        'sector': 'semiconductors',
    }
}

# Genel parametreler
PREDICTION_HORIZON = 7
TRANSACTION_COST_BPS = 5
LOOKBACK = 60
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10  # Test = 0.10
ENSEMBLE_SEEDS = [42, 52, 62]  # 3-model ensemble per regime
EPOCHS = 60

# --- DB ---
DB_FOLDER = r"C:\Projects\ML"
DB_NAME = "data_multi_v15_1.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS gunluk_veriler (
        tarih TEXT, sembol TEXT,
        acilis REAL, yuksek REAL, dusuk REAL, kapanis REAL, hacim REAL,
        UNIQUE(tarih, sembol))''')
    conn.commit()
    conn.close()

def save_to_db(df, ticker):
    """İndirilen verileri veritabanına kaydeder."""
    try:
        conn = sqlite3.connect(DB_PATH)
        temp_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        temp_df['sembol'] = ticker
        temp_df.rename(columns={'Open': 'acilis', 'High': 'yuksek', 'Low': 'dusuk', 
                                'Close': 'kapanis', 'Volume': 'hacim'}, inplace=True)
        temp_df.to_sql('gunluk_veriler', conn, if_exists='append', index_label='tarih', method='multi', chunksize=500)
        conn.close()
    except Exception as e:
        pass # Veri zaten varsa (UNIQUE constraint) sessizce geç

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_random.set_seed(seed)

# ============================================================
# VERİ
# ============================================================
def download_pair_data(ticker, benchmark):
    end = datetime.now()
    start = end - timedelta(days=12*365)
    try:
        df_stock = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        df_bench = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"   HATA: {ticker}/{benchmark} indirilemedi: {e}")
        return None, None
        
    if df_stock is None or df_stock.empty or df_bench is None or df_bench.empty:
        return None, None
        
    if hasattr(df_stock.columns, 'get_level_values'):
        df_stock.columns = df_stock.columns.get_level_values(0)
    if hasattr(df_bench.columns, 'get_level_values'):
        df_bench.columns = df_bench.columns.get_level_values(0)
        
    df_stock = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_bench = df_bench[['Close']].rename(columns={'Close': 'Bench_Close'}).dropna()
    
    # DB Kaydı
    save_to_db(df_stock, ticker)
    
    common_idx = df_stock.index.intersection(df_bench.index)
    return df_stock.loc[common_idx], df_bench.loc[common_idx]

def get_macro_data():
    end = datetime.now()
    start = end - timedelta(days=12*365)
    tickers = {"^VIX": "VIX", "^TNX": "US_10Y_BOND", "DX-Y.NYB": "DXY"}
    try:
        df = yf.download(list(tickers.keys()), start=start, end=end, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.rename(columns=tickers, inplace=True)
        return df.ffill()
    except Exception as e:
        print(f"   UYARI: Makro veri ({e})")
        return pd.DataFrame()

# ============================================================
# REJIM TESPİTİ (HİBRİT, 3 REJIM)
# ============================================================
def detect_regime_probabilities(bench_close, window_trend=50, window_long=200, window_vol=20, vol_quantile=0.7):
    log_ret = np.log(bench_close / bench_close.shift(1))
    sma_short = bench_close.rolling(window_trend).mean()
    sma_long = bench_close.rolling(window_long).mean()
    
    trend_score = (sma_short / sma_long) - 1.0
    trend_strength = 1.0 / (1.0 + np.exp(-trend_score * 50))  
    
    vol = log_ret.rolling(window_vol).std()
    vol_rolling_quantile = vol.expanding(min_periods=252).quantile(vol_quantile)
    
    chaos_signal = (vol / vol_rolling_quantile - 1.0) * 5  
    p_chaos = 1.0 / (1.0 + np.exp(-chaos_signal))
    p_chaos = p_chaos.clip(0.05, 0.95)  
    
    remaining = 1.0 - p_chaos
    p_bull = remaining * trend_strength
    p_bear = remaining * (1.0 - trend_strength)
    
    df_regime = pd.DataFrame({
        'p_bull': p_bull, 'p_bear': p_bear, 'p_chaos': p_chaos,
        'trend_strength': trend_strength, 'vol_20': vol,
    }, index=bench_close.index)
    
    probs = df_regime[['p_bull', 'p_bear', 'p_chaos']].values
    df_regime['regime'] = np.array(['bull', 'bear', 'chaos'])[np.argmax(probs, axis=1)]
    return df_regime

def rolling_beta(stock_returns, bench_returns, window=60):
    beta = np.full(len(stock_returns), np.nan)
    s = stock_returns.values; b = bench_returns.values
    for i in range(window, len(s)):
        s_w = s[i-window:i]; b_w = b[i-window:i]
        if np.std(b_w) == 0: continue
        cov = np.cov(s_w, b_w)[0, 1]
        var_b = np.var(b_w)
        beta[i] = cov / var_b if var_b > 0 else 1.0
    return pd.Series(beta, index=stock_returns.index)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_features(df_stock, df_bench, macro_df, df_regime):
    df = df_stock.copy()
    df['Bench_Close'] = df_bench['Bench_Close']
    
    try:
        import ta
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['SMA20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        df['SMA50'] = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
        df['SMA200'] = ta.trend.SMAIndicator(df['Close'], 200).sma_indicator()
        
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        df['Mom_20'] = df['Close'].pct_change(20)
        
        df['Bench_Log_Ret'] = np.log(df['Bench_Close'] / df['Bench_Close'].shift(1))
        df['Bench_Mom_20'] = df['Bench_Close'].pct_change(20)
        df['Bench_Vol_20'] = df['Bench_Log_Ret'].rolling(20).std()
        
        df['Rel_Mom_20'] = df['Mom_20'] - df['Bench_Mom_20']
        df['Rel_Vol_20'] = df['Vol_20'] - df['Bench_Vol_20']
        
        df['Beta_60'] = rolling_beta(df['Log_Ret'].fillna(0), df['Bench_Log_Ret'].fillna(0), window=60)
        df['Beta_60'] = df['Beta_60'].clip(0.0, 3.0)
        
        # Gelecekteki Residual Getiri (Hedef)
        future_stock = np.log(df['Close'].shift(-PREDICTION_HORIZON) / df['Close'])
        future_bench = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON) / df['Bench_Close'])
        df['Future_Residual_Ret'] = future_stock - df['Beta_60'] * future_bench
        
        df = df.join(df_regime[['p_bull', 'p_bear', 'p_chaos']], how='left')
        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill()
            
        df = df.dropna(subset=['Future_Residual_Ret', 'Beta_60', 'p_bull', 'p_bear', 'p_chaos'])
        df = df.dropna()
        return df
    except Exception as e:
        print(f"   HATA Feature engineering: {e}")
        return None

# ============================================================
# MODEL & ENSEMBLE EĞİTİMİ
# ============================================================
def build_lstm(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def create_dataset_clean(X_arr, y_arr, lb):
    Xs, ys = [], []
    for i in range(lb, len(X_arr)):
        Xs.append(X_arr[i-lb:i]); ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)

def train_regime_experts(X_tr, y_tr, X_v, y_v, tr_probs, v_probs_full):
    experts = {}
    for regime_idx, regime_name in enumerate(['bull', 'bear', 'chaos']):
        print(f"      🎓 Uzman: {regime_name.upper()} eğitiliyor...")
        sw = tr_probs[:, regime_idx]
        sw_norm = np.clip(sw, 0.05, 1.0)
        
        effective_n = sw_norm.sum()
        if effective_n < 100:
            print(f"         ⚠️  Yetersiz veri ({effective_n:.0f}), atlama")
            experts[regime_name] = None
            continue
            
        models = []
        for sd in ENSEMBLE_SEEDS:
            set_seeds(sd)
            m = build_lstm((X_tr.shape[1], X_tr.shape[2]))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
            m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=32, validation_data=(X_v, y_v), 
                  sample_weight=sw_norm, callbacks=[es, rlr], verbose=0)
            models.append(m)
        experts[regime_name] = models
        print(f"         ✓ {regime_name}: effective_n={effective_n:.0f}, {len(models)} model")
    return experts

def soft_ensemble_predict(experts, X, probs):
    regime_preds = {}
    for regime_idx, regime_name in enumerate(['bull', 'bear', 'chaos']):
        if experts.get(regime_name) is None:
            regime_preds[regime_name] = np.zeros(len(X))
            continue
        preds_per_seed = [m.predict(X, verbose=0).flatten() for m in experts[regime_name]]
        regime_preds[regime_name] = np.mean(preds_per_seed, axis=0)
        
    final = (probs[:, 0] * regime_preds['bull'] + 
             probs[:, 1] * regime_preds['bear'] + 
             probs[:, 2] * regime_preds['chaos'])
    return final

# ============================================================
# HİSSE BAZLI ANALİZ AKIŞI
# ============================================================
def analyze_ticker(ticker, config, macro_df):
    print(f"\n{'='*65}")
    print(f"📊 {ticker} ({config['name']}) — Regime-Switching Analiz")
    print(f"{'='*65}")
    
    df_stock, df_bench = download_pair_data(ticker, config['benchmark'])
    if df_stock is None: return None
    
    print("1. Rejim tespit ediliyor...")
    df_regime = detect_regime_probabilities(df_bench['Bench_Close'])
    
    print("2. Özellik mühendisliği (Feature Engineering) uygulanıyor...")
    df = build_features(df_stock, df_bench, macro_df, df_regime)
    if df is None or len(df) < 500: return None
    
    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Close', 'Bench_Close', 'Future_Residual_Ret']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_data = df[feature_cols].values
    y_data = df['Future_Residual_Ret'].values
    probs_arr = df[['p_bull', 'p_bear', 'p_chaos']].values
    
    train_split = int(len(df) * TRAIN_FRAC)
    val_split = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    
    x_sc = MinMaxScaler(); x_sc.fit(X_data[:train_split])
    y_sc = MinMaxScaler((-1, 1)); y_sc.fit(y_data[:train_split].reshape(-1, 1))
    
    X_tr_s = x_sc.transform(X_data[:train_split])
    y_tr_s = y_sc.transform(y_data[:train_split].reshape(-1, 1)).flatten()
    X_v_s = x_sc.transform(X_data[train_split-LOOKBACK:val_split])
    y_v_s = y_sc.transform(y_data[train_split-LOOKBACK:val_split].reshape(-1, 1)).flatten()
    X_te_s = x_sc.transform(X_data[val_split-LOOKBACK:])
    y_te_s = y_sc.transform(y_data[val_split-LOOKBACK:].reshape(-1, 1)).flatten()
    
    Xt, yt = create_dataset_clean(X_tr_s, y_tr_s, LOOKBACK)
    Xv, yv = create_dataset_clean(X_v_s, y_v_s, LOOKBACK)
    Xte, yte = create_dataset_clean(X_te_s, y_te_s, LOOKBACK)
    
    print("3. Modeller eğitiliyor...")
    experts = train_regime_experts(Xt, yt, Xv, yv, probs_arr[LOOKBACK:train_split], probs_arr[train_split:val_split])
    
    print("4. OOS (Out-of-Sample) Test ve Sinyal Üretimi...")
    pred_te_s = soft_ensemble_predict(experts, Xte, probs_arr[val_split:val_split+len(yte)])
    actual_residual = y_sc.inverse_transform(yte.reshape(-1, 1)).flatten()
    pred_residual = y_sc.inverse_transform(pred_te_s.reshape(-1, 1)).flatten()
    
    dir_acc = (np.sign(pred_residual) == np.sign(actual_residual)).mean() * 100
    r2 = r2_score(actual_residual, pred_residual)
    
    # Gelecek Tahmini (Son 60 gün)
    last_X = x_sc.transform(X_data)[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    last_probs = probs_arr[-1:].reshape(1, 3)
    future_pred_s = soft_ensemble_predict(experts, last_X, last_probs)
    pred_resid_future = float(y_sc.inverse_transform(future_pred_s.reshape(-1, 1))[0, 0])
    
    residual_std = float(np.std(actual_residual - pred_residual))
    
    if abs(pred_resid_future) < 0.5 * residual_std:
        sig = "NÖTR"
    elif pred_resid_future > 0:
        sig = "AL" if abs(pred_resid_future) < 1.5 * residual_std else "GÜÇLÜ AL"
    else:
        sig = "ZAYIF SAT" if abs(pred_resid_future) < 1.5 * residual_std else "SAT"
        
    return {
        'ticker': ticker, 'name': config['name'],
        'regime': df_regime['regime'].iloc[-1],
        'dir_acc': dir_acc, 'r2': r2, 'signal': sig, 'pred': pred_resid_future
    }

# ============================================================
# ANA ÇALIŞTIRICI
# ============================================================
if __name__ == "__main__":
    init_db()
    set_seeds()
    print("Makro veriler indiriliyor...")
    macro_df = get_macro_data()
    
    results = []
    for ticker, config in TICKERS_CONFIG.items():
        res = analyze_ticker(ticker, config, macro_df)
        if res:
            results.append(res)
            
    print("\n" + "="*65)
    print("MULTI-STOCK ÖZET TABLO (v15.1):")
    print("="*65)
    print(f"{'TICKER':<10} {'REJİM':<10} {'YÖN DOĞR':<15} {'R²':<10} {'SİNYAL'}")
    print("-" * 65)
    for r in results:
        print(f"{r['ticker']:<10} {r['regime'].upper():<10} %{r['dir_acc']:<14.1f} {r['r2']:<10.4f} {r['signal']}")
    print("="*65)