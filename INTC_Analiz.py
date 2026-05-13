# ============================================================
# INTC v15.0 — REGIME-SWITCHING MARKET-NEUTRAL FRAMEWORK
#                + MULTI-STOCK PARAMETRIC
#
# v14.0'dan farklar:
#   [REGIME-1] HİBRİT REJIM TESPİTİ:
#              BULL  : SMA50 > SMA200 ve vol_20 düşük
#              BEAR  : SMA50 < SMA200
#              CHAOS : Yüksek volatilite (herhangi bir trend)
#              → SOXX (benchmark) üzerinden hesaplanır.
#              → Soft probabilities döner (yumuşak geçiş)
#
#   [REGIME-2] SOFT ENSEMBLE:
#              3 ayrı LSTM (bull/bear/chaos uzmanları) eğitilir.
#              Tahmin = Σ p_regime × model_regime.predict()
#              Sert geçiş yok, rejim olasılıkları ile karışım.
#
#   [MULTI-STOCK] TICKERS_CONFIG dict yapısı:
#                 Her hisse için ayrı analiz, sonunda karşılaştırma raporu.
#                 Yeni hisse eklemek = bir satır eklemek.
#
#   [OUTPUTS] Her hisse için ayrı HTML + karşılaştırma raporu.
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
from sklearn.metrics import r2_score, mean_squared_error
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
# 🎯 KONFİGÜRASYON — HİSSE EKLEMEK İÇİN BURAYI GENİŞLET
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
    # ÖRNEK: Yeni hisse eklemek için:
    # 'NVDA': {'name': 'NVIDIA', 'benchmark': 'SOXX', 'sector': 'semiconductors'},
    # 'AAPL': {'name': 'Apple Inc',  'benchmark': 'SPY', 'sector': 'tech'},
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
DB_NAME = "data_multi_v15_0.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS gunluk_veriler (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tarih TEXT, sembol TEXT,
        acilis REAL, yuksek REAL, dusuk REAL, kapanis REAL, hacim REAL,
        UNIQUE(tarih, sembol))''')
    conn.commit(); conn.close()

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf_random.set_seed(seed)

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
    common_idx = df_stock.index.intersection(df_bench.index)
    return df_stock.loc[common_idx], df_bench.loc[common_idx]

def get_macro_data():
    end = datetime.now(); start = end - timedelta(days=12*365)
    tickers = {"^VIX": "VIX", "^TNX": "US_10Y_BOND", "DX-Y.NYB": "DXY",
               "^GSPC": "SP500", "^IXIC": "NASDAQ"}
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
def detect_regime_probabilities(bench_close, window_trend=50, window_long=200,
                                 window_vol=20, vol_quantile=0.7):
    """
    Hibrit rejim tespit:
      - Trend gücü: SMA50 / SMA200
      - Volatilite seviyesi: vol_20 vs tarihsel quantile
    
    Soft probabilities döner (3 rejim):
      p_bull   : SMA50 > SMA200 ve düşük volatilite (sağlıklı trend)
      p_bear   : SMA50 < SMA200 (negatif trend, vol fark etmez)
      p_chaos  : Yüksek volatilite (rejim ne olursa olsun)
    
    Olasılıklar [0,1], toplam = 1, yumuşak geçişlerle.
    """
    log_ret = np.log(bench_close / bench_close.shift(1))
    sma_short = bench_close.rolling(window_trend).mean()
    sma_long = bench_close.rolling(window_long).mean()
    
    # Trend skoru: SMA50/SMA200 - 1
    # Pozitif → bull, Negatif → bear
    trend_score = (sma_short / sma_long) - 1.0
    # Sigmoid ile [0,1]'e normalize
    trend_strength = 1.0 / (1.0 + np.exp(-trend_score * 50))  # 50 scaling = keskin
    # trend_strength ≈ 1 → güçlü bull, ≈ 0 → güçlü bear
    
    # Volatilite skoru: rolling vol_20'nin tarihsel quantile'a göre konumu
    vol = log_ret.rolling(window_vol).std()
    vol_rolling_quantile = vol.expanding(min_periods=252).quantile(vol_quantile)
    # vol > rolling_quantile → yüksek vol (chaos)
    chaos_signal = (vol / vol_rolling_quantile - 1.0) * 5  # scaling
    p_chaos = 1.0 / (1.0 + np.exp(-chaos_signal))
    p_chaos = p_chaos.clip(0.05, 0.95)  # extremes'i sıkıştır
    
    # Bull ve bear, kalan (1-p_chaos) olasılığı paylaşır
    remaining = 1.0 - p_chaos
    p_bull = remaining * trend_strength
    p_bear = remaining * (1.0 - trend_strength)
    
    df_regime = pd.DataFrame({
        'p_bull': p_bull, 'p_bear': p_bear, 'p_chaos': p_chaos,
        'trend_strength': trend_strength, 'vol_20': vol,
    }, index=bench_close.index)
    
    # Discrete regime (en yüksek olasılıklı)
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
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['SMA20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        df['SMA50'] = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
        df['SMA200'] = ta.trend.SMAIndicator(df['Close'], 200).sma_indicator()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_pct'] = (df['Close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_5'] = df['Log_Ret'].rolling(5).std()
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()
        df['Mom_5'] = df['Close'].pct_change(5)
        df['Mom_20'] = df['Close'].pct_change(20)
        df['Vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Px_SMA50'] = df['Close'] / df['SMA50']
        df['Px_SMA200'] = df['Close'] / df['SMA200']
        
        # Sektör features
        df['Bench_Log_Ret'] = np.log(df['Bench_Close'] / df['Bench_Close'].shift(1))
        df['Bench_Mom_5'] = df['Bench_Close'].pct_change(5)
        df['Bench_Mom_20'] = df['Bench_Close'].pct_change(20)
        df['Bench_Vol_20'] = df['Bench_Log_Ret'].rolling(20).std()
        df['Rel_Mom_5'] = df['Mom_5'] - df['Bench_Mom_5']
        df['Rel_Mom_20'] = df['Mom_20'] - df['Bench_Mom_20']
        df['Rel_Vol_20'] = df['Vol_20'] - df['Bench_Vol_20']
        df['Rel_Strength'] = df['Close'] / df['Bench_Close']
        df['Rel_Strength_SMA20'] = df['Rel_Strength'].rolling(20).mean()
        df['Rel_Strength_Dev'] = (df['Rel_Strength'] - df['Rel_Strength_SMA20']) / df['Rel_Strength_SMA20']
        df['Beta_60'] = rolling_beta(df['Log_Ret'].fillna(0), df['Bench_Log_Ret'].fillna(0), window=60)
        df['Beta_60'] = df['Beta_60'].clip(0.0, 3.0)
        df['Corr_60'] = df['Log_Ret'].rolling(60).corr(df['Bench_Log_Ret'])
        
        # REJIM olasılıkları feature olarak ekle
        df = df.join(df_regime[['p_bull', 'p_bear', 'p_chaos']], how='left')
        
        # Hedef: residual return
        future_stock = np.log(df['Close'].shift(-PREDICTION_HORIZON) / df['Close'])
        future_bench = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON) / df['Bench_Close'])
        df['Future_Stock_Ret'] = future_stock
        df['Future_Bench_Ret'] = future_bench
        df['Future_Residual_Ret'] = future_stock - df['Beta_60'] * future_bench
        
        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill()
        
        df = df.dropna(subset=['Future_Residual_Ret', 'Beta_60', 'p_bull', 'p_bear', 'p_chaos'])
        df = df.dropna()
        return df
    except Exception as e:
        print(f"   HATA Feature engineering: {e}")
        import traceback; traceback.print_exc()
        return None

# ============================================================
# STATISTIC HELPERS
# ============================================================
def binom_dir(c, t, p0=0.5):
    if t < 10: return 1.0
    return stats.binomtest(c, t, p=p0, alternative='greater').pvalue

def information_ratio(returns, periods_per_year):
    arr = np.asarray(returns); arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    sd = np.std(arr, ddof=1)
    if sd <= 0: return 0.0
    return np.sqrt(periods_per_year) * np.mean(arr) / sd

# ============================================================
# MODEL
# ============================================================
def create_dataset_clean(X_arr, y_arr, lb):
    Xs, ys = [], []
    for i in range(lb, len(X_arr)):
        Xs.append(X_arr[i-lb:i]); ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape, dropout=0.3, units=64):
    m = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units, return_sequences=True)),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def train_one_model(Xt, yt, Xv, yv, seed=42, epochs=60, sample_weights=None):
    """Sample weights ile eğitim — soft regime ensemble için."""
    set_seeds(seed)
    m = build_lstm((Xt.shape[1], Xt.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    fit_kwargs = dict(epochs=epochs, batch_size=32, validation_data=(Xv, yv),
                      callbacks=[es, rlr], verbose=0)
    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights
    m.fit(Xt, yt, **fit_kwargs)
    return m

# ============================================================
# REGIME-SOFT-ENSEMBLE TRAINING
# ============================================================
def train_regime_experts(X_tr, y_tr, X_v, y_v,
                         tr_probs, v_probs,
                         seeds=ENSEMBLE_SEEDS, epochs=EPOCHS):
    """
    3 uzman LSTM eğitir. Her uzman, kendi rejimine ait sample'larda
    daha yüksek ağırlık alır (soft sample weighting).
    
    tr_probs, v_probs: (N, 3) rejim olasılıkları [p_bull, p_bear, p_chaos]
                       Eğitim örnekleri için (zaman olarak X_tr ile aynı sırada)
    """
    experts = {}
    for regime_idx, regime_name in enumerate(['bull', 'bear', 'chaos']):
        print(f"      🎓 Uzman: {regime_name.upper()} eğitiliyor...")
        sw = tr_probs[:, regime_idx]
        # Minimum ağırlık 0.05 ki model hiçbir örneği tamamen ignore etmesin
        sw_norm = np.clip(sw, 0.05, 1.0)
        
        # Bu rejime ait yeterli ağırlıklı örnek var mı?
        effective_n = sw_norm.sum()
        if effective_n < 100:
            print(f"         ⚠️  Yetersiz veri ({effective_n:.0f}), atlama")
            experts[regime_name] = None
            continue
        
        # Ensemble (multi-seed)
        models = []
        for sd in seeds:
            m = train_one_model(X_tr, y_tr, X_v, y_v,
                               seed=sd, epochs=epochs, sample_weights=sw_norm)
            models.append(m)
        experts[regime_name] = models
        print(f"         ✓ {regime_name}: effective_n={effective_n:.0f}, {len(models)} model")
    return experts

def soft_ensemble_predict(experts, X, probs):
    """
    Soft ensemble: pred(t) = Σ p_regime(t) × mean(expert_regime.predict(X[t]))
    
    X     : (N, lookback, features)
    probs : (N, 3) — p_bull, p_bear, p_chaos
    """
    regime_preds = {}
    for regime_idx, regime_name in enumerate(['bull', 'bear', 'chaos']):
        if experts.get(regime_name) is None:
            regime_preds[regime_name] = np.zeros(len(X))
            continue
        preds_per_seed = [m.predict(X, verbose=0).flatten() for m in experts[regime_name]]
        regime_preds[regime_name] = np.mean(preds_per_seed, axis=0)
    
    # Soft weighted average
    final = (probs[:, 0] * regime_preds['bull']
           + probs[:, 1] * regime_preds['bear']
           + probs[:, 2] * regime_preds['chaos'])
    return final, regime_preds

# ============================================================
# BACKTEST
# ============================================================
def market_neutral_backtest(pred_resid, actual_resid, cost_bps=5):
    n = len(pred_resid)
    pos = np.sign(pred_resid)
    raw = pos * actual_resid
    pos_ch = np.abs(np.diff(np.concatenate([[0], pos])))
    cost = pos_ch * (2 * cost_bps / 10000.0)
    return raw - cost

# ============================================================
# HİSSE BAZLI ANALİZ — TEK FONKSİYON
# ============================================================
def analyze_ticker(ticker, config, macro_df):
    """Tek bir hisse için tüm analiz pipeline'ı. Sonuçları dict döner."""
    set_seeds()
    name = config['name']
    benchmark = config['benchmark']
    
    print(f"\n{'='*65}")
    print(f"📊 {ticker} ({name}) — Regime-Switching Analiz")
    print(f"{'='*65}")
    
    print(f"\n1. Veri indiriliyor ({ticker} + {benchmark})...")
    df_stock, df_bench = download_pair_data(ticker, benchmark)
    if df_stock is None:
        print(f"   HATA: Veri yok")
        return None
    print(f"   {len(df_stock)} ortak gün")
    
    print(f"\n2. Rejim tespit ({benchmark} üzerinden)...")
    df_regime = detect_regime_probabilities(df_bench['Bench_Close'])
    regime_counts = df_regime['regime'].value_counts()
    print(f"   Tarihsel rejim dağılımı:")
    for r in ['bull', 'bear', 'chaos']:
        n_r = regime_counts.get(r, 0)
        pct = n_r / len(df_regime) * 100 if len(df_regime) > 0 else 0
        print(f"      {r.upper():>6}: {n_r} gün ({pct:.1f}%)")
    
    current_regime = df_regime['regime'].iloc[-1]
    current_probs = df_regime[['p_bull', 'p_bear', 'p_chaos']].iloc[-1].values
    print(f"   📍 GÜNCEL REJİM: {current_regime.upper()} "
          f"(bull={current_probs[0]:.2f}, bear={current_probs[1]:.2f}, chaos={current_probs[2]:.2f})")
    
    print(f"\n3. Feature engineering...")
    df = build_features(df_stock, df_bench, macro_df, df_regime)
    if df is None or len(df) < 500:
        print(f"   HATA: Yetersiz veri")
        return None
    print(f"   {len(df)} kullanılabilir gün")
    
    # Hedef istatistikleri
    fr = df['Future_Residual_Ret']
    print(f"\n4. Hedef (residual) istatistikleri:")
    print(f"   Mean: {fr.mean():+.5f}, Std: {fr.std():.5f}")
    print(f"   Pozitif oran: %{(fr > 0).mean()*100:.1f} (~%50 = bias yok)")
    print(f"   Son beta: {df['Beta_60'].iloc[-1]:.3f}")
    
    # Feature seçimi (regime olasılıkları da feature olarak dahil)
    exclude = ['Open', 'High', 'Low', 'Volume', 'Close', 'Bench_Close', 'Adj Close',
               'Future_Stock_Ret', 'Future_Bench_Ret', 'Future_Residual_Ret']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X_data = df[feature_cols].values
    y_data = df['Future_Residual_Ret'].values
    regime_probs_arr = df[['p_bull', 'p_bear', 'p_chaos']].values
    
    train_split = int(len(df) * TRAIN_FRAC)
    val_split = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    print(f"\n5. Split: Train={train_split}, Val={val_split-train_split}, Test={len(df)-val_split}")
    
    # Scaling
    x_sc = MinMaxScaler((0, 1)); x_sc.fit(X_data[:train_split])
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
    
    # Rejim olasılıkları create_dataset_clean ile aynı sırada
    tr_probs = regime_probs_arr[LOOKBACK:train_split]
    v_probs_full = regime_probs_arr[train_split:val_split]
    te_probs = regime_probs_arr[val_split:val_split+len(yte)]
    
    print(f"\n6. Regime-Soft-Ensemble eğitimi:")
    experts = train_regime_experts(Xt, yt, Xv, yv, tr_probs, v_probs_full,
                                    seeds=ENSEMBLE_SEEDS, epochs=EPOCHS)
    
    # OOS değerlendirme
    print(f"\n7. OOS değerlendirme (soft ensemble):")
    pred_te_s, regime_preds_s = soft_ensemble_predict(experts, Xte, te_probs)
    pred_residual = y_sc.inverse_transform(pred_te_s.reshape(-1, 1)).flatten()
    actual_residual = y_sc.inverse_transform(yte.reshape(-1, 1)).flatten()
    
    test_r2 = r2_score(actual_residual, pred_residual)
    correct = (np.sign(pred_residual) == np.sign(actual_residual)).sum()
    total = len(pred_residual)
    dir_acc = correct / total * 100
    p_bin = binom_dir(int(correct), int(total))
    
    print(f"   Residual R²:   {test_r2:+.4f}")
    print(f"   Yön doğruluğu: %{dir_acc:.1f} ({correct}/{total}, p={p_bin:.4f})")
    
    # Bias kontrolü
    actual_pos_pct = (actual_residual > 0).mean() * 100
    pred_pos_pct = (pred_residual > 0).mean() * 100
    naive_best = max(actual_pos_pct, 100 - actual_pos_pct)
    alpha_edge = dir_acc - naive_best
    print(f"\n   Gerçek pozitif oran: %{actual_pos_pct:.1f}, Tahmin pozitif oran: %{pred_pos_pct:.1f}")
    print(f"   📊 Alfa edge: %{alpha_edge:+.2f} puan (vs en iyi naive)")
    
    # REJIM-BAZLI ALT-ANALİZ
    print(f"\n8. Rejim-bazlı OOS performans:")
    regime_results = {}
    te_regime_labels = np.array(['bull', 'bear', 'chaos'])[np.argmax(te_probs, axis=1)]
    for regime in ['bull', 'bear', 'chaos']:
        mask = te_regime_labels == regime
        n = mask.sum()
        if n < 10:
            print(f"      {regime.upper():>6}: yetersiz örnek ({n})")
            regime_results[regime] = None
            continue
        r_correct = (np.sign(pred_residual[mask]) == np.sign(actual_residual[mask])).sum()
        r_acc = r_correct / n * 100
        r_p = binom_dir(int(r_correct), int(n))
        r_r2 = r2_score(actual_residual[mask], pred_residual[mask]) if n > 5 else np.nan
        print(f"      {regime.upper():>6}: n={n}, yön=%{r_acc:.1f} (p={r_p:.3f}), R²={r_r2:+.3f}")
        regime_results[regime] = {
            'n': int(n), 'dir_acc': float(r_acc), 'p_bin': float(r_p), 'r2': float(r_r2),
        }
    
    # Backtest
    print(f"\n9. Market-neutral backtest:")
    alpha_rets = market_neutral_backtest(pred_residual, actual_residual, TRANSACTION_COST_BPS)
    idx_no = np.arange(0, len(alpha_rets), PREDICTION_HORIZON)
    alpha_no = alpha_rets[idx_no]
    ir_daily = information_ratio(alpha_rets, 252)
    ir_no = information_ratio(alpha_no, 252 / PREDICTION_HORIZON)
    bh_alpha = df['Future_Stock_Ret'].iloc[val_split:val_split+total].values - \
               df['Beta_60'].iloc[val_split:val_split+total].values * \
               df['Future_Bench_Ret'].iloc[val_split:val_split+total].values
    bh_ir = information_ratio(bh_alpha, 252)
    cum_alpha_daily = (np.exp(alpha_rets.cumsum()) - 1) * 100
    cum_alpha_no = (np.exp(alpha_no.cumsum()) - 1) * 100
    print(f"   IR (Daily):     {ir_daily:+.2f}")
    print(f"   IR (NonOver):   {ir_no:+.2f}")
    print(f"   B&H IR:         {bh_ir:+.2f}")
    print(f"   Cumul. alfa:    %{cum_alpha_daily[-1]:+.2f} (daily) / %{cum_alpha_no[-1]:+.2f} (no-over)")
    
    # Gelecek tahmini
    print(f"\n10. Gelecek {PREDICTION_HORIZON}g tahmini:")
    last_X = x_sc.transform(X_data)[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    last_probs = regime_probs_arr[-1:].reshape(1, 3)
    future_pred_s, _ = soft_ensemble_predict(experts, last_X, last_probs)
    pred_resid_future = float(y_sc.inverse_transform(future_pred_s.reshape(-1, 1))[0, 0])
    residual_std = float(np.std(actual_residual - pred_residual))
    cur_stock = float(df['Close'].iloc[-1])
    cur_bench = float(df['Bench_Close'].iloc[-1])
    last_beta = float(df['Beta_60'].iloc[-1])
    
    print(f"   {ticker} mevcut: ${cur_stock:.2f}, {benchmark}: ${cur_bench:.2f}, β={last_beta:.3f}")
    print(f"   Residual tahmin: {pred_resid_future*100:+.2f}% (±{residual_std*100:.2f}%)")
    print(f"   Güncel rejim:    {current_regime.upper()}")
    
    # Sinyal
    if abs(pred_resid_future) < 0.5 * residual_std:
        sig, sig_color = "NÖTR", "gray"
    elif pred_resid_future > 0:
        sig, sig_color = ("AL", "blue") if abs(pred_resid_future) < 1.5*residual_std else ("GÜÇLÜ AL", "green")
    else:
        sig, sig_color = ("ZAYIF SAT", "orange") if abs(pred_resid_future) < 1.5*residual_std else ("SAT", "red")
    
    print(f"   📌 SİNYAL: {sig}")
    
    # SONUÇLAR DICT
    return {
        'ticker': ticker, 'name': name, 'benchmark': benchmark,
        'df': df, 'df_regime': df_regime,
        'pred_residual': pred_residual, 'actual_residual': actual_residual,
        'test_r2': test_r2, 'dir_acc': dir_acc, 'p_bin': p_bin,
        'actual_pos_pct': actual_pos_pct, 'pred_pos_pct': pred_pos_pct,
        'alpha_edge': alpha_edge, 'naive_best': naive_best,
        'regime_results': regime_results,
        'ir_daily': ir_daily, 'ir_no': ir_no, 'bh_ir': bh_ir,
        'cum_alpha_daily': cum_alpha_daily, 'cum_alpha_no': cum_alpha_no,
        'alpha_rets': alpha_rets,
        'cur_stock': cur_stock, 'cur_bench': cur_bench, 'last_beta': last_beta,
        'pred_resid_future': pred_resid_future, 'residual_std': residual_std,
        'signal': sig, 'signal_color': sig_color,
        'current_regime': current_regime, 'current_probs': current_probs.tolist(),
        'val_split': val_split, 'total_samples': total,
    }

# ============================================================
# GRAFİKLER
# ============================================================
def make_charts(result):
    ticker = result['ticker']
    df = result['df']
    val_split = result['val_split']
    pred_residual = result['pred_residual']
    actual_residual = result['actual_residual']
    
    # 1. Ana grafik: fiyat + test tahminleri + rejim arka plan
    fig, ax1 = plt.subplots(figsize=(14, 7))
    show_n = min(500, len(df))
    
    # Rejim arka planı (renkli bantlar)
    df_show = df.iloc[-show_n:]
    regime_arr = df_show[['p_bull', 'p_bear', 'p_chaos']].values
    dom_regime = np.argmax(regime_arr, axis=1)
    regime_colors = {0: '#86efac', 1: '#fca5a5', 2: '#fde047'}  # bull-green, bear-red, chaos-yellow
    # Boyamak için consecutive regions bul
    last_r = dom_regime[0]
    start_idx = 0
    for i in range(1, len(dom_regime)):
        if dom_regime[i] != last_r:
            ax1.axvspan(df_show.index[start_idx], df_show.index[i],
                        alpha=0.15, color=regime_colors[last_r], zorder=0)
            start_idx = i
            last_r = dom_regime[i]
    ax1.axvspan(df_show.index[start_idx], df_show.index[-1],
                alpha=0.15, color=regime_colors[last_r], zorder=0)
    
    ax1.plot(df_show.index, df_show['Close'], label=f'{ticker} Gerçek',
             color='#1f2937', linewidth=1.8)
    
    # Test tahminleri
    n_test = len(pred_residual)
    test_close = df['Close'].iloc[val_split:val_split+n_test].values
    test_beta = df['Beta_60'].iloc[val_split:val_split+n_test].values
    test_bench_ret = df['Future_Bench_Ret'].iloc[val_split:val_split+n_test].values
    pred_prices = test_close * np.exp(pred_residual + test_beta * test_bench_ret)
    pred_dates = df.index[val_split:val_split+n_test] + pd.Timedelta(days=PREDICTION_HORIZON)
    ax1.plot(pred_dates, pred_prices, linestyle='--', color='#f59e0b',
             linewidth=1.4, alpha=0.85, label=f'Model Test (OOS, {PREDICTION_HORIZON}g)')
    
    test_start_date = df.index[val_split]
    ax1.axvline(x=test_start_date, color='red', linestyle=':', alpha=0.5,
                linewidth=1.5, label=f'Test başlangıcı')
    
    # Gelecek tahmin noktası
    fut_date = df.index[-1] + timedelta(days=PREDICTION_HORIZON)
    target_neutral = result['cur_stock'] * np.exp(result['pred_resid_future'])
    ax1.scatter([fut_date], [target_neutral], color=result['signal_color'],
                s=150, zorder=10, edgecolors='black', linewidth=1.5,
                label=f'Gelecek tahmin')
    
    ax1.set_ylabel(f'{ticker} ($)')
    ax1.set_title(f'{result["name"]} ({ticker}) — Rejim-Renkli Tarihçe + Model Test\n'
                  f'🟢 Bull   🔴 Bear   🟡 Chaos',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
    main_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 2. Kümülatif alfa
    fig, ax = plt.subplots(figsize=(12, 4.5))
    alpha_rets = result['alpha_rets']
    cum_curve = (np.exp(np.cumsum(alpha_rets)) - 1) * 100
    test_idx = df.index[val_split:val_split+len(alpha_rets)]
    ax.plot(test_idx, cum_curve, label=f'Strategy α (IR={result["ir_daily"]:.2f})',
            color='steelblue', linewidth=2)
    ax.fill_between(test_idx, cum_curve, 0, where=(cum_curve > 0), alpha=0.2, color='green')
    ax.fill_between(test_idx, cum_curve, 0, where=(cum_curve < 0), alpha=0.2, color='red')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'{ticker} Market-Neutral Kümülatif Alfa (Regime-Switching)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Kümülatif alfa (%)'); ax.grid(alpha=0.2); ax.legend()
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
    alpha_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 3. Rejim olasılıkları zaman serisi
    fig, ax = plt.subplots(figsize=(12, 4))
    df_show = df.iloc[-500:]
    ax.fill_between(df_show.index, 0, df_show['p_bull'], alpha=0.6, color='#86efac', label='Bull')
    ax.fill_between(df_show.index, df_show['p_bull'], df_show['p_bull'] + df_show['p_bear'],
                    alpha=0.6, color='#fca5a5', label='Bear')
    ax.fill_between(df_show.index, df_show['p_bull'] + df_show['p_bear'], 1.0,
                    alpha=0.6, color='#fde047', label='Chaos')
    ax.set_title(f'Rejim Olasılıkları Evrimi ({result["benchmark"]} tabanlı)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Olasılık'); ax.set_ylim(0, 1); ax.legend(loc='lower left'); ax.grid(alpha=0.2)
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
    regime_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # 4. Scatter: actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual_residual, pred_residual, alpha=0.5, edgecolors='k', linewidth=0.4)
    lo = min(actual_residual.min(), pred_residual.min())
    hi = max(actual_residual.max(), pred_residual.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='Mükemmel')
    ax.axhline(0, color='gray', alpha=0.3); ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('Gerçek residual'); ax.set_ylabel('Tahmin residual')
    ax.set_title(f'{ticker} Test Saçılım (R²={result["test_r2"]:+.3f})')
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
    scatter_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return {'main': main_b64, 'alpha': alpha_b64, 'regime': regime_b64, 'scatter': scatter_b64}

# ============================================================
# HTML RAPORLAR
# ============================================================
COMMON_CSS = """
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
h1 { text-align: center; color: #111827; border-bottom: 3px solid #0071c5; padding-bottom: 15px; }
h2 { color: #0071c5; }
.section { margin-bottom: 30px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
.header { background: #0071c5; color: white; padding: 15px 25px; font-size: 1.3em; font-weight: 600; }
.signal { padding: 20px; text-align: center; font-weight: bold; font-size: 1.3em; }
.box { padding: 15px; margin: 15px 20px; border-radius: 4px; font-size: 0.95em; }
.box-info { background: #eef2ff; border-left: 4px solid #4f46e5; }
.box-good { background: #dcfce7; border-left: 4px solid #15803d; }
.box-warn { background: #fef3c7; border-left: 4px solid #f59e0b; }
.box-bad { background: #fee2e2; border-left: 4px solid #b91c1c; }
.chart-area { padding: 20px; text-align: center; background: #f9fafb; }
.chart-area img { max-width: 100%; border-radius: 8px; border: 1px solid #eee; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; }
.stat-box { background: #fff; padding: 15px; text-align: center; }
.stat-label { font-size: 0.78em; color: #6b7280; font-weight: 700; text-transform: uppercase; }
.stat-val { font-size: 1.3em; font-weight: 800; color: #111827; margin: 5px 0; }
.stat-sub { font-size: 0.7em; color: #9ca3af; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 8px 10px; border: 1px solid #eee; text-align: center; }
th { background: #f3f4f6; }
.regime-bull { background: #dcfce7; }
.regime-bear { background: #fee2e2; }
.regime-chaos { background: #fef3c7; }
</style>
"""

def generate_ticker_html(result, charts):
    ticker = result['ticker']
    name = result['name']
    sig = result['signal']
    sig_color = result['signal_color']
    signal_bg = {'green':'#dcfce7', 'blue':'#eff6ff', 'gray':'#f3f4f6',
                 'orange':'#fed7aa', 'red':'#fee2e2'}.get(sig_color, '#eff6ff')
    
    alpha_verdict = "ALFA YOK"
    alpha_verdict_color = "#b91c1c"
    if result['alpha_edge'] > 3:
        alpha_verdict = "GERÇEK ALFA"
        alpha_verdict_color = "#15803d"
    elif result['alpha_edge'] > 0:
        alpha_verdict = "ZAYIF ALFA"
        alpha_verdict_color = "#a16207"
    
    # Rejim performans tablosu
    regime_table = "<table><tr><th>Rejim</th><th>n</th><th>Yön Doğruluğu</th><th>p-değeri</th><th>R²</th></tr>"
    for regime in ['bull', 'bear', 'chaos']:
        r = result['regime_results'].get(regime)
        if r is None:
            regime_table += f"<tr class='regime-{regime}'><td>{regime.upper()}</td><td colspan='4'>Yetersiz veri</td></tr>"
        else:
            regime_table += f"<tr class='regime-{regime}'><td>{regime.upper()}</td><td>{r['n']}</td><td>%{r['dir_acc']:.1f}</td><td>{r['p_bin']:.3f}</td><td>{r['r2']:+.3f}</td></tr>"
    regime_table += "</table>"
    
    # Senaryolar
    scenario_html = "<table><tr><th>SOXX 7g</th><th>INTC tahmin</th><th>Fiyat</th></tr>"
    for soxx_move in [-0.03, -0.01, 0.0, 0.01, 0.03]:
        intc_move = result['pred_resid_future'] + result['last_beta'] * soxx_move
        intc_price = result['cur_stock'] * np.exp(intc_move)
        bg = 'regime-bull' if intc_move > 0 else 'regime-bear'
        scenario_html += f"<tr class='{bg}'><td>{soxx_move*100:+.1f}%</td><td>{intc_move*100:+.2f}%</td><td>${intc_price:.2f}</td></tr>"
    scenario_html += "</table>"
    
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>{ticker} v15.0 — Regime-Switching</title>
{COMMON_CSS}
</head>
<body><div class="container">
<h1>{ticker} ({name}) — Regime-Switching v15.0</h1>

<div class="box box-info">
<b>🎯 Mimari:</b> 3-rejim soft ensemble (bull/bear/chaos). Her rejim için ayrı LSTM uzmanı eğitilir.
Tahmin = Σ p<sub>rejim</sub> × uzman<sub>rejim</sub>(X). Hedef: residual return (INTC - β × {result['benchmark']}).
<br><br>
<b>📍 Güncel rejim:</b> <b>{result['current_regime'].upper()}</b>
(bull={result['current_probs'][0]:.2f}, bear={result['current_probs'][1]:.2f}, chaos={result['current_probs'][2]:.2f})
</div>

<div class="section">
<div class="header">🎯 SİNYAL</div>
<div class="signal" style="background:{signal_bg}; color:{sig_color};">{sig}</div>
<div class="box box-info">
<b>Residual tahmin:</b> {result['pred_resid_future']*100:+.2f}% (±{result['residual_std']*100:.2f}%)<br>
<b>Mevcut:</b> {ticker}=${result['cur_stock']:.2f}, {result['benchmark']}=${result['cur_bench']:.2f}, β={result['last_beta']:.3f}<br>
<b>Yorum:</b> {ticker}, önümüzdeki {PREDICTION_HORIZON} günde {result['benchmark']}'a göre {result['pred_resid_future']*100:+.2f}% farklı performans gösterecek.
</div>
<h3 style="padding:0 20px;">Senaryolar</h3>
<div style="padding:0 20px 20px;">{scenario_html}</div>
</div>

<div class="section">
<div class="header">📊 BİLİMSEL DEĞERLENDİRME</div>
<div class="box" style="background:{alpha_verdict_color}1A; border-left:4px solid {alpha_verdict_color};">
<b style="color:{alpha_verdict_color}; font-size:1.1em;">{alpha_verdict}</b>: Alfa edge = %{result['alpha_edge']:+.2f} puan
</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">Residual Yön</div><div class="stat-val">%{result['dir_acc']:.1f}</div><div class="stat-sub">p={result['p_bin']:.4f}</div></div>
<div class="stat-box"><div class="stat-label">Residual R²</div><div class="stat-val">{result['test_r2']:+.4f}</div><div class="stat-sub">OOS</div></div>
<div class="stat-box"><div class="stat-label">Naive baseline</div><div class="stat-val">%{result['naive_best']:.1f}</div><div class="stat-sub">en iyi</div></div>
<div class="stat-box"><div class="stat-label">Alfa edge</div><div class="stat-val" style="color:{alpha_verdict_color}">%{result['alpha_edge']:+.2f}</div><div class="stat-sub">vs naive</div></div>
<div class="stat-box"><div class="stat-label">IR (Daily)</div><div class="stat-val">{result['ir_daily']:+.2f}</div><div class="stat-sub">Aktif</div></div>
<div class="stat-box"><div class="stat-label">IR (NonOver)</div><div class="stat-val">{result['ir_no']:+.2f}</div><div class="stat-sub">Gerçekçi</div></div>
<div class="stat-box"><div class="stat-label">Passive IR</div><div class="stat-val">{result['bh_ir']:+.2f}</div><div class="stat-sub">Buy&Hold</div></div>
<div class="stat-box"><div class="stat-label">Cum α</div><div class="stat-val">%{result['cum_alpha_daily'][-1]:+.1f}</div><div class="stat-sub">{result['total_samples']}g</div></div>
</div>
</div>

<div class="section">
<div class="header">🎭 REJİM-BAZLI PERFORMANS</div>
<div style="padding:15px 20px;">{regime_table}</div>
<div class="box box-info">
<b>Yorumlama:</b> Bir model genel olarak başarısız görünse bile, belirli bir rejimde gerçek alfa üretiyor olabilir.
Bu tablo o tespiti yapar. Yüksek doğruluk + düşük p-değeri olan rejim, modelin "uzmanlık alanı"dır.
</div>
</div>

<div class="section">
<div class="header">📈 Görsel — Rejim Renkli Fiyat + Model Test</div>
<div class="chart-area"><img src="data:image/png;base64,{charts['main']}"></div>
<div class="box box-info" style="margin-top:0;">
🟢 Bull rejimi  🔴 Bear rejimi  🟡 Chaos rejimi. Turuncu kesikli çizgi = model OOS tahminleri.
Kırmızı dikey çizgi = test setinin başlangıcı.
</div>
</div>

<div class="section">
<div class="header">💰 Kümülatif Alfa</div>
<div class="chart-area"><img src="data:image/png;base64,{charts['alpha']}"></div>
</div>

<div class="section">
<div class="header">🎭 Rejim Olasılıkları (Zaman Serisi)</div>
<div class="chart-area"><img src="data:image/png;base64,{charts['regime']}"></div>
</div>

<div class="section">
<div class="header">🎯 Test Saçılım</div>
<div class="chart-area"><img src="data:image/png;base64,{charts['scatter']}"></div>
</div>

</div></body></html>"""
    return html

def generate_comparison_html(results):
    """Tüm hisseler için karşılaştırma raporu."""
    valid = [r for r in results if r is not None]
    if not valid:
        return None
    
    rows_html = ""
    for r in valid:
        verdict_color = '#15803d' if r['alpha_edge'] > 3 else '#a16207' if r['alpha_edge'] > 0 else '#b91c1c'
        verdict = 'GERÇEK ALFA' if r['alpha_edge'] > 3 else 'ZAYIF' if r['alpha_edge'] > 0 else 'YOK'
        rows_html += f"""<tr>
<td><b>{r['ticker']}</b><br><span style='font-size:0.8em; color:#666'>{r['name']}</span></td>
<td>{r['current_regime'].upper()}</td>
<td>%{r['dir_acc']:.1f}<br><span style='font-size:0.8em; color:#666'>p={r['p_bin']:.3f}</span></td>
<td>{r['test_r2']:+.4f}</td>
<td style='color:{verdict_color};'><b>%{r['alpha_edge']:+.2f}</b><br><span style='font-size:0.8em'>{verdict}</span></td>
<td>{r['ir_no']:+.2f}</td>
<td>{r['bh_ir']:+.2f}</td>
<td>%{r['cum_alpha_daily'][-1]:+.1f}</td>
<td>{r['signal']}</td>
</tr>"""
    
    # Rejim-bazlı kıyaslama tablosu
    regime_table = "<table><tr><th>Ticker</th><th>Bull Yön</th><th>Bear Yön</th><th>Chaos Yön</th></tr>"
    for r in valid:
        row = f"<tr><td><b>{r['ticker']}</b></td>"
        for regime in ['bull', 'bear', 'chaos']:
            rr = r['regime_results'].get(regime)
            if rr is None or rr['n'] < 10:
                row += "<td style='color:#999'>—</td>"
            else:
                color = '#15803d' if rr['dir_acc'] > 55 else '#a16207' if rr['dir_acc'] > 50 else '#b91c1c'
                row += f"<td style='color:{color};'>%{rr['dir_acc']:.1f}<br><span style='font-size:0.75em'>n={rr['n']}, p={rr['p_bin']:.3f}</span></td>"
        row += "</tr>"
        regime_table += row
    regime_table += "</table>"
    
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>Multi-Stock Karşılaştırma — v15.0</title>
{COMMON_CSS}
</head>
<body><div class="container">
<h1>Multi-Stock Regime-Switching Karşılaştırma — v15.0</h1>

<div class="box box-info">
<b>Bu rapor</b> birden fazla hisseyi aynı framework ile analiz eder. Her hisse için ayrı bir
regime-switching market-neutral model eğitilir. Aşağıdaki tablo {len(valid)} hissenin OOS performansını
yan yana gösterir.
</div>

<div class="section">
<div class="header">📊 Genel Karşılaştırma</div>
<div style="padding:15px 20px;">
<table>
<tr style="background:#f3f4f6;">
<th>Ticker</th><th>Güncel Rejim</th><th>Yön Doğr.</th><th>R²</th>
<th>Alfa Edge</th><th>IR (NoOver)</th><th>Passive IR</th><th>Cum α</th><th>Sinyal</th>
</tr>
{rows_html}
</table>
</div>
</div>

<div class="section">
<div class="header">🎭 Rejim-Bazlı Performans Karşılaştırması</div>
<div style="padding:15px 20px;">{regime_table}</div>
<div class="box box-info">
<b>Yorumlama:</b> Bir hisse için model genel olarak başarısız görünse bile, belirli bir rejimde
(örn. BEAR'da) gerçek alfa üretiyor olabilir. Bu tablo "hangi hisse hangi rejimde tahmin edilebilir?"
sorusuna cevap verir. Renk kodları: 🟢 %55+ (anlamlı), 🟠 %50-55 (marjinal), 🔴 %50- (rastgele).
</div>
</div>

<div class="section">
<div class="header">📁 Detaylı Raporlar</div>
<div class="box box-info">
Her hisse için ayrıntılı analiz aşağıdaki dosyalardadır:
<ul>
{''.join(f'<li><a href="{r["ticker"]}_v15.0.html">{r["ticker"]} ({r["name"]})</a></li>' for r in valid)}
</ul>
</div>
</div>

</div></body></html>"""
    return html

# ============================================================
# ANA AKIŞ — ÇOKLU HİSSE
# ============================================================
def main():
    set_seeds()
    init_db()
    
    print("="*65)
    print("v15.0 — MULTI-STOCK REGIME-SWITCHING FRAMEWORK")
    print("="*65)
    print(f"\nAnaliz edilecek hisseler: {list(TICKERS_CONFIG.keys())}")
    print(f"Tahmin ufku: {PREDICTION_HORIZON} gün")
    print(f"İşlem maliyeti: {TRANSACTION_COST_BPS} bps")
    print(f"Ensemble: {len(ENSEMBLE_SEEDS)} model × 3 rejim = "
          f"{len(ENSEMBLE_SEEDS)*3} LSTM eğitimi/hisse")
    print(f"Toplam LSTM eğitimi: {len(TICKERS_CONFIG) * len(ENSEMBLE_SEEDS) * 3}")
    
    print("\nMakro veri indiriliyor...")
    macro_df = get_macro_data()
    
    results = []
    for ticker, config in TICKERS_CONFIG.items():
        result = analyze_ticker(ticker, config, macro_df)
        if result is not None:
            print(f"\n11. {ticker} HTML raporu oluşturuluyor...")
            charts = make_charts(result)
            html = generate_ticker_html(result, charts)
            fname = f"{ticker}_Analiz_v15.0.html"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"    ✓ {fname}")
        results.append(result)
    
    # Karşılaştırma raporu
    if len([r for r in results if r is not None]) >= 1:
        print(f"\n📊 Karşılaştırma raporu oluşturuluyor...")
        comp_html = generate_comparison_html(results)
        if comp_html:
            with open('Karsilastirma_v15.0.html', 'w', encoding='utf-8') as f:
                f.write(comp_html)
            print(f"    ✓ Karsilastirma_v15.0.html")
    
    # Konsol özeti
    print("\n" + "="*65)
    print("MULTI-STOCK ÖZET TABLO:")
    print("="*65)
    print(f"{'TICKER':<8} {'REJİM':<8} {'YÖN':<8} {'R²':<10} {'ALFA':<8} {'IR':<7} {'SİNYAL':<12}")
    print("-"*65)
    for r in results:
        if r is None: continue
        print(f"{r['ticker']:<8} {r['current_regime'].upper():<8} "
              f"%{r['dir_acc']:<6.1f} {r['test_r2']:<+10.4f} "
              f"%{r['alpha_edge']:<+6.2f} {r['ir_no']:<+7.2f} {r['signal']:<12}")
    print("="*65)
    print(f"\n✓ Tüm raporlar oluşturuldu.")
    print(f"  Ana rapor: Karsilastirma_v15.0.html")

if __name__ == "__main__":
    main()
