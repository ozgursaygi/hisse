# ============================================================
# MULTI-STOCK v16.5 — REGIME-SWITCHING & EWT FORECASTING
# ============================================================
import sys, os, sqlite3, warnings, random, base64, json
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Sunucu ortamı uyumluluğu için
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
    'INTC': {'name': 'Intel Corporation', 'benchmark': 'SOXX'},
    'AMD':  {'name': 'Advanced Micro Devices', 'benchmark': 'SOXX'},
    'NVDA': {'name': 'NVIDIA Corporation', 'benchmark': 'SOXX'}
}

PREDICTION_HORIZON = 7
LOOKBACK = 60
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10 
ENSEMBLE_SEEDS = [42, 52, 62]  
EPOCHS = 60

# --- DB AYARLARI ---
DB_FOLDER = r"C:\Projects\ML"
DB_PATH = os.path.join(DB_FOLDER, "market_intelligence_v16.db")

def init_db():
    if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals (
        tarih TEXT, sembol TEXT, rejim TEXT, signal TEXT, price REAL, 
        prediction_ret REAL, UNIQUE(tarih, sembol))''')
    conn.commit(); conn.close()

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf_random.set_seed(seed)

# ============================================================
# 1. VERİ İNDİRME VE HAZIRLIK
# ============================================================
def download_pair_data(ticker, benchmark):
    end = datetime.now()
    start = end - timedelta(days=12*365) # 12 yıllık geniş veri seti
    try:
        s = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        b = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
        if s.empty or b.empty: return None, None
        
        if hasattr(s.columns, 'get_level_values'): s.columns = s.columns.get_level_values(0)
        if hasattr(b.columns, 'get_level_values'): b.columns = b.columns.get_level_values(0)
            
        s = s[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        b = b[['Close']].rename(columns={'Close': 'Bench_Close'}).dropna()
        common = s.index.intersection(b.index)
        return s.loc[common], b.loc[common]
    except: return None, None

def get_macro_data():
    tickers = {"^VIX": "VIX", "^TNX": "US_10Y_BOND", "DX-Y.NYB": "DXY"}
    try:
        df = yf.download(list(tickers.keys()), period="12y", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.rename(columns=tickers).ffill()
    except: return pd.DataFrame()

# ============================================================
# 2. EWT VE REJİM TESPİT MOTORU
# ============================================================
def detect_regimes(bench_close):
    """Benchmark (SOXX) üzerinden piyasa ritmini ve rejimini tespit eder."""
    log_ret = np.log(bench_close / bench_close.shift(1))
    sma50 = bench_close.rolling(50).mean()
    sma200 = bench_close.rolling(200).mean()
    
    trend_strength = 1.0 / (1.0 + np.exp(-((sma50 / sma200) - 1.0) * 50))  
    vol = log_ret.rolling(20).std()
    p_chaos = (1.0 / (1.0 + np.exp(-((vol / vol.expanding().quantile(0.7)) - 1.0) * 5))).clip(0.05, 0.95)  
    
    p_bull = (1.0 - p_chaos) * trend_strength
    p_bear = (1.0 - p_chaos) * (1.0 - trend_strength)
    
    df = pd.DataFrame({'p_bull': p_bull, 'p_bear': p_bear, 'p_chaos': p_chaos}, index=bench_close.index)
    df['regime'] = np.array(['BULL', 'BEAR', 'CHAOS'])[np.argmax(df.values, axis=1)]
    return df

def build_features(df_s, df_b, macro_df, df_r):
    """Elliott Dalga Teorisi ritimlerini (Fibonacci) ve teknik göstergeleri birleştirir."""
    df = df_s.copy()
    df['Bench_Close'] = df_b['Bench_Close']
    try:
        import ta
        # Klasik Göstergeler
        df['RSI'] = ta.momentum.rsi(df['Close'], 14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # EWT Ritim Analizi: Fibonacci Seviyelerine Uzaklık
        # Son 120 günün (bir major dalga boyu) Fibonacci geri çekilme seviyeleri
        wave_h = df['High'].rolling(120).max()
        wave_l = df['Low'].rolling(120).min()
        wave_range = wave_h - wave_l
        
        df['Fib_382'] = (df['Close'] - (wave_h - wave_range * 0.382)) / df['Close']
        df['Fib_618'] = (df['Close'] - (wave_h - wave_range * 0.618)) / df['Close']
        
        # Piyasa Duyarlılığı
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Bench_Log_Ret'] = np.log(df['Bench_Close'] / df['Bench_Close'].shift(1))
        cov = df['Log_Ret'].rolling(60).cov(df['Bench_Log_Ret'])
        var = df['Bench_Log_Ret'].rolling(60).var()
        df['Beta_60'] = (cov / var).clip(0.0, 3.0)
        
        # Hedef: Gelecek 7 Günlük 'Artık' (Residual) Getiri
        f_stock = np.log(df['Close'].shift(-PREDICTION_HORIZON) / df['Close'])
        f_bench = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON) / df['Bench_Close'])
        df['Future_Residual_Ret'] = f_stock - df['Beta_60'] * f_bench
        
        df = df.join(df_r[['p_bull', 'p_bear', 'p_chaos']], how='left').join(macro_df, how='left').dropna()
        return df
    except: return None

# ============================================================
# 3. MODELLEME VE EĞİTİM (SOFT ENSEMBLE)
# ============================================================
def build_lstm(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def create_dataset(X, y, lb):
    return np.array([X[i-lb:i] for i in range(lb, len(X))]), np.array([y[i] for i in range(lb, len(y))])

def train_experts(Xt, yt, Xv, yv, probs, split_idx):
    experts = {}
    for idx, reg in enumerate(['bull', 'bear', 'chaos']):
        sw = np.clip(probs[LOOKBACK:split_idx, idx], 0.05, 1.0)
        if sw.sum() < 100: experts[reg] = None; continue
        
        models = []
        for seed in ENSEMBLE_SEEDS:
            set_seeds(seed)
            m = build_lstm((Xt.shape[1], Xt.shape[2]))
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            m.fit(Xt, yt, epochs=EPOCHS, batch_size=32, validation_data=(Xv, yv), 
                  sample_weight=sw, callbacks=[es], verbose=0)
            models.append(m)
        experts[reg] = models
    return experts

def ensemble_predict(experts, X, p):
    res = {}
    for i, r in enumerate(['bull', 'bear', 'chaos']):
        if experts.get(r): res[r] = np.mean([m.predict(X, verbose=0).flatten() for m in experts[r]], axis=0)
        else: res[r] = np.zeros(len(X))
    return p[:, 0]*res['bull'] + p[:, 1]*res['bear'] + p[:, 2]*res['chaos']

# ============================================================
# 4. GÖRSELLEŞTİRME VE RAPORLAMA
# ============================================================
def make_professional_charts(ticker, df, val_split, test_actual, test_pred, future_prices):
    """Geçmiş, Test ve Gelecek Tahminini içeren bilimsel grafik."""
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    dates = df.index
    # Son 120 günün geçmişi
    plt.plot(dates[val_split-120:val_split], df['Close'].iloc[val_split-120:val_split], label='Geçmiş (Dalga)', color='#94a3b8')
    # Test verisi
    plt.plot(dates[val_split:], test_actual, label='Test (Gerçek)', color='#22d3ee', alpha=0.8)
    plt.plot(dates[val_split:], test_pred, label='Test (Model)', color='#f472b6', linestyle='--')
    
    # Gelecek Tahmini
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]
    plt.plot([dates[-1]] + future_dates, [df['Close'].iloc[-1]] + future_prices, 
             label='7G YZ Tahmini', color='#4ade80', marker='o', linewidth=2)
    
    plt.title(f"{ticker} - EWT Ritim & LSTM Tahmin Modeli")
    plt.legend(); plt.grid(alpha=0.2)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120); plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================
# 5. ANA ANALİZ AKIŞI
# ============================================================
def analyze_ticker(ticker, config, macro_df):
    print(f"\n" + "="*65)
    print(f"📊 {ticker} ({config['name']}) Analizi Başladı")
    print("="*65)
    
    df_s, df_b = download_pair_data(ticker, config['benchmark'])
    if df_s is None: return None
    
    df_r = detect_regimes(df_b['Bench_Close'])
    df = build_features(df_s, df_b, macro_df, df_r)
    if df is None: return None
    
    f_cols = [c for c in df.columns if c not in ['Open','High','Low','Volume','Close','Bench_Close','Future_Residual_Ret']]
    X_raw, y_raw = df[f_cols].values, df['Future_Residual_Ret'].values
    probs = df[['p_bull', 'p_bear', 'p_chaos']].values
    
    train_idx = int(len(df) * TRAIN_FRAC)
    val_idx = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    
    x_sc, y_sc = MinMaxScaler(), MinMaxScaler((-1, 1))
    X_s = x_sc.fit_transform(X_raw)
    y_s = y_sc.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    Xt, yt = create_dataset(X_s[:train_idx], y_s[:train_idx], LOOKBACK)
    Xv, yv = create_dataset(X_s[train_idx-LOOKBACK:val_idx], y_s[train_idx-LOOKBACK:val_idx], LOOKBACK)
    Xte, yte = create_dataset(X_s[val_idx-LOOKBACK:], y_s[val_idx-LOOKBACK:], LOOKBACK)
    
    print("   [Model] Uzmanlar eğitiliyor (EWT Ritimleri İşleniyor)...")
    experts = train_experts(Xt, yt, Xv, yv, probs, train_idx)
    
    # Test Tahmini
    pred_te_s = ensemble_predict(experts, Xte, probs[val_idx:val_idx+len(yte)])
    act_res = y_sc.inverse_transform(yte.reshape(-1, 1)).flatten()
    prd_res = y_sc.inverse_transform(pred_te_s.reshape(-1, 1)).flatten()
    
    # Gelecek 7G Tahmini
    last_seq = X_s[-LOOKBACK:].reshape(1, LOOKBACK, len(f_cols))
    last_probs = probs[-1:].reshape(1, 3)
    future_s = ensemble_predict(experts, last_seq, last_probs)
    future_ret = float(y_sc.inverse_transform(future_s.reshape(-1, 1))[0,0])
    
    # Sonuçların Fiyata Dönüşümü
    test_actual_prices = df['Close'].iloc[val_idx:].values * np.exp(act_res)
    test_pred_prices = df['Close'].iloc[val_idx:].values * np.exp(prd_res)
    target_price = df['Close'].iloc[-1] * np.exp(future_ret)
    
    dir_acc = (np.sign(prd_res) == np.sign(act_res)).mean() * 100
    std_res = float(np.std(act_res - prd_res))
    
    if future_ret > 0.5 * std_res: sig, yon, col = "AL 🚀", "YÜKSELİŞ 📈", "#22c55e"
    elif future_ret < -0.5 * std_res: sig, yon, col = "SAT 💥", "DÜŞÜŞ 📉", "#ef4444"
    else: sig, yon, col = "NÖTR 🟡", "YATAY ➡️", "#f59e0b"
    
    print(f"\n   🎯 [SONUÇLAR]")
    print(f"    Yön Doğruluğu: %{dir_acc:.1f} | Beklenen Yön: {yon}")
    print(f"    Hedef Fiyat: ${target_price:.2f} | Sinyal: {sig}")
    
    chart_b64 = make_professional_charts(ticker, df, val_idx, test_actual_prices, test_pred_prices, 
                                        np.linspace(df['Close'].iloc[-1], target_price, 7).tolist())
    
    return {
        'ticker': ticker, 'name': config['name'], 'price': df['Close'].iloc[-1],
        'regime': df_r['regime'].iloc[-1], 'acc': dir_acc, 'yon': yon, 'sig': sig, 
        'target': target_price, 'chart': chart_b64, 'color': col,
        'p_bull': probs[-1,0], 'p_bear': probs[-1,1], 'p_chaos': probs[-1,2]
    }

# ============================================================
# 6. MAIN VE RAPOR ÜRETİMİ
# ============================================================
def generate_final_html(results):
    html = f"""
    <!DOCTYPE html>
    <html lang="tr"><head><meta charset="UTF-8"><title>EWT & LSTM Multi-Stock Analiz</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        .card {{ background: #1e293b; border-radius: 16px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 15px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }}
        .stat {{ background: #0f172a; padding: 15px; border-radius: 8px; text-align: center; }}
        img {{ width: 100%; border-radius: 12px; margin-top: 20px; border: 1px solid #334155; }}
    </style></head><body>
    <h1 style='text-align:center;'>🚀 YZ Destekli EWT Ritim Analiz Raporu</h1>
    """
    for r in results:
        html += f"""
        <div class='card'>
            <div class='header'>
                <h2 style='color:#38bdf8;'>{r['ticker']} - {r['name']}</h2>
                <span style='background:{r['color']}; padding:8px 16px; border-radius:20px; font-weight:bold;'>{r['sig']}</span>
            </div>
            <div class='grid'>
                <div class='stat'><div>Güncel Fiyat</div><div style='font-size:1.4em; font-weight:bold;'>${r['price']:.2f}</div></div>
                <div class='stat'><div>Hedef Fiyat (7G)</div><div style='font-size:1.4em; color:#4ade80;'>${r['target']:.2f}</div></div>
                <div class='stat'><div>Beklenen Yön</div><div style='font-weight:bold;'>{r['yon']}</div></div>
                <div class='stat'><div>Piyasa Rejimi</div><div style='font-weight:bold;'>{r['regime']}</div></div>
            </div>
            <img src='data:image/png;base64,{r['chart']}'>
        </div>
        """
    html += "</body></html>"
    with open("EWT_MultiStock_Analiz.html", "w", encoding="utf-8") as f: f.write(html)
    print(f"\n[BİLGİ] Nihai HTML Raporu oluşturuldu: EWT_MultiStock_Analiz.html")

if __name__ == "__main__":
    init_db(); set_seeds()
    macro = get_macro_data()
    final_res = []
    for t, cfg in TICKERS_CONFIG.items():
        res = analyze_ticker(t, cfg, macro)
        if res: final_res.append(res)
    if final_res: generate_final_html(final_res)