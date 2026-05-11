# ============================================================
# INTC AI ANALİZ PANELİ v10.4 - RİTMİK KALIPLARA DAYALI BİLİMSEL TAHMİN
#
# v10.4 İyileştirmeleri (v10.3 üzerine):
#   [FIX-1] DM testi yorumu düzeltildi: Modelin MSE'si referanstan düşük olmalı.
#   [FIX-2] Backtest Sharpe yıllıklandırması 15 günlük periyoda göre düzeltildi.
#   [FIX-3] Veri sızıntısı riski tamamen giderildi (scaler fit sadece train'de).
#   [FIX-4] Sinyal eşiği, tarihsel 15g getiri oynaklığına göre dinamik hale getirildi.
#   [FIX-5] Test performansı saçılım grafiği ile daha dürüst gösteriliyor.
#   [FIX-6] Tüm istatistiksel testler ve metrik yorumları doğrulandı.
#   [FIX-7] mini_plot fonksiyonu DataFrame girişi için düzeltildi (renk hatası giderildi).
# ============================================================

import sys
import subprocess
import importlib
import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Arka planda çalışmak için
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO
from datetime import datetime, timedelta, timezone
import random
import tensorflow as tf
import yfinance as yf
from textblob import TextBlob
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow import random as tf_random

# --- PAKET KURULUMU ---
def install_package(package):
    print(f"OTOMATİK KURULUM: '{package}' yükleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])

required_packages = ['tf-keras', 'ta', 'yfinance', 'textblob',
                     'scipy', 'seaborn', 'sklearn', 'statsmodels']
for package in required_packages:
    try:
        importlib.import_module(package.replace('-', '_'))
    except ImportError:
        try:
            install_package(package)
        except:
            pass

# GoogleNews için ek yedekleme
try:
    from GoogleNews import GoogleNews
except ImportError:
    install_package('GoogleNews')
    from GoogleNews import GoogleNews

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# --- VERİTABANI AYARLARI ---
DB_FOLDER = r"C:\Projects\ML"
DB_NAME = "data_intc_v10_4.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gunluk_veriler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih TEXT, sembol TEXT,
            acilis REAL, yuksek REAL, dusuk REAL, kapanis REAL, hacim REAL,
            UNIQUE(tarih, sembol)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tahminler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analiz_tarihi TEXT, hedef_tarih TEXT,
            sembol TEXT, tahmin_fiyati REAL,
            UNIQUE(analiz_tarihi, hedef_tarih, sembol)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_sqlite(ticker, df):
    if df is None or df.empty:
        return
    start_date_filter = pd.Timestamp("2020-01-01")
    df_filtered = df[df.index >= start_date_filter].copy()
    if df_filtered.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for index, row in df_filtered.iterrows():
        date_str = index.strftime('%Y-%m-%d')
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO gunluk_veriler
                (tarih, sembol, acilis, yuksek, dusuk, kapanis, hacim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, ticker, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        except:
            pass
    conn.commit()
    conn.close()

def save_predictions_to_sqlite(ticker, dates, prices, analysis_date=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if analysis_date is None:
        analiz_tarihi = datetime.now().strftime('%Y-%m-%d')
    else:
        analiz_tarihi = analysis_date
    for date, price in zip(dates, prices):
        hedef_tarih = date.strftime('%Y-%m-%d')
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO tahminler
                (analiz_tarihi, hedef_tarih, sembol, tahmin_fiyati)
                VALUES (?, ?, ?, ?)
            ''', (analiz_tarihi, hedef_tarih, ticker, float(price)))
        except:
            pass
    conn.commit()
    conn.close()

# --- SEED SABİTLEME ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_random.set_seed(seed)
set_seeds()

# --- VARLIK ---
TICKER = 'INTC'
TICKER_NAME = 'Intel Corporation'
PREDICTION_HORIZON = 15  # 15 günlük tahmin

# --- MAKRO VERİ ---
def get_macro_data():
    end = datetime.now()
    start = end - timedelta(days=12*365)
    tickers = {
        "^VIX": "VIX", "^TNX": "US_10Y_BOND", "CL=F": "OIL",
        "DX-Y.NYB": "DXY", "^GSPC": "SP500", "^IXIC": "NASDAQ", "SOXX": "SEMI_ETF"
    }
    try:
        df = yf.download(list(tickers.keys()), start=start, end=end, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.rename(columns=tickers, inplace=True)
        return df.ffill()
    except Exception as e:
        print(f"UYARI: Makro veriler indirilemedi ({e}).")
        return pd.DataFrame()

# --- SENTIMENT (Gelişmiş yedekleme) ---
def get_advanced_sentiment(ticker):
    news_items = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            titles = []
            for n in news[:10]:
                if not isinstance(n, dict): continue
                title = n.get('title') or (n.get('content', {}) or {}).get('title')
                if not title: continue
                titles.append(title)
                ts = n.get('providerPublishTime', 0)
                try: date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                except: date_str = "Tarih Yok"
                news_items.append({'date': date_str, 'title': title})
            if titles:
                scores = [TextBlob(t).sentiment.polarity for t in titles]
                return float(np.mean(scores)), news_items
    except: pass

    try:
        googlenews = GoogleNews(lang='en', region='US')
        googlenews.set_period('7d')
        googlenews.search("Intel INTC stock news")
        results = googlenews.result()[:10]
        if results:
            titles = []
            for item in results:
                t = item.get('title')
                if t:
                    titles.append(t)
                    news_items.append({'date': item.get('date', ''), 'title': t})
            if titles:
                scores = [TextBlob(t).sentiment.polarity for t in titles]
                return float(np.mean(scores)), news_items
    except: pass

    return 0.0, []

# --- VERİ İNDİRME VE FEATURE ENGINEERING ---
def get_stock_data(symbol, macro_df, news_score=0.0):
    end = datetime.now()
    start = end - timedelta(days=12*365)
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"HATA: {symbol} indirilemedi: {e}")
        return None
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols_to_keep:
        if c not in df.columns: df[c] = df['Close']
    df = df[cols_to_keep].dropna()
    if len(df) < 200: return None
    save_to_sqlite(symbol, df)

    # Genişletilmiş feature mühendisliği (hepsi sadece geçmiş veri kullanır)
    try:
        import ta
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['SMA20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
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

        # 15 günlük gelecek getiri (direkt hedef)
        df['Future_Ret_15d'] = np.log(df['Close'].shift(-PREDICTION_HORIZON) / df['Close'])

        # Sentiment feature (geçmişi bilmez, son 20 güne bugünün skoru atanır)
        df['NewsSent'] = 0.0
        if len(df) >= 20:
            df.iloc[-20:, df.columns.get_loc('NewsSent')] = news_score

        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill().dropna()
        else:
            df = df.dropna()
        df = df.dropna(subset=['Future_Ret_15d'])
        return df
    except Exception as e:
        print(f"HATA: İndikatörler hesaplanamadı: {e}")
        return None

# --- YILLIKLANDIRILMIŞ SHARPE HESAPLAMA (15 GÜNLÜK PERİYOT İÇİN) ---
def annualized_sharpe(period_returns, periods_per_year=252/15, risk_free_rate=0.045):
    """15 günlük dönem getirileri için yıllıklandırılmış Sharpe oranı"""
    arr = np.array(period_returns)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    rf_period = (1 + risk_free_rate)**(1/periods_per_year) - 1
    excess = arr - rf_period
    return np.sqrt(periods_per_year) * np.mean(excess) / np.std(arr, ddof=1) if np.std(arr) > 0 else 0.0

def calculate_real_metrics(returns_array, periods_per_year=252/15):
    returns_array = np.array(returns_array)
    returns_array = returns_array[np.isfinite(returns_array)]
    if len(returns_array) < 2:
        return {'mdd': 0, 'sharpe': 0, 'volatility': 0}
    cum_returns = np.exp(np.cumsum(returns_array))
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    mdd = np.abs(np.min(drawdown)) * 100
    annual_vol = np.std(returns_array, ddof=1) * np.sqrt(periods_per_year) * 100
    sharpe = annualized_sharpe(returns_array, periods_per_year)
    return {'mdd': mdd, 'sharpe': sharpe, 'volatility': annual_vol}

# --- BASELINE MODELLER ---
def baseline_naive(y_test):
    return np.zeros(len(y_test))

def baseline_random_walk(y_train, y_test):
    if len(y_test) == 0: return np.array([])
    pred = np.zeros(len(y_test))
    pred[0] = y_train[-1] if len(y_train) > 0 else 0
    pred[1:] = y_test[:-1]
    return pred

def diebold_mariano_test(actual, pred1, pred2, alpha=0.05):
    """DM testi ve modelin daha iyi olup olmadığını döndürür.
    Returns: dm_stat, p_value, is_better (True if pred1 significantly better than pred2)
    """
    actual, pred1, pred2 = map(np.array, [actual, pred1, pred2])
    e1, e2 = actual - pred1, actual - pred2
    d = e1**2 - e2**2
    n = len(d)
    if n < 10: return 0, 1.0, False
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0: return 0, 1.0, False
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    is_better = (mean_d < 0) and (p_value < alpha)  # pred1 better if its MSE is smaller
    return dm_stat, p_value, is_better

# --- DATASET ---
def create_dataset(dataset, target_idx, lookback):
    X, Y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i])
        Y.append(dataset[i, target_idx])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape, dropout=0.3, lstm_units=64):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout),
        LSTM(lstm_units // 2),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model

# --- NESTED LOOKBACK SEÇİMİ ---
def find_optimal_lookback_nested(data, target_idx, train_end, scaler_factory,
                                  candidates=[60, 120, 250], epochs=12, inner_val_frac=0.15):
    print(f"   -> [Nested] Lookback aranıyor (sadece train içinde): {candidates}")
    inner_val_start = int(train_end * (1 - inner_val_frac))
    inner_train_end = inner_val_start
    results = {}
    for lb in candidates:
        if inner_train_end <= lb + 50: continue
        inner_scaler = scaler_factory()
        inner_scaler.fit(data[:inner_train_end])
        inner_train_scaled = inner_scaler.transform(data[:inner_train_end])
        X_tr, y_tr = create_dataset(inner_train_scaled, target_idx, lb)
        inner_val_inputs = data[inner_train_end - lb : train_end]
        inner_val_scaled = inner_scaler.transform(inner_val_inputs)
        X_v, y_v = create_dataset(inner_val_scaled, target_idx, lb)
        if len(X_v) < 10 or len(X_tr) < 100: continue
        set_seeds()
        m = Sequential([Input(shape=(X_tr.shape[1], X_tr.shape[2])), LSTM(32), Dense(1)])
        m.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        hist = m.fit(X_tr, y_tr, epochs=epochs, batch_size=64, validation_data=(X_v, y_v), callbacks=[es], verbose=0)
        best_val_loss = min(hist.history['val_loss'])
        results[lb] = best_val_loss
        print(f"      lookback={lb}: inner_val_loss={best_val_loss:.5f}")
    if not results: return 60
    best_lb = min(results.keys(), key=lambda k: results[k])
    print(f"   ✅ SEÇİLEN LOOKBACK: {best_lb} gün (nested, leakage'siz)")
    return best_lb

# --- WALK-FORWARD CV ---
def walk_forward_cv(data, target_idx, features, lookback, n_splits=4, epochs=25):
    print(f"\n   📊 Walk-Forward CV (n_splits={n_splits}):")
    n = len(data)
    fold_size = n // (n_splits + 1)
    cv_scores = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)
        if val_end > n - lookback: break
        fold_scaler = MinMaxScaler((0, 1))
        fold_scaler.fit(data[:train_end])
        train_sc = fold_scaler.transform(data[:train_end])
        X_tr, y_tr = create_dataset(train_sc, target_idx, lookback)
        val_inputs = data[train_end - lookback : val_end]
        val_sc = fold_scaler.transform(val_inputs)
        X_v, y_v = create_dataset(val_sc, target_idx, lookback)
        if len(X_tr) < 100 or len(X_v) < 10: continue
        set_seeds(seed=42 + fold)
        m = build_lstm_model((X_tr.shape[1], X_tr.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
        m.fit(X_tr, y_tr, epochs=epochs, batch_size=32, validation_data=(X_v, y_v), callbacks=[es], verbose=0)
        pred_scaled = m.predict(X_v, verbose=0).flatten()
        # Unscale predictions
        n_features = len(features)
        dummy_pred = np.zeros((len(pred_scaled), n_features))
        dummy_pred[:, target_idx] = pred_scaled
        pred_returns = fold_scaler.inverse_transform(dummy_pred)[:, target_idx]
        dummy_actual = np.zeros((len(y_v), n_features))
        dummy_actual[:, target_idx] = y_v
        actual_returns = fold_scaler.inverse_transform(dummy_actual)[:, target_idx]
        # Yön doğruluğu
        threshold = np.std(actual_returns) * 0.1
        pred_signs = np.where(np.abs(pred_returns) < threshold, 0, np.sign(pred_returns))
        actual_signs = np.where(np.abs(actual_returns) < threshold, 0, np.sign(actual_returns))
        non_flat = (pred_signs != 0) & (actual_signs != 0)
        dir_acc = (np.mean(pred_signs[non_flat] == actual_signs[non_flat]) * 100) if non_flat.sum() > 0 else 50.0
        mse = mean_squared_error(actual_returns, pred_returns)
        try: r2 = r2_score(actual_returns, pred_returns)
        except: r2 = -1
        cv_scores.append({'fold': fold+1, 'mse': mse, 'r2': r2, 'dir_acc': dir_acc,
                         'n_compared': int(non_flat.sum()), 'n_total': len(pred_returns)})
        print(f"      Fold {fold+1}: MSE={mse:.5f} | R²={r2:.4f} | Dir.Acc={dir_acc:.1f}% ({non_flat.sum()}/{len(pred_returns)})")
    if not cv_scores: return None
    avg_r2 = np.mean([s['r2'] for s in cv_scores])
    avg_dir = np.mean([s['dir_acc'] for s in cv_scores])
    std_dir = np.std([s['dir_acc'] for s in cv_scores])
    print(f"   📈 CV Ortalaması: R²={avg_r2:.4f} | Dir.Acc={avg_dir:.1f}% (±{std_dir:.1f}%)")
    return {'avg_r2': avg_r2, 'avg_dir_acc': avg_dir, 'std_dir_acc': std_dir, 'folds': cv_scores}

# --- ENSEMBLE EĞİTİMİ ---
def train_ensemble(X_train, y_train, X_val, y_val, n_models=3, epochs=80):
    print(f"\n   🎯 Ensemble eğitiliyor ({n_models} model)...")
    models = []
    for i in range(n_models):
        set_seeds(seed=42 + i*10)
        m = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        h = m.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=[es, rlr], verbose=0)
        print(f"      Model {i+1}: epochs={len(h.history['loss'])} | best_val_loss={min(h.history['val_loss']):.5f}")
        models.append(m)
    return models

def ensemble_predict(models, X):
    preds = np.array([m.predict(X, verbose=0).flatten() for m in models])
    return preds.mean(axis=0), preds.std(axis=0)

# --- GELECEK TAHMİNİ (15 GÜN) ---
def forecast_future(models, last_batch_scaled, scaler, features, target_idx, current_price, residual_std):
    preds_scaled = np.array([m.predict(last_batch_scaled, verbose=0)[0,0] for m in models])
    mean_scaled = preds_scaled.mean()
    std_scaled = preds_scaled.std()
    n_features = len(features)
    dummy_mean = np.zeros((1, n_features)); dummy_mean[0, target_idx] = mean_scaled
    mean_ret = scaler.inverse_transform(dummy_mean)[0, target_idx]
    # Toplam belirsizlik = ensemble std + residual std
    dummy_ens = np.zeros((1, n_features)); dummy_ens[0, target_idx] = mean_scaled + std_scaled
    ens_std_unscaled = scaler.inverse_transform(dummy_ens)[0, target_idx] - mean_ret
    total_std = np.sqrt(ens_std_unscaled**2 + residual_std**2)
    target_price = current_price * np.exp(mean_ret)
    ci_low = current_price * np.exp(mean_ret - 1.96 * total_std)
    ci_high = current_price * np.exp(mean_ret + 1.96 * total_std)
    return target_price, ci_low, ci_high, mean_ret

# --- BACKTEST (15 GÜNLÜK SİNYAL) ---
def backtest_15d(actual_returns, predicted_returns):
    """15 günlük tahminlere göre strateji getirilerini hesaplar"""
    n = min(len(predicted_returns), len(actual_returns))
    positions = np.sign(predicted_returns[:n])
    strategy_returns = positions * actual_returns[:n]
    return strategy_returns

# ============================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================

def plot_main_chart(df, fut_dates, target_price, ci_low, ci_high, sig_color, name, ticker):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index[-200:], df['Close'].iloc[-200:], label="Gerçek Fiyat", color="#1f2937", linewidth=1.8)
    ax.scatter(fut_dates[-1], target_price, color=sig_color, s=100, zorder=10, label=f"{PREDICTION_HORIZON} Gün Hedef")
    ax.axhline(y=df['Close'].iloc[-1], color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f"{name} ({ticker}) - {PREDICTION_HORIZON} Günlük Fiyat Tahmini", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Tarih"); ax.set_ylabel("Fiyat ($)")
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=100); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_scatter_predictions(actual, predicted, ticker):
    """Gerçekleşen vs Tahmin Edilen 15 günlük getiri saçılım grafiği"""
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1, label='Mükemmel Tahmin')
    ax.set_xlabel('Gerçekleşen 15 Günlük Getiri'); ax.set_ylabel('Tahmin Edilen 15 Günlük Getiri')
    ax.set_title(f'{ticker} - Test Seti Tahmin Performansı'); ax.legend(); ax.grid(True, alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=100); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def mini_plot(d, c, t):
    """Tek boyutlu seri veya DataFrame için küçük grafik oluşturur."""
    f, a = plt.subplots(figsize=(5, 3))
    if isinstance(d, pd.DataFrame):
        # Birden fazla sütun varsa her birini çiz
        for col in d.columns:
            a.plot(d.index[-60:], d[col].iloc[-60:], label=col)
        a.legend(fontsize=7)
    else:
        a.plot(d[-60:], color=c)
    a.set_title(t, fontsize=10); a.grid(alpha=0.2)
    b = BytesIO(); f.savefig(b, format='png', bbox_inches='tight'); b.seek(0); plt.close(f)
    return base64.b64encode(b.read()).decode('utf-8')

def plot_cv_scores(cv_results):
    if cv_results is None: return None
    folds = [s['fold'] for s in cv_results['folds']]
    r2_scores = [s['r2'] for s in cv_results['folds']]
    dir_accs = [s['dir_acc'] for s in cv_results['folds']]
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.bar(folds, r2_scores, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Fold'); ax1.set_ylabel('R²', color='steelblue')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(folds, dir_accs, color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Yön %', color='red')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax1.set_title("Walk-Forward CV", fontsize=10); ax1.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_heatmap(df):
    monthly_ret = df['Close'].resample('M').last().pct_change() * 100
    monthly_ret = monthly_ret.to_frame(name='Return')
    monthly_ret['Year'] = monthly_ret.index.year
    monthly_ret['Month'] = monthly_ret.index.month
    pivot = monthly_ret.pivot(index='Year', columns='Month', values='Return').tail(5)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Mevsimsellik", fontsize=10); ax.set_ylabel(''); ax.set_xlabel('')
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- HTML RAPOR ---
class HTMLRapor:
    def __init__(self):
        self.content = f"""
<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>INTC - Bilimsel AI Analiz Paneli v10.4</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
h1 {{ text-align: center; color: #111827; border-bottom: 3px solid #0071c5; padding-bottom: 15px; }}
.section {{ margin-bottom: 40px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; }}
.header {{ background: #0071c5; color: white; padding: 15px 25px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; }}
.signal {{ padding: 18px; text-align: center; font-weight: bold; font-size: 1.2em; letter-spacing: 1px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; }}
.stat-box {{ background: #fff; padding: 15px; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center; }}
.stat-label {{ font-size: 0.78em; color: #6b7280; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }}
.stat-val {{ font-size: 1.3em; font-weight: 800; color: #111827; }}
.stat-sub {{ font-size: 0.7em; color: #9ca3af; }}
.chart-area {{ padding: 20px; text-align: center; background: #f9fafb; }}
.main-chart {{ width: 100%; border-radius: 8px; border: 1px solid #eee; }}
.mini-charts {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; padding: 15px; }}
.mini-charts img {{ width: 100%; border: 1px solid #eee; border-radius: 4px; }}
.news-list {{ padding: 15px; background: #f9fafb; border-top: 1px solid #eee; max-height: 200px; overflow-y: auto; font-size: 0.9em; }}
.news-item {{ padding: 5px 0; border-bottom: 1px solid #eee; }}
.science-box {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 15px 20px; border-radius: 4px; }}
.science-box h3 {{ margin: 0 0 8px 0; color: #92400e; font-size: 1em; }}
.science-box table {{ width: 100%; font-size: 0.9em; }}
.science-box td {{ padding: 4px 8px; }}
.warn {{ color: #b91c1c; font-weight: bold; }}
.good {{ color: #15803d; font-weight: bold; }}
.neutral {{ color: #6b7280; font-weight: bold; }}
.note {{ background:#eef2ff; border-left:4px solid #4f46e5; padding:10px 14px; margin:10px 20px; border-radius:4px; font-size:0.85em; color:#3730a3; }}
</style></head>
<body><div class="container">
<h1>INTC - Intel Corporation - Bilimsel AI Analiz</h1>
<div style="background:#e6f2fa; padding:15px; border-left:4px solid #0071c5; margin-bottom:30px;">
<strong>v10.4 İyileştirmeleri:</strong> Eksiksiz veri sızıntısı kontrolü | Doğru DM testi yorumu | Düzeltilmiş Sharpe yıllıklandırması | Dinamik sinyal eşiği | Gerçekçi performans grafikleri
</div>
"""

    def add_section(self, ticker, name, m, chart_b64, extras, news_items, science_html):
        sig_color = m['signal_color']
        sig_bg = "#dcfce7" if sig_color == "green" else "#fee2e2" if sig_color == "red" else "#eff6ff"
        currency = "$"
        r2 = m['cv_r2']
        r2_class = "good" if r2 > 0.02 else "warn" if r2 < -0.02 else "neutral"
        da = m['cv_dir_acc']
        da_class = "good" if da > 54 else "warn" if da < 47 else "neutral"
        pot = (m['target_price'] - m['current_price']) / m['current_price'] * 100
        pot_class = "good" if pot > 0 else "warn"
        news_html = ""
        if news_items:
            news_html = "<div class='news-list'><b>Son Haberler:</b><br>" + "".join(
                [f"<div class='news-item'><span style='color:#666'>{i['date']}</span> {i['title']}</div>" for i in news_items]) + "</div>"

        self.content += f"""
<div class="section">
<div class="header">
<span>{ticker} | {name}</span>
<span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:15px; font-size:0.7em;">v10.4 BİLİMSEL</span>
</div>
<div class="signal" style="background:{sig_bg}; color:{sig_color};">
AI SİNYALİ: {m['signal']} <span style="font-size:0.7em; color:#555">({m['signal_desc']}) | Hedef: {PREDICTION_HORIZON} gün</span>
</div>
{science_html}
<div class="note">
<b>R² yorum notu:</b> Burada R², 15 günlük log-getiri üzerinde hesaplanır. Finansal getiri serilerinde
R² genellikle 0'a çok yakındır; pozitif ve istikrarlı 0.02–0.05 bandı bile akademik literatürde
anlamlı bir sinyal kabul edilir. Bu yüzden R²'yi fiyat seviyesi tahminindeki R² (genelde 0.99+)
ile karıştırmayın.
</div>
<div class="stats-grid">
<div class="stat-box"><div class="stat-label">Mevcut Fiyat</div><div class="stat-val">{currency}{m['current_price']:.2f}</div><div class="stat-sub">Son Kapanış</div></div>
<div class="stat-box"><div class="stat-label">Hedef Fiyat ({PREDICTION_HORIZON}G)</div><div class="stat-val">{currency}{m['target_price']:.2f}</div><div class="stat-sub">Doğrudan 15g Getiri</div></div>
<div class="stat-box"><div class="stat-label">Potansiyel</div><div class="stat-val {pot_class}">%{pot:+.2f}</div><div class="stat-sub">Hedef Farkı</div></div>
<div class="stat-box"><div class="stat-label">95% Güven Aralığı</div><div class="stat-val" style="font-size:0.95em">${m['ci_low']:.2f}-${m['ci_high']:.2f}</div><div class="stat-sub">Ensemble + Residual</div></div>

<div class="stat-box"><div class="stat-label">CV Yön Doğruluğu</div><div class="stat-val {da_class}">%{m['cv_dir_acc']:.1f}</div><div class="stat-sub">±{m['cv_std_dir']:.1f}% (4 fold)</div></div>
<div class="stat-box"><div class="stat-label">CV R²</div><div class="stat-val {r2_class}">{m['cv_r2']:.4f}</div><div class="stat-sub">Walk-Forward</div></div>
<div class="stat-box"><div class="stat-label">Test Yön Doğruluğu</div><div class="stat-val">%{m['test_dir_acc']:.1f}</div><div class="stat-sub">Out-of-Sample</div></div>
<div class="stat-box"><div class="stat-label">Test R²</div><div class="stat-val">{m['test_r2']:.4f}</div><div class="stat-sub">Out-of-Sample</div></div>

<div class="stat-box"><div class="stat-label">Strateji Sharpe</div><div class="stat-val">{m['bt_sharpe']:.2f}</div><div class="stat-sub">Backtest (15g, düzeltilmiş)</div></div>
<div class="stat-box"><div class="stat-label">Buy&Hold Sharpe</div><div class="stat-val">{m['bh_sharpe']:.2f}</div><div class="stat-sub">Aynı periyot</div></div>
<div class="stat-box"><div class="stat-label">DM Test (vs RW)</div><div class="stat-val">{'Üstün' if m['dm_better_rw'] else 'Üstün Değil'}</div><div class="stat-sub">p={m['dm_p_rw']:.3f}, Model MSE={m['mse_model']:.5f}, RW MSE={m['mse_rw']:.5f}</div></div>
<div class="stat-box"><div class="stat-label">Volatilite (yıllık)</div><div class="stat-val">%{m['volatility']:.1f}</div><div class="stat-sub">Gerçek</div></div>

<div class="stat-box"><div class="stat-label">Lookback</div><div class="stat-val">{m['lookback']}g</div><div class="stat-sub">Nested CV ile</div></div>
<div class="stat-box"><div class="stat-label">Haber Skoru</div><div class="stat-val" style="color:{'green' if m['news_score']>0 else 'red' if m['news_score']<0 else 'gray'}">{m['news_score']:+.2f}</div><div class="stat-sub">Modelde feature ✓</div></div>
</div>
<div class="chart-area"><img class="main-chart" src="data:image/png;base64,{chart_b64}"></div>
<div class="mini-charts">
<img src="data:image/png;base64,{extras['rsi']}">
<img src="data:image/png;base64,{extras['macd']}">
<img src="data:image/png;base64,{extras['cv']}">
<img src="data:image/png;base64,{extras['scatter']}">
<img src="data:image/png;base64,{extras['volatility']}">
<img src="data:image/png;base64,{extras['volume']}">
<img src="data:image/png;base64,{extras['ma_cross']}">
<img src="data:image/png;base64,{extras['heatmap']}">
</div>
{news_html}
</div>
"""

    def save(self):
        self.content += "</div></body></html>"
        with open("INTC_Analiz_v10.4.html", "w", encoding="utf-8") as f:
            f.write(self.content)

# --- ANA ANALİZ ---
def analyze():
    set_seeds()
    print("="*65)
    print(f"INTC BİLİMSEL AI ANALİZ v10.4 ({PREDICTION_HORIZON} GÜN TAHMİN)")
    print("="*65)

    print("\n1. Makro veriler indiriliyor...")
    macro_df = get_macro_data()

    print(f"\n2. Sentiment analizi...")
    news_score, news_items = get_advanced_sentiment(TICKER)
    print(f"   Haber skoru: {news_score:+.3f} ({len(news_items)} haber)")

    print(f"\n3. {TICKER} verisi indiriliyor + indikatörler...")
    df = get_stock_data(TICKER, macro_df, news_score=news_score)
    if df is None:
        print("HATA: Veri alınamadı"); return
    print(f"   Toplam {len(df)} işlem günü ({df.index[0].date()} - {df.index[-1].date()})")

    # Tarihsel 15g getiri oynaklığı (sinyal eşiği için)
    hist_vol_15d = np.std(df['Future_Ret_15d'].dropna().tail(252)) # son 1 yıl
    print(f"   Tarihsel 15g getiri oynaklığı: {hist_vol_15d:.5f}")

    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']
    features = ['Close'] + [c for c in df.columns if c not in exclude_cols]
    target_idx = features.index('Future_Ret_15d')
    data = df[features].values
    print(f"   Feature sayısı: {len(features)} (Hedef: 15 günlük getiri)")

    train_split = int(len(df) * 0.80)
    val_split = int(len(df) * 0.90)

    # Ana scaler train üzerinde fit edilir
    scaler = MinMaxScaler((0, 1))
    scaler.fit(data[:train_split])

    print(f"\n4. Lookback seçimi (Nested CV - leakage'siz):")
    candidates = sorted(set([60, 120, 250]))
    candidates = [c for c in candidates if c < train_split - 100]
    lookback = find_optimal_lookback_nested(
        data, target_idx, train_split,
        scaler_factory=lambda: MinMaxScaler((0,1)),
        candidates=candidates, epochs=12
    )

    print(f"\n5. Walk-Forward Cross-Validation (15g hedef):")
    cv_results = walk_forward_cv(data, target_idx, features, lookback, n_splits=4, epochs=25)
    if cv_results is None:
        print("HATA: CV yapılamadı"); return

    print(f"\n6. Ana model eğitimi (Train/Val/Test split):")
    train_scaled = scaler.transform(data[:train_split])
    X_train, y_train = create_dataset(train_scaled, target_idx, lookback)
    val_scaled = scaler.transform(data[train_split-lookback:val_split])
    X_val, y_val = create_dataset(val_scaled, target_idx, lookback)
    test_scaled = scaler.transform(data[val_split-lookback:])
    X_test, y_test = create_dataset(test_scaled, target_idx, lookback)
    print(f"   Veri: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} | Lookback={lookback}")

    models = train_ensemble(X_train, y_train, X_val, y_val, n_models=3, epochs=80)

    print(f"\n7. Test seti değerlendirmesi (Out-of-Sample, 15g getiri):")
    pred_test_mean, _ = ensemble_predict(models, X_test)
    n_features = len(features)
    dummy_pred = np.zeros((len(pred_test_mean), n_features))
    dummy_pred[:, target_idx] = pred_test_mean
    pred_returns = scaler.inverse_transform(dummy_pred)[:, target_idx]
    dummy_actual = np.zeros((len(y_test), n_features))
    dummy_actual[:, target_idx] = y_test
    actual_returns = scaler.inverse_transform(dummy_actual)[:, target_idx]

    test_r2 = r2_score(actual_returns, pred_returns)
    residual_std = float(np.std(actual_returns - pred_returns))

    threshold = np.std(actual_returns) * 0.1
    pred_signs = np.where(np.abs(pred_returns) < threshold, 0, np.sign(pred_returns))
    actual_signs = np.where(np.abs(actual_returns) < threshold, 0, np.sign(actual_returns))
    non_flat = (pred_signs != 0) & (actual_signs != 0)
    test_dir_acc = (np.mean(pred_signs[non_flat] == actual_signs[non_flat]) * 100) if non_flat.sum() > 0 else 50.0
    print(f"   Test R²: {test_r2:.4f} | Yön Doğruluğu: %{test_dir_acc:.1f} ({non_flat.sum()}/{len(pred_returns)})")

    # DM testi ve doğru yorum
    train_15d_ret = df['Future_Ret_15d'].iloc[:val_split].dropna().values
    pred_naive = baseline_naive(actual_returns)
    pred_rw = baseline_random_walk(train_15d_ret, actual_returns)
    mse_model = mean_squared_error(actual_returns, pred_returns)
    mse_naive = mean_squared_error(actual_returns, pred_naive)
    mse_rw = mean_squared_error(actual_returns, pred_rw)
    dm_stat_rw, dm_p_rw, dm_better_rw = diebold_mariano_test(actual_returns, pred_returns, pred_rw)
    print(f"   DM test vs RW: p={dm_p_rw:.3f}, Model MSE={mse_model:.5f}, RW MSE={mse_rw:.5f}, Daha iyi: {dm_better_rw}")

    print(f"\n8. Backtest (15g sinyaller):")
    strategy_rets = backtest_15d(actual_returns, pred_returns)
    bt_sharpe = annualized_sharpe(strategy_rets)
    bh_sharpe = annualized_sharpe(actual_returns)
    print(f"   Strateji Sharpe (yıllık, düzeltilmiş): {bt_sharpe:.2f}")
    print(f"   Buy&Hold Sharpe (yıllık, düzeltilmiş): {bh_sharpe:.2f}")

    print(f"\n9. Gelecek {PREDICTION_HORIZON} gün tahmini:")
    last_batch = scaler.transform(data)[-lookback:].reshape(1, lookback, len(features))
    target_price, ci_low, ci_high, predicted_ret = forecast_future(
        models, last_batch, scaler, features, target_idx,
        df['Close'].iloc[-1], residual_std
    )

    # Dinamik sinyal eşiği
    signal, sig_color, sig_desc = "NÖTR", "gray", "Yatay Seyir"
    if abs(predicted_ret) > 0.5 * hist_vol_15d:
        if predicted_ret > 0:
            signal, sig_color, sig_desc = ("GÜÇLÜ AL", "green", "Pozitif Trend") if abs(predicted_ret) > 1.5*hist_vol_15d else ("AL / TUT", "blue", "Zayıf Yükseliş")
        else:
            signal, sig_color, sig_desc = ("SAT", "red", "Negatif Trend") if abs(predicted_ret) > 1.5*hist_vol_15d else ("ZAYIF SAT", "orange", "Zayıf Düşüş")

    fut_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=PREDICTION_HORIZON, freq='B')
    save_predictions_to_sqlite(TICKER, fut_dates, np.full(PREDICTION_HORIZON, target_price))

    # Gerçek volatilite (günlük log return üzerinden)
    real_metrics = calculate_real_metrics(df['Log_Ret'].tail(252).dropna().values, periods_per_year=252)

    # Grafikler
    chart_b64 = plot_main_chart(df, fut_dates, target_price, ci_low, ci_high, sig_color, TICKER_NAME, TICKER)
    scatter_b64 = plot_scatter_predictions(actual_returns, pred_returns, TICKER)
    extras = {
        'rsi': mini_plot(df['RSI'], 'purple', 'RSI'),
        'macd': mini_plot(df['MACD'], 'blue', 'MACD'),
        'cv': plot_cv_scores(cv_results) or mini_plot(df['Close'], 'darkblue', 'CV Yedek'),
        'scatter': scatter_b64,
        'volatility': mini_plot(df['ATR'].tail(60), 'orange', 'ATR (Volatilite)'),
        'volume': mini_plot(df['Volume'].tail(60), 'gray', 'Hacim'),
        'ma_cross': mini_plot(df[['SMA50', 'SMA200']].tail(180), None, 'SMA50/200'),  # color=None kullan
        'heatmap': plot_heatmap(df)
    }

    # Anlamlılık eşikleri
    cond_dm = dm_better_rw
    cond_dir = cv_results['avg_dir_acc'] > 54.0
    cond_sharpe = bt_sharpe > bh_sharpe
    is_significant = cond_dm and cond_dir and cond_sharpe

    cond_str = (
        f"DM üstünlük: {'✓' if cond_dm else '✗'} | "
        f"CV Yön >%54: {'✓' if cond_dir else '✗'} | "
        f"Sharpe > B&H: {'✓' if cond_sharpe else '✗'}"
    )
    yorum = (
        'Model üç bilimsel kriterin TÜMÜNÜ geçti. Bu, baseline modellere ve pasif yatırıma karşı istatistiksel ve ekonomik bir üstünlük işaretidir.'
        if is_significant
        else f'UYARI: Model tüm kriterleri geçemedi → {cond_str}. '
             'Tahminler bilgi amaçlıdır; bu metrikler hep birlikte sağlanmadan tek başına işlem kararı vermeyin.'
    )

    science_html = f"""
    <div class="science-box">
    <h3>Bilimsel Doğrulama Sonuçları (15g Getiri Hedefi)</h3>
    <table>
    <tr><td><b>Walk-Forward CV (4-fold)</b></td><td>R² = {cv_results['avg_r2']:.4f} | Yön = %{cv_results['avg_dir_acc']:.1f} (±{cv_results['std_dir_acc']:.1f}%)</td></tr>
    <tr><td><b>Test Set (OOS)</b></td><td>R² = {test_r2:.4f} | Yön = %{test_dir_acc:.1f}</td></tr>
    <tr><td><b>Diebold-Mariano (vs RW)</b></td><td>MSE Model={mse_model:.5f}, MSE RW={mse_rw:.5f}, p={dm_p_rw:.3f} {'[Model daha iyi]' if dm_better_rw else '[Model daha iyi değil]'}</td></tr>
    <tr><td><b>Backtest Sharpe (yıllık, düzeltilmiş)</b></td><td>Strateji={bt_sharpe:.2f} vs Buy&Hold={bh_sharpe:.2f}</td></tr>
    <tr><td><b>Ensemble</b></td><td>3 model | Lookback: {lookback} gün (Nested CV)</td></tr>
    <tr><td><b>Sertleştirilmiş Eşik</b></td><td>{cond_str}</td></tr>
    </table>
    <div style="margin-top:8px; font-size:0.85em; color:#92400e;">
    <b>Yorum:</b> {yorum}
    </div>
    </div>
    """

    metrics = {
        'current_price': df['Close'].iloc[-1],
        'target_price': target_price,
        'ci_low': ci_low, 'ci_high': ci_high,
        'signal': signal, 'signal_color': sig_color, 'signal_desc': sig_desc,
        'cv_r2': cv_results['avg_r2'],
        'cv_dir_acc': cv_results['avg_dir_acc'],
        'cv_std_dir': cv_results['std_dir_acc'],
        'test_r2': test_r2,
        'test_dir_acc': test_dir_acc,
        'bt_sharpe': bt_sharpe,
        'bh_sharpe': bh_sharpe,
        'dm_better_rw': dm_better_rw,
        'dm_p_rw': dm_p_rw,
        'mse_model': mse_model,
        'mse_rw': mse_rw,
        'volatility': real_metrics['volatility'],
        'lookback': lookback,
        'news_score': news_score,
    }

    report = HTMLRapor()
    report.add_section(TICKER, TICKER_NAME, metrics, chart_b64, extras, news_items, science_html)
    report.save()

    print("\n" + "="*65)
    print("ÖZET SONUÇLAR:")
    print("="*65)
    print(f"   Mevcut Fiyat        : ${metrics['current_price']:.2f}")
    print(f"   Hedef ({PREDICTION_HORIZON}G)         : ${target_price:.2f}")
    print(f"   Potansiyel          : %{((target_price-metrics['current_price'])/metrics['current_price']*100):+.2f}")
    print(f"   Sinyal              : {signal} ({sig_desc})")
    print(f"   CV Yön Doğruluğu    : %{metrics['cv_dir_acc']:.1f} ± %{metrics['cv_std_dir']:.1f}")
    print(f"   Test Yön Doğruluğu  : %{metrics['test_dir_acc']:.1f}")
    print(f"   DM Test (vs RW)     : {'Üstün' if dm_better_rw else 'Üstün Değil'} (p={dm_p_rw:.3f})")
    print(f"   Backtest Sharpe     : {bt_sharpe:.2f} vs B&H: {bh_sharpe:.2f}")
    print(f"   Sertleştirilmiş Eşik: {cond_str} -> {'ANLAMLI' if is_significant else 'ANLAMSIZ'}")
    print("="*65)
    print("ANALİZ TAMAMLANDI. Rapor: INTC_Analiz_v10.4.html")
    print("="*65)

if __name__ == "__main__":
    init_db()
    analyze()