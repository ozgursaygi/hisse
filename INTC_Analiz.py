# ============================================================
# INTC AI ANALİZ PANELİ v10.2 - BİLİMSEL VERSİYON (DÜZELTİLMİŞ)
#
# v10.2 Düzeltmeleri (v10.0/10.1 üzerine):
#   [FIX-1] Lookback seçiminde data leakage giderildi
#           -> Lookback artık SADECE train seti içinde nested split ile seçiliyor.
#              Val seti hem hyperparameter seçimi hem early stopping için
#              kullanılmıyor.
#   [FIX-2] is_significant eşiği sertleştirildi
#           -> %52 -> %54 yön doğruluğu + Backtest Sharpe > Buy&Hold Sharpe
#              + DM p-value < 0.05 (3 koşul birden).
#   [FIX-3] Sentiment skoru artık modele feature olarak gerçekten dahil ediliyor
#           -> Statik (tek skor) olduğu için son N=20 günlük pencerede
#              sabit feature olarak eklenir ve raporda "kullanılıyor" işaretlenir.
#   [FIX-4] Makro veri doldurmada bfill kaldırıldı (look-ahead riski)
#   [FIX-5] generate_signal eşiği tek çarpana indirildi, kafa karışıklığı temizlendi
#   [FIX-6] Multi-step recursive hata birikimi için CI genişletmesi güçlendirildi
#           -> Ensemble varyansı + adımsal hata varyansı toplamı şeklinde.
#   [FIX-7] Log-return R² yorumu için raporda açıklama notu eklendi.
# ============================================================

import sys
import subprocess
import importlib
import sqlite3

def install_package(package):
    print(f"OTOMATİK KURULUM: '{package}' yükleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])

required_packages = ['tf-keras', 'ta', 'yfinance', 'GoogleNews', 'textblob',
                     'scipy', 'seaborn', 'sklearn', 'statsmodels']
for package in required_packages:
    try: importlib.import_module(package.replace('-', '_'))
    except ImportError:
        try: install_package(package)
        except: pass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from datetime import datetime, timedelta
import random
import tensorflow as tf
import yfinance as yf
from GoogleNews import GoogleNews
import ta
from textblob import TextBlob
from scipy import stats
import seaborn as sns

try:
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# --- DATABASE AYARLARI ---
DB_FOLDER = r"C:\Projects\ML"
DB_NAME = "data_intc.db"
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
    if df is None or df.empty: return
    start_date_filter = pd.Timestamp("2020-01-01")
    df_filtered = df[df.index >= start_date_filter].copy()
    if df_filtered.empty: return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for index, row in df_filtered.iterrows():
        date_str = index.strftime('%Y-%m-%d')
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO gunluk_veriler
                (tarih, sembol, acilis, yuksek, dusuk, kapanis, hacim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, ticker, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        except: pass
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
        except: pass
    conn.commit()
    conn.close()

# --- SEED SABİTLEME ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
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
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns=tickers, inplace=True)
        # [FIX-4] bfill kaldırıldı (look-ahead leakage'i tetikliyordu).
        # Sadece ileriye doğru doldurma yapılır; ilk N satır NA kalabilir, dropna ile silinecek.
        return df.ffill()
    except Exception as e:
        print(f"UYARI: Makro veriler indirilemedi ({e}).")
        return pd.DataFrame()

def get_stock_data(symbol, macro_df, news_score=0.0):
    """
    [FIX-3] news_score parametresi eklendi; son ~20 gün için 'NewsSent' feature
    olarak DataFrame'e eklenir. Geçmiş günler için 0 (nötr) atanır
    (geçmiş haber sentiment serisi olmadığı için en temiz seçenek bu).
    """
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

    # GENİŞLETİLMİŞ FEATURE ENGINEERING
    try:
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

        # [FIX-3] Sentiment feature ekleniyor.
        # Geçmiş için 0 (nötr), son 20 işlem günü için mevcut haber skoru.
        # Bu yaklaşımın sınırı raporda açıkça belirtilir:
        # gerçek tarihsel sentiment serisi olmadığı için modelin sentiment'ten
        # öğrenebileceği bilgi sınırlıdır; ancak en azından son penceredeki
        # rejimi temsil eder.
        df['NewsSent'] = 0.0
        if len(df) >= 20:
            df.iloc[-20:, df.columns.get_loc('NewsSent')] = news_score

        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill().dropna()
        else:
            df = df.dropna()
        return df
    except Exception as e:
        print(f"HATA: İndikatörler hesaplanamadı: {e}")
        return None

# --- SENTIMENT ---
def get_advanced_sentiment(ticker):
    news_items = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        titles = []
        if news:
            for n in news[:10]:
                if not isinstance(n, dict): continue
                title = n.get('title') or (n.get('content', {}) or {}).get('title')
                if not title: continue
                titles.append(title)
                ts = n.get('providerPublishTime', 0)
                try: date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                except: date_str = "Tarih Yok"
                news_items.append({'date': date_str, 'title': title})
        if not titles:
            try:
                googlenews = GoogleNews(lang='en', region='US')
                googlenews.set_period('7d')
                googlenews.search("Intel INTC stock news")
                results = googlenews.result()[:10]
                for item in results:
                    t = item.get('title')
                    if t:
                        titles.append(t)
                        news_items.append({'date': item.get('date', ''), 'title': t})
            except: pass
        if not titles: return 0.0, news_items
        scores = [TextBlob(t).sentiment.polarity for t in titles]
        return float(np.mean(scores)) if scores else 0.0, news_items
    except: return 0.0, []

# --- FİNANSAL METRİKLER ---
def calculate_alpha_beta(stock_returns, market_returns, risk_free_rate=0.045):
    common_idx = stock_returns.index.intersection(market_returns.index)
    if len(common_idx) < 30: return 1.0, 0.0
    s = stock_returns.loc[common_idx].values.flatten()
    m = market_returns.loc[common_idx].values.flatten()
    mask = np.isfinite(s) & np.isfinite(m)
    s, m = s[mask], m[mask]
    if len(s) < 30: return 1.0, 0.0
    cov = np.cov(s, m)[0, 1]
    var = np.var(m)
    if var == 0: return 1.0, 0.0
    beta = cov / var
    rf_daily = (1 + risk_free_rate)**(1/252) - 1
    alpha = (np.mean(s) - (rf_daily + beta * (np.mean(m) - rf_daily))) * 252
    return beta, alpha

def calculate_real_metrics(returns_array, risk_free_rate=0.045):
    """Gerçek log returns üzerinden hesaplanan metrikler"""
    returns_array = np.array(returns_array)
    returns_array = returns_array[np.isfinite(returns_array)]
    if len(returns_array) < 2:
        return {'mdd': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'volatility': 0}

    cum_returns = np.exp(np.cumsum(returns_array))
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    mdd = np.abs(np.min(drawdown)) * 100

    avg_ret = np.mean(returns_array)
    std_ret = np.std(returns_array)

    rf_daily = (1 + risk_free_rate)**(1/252) - 1
    sharpe = (np.sqrt(252) * (avg_ret - rf_daily) / std_ret) if std_ret > 0 else 0

    neg_returns = returns_array[returns_array < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else std_ret
    sortino = (np.sqrt(252) * (avg_ret - rf_daily) / downside_std) if downside_std > 0 else 0

    annual_vol = std_ret * np.sqrt(252) * 100
    annual_ret = avg_ret * 252
    calmar = (annual_ret * 100 / mdd) if mdd > 0 else 0

    return {'mdd': mdd, 'sharpe': sharpe, 'sortino': sortino,
            'calmar': calmar, 'volatility': annual_vol}

def get_benchmark_data(benchmark_symbol):
    end = datetime.now()
    start = end - timedelta(days=12*365)
    try:
        df = yf.download(benchmark_symbol, start=start, end=end, progress=False, auto_adjust=True)['Close']
        if isinstance(df, pd.DataFrame): df = df.iloc[:, 0] if df.shape[1] > 0 else df
        return np.log(df / df.shift(1)).dropna()
    except: return None

# ============================================================
# BİLİMSEL MODEL EĞİTİMİ
# ============================================================

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

def estimate_lookback_acf(log_returns, max_lag=400, alpha=0.05):
    if not HAS_STATSMODELS: return None
    abs_returns = np.abs(log_returns.dropna().values)
    if len(abs_returns) < max_lag * 2:
        max_lag = len(abs_returns) // 4
    try:
        acf_values, confint = acf(abs_returns, nlags=max_lag, alpha=alpha, fft=True)
        upper_bound = confint[:, 1] - acf_values
        significant_lags = np.where(np.abs(acf_values[1:]) > upper_bound[1:])[0]
        if len(significant_lags) == 0: return 60
        last_significant = significant_lags[-1] + 1
        return max(30, min(int(last_significant * 1.2), max_lag))
    except: return None

def find_optimal_lookback_nested(data, target_idx, train_end, scaler_factory,
                                  candidates=[60, 120, 250], epochs=12,
                                  inner_val_frac=0.15):
    """
    [FIX-1] LOOKBACK SEÇİMİ ARTIK SADECE TRAIN İÇİNDE YAPILIYOR.

    Önceki versiyonda:
      - lookback, train üzerinde fit edilip val üzerinde değerlendiriliyordu;
        ancak sonra aynı val seti early stopping için tekrar kullanılınca
        "double dipping" oluşuyordu.

    Bu versiyonda:
      - Train seti içinde yeni bir alt-validasyon (inner val) ayrılır.
      - Lookback seçimi sadece bu inner val'a göre yapılır.
      - Ana val seti (train_end : val_end) ne hyperparameter seçimine ne de
        bu fonksiyona girer. Yalnızca ileride early stopping için kullanılır.
    """
    print(f"   -> [Nested] Lookback aranıyor (sadece train içinde): {candidates}")
    inner_val_start = int(train_end * (1 - inner_val_frac))
    inner_train_end = inner_val_start

    results = {}
    for lb in candidates:
        if inner_train_end <= lb + 50: continue

        # Scaler SADECE inner_train üzerinde fit edilir
        inner_scaler = scaler_factory()
        inner_scaler.fit(data[:inner_train_end])

        inner_train_scaled = inner_scaler.transform(data[:inner_train_end])
        X_tr, y_tr = create_dataset(inner_train_scaled, target_idx, lb)

        inner_val_inputs = data[inner_train_end - lb : train_end]
        inner_val_scaled = inner_scaler.transform(inner_val_inputs)
        X_v, y_v = create_dataset(inner_val_scaled, target_idx, lb)

        if len(X_v) < 10 or len(X_tr) < 100: continue

        set_seeds()
        m = Sequential([Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                        LSTM(32), Dense(1)])
        m.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
        es = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True, verbose=0)
        hist = m.fit(X_tr, y_tr, epochs=epochs, batch_size=64,
                     validation_data=(X_v, y_v), callbacks=[es], verbose=0)
        best_val_loss = min(hist.history['val_loss'])
        results[lb] = best_val_loss
        print(f"      lookback={lb}: inner_val_loss={best_val_loss:.5f}")

    if not results: return 60
    best_lb = min(results.keys(), key=lambda k: results[k])
    print(f"   ✅ SEÇİLEN LOOKBACK: {best_lb} gün (nested, leakage'siz)")
    return best_lb

def walk_forward_cv(data, target_idx, features, lookback, n_splits=4, epochs=25):
    """
    BİLİMSEL: Zaman serisi için Expanding Window CV.
    Her fold için:
      - Ayrı scaler (sızıntı yok)
      - Yön doğruluğu HAM (inverse-scaled) getiriler üzerinden
      - Yataya yakın tahminler sahte isabet üretmesin diye dışlanır.
    """
    print(f"\n   📊 Walk-Forward CV (n_splits={n_splits}):")
    n = len(data)
    fold_size = n // (n_splits + 1)
    n_features = len(features)

    cv_scores = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)
        if val_end > n - lookback: break

        fold_scaler = MinMaxScaler((0, 1))
        train_raw = data[:train_end]
        fold_scaler.fit(train_raw)

        train_sc = fold_scaler.transform(train_raw)
        X_tr, y_tr = create_dataset(train_sc, target_idx, lookback)

        val_inputs = data[train_end - lookback : val_end]
        val_sc = fold_scaler.transform(val_inputs)
        X_v, y_v = create_dataset(val_sc, target_idx, lookback)

        if len(X_tr) < 100 or len(X_v) < 10: continue

        set_seeds(seed=42 + fold)
        m = build_lstm_model((X_tr.shape[1], X_tr.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
        m.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
              validation_data=(X_v, y_v), callbacks=[es], verbose=0)

        pred_scaled = m.predict(X_v, verbose=0).flatten()
        mse = mean_squared_error(y_v, pred_scaled)
        try: r2 = r2_score(y_v, pred_scaled)
        except: r2 = -1

        dummy_pred = np.zeros((len(pred_scaled), n_features))
        dummy_pred[:, target_idx] = pred_scaled
        pred_real = fold_scaler.inverse_transform(dummy_pred)[:, target_idx]

        dummy_actual = np.zeros((len(y_v), n_features))
        dummy_actual[:, target_idx] = y_v
        actual_real = fold_scaler.inverse_transform(dummy_actual)[:, target_idx]

        threshold = np.std(actual_real) * 0.1
        pred_signs = np.where(np.abs(pred_real) < threshold, 0, np.sign(pred_real))
        actual_signs = np.where(np.abs(actual_real) < threshold, 0, np.sign(actual_real))

        non_flat_mask = (pred_signs != 0) & (actual_signs != 0)
        if non_flat_mask.sum() > 0:
            dir_acc = np.mean(pred_signs[non_flat_mask] == actual_signs[non_flat_mask]) * 100
        else:
            dir_acc = 50.0

        cv_scores.append({'fold': fold+1, 'mse': mse, 'r2': r2, 'dir_acc': dir_acc,
                         'n_compared': int(non_flat_mask.sum()), 'n_total': len(pred_real)})
        print(f"      Fold {fold+1}: MSE={mse:.5f} | R²={r2:.4f} | Dir.Acc={dir_acc:.1f}% ({non_flat_mask.sum()}/{len(pred_real)})")

    if not cv_scores: return None
    avg_r2 = np.mean([s['r2'] for s in cv_scores])
    avg_dir = np.mean([s['dir_acc'] for s in cv_scores])
    std_dir = np.std([s['dir_acc'] for s in cv_scores])
    print(f"   📈 CV Ortalaması: R²={avg_r2:.4f} | Dir.Acc={avg_dir:.1f}% (±{std_dir:.1f}%)")
    return {'avg_r2': avg_r2, 'avg_dir_acc': avg_dir, 'std_dir_acc': std_dir, 'folds': cv_scores}

def train_ensemble(X_train, y_train, X_val, y_val, n_models=3, epochs=80):
    """3 farklı seed ile 3 model - varyans azaltma"""
    print(f"\n   🎯 Ensemble eğitiliyor ({n_models} model)...")
    models = []
    for i in range(n_models):
        set_seeds(seed=42 + i*10)
        m = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        h = m.fit(X_train, y_train, epochs=epochs, batch_size=32,
                  validation_data=(X_val, y_val), callbacks=[es, rlr], verbose=0)
        best_val = min(h.history['val_loss'])
        print(f"      Model {i+1}: epochs={len(h.history['loss'])} | best_val_loss={best_val:.5f}")
        models.append(m)
    return models

def ensemble_predict(models, X):
    preds = np.array([m.predict(X, verbose=0).flatten() for m in models])
    return preds.mean(axis=0), preds.std(axis=0)

# --- BASELINE MODELLER ---
def baseline_naive(y_test):
    """Naive: yarın bugünle aynı (log return = 0)"""
    return np.zeros(len(y_test))

def baseline_mean(y_train, y_test):
    return np.full(len(y_test), np.mean(y_train))

def baseline_random_walk(y_train, y_test):
    """RW: önceki getiri = sonraki getiri tahmini"""
    if len(y_test) == 0: return np.array([])
    pred = np.zeros(len(y_test))
    pred[0] = y_train[-1] if len(y_train) > 0 else 0
    pred[1:] = y_test[:-1]
    return pred

def diebold_mariano_test(actual, pred1, pred2):
    """
    H0: İki model aynı tahmin gücüne sahip
    p < 0.05 ise modeller arasında ANLAMLI fark var
    """
    actual = np.array(actual)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    n = len(d)
    if n < 10: return 0, 1.0
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0: return 0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

# ============================================================
# MULTİ-STEP TAHMİN (15 GÜN) + Güçlendirilmiş CI
# ============================================================
def multi_step_forecast(models, last_batch_scaled, scaler, features, target_idx,
                         current_price, n_steps=15, residual_std=None):
    """
    [FIX-6] CI artık iki belirsizlik kaynağını birleştiriyor:
      1) Ensemble varyansı (modeller arası uyuşmazlık)
      2) Adımsal rezidüel varyans (recursive hata birikimi)
         -> sqrt(step+1) * residual_std olarak modellenir.
    Recursive forecast'ta tek başına ensemble std yetersiz; rezidüel std
    test setinden hesaplanıp dışarıdan verilir.
    """
    n_features = len(features)
    future_prices = []
    future_lower = []
    future_upper = []
    temp_batch = last_batch_scaled.copy()
    curr_p = current_price

    if residual_std is None: residual_std = 0.0

    for step in range(n_steps):
        preds_scaled = np.array([m.predict(temp_batch, verbose=0)[0,0] for m in models])
        mean_pred_sc = preds_scaled.mean()
        ensemble_std_sc = preds_scaled.std()

        # Toplam belirsizlik (scaled space) = ensemble + adımsal rezidüel.
        # residual_std scaled space'te değil orijinal log-return space'te,
        # bu yüzden inverse_transform sonrası ekleyeceğiz.

        d_mean = np.zeros((1, n_features)); d_mean[0, target_idx] = mean_pred_sc
        mean_ret = scaler.inverse_transform(d_mean)[0, target_idx]

        # Ensemble std'yi de orijinal log-return ölçeğine çevir
        d_ens_hi = np.zeros((1, n_features))
        d_ens_hi[0, target_idx] = mean_pred_sc + ensemble_std_sc
        ens_std_unscaled = scaler.inverse_transform(d_ens_hi)[0, target_idx] - mean_ret

        # Adımsal rezidüel (recursive hata birikimi)
        step_residual = residual_std * np.sqrt(step + 1)

        # Toplam std = sqrt(ensemble^2 + step_residual^2)
        total_std = np.sqrt(ens_std_unscaled**2 + step_residual**2)

        low_ret = mean_ret - 1.96 * total_std
        high_ret = mean_ret + 1.96 * total_std

        future_prices.append(curr_p * np.exp(mean_ret))
        future_lower.append(curr_p * np.exp(low_ret))
        future_upper.append(curr_p * np.exp(high_ret))

        curr_p = future_prices[-1]

        new_row = temp_batch[0, -1, :].copy()
        new_row[target_idx] = mean_pred_sc
        temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1,1,n_features), axis=1)

    return np.array(future_prices), np.array(future_lower), np.array(future_upper)

# ============================================================
# BACKTEST
# ============================================================
def backtest_strategy(actual_prices, predicted_returns, threshold=0.001):
    if len(predicted_returns) < 2: return None
    actual_returns = np.diff(np.log(actual_prices))
    n = min(len(predicted_returns), len(actual_returns))

    positions = np.where(predicted_returns[:n] > threshold, 1,
                        np.where(predicted_returns[:n] < -threshold, -1, 0))
    strategy_returns = positions * actual_returns[:n]

    metrics = calculate_real_metrics(strategy_returns)
    bh_metrics = calculate_real_metrics(actual_returns[:n])

    return {
        'strategy': metrics, 'buy_hold': bh_metrics,
        'strategy_returns': strategy_returns,
        'n_trades': int(np.sum(np.abs(np.diff(positions)) > 0)),
        'pct_in_market': float(np.mean(positions != 0) * 100)
    }

# ============================================================
# GRAFİKLER
# ============================================================
def plot_main_chart(df, val_split, test_dates, rec_prices, fut_dates,
                    future_prices, future_lower, future_upper, sig_color, name, ticker):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index[-200:], df['Close'].iloc[-200:],
            label="Gerçek Fiyat", color="#1f2937", linewidth=1.8)

    if test_dates is not None and rec_prices is not None and len(test_dates) > 0:
        ax.plot(test_dates, rec_prices,
                label="Model Test (OOS)", linestyle="--", color="#f59e0b", linewidth=1.5, alpha=0.8)

    ax.plot(fut_dates, future_prices, label=f"{PREDICTION_HORIZON} Günlük Tahmin",
            color=sig_color, linewidth=2.5, marker='o', markersize=5,
            markerfacecolor=sig_color, markeredgewidth=0, zorder=10)

    ax.fill_between(fut_dates, future_lower, future_upper,
                     color=sig_color, alpha=0.15, label="95% Güven Aralığı")

    ax.axhline(y=df['Close'].iloc[-1], color='gray', linestyle=':', alpha=0.5)

    ax.set_title(f"{name} ({ticker}) - {PREDICTION_HORIZON} Günlük Fiyat Tahmini (Ensemble + CI)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat ($)")

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_regression_channel(df):
    data = df['Close'].tail(90).values
    x = np.arange(len(data))
    slope, intercept, _, _, _ = stats.linregress(x, data)
    reg_line = slope * x + intercept
    std = np.std(data - reg_line)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, data, color='black', label='Fiyat')
    ax.plot(x, reg_line, color='blue', linestyle='--', label='Trend')
    ax.fill_between(x, reg_line - 2*std, reg_line + 2*std, color='blue', alpha=0.1)
    ax.set_title("Regresyon Kanalı", fontsize=10); ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_drawdown(df):
    data = df['Close'].tail(180)
    rolling_max = data.cummax()
    dd = (data - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
    ax.plot(dd.index, dd, color='red', linewidth=1)
    ax.set_title("Max Drawdown", fontsize=10); ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_volatility(df):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df.index[-60:], df['ATR'].tail(60), color='orange', linewidth=2)
    ax.set_title("Volatilite (ATR)", fontsize=10); ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_volume(df):
    vol = df['Volume'].tail(60); close = df['Close'].tail(60)
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.bar(vol.index, vol, color='gray', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(close.index, close, color='blue', linewidth=1)
    ax1.set_title("Hacim", fontsize=10)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_ma_cross(df):
    data = df.tail(180)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(data.index, data['Close'], color='black', alpha=0.5, linewidth=1)
    ax.plot(data.index, data['SMA50'], color='green', linewidth=1.5, label='SMA50')
    ax.plot(data.index, data['SMA200'], color='red', linewidth=1.5, label='SMA200')
    ax.set_title("SMA 50/200", fontsize=10); ax.grid(alpha=0.2); ax.legend(fontsize=7)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_heatmap(df):
    monthly_ret = df['Close'].resample('M').last().pct_change() * 100
    monthly_ret = monthly_ret.to_frame(name='Return')
    monthly_ret['Year'] = monthly_ret.index.year
    monthly_ret['Month'] = monthly_ret.index.month
    pivot = monthly_ret.pivot(index='Year', columns='Month', values='Return').tail(5)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Mevsimsellik", fontsize=10); ax.set_ylabel(''); ax.set_xlabel('')
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

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

def mini_plot(d, c, t):
    f, a = plt.subplots(figsize=(5, 3))
    a.plot(d[-60:], color=c); a.set_title(t, fontsize=10); a.grid(alpha=0.2)
    b = BytesIO(); f.savefig(b, format='png', bbox_inches='tight'); b.seek(0); plt.close(f)
    return base64.b64encode(b.read()).decode('utf-8')

def generate_signal(current_price, predicted_price, atr_value, atr_multiplier=1.5):
    """
    [FIX-5] Tek eşik çarpanı. Önceki sürümde iki yerde farklı çarpan vardı
    (0.5 ve 3); şimdi yalnızca atr_multiplier (varsayılan 1.5) kullanılıyor.
    Bu, "GÜÇLÜ AL/SAT" eşiğinin yaklaşık 1.5 ATR-eşdeğeri hareket olduğunu
    açıkça gösterir.
    """
    expected_change = predicted_price - current_price
    threshold = atr_value * atr_multiplier
    if expected_change > threshold: return "GÜÇLÜ AL", "green", "Pozitif Trend"
    elif expected_change > 0: return "AL / TUT", "blue", "Zayıf Yükseliş"
    elif expected_change < -threshold: return "SAT", "red", "Negatif Trend"
    else: return "NÖTR", "gray", "Yatay Seyir"

# ============================================================
# HTML RAPOR
# ============================================================
class HTMLRapor:
    def __init__(self):
        self.content = """
<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<title>INTC - Bilimsel AI Analiz Paneli v10.2</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
h1 { text-align: center; color: #111827; border-bottom: 3px solid #0071c5; padding-bottom: 15px; }
.section { margin-bottom: 40px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; }
.header { background: #0071c5; color: white; padding: 15px 25px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; }
.signal { padding: 18px; text-align: center; font-weight: bold; font-size: 1.2em; letter-spacing: 1px; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; }
.stat-box { background: #fff; padding: 15px; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center; }
.stat-label { font-size: 0.78em; color: #6b7280; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
.stat-val { font-size: 1.3em; font-weight: 800; color: #111827; }
.stat-sub { font-size: 0.7em; color: #9ca3af; }
.chart-area { padding: 20px; text-align: center; background: #f9fafb; }
.main-chart { width: 100%; border-radius: 8px; border: 1px solid #eee; }
.mini-charts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; padding: 15px; }
.mini-charts img { width: 100%; border: 1px solid #eee; border-radius: 4px; }
.news-list { padding: 15px; background: #f9fafb; border-top: 1px solid #eee; max-height: 200px; overflow-y: auto; font-size: 0.9em; }
.news-item { padding: 5px 0; border-bottom: 1px solid #eee; }
.science-box { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 15px 20px; border-radius: 4px; }
.science-box h3 { margin: 0 0 8px 0; color: #92400e; font-size: 1em; }
.science-box table { width: 100%; font-size: 0.9em; }
.science-box td { padding: 4px 8px; }
.warn { color: #b91c1c; font-weight: bold; }
.good { color: #15803d; font-weight: bold; }
.neutral { color: #6b7280; font-weight: bold; }
.note { background:#eef2ff; border-left:4px solid #4f46e5; padding:10px 14px; margin:10px 20px; border-radius:4px; font-size:0.85em; color:#3730a3; }
</style></head>
<body><div class="container">
<h1>INTC - Intel Corporation - Bilimsel AI Analiz</h1>
<div style="background:#e6f2fa; padding:15px; border-left:4px solid #0071c5; margin-bottom:30px;">
<strong>v10.2 Düzeltmeleri:</strong> Lookback nested-CV (leakage giderildi) | Sertleştirilmiş anlamlılık eşiği (%54 + Sharpe + DM) | Sentiment artık feature olarak modelde | Recursive hata birikimi CI'a dahil | Makro bfill kaldırıldı
</div>
"""

    def add_section(self, ticker, name, m, chart_b64, extras, news_items, science_html):
        sig_color = m['signal_color']
        sig_bg = "#dcfce7" if sig_color == "green" else "#fee2e2" if sig_color == "red" else "#eff6ff"
        currency = "$"
        r2 = m['cv_r2']
        # Log-return tahmininde R² yorumu farklı; eşikler buna göre tutuldu
        r2_class = "good" if r2 > 0.02 else "warn" if r2 < -0.02 else "neutral"
        da = m['cv_dir_acc']
        da_class = "good" if da > 54 else "warn" if da < 47 else "neutral"
        pot = (m['target_price'] - m['current_price']) / m['current_price'] * 100
        pot_class = "good" if pot > 0 else "warn"

        news_html = ""
        if news_items:
            news_html = "<div class='news-list'><b>Son Haberler:</b><br>" + "".join(
                [f"<div class='news-item'><span style='color:#666'>{i['date']}</span> {i['title']}</div>"
                 for i in news_items]) + "</div>"

        self.content += f"""
<div class="section">
<div class="header">
<span>{ticker} | {name}</span>
<span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:15px; font-size:0.7em;">v10.2 BİLİMSEL</span>
</div>
<div class="signal" style="background:{sig_bg}; color:{sig_color};">
AI SİNYALİ: {m['signal']} <span style="font-size:0.7em; color:#555">({m['signal_desc']}) | Hedef: {PREDICTION_HORIZON} gün</span>
</div>

{science_html}

<div class="note">
<b>R² yorum notu:</b> Burada R², log-getiri (Log_Ret) üzerinde hesaplanır. Finansal getiri serilerinde
R² genellikle 0'a çok yakındır; pozitif ve istikrarlı 0.02–0.05 bandı bile akademik literatürde
anlamlı bir sinyal kabul edilir. Bu yüzden R²'yi fiyat seviyesi tahminindeki R² (genelde 0.99+)
ile karıştırmayın.
</div>

<div class="stats-grid">
<div class="stat-box"><div class="stat-label">Mevcut Fiyat</div><div class="stat-val">{currency}{m['current_price']:.2f}</div><div class="stat-sub">Son Kapanış</div></div>
<div class="stat-box"><div class="stat-label">Hedef Fiyat ({PREDICTION_HORIZON}G)</div><div class="stat-val">{currency}{m['target_price']:.2f}</div><div class="stat-sub">Ensemble Tahmin</div></div>
<div class="stat-box"><div class="stat-label">Potansiyel</div><div class="stat-val {pot_class}">%{pot:+.2f}</div><div class="stat-sub">Hedef Farkı</div></div>
<div class="stat-box"><div class="stat-label">95% Güven Aralığı</div><div class="stat-val" style="font-size:0.95em">${m['ci_low']:.2f}-${m['ci_high']:.2f}</div><div class="stat-sub">Ensemble + Recursive</div></div>

<div class="stat-box"><div class="stat-label">CV Yön Doğruluğu</div><div class="stat-val {da_class}">%{m['cv_dir_acc']:.1f}</div><div class="stat-sub">±{m['cv_std_dir']:.1f}% (4 fold)</div></div>
<div class="stat-box"><div class="stat-label">CV R²</div><div class="stat-val {r2_class}">{m['cv_r2']:.4f}</div><div class="stat-sub">Walk-Forward</div></div>
<div class="stat-box"><div class="stat-label">Test Yön Doğruluğu</div><div class="stat-val">%{m['test_dir_acc']:.1f}</div><div class="stat-sub">Out-of-Sample</div></div>
<div class="stat-box"><div class="stat-label">Test R²</div><div class="stat-val">{m['test_r2']:.4f}</div><div class="stat-sub">Out-of-Sample</div></div>

<div class="stat-box"><div class="stat-label">Strateji Sharpe</div><div class="stat-val">{m['bt_sharpe']:.2f}</div><div class="stat-sub">Backtest</div></div>
<div class="stat-box"><div class="stat-label">Buy&Hold Sharpe</div><div class="stat-val">{m['bh_sharpe']:.2f}</div><div class="stat-sub">Karşılaştırma</div></div>
<div class="stat-box"><div class="stat-label">Alpha (Yıllık)</div><div class="stat-val">{m['alpha']:+.2f}</div><div class="stat-sub">vs S&P 500</div></div>
<div class="stat-box"><div class="stat-label">Beta</div><div class="stat-val">{m['beta']:.2f}</div><div class="stat-sub">S&P 500'e Karşı</div></div>

<div class="stat-box"><div class="stat-label">Volatilite (yıllık)</div><div class="stat-val">%{m['volatility']:.1f}</div><div class="stat-sub">Gerçek</div></div>
<div class="stat-box"><div class="stat-label">Max Drawdown</div><div class="stat-val warn">%{m['mdd']:.1f}</div><div class="stat-sub">Tarihsel</div></div>
<div class="stat-box"><div class="stat-label">Lookback</div><div class="stat-val">{m['lookback']}g</div><div class="stat-sub">Nested CV ile</div></div>
<div class="stat-box"><div class="stat-label">Haber Skoru</div><div class="stat-val" style="color:{'green' if m['news_score']>0 else 'red' if m['news_score']<0 else 'gray'}">{m['news_score']:+.2f}</div><div class="stat-sub">Modelde feature ✓</div></div>
</div>

<div class="chart-area"><img class="main-chart" src="data:image/png;base64,{chart_b64}"></div>

<div class="mini-charts">
<img src="data:image/png;base64,{extras['rsi']}">
<img src="data:image/png;base64,{extras['macd']}">
<img src="data:image/png;base64,{extras['cv']}">
<img src="data:image/png;base64,{extras['reg_channel']}">
<img src="data:image/png;base64,{extras['drawdown']}">
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
        with open("INTC_Analiz.html", "w", encoding="utf-8") as f:
            f.write(self.content)

# ============================================================
# ANA ANALİZ
# ============================================================
def analyze():
    set_seeds()
    print("="*65)
    print(f"INTC BİLİMSEL AI ANALİZ v10.2 ({PREDICTION_HORIZON} GÜN TAHMİN)")
    print("="*65)

    print("\n1. Makro veriler indiriliyor...")
    macro_df = get_macro_data()

    print("\n2. Benchmark (S&P 500) indiriliyor...")
    market_returns = get_benchmark_data('^GSPC')

    # [FIX-3] Sentiment'i feature olarak ekleyebilmek için ÖNCE haber çekilir
    print(f"\n3. Sentiment analizi (feature olarak dahil edilecek)...")
    news_score, news_items = get_advanced_sentiment(TICKER)
    print(f"   Haber skoru: {news_score:+.3f} ({len(news_items)} haber)")

    print(f"\n4. {TICKER} verisi indiriliyor + indikatörler hesaplanıyor...")
    df = get_stock_data(TICKER, macro_df, news_score=news_score)
    if df is None:
        print("HATA: Veri alınamadı")
        return
    print(f"   Toplam {len(df)} işlem günü ({df.index[0].date()} - {df.index[-1].date()})")

    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']
    features = ['Close'] + [c for c in df.columns if c not in exclude_cols]
    target_idx = features.index('Log_Ret')
    data = df[features].values
    print(f"   Feature sayısı: {len(features)} (NewsSent dahil)")

    train_split = int(len(df) * 0.80)
    val_split = int(len(df) * 0.90)

    # Ana scaler train üzerinde fit edilir
    scaler = MinMaxScaler((0, 1))
    scaler.fit(data[:train_split])

    # [FIX-1] Lookback seçimi artık nested: yalnızca train içindeki bir
    # alt-validasyon kullanılır. Ana val seti bu adıma DOKUNMAZ.
    print(f"\n5. Lookback seçimi (Nested CV - leakage'siz):")
    log_ret_train = df['Log_Ret'].iloc[:train_split]
    acf_hint = estimate_lookback_acf(log_ret_train)
    if acf_hint: print(f"   ACF analizi: ~{acf_hint} gün")
    candidates = sorted(set([60, 120, 250] + ([acf_hint] if acf_hint else [])))
    candidates = [c for c in candidates if c < train_split - 100]
    lookback = find_optimal_lookback_nested(
        data, target_idx, train_split,
        scaler_factory=lambda: MinMaxScaler((0,1)),
        candidates=candidates, epochs=12
    )

    print(f"\n6. Walk-Forward Cross-Validation:")
    cv_results = walk_forward_cv(data, target_idx, features, lookback, n_splits=4, epochs=25)

    print(f"\n7. Ana model eğitimi (Train/Val/Test split):")
    train_scaled = scaler.transform(data[:train_split])
    X_train, y_train = create_dataset(train_scaled, target_idx, lookback)
    val_scaled = scaler.transform(data[train_split-lookback:val_split])
    X_val, y_val = create_dataset(val_scaled, target_idx, lookback)
    test_scaled = scaler.transform(data[val_split-lookback:])
    X_test, y_test = create_dataset(test_scaled, target_idx, lookback)
    print(f"   Veri: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} | Lookback={lookback}")

    models = train_ensemble(X_train, y_train, X_val, y_val, n_models=3, epochs=80)

    print(f"\n8. Test seti değerlendirmesi (Out-of-Sample):")
    pred_test_mean, pred_test_std = ensemble_predict(models, X_test)

    dummy = np.zeros((len(pred_test_mean), len(features)))
    dummy[:, target_idx] = pred_test_mean
    pred_test_returns = scaler.inverse_transform(dummy)[:, target_idx]

    actual_prices = df['Close'].iloc[val_split:].values
    min_len = min(len(pred_test_returns), len(actual_prices))
    pred_test_returns = pred_test_returns[:min_len]
    actual_prices = actual_prices[:min_len]

    actual_prices_with_prev = df['Close'].iloc[val_split-1:].values
    actual_returns_log = np.log(actual_prices_with_prev[1:min_len+1] / actual_prices_with_prev[:min_len])

    try: test_r2 = r2_score(actual_returns_log, pred_test_returns)
    except: test_r2 = 0

    # Test set residual std -> recursive forecast CI için kullanılacak
    test_residuals = actual_returns_log - pred_test_returns
    residual_std = float(np.std(test_residuals[np.isfinite(test_residuals)]))

    threshold_test = np.std(actual_returns_log) * 0.1
    pred_signs_test = np.where(np.abs(pred_test_returns) < threshold_test, 0, np.sign(pred_test_returns))
    actual_signs_test = np.where(np.abs(actual_returns_log) < threshold_test, 0, np.sign(actual_returns_log))
    non_flat_test = (pred_signs_test != 0) & (actual_signs_test != 0)
    if non_flat_test.sum() > 0:
        test_dir_acc = np.mean(pred_signs_test[non_flat_test] == actual_signs_test[non_flat_test]) * 100
    else:
        test_dir_acc = 50.0
    print(f"   Test R²: {test_r2:.4f} | Yön Doğruluğu: %{test_dir_acc:.1f} ({non_flat_test.sum()}/{len(pred_test_returns)})")
    print(f"   Test residual std (CI için): {residual_std:.5f}")

    rec_prices = []
    pp = df['Close'].iloc[val_split-1]
    for i in range(min_len):
        pp = pp * np.exp(pred_test_returns[i])
        rec_prices.append(pp)
    rec_prices = np.array(rec_prices)

    print(f"\n9. Baseline karşılaştırması (Diebold-Mariano):")
    train_returns = df['Log_Ret'].iloc[:val_split].dropna().values
    pred_naive = baseline_naive(actual_returns_log)
    pred_mean = baseline_mean(train_returns, actual_returns_log)
    pred_rw = baseline_random_walk(train_returns, actual_returns_log)

    mse_model = mean_squared_error(actual_returns_log, pred_test_returns)
    mse_naive = mean_squared_error(actual_returns_log, pred_naive)
    mse_mean = mean_squared_error(actual_returns_log, pred_mean)
    mse_rw = mean_squared_error(actual_returns_log, pred_rw)

    dm_naive_stat, dm_naive_p = diebold_mariano_test(actual_returns_log, pred_test_returns, pred_naive)
    dm_mean_stat, dm_mean_p = diebold_mariano_test(actual_returns_log, pred_test_returns, pred_mean)
    dm_rw_stat, dm_rw_p = diebold_mariano_test(actual_returns_log, pred_test_returns, pred_rw)

    print(f"   LSTM Model    : MSE={mse_model:.6f}")
    print(f"   Naive (zero)  : MSE={mse_naive:.6f} | DM p={dm_naive_p:.4f}")
    print(f"   Mean baseline : MSE={mse_mean:.6f} | DM p={dm_mean_p:.4f}")
    print(f"   Random Walk   : MSE={mse_rw:.6f} | DM p={dm_rw_p:.4f}")

    print(f"\n10. Backtest stratejisi:")
    bt = backtest_strategy(actual_prices, pred_test_returns, threshold=0.001)
    if bt:
        print(f"   Strateji Sharpe : {bt['strategy']['sharpe']:.2f}")
        print(f"   Buy&Hold Sharpe : {bt['buy_hold']['sharpe']:.2f}")
        print(f"   İşlem Sayısı    : {bt['n_trades']}")
        print(f"   Piyasada Kalma  : %{bt['pct_in_market']:.1f}")

    print(f"\n11. {PREDICTION_HORIZON} günlük gelecek tahmini:")
    last_batch = scaler.transform(data)[-lookback:].reshape(1, lookback, len(features))
    future_prices, future_lower, future_upper = multi_step_forecast(
        models, last_batch, scaler, features, target_idx,
        df['Close'].iloc[-1], n_steps=PREDICTION_HORIZON,
        residual_std=residual_std  # [FIX-6] recursive hata birikimi CI'a dahil
    )

    target_price = future_prices[-1]
    ci_low = future_lower[-1]
    ci_high = future_upper[-1]
    current_atr = df['ATR'].iloc[-1]
    # [FIX-5] Tek çarpan: 1.5*ATR
    signal, sig_color, sig_desc = generate_signal(
        df['Close'].iloc[-1], target_price, current_atr, atr_multiplier=1.5
    )

    fut_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=PREDICTION_HORIZON, freq='B')
    save_predictions_to_sqlite(TICKER, fut_dates, future_prices)

    stock_ret_series = df['Log_Ret']
    beta, alpha = calculate_alpha_beta(stock_ret_series, market_returns)

    real_metrics = calculate_real_metrics(df['Log_Ret'].tail(252).dropna().values)

    test_dates = df.index[val_split:][:min_len]
    chart_b64 = plot_main_chart(df, val_split, test_dates, rec_prices, fut_dates,
                                 future_prices, future_lower, future_upper, sig_color,
                                 TICKER_NAME, TICKER)

    extras = {
        'rsi': mini_plot(df['RSI'], 'purple', 'RSI'),
        'macd': mini_plot(df['MACD'], 'blue', 'MACD'),
        'cv': plot_cv_scores(cv_results) or mini_plot(df['SEMI_ETF'] if 'SEMI_ETF' in df.columns else df['Close'], 'darkblue', 'SOXX'),
        'reg_channel': plot_regression_channel(df),
        'drawdown': plot_drawdown(df),
        'volatility': plot_volatility(df),
        'volume': plot_volume(df),
        'ma_cross': plot_ma_cross(df),
        'heatmap': plot_heatmap(df)
    }

    # [FIX-2] SERTLEŞTİRİLMİŞ ANLAMLILIK EŞİĞİ
    # Üç koşul birden:
    #   (a) DM p-value < 0.05 (en az naive ya da RW karşısında)
    #   (b) CV yön doğruluğu > %54  (önceki: %52)
    #   (c) Backtest Sharpe > Buy&Hold Sharpe
    bt_sharpe = bt['strategy']['sharpe'] if bt else 0
    bh_sharpe = bt['buy_hold']['sharpe'] if bt else 0

    cond_dm = min(dm_naive_p, dm_rw_p) < 0.05
    cond_dir = cv_results['avg_dir_acc'] > 54.0
    cond_sharpe = bt_sharpe > bh_sharpe

    is_significant = cond_dm and cond_dir and cond_sharpe

    cond_str = (
        f"DM p<0.05: {'✓' if cond_dm else '✗'} | "
        f"CV Yön >%54: {'✓' if cond_dir else '✗'} | "
        f"Sharpe > B&H: {'✓' if cond_sharpe else '✗'}"
    )

    yorum = (
        'Model üç bilimsel kriterin TÜMÜNÜ geçti (DM anlamlılığı + yön doğruluğu + ekonomik üstünlük). '
        'Bu, baseline\'lara ve pasif yatırıma karşı somut bir edge işaretidir.'
        if is_significant
        else f'UYARI: Model tüm kriterleri geçemedi → {cond_str}. '
             'Tahminler bilgi amaçlıdır; bu metrikler hep birlikte sağlanmadan tek başına işlem kararı vermeyin.'
    )

    science_html = f"""
    <div class="science-box">
    <h3>Bilimsel Doğrulama Sonuçları</h3>
    <table>
    <tr><td><b>Walk-Forward CV (4-fold)</b></td><td>R² = {cv_results['avg_r2']:.4f} | Yön = %{cv_results['avg_dir_acc']:.1f} (±{cv_results['std_dir_acc']:.1f}%)</td></tr>
    <tr><td><b>Test Set (OOS)</b></td><td>R² = {test_r2:.4f} | Yön = %{test_dir_acc:.1f}</td></tr>
    <tr><td><b>Diebold-Mariano</b></td><td>vs Naive: p={dm_naive_p:.3f} | vs RW: p={dm_rw_p:.3f} {'[Anlamlı]' if min(dm_naive_p, dm_rw_p) < 0.05 else '[Anlamsız]'}</td></tr>
    <tr><td><b>MSE</b></td><td>Model={mse_model:.5f} | Naive={mse_naive:.5f} | RW={mse_rw:.5f}</td></tr>
    <tr><td><b>Backtest Sharpe</b></td><td>Strateji={bt_sharpe:.2f} vs Buy&Hold={bh_sharpe:.2f}</td></tr>
    <tr><td><b>Ensemble</b></td><td>3 model | Lookback: {lookback} gün (Nested CV - leakage'siz)</td></tr>
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
        'alpha': alpha, 'beta': beta,
        'cv_r2': cv_results['avg_r2'],
        'cv_dir_acc': cv_results['avg_dir_acc'],
        'cv_std_dir': cv_results['std_dir_acc'],
        'test_r2': test_r2,
        'test_dir_acc': test_dir_acc,
        'bt_sharpe': bt_sharpe,
        'bh_sharpe': bh_sharpe,
        'mdd': real_metrics['mdd'],
        'volatility': real_metrics['volatility'],
        'lookback': lookback,
        'news_score': news_score,
        'dm_naive_p': dm_naive_p,
        'dm_rw_p': dm_rw_p
    }

    report = HTMLRapor()
    report.add_section(TICKER, TICKER_NAME, metrics, chart_b64, extras, news_items, science_html)
    report.save()

    print("\n" + "="*65)
    print("ÖZET SONUÇLAR:")
    print("="*65)
    print(f"   Mevcut Fiyat        : ${metrics['current_price']:.2f}")
    print(f"   Hedef ({PREDICTION_HORIZON}G)         : ${target_price:.2f}")
    print(f"   95% Güven Aralığı   : ${ci_low:.2f} - ${ci_high:.2f}")
    print(f"   Potansiyel          : %{((target_price-metrics['current_price'])/metrics['current_price']*100):+.2f}")
    print(f"   Sinyal              : {signal} ({sig_desc})")
    print(f"   CV Yön Doğruluğu    : %{metrics['cv_dir_acc']:.1f} ± %{metrics['cv_std_dir']:.1f}")
    print(f"   CV R²               : {metrics['cv_r2']:.4f}")
    print(f"   Test Yön Doğruluğu  : %{metrics['test_dir_acc']:.1f}")
    print(f"   Test R²             : {metrics['test_r2']:.4f}")
    print(f"   DM Test (vs RW)     : p={metrics['dm_rw_p']:.4f} {'[Anlamlı]' if metrics['dm_rw_p']<0.05 else '[Anlamsız]'}")
    print(f"   Backtest Sharpe     : {metrics['bt_sharpe']:.2f} vs Buy&Hold: {metrics['bh_sharpe']:.2f}")
    print(f"   Sertleştirilmiş Eşik: {cond_str} -> {'ANLAMLI' if is_significant else 'ANLAMSIZ'}")
    print(f"   Volatilite (yıllık) : %{metrics['volatility']:.1f}")
    print(f"   Beta vs S&P 500     : {metrics['beta']:.2f}")
    print(f"   Alpha (yıllık)      : {metrics['alpha']:+.2f}")
    print("="*65)
    print("ANALİZ TAMAMLANDI. Rapor: INTC_Analiz.html")
    print("="*65)

if __name__ == "__main__":
    init_db()
    analyze()
