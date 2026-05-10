# --- OTOMATİK KURULUM VE BAĞLILIK KONTROLÜ (AUTO-INSTALLER v3) ---
import sys
import subprocess
import importlib
import sqlite3 

def install_package(package):
    print(f"OTOMATİK KURULUM: '{package}' yükleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])

# SADECE GEREKLİ KÜTÜPHANELER
required_packages = ['tf-keras', 'ta', 'yfinance', 'GoogleNews', 'textblob', 'scipy', 'seaborn', 'sklearn', 'imageio', 'statsmodels']
for package in required_packages:
    try: importlib.import_module(package.replace('-', '_'))
    except ImportError:
        try: install_package(package)
        except: pass

# --- STANDART İMPORTLAR ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tüm TF loglarını kapat
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # OneDNN uyarısını kapat
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
try:
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import seaborn as sns
import imageio.v2 as imageio 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_percentage_error

# --- DATABASE AYARLARI ---
DB_FOLDER = r"C:\Projects\ML"
DB_NAME = "data_intc.db"  # INTC için özel veritabanı
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def init_db():
    """Veritabanı ve tabloları oluşturur."""
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gunluk_veriler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih TEXT,
            sembol TEXT,
            acilis REAL,
            yuksek REAL,
            dusuk REAL,
            kapanis REAL,
            hacim REAL,
            UNIQUE(tarih, sembol)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tahminler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analiz_tarihi TEXT,
            hedef_tarih TEXT,
            sembol TEXT,
            tahmin_fiyati REAL,
            UNIQUE(analiz_tarihi, hedef_tarih, sembol)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_sqlite(ticker, df):
    """Gerçekleşen verileri kaydeder (2020 sonrası - Eksikleri tamamlar)."""
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
                INSERT OR IGNORE INTO gunluk_veriler (tarih, sembol, acilis, yuksek, dusuk, kapanis, hacim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, ticker, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        except: pass

    conn.commit()
    conn.close()

def save_predictions_to_sqlite(ticker, dates, prices, analysis_date=None):
    """Modelin ürettiği 7 günlük tahmini veritabanına kaydeder."""
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
                INSERT OR REPLACE INTO tahminler (analiz_tarihi, hedef_tarih, sembol, tahmin_fiyati)
                VALUES (?, ?, ?, ?)
            ''', (analiz_tarihi, hedef_tarih, ticker, float(price)))
        except: pass
    conn.commit()
    conn.close()

# --- GÜNCELLENMİŞ EKSİK TAHMİNLERİ DOLDURMA (2020'TEN İTİBAREN) ---
def fill_missing_predictions(ticker, df, model, scaler, features, target_idx, lookback):
    """
    2020-01-01 tarihinden bugüne kadar olan verilerde veritabanında tahmini olmayan günleri bulur 
    ve o günün şartlarıyla (geleceği görmeden) tahmin üretip kaydeder.
    """
    conn = sqlite3.connect(DB_PATH)

    start_date = pd.Timestamp("2020-01-01")
    valid_dates = df[df.index >= start_date].index

    filled_count = 0
    print(f"   -> Geçmiş analizler kontrol ediliyor (Başlangıç: {start_date.strftime('%Y-%m-%d')})...")

    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT analiz_tarihi FROM tahminler WHERE sembol=?", (ticker,))
    existing_dates = set(row[0] for row in cursor.fetchall())

    for current_date in valid_dates:
        date_str = current_date.strftime('%Y-%m-%d')

        if date_str in existing_dates:
            continue

        historical_df = df[df.index <= current_date]

        if len(historical_df) < lookback + 10: 
            continue

        data = historical_df[features].values
        full_data_scaled = scaler.transform(data)

        last_batch = full_data_scaled[-lookback:].reshape(1, lookback, len(features))

        future_prices = []
        curr_p = historical_df['Close'].iloc[-1]

        temp_batch = last_batch.copy()
        # Yumuşatma faktörü - 0.0: dürüst, 0.3: daha tutucu
        SMOOTH_FACTOR_HIST = 0.0
        recent_trend = np.mean(temp_batch[0, -10:, target_idx])

        for i in range(7):
            raw_pred = model.predict(temp_batch, verbose=0)[0,0]
            p_sc = (raw_pred * (1 - SMOOTH_FACTOR_HIST)) + (recent_trend * SMOOTH_FACTOR_HIST)

            d = np.zeros((1, len(features)))
            d[0, target_idx] = p_sc

            p_ret = scaler.inverse_transform(d)[0, target_idx]

            curr_p = curr_p * np.exp(p_ret)
            future_prices.append(curr_p)

            new_row = temp_batch[0, -1, :].copy()
            new_row[target_idx] = p_sc 
            temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1,1,len(features)), axis=1)

        fut_dates = pd.date_range(current_date + timedelta(days=1), periods=7)
        save_predictions_to_sqlite(ticker, fut_dates, future_prices, analysis_date=date_str)

        filled_count += 1
        if filled_count % 500 == 0:
             print(f"      ... {filled_count} gün işlendi ({date_str})")

    conn.close()
    if filled_count > 0:
        print(f"   TOPLAM {filled_count} adet eksik gün (2020-Bugün arası) veritabanına eklendi.")
    else:
        print("   Tüm geçmiş tahminler zaten mevcut, güncel veri ile devam ediliyor.")


# --- HIZLI VERSİYON: SADECE SON N GÜNÜ DOLDUR ---
def fill_recent_predictions(ticker, df, model, scaler, features, target_idx, lookback, days=10):
    """
    Sadece son 'days' günü dolduran hızlı versiyon. GIF animasyonu için kullanılır.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Son N iş gününü al
    valid_dates = df.index[-days:]
    
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT analiz_tarihi FROM tahminler WHERE sembol=?", (ticker,))
    existing_dates = set(row[0] for row in cursor.fetchall())
    
    filled_count = 0
    print(f"   -> Son {days} gün için tahmin geçmişi hazırlanıyor (GIF için)...")
    
    for current_date in valid_dates:
        date_str = current_date.strftime('%Y-%m-%d')
        
        if date_str in existing_dates:
            continue
        
        historical_df = df[df.index <= current_date]
        if len(historical_df) < lookback + 10:
            continue
        
        data = historical_df[features].values
        full_data_scaled = scaler.transform(data)
        last_batch = full_data_scaled[-lookback:].reshape(1, lookback, len(features))
        
        future_prices = []
        curr_p = historical_df['Close'].iloc[-1]
        temp_batch = last_batch.copy()
        
        SMOOTH_FACTOR_HIST = 0.0
        recent_trend = np.mean(temp_batch[0, -10:, target_idx])
        
        for i in range(7):
            raw_pred = model.predict(temp_batch, verbose=0)[0, 0]
            p_sc = (raw_pred * (1 - SMOOTH_FACTOR_HIST)) + (recent_trend * SMOOTH_FACTOR_HIST)
            d = np.zeros((1, len(features)))
            d[0, target_idx] = p_sc
            p_ret = scaler.inverse_transform(d)[0, target_idx]
            curr_p = curr_p * np.exp(p_ret)
            future_prices.append(curr_p)
            
            new_row = temp_batch[0, -1, :].copy()
            new_row[target_idx] = p_sc
            temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)
        
        fut_dates = pd.date_range(current_date + timedelta(days=1), periods=7)
        save_predictions_to_sqlite(ticker, fut_dates, future_prices, analysis_date=date_str)
        filled_count += 1
    
    conn.close()
    if filled_count > 0:
        print(f"   {filled_count} günlük tahmin geçmişi eklendi.")


# --- GIF OLUŞTURMA ---
def create_prediction_gif(ticker, current_df, prediction_dates, prediction_prices):
    """Son 10 analiz gününün tahminlerini animasyon (GIF) haline getirir."""
    conn = sqlite3.connect(DB_PATH)
    dates_query = """
        SELECT DISTINCT analiz_tarihi 
        FROM tahminler 
        WHERE sembol = ? 
        ORDER BY analiz_tarihi DESC 
        LIMIT 10 
    """
    try:
        past_dates = pd.read_sql(dates_query, conn, params=(ticker,))
        if past_dates.empty:
            conn.close()
            return None 

        past_dates = past_dates.sort_values('analiz_tarihi')
        unique_dates = past_dates['analiz_tarihi'].tolist()

        today_str = datetime.now().strftime('%Y-%m-%d')
        if today_str not in unique_dates:
            unique_dates.append(today_str)

    except:
        conn.close()
        return None

    frames = []

    y_min = min(current_df['Close'].tail(60).min(), min(prediction_prices)) * 0.98
    y_max = max(current_df['Close'].tail(60).max(), max(prediction_prices)) * 1.02
    x_start = current_df.index[-60]
    x_end = prediction_dates[-1] + timedelta(days=2)

    for date_str in unique_dates:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(current_df.index[-90:], current_df['Close'].tail(90), color='black', alpha=0.3, label='Gerçek Fiyat', linewidth=1.5)

        ax.plot(prediction_dates, prediction_prices, color='blue', linestyle='-', linewidth=2.5, alpha=0.9, label=f"GÜNCEL ({today_str})")

        if date_str != today_str:
            q = "SELECT hedef_tarih, tahmin_fiyati FROM tahminler WHERE sembol = ? AND analiz_tarihi = ?"
            hist_data = pd.read_sql(q, conn, params=(ticker, date_str))

            if not hist_data.empty:
                hist_data['hedef_tarih'] = pd.to_datetime(hist_data['hedef_tarih'])
                ax.plot(hist_data['hedef_tarih'], hist_data['tahmin_fiyati'], 
                        color='red', marker='o', linestyle='--', linewidth=2, label=f"Eski ({date_str})")

                ax.set_title(f"Tahmin Evrimi: {date_str} vs GÜNCEL", fontsize=14, fontweight='bold', color='#374151')
        else:
             ax.set_title(f"Tahmin Evrimi: {date_str} (Bugün)", fontsize=14, fontweight='bold', color='#374151')

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_start, x_end)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left', fontsize=9)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    conn.close()

    if not frames: return None

    gif_buf = BytesIO()
    imageio.mimsave(gif_buf, frames, format='GIF', duration=800, loop=0)
    return base64.b64encode(gif_buf.getvalue()).decode('utf-8')


# --- SEED SABİTLEME ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

# --- VARLIK LİSTESİ - SADECE INTC ---
etf_map = {
    'INTC': {'name': 'Intel Corporation', 'type': 'us_stock'}
}

# --- MAKRO VERİ (ABD ODAKLI) ---
def get_macro_data():
    end = datetime.now()
    start = end - timedelta(days=12*365)
    # ABD hissesi için ABD odaklı makro veriler
    tickers = {
        "^VIX": "VIX",                # Korku Endeksi
        "^TNX": "US_10Y_BOND",        # ABD 10 Yıllık Tahvil Faizi
        "CL=F": "OIL",                # Ham Petrol
        "DX-Y.NYB": "DXY",            # Dolar Endeksi
        "^GSPC": "SP500",             # S&P 500
        "^IXIC": "NASDAQ",            # NASDAQ Composite
        "SOXX": "SEMI_ETF"            # Yarı İletken ETF'i (Intel'in sektör endeksi)
    }
    try:
        df = yf.download(list(tickers.keys()), start=start, end=end, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns=tickers, inplace=True)
        return df.ffill().bfill()
    except Exception as e:
        print(f"UYARI: Makro veriler indirilemedi ({e}). Analiz makro verisiz devam edecek.")
        return pd.DataFrame()

def get_stock_data(symbol, macro_df):
    end = datetime.now()
    start = end - timedelta(days=12*365)

    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"HATA: {symbol} indirilirken bağlantı hatası oluştu: {e}")
        return None

    if df is None or df.empty: 
        print(f"UYARI: {symbol} için veri bulunamadı veya boş döndü.")
        return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols_to_keep:
        if c not in df.columns: df[c] = df['Close']
    df = df[cols_to_keep].dropna()

    if len(df) < 60:
        print(f"UYARI: {symbol} için yeterli geçmiş veri yok (Sadece {len(df)} gün bulundu). Analiz atlanıyor.")
        return None

    save_to_sqlite(symbol, df)

    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['SMA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill().dropna()
        else:
            df = df.dropna()
        return df
    except Exception as e:
        print(f"HATA: {symbol} için teknik indikatörler hesaplanırken hata oluştu: {e}")
        return None

# --- SENTIMENT FONKSİYONU ---
def get_advanced_sentiment(ticker):
    news_items = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        titles = []
        if news:
            for n in news[:10]:  # INTC için daha fazla haber
                title = n.get('title')
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
        scores = []
        for t in titles:
            analysis = TextBlob(t)
            scores.append(analysis.sentiment.polarity)
        return np.mean(scores) if scores else 0.0, news_items
    except: return 0.0, []

def calculate_alpha_beta(stock_returns, market_returns, risk_free_rate=0.045):
    """ABD için risk-free rate %4.5 (US Treasury 10Y civarı)"""
    common_idx = stock_returns.index.intersection(market_returns.index)
    if len(common_idx) < 30: return 1.0, 0.0
    stock_ret = stock_returns.loc[common_idx].values.flatten()
    mkt_ret = market_returns.loc[common_idx].values.flatten()
    valid_mask = np.isfinite(stock_ret) & np.isfinite(mkt_ret)
    stock_ret = stock_ret[valid_mask]
    mkt_ret = mkt_ret[valid_mask]
    if len(stock_ret) < 30: return 1.0, 0.0
    cov_matrix = np.cov(stock_ret, mkt_ret)
    covariance = cov_matrix[0, 1]
    variance = np.var(mkt_ret)
    if variance == 0: return 1.0, 0.0
    beta = covariance / variance
    rf_daily = (1 + risk_free_rate)**(1/252) - 1
    alpha = np.mean(stock_ret) - (rf_daily + beta * (np.mean(mkt_ret) - rf_daily))
    return beta, alpha * 252 

def generate_signal(current_price, predicted_price, atr_value):
    expected_change = (predicted_price - current_price)
    threshold = atr_value * 0.5 
    if expected_change > threshold: return "GÜÇLÜ AL", "green", "Pozitif Trend"
    elif expected_change > 0: return "AL / TUT", "blue", "Zayıf Yükseliş"
    elif expected_change < -threshold: return "SAT", "red", "Negatif Trend"
    else: return "NÖTR", "gray", "Yatay Seyir"

# --- İSTATİSTİK FONKSİYONLARI ---
def calculate_metrics_extended(prices, predicted_returns, market_returns=None, risk_free_rate=0.045):
    """ABD için risk-free rate %4.5"""
    prices = np.array(prices)
    if len(prices) < 2: return {}
    real_returns = np.diff(np.log(prices))
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak: peak = p
        dd = (peak - p) / peak if peak != 0 else 0
        if dd > max_dd: max_dd = dd
    mdd_pct = max_dd * 100
    avg_ret = np.mean(predicted_returns)
    std_ret = np.std(predicted_returns)
    sharpe = (np.sqrt(252) * (avg_ret - risk_free_rate/252) / std_ret) if std_ret > 0 else 0
    annual_volatility = std_ret * np.sqrt(252) * 100 
    negative_returns = predicted_returns[predicted_returns < 0]
    downside_std = np.std(negative_returns)
    sortino = (np.sqrt(252) * (avg_ret - risk_free_rate/252) / downside_std) if downside_std > 0 else 0
    annualized_return = avg_ret * 252
    calmar = (annualized_return / max_dd) if max_dd > 0 else 0
    return {'mdd': mdd_pct, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'volatility': annual_volatility}

def get_benchmark_data(benchmark_symbol):
    end = datetime.now()
    start = end - timedelta(days=12*365)
    try:
        df = yf.download(benchmark_symbol, start=start, end=end, progress=False, auto_adjust=True)['Close']
        if isinstance(df, pd.DataFrame): df = df.iloc[:, 0] if df.shape[1] > 0 else df
        return np.log(df / df.shift(1)).dropna()
    except: return None

# --- HTML RAPOR ---
class HTMLRapor:
    def __init__(self):
        self.content = """
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <title>INTC - AI Destekli Finansal Öngörü Paneli</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { text-align: center; color: #111827; border-bottom: 3px solid #0071c5; padding-bottom: 15px; }
                .report-section { margin-bottom: 50px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; }
                .header { background: #0071c5; color: white; padding: 15px 25px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; }
                .signal-banner { padding: 15px; text-align: center; font-weight: bold; font-size: 1.2em; letter-spacing: 1px; }
                .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; }
                .stat-box { background: #fff; padding: 15px; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center; }
                .stat-label { font-size: 0.8em; color: #6b7280; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
                .stat-val { font-size: 1.3em; font-weight: 800; color: #111827; }
                .stat-sub { font-size: 0.7em; color: #9ca3af; }
                .chart-area { padding: 20px; text-align: center; background: #f9fafb; }
                .main-chart { width: 100%; border-radius: 8px; border: 1px solid #eee; }
                .mini-charts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; padding: 15px; background: #fff; }
                .mini-charts img { width: 100%; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
                .news-list { padding: 15px; background: #f9fafb; border-top: 1px solid #eee; max-height: 200px; overflow-y: auto; font-size: 0.9em; }
                .news-item { padding: 5px 0; border-bottom: 1px solid #eee; }
                .gif-container { text-align: center; padding: 20px; background: #fff; border-top: 1px solid #eee; }
                .gif-img { width: 100%; border-radius: 8px; border: 1px solid #eee; } 
            </style>
        </head>
        <body><div class="container">
        <h1>INTC - Intel Corporation AI Analiz Paneli</h1>
        <div style="background:#e6f2fa; padding:15px; border-left:4px solid #0071c5; margin-bottom:30px;">
            <strong>Analiz Modülleri:</strong> 1. Realistic AI (7 Günlük) | 2. İstatistik (R², Sortino, Alpha) | 3. Makro Ekonomi (S&P500, NASDAQ, SOXX) | 4. Teknik & Hacim | 5. Tahmin Evrimi (GIF)
        </div>
        """

    def add_section(self, ticker, name, metrics, chart_b64, extra_charts, news_items, gif_b64=None):
        sig_color = metrics['signal_color']
        sig_bg = "#dcfce7" if sig_color == "green" else "#fee2e2" if sig_color == "red" else "#eff6ff"
        currency = "$"  # INTC ABD hissesi - Dolar
        r2_val = metrics['r2_score']
        r2_color = "green" if r2_val > 0.10 else "orange" if r2_val > 0.0 else "red"
        da_val = metrics['dir_acc']
        da_color = "green" if da_val > 55 else "orange" if da_val > 50 else "red"
        potansiyel = (metrics['target_price'] - metrics['current_price']) / metrics['current_price'] * 100
        pot_color = "green" if potansiyel > 0 else "red"

        news_html = ""
        if news_items:
            news_html = "<div class='news-list'><b>Son Haberler:</b><br>" + "".join([f"<div class='news-item'><span style='color:#666'>{i['date']}</span> {i['title']}</div>" for i in news_items]) + "</div>"

        gif_html = ""
        if gif_b64:
            gif_html = f"""
            <div class="gif-container">
                <h3 style="color:#4b5563; margin-bottom:15px; font-size:1.1em;">Tahminlerin Zaman İçindeki Evrimi (Simülasyon)</h3>
                <img class="gif-img" src="data:image/gif;base64,{gif_b64}" alt="Tahmin Animasyonu">
                <div style="font-size:0.8em; color:#666; margin-top:5px;">Not: Bu animasyon modelin son 10 günde fikrini nasıl değiştirdiğini gösterir.</div>
            </div>
            """

        self.content += f"""
        <div class="report-section">
            <div class="header">
                <span>{ticker} | {name}</span>
                <span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:15px; font-size:0.7em;">MEGA MODEL v9.1 - INTC Edition</span>
            </div>
            <div class="signal-banner" style="background:{sig_bg}; color:{sig_color};">
                AI SİNYALİ: {metrics['signal']} <span style="font-size:0.7em; color:#555">({metrics['signal_desc']})</span>
            </div>
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-label">Mevcut Fiyat</div><div class="stat-val">{currency}{metrics['current_price']:.2f}</div><div class="stat-sub">Son Kapanış</div></div>
                <div class="stat-box"><div class="stat-label">Hedef Fiyat (7G)</div><div class="stat-val">{currency}{metrics['target_price']:.2f}</div><div class="stat-sub">AI Tahmini</div></div>
                <div class="stat-box"><div class="stat-label">Potansiyel Getiri</div><div class="stat-val" style="color:{pot_color}">%{potansiyel:.2f}</div><div class="stat-sub">Hedef Farkı</div></div>
                <div class="stat-box"><div class="stat-label">Haber Skoru</div><div class="stat-val" style="color:{'green' if metrics['news_score']>0 else 'red'}">{metrics['news_score']:.2f}</div><div class="stat-sub">Sentiment Analizi</div></div>

                <div class="stat-box"><div class="stat-label">Yön Başarısı</div><div class="stat-val" style="color:{da_color}">%{metrics['dir_acc']:.1f}</div><div class="stat-sub">Yukarı/Aşağı Tahmini</div></div>
                <div class="stat-box"><div class="stat-label">R-Kare ($R^2$)</div><div class="stat-val" style="color:{r2_color}">{metrics['r2_score']:.3f}</div><div class="stat-sub">Model Güvenilirliği</div></div>
                <div class="stat-box"><div class="stat-label">Fiyat Doğruluğu</div><div class="stat-val">%{metrics['price_acc']:.1f}</div><div class="stat-sub">Sapma Oranı</div></div>
                <div class="stat-box"><div class="stat-label">Yıllık Volatilite</div><div class="stat-val">%{metrics['volatility']:.1f}</div><div class="stat-sub">Risk (Oynaklık)</div></div>

                <div class="stat-box"><div class="stat-label">Alpha (Yıllık)</div><div class="stat-val" style="color:{'green' if metrics['alpha']>0 else 'red'}">{metrics['alpha']:.2f}</div><div class="stat-sub">Endeks Üstü Getiri</div></div>
                 <div class="stat-box"><div class="stat-label">Beta ({metrics['bench_name']})</div><div class="stat-val">{metrics['beta']:.2f}</div><div class="stat-sub">Piyasa Hassasiyeti</div></div>
                <div class="stat-box"><div class="stat-label">Sharpe Oranı</div><div class="stat-val">{metrics['sharpe']:.2f}</div><div class="stat-sub">Risk/Getiri Dengesi</div></div>
                <div class="stat-box"><div class="stat-label">Sortino Oranı</div><div class="stat-val">{metrics['sortino']:.2f}</div><div class="stat-sub">Negatif Risk Getirisi</div></div>

                <div class="stat-box"><div class="stat-label">Calmar Oranı</div><div class="stat-val">{metrics['calmar']:.2f}</div><div class="stat-sub">Getiri / Max Risk</div></div>
                <div class="stat-box"><div class="stat-label">Max Drawdown</div><div class="stat-val" style="color:red">%{metrics['mdd']:.1f}</div><div class="stat-sub">En Büyük Düşüş</div></div>
                <div class="stat-box" style="background:#f9fafb;"><div class="stat-label">-</div><div class="stat-val">-</div></div>
                <div class="stat-box" style="background:#f9fafb;"><div class="stat-label">-</div><div class="stat-val">-</div></div>
            </div>
            <div class="chart-area"><img class="main-chart" src="data:image/png;base64,{chart_b64}"></div>
            {gif_html}
            <div class="mini-charts">
                <img src="data:image/png;base64,{extra_charts['rsi']}">
                <img src="data:image/png;base64,{extra_charts['macd']}">
                <img src="data:image/png;base64,{extra_charts['bollinger']}">
                <img src="data:image/png;base64,{extra_charts['reg_channel']}">
                <img src="data:image/png;base64,{extra_charts['drawdown']}">
                <img src="data:image/png;base64,{extra_charts['volatility']}">
                <img src="data:image/png;base64,{extra_charts['volume']}">
                <img src="data:image/png;base64,{extra_charts['ma_cross']}">
                <img src="data:image/png;base64,{extra_charts['heatmap']}">
            </div>
            {news_html}
        </div>
        """

    def save(self):
        self.content += "</div></body></html>"
        with open("INTC_Analiz.html", "w", encoding="utf-8") as f:
            f.write(self.content)

def plot_regression_channel(df):
    data = df['Close'].tail(90).values
    x = np.arange(len(data))
    slope, intercept, _, _, _ = stats.linregress(x, data)
    reg_line = slope * x + intercept
    std = np.std(data - reg_line)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(x, data, color='black', label='Fiyat')
    ax.plot(x, reg_line, color='blue', linestyle='--', label='Trend')
    ax.fill_between(x, reg_line - 2*std, reg_line + 2*std, color='blue', alpha=0.1, label='+-2SD')
    ax.set_title("Regresyon Kanalı", fontsize=10)
    ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_drawdown(df):
    data = df['Close'].tail(180)
    rolling_max = data.cummax()
    drawdown = (data - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(5,3))
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax.set_title("Max Drawdown", fontsize=10)
    ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_volatility_cone(df):
    atr = df['ATR'].tail(60)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(atr.index, atr, color='orange', linewidth=2)
    ax.set_title("Volatilite (ATR)", fontsize=10)
    ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_volume_osc(df):
    vol = df['Volume'].tail(60)
    close = df['Close'].tail(60)
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.bar(vol.index, vol, color='gray', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(close.index, close, color='blue', linewidth=1)
    ax1.set_title("Hacim", fontsize=10)
    ax1.grid(False); ax2.grid(False)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_ma_cross(df):
    data = df.tail(180)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(data.index, data['Close'], color='black', alpha=0.5, linewidth=1)
    ax.plot(data.index, data['SMA50'], color='green', linewidth=1.5)
    ax.plot(data.index, data['SMA200'], color='red', linewidth=1.5)
    ax.set_title("SMA 50/200", fontsize=10)
    ax.grid(alpha=0.2)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_heatmap(df):
    monthly_ret = df['Close'].resample('M').last().pct_change() * 100
    monthly_ret = monthly_ret.to_frame(name='Return')
    monthly_ret['Year'] = monthly_ret.index.year
    monthly_ret['Month'] = monthly_ret.index.month
    pivot = monthly_ret.pivot(index='Year', columns='Month', values='Return').tail(5)
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Mevsimsellik", fontsize=10)
    ax.set_ylabel(''); ax.set_xlabel('')
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

def mini_plot(d, c, t):
    f, a = plt.subplots(figsize=(5,3)); a.plot(d[-60:], color=c); a.set_title(t, fontsize=10); a.grid(alpha=0.2)
    b = BytesIO(); f.savefig(b, format='png', bbox_inches='tight'); b.seek(0); plt.close(f)
    return base64.b64encode(b.read()).decode('utf-8')

# --- BİLİMSEL LOOKBACK OPTİMİZASYONU ---
def estimate_lookback_acf(log_returns, max_lag=500, alpha=0.05):
    """
    ACF (Autocorrelation Function) ile başlangıç lookback tahmini.
    Anlamlı korelasyonun bittiği son lag + güvenlik payı dönülür.
    
    Finansal getiriler genelde düşük otokorelasyona sahiptir, bu yüzden
    ABSOLUTE returns (volatilite kümelenmesi) daha bilgilendirici olur.
    """
    if not HAS_STATSMODELS:
        return None
    
    # Volatilite kümelenmesi için mutlak getiriler (ARCH etkisi)
    abs_returns = np.abs(log_returns.dropna().values)
    
    if len(abs_returns) < max_lag * 2:
        max_lag = len(abs_returns) // 4
    
    try:
        acf_values, confint = acf(abs_returns, nlags=max_lag, alpha=alpha, fft=True)
        # Güven aralığının üst sınırı (anlamlılık eşiği)
        upper_bound = confint[:, 1] - acf_values  # bootstrap CI
        
        # ACF değerinin güven aralığından çıktığı son lag
        significant_lags = np.where(np.abs(acf_values[1:]) > upper_bound[1:])[0]
        
        if len(significant_lags) == 0:
            return 60  # default
        
        # Anlamlı son lag + %20 güvenlik payı
        last_significant = significant_lags[-1] + 1
        suggested = int(last_significant * 1.2)
        return max(30, min(suggested, max_lag))
    except Exception as e:
        print(f"   ACF hesaplama hatası: {e}")
        return None


def find_optimal_lookback(data, features, target_idx, train_split, val_split, scaler,
                           candidate_lookbacks=[60, 120, 250, 500],
                           epochs=20, verbose=True):
    """
    Walk-forward grid search: Aday lookback değerlerini validation setinde dener.
    En düşük val_loss'u veren lookback seçilir.
    
    HIZLANDIRMA: Her aday için sadece 20 epoch eğitilir (tam eğitim değil).
    Bu, hangi lookback'in daha iyi öğrendiğini gösterir.
    """
    if verbose:
        print(f"   -> Optimum lookback aranıyor: {candidate_lookbacks}")
    
    results = {}
    
    for lb in candidate_lookbacks:
        # Yetersiz veri kontrolü
        if train_split <= lb + 50:
            if verbose:
                print(f"      lookback={lb}: yetersiz train verisi, atlandı")
            continue
        
        # Veri hazırlama
        train_raw = data[:train_split]
        train_scaled = scaler.transform(train_raw)
        
        X_tr, y_tr = [], []
        for i in range(lb, len(train_scaled)):
            X_tr.append(train_scaled[i-lb:i])
            y_tr.append(train_scaled[i, target_idx])
        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        
        val_inputs = data[train_split - lb : val_split]
        val_scaled = scaler.transform(val_inputs)
        X_v, y_v = [], []
        for i in range(lb, len(val_scaled)):
            X_v.append(val_scaled[i-lb:i])
            y_v.append(val_scaled[i, target_idx])
        X_v, y_v = np.array(X_v), np.array(y_v)
        
        if len(X_v) < 10:
            continue
        
        # Hızlı eğitim için küçük model
        set_seeds()
        m = Sequential([
            Input(shape=(X_tr.shape[1], X_tr.shape[2])),
            LSTM(32),
            Dense(1)
        ])
        m.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
        
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        
        hist = m.fit(X_tr, y_tr, epochs=epochs, batch_size=64,
                     validation_data=(X_v, y_v), callbacks=[es], verbose=0)
        
        # Val R² hesabı
        pred_v = m.predict(X_v, verbose=0).flatten()
        try:
            r2_v = r2_score(y_v, pred_v)
        except:
            r2_v = -1.0
        
        best_val_loss = min(hist.history['val_loss'])
        results[lb] = {'val_loss': best_val_loss, 'r2': r2_v, 'epochs': len(hist.history['loss'])}
        
        if verbose:
            print(f"      lookback={lb}: val_loss={best_val_loss:.5f} | val_R²={r2_v:.4f} | epochs={len(hist.history['loss'])}")
    
    if not results:
        return 60  # fallback
    
    # En düşük val_loss kazanır (R² yerine loss daha güvenilir, çünkü R² negatif olabilir)
    best_lb = min(results.keys(), key=lambda k: results[k]['val_loss'])
    
    if verbose:
        print(f"   ✅ SEÇİLEN LOOKBACK: {best_lb} gün (val_loss={results[best_lb]['val_loss']:.5f})")
    
    return best_lb


# --- ANALİZ ÇEKİRDEĞİ ---
def analyze_ticker(ticker, info, macro_df):
    set_seeds()
    name = info['name']
    asset_type = info['type']

    # INTC için benchmark: S&P 500 (^GSPC)
    # Alternatif: NASDAQ (^IXIC) veya SOXX (Yarı iletken ETF)
    if asset_type == 'us_stock': 
        bench_symbol, bench_name = '^GSPC', 'S&P 500'
    else: 
        bench_symbol, bench_name = '^GSPC', 'S&P 500'

    market_returns = get_benchmark_data(bench_symbol)
    if market_returns is None: return None

    df = get_stock_data(ticker, macro_df)
    if df is None: return None

    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']
    features = ['Close'] + [c for c in df.columns if c not in exclude_cols]

    target_idx = features.index('Log_Ret')
    data = df[features].values

    train_split = int(len(df) * 0.80)
    val_split = int(len(df) * 0.90)
    train_raw = data[:train_split]

    scaler = MinMaxScaler((0,1))
    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)

    # ============================================================
    # BİLİMSEL LOOKBACK SEÇİMİ
    # 1. ACF ile başlangıç tahmini (volatilite kümelenmesi)
    # 2. Grid search ile validation setinde en iyi lookback bulunur
    # ============================================================
    
    # Adım 1: ACF tabanlı hint
    log_ret_train = df['Log_Ret'].iloc[:train_split]
    acf_hint = estimate_lookback_acf(log_ret_train, max_lag=500)
    if acf_hint:
        print(f"   -> ACF analizi öneriyor: ~{acf_hint} gün")
    
    # Adım 2: Grid search adayları (ACF hint'ini de dahil et)
    candidates = sorted(set([60, 120, 250, 500] + ([acf_hint] if acf_hint else [])))
    # Veriye göre filtrele
    candidates = [c for c in candidates if c < train_split - 100]
    
    lookback = find_optimal_lookback(
        data=data, features=features, target_idx=target_idx,
        train_split=train_split, val_split=val_split, scaler=scaler,
        candidate_lookbacks=candidates,
        epochs=20  # hızlı arama için
    )

    def create_dataset(dataset, lb=60):
        X, Y = [], []
        for i in range(lb, len(dataset)):
            X.append(dataset[i-lb:i])
            Y.append(dataset[i, target_idx])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_scaled, lookback)

    val_inputs = data[train_split - lookback : val_split]
    val_inputs_scaled = scaler.transform(val_inputs)
    X_val, y_val = create_dataset(val_inputs_scaled, lookback)

    test_inputs = data[val_split - lookback :]
    test_inputs_scaled = scaler.transform(test_inputs)
    X_test, y_test = create_dataset(test_inputs_scaled, lookback)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)

    print(f"   -> Model eğitiliyor...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=0)
    
    # =====================================================
    # OVERFIT KONTROLÜ - Train ve Val performansı
    # =====================================================
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    epochs_run = len(history.history['loss'])
    
    # Train R² (in-sample)
    pred_train = model.predict(X_train, verbose=0).flatten()
    try: r2_train = r2_score(y_train, pred_train)
    except: r2_train = 0
    
    # Val R² (out-of-sample, eğitimde "görüldü" ama label olarak kullanılmadı)
    pred_val = model.predict(X_val, verbose=0).flatten()
    try: r2_val_set = r2_score(y_val, pred_val)
    except: r2_val_set = 0
    
    print(f"   -> Eğitim: {epochs_run} epoch | Train Loss: {final_train_loss:.5f} | Val Loss: {final_val_loss:.5f} (best: {best_val_loss:.5f})")
    print(f"   -> Train R²: {r2_train:.4f} | Val R²: {r2_val_set:.4f}")
    if r2_train - r2_val_set > 0.15:
        print(f"   ⚠️  UYARI: Train ve Val R² arasında büyük fark var. Olası overfitting!")
    
    # Veri seti boyutları
    print(f"   -> Veri: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test) if 'X_test' in dir() else 'N/A'} | Lookback={lookback}")
    
    # ============================================================
    # GEÇMİŞ TAHMİN DOLDURMA - HIZ İÇİN KAPATILDI
    # GIF için sadece son 10 günün tahminleri yapılır (hızlı).
    # Tam geçmiş için: ENABLE_FILL_HISTORY = True yap
    # ============================================================
    ENABLE_FILL_HISTORY = False
    if ENABLE_FILL_HISTORY:
        fill_missing_predictions(ticker, df, model, scaler, features, target_idx, lookback)
    else:
        # Sadece son 10 günü doldur - GIF için yeterli
        fill_recent_predictions(ticker, df, model, scaler, features, target_idx, lookback, days=10)

    pred_test_scaled = model.predict(X_test, verbose=0)
    dummy = np.zeros((len(pred_test_scaled), len(features)))
    dummy[:, target_idx] = pred_test_scaled.flatten()
    pred_test_ret = scaler.inverse_transform(dummy)[:, target_idx]

    # ============================================================
    # DOĞRU HİZALAMA: 
    # X_test, val_split-lookback'ten başlar, indekslemesi şöyledir:
    # X_test[i] = data[val_split-lookback+i : val_split+i]
    # y_test[i] = data[val_split+i, target_idx] = günün log_return'i
    # 
    # Yani pred_test_ret[i], val_split+i. günün log return tahmini.
    # actual_prices[i] = df['Close'].iloc[val_split + i]
    # 
    # Log return: log(P[t]/P[t-1]) - yani val_split+i. günün getirisi
    # P[val_split+i-1] -> P[val_split+i] hareketini gösterir.
    # ============================================================
    
    actual_prices = df['Close'].iloc[val_split:].values
    actual_prices_with_prev = df['Close'].iloc[val_split-1:].values  # bir gün öncesi dahil
    min_len = min(len(pred_test_ret), len(actual_prices))
    pred_test_ret = pred_test_ret[:min_len]
    actual_prices = actual_prices[:min_len]

    # GERÇEK log returns: P[val_split+i] / P[val_split+i-1]
    actual_returns_log = np.log(actual_prices_with_prev[1:min_len+1] / actual_prices_with_prev[:min_len])
    
    # R2 - tahmin edilen ve gerçek log return aynı günü temsil eder (DOĞRU HİZALAMA)
    try: r2_val = r2_score(actual_returns_log, pred_test_ret)
    except: r2_val = 0

    # ============================================================
    # KRİTİK: Dürüst test metriği için TEACHER FORCING YOK!
    # Gerçek koşulda model sadece tahminlerine güvenir.
    # rec_prices: önceki TAHMİN edilen fiyattan başlayarak ileri yürür
    # ============================================================
    rec_prices = []
    prev_price = df['Close'].iloc[val_split-1]  # Sadece başlangıç noktası gerçek
    for i in range(min_len):
        prev_price = prev_price * np.exp(pred_test_ret[i])  # ÖNCEKİ TAHMİN kullanılır
        rec_prices.append(prev_price)
    rec_prices = np.array(rec_prices)

    # 1-ADIM ÖNCELİK YÖN DOĞRULUĞU: 
    # Tahmin edilen günlük getirinin yönü, gerçek günlük getirinin yönüyle eşleşiyor mu?
    dir_acc = (np.sum(np.sign(pred_test_ret) == np.sign(actual_returns_log)) / len(actual_returns_log)) * 100 if len(actual_returns_log)>0 else 0
    
    # 1-ADIM TEACHER FORCING ile fiyat doğruluğu (referans için)
    rec_prices_tf = []
    for i in range(min_len):
        prev = df['Close'].iloc[val_split-1] if i == 0 else actual_prices[i-1]
        rec_prices_tf.append(prev * np.exp(pred_test_ret[i]))
    rec_prices_tf = np.array(rec_prices_tf)
    price_acc = 100 - (mean_absolute_percentage_error(actual_prices, rec_prices_tf) * 100)

    stock_ret_series = df['Log_Ret']
    beta, alpha = calculate_alpha_beta(stock_ret_series, market_returns)
    extended_metrics = calculate_metrics_extended(rec_prices, pred_test_ret)

    # ============================================================
    # 7 GÜNLÜK GELECEK TAHMİNİ (Rekürsif - saf model çıktısı)
    # SMOOTH_FACTOR=0 -> Saf model tahmini (dürüst)
    # SMOOTH_FACTOR=0.3 -> Trendle yumuşatılmış (daha tutucu, ama metriğe sızabilir)
    # ============================================================
    SMOOTH_FACTOR = 0.0  # Dürüst tahmin için 0, daha tutucu için 0.3
    
    full_data_scaled = scaler.transform(data)
    last_batch = full_data_scaled[-lookback:].reshape(1, lookback, len(features))
    future_prices = []
    curr_p = df['Close'].iloc[-1]

    temp_batch = last_batch.copy()
    recent_trend = np.mean(temp_batch[0, -10:, target_idx])

    for i in range(7):
        raw_pred = model.predict(temp_batch, verbose=0)[0,0]
        # Yumuşatma faktörü 0 ise saf model tahmini kullanılır
        p_sc = (raw_pred * (1 - SMOOTH_FACTOR)) + (recent_trend * SMOOTH_FACTOR)
        d = np.zeros((1, len(features)))
        d[0, target_idx] = p_sc
        p_ret = scaler.inverse_transform(d)[0, target_idx]

        curr_p = curr_p * np.exp(p_ret)
        future_prices.append(curr_p)

        new_row = temp_batch[0, -1, :].copy()
        new_row[target_idx] = p_sc 
        temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1,1,len(features)), axis=1)

    target_price = future_prices[-1]
    current_atr = df['ATR'].iloc[-1]
    signal, sig_color, sig_desc = generate_signal(df['Close'].iloc[-1], target_price, current_atr * 3)

    fut_dates = pd.date_range(df.index[-1]+timedelta(days=1), periods=7)
    save_predictions_to_sqlite(ticker, fut_dates, future_prices)

    gif_b64 = create_prediction_gif(ticker, df, fut_dates, future_prices)

    fig, ax = plt.subplots(figsize=(12, 6))
    test_dates = df.index[val_split:][:min_len]
    ax.plot(df.index[-150:], df['Close'].iloc[-150:], label="Gerçek Fiyat", color="#1f2937")
    ax.plot(test_dates, rec_prices, label="Model Testi (OOS)", linestyle="--", color="#f59e0b")

    ax.plot(fut_dates, future_prices, 
            label="7 Günlük Tahmin", 
            color=sig_color, 
            linewidth=2, 
            marker='o',      
            markersize=4,            
            markerfacecolor=sig_color, 
            markeredgewidth=0,
            zorder=10)

    ax.set_title(f"{name} ({ticker}) Fiyat Tahmini")
    ax.legend(); ax.grid(True, alpha=0.2)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    extras = {
        'rsi': mini_plot(df['RSI'], 'purple', 'RSI'),
        'macd': mini_plot(df['MACD'], 'blue', 'MACD'),
        'bollinger': mini_plot(df['SEMI_ETF'] if 'SEMI_ETF' in df.columns else df['SP500'], 'darkblue', 'SOXX (Yarı İletken ETF)'),
        'reg_channel': plot_regression_channel(df),
        'drawdown': plot_drawdown(df),
        'volatility': plot_volatility_cone(df),
        'volume': plot_volume_osc(df),
        'ma_cross': plot_ma_cross(df),
        'heatmap': plot_heatmap(df)
    }

    metrics = {
        'current_price': df['Close'].iloc[-1], 'target_price': target_price,
        'signal': signal, 'signal_color': sig_color, 'signal_desc': sig_desc,
        'alpha': alpha, 'beta': beta, 'bench_name': bench_name,
        'price_acc': price_acc if price_acc > 0 else 0,
        'dir_acc': dir_acc, 
        'mdd': extended_metrics['mdd'], 
        'sharpe': extended_metrics['sharpe'],
        'sortino': extended_metrics['sortino'],
        'calmar': extended_metrics['calmar'],
        'volatility': extended_metrics['volatility'],
        'r2_score': r2_val,
        'lookback': lookback
    }
    return metrics, chart_b64, extras, gif_b64

# --- MAIN ---
def main():
    report = HTMLRapor()
    init_db()

    print("="*60)
    print("INTC (Intel Corporation) AI ANALİZ PANELİ v9.1")
    print("="*60)
    print("1. Makro Veriler İndiriliyor (S&P500, NASDAQ, SOXX, VIX, Petrol, Tahvil)...")
    macro_df = get_macro_data()

    for ticker, info in etf_map.items():
        print(f"\n>> {ticker} ({info['name']}) Analiz Ediliyor...")
        news_score, news_items = get_advanced_sentiment(ticker)
        result = analyze_ticker(ticker, info, macro_df)

        if result:
            metrics, chart, extras, gif_b64 = result
            metrics['news_score'] = news_score
            report.add_section(ticker, info['name'], metrics, chart, extras, news_items, gif_b64)
            print(f"\n   {'='*50}")
            print(f"   ANALİZ TAMAMLANDI:")
            print(f"   {'='*50}")
            print(f"   Mevcut Fiyat   : ${metrics['current_price']:.2f}")
            print(f"   Hedef Fiyat 7G : ${metrics['target_price']:.2f}")
            print(f"   Potansiyel     : %{((metrics['target_price']-metrics['current_price'])/metrics['current_price']*100):+.2f}")
            print(f"   Sinyal         : {metrics['signal']} ({metrics['signal_desc']})")
            print(f"   Yön Başarısı   : %{metrics['dir_acc']:.1f}")
            print(f"   R-Kare         : {metrics['r2_score']:.3f}")
            print(f"   Sharpe         : {metrics['sharpe']:.2f}")
            print(f"   Sortino        : {metrics['sortino']:.2f}")
            print(f"   Alpha (yıllık) : {metrics['alpha']:.2f}")
            print(f"   Beta vs SP500  : {metrics['beta']:.2f}")
            print(f"   Volatilite     : %{metrics['volatility']:.1f}")
            print(f"   Max Drawdown   : %{metrics['mdd']:.1f}")
            print(f"   Haber Skoru    : {metrics['news_score']:+.2f}")
            print(f"   Lookback (Bil.): {metrics['lookback']} gün (geriye bakma penceresi)")

    report.save()
    print("\n" + "="*60)
    print("ANALİZ BİTTİ. Rapor: INTC_Analiz.html")
    print("="*60)

if __name__ == "__main__":
    main()
