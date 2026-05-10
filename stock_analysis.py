# --- OTOMATİK KURULUM VE BAĞLILIK KONTROLÜ (AUTO-INSTALLER v3) ---
import sys
import subprocess
import importlib
import sqlite3 

def install_package(package):
    print(f"OTOMATİK KURULUM: '{package}' yükleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir"])

# SADECE GEREKLİ KÜTÜPHANELER
required_packages = ['tf-keras', 'ta', 'yfinance', 'GoogleNews', 'textblob', 'scipy', 'seaborn', 'sklearn', 'imageio']
for package in required_packages:
    try: importlib.import_module(package.replace('-', '_'))
    except ImportError:
        try: install_package(package)
        except: pass

# --- STANDART İMPORTLAR ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
DB_NAME = "data_gunluk.db"
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

    # Eğer analiz tarihi dışarıdan verilmediyse bugünü al
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

    # Başlangıç tarihini belirle (2020-01-01)
    start_date = pd.Timestamp("2020-01-01")

    # Veri setindeki tarihlerin bu tarihten sonraki kısmını al
    valid_dates = df[df.index >= start_date].index

    filled_count = 0
    print(f"   -> Geçmiş analizler kontrol ediliyor (Başlangıç: {start_date.strftime('%Y-%m-%d')})...")

    # Performans optimizasyonu: Mevcut tahmin tarihlerini hafızaya al
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT analiz_tarihi FROM tahminler WHERE sembol=?", (ticker,))
    existing_dates = set(row[0] for row in cursor.fetchall())

    # Tüm geçerli tarihleri döngüye al
    for current_date in valid_dates:
        date_str = current_date.strftime('%Y-%m-%d')

        # Eğer bu tarih veritabanında zaten varsa atla
        if date_str in existing_dates:
            continue

        # O tarihteki veriyi simüle et (Lookback kadar geriye gidebilmeliyiz)
        historical_df = df[df.index <= current_date]

        # Yeterli veri yoksa (Lookback + 10 gün tampon) atla
        if len(historical_df) < lookback + 10: 
            continue

        # Tahmin için veriyi hazırla
        # Sadece son 'lookback' kadar veriyi alıp modele sokacağız
        data = historical_df[features].values
        # Ölçeklendirme (Tüm veri seti üzerinde eğitilmiş scaler kullanılıyor)
        full_data_scaled = scaler.transform(data)

        # Son pencereyi al
        last_batch = full_data_scaled[-lookback:].reshape(1, lookback, len(features))

        future_prices = []
        curr_p = historical_df['Close'].iloc[-1]

        # Tahmin döngüsü için batch kopyası
        temp_batch = last_batch.copy()

        # Trend faktörü (Son 10 gün ortalaması)
        recent_trend = np.mean(temp_batch[0, -10:, target_idx])

        # 7 Günlük tahmin üret
        for i in range(7):
            raw_pred = model.predict(temp_batch, verbose=0)[0,0]
            # Model çıktısını biraz yumuşat
            p_sc = (raw_pred * 0.7) + (recent_trend * 0.3)

            d = np.zeros((1, len(features)))
            d[0, target_idx] = p_sc

            # Geri dönüştür (Inverse Scale)
            p_ret = scaler.inverse_transform(d)[0, target_idx]

            # Fiyatı hesapla
            curr_p = curr_p * np.exp(p_ret)
            future_prices.append(curr_p)

            # Batch güncelle (Kayan pencere)
            new_row = temp_batch[0, -1, :].copy()
            new_row[target_idx] = p_sc 
            temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1,1,len(features)), axis=1)

        # Veritabanına kaydet
        # Hedef tarihler: Analiz tarihinden sonraki 7 gün
        fut_dates = pd.date_range(current_date + timedelta(days=1), periods=7)
        save_predictions_to_sqlite(ticker, fut_dates, future_prices, analysis_date=date_str)

        filled_count += 1
        # Kullanıcıya işlem hakkında bilgi ver (her 500 günde bir)
        if filled_count % 500 == 0:
             print(f"      ... {filled_count} gün işlendi ({date_str})")

    conn.close()
    if filled_count > 0:
        print(f"   TOPLAM {filled_count} adet eksik gün (2020-Bugün arası) veritabanına eklendi.")
    else:
        print("   Tüm geçmiş tahminler zaten mevcut, güncel veri ile devam ediliyor.")


# --- GIF OLUŞTURMA (BÜYÜK BOYUT) ---
def create_prediction_gif(ticker, current_df, prediction_dates, prediction_prices):
    """Son 7 analiz gününün tahminlerini animasyon (GIF) haline getirir."""
    conn = sqlite3.connect(DB_PATH)
    # Sorguyu genişlettik, daha fazla geçmiş tahmin görelim
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

    # Eksen Ayarları
    y_min = min(current_df['Close'].tail(60).min(), min(prediction_prices)) * 0.98
    y_max = max(current_df['Close'].tail(60).max(), max(prediction_prices)) * 1.02
    x_start = current_df.index[-60]
    x_end = prediction_dates[-1] + timedelta(days=2)

    for date_str in unique_dates:
        fig, ax = plt.subplots(figsize=(12, 6))

        # 1. Arka plan: Gerçek Fiyat
        ax.plot(current_df.index[-90:], current_df['Close'].tail(90), color='black', alpha=0.3, label='Gerçek Fiyat', linewidth=1.5)

        # 2. SABİT GÜNCEL TAHMİN (Mavi Çizgi)
        ax.plot(prediction_dates, prediction_prices, color='blue', linestyle='-', linewidth=2.5, alpha=0.9, label=f"GÜNCEL ({today_str})")

        # 3. GEÇMİŞ TAHMİN (Kırmızı Çizgi)
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
    imageio.mimsave(gif_buf, frames, format='GIF', duration=800, loop=0) # Daha akıcı olması için süreyi kısalttık
    return base64.b64encode(gif_buf.getvalue()).decode('utf-8')


# --- SEED SABİTLEME ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

# --- VARLIK LİSTESİ ---
etf_map = {
    'USDTRY=X': {'name': 'USD/TRY', 'type': 'forex'},
    'EURTRY=X': {'name': 'EUR/TRY', 'type': 'forex'},
    'GC=F':     {'name': 'Gram Altın (TL)', 'type': 'commodity'}, 
    'SI=F':     {'name': 'Gümüş', 'type': 'commodity'}, 
    'THYAO.IS': {'name': 'Türk Hava Yolları', 'type': 'stock'},
    'KCHOL.IS': {'name': 'Koç Holding', 'type': 'stock'},
    'AKBNK.IS': {'name': 'Akbank', 'type': 'stock'},
    'ASELS.IS': {'name': 'Aselsan', 'type': 'stock'},
    'BTC-USD':  {'name': 'Bitcoin', 'type': 'crypto'},
    'ETH-USD':  {'name': 'Ethereum', 'type': 'crypto'}
}

# --- MAKRO VERİ ---
def get_macro_data():
    end = datetime.now()
    start = end - timedelta(days=12*365)
    tickers = {
        "TRY=X": "USD_TRY",
        "^VIX": "VIX",
        "^TNX": "US_10Y_BOND", 
        "CL=F": "OIL",         
        "DX-Y.NYB": "DXY"      
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

    if symbol == 'GC=F':
        try:
            gold_df = yf.download('GC=F', start=start, end=end, progress=False, auto_adjust=True)
            usd_df = yf.download('USDTRY=X', start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(gold_df.columns, pd.MultiIndex): gold_df.columns = gold_df.columns.get_level_values(0)
            if isinstance(usd_df.columns, pd.MultiIndex): usd_df.columns = usd_df.columns.get_level_values(0)
            merged = pd.DataFrame(index=gold_df.index)
            merged['Gold_Close'] = gold_df['Close']
            merged['USD_Close'] = usd_df['Close']
            merged = merged.dropna()
            df = pd.DataFrame(index=merged.index)
            df['Close'] = (merged['Gold_Close'] * merged['USD_Close']) / 31.1035
            df['Open'] = df['Close']; df['High'] = df['Close']; df['Low'] = df['Close']; df['Volume'] = 1000000 
        except Exception as e:
            print(f"HATA: Gram Altın hesaplanamadı -> {e}")
            return None
    else:
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
        search_ticker = 'GC=F' if ticker == 'GC=F' else ticker
        stock = yf.Ticker(search_ticker)
        news = stock.news
        titles = []
        if news:
            for n in news[:5]:
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
                if ticker == 'GC=F': search_term = "Gold Price Turkey News"
                else: search_term = ticker.replace('.IS', '') + " stock news"
                googlenews.search(search_term)
                results = googlenews.result()[:5]
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

def calculate_alpha_beta(stock_returns, market_returns, risk_free_rate=0.30):
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
def calculate_metrics_extended(prices, predicted_returns, market_returns=None, risk_free_rate=0.30):
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
    # GÜNCELLEME: Benchmark (XU100 vb) verisi için de 12 yıl geriye gidiyoruz.
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
            <title>AI Destekli Finansal Öngörü Paneli</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { text-align: center; color: #111827; border-bottom: 3px solid #6366f1; padding-bottom: 15px; }
                .report-section { margin-bottom: 50px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; }
                .header { background: #6366f1; color: white; padding: 15px 25px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; }
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
        <h1>AI Destekli Finansal Öngörü Paneli</h1>
        <div style="background:#eef2ff; padding:15px; border-left:4px solid #6366f1; margin-bottom:30px;">
            <strong>Analiz Modülleri:</strong> 1. Realistic AI (7 Günlük) | 2. İstatistik (R², Sortino, Alpha) | 3. Makro Ekonomi | 4. Teknik & Hacim | 5. Tahmin Evrimi (GIF)
        </div>
        """

    def add_section(self, ticker, name, metrics, chart_b64, extra_charts, news_items, gif_b64=None):
        sig_color = metrics['signal_color']
        sig_bg = "#dcfce7" if sig_color == "green" else "#fee2e2" if sig_color == "red" else "#eff6ff"
        currency = "₺" if ".IS" in ticker or ticker=="GC=F" else "$"
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
                <div style="font-size:0.8em; color:#666; margin-top:5px;">Not: Bu animasyon modelin son 7 günde fikrini nasıl değiştirdiğini gösterir.</div>
            </div>
            """

        self.content += f"""
        <div class="report-section">
            <div class="header">
                <span>{ticker} | {name}</span>
                <span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:15px; font-size:0.7em;">MEGA MODEL v9.1</span>
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
        with open("Hisse_Doviz_Kripto_Analiz.html", "w", encoding="utf-8") as f:
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

# --- ANALİZ ÇEKİRDEĞİ ---
def analyze_ticker(ticker, info, macro_df):
    set_seeds()
    name = info['name']
    asset_type = info['type']

    if asset_type == 'stock': bench_symbol, bench_name = 'XU100.IS', 'BIST 100'
    elif asset_type == 'crypto': bench_symbol, bench_name = 'BTC-USD', 'BTC'
    elif asset_type == 'forex': bench_symbol, bench_name = 'DX-Y.NYB', 'DXY'
    elif asset_type == 'commodity': bench_symbol, bench_name = 'GC=F', 'Gold Futures'
    else: bench_symbol, bench_name = 'XU100.IS', 'BIST 100'

    market_returns = get_benchmark_data(bench_symbol)
    if market_returns is None: return None

    df = get_stock_data(ticker, macro_df)
    if df is None: return None

    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close', 'Gold_Close', 'USD_Close']
    features = ['Close'] + [c for c in df.columns if c not in exclude_cols]

    target_idx = features.index('Log_Ret')
    data = df[features].values

    train_split = int(len(df) * 0.80)
    val_split = int(len(df) * 0.90)
    train_raw = data[:train_split]

    scaler = MinMaxScaler((0,1))
    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)

    lookback = 60

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

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=0)

    # --- GEÇMİŞ GÜNLERİ TAMAMLA (2020'TEN BERİ) ---
    fill_missing_predictions(ticker, df, model, scaler, features, target_idx, lookback)
    # ---------------------------------------------

    pred_test_scaled = model.predict(X_test, verbose=0)
    dummy = np.zeros((len(pred_test_scaled), len(features)))
    dummy[:, target_idx] = pred_test_scaled.flatten()
    pred_test_ret = scaler.inverse_transform(dummy)[:, target_idx]

    actual_prices = df['Close'].iloc[val_split:].values
    min_len = min(len(pred_test_ret), len(actual_prices))
    pred_test_ret = pred_test_ret[:min_len]
    actual_prices = actual_prices[:min_len]

    actual_returns_log = np.diff(np.log(actual_prices))
    if len(pred_test_ret) > len(actual_returns_log):
        pred_test_ret_aligned = pred_test_ret[:len(actual_returns_log)]
    else:
        pred_test_ret_aligned = pred_test_ret

    try: r2_val = r2_score(actual_returns_log, pred_test_ret_aligned)
    except: r2_val = 0

    rec_prices = []
    initial_price = df['Close'].iloc[val_split-1]

    for i in range(min_len):
        prev = initial_price if i == 0 else actual_prices[i-1]
        rec_prices.append(prev * np.exp(pred_test_ret[i]))
    rec_prices = np.array(rec_prices)

    price_acc = 100 - (mean_absolute_percentage_error(actual_prices, rec_prices) * 100)
    dir_acc = (np.sum(np.sign(np.diff(actual_prices)) == np.sign(np.diff(rec_prices))) / (len(actual_prices)-1)) * 100 if len(actual_prices)>1 else 0

    stock_ret_series = df['Log_Ret']
    beta, alpha = calculate_alpha_beta(stock_ret_series, market_returns)
    extended_metrics = calculate_metrics_extended(rec_prices, pred_test_ret)

    full_data_scaled = scaler.transform(data)
    last_batch = full_data_scaled[-lookback:].reshape(1, lookback, len(features))
    future_prices = []
    curr_p = df['Close'].iloc[-1]

    temp_batch = last_batch.copy()

    recent_trend = np.mean(temp_batch[0, -10:, target_idx])

    for i in range(7):
        raw_pred = model.predict(temp_batch, verbose=0)[0,0]
        p_sc = (raw_pred * 0.7) + (recent_trend * 0.3)
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

    # --- GÜNCEL TAHMİNİ KAYDET (Bugün) ---
    fut_dates = pd.date_range(df.index[-1]+timedelta(days=1), periods=7)
    save_predictions_to_sqlite(ticker, fut_dates, future_prices)
    # -------------------------------

    # --- GIF OLUŞTUR ---
    gif_b64 = create_prediction_gif(ticker, df, fut_dates, future_prices)
    # ---------------------------------

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

    ax.set_title(f"{name} Fiyat Tahmini (Enflasyon Destekli)")
    ax.legend(); ax.grid(True, alpha=0.2)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    extras = {
        'rsi': mini_plot(df['RSI'], 'purple', 'RSI'),
        'macd': mini_plot(df['MACD'], 'blue', 'MACD'),
        'bollinger': mini_plot(df['US_10Y_BOND'], 'brown', 'ABD 10Y Tahvil (Faiz)'),
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
        'r2_score': r2_val
    }
    return metrics, chart_b64, extras, gif_b64

# --- MAIN ---
def main():
    report = HTMLRapor()
    init_db()

    print("AI DESTEKLİ FİNANSAL ÖNGÖRÜ PANELİ (v9.1 - Backtest View)...")
    print("1. Makro Veriler İndiriliyor (Enflasyon, Petrol, Tahvil)...")
    macro_df = get_macro_data()

    for ticker, info in etf_map.items():
        print(f"\n>> {ticker} ({info['name']}) Analiz Ediliyor...")
        news_score, news_items = get_advanced_sentiment(ticker)
        result = analyze_ticker(ticker, info, macro_df)

        if result:
            metrics, chart, extras, gif_b64 = result
            metrics['news_score'] = news_score
            report.add_section(ticker, info['name'], metrics, chart, extras, news_items, gif_b64)
            print(f"   TAMAMLANDI -> Sinyal: {metrics['signal']} | Yön:%{metrics['dir_acc']:.1f} | R2:{metrics['r2_score']:.2f} | Sharpe:{metrics['sharpe']:.1f} | Sortino:{metrics['sortino']:.1f} | Alpha:{metrics['alpha']:.2f}")

    report.save()
    print("\n" + "="*50)
    print("ANALİZ BİTTİ. Rapor: Hisse_Doviz_Kripto_Analiz.html")

if __name__ == "__main__":
    main()
