# --- UYARILARI GİZLEME VE KURULUM KONTROLÜ ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Gerekli kütüphaneleri kontrol et
try: import ta
except ImportError: print("UYARI: 'ta' kütüphanesi bulunamadı. pip install ta yapınız.")
try: import yfinance as yf
except ImportError: print("UYARI: 'yfinance' kütüphanesi bulunamadı. pip install yfinance yapınız.")
try: from GoogleNews import GoogleNews
except ImportError: print("UYARI: 'GoogleNews' kütüphanesi bulunamadı. pip install GoogleNews yapınız.")
try: from textblob import TextBlob
except ImportError: print("UYARI: 'textblob' kütüphanesi bulunamadı. pip install textblob yapınız.")
try: from deep_translator import GoogleTranslator
except ImportError: print("UYARI: 'deep_translator' bulunamadı. pip install deep-translator yapınız.")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from datetime import datetime, timedelta

# --- SCIENTIFIC IMPORTS ---
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# --- VARLIK LİSTESİ ---
#etf_map = {
#    'XU100.IS': 'BIST 100 Endeksi',
#    'THYAO.IS': 'Türk Hava Yolları',
#    'ASELS.IS': 'Aselsan',
#    'KCHOL.IS': 'Koç Holding',
#    'AKBNK.IS': 'Akbank',
#    'TUPRS.IS': 'Tüpraş',
#    'SISE.IS':  'Şişecam'
#}

etf_map = {
    'SI=F': 'Silver',
    'GC=F' : 'Gold',
    'BTC-USD' : 'Bitcoin',
    'ETH-USD': 'Etherum',
}

# --- FİNANSAL HESAPLAMALAR (ALPHA, BETA, SİNYAL) ---
def calculate_alpha_beta(stock_returns, market_returns, risk_free_rate=0.30):
    """CAPM Modeli ile Alpha ve Beta hesaplar."""
    # Tarihleri hizala (Intersection)
    common_idx = stock_returns.index.intersection(market_returns.index)

    if len(common_idx) < 30: 
        return 1.0, 0.0

    # Verileri seç ve Numpy array'e çevirip düzleştir (flatten)
    stock_ret = stock_returns.loc[common_idx].values.flatten()
    mkt_ret = market_returns.loc[common_idx].values.flatten()

    # Hata Kontrolü: NaN veya Sonsuz değerler varsa temizle
    valid_mask = np.isfinite(stock_ret) & np.isfinite(mkt_ret)
    stock_ret = stock_ret[valid_mask]
    mkt_ret = mkt_ret[valid_mask]

    if len(stock_ret) < 30: return 1.0, 0.0

    # Kovaryans ve Varyans
    # np.cov 2x2 matris döndürür: [[var(a), cov(a,b)], [cov(b,a), var(b)]]
    cov_matrix = np.cov(stock_ret, mkt_ret)
    covariance = cov_matrix[0, 1]
    variance = np.var(mkt_ret)

    if variance == 0: return 1.0, 0.0

    beta = covariance / variance

    # Yıllık Alpha
    rf_daily = (1 + risk_free_rate)**(1/252) - 1
    alpha = np.mean(stock_ret) - (rf_daily + beta * (np.mean(mkt_ret) - rf_daily))

    return beta, alpha * 252 

def generate_signal(current_price, predicted_price, atr_value):
    """Volatilite (ATR) tabanlı Sinyal Üreticisi"""
    expected_change = (predicted_price - current_price)
    threshold = atr_value * 0.5 # ATR'nin yarısı kadar hareket bekleniyorsa sinyal ver

    if expected_change > threshold:
        return "GÜÇLÜ AL", "green", "Pozitif Trend Beklentisi"
    elif expected_change > 0:
        return "AL / TUT", "blue", "Zayıf Yükseliş"
    elif expected_change < -threshold:
        return "SAT", "red", "Negatif Trend Beklentisi"
    else:
        return "NÖTR", "gray", "Yatay Seyir"

# --- DİĞER METRİKLER ---
def calculate_max_drawdown(prices):
    prices = np.array(prices)
    if len(prices) == 0: return 0
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak: peak = p
        dd = (peak - p) / peak if peak != 0 else 0
        if dd > max_dd: max_dd = dd
    return max_dd * 100

def calculate_sharpe(returns, rf=0.0):
    if len(returns) < 2 or np.std(returns) == 0: return 0
    return np.sqrt(252) * np.mean(returns - rf/252) / np.std(returns)

def calculate_theil(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = np.sqrt(np.mean((y_pred - y_true)**2))
    den = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    return num / den if den != 0 else 0

# --- VERİ ÇEKME FONKSİYONLARI ---
def get_benchmark_data():
    """BIST 100 Endeks verisini çeker (Beta hesabı için)"""
    end = datetime.now()
    start = end - timedelta(days=5*365)
    # auto_adjust=True fiyatları split/temettüye göre düzeltir
    df = yf.download('XU100.IS', start=start, end=end, progress=False, auto_adjust=True)['Close']
    if isinstance(df, pd.DataFrame): 
        # Bazen yfinance DataFrame bazen Series dönebilir, garantiye alalım
        df = df.iloc[:, 0] if df.shape[1] > 0 else df

    returns = np.log(df / df.shift(1)).dropna()
    return returns

def get_macro_data():
    end = datetime.now()
    start = end - timedelta(days=5*365)
    tickers = ["TRY=X", "^VIX"]
    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.rename(columns={'TRY=X': 'USD_TRY', '^VIX': 'VIX'}, inplace=True)
    return df.fillna(method='ffill')

def get_stock_data(symbol, macro_df):
    end = datetime.now()
    start = end - timedelta(days=5*365)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Teknik İndikatörler
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Feature Engineering
    df['Log_Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Log_Ret_Lag2'] = df['Log_Ret'].shift(2)

    df = df.join(macro_df, how='left').fillna(method='ffill').dropna()
    return df

# --- GOOGLE NEWS ---
def get_news_sentiment(symbol):
    try:
        googlenews = GoogleNews(lang='tr', region='TR')
        googlenews.set_period('10d')
        q = f"{symbol.replace('.IS','')} hisse" if ".IS" in symbol else symbol
        googlenews.search(q)
        results = googlenews.result()[:10]

        scores = []
        items = []
        translator = GoogleTranslator(source='auto', target='en')

        for item in results:
            title = item['title']
            if len(title) > 5:
                try:
                    en_title = translator.translate(title)
                    scores.append(TextBlob(en_title).sentiment.polarity)
                    items.append({'date': item['date'], 'title': title})
                except: pass

        return (np.mean(scores) if scores else 0.0), items
    except: return 0.0, []

# --- HTML RAPOR ---
class HTMLRapor:
    def __init__(self):
        self.content = """
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <title>BİST Mega Analiz Raporu</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #f3f4f6; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { text-align: center; color: #111827; border-bottom: 3px solid #6366f1; padding-bottom: 15px; }
                .report-section { margin-bottom: 50px; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #fff; }
                .header { background: #6366f1; color: white; padding: 15px 25px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; }

                .signal-banner { padding: 15px; text-align: center; font-weight: bold; font-size: 1.2em; letter-spacing: 1px; }

                .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #e5e7eb; }
                .stat-box { background: #fff; padding: 15px; text-align: center; height: 100px; display: flex; flex-direction: column; justify-content: center; }
                .stat-label { font-size: 0.8em; color: #6b7280; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
                .stat-val { font-size: 1.3em; font-weight: 800; color: #111827; }
                .stat-sub { font-size: 0.7em; color: #9ca3af; }

                .chart-area { padding: 20px; text-align: center; background: #f9fafb; }
                .main-chart { width: 100%; border-radius: 8px; border: 1px solid #eee; }

                .mini-charts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; padding: 15px; background: #fff; }
                .mini-charts img { width: 100%; border: 1px solid #eee; border-radius: 4px; }

                .news-list { padding: 15px; background: #f9fafb; border-top: 1px solid #eee; max-height: 200px; overflow-y: auto; font-size: 0.9em; }
                .news-item { padding: 5px 0; border-bottom: 1px solid #eee; }
            </style>
        </head>
        <body><div class="container">
        <h1>BİST MEGA ANALİZ: AI + FINANSAL GEÇERLİLİK</h1>
        <div style="background:#eef2ff; padding:15px; border-left:4px solid #6366f1; margin-bottom:30px;">
            <strong>Analiz Modülleri:</strong> 1. Bi-LSTM Fiyat Tahmini | 2. Alpha/Beta Risk Analizi | 3. Volatilite Bazlı Sinyal | 4. Haber Duygu Analizi
        </div>
        """

    def add_section(self, ticker, name, metrics, chart_b64, extra_charts, news_items):
        sig_color = metrics['signal_color']
        sig_bg = "#dcfce7" if sig_color == "green" else "#fee2e2" if sig_color == "red" else "#eff6ff"
        currency = "₺" if ".IS" in ticker else "$"

        # Haber HTML
        news_html = ""
        if news_items:
            news_html = "<div class='news-list'><b>Son Haberler:</b><br>" + "".join([f"<div class='news-item'><span style='color:#666'>{i['date']}</span> {i['title']}</div>" for i in news_items]) + "</div>"

        self.content += f"""
        <div class="report-section">
            <div class="header">
                <span>{ticker} | {name}</span>
                <span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:15px; font-size:0.7em;">MEGA MODEL v5</span>
            </div>

            <div class="signal-banner" style="background:{sig_bg}; color:{sig_color};">
                AI SİNYALİ: {metrics['signal']} <span style="font-size:0.7em; color:#555">({metrics['signal_desc']})</span>
            </div>

            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Hedef Fiyat (10G)</div>
                    <div class="stat-val">{currency}{metrics['target_price']:.2f}</div>
                    <div class="stat-sub">Mevcut: {currency}{metrics['current_price']:.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Alpha (Yıllık)</div>
                    <div class="stat-val" style="color:{'green' if metrics['alpha']>0 else 'red'}">{metrics['alpha']:.2f}</div>
                    <div class="stat-sub">Piyasadan Bağımsız Getiri</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Beta (Hassasiyet)</div>
                    <div class="stat-val">{metrics['beta']:.2f}</div>
                    <div class="stat-sub">Endeks Korelasyonu</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Fiyat Doğruluğu</div>
                    <div class="stat-val">%{metrics['price_acc']:.1f}</div>
                    <div class="stat-sub">Model Güvenilirliği</div>
                </div>

                <div class="stat-box">
                    <div class="stat-label">Haber Skoru</div>
                    <div class="stat-val" style="color:{'green' if metrics['news_score']>0 else 'red'}">{metrics['news_score']:.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Sharpe Oranı</div>
                    <div class="stat-val">{metrics['sharpe']:.2f}</div>
                    <div class="stat-sub">Risk/Getiri (>1 İyi)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Max Drawdown</div>
                    <div class="stat-val" style="color:red">%{metrics['mdd']:.1f}</div>
                    <div class="stat-sub">Max Risk</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Yön Başarısı</div>
                    <div class="stat-val">%{metrics['dir_acc']:.1f}</div>
                </div>
            </div>

            <div class="chart-area">
                <img class="main-chart" src="data:image/png;base64,{chart_b64}">
            </div>

            <div class="mini-charts">
                <img src="data:image/png;base64,{extra_charts['bollinger']}">
                <img src="data:image/png;base64,{extra_charts['macd']}">
                <img src="data:image/png;base64,{extra_charts['rsi']}">
            </div>

            {news_html}
        </div>
        """

    def save(self):
        self.content += "</div></body></html>"
        with open("BIST_Fon_Analiz.html", "w", encoding="utf-8") as f: f.write(self.content)

# --- MODEL VE ANALİZ ÇEKİRDEĞİ ---
def analyze_ticker(ticker, name, macro_df, market_returns):
    # 1. Veri Hazırlığı
    df = get_stock_data(ticker, macro_df)
    if df is None: return None

    # Feature Selection
    features = ['Close', 'RSI', 'MACD', 'ATR', 'CCI', 'Log_Ret']
    if 'USD_TRY' in df.columns: features += ['USD_TRY', 'VIX']

    data = df[features].values
    target = df[['Log_Ret']].values

    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(data)

    target_idx = features.index('Log_Ret')

    # Train/Test Split
    split = int(len(df) * 0.90)
    lookback = 60

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, target_idx])
    X, y = np.array(X), np.array(y)

    X_train, y_train = X[:split-lookback], y[:split-lookback]
    X_test, y_test = X[split-lookback:], y[split-lookback:]

    # 2. Bi-LSTM Model
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss=Huber())
    model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=0)

    # 3. Tahmin ve Metrikler
    # Test Seti Tahminleri
    pred_test_scaled = model.predict(X_test, verbose=0)

    # Inverse Transform
    dummy = np.zeros((len(pred_test_scaled), len(features)))
    dummy[:, target_idx] = pred_test_scaled.flatten()
    pred_test_ret = scaler.inverse_transform(dummy)[:, target_idx]

    # Gerçek Değerler
    actual_prices = df['Close'].iloc[split:].values
    rec_prices = []
    last_p = df['Close'].iloc[split-1]

    for r in pred_test_ret:
        p = last_p * np.exp(r)
        rec_prices.append(p)
        last_p = actual_prices[len(rec_prices)-1] if len(rec_prices) <= len(actual_prices) else p

    rec_prices = np.array(rec_prices[:len(actual_prices)])

    # Metrik Hesapları
    price_acc = 100 - (mean_absolute_percentage_error(actual_prices, rec_prices) * 100)
    dir_acc = (np.sum(np.sign(np.diff(actual_prices)) == np.sign(np.diff(rec_prices))) / (len(actual_prices)-1)) * 100
    mdd = calculate_max_drawdown(np.cumprod(1 + pred_test_ret))
    sharpe = calculate_sharpe(pred_test_ret)

    # Alpha / Beta (DÜZELTİLMİŞ KISIM BURASI)
    stock_ret_series = df['Log_Ret']
    beta, alpha = calculate_alpha_beta(stock_ret_series, market_returns)

    # 4. Gelecek Tahmini ve Sinyal
    last_batch = scaled_data[-lookback:].reshape(1, lookback, len(features))
    future_prices = []
    curr_p = df['Close'].iloc[-1]

    # 10 Günlük Recursive Tahmin
    temp_batch = last_batch.copy()
    for _ in range(10):
        p_sc = model.predict(temp_batch, verbose=0)[0,0]

        d = np.zeros((1, len(features)))
        d[0, target_idx] = p_sc
        p_ret = scaler.inverse_transform(d)[0, target_idx]

        next_p = curr_p * np.exp(p_ret)
        future_prices.append(next_p)
        curr_p = next_p

        # Batch güncelle
        new_row = temp_batch[0, -1, :].copy()
        new_row[target_idx] = p_sc
        temp_batch = np.append(temp_batch[:, 1:, :], new_row.reshape(1,1,len(features)), axis=1)

    target_price = future_prices[-1]
    current_atr = df['ATR'].iloc[-1]
    signal, sig_color, sig_desc = generate_signal(df['Close'].iloc[-1], target_price, current_atr * 3)

    # 5. Görselleştirme
    # Ana Grafik
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-150:], df['Close'].iloc[-150:], label="Gerçek Fiyat", color="#1f2937")
    ax.plot(df.index[split:], rec_prices, label="Model Testi", linestyle="--", color="#f59e0b")

    fut_dates = pd.date_range(df.index[-1]+timedelta(days=1), periods=10)
    ax.plot(fut_dates, future_prices, label="10 Günlük Tahmin", color=sig_color, linewidth=3)

    ax.set_title(f"{name} Fiyat Tahmini ve Testi")
    ax.legend()
    ax.grid(True, alpha=0.2)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Mini Grafikler
    def mini_plot(data, color, title):
        f, a = plt.subplots(figsize=(5,3))
        a.plot(data[-60:], color=color)
        a.set_title(title, fontsize=10)
        a.grid(alpha=0.2)
        b = BytesIO()
        f.savefig(b, format='png', bbox_inches='tight')
        b.seek(0)
        plt.close(f)
        return base64.b64encode(b.read()).decode('utf-8')

    extras = {
        'rsi': mini_plot(df['RSI'], 'purple', 'RSI (Momentum)'),
        'macd': mini_plot(df['MACD'], 'blue', 'MACD (Trend)'),
        'bollinger': mini_plot(df['ATR'], 'brown', 'ATR (Volatilite)')
    }

    metrics = {
        'current_price': df['Close'].iloc[-1],
        'target_price': target_price,
        'signal': signal, 'signal_color': sig_color, 'signal_desc': sig_desc,
        'alpha': alpha, 'beta': beta,
        'price_acc': price_acc if price_acc > 0 else 0,
        'dir_acc': dir_acc, 'mdd': mdd, 'sharpe': sharpe
    }

    return metrics, chart_b64, extras

# --- MAIN ---
def main():
    report = HTMLRapor()
    print("BİST MEGA ANALİZ BAŞLIYOR...")
    print("1. Benchmark ve Makro Veriler İndiriliyor...")
    market_returns = get_benchmark_data()
    macro_df = get_macro_data()

    for ticker, name in etf_map.items():
        if ticker == 'XU100.IS': continue
        print(f"\n>> {ticker} ({name}) Analiz Ediliyor...")

        # Haber Skoru
        news_score, news_items = get_news_sentiment(ticker)

        # Teknik ve Finansal Analiz
        result = analyze_ticker(ticker, name, macro_df, market_returns)

        if result:
            metrics, chart, extras = result
            metrics['news_score'] = news_score
            report.add_section(ticker, name, metrics, chart, extras, news_items)
            print(f"   TAMAMLANDI -> Sinyal: {metrics['signal']} | Doğruluk: %{metrics['price_acc']:.1f}")

    report.save()
    print("\n" + "="*50)
    print("ANALİZ BİTTİ. Rapor: BIST_Fon_Analiz.html")
    print("="*50)

if __name__ == "__main__":
    main()
