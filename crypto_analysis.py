# --- UYARILARI GİZLEME VE KURULUM KONTROLÜ ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import time
import requests 
import xml.etree.ElementTree as ET 
import random

required_libs = ['ta', 'yfinance', 'textblob', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow']
for lib in required_libs:
    try:
        __import__(lib)
    except ImportError:
        pass 

import ta
import yfinance as yf
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf 
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# --- SEED SABİTLEME ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- AYARLAR ---
tickers_map = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'XRP-USD': 'Ripple',
    'DOGE-USD': 'Dogecoin',
    'SOL-USD': 'Solana',
    'BNB-USD': 'Binance Coin',
    'ADA-USD': 'Cardano'
}

# Grafik Stili
plt.style.use('dark_background')
sns.set_style("darkgrid", {"axes.facecolor": "#0f172a", "grid.color": "#334155", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# --- HTML RAPOR ---
class HTMLRapor:
    def __init__(self):
        self.content = """
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <title>AI Crypto Analyzer</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background-color: #0f172a; color: #e2e8f0; }
                .container { max-width: 1400px; margin: 0 auto; background: #1e293b; padding: 40px; border-radius: 12px; border: 1px solid #334155; }
                h1 { text-align: center; color: #38bdf8; border-bottom: 2px solid #0ea5e9; padding-bottom: 15px; }
                .report-section { margin-bottom: 50px; border: 1px solid #334155; padding: 25px; border-radius: 12px; background: #0f172a; }
                .report-header { background: #1e293b; color: #38bdf8; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; font-size: 1.4em; font-weight: 600; display: flex; justify-content: space-between; align-items: center; border: 1px solid #334155; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 25px; }
                .stat-box { background: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #ec4899; }
                .stat-label { font-size: 0.75em; color: #94a3b8; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }
                .stat-value { font-size: 1.3em; font-weight: 700; color: #f8fafc; }

                /* Grid 3 sütunlu */
                .chart-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px;}
                .chart-box { background: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155; text-align: center; }
                .chart-box.full-width { grid-column: 1 / -1; }
                .chart-box.span-2 { grid-column: span 2; }

                table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.85em; }
                th { text-align: left; color: #94a3b8; padding: 10px; border-bottom: 1px solid #334155; }
                td { padding: 8px; border-bottom: 1px solid #334155; color: #cbd5e1; }
                .pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #94a3b8; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Crypto Analysis & Forecast Report</h1>
                <p style="text-align: center; color:#94a3b8;">Technical Analysis + Sentiment + LSTM Prediction + Drawdown Metrics</p>
        """

    def add_section(self, ticker, company_name, price, sentiment_score, sentiment_status, r2, model_accuracy, pred_price, direction_acc, volatility, charts, news_list):
        sent_color = "#4ade80" if sentiment_score > 0 else "#f87171" if sentiment_score < 0 else "#94a3b8"
        r2_color = "#4ade80" if r2 > 0 else "#f87171"
        acc_color = "#4ade80" if model_accuracy > 97 else "#fbbf24"
        dir_color = "#4ade80" if direction_acc > 55 else "#f87171"

        vol_text = "Düşük" if volatility < 0.02 else "Orta" if volatility < 0.04 else "Yüksek"
        vol_color = "#4ade80" if volatility < 0.02 else "#fbbf24" if volatility < 0.04 else "#f87171"

        news_html = "<table><tr><th>Haber Başlığı</th><th>Duygu</th><th>Skor</th></tr>"
        for news in news_list[:10]:
            score = news['sentiment']
            cls = "pos" if score > 0.05 else "neg" if score < -0.05 else "neu"
            news_html += f"<tr><td>{news['title'][:80]}...</td><td class='{cls}'>{cls.upper()}</td><td>{score:.2f}</td></tr>"
        news_html += "</table>"

        self.content += f"""
        <div class="report-section">
            <div class="report-header">
                <span>{ticker} | {company_name}</span>
                <span style="color:{sent_color}">Sentiment: {sentiment_status} ({sentiment_score:.2f})</span>
            </div>
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-label">Son Fiyat</div><div class="stat-value">${price:.4f}</div></div>
                <div class="stat-box" style="border-left-color:{vol_color}"><div class="stat-label">Volatilite (Risk)</div><div class="stat-value" style="color:{vol_color}">{vol_text} (%{volatility*100:.1f})</div></div>
                <div class="stat-box" style="border-left-color:{acc_color}"><div class="stat-label">Model Doğruluğu</div><div class="stat-value" style="color:{acc_color}">%{model_accuracy:.2f}</div></div>
                <div class="stat-box" style="border-left-color:{dir_color}"><div class="stat-label">Trend İsabet Oranı</div><div class="stat-value" style="color:{dir_color}">%{direction_acc:.1f}</div></div>
                <div class="stat-box" style="border-left-color:{r2_color}"><div class="stat-label">R² Skoru</div><div class="stat-value" style="color:{r2_color}">{r2:.3f}</div></div>
                <div class="stat-box"><div class="stat-label">10G Tahmin</div><div class="stat-value">${pred_price:.4f}</div></div>
            </div>

            <div class="chart-grid">
                <div class="chart-box full-width"><img src="data:image/png;base64,{charts['main']}"></div>

                <div class="chart-box"><img src="data:image/png;base64,{charts['ma']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['vol']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['dist']}"></div> 

                <div class="chart-box span-2"><img src="data:image/png;base64,{charts['tech']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['obv']}"></div> 

                <!-- YENİ GRAFİKLER -->
                <div class="chart-box span-2"><img src="data:image/png;base64,{charts['drawdown']}"></div>
                <div class="chart-box"><img src="data:image/png;base64,{charts['scatter']}"></div>
            </div>

            <div style="margin-top:20px;">
                <h3 style="color:#a5b4fc; font-size:1em;">Son Haberler (Yahoo RSS)</h3>
                {news_html}
            </div>
        </div>
        """

    def save(self, filename):
        self.content += "</div></body></html>"
        with open(filename, "w", encoding="utf-8") as f: f.write(self.content)

# --- HABER MOTORU ---
class SentimentEngine:
    def __init__(self): self.headers = {'User-Agent': 'Mozilla/5.0'}
    def get_sentiment(self, ticker):
        try:
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            response = requests.get(rss_url, headers=self.headers)
            if response.status_code != 200: return 0, "Connection Error", []
            root = ET.fromstring(response.content)
            news_items = root.findall('.//item')
            if not news_items: return 0, "No RSS Data", []
            processed = []
            total = 0; count = 0
            for item in news_items:
                title = item.find('title').text
                if not title: continue
                pol = TextBlob(title).sentiment.polarity
                processed.append({'title': title, 'sentiment': pol})
                total += pol; count += 1
                if count >= 15: break 
            if count == 0: return 0, "Neutral", []
            avg = total / count
            stat = "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral"
            return avg, stat, processed
        except: return 0, "Error", []

# --- VERİ VE MODEL ---
def calculate_indicators(close_prices, volume):
    df = pd.DataFrame({'Close': close_prices, 'Volume': volume})
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['MA200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['OBV_EMA'] = ta.trend.EMAIndicator(df['OBV'], window=20).ema_indicator()

    # YENİ: Drawdown Hesaplama
    # Her gün için o güne kadarki en yüksek fiyatı bul
    rolling_max = df['Close'].expanding().max()
    # Şu anki fiyatın zirveye göre kaybı
    df['Drawdown'] = (df['Close'] / rolling_max) - 1

    return df

def get_data(ticker):
    try:
        end = pd.Timestamp.now()
        start = end - pd.DateOffset(years=5)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[['Close', 'Volume']].fillna(method='ffill')
        df = df.resample('D').ffill()
        ind = calculate_indicators(df['Close'], df['Volume'])
        cols_to_add = ['Log_Ret', 'RSI', 'MACD', 'MA50', 'MA200', 'BB_High', 'BB_Low', 'OBV', 'OBV_EMA', 'Drawdown']
        df = pd.concat([df, ind[cols_to_add]], axis=1).dropna()
        return df
    except: return None

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0f172a')
    buf.seek(0)
    res = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return res

def run_analysis():
    report = HTMLRapor()
    sent_engine = SentimentEngine()

    print("AI Crypto Analyzer - Başlatılıyor...")
    print("-" * 70)

    for ticker, name in tickers_map.items():
        print(f" >> {name} ({ticker})...")
        clear_session()
        time.sleep(1)
        score, status, news = sent_engine.get_sentiment(ticker)

        df = get_data(ticker)
        if df is None: continue

        recent_volatility = df['Log_Ret'].tail(90).std()

        features = ['Log_Ret', 'RSI', 'MACD', 'Volume']
        data = df[features].values

        train_len = int(len(data) * 0.9)
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(data[:train_len])
        test_scaled = scaler.transform(data[train_len:])
        full_scaled = np.concatenate((train_scaled, test_scaled))

        t_scaler = MinMaxScaler()
        t_scaler.fit(data[:train_len, 0].reshape(-1, 1))

        lookback = 14
        X, y = [], []
        for i in range(lookback, len(train_scaled)):
            X.append(train_scaled[i-lookback:i])
            y.append(train_scaled[i, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            Input(shape=(lookback, len(features))),
            LSTM(64, return_sequences=False), 
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(X, y, epochs=25, batch_size=16, verbose=0)

        # TAHMİN
        future_days = 10
        future_prices = []
        curr_batch = full_scaled[-lookback:].reshape(1, lookback, 4)
        curr_price = df['Close'].iloc[-1]
        sim_h_p = df['Close'].tolist()
        sim_h_v = df['Volume'].tolist()

        for _ in range(future_days):
            pred_s = model.predict(curr_batch, verbose=0)[0, 0]
            pred_ret = t_scaler.inverse_transform([[pred_s]])[0, 0]
            vol_multiplier = 1.0 + (recent_volatility * 10) 
            adj_ret = (pred_ret * vol_multiplier) + (score * 0.005 * vol_multiplier)
            next_price = curr_price * np.exp(adj_ret)
            future_prices.append(next_price)
            sim_h_p.append(next_price)
            sim_h_v.append(pd.Series(sim_h_v[-20:]).mean())
            curr_price = next_price
            s_series = pd.Series(sim_h_p[-100:])
            n_row = scaler.transform([[adj_ret, ta.momentum.rsi(s_series, 14).iloc[-1], ta.trend.MACD(s_series).macd().iloc[-1], sim_h_v[-1]]])
            curr_batch = np.append(curr_batch[:, 1:, :], n_row.reshape(1, 1, 4), axis=1)

        # BACKTEST
        test_inputs = full_scaled[len(full_scaled)-len(test_scaled)-lookback:]
        x_test = []
        for i in range(lookback, len(test_inputs)): x_test.append(test_inputs[i-lookback:i])
        x_test = np.array(x_test)

        pred_log_ret = t_scaler.inverse_transform(model.predict(x_test, verbose=0))
        real_test_prices = df['Close'].iloc[train_len:].values
        rolling_pred_prices = []

        for i in range(len(pred_log_ret)):
            prev_real_price = df['Close'].iloc[train_len + i - 1] 
            next_pred_price = prev_real_price * np.exp(pred_log_ret[i, 0])
            rolling_pred_prices.append(next_pred_price)
        rolling_pred_prices = np.array(rolling_pred_prices)

        rmse = np.sqrt(mean_squared_error(real_test_prices, rolling_pred_prices))
        r2 = r2_score(real_test_prices, rolling_pred_prices)
        mape = mean_absolute_percentage_error(real_test_prices, rolling_pred_prices)
        model_accuracy = 100 * (1 - mape)

        diff_act = np.diff(real_test_prices)
        diff_pred = np.diff(rolling_pred_prices)
        if np.sum(diff_act != 0) > 0: direction_acc = np.mean(np.sign(diff_act) == np.sign(diff_pred)) * 100
        else: direction_acc = 50.0

        # GRAFİKLER
        charts = {}
        mask = df.index > (df.index[-1] - pd.Timedelta(days=120))
        sub = df.loc[mask]

        # 1. Ana Grafik
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        dates_future = pd.date_range(df.index[-1], periods=future_days+1)[1:]
        test_dates = df.index[train_len:]
        mask_test = test_dates > (df.index[-1] - pd.Timedelta(days=120))

        ax1.plot(df.index[mask], df['Close'][mask], label='Gerçek Fiyat', color='#94a3b8', alpha=0.7)
        if len(test_dates[mask_test]) > 0:
            plot_data = rolling_pred_prices[-len(test_dates[mask_test]):]
            ax1.plot(test_dates[mask_test], plot_data, label='AI Rolling Test', color='#fbbf24', linestyle='--', alpha=0.8)
        ax1.plot(dates_future, future_prices, label='Gelecek Tahmini', color='#38bdf8', marker='o')
        ax1.set_title(f"{name} - Fiyat Analizi")
        leg1 = ax1.legend(loc='upper right', frameon=True)
        leg1.get_frame().set_facecolor('black')
        for t in leg1.get_texts(): t.set_color("white")
        charts['main'] = plot_to_base64(fig1)

        # 2. MA
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(sub.index, sub['Close'], color='#94a3b8', alpha=0.5, label='Fiyat')
        ax2.plot(sub.index, sub['MA50'], color='#fbbf24', label='MA50')
        ax2.plot(sub.index, sub['MA200'], color='#f472b6', label='MA200')
        ax2.set_title("Trend Analizi")
        leg2 = ax2.legend(frameon=True)
        leg2.get_frame().set_facecolor('black')
        for t in leg2.get_texts(): t.set_color("white")
        charts['ma'] = plot_to_base64(fig2)

        # 3. Volatilite
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(sub.index, sub['Close'], color='#94a3b8')
        ax3.fill_between(sub.index, sub['BB_High'], sub['BB_Low'], color='#22d3ee', alpha=0.1)
        ax3.set_title("Bollinger")
        charts['vol'] = plot_to_base64(fig3)

        # 4. Teknik
        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax4a.plot(sub.index, sub['RSI'], color='#c084fc', label='RSI')
        ax4a.axhline(70, color='r', linestyle='--', alpha=0.3); ax4a.axhline(30, color='g', linestyle='--', alpha=0.3)
        leg4a = ax4a.legend(loc='upper left', frameon=True)
        leg4a.get_frame().set_facecolor('black')
        for t in leg4a.get_texts(): t.set_color("white")
        ax4b.bar(sub.index, sub['MACD'], color='#22d3ee', label='MACD')
        leg4b = ax4b.legend(loc='upper left', frameon=True)
        leg4b.get_frame().set_facecolor('black')
        for t in leg4b.get_texts(): t.set_color("white")
        charts['tech'] = plot_to_base64(fig4)

        # 5. OBV
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(sub.index, sub['OBV'], color='#a3e635', label='OBV')
        ax5.plot(sub.index, sub['OBV_EMA'], color='#facc15', label='OBV Trend', linestyle='--')
        ax5.set_title("Hacim Dengesi (OBV)")
        leg5 = ax5.legend(frameon=True)
        leg5.get_frame().set_facecolor('black')
        for t in leg5.get_texts(): t.set_color("white")
        charts['obv'] = plot_to_base64(fig5)

        # 6. Risk Dağılımı
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        daily_rets = df['Log_Ret'].tail(365) * 100
        sns.histplot(daily_rets, kde=True, color='#f472b6', ax=ax6, bins=30)
        ax6.axvline(0, color='white', linestyle='--', alpha=0.5)
        ax6.set_title("Risk Analizi (Günlük %)")
        charts['dist'] = plot_to_base64(fig6)

        # --- YENİ GRAFİK 1: Drawdown (Tepe-Dip Kaybı) ---
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        # Drawdown'ı yüzdeye çevir
        dd_series = sub['Drawdown'] * 100 
        ax7.fill_between(sub.index, dd_series, 0, color='#ef4444', alpha=0.3, label='Kayip %')
        ax7.plot(sub.index, dd_series, color='#ef4444')
        ax7.set_title("Maximum Drawdown (Zirveden Kayıp)")
        ax7.set_ylabel("Kayıp (%)")
        leg7 = ax7.legend(frameon=True)
        leg7.get_frame().set_facecolor('black')
        for t in leg7.get_texts(): t.set_color("white")
        charts['drawdown'] = plot_to_base64(fig7)

        # --- YENİ GRAFİK 2: Prediction vs Actual (Doğrulama) ---
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        # Gerçek vs Tahmin (Rolling Test Sonuçları)
        ax8.scatter(real_test_prices, rolling_pred_prices, alpha=0.5, color='#38bdf8', s=10)

        # Mükemmel tahmin çizgisi (y=x)
        lims = [
            np.min([ax8.get_xlim(), ax8.get_ylim()]),  
            np.max([ax8.get_xlim(), ax8.get_ylim()]),  
        ]
        ax8.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Mükemmel Hat')
        ax8.set_title("Tahmin vs Gerçek Sapması")
        ax8.set_xlabel("Gerçek Fiyat")
        ax8.set_ylabel("AI Tahmini")
        leg8 = ax8.legend(frameon=True)
        leg8.get_frame().set_facecolor('black')
        for t in leg8.get_texts(): t.set_color("white")
        charts['scatter'] = plot_to_base64(fig8)

        report.add_section(ticker, name, df['Close'].iloc[-1], score, status, r2, model_accuracy, future_prices[-1], direction_acc, recent_volatility, charts, news)
        print(f"    -> Tamamlandı (Trend İsabeti: %{direction_acc:.1f})")

    report.save("AI_Crypto_Analyzer_Report.html")
    print("\nRAPOR: AI_Crypto_Analyzer_Report.html")

if __name__ == "__main__":
    run_analysis()
