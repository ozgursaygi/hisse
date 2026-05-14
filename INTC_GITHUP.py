# ============================================================
# MULTI-STOCK v17.0 — SCIENTIFIC REGIME AI FORECAST ENGINE
# ============================================================

import os
import sqlite3
import warnings
import random
import base64

from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    Bidirectional
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from tensorflow import random as tf_random

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# CONFIG
# ============================================================

TICKERS_CONFIG = {
    "INTC": {
        "name": "Intel Corporation",
        "benchmark": "SOXX"
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "benchmark": "SOXX"
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "benchmark": "SOXX"
    }
}

PREDICTION_HORIZON = 7
LOOKBACK = 60

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

ENSEMBLE_SEEDS = [42, 52, 62]

EPOCHS = 40
BATCH_SIZE = 32

DB_FOLDER = r"C:\Projects\ML"
DB_PATH = os.path.join(DB_FOLDER, "market_intelligence_v17.db")

# ============================================================
# DATABASE
# ============================================================

def init_db():

    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        tarih TEXT,
        sembol TEXT,
        rejim TEXT,
        signal TEXT,
        price REAL,
        prediction_ret REAL,
        UNIQUE(tarih, sembol)
    )
    """)

    conn.commit()
    conn.close()

# ============================================================
# RANDOM SEED
# ============================================================

def set_seeds(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf_random.set_seed(seed)

# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_ticker_data(ticker):

    try:

        end = datetime.now()
        start = end - timedelta(days=365 * 12)

        t = yf.Ticker(ticker)

        df = t.history(
            start=start,
            end=end,
            auto_adjust=True
        )

        if df.empty:
            return None

        df = df[[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]].copy()

        df = df.dropna()

        return df

    except Exception as e:

        print(f"[ERROR] download_ticker_data {ticker}: {e}")

        return None

# ============================================================
# MACRO DATA
# ============================================================

def get_macro_data():

    tickers = {
        "^VIX": "VIX",
        "^TNX": "US10Y",
        "DX-Y.NYB": "DXY"
    }

    try:

        df = yf.download(
            list(tickers.keys()),
            period="12y",
            auto_adjust=True,
            progress=False
        )["Close"]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=tickers)

        df = df.ffill()

        return df

    except Exception as e:

        print(f"[ERROR] macro data: {e}")

        return pd.DataFrame()

# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regimes(bench_close):

    df = pd.DataFrame(index=bench_close.index)

    log_ret = np.log(
        bench_close / bench_close.shift(1)
    )

    sma50 = bench_close.rolling(50).mean()
    sma200 = bench_close.rolling(200).mean()

    trend_strength = (
        1.0 /
        (
            1.0 +
            np.exp(
                -((sma50 / sma200) - 1.0) * 50
            )
        )
    )

    vol20 = log_ret.rolling(20).std()

    vol_q70 = vol20.expanding().quantile(0.7)

    p_chaos = (
        1.0 /
        (
            1.0 +
            np.exp(
                -((vol20 / vol_q70) - 1.0) * 5
            )
        )
    )

    p_chaos = p_chaos.clip(0.05, 0.95)

    p_bull = (1.0 - p_chaos) * trend_strength
    p_bear = (1.0 - p_chaos) * (1.0 - trend_strength)

    total = p_bull + p_bear + p_chaos

    df["p_bull"] = p_bull / total
    df["p_bear"] = p_bear / total
    df["p_chaos"] = p_chaos / total

    probs = df[[
        "p_bull",
        "p_bear",
        "p_chaos"
    ]].values

    labels = ["BULL", "BEAR", "CHAOS"]

    df["regime"] = [
        labels[np.argmax(x)] for x in probs
    ]

    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    stock_df,
    bench_df,
    macro_df,
    regime_df
):

    try:

        import ta

        df = stock_df.copy()

        df["Bench_Close"] = bench_df["Close"]

        # =========================
        # RETURNS
        # =========================

        df["Log_Ret"] = np.log(
            df["Close"] / df["Close"].shift(1)
        )

        df["Bench_Log_Ret"] = np.log(
            df["Bench_Close"] /
            df["Bench_Close"].shift(1)
        )

        # =========================
        # RSI
        # =========================

        df["RSI"] = ta.momentum.rsi(
            df["Close"],
            window=14
        )

        # =========================
        # MACD
        # =========================

        df["MACD"] = ta.trend.macd_diff(
            df["Close"]
        )

        # =========================
        # ATR
        # =========================

        df["ATR"] = ta.volatility.average_true_range(
            df["High"],
            df["Low"],
            df["Close"]
        )

        # =========================
        # VOLATILITY
        # =========================

        df["VOL_20"] = (
            df["Log_Ret"]
            .rolling(20)
            .std()
        )

        # =========================
        # MOMENTUM
        # =========================

        df["MOM_20"] = (
            df["Close"] /
            df["Close"].shift(20)
        ) - 1

        # =========================
        # BETA
        # =========================

        cov = (
            df["Log_Ret"]
            .rolling(60)
            .cov(df["Bench_Log_Ret"])
        )

        var = (
            df["Bench_Log_Ret"]
            .rolling(60)
            .var()
        )

        df["Beta_60"] = (
            cov / var
        ).clip(0.0, 5.0)

        # =========================
        # FIBONACCI FEATURES
        # =========================

        wave_high = (
            df["High"]
            .rolling(120)
            .max()
        )

        wave_low = (
            df["Low"]
            .rolling(120)
            .min()
        )

        wave_range = wave_high - wave_low

        fib382 = (
            wave_high -
            wave_range * 0.382
        )

        fib618 = (
            wave_high -
            wave_range * 0.618
        )

        df["Fib_382_Dist"] = (
            (df["Close"] - fib382)
            / df["Close"]
        )

        df["Fib_618_Dist"] = (
            (df["Close"] - fib618)
            / df["Close"]
        )

        # =========================
        # FUTURE RETURN
        # =========================

        future_stock = np.log(
            df["Close"].shift(-PREDICTION_HORIZON)
            / df["Close"]
        )

        future_bench = np.log(
            df["Bench_Close"].shift(-PREDICTION_HORIZON)
            / df["Bench_Close"]
        )

        df["Future_Residual_Ret"] = (
            future_stock -
            (
                df["Beta_60"] *
                future_bench
            )
        )

        # =========================
        # MERGE
        # =========================

        df = (
            df
            .join(
                regime_df[[
                    "p_bull",
                    "p_bear",
                    "p_chaos"
                ]],
                how="left"
            )
            .join(
                macro_df,
                how="left"
            )
        )

        df = df.ffill()

        df = df.dropna()

        return df

    except Exception as e:

        print(f"[ERROR] feature engineering: {e}")

        return None

# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(X, y, lookback):

    Xs = []
    ys = []

    for i in range(lookback, len(X)):

        Xs.append(
            X[i-lookback:i]
        )

        ys.append(y[i])

    return (
        np.array(Xs),
        np.array(ys)
    )

# ============================================================
# MODEL
# ============================================================

def build_lstm(input_shape):

    model = Sequential([

        Input(shape=input_shape),

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        Dropout(0.3),

        LSTM(32),

        Dropout(0.2),

        Dense(
            16,
            activation="relu"
        ),

        Dense(1)

    ])

    model.compile(
        optimizer=Adam(0.001),
        loss=Huber()
    )

    return model

# ============================================================
# TRAINING
# ============================================================

def train_experts(
    Xt,
    yt,
    Xv,
    yv,
    probs,
    split_idx
):

    experts = {}

    regimes = [
        "bull",
        "bear",
        "chaos"
    ]

    for idx, regime in enumerate(regimes):

        print(f"   [TRAIN] {regime.upper()}")

        sw = np.clip(
            probs[
                LOOKBACK:split_idx,
                idx
            ],
            0.05,
            1.0
        )

        if sw.sum() < 100:

            experts[regime] = None

            continue

        models = []

        for seed in ENSEMBLE_SEEDS:

            set_seeds(seed)

            model = build_lstm(
                (
                    Xt.shape[1],
                    Xt.shape[2]
                )
            )

            es = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

            rl = ReduceLROnPlateau(
                monitor="val_loss",
                patience=5,
                factor=0.5
            )

            model.fit(
                Xt,
                yt,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(Xv, yv),
                sample_weight=sw,
                callbacks=[es, rl],
                verbose=0
            )

            models.append(model)

        experts[regime] = models

    return experts

# ============================================================
# ENSEMBLE PREDICT
# ============================================================

def ensemble_predict(
    experts,
    X,
    probs
):

    outputs = {}

    for i, regime in enumerate([
        "bull",
        "bear",
        "chaos"
    ]):

        if experts.get(regime):

            preds = []

            for model in experts[regime]:

                p = model.predict(
                    X,
                    verbose=0
                ).flatten()

                preds.append(p)

            outputs[regime] = np.mean(
                preds,
                axis=0
            )

        else:

            outputs[regime] = np.zeros(len(X))

    final = (
        probs[:, 0] * outputs["bull"] +
        probs[:, 1] * outputs["bear"] +
        probs[:, 2] * outputs["chaos"]
    )

    return final

# ============================================================
# CHARTS
# ============================================================

def make_chart(
    ticker,
    df,
    split_idx,
    actual,
    predicted,
    future_prices
):

    plt.figure(figsize=(14, 7))

    plt.style.use("dark_background")

    dates = df.index

    hist_start = max(0, split_idx - 120)

    plt.plot(
        dates[hist_start:split_idx],
        df["Close"].iloc[
            hist_start:split_idx
        ],
        label="History"
    )

    plt.plot(
        dates[split_idx:split_idx+len(actual)],
        actual,
        label="Actual"
    )

    plt.plot(
        dates[split_idx:split_idx+len(predicted)],
        predicted,
        linestyle="--",
        label="Predicted"
    )

    future_dates = pd.bdate_range(
        dates[-1],
        periods=8
    )[1:]

    plt.plot(
        [dates[-1]] + list(future_dates),
        [df["Close"].iloc[-1]] + future_prices,
        marker="o",
        linewidth=2,
        label="Future Forecast"
    )

    plt.title(
        f"{ticker} Scientific AI Forecast"
    )

    plt.grid(alpha=0.2)

    plt.legend()

    buf = BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    return base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

# ============================================================
# ANALYSIS
# ============================================================

def analyze_ticker(
    ticker,
    config,
    macro_df
):

    print("\n" + "=" * 70)
    print(f"ANALYZING {ticker}")
    print("=" * 70)

    stock_df = download_ticker_data(
        ticker
    )

    bench_df = download_ticker_data(
        config["benchmark"]
    )

    if stock_df is None:
        return None

    if bench_df is None:
        return None

    common = stock_df.index.intersection(
        bench_df.index
    )

    stock_df = stock_df.loc[common]
    bench_df = bench_df.loc[common]

    regime_df = detect_regimes(
        bench_df["Close"]
    )

    df = build_features(
        stock_df,
        bench_df,
        macro_df,
        regime_df
    )

    if df is None:
        return None

    excluded = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Close",
        "Bench_Close",
        "Future_Residual_Ret"
    ]

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    X_raw = df[feature_cols].values

    y_raw = df[
        "Future_Residual_Ret"
    ].values

    probs = df[[
        "p_bull",
        "p_bear",
        "p_chaos"
    ]].values

    train_idx = int(
        len(df) * TRAIN_FRAC
    )

    val_idx = int(
        len(df) * (
            TRAIN_FRAC + VAL_FRAC
        )
    )

    # ========================================================
    # NO DATA LEAKAGE
    # ========================================================

    x_sc = MinMaxScaler()

    x_sc.fit(
        X_raw[:train_idx]
    )

    X_scaled = x_sc.transform(X_raw)

    y_sc = MinMaxScaler(
        feature_range=(-1, 1)
    )

    y_sc.fit(
        y_raw[:train_idx].reshape(-1, 1)
    )

    y_scaled = y_sc.transform(
        y_raw.reshape(-1, 1)
    ).flatten()

    # ========================================================
    # DATASETS
    # ========================================================

    Xt, yt = create_dataset(
        X_scaled[:train_idx],
        y_scaled[:train_idx],
        LOOKBACK
    )

    Xv, yv = create_dataset(
        X_scaled[
            train_idx-LOOKBACK:val_idx
        ],
        y_scaled[
            train_idx-LOOKBACK:val_idx
        ],
        LOOKBACK
    )

    Xte, yte = create_dataset(
        X_scaled[val_idx-LOOKBACK:],
        y_scaled[val_idx-LOOKBACK:],
        LOOKBACK
    )

    # ========================================================
    # TRAIN
    # ========================================================

    experts = train_experts(
        Xt,
        yt,
        Xv,
        yv,
        probs,
        train_idx
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    pred_scaled = ensemble_predict(
        experts,
        Xte,
        probs[val_idx:val_idx+len(yte)]
    )

    pred_ret = y_sc.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    actual_ret = y_sc.inverse_transform(
        yte.reshape(-1, 1)
    ).flatten()

    # ========================================================
    # FUTURE FORECAST
    # ========================================================

    last_seq = X_scaled[
        -LOOKBACK:
    ].reshape(
        1,
        LOOKBACK,
        len(feature_cols)
    )

    last_probs = probs[-1:].reshape(1, 3)

    future_scaled = ensemble_predict(
        experts,
        last_seq,
        last_probs
    )

    future_ret = float(
        y_sc.inverse_transform(
            future_scaled.reshape(-1, 1)
        )[0, 0]
    )

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    try:

        current_price = float(
            yf.Ticker(ticker)
            .fast_info["lastPrice"]
        )

    except:

        current_price = float(
            df["Close"].iloc[-1]
        )

    # ========================================================
    # TARGET
    # ========================================================

    target_price = (
        current_price *
        np.exp(future_ret)
    )

    # ========================================================
    # TEST PRICE SERIES
    # ========================================================

    base_prices = df["Close"].iloc[
        val_idx:val_idx+len(actual_ret)
    ].values

    actual_prices = (
        base_prices *
        np.exp(actual_ret)
    )

    predicted_prices = (
        base_prices *
        np.exp(pred_ret)
    )

    # ========================================================
    # METRICS
    # ========================================================

    mae = mean_absolute_error(
        actual_ret,
        pred_ret
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_ret,
            pred_ret
        )
    )

    r2 = r2_score(
        actual_ret,
        pred_ret
    )

    direction_acc = (
        (
            np.sign(actual_ret) ==
            np.sign(pred_ret)
        ).mean()
    ) * 100

    residual_std = np.std(
        actual_ret - pred_ret
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    if future_ret > 0.5 * residual_std:

        signal = "BUY 🚀"
        direction = "BULLISH 📈"
        color = "#22c55e"

    elif future_ret < -0.5 * residual_std:

        signal = "SELL 💥"
        direction = "BEARISH 📉"
        color = "#ef4444"

    else:

        signal = "NEUTRAL 🟡"
        direction = "SIDEWAYS ➡️"
        color = "#f59e0b"

    # ========================================================
    # CHART
    # ========================================================

    future_path = np.linspace(
        current_price,
        target_price,
        7
    ).tolist()

    chart_b64 = make_chart(
        ticker,
        df,
        val_idx,
        actual_prices,
        predicted_prices,
        future_path
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(f"Current Price : ${current_price:.2f}")
    print(f"Target Price  : ${target_price:.2f}")
    print(f"Direction Acc : %{direction_acc:.2f}")
    print(f"RMSE          : {rmse:.6f}")
    print(f"R2            : {r2:.4f}")
    print(f"Signal        : {signal}")

    return {

        "ticker": ticker,
        "name": config["name"],

        "price": current_price,
        "target": target_price,

        "signal": signal,
        "direction": direction,

        "regime": regime_df["regime"].iloc[-1],

        "acc": direction_acc,
        "rmse": rmse,
        "r2": r2,

        "chart": chart_b64,

        "color": color,

        "p_bull": probs[-1, 0],
        "p_bear": probs[-1, 1],
        "p_chaos": probs[-1, 2]
    }

# ============================================================
# HTML REPORT
# ============================================================

def generate_html(results):

    html = """
    <!DOCTYPE html>
    <html lang="tr">

    <head>

    <meta charset="UTF-8">

    <title>
    Scientific Multi-Stock AI Report
    </title>

    <style>

    body{
        background:#0f172a;
        color:#f8fafc;
        font-family:Segoe UI;
        padding:20px;
    }

    .card{
        background:#1e293b;
        border-radius:16px;
        padding:25px;
        margin-bottom:30px;
        box-shadow:0 4px 20px rgba(0,0,0,0.3);
    }

    .header{
        display:flex;
        justify-content:space-between;
        align-items:center;
        border-bottom:1px solid #334155;
        padding-bottom:15px;
    }

    .grid{
        display:grid;
        grid-template-columns:
        repeat(auto-fit,minmax(200px,1fr));
        gap:20px;
        margin-top:20px;
    }

    .stat{
        background:#0f172a;
        padding:15px;
        border-radius:10px;
        text-align:center;
    }

    img{
        width:100%;
        margin-top:20px;
        border-radius:12px;
        border:1px solid #334155;
    }

    </style>

    </head>

    <body>

    <h1 style='text-align:center;'>
    Scientific Regime AI Forecast Engine
    </h1>
    """

    for r in results:

        html += f"""

        <div class='card'>

            <div class='header'>

                <h2>
                {r['ticker']} - {r['name']}
                </h2>

                <span style='
                    background:{r["color"]};
                    padding:10px 20px;
                    border-radius:20px;
                    font-weight:bold;
                '>

                {r['signal']}

                </span>

            </div>

            <div class='grid'>

                <div class='stat'>
                    <div>Current Price</div>
                    <div style='font-size:1.5em;font-weight:bold;'>
                    ${r['price']:.2f}
                    </div>
                </div>

                <div class='stat'>
                    <div>Target Price</div>
                    <div style='font-size:1.5em;color:#4ade80;'>
                    ${r['target']:.2f}
                    </div>
                </div>

                <div class='stat'>
                    <div>Direction Accuracy</div>
                    <div>
                    %{r['acc']:.2f}
                    </div>
                </div>

                <div class='stat'>
                    <div>Regime</div>
                    <div>
                    {r['regime']}
                    </div>
                </div>

                <div class='stat'>
                    <div>RMSE</div>
                    <div>
                    {r['rmse']:.6f}
                    </div>
                </div>

                <div class='stat'>
                    <div>R²</div>
                    <div>
                    {r['r2']:.4f}
                    </div>
                </div>

            </div>

            <img src='data:image/png;base64,{r["chart"]}'>

        </div>
        """

    html += """
    </body>
    </html>
    """

    with open(
        "Scientific_AI_Report.html",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print("\n[OK] HTML report created")
    print("Scientific_AI_Report.html")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    init_db()

    set_seeds()

    macro_df = get_macro_data()

    results = []

    for ticker, config in TICKERS_CONFIG.items():

        try:

            res = analyze_ticker(
                ticker,
                config,
                macro_df
            )

            if res:
                results.append(res)

        except Exception as e:

            print(f"[FATAL ERROR] {ticker}: {e}")

    if results:

        generate_html(results)

    else:

        print("No results generated.")