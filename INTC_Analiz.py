# ============================================================
# MULTI-STOCK v15.2
# Değişiklikler (v15.1 üzerine):
#   + NVDA eklendi
#   + Tek HTML rapor (tüm hisseler sırayla, ayrı dosya yok)
#   + Sadece ana grafik (diğerleri kaldırıldı)
#   + Tüm v15.1 fix'leri korundu
# ============================================================

import os, sqlite3, warnings, random, base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
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
# 🎯 KONFİGÜRASYON — Hisse eklemek/çıkarmak için yalnızca burası
# ============================================================
TICKERS_CONFIG = {
    'INTC': {'name': 'Intel Corporation',     'benchmark': 'SOXX'},
    'AMD':  {'name': 'Advanced Micro Devices', 'benchmark': 'SOXX'},
    'NVDA': {'name': 'NVIDIA Corporation',     'benchmark': 'SOXX'},
    # Örnek ek hisseler:
    # 'AAPL': {'name': 'Apple Inc.',       'benchmark': 'SPY'},
    # 'TSLA': {'name': 'Tesla Inc.',       'benchmark': 'SPY'},
    # 'JPM':  {'name': 'JPMorgan Chase',   'benchmark': 'XLF'},
}

PREDICTION_HORIZON   = 7
TRANSACTION_COST_BPS = 5
LOOKBACK             = 60
TRAIN_FRAC           = 0.80
VAL_FRAC             = 0.10
ENSEMBLE_SEEDS       = [42, 52, 62]
EPOCHS               = 60
DB_FOLDER            = r"C:\Projects\ML"
OUTPUT_HTML          = "MultiStock_Analiz_v15.2.html"

# ============================================================
# YARDIMCI
# ============================================================

def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_random.set_seed(seed)

# ============================================================
# VERİ
# ============================================================

def download_pair(ticker, benchmark):
    end   = datetime.now()
    start = end - timedelta(days=12 * 365)
    try:
        ds = yf.download(ticker,    start=start, end=end, progress=False, auto_adjust=False)
        db = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"   HATA {ticker}: {e}"); return None, None
    if ds is None or ds.empty or db is None or db.empty:
        return None, None
    for df in [ds, db]:
        if hasattr(df.columns, 'get_level_values'):
            df.columns = df.columns.get_level_values(0)
    ds = ds[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    db = db[['Close']].rename(columns={'Close': 'Bench_Close'}).dropna()
    idx = ds.index.intersection(db.index)
    return ds.loc[idx], db.loc[idx]

def get_macro():
    end  = datetime.now(); start = end - timedelta(days=12 * 365)
    tmap = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY",
            "^GSPC": "SP500", "^IXIC": "NASDAQ"}
    try:
        df = yf.download(list(tmap.keys()), start=start, end=end, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.rename(columns=tmap, inplace=True)
        return df.ffill()
    except:
        return pd.DataFrame()

# ============================================================
# REJİM TESPİTİ
# ============================================================

def detect_regime(bench_close):
    log_ret     = np.log(bench_close / bench_close.shift(1))
    sma50       = bench_close.rolling(50).mean()
    sma200      = bench_close.rolling(200).mean()
    trend_score = (sma50 / sma200) - 1.0
    trend_str   = 1.0 / (1.0 + np.exp(-trend_score * 50))
    vol         = log_ret.rolling(20).std()
    vol_q       = vol.expanding(min_periods=252).quantile(0.70)
    p_chaos     = (1.0 / (1.0 + np.exp(-(vol / vol_q - 1.0) * 5))).clip(0.05, 0.95)
    remaining   = 1.0 - p_chaos
    p_bull      = remaining * trend_str
    p_bear      = remaining * (1.0 - trend_str)
    df_r = pd.DataFrame({'p_bull': p_bull, 'p_bear': p_bear, 'p_chaos': p_chaos,
                         'vol_20': vol}, index=bench_close.index)
    probs = df_r[['p_bull', 'p_bear', 'p_chaos']].values
    df_r['regime'] = np.array(['bull', 'bear', 'chaos'])[np.argmax(probs, axis=1)]
    return df_r

def rolling_beta(sr, br, window=60):
    beta = np.full(len(sr), np.nan)
    sv, bv = sr.values, br.values
    for i in range(window, len(sv)):
        sw, bw = sv[i - window:i], bv[i - window:i]
        if np.std(bw) == 0: continue
        cov = np.cov(sw, bw)[0, 1]
        vb  = np.var(bw)
        beta[i] = cov / vb if vb > 0 else 1.0
    return pd.Series(beta, index=sr.index)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(ds, db, macro, df_regime):
    df = ds.copy()
    df['Bench_Close'] = db['Bench_Close']
    try:
        import ta
        df['RSI']         = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD']        = ta.trend.MACD(df['Close']).macd()
        df['MACD_Sig']    = ta.trend.MACD(df['Close']).macd_signal()
        df['ATR']         = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['CCI']         = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['SMA20']       = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        df['SMA50']       = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
        df['SMA200']      = ta.trend.SMAIndicator(df['Close'], 200).sma_indicator()
        bb                = ta.volatility.BollingerBands(df['Close'])
        df['BB_pct']      = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        df['Log_Ret']     = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_5']       = df['Log_Ret'].rolling(5).std()
        df['Vol_20']      = df['Log_Ret'].rolling(20).std()
        df['Mom_5']       = df['Close'].pct_change(5)
        df['Mom_20']      = df['Close'].pct_change(20)
        df['Vol_ratio']   = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Px_SMA50']    = df['Close'] / df['SMA50']
        df['Px_SMA200']   = df['Close'] / df['SMA200']
        df['B_LogRet']    = np.log(df['Bench_Close'] / df['Bench_Close'].shift(1))
        df['B_Mom5']      = df['Bench_Close'].pct_change(5)
        df['B_Mom20']     = df['Bench_Close'].pct_change(20)
        df['B_Vol20']     = df['B_LogRet'].rolling(20).std()
        df['Rel_Mom5']    = df['Mom_5']  - df['B_Mom5']
        df['Rel_Mom20']   = df['Mom_20'] - df['B_Mom20']
        df['Rel_Vol20']   = df['Vol_20'] - df['B_Vol20']
        df['Rel_Str']     = df['Close']  / df['Bench_Close']
        df['Rel_Str_SMA'] = df['Rel_Str'].rolling(20).mean()
        df['Rel_Str_Dev'] = (df['Rel_Str'] - df['Rel_Str_SMA']) / df['Rel_Str_SMA']
        df['Beta_60']     = rolling_beta(df['Log_Ret'].fillna(0), df['B_LogRet'].fillna(0))
        df['Beta_60']     = df['Beta_60'].clip(0.0, 3.0)
        df['Corr_60']     = df['Log_Ret'].rolling(60).corr(df['B_LogRet'])
        df = df.join(df_regime[['p_bull', 'p_bear', 'p_chaos']], how='left')
        fut_s = np.log(df['Close'].shift(-PREDICTION_HORIZON)       / df['Close'])
        fut_b = np.log(df['Bench_Close'].shift(-PREDICTION_HORIZON) / df['Bench_Close'])
        df['Fut_Stock']   = fut_s
        df['Fut_Bench']   = fut_b
        df['Fut_Resid']   = fut_s - df['Beta_60'] * fut_b
        if not macro.empty:
            df = df.join(macro, how='left').ffill()
        df = df.dropna(subset=['Fut_Resid', 'Beta_60', 'p_bull', 'p_bear', 'p_chaos'])
        df = df.dropna()
        return df
    except Exception as e:
        import traceback; print(f"   Feature Hata: {e}"); traceback.print_exc()
        return None

# ============================================================
# İSTATİSTİK
# ============================================================

def binom_p(c, t):
    if t < 10: return 1.0
    return stats.binomtest(c, t, p=0.5, alternative='greater').pvalue

def info_ratio(rets, ppy):
    arr = np.asarray(rets); arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    sd = np.std(arr, ddof=1)
    return float(np.sqrt(ppy) * np.mean(arr) / sd) if sd > 0 else 0.0

# ============================================================
# MODEL
# ============================================================

def make_dataset(X, y, lb):
    Xs, ys = [], []
    for i in range(lb, len(X)):
        Xs.append(X[i - lb:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm(shape):
    m = Sequential([
        Input(shape=shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    m.compile(optimizer=Adam(0.001), loss=Huber())
    return m

def train_model(Xt, yt, Xv, yv, seed, epochs, sw=None):
    set_seeds(seed)
    m  = build_lstm((Xt.shape[1], Xt.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    kw = dict(epochs=epochs, batch_size=32, validation_data=(Xv, yv),
              callbacks=[es, rl], verbose=0)
    if sw is not None: kw['sample_weight'] = sw
    m.fit(Xt, yt, **kw)
    return m

def train_experts(Xt, yt, Xv, yv, tr_probs):
    experts = {}
    for i, name in enumerate(['bull', 'bear', 'chaos']):
        print(f"      🎓 {name.upper()}...")
        sw    = np.clip(tr_probs[:, i], 0.05, 1.0)
        eff_n = sw.sum()
        if eff_n < 100:
            print(f"         ⚠ yetersiz ({eff_n:.0f})"); experts[name] = None; continue
        models = [train_model(Xt, yt, Xv, yv, seed=sd, epochs=EPOCHS, sw=sw)
                  for sd in ENSEMBLE_SEEDS]
        experts[name] = models
        print(f"         ✓ eff_n={eff_n:.0f}")
    return experts

def predict_soft(experts, X, probs):
    preds = {}
    for i, name in enumerate(['bull', 'bear', 'chaos']):
        if experts.get(name) is None:
            preds[name] = np.zeros(len(X)); continue
        preds[name] = np.mean([m.predict(X, verbose=0).flatten()
                               for m in experts[name]], axis=0)
    return (probs[:, 0] * preds['bull']
          + probs[:, 1] * preds['bear']
          + probs[:, 2] * preds['chaos'])

# ============================================================
# BACKTEST
# ============================================================

def backtest(pred, actual):
    pos   = np.sign(pred)
    raw   = pos * actual
    chg   = np.abs(np.diff(np.concatenate([[0], pos])))
    cost  = chg * (2 * TRANSACTION_COST_BPS / 10000.0)
    daily = raw - cost
    idx   = np.arange(0, len(pred), PREDICTION_HORIZON)
    pos_n = pos[idx]; raw_n = pos_n * actual[idx]
    chg_n = np.abs(np.diff(np.concatenate([[0], pos_n])))
    no    = raw_n - chg_n * (2 * TRANSACTION_COST_BPS / 10000.0)
    return daily, no

# ============================================================
# ANA ANALİZ
# ============================================================

def analyze(ticker, config, macro):
    set_seeds(42)  # FIX-4: izole seed
    name, bench = config['name'], config['benchmark']
    print(f"\n{'='*60}")
    print(f"📊 {ticker} — {name}")
    print(f"{'='*60}")

    ds, db = download_pair(ticker, bench)
    if ds is None: return None
    print(f"   {len(ds)} gün")

    df_regime = detect_regime(db['Bench_Close'])
    cur_reg   = df_regime['regime'].iloc[-1]
    cur_probs = df_regime[['p_bull', 'p_bear', 'p_chaos']].iloc[-1].values
    print(f"   Güncel rejim: {cur_reg.upper()} "
          f"(bull={cur_probs[0]:.2f} bear={cur_probs[1]:.2f} chaos={cur_probs[2]:.2f})")

    df = build_features(ds, db, macro, df_regime)
    if df is None or len(df) < 500: return None
    print(f"   {len(df)} kullanılabilir gün")

    excl = ['Open','High','Low','Volume','Close','Bench_Close','Adj Close',
            'Fut_Stock','Fut_Bench','Fut_Resid']
    fcols = [c for c in df.columns if c not in excl]
    X_arr = df[fcols].values
    y_arr = df['Fut_Resid'].values
    rp    = df[['p_bull','p_bear','p_chaos']].values

    tr = int(len(df) * TRAIN_FRAC)
    vl = int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    print(f"   Split: train={tr} val={vl-tr} test={len(df)-vl}")

    # FIX-1: StandardScaler
    xs = StandardScaler(); xs.fit(X_arr[:tr])
    ys = StandardScaler(); ys.fit(y_arr[:tr].reshape(-1, 1))

    Xtr_s = xs.transform(X_arr[:tr])
    ytr_s = ys.transform(y_arr[:tr].reshape(-1,1)).flatten()
    Xv_s  = xs.transform(X_arr[tr - LOOKBACK:vl])
    yv_s  = ys.transform(y_arr[tr - LOOKBACK:vl].reshape(-1,1)).flatten()
    Xte_s = xs.transform(X_arr[vl - LOOKBACK:])
    yte_s = ys.transform(y_arr[vl - LOOKBACK:].reshape(-1,1)).flatten()

    Xt, yt = make_dataset(Xtr_s, ytr_s, LOOKBACK)
    Xv, yv = make_dataset(Xv_s,  yv_s,  LOOKBACK)
    Xte,yte= make_dataset(Xte_s, yte_s, LOOKBACK)

    # FIX-3: hizalı prob dizileri
    tr_probs = rp[LOOKBACK:tr]          # len = len(Xt) ✓
    xv_probs = rp[tr:vl]               # len = len(Xv) ✓
    te_probs = rp[vl:vl + len(yte)]    # len = len(Xte) ✓

    print(f"\n   Uzman eğitimi ({len(ENSEMBLE_SEEDS)*3} LSTM):")
    experts = train_experts(Xt, yt, Xv, yv, tr_probs)

    pred_s  = predict_soft(experts, Xte, te_probs)
    pred_r  = ys.inverse_transform(pred_s.reshape(-1,1)).flatten()
    act_r   = ys.inverse_transform(yte.reshape(-1,1)).flatten()

    r2      = float(np.corrcoef(act_r, pred_r)[0,1]**2) if np.std(pred_r)>0 else 0.0
    correct = int((np.sign(pred_r) == np.sign(act_r)).sum())
    total   = len(pred_r)
    dir_acc = correct / total * 100
    p_val   = binom_p(correct, total)
    act_pos = (act_r > 0).mean() * 100
    prd_pos = (pred_r > 0).mean() * 100
    naive   = max(act_pos, 100 - act_pos)
    edge    = dir_acc - naive

    print(f"   R²={r2:+.4f}  Yön=%{dir_acc:.1f} ({correct}/{total} p={p_val:.4f})")
    print(f"   Pos oran: gerçek=%{act_pos:.1f} tahmin=%{prd_pos:.1f}  edge=%{edge:+.2f}")

    # Rejim bazlı
    reg_lab = np.array(['bull','bear','chaos'])[np.argmax(te_probs, axis=1)]
    reg_res = {}
    for reg in ['bull','bear','chaos']:
        mask = reg_lab == reg; n = int(mask.sum())
        if n < 10: reg_res[reg] = None; continue
        rc  = int((np.sign(pred_r[mask]) == np.sign(act_r[mask])).sum())
        reg_res[reg] = {'n': n, 'dir': rc/n*100, 'p': binom_p(rc, n)}
    for reg, v in reg_res.items():
        if v: print(f"      {reg.upper():>6}: n={v['n']} yön=%{v['dir']:.1f} p={v['p']:.3f}")

    # FIX-2: non-overlapping backtest
    daily, no = backtest(pred_r, act_r)
    ir_no     = info_ratio(no, 252 / PREDICTION_HORIZON)
    bh_alpha  = (df['Fut_Stock'].iloc[vl:vl+total].values
                 - df['Beta_60'].iloc[vl:vl+total].values
                 * df['Fut_Bench'].iloc[vl:vl+total].values)
    bh_ir     = info_ratio(bh_alpha, 252)
    cum_no    = float((np.exp(no.sum()) - 1) * 100)
    print(f"   IR(NoOver)={ir_no:+.2f}  B&H IR={bh_ir:+.2f}  Cum α(NoOver)=%{cum_no:+.1f}")

    # Gelecek tahmini
    last_X    = xs.transform(X_arr)[-LOOKBACK:].reshape(1, LOOKBACK, len(fcols))
    last_prob = rp[-1:].reshape(1, 3)
    fut_s     = predict_soft(experts, last_X, last_prob)
    fut_ret   = float(ys.inverse_transform(fut_s.reshape(-1,1))[0,0])
    resid_std = float(np.std(act_r - pred_r))
    cur_price = float(df['Close'].iloc[-1])
    cur_bench = float(db['Bench_Close'].iloc[-1])
    last_beta = float(df['Beta_60'].iloc[-1])

    if   abs(fut_ret) < 0.5 * resid_std: sig, sc = "NÖTR",      "gray"
    elif fut_ret > 0:  sig, sc = ("GÜÇLÜ AL","green") if fut_ret>1.5*resid_std else ("AL","blue")
    else:              sig, sc = ("SAT","red")         if fut_ret<-1.5*resid_std else ("ZAYIF SAT","orange")
    print(f"   Tahmin: {fut_ret*100:+.2f}%  Sinyal: {sig}  Rejim: {cur_reg.upper()}")

    return dict(
        ticker=ticker, name=name, bench=bench,
        df=df, df_regime=df_regime,
        pred_r=pred_r, act_r=act_r,
        r2=r2, dir_acc=dir_acc, p_val=p_val,
        act_pos=act_pos, prd_pos=prd_pos, edge=edge,
        reg_res=reg_res,
        ir_no=ir_no, bh_ir=bh_ir, cum_no=cum_no,
        daily=daily, no=no,
        cur_price=cur_price, cur_bench=cur_bench, last_beta=last_beta,
        fut_ret=fut_ret, resid_std=resid_std,
        signal=sig, sig_color=sc,
        cur_reg=cur_reg, cur_probs=cur_probs.tolist(),
        val_split=vl, total=total,
    )

# ============================================================
# ANA GRAFİK
# ============================================================

def make_chart(res):
    """Tek grafik: rejim-renkli fiyat + OOS test tahminleri + gelecek nokta."""
    ticker = res['ticker']; df = res['df']
    vl     = res['val_split']; pred_r = res['pred_r']

    fig, ax = plt.subplots(figsize=(14, 6))
    show_n  = min(600, len(df))
    df_show = df.iloc[-show_n:]

    # Rejim renk bantları
    dom    = np.argmax(df_show[['p_bull','p_bear','p_chaos']].values, axis=1)
    rcol   = {0: '#86efac', 1: '#fca5a5', 2: '#fde047'}
    s, lr  = 0, dom[0]
    for i in range(1, len(dom)):
        if dom[i] != lr:
            ax.axvspan(df_show.index[s], df_show.index[i],
                       alpha=0.13, color=rcol[lr], zorder=0)
            s, lr = i, dom[i]
    ax.axvspan(df_show.index[s], df_show.index[-1],
               alpha=0.13, color=rcol[lr], zorder=0)

    # Gerçek fiyat
    ax.plot(df_show.index, df_show['Close'],
            color='#1f2937', linewidth=1.8, label=f'{ticker} Gerçek')

    # OOS test tahmin çizgisi
    n_te    = len(pred_r)
    tc      = df['Close'].iloc[vl:vl + n_te].values
    tb      = df['Beta_60'].iloc[vl:vl + n_te].values
    tbr     = df['Fut_Bench'].iloc[vl:vl + n_te].values
    pp      = tc * np.exp(pred_r + tb * tbr)
    pd_idx  = df.index[vl:vl + n_te] + pd.Timedelta(days=PREDICTION_HORIZON)
    ax.plot(pd_idx, pp, linestyle='--', color='#f59e0b',
            linewidth=1.4, alpha=0.85, label=f'Model OOS ({PREDICTION_HORIZON}g)')

    # Test başlangıç çizgisi
    ax.axvline(df.index[vl], color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label='Test başlangıcı')

    # Gelecek tahmin noktası (SOXX=0 varsayımı)
    fut_date  = df.index[-1] + timedelta(days=PREDICTION_HORIZON)
    fut_price = res['cur_price'] * np.exp(res['fut_ret'])
    ax.scatter([fut_date], [fut_price], color=res['sig_color'],
               s=160, zorder=10, edgecolors='black', linewidth=1.5,
               label=f'Gelecek {PREDICTION_HORIZON}g ({res["signal"]})')

    ax.set_title(
        f"{res['name']} ({ticker})   |   "
        f"Rejim: {res['cur_reg'].upper()}   |   "
        f"Sinyal: {res['signal']}   |   "
        f"IR(NoOver): {res['ir_no']:+.2f}   |   "
        f"Yön: %{res['dir_acc']:.1f}   |   "
        f"Edge: %{res['edge']:+.2f}\n"
        f"🟢 Bull   🔴 Bear   🟡 Chaos   |   "
        f"Turuncu kesikli = OOS model tahmini   |   "
        f"Kırmızı dikey = test başlangıcı",
        fontsize=11, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylabel(f'{ticker} ($)')
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================
# TEK HTML — TÜM HİSSELER SIRALI
# ============================================================

CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #f1f5f9; padding: 24px; margin: 0; }
.page { max-width: 1280px; margin: 0 auto; }
h1 { text-align: center; color: #0f172a; font-size: 1.6em; margin-bottom: 8px; }
.subtitle { text-align: center; color: #64748b; font-size: .9em; margin-bottom: 32px; }
/* Hisse kartı */
.card { background: #fff; border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
        margin-bottom: 36px; overflow: hidden; }
.card-header { padding: 14px 24px; display: flex; align-items: center;
               justify-content: space-between; background: #0f172a; color: #fff; }
.card-header .ticker { font-size: 1.4em; font-weight: 700; }
.card-header .regime-badge { font-size: .8em; padding: 4px 12px; border-radius: 20px;
                              font-weight: 600; }
.bull-badge  { background: #16a34a; }
.bear-badge  { background: #dc2626; }
.chaos-badge { background: #ca8a04; color: #fff; }
/* Sinyal */
.signal-bar { padding: 14px 24px; text-align: center; font-weight: 700;
              font-size: 1.1em; letter-spacing: .5px; }
/* Metrik grid */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr);
           gap: 1px; background: #e2e8f0; border-top: 1px solid #e2e8f0; }
.m { background: #fff; padding: 14px; text-align: center; }
.m-label { font-size: .72em; color: #64748b; font-weight: 700;
           text-transform: uppercase; letter-spacing: .5px; }
.m-val   { font-size: 1.25em; font-weight: 800; margin: 4px 0; color: #0f172a; }
.m-sub   { font-size: .68em; color: #94a3b8; }
/* Rejim tablosu */
.reg-table { padding: 16px 24px; background: #f8fafc; border-top: 1px solid #e2e8f0; }
.reg-table table { width: 100%; border-collapse: collapse; font-size: .88em; }
.reg-table th { background: #f1f5f9; padding: 7px 10px; font-weight: 700;
                color: #374151; border: 1px solid #e2e8f0; }
.reg-table td { padding: 7px 10px; border: 1px solid #e2e8f0; text-align: center; }
.bull-row { background: #f0fdf4; }
.bear-row { background: #fef2f2; }
.chaos-row{ background: #fefce8; }
/* Grafik */
.chart-wrap { padding: 20px 24px; background: #f8fafc;
              border-top: 1px solid #e2e8f0; text-align: center; }
.chart-wrap img { max-width: 100%; border-radius: 10px;
                  border: 1px solid #e2e8f0; }
/* Senaryo tablosu */
.scen { padding: 16px 24px; border-top: 1px solid #e2e8f0; }
.scen h4 { margin: 0 0 10px; font-size: .9em; color: #475569; }
.scen table { width: 100%; border-collapse: collapse; font-size: .87em; }
.scen th { background: #f1f5f9; padding: 6px 10px; border: 1px solid #e2e8f0; }
.scen td { padding: 6px 10px; border: 1px solid #e2e8f0; text-align: center; }
.pos { background: #f0fdf4; color: #15803d; font-weight: 600; }
.neg { background: #fef2f2; color: #b91c1c; font-weight: 600; }
/* Uyarı kutusu */
.note { margin: 8px 24px 16px; padding: 10px 14px; border-radius: 8px;
        font-size: .82em; background: #fefce8; border-left: 4px solid #ca8a04;
        color: #713f12; }
/* Özet tablosu */
.summary-wrap { background: #fff; border-radius: 14px;
                box-shadow: 0 2px 8px rgba(0,0,0,.08);
                margin-bottom: 36px; overflow: hidden; }
.summary-wrap .sh { background: #0f172a; color: #fff; padding: 14px 24px;
                    font-size: 1.1em; font-weight: 700; }
.summary-wrap table { width: 100%; border-collapse: collapse; font-size: .88em; }
.summary-wrap th { background: #f1f5f9; padding: 9px 12px;
                   border: 1px solid #e2e8f0; font-weight: 700; color: #374151; }
.summary-wrap td { padding: 9px 12px; border: 1px solid #e2e8f0; text-align: center; }
.green { color: #15803d; font-weight: 700; }
.orange{ color: #ca8a04; font-weight: 700; }
.red   { color: #b91c1c; font-weight: 700; }
</style>
"""

def sig_bg(sc):
    return {'green':'#dcfce7','blue':'#dbeafe','gray':'#f1f5f9',
            'orange':'#ffedd5','red':'#fee2e2'}.get(sc,'#f1f5f9')

def edge_cls(e):
    return 'green' if e > 3 else 'orange' if e > 0 else 'red'

def dir_cls(d):
    return 'green' if d > 55 else 'orange' if d > 50 else 'red'

def build_html(results):
    valid = [r for r in results if r is not None]

    # ── Özet tablosu ──
    sum_rows = ""
    for r in valid:
        ec = edge_cls(r['edge']); dc = dir_cls(r['dir_acc'])
        rb = f"{r['cur_reg']}-badge"
        sum_rows += (
            f"<tr>"
            f"<td><b>{r['ticker']}</b><br><span style='font-size:.8em;color:#64748b'>{r['name']}</span></td>"
            f"<td><span class='regime-badge {rb}' style='padding:3px 8px;border-radius:10px;"
            f"font-size:.78em;font-weight:700'>{r['cur_reg'].upper()}</span></td>"
            f"<td class='{dc}'>%{r['dir_acc']:.1f}<br>"
            f"<span style='font-size:.75em;font-weight:400'>p={r['p_val']:.3f}</span></td>"
            f"<td>{r['r2']:+.4f}</td>"
            f"<td class='{ec}'>%{r['edge']:+.2f}</td>"
            f"<td>{r['ir_no']:+.2f}</td>"
            f"<td>{r['bh_ir']:+.2f}</td>"
            f"<td>%{r['cum_no']:+.1f}</td>"
            f"<td style='color:{r['sig_color']};font-weight:700'>{r['signal']}</td>"
            f"</tr>"
        )

    summary_html = f"""
<div class="summary-wrap">
  <div class="sh">📊 Özet Karşılaştırma</div>
  <div style="padding:16px 24px;overflow-x:auto">
  <table>
  <tr><th>Ticker</th><th>Rejim</th><th>Yön Doğr.</th><th>R²</th>
      <th>Alfa Edge</th><th>IR (NoOver)</th><th>Passive IR</th>
      <th>Cum α (NoOver)</th><th>Sinyal</th></tr>
  {sum_rows}
  </table>
  </div>
  <div class="note">
    <b>Metrik rehberi:</b>
    Yön Doğr. = OOS test setinde gerçek residual yönü doğru tahmin oranı.
    Alfa Edge = yön doğruluğu − naive baseline (bias-free ortamda gerçek bilgi).
    IR (NoOver) = non-overlapping {PREDICTION_HORIZON}g periyotlarda Information Ratio (gerçekçi).
    Cum α (NoOver) = toplam portföy büyümesi (bağımsız periyotlar toplamı).
    Passive IR = modelsiz sadece hisse long + beta×SOXX short tutulursa IR.
  </div>
</div>"""

    # ── Hisse kartları ──
    cards_html = ""
    for r in valid:
        chart_b64 = make_chart(r)

        # Rejim performans tablosu
        reg_rows = ""
        for reg in ['bull', 'bear', 'chaos']:
            rv  = r['reg_res'].get(reg); cls = f"{reg}-row"
            if rv is None:
                reg_rows += f"<tr class='{cls}'><td><b>{reg.upper()}</b></td><td colspan=3 style='color:#94a3b8'>Yetersiz veri</td></tr>"
            else:
                dc = dir_cls(rv['dir'])
                reg_rows += (f"<tr class='{cls}'>"
                             f"<td><b>{reg.upper()}</b></td>"
                             f"<td>{rv['n']}</td>"
                             f"<td class='{dc}'>%{rv['dir']:.1f}</td>"
                             f"<td>{rv['p']:.3f}</td></tr>")

        # Senaryo tablosu
        scen_rows = ""
        for sm in [-0.04, -0.02, 0.0, 0.02, 0.04]:
            im = r['fut_ret'] + r['last_beta'] * sm
            ip = r['cur_price'] * np.exp(im)
            cls = 'pos' if im > 0 else 'neg'
            scen_rows += (f"<tr>"
                          f"<td>{sm*100:+.1f}%</td>"
                          f"<td class='{cls}'>{im*100:+.2f}%</td>"
                          f"<td class='{cls}'>${ip:.2f}</td></tr>")

        sig_bar_bg = sig_bg(r['sig_color'])
        rb = f"{r['cur_reg']}-badge"
        avc = edge_cls(r['edge'])
        avt = 'GERÇEK ALFA' if r['edge']>3 else 'ZAYIF ALFA' if r['edge']>0 else 'ALFA YOK'

        cards_html += f"""
<div class="card">
  <!-- Başlık -->
  <div class="card-header">
    <span class="ticker">{r['ticker']} — {r['name']}</span>
    <span class="regime-badge {rb}">{r['cur_reg'].upper()}
      &nbsp;|&nbsp; bull={r['cur_probs'][0]:.2f} bear={r['cur_probs'][1]:.2f} chaos={r['cur_probs'][2]:.2f}</span>
  </div>

  <!-- Sinyal -->
  <div class="signal-bar" style="background:{sig_bar_bg};color:{r['sig_color']}">
    {r['signal']}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Residual tahmin: {r['fut_ret']*100:+.2f}%
    &nbsp;(β={r['last_beta']:.2f}, benchmark={r['bench']})
  </div>

  <!-- Metrik grid -->
  <div class="metrics">
    <div class="m">
      <div class="m-label">Mevcut Fiyat</div>
      <div class="m-val">${r['cur_price']:.2f}</div>
      <div class="m-sub">{r['bench']}: ${r['cur_bench']:.2f}</div>
    </div>
    <div class="m">
      <div class="m-label">Yön Doğruluğu</div>
      <div class="m-val" style="color:{['#b91c1c','#ca8a04','#15803d'][0 if r['dir_acc']<50 else 1 if r['dir_acc']<55 else 2]}">
        %{r['dir_acc']:.1f}</div>
      <div class="m-sub">{r['total']} OOS örnek · p={r['p_val']:.3f}</div>
    </div>
    <div class="m">
      <div class="m-label">Alfa Edge</div>
      <div class="m-val" style="color:{'#15803d' if r['edge']>3 else '#ca8a04' if r['edge']>0 else '#b91c1c'}">
        %{r['edge']:+.2f}</div>
      <div class="m-sub">{avt}</div>
    </div>
    <div class="m">
      <div class="m-label">R² (Residual)</div>
      <div class="m-val">{r['r2']:+.4f}</div>
      <div class="m-sub">OOS test seti</div>
    </div>
    <div class="m">
      <div class="m-label">IR (NoOver) ✓</div>
      <div class="m-val" style="color:{'#15803d' if r['ir_no']>0.3 else '#ca8a04' if r['ir_no']>0 else '#b91c1c'}">
        {r['ir_no']:+.2f}</div>
      <div class="m-sub">Gerçekçi bağımsız</div>
    </div>
    <div class="m">
      <div class="m-label">Passive IR</div>
      <div class="m-val">{r['bh_ir']:+.2f}</div>
      <div class="m-sub">Modelsiz B&H</div>
    </div>
    <div class="m">
      <div class="m-label">Cum α (NoOver)</div>
      <div class="m-val" style="color:{'#15803d' if r['cum_no']>0 else '#b91c1c'}">
        %{r['cum_no']:+.1f}</div>
      <div class="m-sub">Bağımsız periyot</div>
    </div>
    <div class="m">
      <div class="m-label">Pos Oran Δ</div>
      <div class="m-val">%{abs(r['prd_pos']-r['act_pos']):.1f}</div>
      <div class="m-sub">tahmin-gerçek farkı</div>
    </div>
  </div>

  <!-- Grafik -->
  <div class="chart-wrap">
    <img src="data:image/png;base64,{chart_b64}" alt="{r['ticker']} grafik">
  </div>

  <!-- Rejim tablosu + Senaryo yan yana -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
    <div class="reg-table">
      <b style="font-size:.85em;color:#475569">🎭 Rejim-Bazlı OOS Performans</b>
      <table style="margin-top:8px">
        <tr><th>Rejim</th><th>n</th><th>Yön %</th><th>p</th></tr>
        {reg_rows}
      </table>
    </div>
    <div class="scen">
      <h4>📐 Senaryo Tablosu ({r['bench']} hareketine göre)</h4>
      <table>
        <tr><th>{r['bench']} 7g</th><th>{r['ticker']} Tahmin</th><th>Fiyat</th></tr>
        {scen_rows}
      </table>
    </div>
  </div>

</div>"""

    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    tickers_str = ', '.join(r['ticker'] for r in valid)

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Stock Analiz v15.2 — {tickers_str}</title>
  {CSS}
</head>
<body>
<div class="page">

  <h1>🔬 Multi-Stock Regime-Switching Analiz — v15.2</h1>
  <div class="subtitle">
    {tickers_str} &nbsp;·&nbsp; Market-neutral residual return &nbsp;·&nbsp;
    Soft ensemble 3 rejim &nbsp;·&nbsp; {now}
  </div>

  {summary_html}

  {cards_html}

  <div class="note" style="margin:0 0 24px">
    <b>v15.2 metodoloji notu:</b>
    Hedef = log(hisse<sub>t+{PREDICTION_HORIZON}</sub>/hisse<sub>t</sub>) − β × log(benchmark<sub>t+{PREDICTION_HORIZON}</sub>/benchmark<sub>t</sub>).
    Bu formül bull market bias'ını yapısal olarak kaldırır.
    Scaler: StandardScaler (FIX-1).
    Backtest: non-overlapping {PREDICTION_HORIZON}g bağımsız periyotlar (FIX-2).
    Rejim prob hizalaması düzeltildi (FIX-3). Seed izolasyonu her hisse için (FIX-4).
  </div>

</div>
</body>
</html>"""

# ============================================================
# ANA AKIŞ
# ============================================================

def main():
    set_seeds()
    init_db()

    tickers = list(TICKERS_CONFIG.keys())
    n_lstm  = len(tickers) * len(ENSEMBLE_SEEDS) * 3

    print("=" * 60)
    print(f"MULTI-STOCK v15.2  |  {', '.join(tickers)}")
    print("=" * 60)
    print(f"Hisseler : {tickers}")
    print(f"Ufuk     : {PREDICTION_HORIZON} gün")
    print(f"Toplam LSTM: {n_lstm}")

    macro = get_macro()

    results = []
    for ticker, cfg in TICKERS_CONFIG.items():
        res = analyze(ticker, cfg, macro)
        results.append(res)

    # Tek HTML
    print(f"\n📄 HTML oluşturuluyor → {OUTPUT_HTML}")
    html = build_html(results)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✓ {OUTPUT_HTML}")

    # Konsol özet
    print("\n" + "=" * 60)
    print(f"{'TICKER':<6} {'YÖN%':<7} {'EDGE%':<8} {'IR':<7} {'CUM_NO%':<10} {'SİNYAL'}")
    print("-" * 60)
    for r in results:
        if r is None: continue
        print(f"{r['ticker']:<6} %{r['dir_acc']:<5.1f} %{r['edge']:<+6.2f}  "
              f"{r['ir_no']:<+5.2f}  %{r['cum_no']:<+8.1f}  {r['signal']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
