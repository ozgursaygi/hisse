# ============================================================
# TEŞHİS: 15 günlük forward EĞİM YÖNÜ öğrenilebilir mi?
# ============================================================
# Amaç: TAM modeli (135 LSTM) kurmadan ÖNCE, ucuz bir testle şunu ölç:
#   - Hedef = sign(önümüzdeki 15 günün kapanışlarına uydurulan doğrunun eğimi)
#     3 sınıf: Yukarı / Yatay / Aşağı  (Yatay = |eğim| < tarihsel std)
#   - "benzer geçmiş → benzer gelecek" hipotezi gerçekten sinyal taşıyor mu?
#
# Yöntem (LEAKAGE'SIZ):
#   - Özellikler yalnızca GEÇMİŞE bakar (t anına kadar).
#   - Hedef GELECEĞE bakar (t+1..t+15), shift(-) ile; son 15 gün düşürülür.
#   - Zamansal train/test ayrımı (%70/%30, karıştırma YOK).
#   - Baseline = "her zaman çoğunluk sınıfını tahmin et".
#   - Model = basit Logistic Regression (LSTM değil — ucuz, hızlı, sinyal var mı yok mu görmek için yeterli).
#   - Ek olarak: özellik-hedef mutual information ve permütasyon testi.
# ============================================================

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

TICKERS   = ['INTC', 'AMD', 'NVDA']
BENCH     = 'SOXX'
SLOPE_H   = 15        # forward eğim ufku (gün)
LOOKBACK_FEAT = 60    # özelliklerin baktığı geçmiş pencere
TRAIN_FRAC = 0.70
FLAT_K    = 0.5       # Yatay eşiği: |z-eğim| < FLAT_K*std → Yatay
N_PERM    = 200       # permütasyon testi tekrarı

def download(tk):
    end = datetime.now(); start = end - timedelta(days=12*365)
    df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty: return None
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    return df[['Open','High','Low','Close','Volume']].dropna()

def forward_slope(close, h=SLOPE_H):
    """Önümüzdeki h günün log-fiyatına uydurulan doğrunun eğimi (leakage kontrollü)."""
    logp = np.log(close.values)
    n = len(logp)
    out = np.full(n, np.nan)
    x = np.arange(h)
    x = (x - x.mean())
    denom = (x**2).sum()
    for i in range(n - h):
        seg = logp[i+1:i+1+h]          # t+1..t+h  (GELECEK)
        if len(seg) < h: continue
        y = seg - seg.mean()
        out[i] = (x * y).sum() / denom  # eğim (günlük log-getiri/gün)
    return pd.Series(out, index=close.index)

def build_features(df, bench):
    """Yalnızca GEÇMİŞ bilgi kullanan özellikler (t anına kadar)."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']; lr = np.log(c/c.shift(1))
    # momentum / trend (geçmiş)
    f['mom_5']   = c.pct_change(5)
    f['mom_10']  = c.pct_change(10)
    f['mom_20']  = c.pct_change(20)
    f['mom_60']  = c.pct_change(60)
    # geçmiş eğim (son 15g) — "benzer geçmiş" çekirdeği
    f['past_slope_15'] = lr.rolling(15).mean()
    f['past_slope_30'] = lr.rolling(30).mean()
    # SMA konumu
    f['px_sma20']  = c/c.rolling(20).mean() - 1
    f['px_sma50']  = c/c.rolling(50).mean() - 1
    f['px_sma200'] = c/c.rolling(200).mean() - 1
    f['sma20_50']  = c.rolling(20).mean()/c.rolling(50).mean() - 1
    # volatilite rejimi
    f['vol_20']  = lr.rolling(20).std()
    f['vol_ratio'] = lr.rolling(5).std()/(lr.rolling(20).std()+1e-9)
    # RSI benzeri
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi'] = 100 - 100/(1 + up/(dn+1e-9))
    # range konumu (60g min/max retracement — TA dürüst versiyon)
    hi = df['High'].rolling(60).max(); lo = df['Low'].rolling(60).min()
    f['range_pos'] = (c - lo)/((hi-lo)+1e-9)
    # hacim
    f['vol_z'] = (df['Volume'] - df['Volume'].rolling(20).mean())/(df['Volume'].rolling(20).std()+1e-9)
    # benchmark göreceli
    bc = bench['Close'].reindex(df.index, method='ffill')
    f['rel_mom20'] = c.pct_change(20) - bc.pct_change(20)
    f['rel_str']   = (c/bc).pct_change(20)
    return f

def make_labels(slope, train_end_idx):
    """3 sınıf: eğimi TRAIN'deki std'ye göre standartlaştır, Yatay bandı uygula.
    Eşik SADECE train'den hesaplanır (leakage yok)."""
    s = slope.copy()
    tr_std = s.iloc[:train_end_idx].std()
    z = s / (tr_std + 1e-12)
    lab = pd.Series(1, index=s.index)         # 1 = Yatay
    lab[z >  FLAT_K] = 2                       # 2 = Yukarı
    lab[z < -FLAT_K] = 0                       # 0 = Aşağı
    lab[s.isna()] = np.nan
    return lab, tr_std

print("="*64)
print("TEŞHİS: 15 günlük forward EĞİM YÖNÜ — öğrenilebilir mi?")
print("="*64)
print(f"Ufuk={SLOPE_H}g | Yatay eşiği=±{FLAT_K}σ | Model=LogReg (ucuz teşhis)")
print(f"Train/Test=%{int(TRAIN_FRAC*100)}/%{int((1-TRAIN_FRAC)*100)} zamansal | Permütasyon n={N_PERM}\n")

bench = download(BENCH)
summary = []

for tk in TICKERS:
    print("-"*64); print(f"📊 {tk}"); print("-"*64)
    df = download(tk)
    if df is None: print("  veri yok"); continue

    feat = build_features(df, bench)
    slope = forward_slope(df['Close'])
    data = feat.copy(); data['__slope__'] = slope
    data = data.dropna()
    n = len(data)
    tr_end = int(n*TRAIN_FRAC)

    lab, tr_std = make_labels(data['__slope__'], tr_end)
    data['__y__'] = lab
    data = data.dropna()
    n = len(data); tr_end = int(n*TRAIN_FRAC)

    X = data.drop(columns=['__slope__','__y__']).values
    y = data['__y__'].astype(int).values
    Xtr, Xte = X[:tr_end], X[tr_end:]
    ytr, yte = y[:tr_end], y[tr_end:]

    # sınıf dağılımı
    cls, cnt = np.unique(ytr, return_counts=True)
    dist = {int(c): int(n) for c,n in zip(cls,cnt)}
    names = {0:'Aşağı',1:'Yatay',2:'Yukarı'}
    dist_str = " ".join(f"{names[c]}=%{100*dist.get(c,0)/len(ytr):.0f}" for c in [0,1,2])
    print(f"  Kullanılabilir: {n}g (train={tr_end}, test={n-tr_end})")
    print(f"  Train sınıf dağılımı: {dist_str}")

    # ölçekle + eğit
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)

    acc  = accuracy_score(yte, pred)
    bacc = balanced_accuracy_score(yte, pred)

    # baseline: her zaman çoğunluk sınıfı (train'den)
    maj = cls[np.argmax(cnt)]
    base_acc = accuracy_score(yte, np.full_like(yte, maj))
    # dengeli baseline = 1/sınıf sayısı
    n_cls = len(np.unique(y))
    chance = 1.0/n_cls

    # permütasyon testi: y'yi karıştırıp acc dağılımı
    rng = np.random.default_rng(42)
    perm_acc = []
    for _ in range(N_PERM):
        yp = rng.permutation(ytr)
        c2 = LogisticRegression(max_iter=500, class_weight='balanced')
        try:
            c2.fit(Xtr_s, yp)
            perm_acc.append(balanced_accuracy_score(yte, c2.predict(Xte_s)))
        except: pass
    perm_acc = np.array(perm_acc)
    p_val = (perm_acc >= bacc).mean()

    # mutual information (özellik–hedef bağı)
    mi = mutual_info_classif(Xtr_s, ytr, random_state=42)
    fnames = list(data.drop(columns=['__slope__','__y__']).columns)
    top = sorted(zip(fnames, mi), key=lambda z:-z[1])[:5]

    print(f"  Model bal-acc = {bacc:.3f}  (şans={chance:.3f}, çoğunluk-acc={base_acc:.3f})")
    print(f"  Ham acc       = {acc:.3f}")
    print(f"  Permütasyon p = {p_val:.3f}  ({'✅ ANLAMLI' if p_val<0.05 else '❌ anlamsız'})")
    print(f"  En bilgili 5 özellik (MI): " + ", ".join(f"{n}={v:.3f}" for n,v in top))
    edge = bacc - chance
    summary.append((tk, bacc, chance, base_acc, p_val, edge))

print("\n" + "="*64)
print("ÖZET")
print("="*64)
print(f"{'Hisse':<6} {'bal-acc':<9} {'şans':<7} {'çoğunluk':<9} {'p-değeri':<10} {'edge':<8} sonuç")
print("-"*64)
for tk,b,ch,ba,p,e in summary:
    res = '✅ sinyal var' if (p<0.05 and e>0.02) else '❌ sinyal yok/zayıf'
    print(f"{tk:<6} {b:<9.3f} {ch:<7.3f} {ba:<9.3f} {p:<10.3f} {e:<+8.3f} {res}")
print("="*64)
print("\nYORUM:")
print("  edge = bal-acc - şans. >0.02 ve p<0.05 ise: öğrenilebilir sinyal VAR.")
print("  Hepsi ❌ ise: 15g eğim de bu özelliklerle öğrenilemiyor → tanımı değiştir.")
print("  LogReg sinyal buluyorsa, LSTM muhtemelen daha iyisini bulur → tam modele değer.")
