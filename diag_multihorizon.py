# ============================================================
# TEŞHİS v2: ÇOK-UFUKLU forward EĞİM YÖNÜ — öğrenilebilir mi?
# ============================================================
# Önceki teşhis 15g'de "yön öğrenilemiyor" dedi. Hipotez: ufuk uzadıkça
# günlük gürültü ortalanır, trend bileşeni güçlenir (hisselerin
# DFA-Hurst'ü 0.86-0.95 → güçlü trend). Bu scripti AYNI leakage'sız
# mantıkla 15/30/45/60g ufuklarda çalıştırır ve hangi ufukta (varsa)
# sinyal belirdiğini gösterir.
#
# Her ufuk için: 3 sınıf (Yukarı/Yatay/Aşağı), eşik SADECE train'den,
# zamansal split, permütasyon testi (gerçek sinyal vs şans).
# Model = LogReg (ucuz). Sinyal çıkarsa → LSTM'e değer.
#
# ÖNEMLİ: Ufuk uzadıkça bağımsız gözlem azalır (2800g/60≈47 pencere),
# bu yüzden permütasyon p-değeri uzun ufukta daha gürültülü olur.
# "n_indep" sütunu bunu gösterir; düşükse sonuca temkinli yaklaş.
# ============================================================

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif

TICKERS    = ['INTC', 'AMD', 'NVDA']
BENCH      = 'SOXX'
HORIZONS   = [15, 30, 45, 60]   # ← çok-ufuklu test
TRAIN_FRAC = 0.70
FLAT_K     = 0.5
N_PERM     = 200

def download(tk):
    end = datetime.now(); start = end - timedelta(days=12*365)
    df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty: return None
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    return df[['Open','High','Low','Close','Volume']].dropna()

def forward_slope(close, h):
    """Önümüzdeki h günün log-fiyatına uydurulan doğrunun eğimi (leakage kontrollü)."""
    logp = np.log(close.values); n = len(logp); out = np.full(n, np.nan)
    x = np.arange(h); x = x - x.mean(); denom = (x**2).sum()
    for i in range(n - h):
        seg = logp[i+1:i+1+h]
        if len(seg) < h: continue
        y = seg - seg.mean(); out[i] = (x * y).sum() / denom
    return pd.Series(out, index=close.index)

def build_features(df, bench):
    """Yalnızca GEÇMİŞ bilgi kullanan özellikler (t anına kadar)."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']; lr = np.log(c/c.shift(1))
    f['mom_5']=c.pct_change(5); f['mom_10']=c.pct_change(10)
    f['mom_20']=c.pct_change(20); f['mom_60']=c.pct_change(60)
    f['past_slope_15']=lr.rolling(15).mean(); f['past_slope_30']=lr.rolling(30).mean()
    f['past_slope_60']=lr.rolling(60).mean()
    f['px_sma20']=c/c.rolling(20).mean()-1; f['px_sma50']=c/c.rolling(50).mean()-1
    f['px_sma200']=c/c.rolling(200).mean()-1; f['sma20_50']=c.rolling(20).mean()/c.rolling(50).mean()-1
    f['sma50_200']=c.rolling(50).mean()/c.rolling(200).mean()-1
    f['vol_20']=lr.rolling(20).std(); f['vol_60']=lr.rolling(60).std()
    f['vol_ratio']=lr.rolling(5).std()/(lr.rolling(20).std()+1e-9)
    delta=c.diff(); up=delta.clip(lower=0).rolling(14).mean(); dn=(-delta.clip(upper=0)).rolling(14).mean()
    f['rsi']=100-100/(1+up/(dn+1e-9))
    hi=df['High'].rolling(60).max(); lo=df['Low'].rolling(60).min()
    f['range_pos']=(c-lo)/((hi-lo)+1e-9)
    f['vol_z']=(df['Volume']-df['Volume'].rolling(20).mean())/(df['Volume'].rolling(20).std()+1e-9)
    bc=bench['Close'].reindex(df.index, method='ffill')
    f['rel_mom20']=c.pct_change(20)-bc.pct_change(20)
    f['rel_str']=(c/bc).pct_change(20)
    return f

def eval_horizon(feat, slope, h):
    """Tek ufuk için leakage'sız değerlendirme. Döner: bacc, chance, base_acc, p, n, n_indep, top_mi"""
    data = feat.copy(); data['__slope__'] = slope; data = data.dropna()
    n = len(data); tr_end = int(n*TRAIN_FRAC)
    if tr_end < 200 or (n-tr_end) < 100: return None
    s = data['__slope__']; tr_std = s.iloc[:tr_end].std(); z = s/(tr_std+1e-12)
    lab = pd.Series(1, index=s.index); lab[z>FLAT_K]=2; lab[z<-FLAT_K]=0
    data['__y__'] = lab; data = data.dropna()
    n = len(data); tr_end = int(n*TRAIN_FRAC)
    X = data.drop(columns=['__slope__','__y__']).values
    y = data['__y__'].astype(int).values
    Xtr,Xte=X[:tr_end],X[tr_end:]; ytr,yte=y[:tr_end],y[tr_end:]
    cls,cnt=np.unique(ytr,return_counts=True)
    sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
    clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Xtr_s,ytr)
    pred=clf.predict(Xte_s)
    bacc=balanced_accuracy_score(yte,pred); acc=accuracy_score(yte,pred)
    maj=cls[np.argmax(cnt)]; base_acc=accuracy_score(yte,np.full_like(yte,maj))
    chance=1.0/len(np.unique(y))
    rng=np.random.default_rng(42); pa=[]
    for _ in range(N_PERM):
        c2=LogisticRegression(max_iter=400,class_weight='balanced')
        try: c2.fit(Xtr_s,rng.permutation(ytr)); pa.append(balanced_accuracy_score(yte,c2.predict(Xte_s)))
        except: pass
    p=(np.array(pa)>=bacc).mean()
    mi=mutual_info_classif(Xtr_s,ytr,random_state=42)
    fnames=list(data.drop(columns=['__slope__','__y__']).columns)
    top=sorted(zip(fnames,mi),key=lambda z:-z[1])[:3]
    return dict(bacc=bacc,acc=acc,chance=chance,base_acc=base_acc,p=p,
                n=n,n_indep=(n-tr_end)//h,top=top,
                dist={int(c):int(k) for c,k in zip(cls,cnt)})

print("="*72)
print("TEŞHİS v2: ÇOK-UFUKLU forward EĞİM YÖNÜ — öğrenilebilir mi?")
print("="*72)
print(f"Ufuklar={HORIZONS}g | Yatay=±{FLAT_K}σ | LogReg | perm n={N_PERM} | train=%{int(TRAIN_FRAC*100)}\n")

bench = download(BENCH)
all_rows = []

for tk in TICKERS:
    print("-"*72); print(f"📊 {tk}"); print("-"*72)
    df = download(tk)
    if df is None: print("  veri yok"); continue
    feat = build_features(df, bench)
    for h in HORIZONS:
        slope = forward_slope(df['Close'], h)
        r = eval_horizon(feat, slope, h)
        if r is None: print(f"  {h:>2}g: yetersiz veri"); continue
        edge = r['bacc']-r['chance']
        sig = '✅ SİNYAL' if (r['p']<0.05 and edge>0.02) else '❌ yok'
        names={0:'Aşağı',1:'Yatay',2:'Yukarı'}
        dstr=" ".join(f"{names[c]}%{100*r['dist'].get(c,0)/sum(r['dist'].values()):.0f}" for c in [0,1,2])
        topstr=", ".join(f"{n}={v:.3f}" for n,v in r['top'])
        print(f"  {h:>2}g: bal-acc={r['bacc']:.3f} (şans={r['chance']:.3f}, çoğ={r['base_acc']:.3f}) "
              f"edge={edge:+.3f} p={r['p']:.3f} {sig}")
        print(f"        n_indep={r['n_indep']:<4} dağılım[{dstr}]  MI: {topstr}")
        all_rows.append((tk,h,r['bacc'],r['chance'],r['base_acc'],edge,r['p'],r['n_indep'],sig))

print("\n" + "="*72)
print("ÖZET TABLOSU")
print("="*72)
print(f"{'Hisse':<6}{'Ufuk':<6}{'bal-acc':<9}{'çoğunluk':<10}{'edge':<9}{'p':<8}{'n_ind':<7}sonuç")
print("-"*72)
for tk,h,b,ch,ba,e,p,ni,sig in all_rows:
    print(f"{tk:<6}{h:<6}{b:<9.3f}{ba:<10.3f}{e:<+9.3f}{p:<8.3f}{ni:<7}{sig}")
print("="*72)

# en iyi ufku öne çıkar
best = [r for r in all_rows if r[8].startswith('✅')]
print("\nKARAR:")
if best:
    best.sort(key=lambda z:-z[5])
    print(f"  ✅ Sinyal bulunan ufuk(lar) VAR. En güçlü: ", end="")
    print(", ".join(f"{r[0]}@{r[1]}g (edge={r[5]:+.3f}, p={r[6]:.3f})" for r in best[:4]))
    print("  → Bu ufukta tam LSTM modeli kurmaya DEĞER.")
else:
    print("  ❌ Hiçbir hissede, hiçbir ufukta anlamlı yön sinyali YOK.")
    print("  → Yön tahmini bu özelliklerle bu hisselerde öğrenilemiyor.")
    print("  → Seçenekler: (a) belirgin-hareket filtresi, (b) volatiliteye dön,")
    print("     (c) farklı varlık sınıfı/özellik ailesi. Yön ufku tek başına çözmüyor.")
print("="*72)
