# ============================================================
# INTC v13.1 SAĞLIK KONTROLÜ
# 
# Amaç: %62 yön doğruluğu GERÇEK alpha mı yoksa BİAS mı?
# 
# Test edilecek hipotezler:
#   H1: Model her zaman pozitif tahmin yaparak yüksek doğruluk alıyor
#   H2: Test periyodu pozitif yön ağırlıklı (bull market bias)
#   H3: Naive baseline'lar (momentum, mean, sign-of-last-N) %62'ye yaklaşıyor
#   H4: Model gerçekten bilgi öğreniyor → naive baseline'lardan ANLAMLI ÜSTÜN
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

PREDICTION_HORIZON = 7
TICKER = 'INTC'

print("="*65)
print(f"INTC SAĞLIK KONTROLÜ - %62 doğruluk gerçek mi yoksa bias mı?")
print("="*65)

# Veri çek (v13.1 ile aynı setup)
end = datetime.now()
start = end - timedelta(days=12*365)
df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
if hasattr(df.columns, 'get_level_values'):
    df.columns = df.columns.get_level_values(0)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 7-günlük gelecek log-getiri (aynı target)
df['Future_Ret'] = np.log(df['Close'].shift(-PREDICTION_HORIZON) / df['Close'])
df = df.dropna(subset=['Future_Ret'])

# Test set: v13.1 ile aynı bölünme (son %10)
val_split = int(len(df) * 0.90)
test_df = df.iloc[val_split:].copy()
test_ret = test_df['Future_Ret'].values
n = len(test_ret)

print(f"\nTest set boyutu: {n} gün ({test_df.index[0].date()} → {test_df.index[-1].date()})")
print(f"Test periyodu istatistikleri:")
print(f"   Mean 7g return:    {test_ret.mean():+.5f}")
print(f"   Pozitif oranı:     %{(test_ret > 0).mean()*100:.1f}")
print(f"   Negatif oranı:     %{(test_ret < 0).mean()*100:.1f}")

print("\n" + "="*65)
print("NAIVE BASELINE'LAR (sığ stratejiler ne kadar iyi?)")
print("="*65)

# Baseline 1: Her zaman pozitif tahmin
b1_acc = (test_ret > 0).mean() * 100
b1_p = stats.binomtest(int((test_ret > 0).sum()), n, p=0.5, alternative='greater').pvalue
print(f"\n1. HER ZAMAN POZİTİF tahmin:")
print(f"   Yön doğruluğu: %{b1_acc:.1f}  (p={b1_p:.4f})")

# Baseline 2: Her zaman negatif tahmin
b2_acc = (test_ret < 0).mean() * 100
print(f"\n2. HER ZAMAN NEGATİF tahmin:")
print(f"   Yön doğruluğu: %{b2_acc:.1f}")

# Baseline 3: Last 7-day momentum
mom_7d = df['Close'].pct_change(7).iloc[val_split:].values
pred_mom = np.sign(mom_7d)
correct_mom = (pred_mom == np.sign(test_ret)).sum()
total_mom = (~np.isnan(mom_7d)).sum()
b3_acc = correct_mom / total_mom * 100
b3_p = stats.binomtest(int(correct_mom), int(total_mom), p=0.5, alternative='greater').pvalue
print(f"\n3. SON 7G MOMENTUM (sign(pct_change(7))):")
print(f"   Yön doğruluğu: %{b3_acc:.1f} ({correct_mom}/{total_mom})  (p={b3_p:.4f})")

# Baseline 4: 50/200 SMA crossover (trend following)
sma50 = df['Close'].rolling(50).mean()
sma200 = df['Close'].rolling(200).mean()
sma_signal = np.sign(sma50 - sma200).iloc[val_split:].values
correct_sma = (sma_signal == np.sign(test_ret)).sum()
total_sma = (~np.isnan(sma_signal)).sum()
b4_acc = correct_sma / total_sma * 100
b4_p = stats.binomtest(int(correct_sma), int(total_sma), p=0.5, alternative='greater').pvalue
print(f"\n4. SMA 50/200 CROSSOVER:")
print(f"   Yön doğruluğu: %{b4_acc:.1f} ({correct_sma}/{total_sma})  (p={b4_p:.4f})")

# Baseline 5: RSI < 30 / > 70 (mean reversion)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - 100 / (1 + rs)
# RSI > 50 → pozitif tahmin, RSI < 50 → negatif tahmin
rsi_signal = np.sign(rsi.iloc[val_split:].values - 50)
correct_rsi = (rsi_signal == np.sign(test_ret)).sum()
total_rsi = (~np.isnan(rsi_signal)).sum()
b5_acc = correct_rsi / total_rsi * 100
b5_p = stats.binomtest(int(correct_rsi), int(total_rsi), p=0.5, alternative='greater').pvalue
print(f"\n5. RSI > 50 / < 50 (RSI trend):")
print(f"   Yön doğruluğu: %{b5_acc:.1f} ({correct_rsi}/{total_rsi})  (p={b5_p:.4f})")

# Baseline 6: SMA20 > Close → negatif, < Close → pozitif
sma20 = df['Close'].rolling(20).mean()
px_sma20 = (df['Close'] - sma20).iloc[val_split:].values
pxs_signal = np.sign(px_sma20)
correct_pxs = (pxs_signal == np.sign(test_ret)).sum()
total_pxs = (~np.isnan(pxs_signal)).sum()
b6_acc = correct_pxs / total_pxs * 100
b6_p = stats.binomtest(int(correct_pxs), int(total_pxs), p=0.5, alternative='greater').pvalue
print(f"\n6. Close > SMA20 → pozitif (trend takip):")
print(f"   Yön doğruluğu: %{b6_acc:.1f} ({correct_pxs}/{total_pxs})  (p={b6_p:.4f})")

print("\n" + "="*65)
print("KARŞILAŞTIRMA: Modeliniz vs Naive Baseline'lar")
print("="*65)
print(f"   Modeliniz (B):      %62.1  ✅")
print(f"   1. Always positive: %{b1_acc:.1f}")
print(f"   2. Always negative: %{b2_acc:.1f}")
print(f"   3. 7g momentum:     %{b3_acc:.1f}")
print(f"   4. SMA 50/200:      %{b4_acc:.1f}")
print(f"   5. RSI 50:          %{b5_acc:.1f}")
print(f"   6. Close vs SMA20:  %{b6_acc:.1f}")

# Sonuç değerlendirmesi
best_naive = max(b1_acc, b3_acc, b4_acc, b5_acc, b6_acc)
gap = 62.1 - best_naive
print(f"\n   En iyi naive baseline: %{best_naive:.1f}")
print(f"   Modelin avantajı:      %{gap:+.1f} puan")
if gap > 5:
    print(f"   ✅ MODEL GERÇEK ALFA ÜRETİYOR — Naive'lere göre belirgin üstün")
elif gap > 2:
    print(f"   🟡 MODEL MARJİNAL OLARAK ÜSTÜN — Şüpheli ama mümkün")
else:
    print(f"   🔴 MODEL NAIVE BASELINE'LARI YENMİYOR — %62 çoğunlukla bias")

print("\n" + "="*65)
print("EK BULGU: Test periyodunun karakteri")
print("="*65)
# Test periyodu birikimli getiri
cum_ret = (1 + test_df['Close'].pct_change()).cumprod().iloc[-1] - 1
print(f"   Test periyodu toplam getiri: %{cum_ret*100:+.1f}")
if cum_ret > 0.2:
    print(f"   ⚠️  Test periyodu güçlü BULL MARKET — pozitif bias riski yüksek")
elif cum_ret < -0.2:
    print(f"   ⚠️  Test periyodu güçlü BEAR MARKET — negatif bias riski yüksek")
else:
    print(f"   ✅ Test periyodu balanced")