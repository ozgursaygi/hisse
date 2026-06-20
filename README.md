# 📈 Hisse / Multi-Stock Quant Backtester

Python tabanlı, rejim değişimi (regime-switching) modelleri ve derin öğrenme (LSTM) algoritmaları kullanarak hisse senedi piyasalarında algoritmik alım-satım stratejileri geliştiren ve geriye dönük test (backtesting) yapan analitik bir makine öğrenmesi çerçevesidir (framework).

## 🚀 Proje Hakkında
Bu proje, geleneksel teknik analiz yöntemlerini modern makine öğrenmesi yaklaşımlarıyla birleştirerek özellikle teknoloji hisselerinin (INTC, AMD, NVDA) fiyat hareketlerini modellemeyi amaçlar. Farklı piyasa koşullarını tespit etmek için rejim değişimi mantığını kullanır ve stratejilerin başarısını kurumsal finansal metriklerle doğrular. Proje, tekrarlı testler ve optimizasyonlarla sürekli olarak geliştirilmiştir (v24).

## ✨ Temel Özellikler
* **Derin Öğrenme ile Zaman Serisi Analizi:** Gelecekteki fiyat hareketlerini tahmin etmek için LSTM (Long Short-Term Memory) sinir ağları entegrasyonu.
* **Rejim Değişimi (Regime-Switching):** Farklı piyasa döngülerini (boğa/ayı/yatay) tespit ederek stratejiyi dinamik olarak ayarlayan istatistiksel modelleme.
* **Teknik & Algoritmik Sinyaller:** Makine öğrenmesi modellerini desteklemek amacıyla Elliott Dalga Teorisi (EWT - Elliott Wave Theory) hesaplamalarının entegrasyonu.
* **Kantitatif Performans Metrikleri:** Geliştirilen stratejilerin başarısının sadece getiri ile değil, *Information Ratio (IR)* ve *Cumulative Alpha* gibi profesyonel risk/getiri metrikleriyle geriye dönük test edilmesi.

## 📂 Ana Dosyalar (Core Files)
* `INTC_Analiz.py`: Veri işleme, LSTM model eğitimi, hisse analizleri ve backtest süreçlerini yürüten temel betik (script). 

## ⚙️ Kullanılan Teknolojiler
* **Dil:** Python
* **Makine Öğrenmesi & Veri Bilimi:** TensorFlow/Keras (LSTM), Pandas, NumPy, Scikit-learn
* **Finansal Analiz:** Kantitatif Backtesting, EWT, Regime-Switching Models
