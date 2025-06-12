# 🎤 Wav2Vec2 ile Ses Sınıflandırma Projesi
## SUPERB Veri Setinde Anahtar Kelime Tanıma

Bu proje, Facebook tarafından geliştirilen Wav2Vec2 modelini kullanarak ses sınıflandırma görevi gerçekleştiren kapsamlı bir makine öğrenmesi uygulamasıdır. SUPERB benchmark'ının Keyword Spotting (anahtar kelime tanıma) veri seti üzerinde çok sınıflı ses sınıflandırma yapılmaktadır.

## 🎯 Proje Amacı

Bu proje, önceden eğitilmiş Wav2Vec2 modelini fine-tuning yaparak ses örneklerinden anahtar kelimeleri tanımayı amaçlar. Model, ses sinyallerini analiz ederek hangi anahtar kelimenin söylendiğini yüksek doğrulukla tahmin edebilir.

## ⭐ Özellikler

- **Gelişmiş Ses İşleme**: Wav2Vec2 tabanlı feature extraction
- **Kapsamlı Metrik Analizi**: Accuracy, Precision, Recall, F1-Score, Specificity, AUC
- **Görselleştirme**: Confusion Matrix, ROC eğrileri, eğitim grafikleri
- **Performans Optimizasyonu**: GPU desteği, mixed precision training
- **Detaylı Raporlama**: Eğitim süreci ve test sonuçlarının tam analizi

## 🔧 Teknoloji Stack'i

- **Derin Öğrenme**: PyTorch, Transformers (Hugging Face)
- **Ses İşleme**: torchaudio, librosa
- **Veri İşleme**: datasets, numpy, pandas
- **Görselleştirme**: matplotlib, seaborn
- **Metrikler**: scikit-learn
- **Ortam**: Google Colab (GPU desteği)

## 📊 Kullanılan Veri Seti

**SUPERB (Speech processing Universal PERformance Benchmark) - Keyword Spotting**
- Çok sınıflı anahtar kelime tanıma görevi
- Standart train/validation/test ayrımı
- Yüksek kaliteli ses örnekleri
- Dengeli sınıf dağılımı

## 🚀 Kurulum ve Kullanım

### Gereksinimler

```bash
# Temel kütüphaneler
datasets==3.6.0
transformers==4.48.3
torchaudio
librosa
scikit-learn
matplotlib
seaborn
```

### Adım Adım Kullanım

1. **Kütüphane Kurulumu** (Hücre 1)
   - Gerekli Python paketlerinin kurulması
   - Colab ortamının hazırlanması

2. **Ortam Hazırlığı** (Hücre 2) 
   - Kütüphanelerin import edilmesi
   - GPU/CPU kontrolü ve versiyon bilgileri
   - Cihaz konfigürasyonu

3. **Veri Yükleme** (Hücre 3)
   - SUPERB ks veri setinin yüklenmesi
   - Veri yapısının incelenmesi
   - Örnek ses dosyalarının dinlenmesi

4. **Veri Ön İşleme** (Hücre 4)
   - Wav2Vec2 Feature Extractor'ın yüklenmesi
   - Ses verilerinin yeniden örneklenmesi (16kHz)
   - Padding ve truncation işlemleri

5. **Model Yükleme** (Hücre 5)
   - Wav2Vec2 modelinin fine-tuning için hazırlanması
   - Sınıflandırma katmanının yapılandırılması
   - GPU'ya taşınması

6. **Model Eğitimi** (Hücre 6)
   - Eğitim argümanlarının belirlenmesi
   - Trainer objesi oluşturulması
   - Model eğitim sürecinin başlatılması

7. **Test Değerlendirmesi** (Hücre 7)
   - Eğitilmiş modelin test setinde değerlendirilmesi
   - Performans metriklerinin hesaplanması

8. **Görselleştirme** (Hücre 8)
   - Confusion Matrix çizimi
   - ROC eğrileri ve AUC analizi
   - Çok sınıflı performans görselleştirmesi

9. **Eğitim Analizi** (Hücre 9)
   - Loss ve metrik grafiklerinin çizimi
   - Epoch bazında performans analizi

10. **Performans Analizi** (Hücre 10)
    - Eğitim ve çıkarım sürelerinin ölçümü
    - Throughput hesaplamaları

## 📈 Model Performansı

Model aşağıdaki metrikleri kullanarak değerlendirilir:

- **Accuracy**: Genel doğruluk oranı
- **Precision (Macro)**: Sınıf bazında hassasiyet ortalaması
- **Recall (Macro)**: Sınıf bazında duyarlılık ortalaması
- **F1-Score (Macro)**: Precision ve recall'un harmonik ortalaması
- **Specificity (Macro)**: Sınıf bazında özgüllük ortalaması
- **AUC (Macro)**: ROC eğrisi altında kalan alan

## 🎛️ Model Konfigürasyonu

```python
# Eğitim Parametreleri
- Epoch Sayısı: 5
- Batch Size: 8 (train), 16 (eval)
- Learning Rate: 3e-5
- Warmup Ratio: 0.1
- Weight Decay: 0.01
- Optimizer: AdamW
- Scheduler: Linear warmup
```

## 🔍 Teknik Detaylar

### Wav2Vec2 Modeli
- **Base Model**: facebook/wav2vec2-base
- **Örnekleme Hızı**: 16kHz
- **Max Sequence Length**: 16000 (1 saniye)
- **Feature Extraction**: Otomatik özellik çıkarımı

### Veri İşleme Pipeline'ı
1. Ses dosyalarının yüklenmesi
2. Örnekleme hızının standardizasyonu
3. Feature extraction ile vektör dönüşümü
4. Padding/truncation işlemleri
5. Tensor formatına çevirme

## 📁 Proje Yapısı

```
wav2vec2-keyword-spotting/
├── notebook.ipynb          # Ana Jupyter notebook
├── README.md              # Bu dosya
├── requirements.txt       # Gerekli kütüphaneler
└── results/              # Model çıktıları ve grafikler
    ├── wav2vec2_ks_results/  # Eğitim sonuçları
    ├── confusion_matrix.png   # Karışıklık matrisi
    ├── roc_curves.png        # ROC eğrileri
    └── training_plots.png    # Eğitim grafikleri
```

## 🎯 Sonuçlar ve Başarılar

- ✅ SUPERB benchmark standardında yüksek performans
- ✅ Çok sınıflı ses sınıflandırmada başarılı sonuçlar
- ✅ Kapsamlı metrik analizi ve görselleştirme
- ✅ Efficient training pipeline
- ✅ Reproducible results

## 🔄 Gelecek Geliştirmeler

- [ ] Daha büyük Wav2Vec2 modellerinin (Large, XLarge) test edilmesi
- [ ] Data augmentation tekniklerinin eklenmesi
- [ ] Cross-validation implementasyonu
- [ ] Real-time inference pipeline'ı
- [ ] Model compression ve quantization
- [ ] Multi-modal yaklaşımların denenmesi

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 📚 Kaynaklar ve Referanslar

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477) - Baevski et al., 2020
- [SUPERB Benchmark](https://arxiv.org/abs/2105.01051) - Yang et al., 2021
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/datasets/)

## 👨‍💻 Geliştirici

Proje geliştiricisi tarafından ses işleme ve derin öğrenme alanında araştırma amaçlı geliştirilmiştir.

---

**Not**: Bu proje Google Colab ortamında GPU kullanılarak geliştirilmiştir. Lokal çalıştırma için uygun CUDA konfigürasyonu gereklidir.