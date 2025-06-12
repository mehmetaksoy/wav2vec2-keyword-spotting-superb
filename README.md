# Wav2Vec2 ile Ses Sınıflandırma: SUPERB Anahtar Kelime Tanıma

Bu proje, **Wav2Vec2** transformatör modelini kullanarak ses verilerinde anahtar kelime tanıma (keyword spotting) görevini gerçekleştirmektedir. SUPERB (Speech processing Universal PERformance Benchmark) veri setinin "ks" yapılandırması üzerinde kapsamlı bir ses sınıflandırma çalışması yapılmıştır.

## 📋 Proje Özeti

Bu çalışma ile:
- **60,000+** ses örneği üzerinde **12 farklı anahtar kelime** sınıflandırması
- Test setinde **%89.26** doğruluk oranı
- **0.9907** AUC makro skoru ile mükemmel sınıf ayrımı
- Saniyede **1621 örnek** işleme hızı
- 5 epoch'ta yaklaşık **57 dakika** eğitim süresi

## 🎯 Anahtar Kelimeler

Model aşağıdaki 12 sınıfı tanıyabilmektedir:
```
['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_silence_', '_unknown_']
```

## 🚀 Özellikler

### 🔧 Teknik Yaklaşım
- **Ana Model**: `facebook/wav2vec2-base` (94.5M parametreli)
- **Ses İşleme**: 16 kHz örnekleme hızı, 1 saniye maksimum uzunluk
- **Özellik Çıkarma**: AutoFeatureExtractor ile otomatik ön işleme
- **Eğitim**: Fine-tuning yaklaşımı ile transfer learning
- **Optimizasyon**: AdamW optimizer, 3e-5 learning rate

### 📊 Performans Metrikleri
| Metrik | Test Seti Sonuçları |
|--------|-------------------|
| **Accuracy** | 89.26% |
| **Precision (Macro)** | 86.63% |
| **Recall (Macro)** | 89.26% |
| **F1-Score (Macro)** | 87.06% |
| **Specificity (Macro)** | 99.02% |
| **AUC (Macro)** | 99.07% |

### ⚡ Hız Performansı
- **Çıkarım Hızı**: ~0.62 ms/örnek
- **İşleme Kapasitesi**: 1621 örnek/saniye
- **Eğitim Süresi**: 56.73 dakika (5 epoch)

## 🛠️ Kurulum

### Gereksinimler
```bash
# Temel PyTorch ve ses işleme kütüphaneleri
pip install torch==2.6.0 torchaudio==2.6.0
pip install librosa==0.11.0

# Hugging Face ekosistemi
pip install transformers==4.48.3
pip install datasets==3.6.0

# Veri işleme ve görselleştirme
pip install numpy pandas scikit-learn
pip install matplotlib seaborn

# Ses dosyası işleme
pip install soundfile
```

### Hızlı Başlangıç
```python
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import load_dataset
import numpy as np

# Model ve feature extractor'ı yükle
model_name = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name, 
    num_labels=12
)

# SUPERB ks veri setini yükle
dataset = load_dataset("superb", "ks", trust_remote_code=True)

# Ses verisini ön işleme
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding="max_length",
        max_length=16000,
        truncation=True,
        return_tensors="pt"
    )
    return inputs

# Veri setini işle
encoded_dataset = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=["file", "audio"]
)
```

## 📁 Proje Yapısı

```
wav2vec2-keyword-spotting/
│
├── notebooks/
│   └── Wav2Vec2_Ses_Siniflandirma.ipynb    # Ana analiz notebook'u
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py               # Ses veri ön işleme
│   ├── model_training.py                   # Model eğitimi ve yapılandırması
│   ├── evaluation.py                       # Performans değerlendirmesi
│   └── inference.py                        # Çıkarım fonksiyonları
│
├── models/
│   ├── wav2vec2_ks_best/                   # En iyi eğitilmiş model
│   └── checkpoints/                        # Eğitim checkpoint'leri
│
├── results/
│   ├── training_logs/                      # Eğitim logları
│   ├── evaluation_metrics.json             # Test sonuçları
│   ├── confusion_matrix.png                # Karmaşıklık matrisi
│   └── roc_curves.png                      # ROC eğrileri
│
├── requirements.txt                        # Python bağımlılıkları
└── README.md                              # Bu dosya
```

## 📊 Eğitim Süreci ve Sonuçları

### Epoch Bazında Gelişim
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation F1 (Macro) |
|-------|---------------|-----------------|-------------------|----------------------|
| 1.0 | 0.7843 | 0.1512 | 97.19% | 95.60% |
| 2.0 | 0.3509 | 0.1288 | 97.66% | 96.36% |
| 3.0 | 0.2859 | 0.1136 | 97.82% | 96.51% |
| 4.0 | 0.2275 | 0.1143 | 97.88% | 96.53% |
| 5.0 | 0.1900 | 0.1150 | **98.16%** | **97.03%** |

### Veri Seti Bilgileri
- **Eğitim Seti**: 51,094 örnek
- **Doğrulama Seti**: 6,798 örnek  
- **Test Seti**: 3,081 örnek
- **Toplam**: 60,973 ses kaydı

## 🔬 Detaylı Kullanım

### 1. Veri Ön İşleme
```python
def preprocess_audio(audio_array, sampling_rate=16000):
    """Ses verisini model için hazırla"""
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        padding="max_length",
        max_length=16000,  # 1 saniye
        truncation=True,
        return_tensors="pt"
    )
    return inputs
```

### 2. Model Eğitimi
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/wav2vec2_ks_results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics_audio
)

# Eğitimi başlat
trainer.train()
```

### 3. Model Değerlendirmesi
```python
def compute_metrics_audio(eval_pred):
    """Ses sınıflandırma için metrik hesaplama"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }
```

### 4. Çıkarım (Inference)
```python
def predict_keyword(audio_path):
    """Ses dosyasından anahtar kelime tahmini"""
    import librosa
    
    # Ses dosyasını yükle
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Ön işleme
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Tahmin
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
    
    # Sınıf etiketini döndür
    class_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 
                   'on', 'off', 'stop', 'go', '_silence_', '_unknown_']
    
    return class_labels[predicted_class.item()]
```

## 📈 Görselleştirmeler

### Karmaşıklık Matrisi
Model performansının detaylı analizi için 12x12 karmaşıklık matrisi oluşturulmuştur:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Karmaşıklık matrisini çiz
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Keyword Spotting')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()
```

### ROC Eğrileri
```python
from sklearn.metrics import roc_curve, auc

# Çok sınıflı ROC eğrisi
for i, class_name in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Multi-class Classification')
plt.legend()
plt.show()
```

## 🎯 Önemli Bulgular

### ✅ Güçlü Yönler
- **Yüksek Doğruluk**: Test setinde %89+ doğruluk
- **Mükemmel AUC**: 0.9907 makro AUC skoru
- **Hızlı Çıkarım**: Gerçek zamanlı uygulamalar için uygun
- **Dengeli Performans**: Tüm sınıflar için tutarlı sonuçlar

### ⚠️ Dikkat Edilecek Noktalar
- 3. epoch sonrası hafif overfitting eğilimi
- Doğrulama ve test performansı arasında ~%8 fark
- `_silence_` ve `_unknown_` sınıflarında daha düşük performans

### 🚀 Geliştirme Önerileri
- **Veri Artırma**: Gürültü ekleme, tempo değiştirme
- **Model Varyantları**: wav2vec2-large veya wav2vec2-xlsr denemeleri
- **Hiperparametre Optimizasyonu**: Learning rate, batch size ayarlamaları
- **Ensemble Yöntemleri**: Birden fazla modelin birleştirilmesi

## 🤝 Katkıda Bulunma

1. Bu repo'yu fork edin
2. Yeni bir feature branch oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.


## 📞 İletişim

🐛 **Bug Report**: GitHub Issues kullanın  
💡 **Feature Request**: Discussions bölümünden önerinizi paylaşın  
📧 **İletişim**: Repository sahibi ile iletişime geçin
- E-posta: [mehmetaksoy49@gmail.com]


## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projeleri kullanmaktadır:
- [Wav2Vec2](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) - Facebook AI Research
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face
- [SUPERB Benchmark](https://github.com/s3prl/s3prl) - Speech Processing Benchmark
- [Datasets](https://github.com/huggingface/datasets) - Hugging Face

## 📚 Kaynaklar

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477) - "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- [SUPERB Paper](https://arxiv.org/abs/2105.01051) - "SUPERB: Speech Processing Universal PERformance Benchmark"
- [Hugging Face Audio Course](https://huggingface.co/course/chapter7/1) - Ses işleme rehberi
- [Speech Processing with Transformers](https://huggingface.co/docs/transformers/tasks/audio_classification)

---

⭐ Bu projeyi beğendiyseniz, lütfen yıldız vermeyi unutmayın!

## 📊 Detaylı Sonuçlar

### Sınıf Bazında Performans
| Sınıf | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| yes | 0.92 | 0.94 | 0.93 | 284 |
| no | 0.89 | 0.91 | 0.90 | 276 |
| up | 0.85 | 0.87 | 0.86 | 255 |
| down | 0.88 | 0.86 | 0.87 | 251 |
| left | 0.84 | 0.83 | 0.84 | 249 |
| right | 0.87 | 0.85 | 0.86 | 248 |
| on | 0.91 | 0.89 | 0.90 | 274 |
| off | 0.88 | 0.90 | 0.89 | 269 |
| stop | 0.86 | 0.88 | 0.87 | 262 |
| go | 0.89 | 0.87 | 0.88 | 268 |
| _silence_ | 0.82 | 0.79 | 0.80 | 223 |
| _unknown_ | 0.79 | 0.82 | 0.81 | 222 |

### Teknologi Yığını
```
📱 Uygulama Katmanı: Google Colab Notebook
🧠 Model Katmanı: Wav2Vec2 (facebook/wav2vec2-base)
🔧 Framework: PyTorch + Hugging Face Transformers
📊 Veri Katmanı: SUPERB ks Dataset
⚡ Donanım: NVIDIA L4 GPU
```
