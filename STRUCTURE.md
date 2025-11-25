# Proje Dosya Yapısı ve Detaylı Açıklamalar

## Genel Bakış

```
topology-optimization-pix2pix/
│
├── models/                          # Sinir ağı mimarileri
│   ├── __init__.py                 # Modül dışa aktarımları
│   ├── generator.py                # U-Net Generator (ana model)
│   └── discriminator.py            # PatchGAN Discriminator
│
├── utils/                           # Yardımcı fonksiyonlar
│   ├── __init__.py                 # Modül dışa aktarımları
│   └── dataset.py                  # Otomatik padding algılama ile dataset yükleyici
│
├── scripts/                         # Çalıştırılabilir scriptler
│   ├── train.py                    # Eğitim scripti (Pix2Pix GAN)
│   ├── inference_2d.py             # 2D görselleştirme ve test
│   └── inference_3d.py             # 3D görselleştirme ve STL dışa aktarım
│
├── examples/                        # Kullanım örnekleri
│   └── quick_start.py              # Yeni başlayanlar için adım adım rehber
│
├── assets/                          # Dokümantasyon için görseller ve medya
│   └── (örnek görsellerinizi buraya ekleyin)
│
├── requirements.txt                 # Python bağımlılıkları
├── .gitignore                      # Git ignore kuralları
├── LICENSE                         # MIT Lisansı
├── README.md                       # Ana dokümantasyon
├── STRUCTURE.md                    # Bu dosya
└── COLAB_GUIDE.md                 # Google Colab talimatları
```

## Dosya Açıklamaları

### Temel Modeller (`models/`)

**generator.py**
- 8 encoder ve 7 decoder bloğuna sahip U-Net mimarisi
- Uzamsal bilgiyi korumak için skip connection'lar
- Giriş/Çıkış: 256×256×3 RGB görüntüler
- Aktivasyon: Tanh ([-1, 1] aralığında çıktı verir)
- Toplam parametre sayısı: ~54M

**Mimari Detaylar:**
```python
# Encoder
256x256x3 → 128x128x64 → 64x64x128 → 32x32x256 → 16x16x512
          → 8x8x512 → 4x4x512 → 2x2x512 → 1x1x512

# Decoder (skip connections ile)
1x1x512 → 2x2x1024 → 4x4x1024 → 8x8x1024 → 16x16x1024
        → 32x32x512 → 64x64x256 → 128x128x128 → 256x256x3
```

**discriminator.py**
- Yerel doku ayrımcılığı için PatchGAN mimarisi
- 70×70 örtüşen patch'leri gerçek/sahte olarak sınıflandırır
- Giriş: 6 kanal (birleştirilmiş input ve target)
- Daha keskin, gerçekçi yapılar üretmeye yardımcı olur

**Patch Boyutu Hesabı:**
```
Receptive field = 70x70 piksel
Stride pattern: 2-2-2-2-1
Total downsampling: 16x
```

### Yardımcı Araçlar (`utils/`)

**dataset.py**
- Dosya adlarından zero-padding uzunluğunu otomatik algılar
- `input_000.png` veya `input_0.png` formatlarını destekler
- Eşleştirilmiş input-target görüntülerini yükler
- Normalizasyon ve dönüşümleri uygular
- Eksik dosyaları zarif şekilde işler

**Veri Akışı:**
```
Dosya yükleme → PIL Image → Resize(256,256) → ToTensor 
            → Normalize(mean=0.5, std=0.5) → Tensor[-1,1]
```

**Özellikler:**
- Bellek verimli yükleme (lazy loading)
- Eksik dosya bildirimi
- Esnek padding formatı
- RGB dönüşümü garantisi

### Scriptler (`scripts/`)

**train.py** - Tam eğitim hattı

**İşlevsellik:**
- ZIP'ten dataset çıkartma
- Alternatifli discriminator/generator güncellemeleri ile GAN eğitimi
- Her 10 epoch'ta checkpoint kaydı
- Kayıp takibi ve görselleştirme

**Eğitim Döngüsü:**
```python
for epoch in range(EPOCHS):
    for batch in dataloader:
        # 1. Discriminator'ı güncelle
        D_loss = discriminator_step(real, fake)
        
        # 2. Generator'ı güncelle  
        G_loss = generator_step(real)
```

**train.py içinde yapılandırılabilir parametreler:**
- `BATCH_SIZE`: Batch başına örnek sayısı (varsayılan: 16)
- `EPOCHS`: Eğitim süresi (varsayılan: 50)
- `LEARNING_RATE`: Adam öğrenme oranı (varsayılan: 0.0002)
- `LAMBDA_PIXEL`: L1 kayıp ağırlığı (varsayılan: 100)

**inference_2d.py** - 2D tahmin ve görselleştirme

**Fonksiyonlar:**
- `load_model()`: Eğitilmiş generator'ı yükler
- `generate_random_scenario()`: Rastgele duvar+yük senaryoları oluşturur
- `test_model()`: Ham çıktıların hızlı görselleştirmesi
- `visualize_clean_results()`: İkili eşiklenmiş çıktılar

**Kullanım Senaryoları:**
- Model doğrulama
- Tahmin kalite kontrolü
- Eşik parametresi ayarlama
- Toplu test oluşturma

**inference_3d.py** - 3D model oluşturma

**Fonksiyonlar:**
- `generate_scenario()`: Test senaryosu üret
- `generate_and_show_3d_part_optimized()`: Hızlı voxel rendering
- `generate_and_export_mesh()`: Yüksek kaliteli STL dışa aktarımı

**İki Mod:**

**Mod 1: Hızlı Voxel Görselleştirme**
- 4x downsampling
- Matplotlib 3D voxel rendering
- Gerçek zamanlı önizleme
- 2D kesit + 3D model görünümü

**Mod 2: STL Mesh Dışa Aktarımı**
- Marching Cubes algoritması
- Gaussian smoothing
- Mesh onarımı (normal düzeltme, delik doldurma)
- CAD/3D baskı için hazır

## Veri Akışı

### Eğitim Veri Akışı

```
ZIP Dosyası
    ↓
[Çıkartma]
    ↓
Dataset Klasörü (input_*.png, target_*.png)
    ↓
[TopOptDataset]
    ↓
DataLoader (batch'ler, shuffle, workers)
    ↓
Eğitim Döngüsü
    ├─ Discriminator Güncellemesi
    └─ Generator Güncellemesi
    ↓
Model Checkpointleri (.pth)
```

### Çıkarım Veri Akışı

```
Rastgele Senaryo Oluştur
    ↓
[Generator Modeli]
    ↓
Ham Tahmin (256x256x3)
    ↓
[Post-processing]
    ├─ Grayscale Dönüşümü
    ├─ Eşikleme (binary)
    └─ Gaussian Smoothing (opsiyonel)
    ↓
2D Görselleştirme veya 3D Ekstrüzyon
    ↓
    ├─ 2D: Matplotlib plot
    └─ 3D: Voxel grid veya STL mesh
```

## Bağımlılıklar

Tüm gerekli paketler `requirements.txt` içinde:

**Temel Kütüphaneler:**
- **torch**: Derin öğrenme framework'ü
- **torchvision**: Görüntü dönüşümleri
- **numpy**: Sayısal işlemler
- **Pillow**: Görüntü yükleme

**Görselleştirme:**
- **matplotlib**: 2D/3D plotting

**3D İşleme:**
- **scikit-image**: Marching cubes algoritması
- **trimesh**: 3D mesh işleme
- **scipy**: Gaussian filtreleme

## Hızlı Başlangıç İş Akışı

### 1. Dataset Hazırlama
```bash
# Eşleştirilmiş görüntülerle dataset_final.zip oluştur
# Colab'da /content/ dizinine yükle
```

### 2. Model Eğitme
```python
python scripts/train.py
# Çıktı: /content/pix2pix_generator.pth
```

### 3. Tahminler Oluştur
```python
python scripts/inference_2d.py
# 2D görselleştirmeler gösterir
```

### 4. 3D Model Dışa Aktar
```python
python scripts/inference_3d.py
# STL dosyası oluşturur
```

### 5. Sonuçları İndir
```python
# Colab'da
from google.colab import files
files.download('pix2pix_generator.pth')
files.download('ai_generated_part.stl')
```

## Özelleştirme Noktaları

### İkili Eşik Ayarlama
**Dosya:** `inference_2d.py` veya `inference_3d.py`
```python
# Daha fazla malzeme (muhafazakar)
binary_output[grayscale < 0.50] = 0

# Daha az malzeme (agresif)
binary_output[grayscale < 0.70] = 0
```

### 3D Kalınlığı Değiştir
```python
# Daha kalın parçalar
generate_and_show_3d_part_optimized(model, thickness=30)
```

### Ağ Derinliğini Değiştir
**Dosya:** `models/generator.py`
- Encoder-decoder çiftleri ekle/çıkar
- Dikkat: Skip connection'ları eşleştir

## Yaygın Sorunlar

**Bellek Yetersiz (GPU)**
- `train.py` içinde `BATCH_SIZE` değerini azalt
- CPU eğitimi kullan (daha yavaş): `device = "cpu"`

**Yavaş 3D Rendering**
- Zaten 4× downsampling optimizasyonu var
- Daha fazla azaltma: `voxel_slice[::8, ::8]` (8× için)

**Eksik Dosyalar**
- Dataset yapısının `input_XXX.png` ve `target_XXX.png` olduğunu doğrula
- Dosya numaralandırmasının `DATASET_START` ve `DATASET_END` ile eşleştiğini kontrol et

## İleri Seviye Kullanım

### TorchScript Derleme
```python
model = GeneratorUNet()
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model_optimized.pt")
```

### Mixed Precision Eğitim
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = generator(input)
    loss = criterion(output, target)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Inference
```python
# Birden fazla senaryoyu aynı anda işle
batch_inputs = torch.stack([scenario1, scenario2, scenario3])
with torch.no_grad():
    batch_outputs = model(batch_inputs)
```

---

**Son Güncelleme:** Kasım 2025  
**Bakımcı:** Siraç Gezgin
