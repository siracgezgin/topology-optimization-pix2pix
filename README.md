![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Maintenance](https://img.shields.io/badge/maintained-yes-green.svg)

# Pix2Pix ile Topoloji Optimizasyonu

Yükleme koşullarına göre hafif yapısal tasarımlar oluşturmak için koşullu GAN (Pix2Pix) kullanan yapay zeka tabanlı topoloji optimizasyon sistemi.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Proje Yapısı](#proje-yapısı)
- [Mimari Detayları](#mimari-detayları)
- [Kullanım Örnekleri](#kullanım-örnekleri)
- [Eğitim](#eğitim)
- [Çıkarım ve Görselleştirme](#çıkarım-ve-görselleştirme)
- [Teknik Detaylar](#teknik-detaylar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

## Proje Hakkında

Bu proje, derin öğrenme kullanarak optimize edilmiş yapısal tasarımları otomatik olarak oluşturur. Bir yükleme senaryosu (sınır koşulları ve kuvvetler) verildiğinde, model yapısal bütünlüğü korurken ağırlığı minimize eden verimli bir malzeme dağılımı tahmin eder.

### Temel Özellikler

- Skip connection'lı U-Net Generator mimarisi
- Gerçekçi çıktılar için PatchGAN Discriminator
- İkili eşikleme ile 2D görselleştirme
- 3D model oluşturma ve STL dışa aktarım
- Google Colab desteği ve ZIP dataset yükleme
- Rastgele yük senaryoları ile test yetenekleri
- Marching Cubes algoritması ile mesh oluşturma

### Uygulama Alanları

- Mekanik parça tasarımı
- Havacılık ve uzay mühendisliği
- Otomotiv yapısal optimizasyon
- Biyomedikal implant tasarımı
- Mimari yapı optimizasyonu

## Gereksinimler

### Sistem Gereksinimleri

- Python 3.8 veya üzeri
- CUDA destekli GPU (eğitim için önerilir)
- En az 8GB RAM (GPU eğitimi için 12GB+ önerilir)
- 2GB+ disk alanı (model ve dataset için)

### Python Kütüphaneleri

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
scikit-image>=0.21.0
scipy>=1.10.0
trimesh>=3.23.0
```

## Kurulum

### 1. Projeyi Klonlama

```bash
git clone https://github.com/siracgezgin/topology-optimization-pix2pix.git
cd topology-optimization-pix2pix
```

### 2. Sanal Ortam Oluşturma (Önerilir)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 4. Kurulum Doğrulama

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Proje Yapısı

```
topology-optimization-pix2pix/
│
├── README.md                      # Bu dosya
├── requirements.txt               # Gerekli paketler
├── .gitignore                     # Git ignore kuralları
├── LICENSE                        # MIT Lisans
│
├── models/                        # Sinir ağı mimarileri
│   ├── __init__.py
│   ├── generator.py               # U-Net Generator (ana model)
│   └── discriminator.py           # PatchGAN Discriminator
│
├── utils/                         # Yardımcı fonksiyonlar
│   ├── __init__.py
│   └── dataset.py                 # Otomatik padding algılamalı dataset yükleyici
│
├── scripts/                       # Çalıştırılabilir scriptler
│   ├── train.py                   # Eğitim scripti (Pix2Pix GAN)
│   ├── inference_2d.py            # 2D görselleştirme ve test
│   └── inference_3d.py            # 3D görselleştirme ve STL dışa aktarım
│
├── examples/                      # Kullanım örnekleri
│   └── quick_start.py             # Yeni başlayanlar için adım adım rehber
│
└── assets/                        # Görseller ve medya
    └── (örnek görseller buraya)
```

## Mimari Detayları

### GAN Mimarisi

![GAN Architecture](assets/gan_architecture.jpeg)

*Generative Adversarial Network (GAN) yapısı: Generator sahte örnekler üretir, Discriminator gerçek ve sahte örnekleri ayırt eder.*

### Generator (U-Net)

8 encoder ve 7 decoder bloğuna sahip U-Net mimarisi.

**Encoder (Örnekleme Azaltma):**
- Conv2d (4x4, stride=2) 
- BatchNorm2d
- LeakyReLU (0.2)
- Dropout (derin katmanlarda)

**Decoder (Örnekleme Artırma):**
- ConvTranspose2d (4x4, stride=2)
- BatchNorm2d
- ReLU
- Skip connections (encoder katmanlarından)
- Dropout (ilk 3 decoder bloğunda)

**Giriş/Çıkış:**
- Giriş: 256x256x3 RGB görüntü
- Çıkış: 256x256x3 RGB görüntü
- Aktivasyon: Tanh ([-1, 1] aralığında çıktı)

### Discriminator (PatchGAN)

70x70 örtüşen görüntü parçalarının gerçek veya sahte olduğunu sınıflandırır.

**Mimari:**
- 4 adet downsampling bloğu
- Her blok: Conv2d → BatchNorm → LeakyReLU
- Son katman: 1 kanallı Conv2d (gerçeklik haritası)

**Giriş:**
- 6 kanal (birleştirilmiş giriş ve hedef görüntüler)

**Özellikler:**
- Yerel doku ayrımcılığı
- Daha keskin, gerçekçi yapılar oluşturur
- Tam görüntü discriminator'a göre daha hızlı eğitim

### Kayıp Fonksiyonları

**Adversarial Loss (GAN Loss):**
```
L_GAN = MSE(D(x, G(x)), 1)
```

**L1 Pixel Loss:**
```
L_L1 = ||y - G(x)||_1
```

**Toplam Generator Loss:**
```
L_total = L_GAN + λ * L_L1
```
- λ (lambda) = 100 (L1 kaybı ağırlığı)

**Discriminator Loss:**
```
L_D = 0.5 * (MSE(D(x,y), 1) + MSE(D(x,G(x)), 0))
```

## Kullanım Örnekleri

### Temel 2D Tahmin

```python
from models import GeneratorUNet
import torch

# Model yükle
model = GeneratorUNet()
model.load_state_dict(torch.load('pix2pix_generator.pth'))
model.eval()

# Tahmin yap
with torch.no_grad():
    output = model(input_tensor)
```

### 3D Model Oluşturma

```python
from scripts.inference_3d import generate_and_show_3d_part_optimized

# 3D tasarımı görselleştir
generate_and_show_3d_part_optimized(model, thickness=15)
```

### STL Dışa Aktarma

```python
from scripts.inference_3d import generate_and_export_mesh

# 3D baskı için dışa aktar
generate_and_export_mesh(model, thickness=30, output_path="design.stl")
```

## Eğitim

### Dataset Hazırlama

Eşleştirilmiş görüntülerle dataset hazırlayın:
- `input_000.png` - `input_453.png` (sınır koşulları)
- `target_000.png` - `target_453.png` (optimize tasarımlar)

**Dataset Formatı:**

**Giriş görüntüleri şunları temsil etmelidir:**
- Mavi pikseller: Sabit sınır (duvar)
- Kırmızı pikseller: Yük/kuvvet uygulama noktası
- Beyaz pikseller: Tasarım alanı

**Hedef görüntüler şunları temsil etmelidir:**
- Siyah pikseller: Malzeme (yapı)
- Beyaz pikseller: Boşluk (hava)

### Google Colab'de Eğitim

```python
# dataset_final.zip dosyasını /content/ dizinine yükleyin
# Eğitim scriptini çalıştırın
!python scripts/train.py
```

### Yerel Eğitim

```python
# train.py içinde ZIP_PATH ve EXTRACT_PATH'i düzenleyin
python scripts/train.py
```

### Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Görüntü Boyutu | 256x256 | Giriş/çıkış görüntü boyutu |
| Batch Boyutu | 16 | Batch başına örnek sayısı |
| Öğrenme Oranı | 0.0002 | Adam optimizer öğrenme oranı |
| Optimizer | Adam | β₁=0.5, β₂=0.999 |
| Epoch Sayısı | 50 | Toplam eğitim epoch'u |
| Lambda (L1) | 100 | L1 kayıp ağırlığı |

### Eğitim Süreci

1. Dataset ZIP dosyasından çıkartılır
2. Her epoch'ta:
   - Discriminator güncellenir (gerçek/sahte sınıflandırma)
   - Generator güncellenir (sahte görüntü oluşturma)
3. Her 10 epoch'ta örnek görselleştirme
4. Model `pix2pix_generator.pth` olarak kaydedilir

## Çıkarım ve Görselleştirme

### 2D Görselleştirme

```bash
python scripts/inference_2d.py
```

**Özellikler:**
- Rastgele yük senaryoları oluşturma
- Ham ve işlenmiş çıktı karşılaştırması
- İkili eşikleme ile temiz tasarımlar
- Toplu test yetenekleri

**Eşikleme Parametrelerini Ayarlama:**

```python
# inference_2d.py veya inference_3d.py içinde
# Daha fazla malzeme tutma
binary_output[grayscale < 0.50] = 0  # Daha muhafazakar

# Daha agresif malzeme çıkarma
binary_output[grayscale < 0.70] = 0  # Daha agresif
```

### 3D Görselleştirme

```bash
python scripts/inference_3d.py
```

**Seçenekler:**

**Seçenek 1: Hızlı Voxel Görselleştirme**
- 4x downsampling ile hızlı render
- 2D kesit + 3D model görüntüleme
- Gerçek zamanlı önizleme

**Seçenek 2: STL Mesh Dışa Aktarımı**
- Marching Cubes algoritması
- Yüksek kaliteli mesh
- 3D baskı için hazır
- CAD yazılımı ile uyumlu

### 3D Kalınlığı Değiştirme

```python
# Daha kalın parçalar
generate_and_show_3d_part_optimized(model, thickness=30)

# STL dışa aktarımda
generate_and_export_mesh(model, thickness=50, output_path="thick_part.stl")
```

## Teknik Detaylar

### PyTorch Versiyonu

Bu proje PyTorch 2.0+ ile geliştirilmiştir. Temel özellikler:

- Otomatik mixed precision (AMP) desteği
- GPU bellek optimizasyonları
- TorchScript derleme desteği
- CUDA 11.8+ uyumluluğu

### Performans Optimizasyonu

**Eğitim:**
- GPU kullanın (CUDA)
- Batch boyutunu artırın (GPU belleği yetiyorsa)
- Mixed precision training (AMP)
- DataLoader num_workers optimizasyonu

**Çıkarım:**
- model.eval() modunu kullanın
- torch.no_grad() ile gradyan hesaplamayı devre dışı bırakın
- Batch inference (birden fazla örnek)

**3D Rendering:**
- Downsampling kullanın (zaten uygulanmış: 4x)
- Voxel çözünürlüğünü azaltın
- Trimesh yerine daha basit görselleştirme

### Dataset Özellikleri

**Otomatik Padding Algılama:**
- `input_000.png` formatını otomatik algılar
- `input_0.png` formatını da destekler
- Eksik dosyaları raporlar

**Görüntü Transformasyonları:**
- Resize: 256x256 boyuta
- ToTensor: [0,255] → [0,1]
- Normalize: [0,1] → [-1,1] (Tanh aktivasyonu için)

### Yaygın Sorunlar ve Çözümleri

**Sorun:** GPU belleği yetersiz
- **Çözüm:** Batch boyutunu azaltın (train.py içinde BATCH_SIZE = 8)

**Sorun:** Eğitim çok yavaş
- **Çözüm:** GPU kullanın, num_workers artırın, görüntü boyutunu azaltın

**Sorun:** Model kötü sonuçlar veriyor
- **Çözüm:** Daha fazla epoch eğitin, dataset kalitesini kontrol edin, lambda değerini ayarlayın

**Sorun:** 3D render çok yavaş
- **Çözüm:** Downsampling oranını artırın (::8 yerine ::4)

**Sorun:** STL dosyası açılmıyor
- **Çözüm:** Mesh'i kontrol edin (vertices/faces sayısı), farklı mesh viewer deneyin

## Katkıda Bulunma

Bu proje açık kaynaklıdır ve katkılara açıktır. Katkıda bulunmak için:

1. Projeyi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

### Kod Standartları

- PEP 8 stil rehberine uyun
- Fonksiyonlar için docstring yazın (Türkçe)
- Değişken isimleri açıklayıcı olsun
- Yorumları Türkçe yazın

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## İletişim

**Geliştirici:** Siraç Gezgin

**GitHub:** [github.com/siracgezgin](https://github.com/siracgezgin)

**Proje Deposu:** [github.com/siracgezgin/topology-optimization-pix2pix](https://github.com/siracgezgin/topology-optimization-pix2pix)

## Referanslar

**Pix2Pix Makalesi:**
```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```

**Kaynaklar:**
- [Pix2Pix Resmi Makale](https://arxiv.org/abs/1611.07004)
- [PyTorch Dokümantasyonu](https://pytorch.org/docs/)
- [Topoloji Optimizasyonu Temelleri](https://en.wikipedia.org/wiki/Topology_optimization)
- [Marching Cubes Algoritması](https://en.wikipedia.org/wiki/Marching_cubes)

## Teşekkürler

Pix2Pix makale yazarlarına, PyTorch ekibine ve açık kaynak topluluğuna teşekkür ederim.
