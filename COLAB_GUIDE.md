# Google Colab Kullanım Rehberi

Bu rehber, projeyi Google Colab'de çalıştırmanıza yardımcı olur.

## Google Colab'de Kurulum

### Seçenek 1: GitHub'dan Klonlama (Önerilir)

```python
# Repository'yi klonla
!git clone https://github.com/siracgezgin/topology-optimization-pix2pix.git
%cd topology-optimization-pix2pix

# Bağımlılıkları yükle
!pip install -q -r requirements.txt
```

### Seçenek 2: Manuel Yükleme

1. Bu dosyaları `/content/` dizinine yükleyin:
   - `models/` klasöründeki tüm dosyalar
   - `utils/` klasöründeki tüm dosyalar
   - `scripts/` klasöründeki tüm dosyalar
   - `dataset_final.zip` dosyanız

2. Bağımlılıkları yükleyin:
```python
!pip install torch torchvision matplotlib pillow scikit-image scipy trimesh
```

## Eğitim

```python
# Önce dataset_final.zip dosyanızı /content/ dizinine yükleyin!

# Eğitimi başlat
!python scripts/train.py

# Model /content/pix2pix_generator.pth olarak kaydedilecek
```

**Beklenen Çıktı:**
```
Kullanılan cihaz: cuda
Zip açılıyor...
Dataset hazır!
Dataset yüklendi: 454 dosya bulundu, 0 eksik.
Eğitim Başladı...
Epoch [1/50]  Kayip_D: 0.4523  Kayip_G: 12.3456
...
```

## Çıkarım (2D)

```python
# Modeli rastgele senaryolarla test et
!python scripts/inference_2d.py
```

**Bu işlem şunları üretir:**
- 5 temel test örneği (girdi → çıktı)
- Eşikleme ile 3 temiz ikili çıktı

## Çıkarım (3D)

```python
# 3D görselleştirme oluştur
!python scripts/inference_3d.py
```

**Özellikler:**
- 2D kesit görünümü
- 3D voxel rendering
- İsteğe bağlı STL dışa aktarımı

## İnteraktif Kullanım

```python
# Modülleri içe aktar
import sys
sys.path.append('/content/topology-optimization-pix2pix')

from models import GeneratorUNet
from scripts.inference_2d import load_model, visualize_clean_results
from scripts.inference_3d import generate_and_show_3d_part_optimized

# Model yükle
model = load_model('/content/pix2pix_generator.pth')

# Tahminler oluştur
visualize_clean_results(model, num_samples=3)

# 3D model oluştur
generate_and_show_3d_part_optimized(model, thickness=20)
```

## Sonuçları Dışa Aktarma

### Eğitilmiş Modeli İndir

```python
from google.colab import files
files.download('/content/pix2pix_generator.pth')
```

### STL Dosyasını İndir

```python
from scripts.inference_3d import generate_and_export_mesh

# Oluştur ve dışa aktar
generate_and_export_mesh(model, thickness=30, output_path="my_design.stl")

# İndir
from google.colab import files
files.download('my_design.stl')
```

## GPU Hızlandırma

GPU'nun etkin olup olmadığını kontrol edin:

```python
import torch
print(f"CUDA Mevcut: {torch.cuda.is_available()}")
print(f"Cihaz: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
```

Colab'de GPU'yu etkinleştirmek için:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 önerilir)
3. Save

## Sorun Giderme

### Bellek Yetersiz

```python
# scripts/train.py içinde batch boyutunu azalt
BATCH_SIZE = 8  # 16 yerine
```

### Modül Bulunamadı

```python
# Projeyi path'e ekle
import sys
sys.path.append('/content/topology-optimization-pix2pix')
```

### Yavaş 3D Rendering

```python
# inference_3d.py içinde downsampling'i artır
voxel_slice_low_res = voxel_slice[::8, ::8]  # 4x yerine 8x
```

## Tam İş Akışı Örneği

```python
# 1. Kurulum
!git clone https://github.com/siracgezgin/topology-optimization-pix2pix.git
%cd topology-optimization-pix2pix
!pip install -q -r requirements.txt

# 2. dataset_final.zip dosyasını /content/ dizinine yükle

# 3. Eğit
!python scripts/train.py

# 4. 2D Test
!python scripts/inference_2d.py

# 5. 3D Oluştur
!python scripts/inference_3d.py

# 6. STL Dışa Aktar
from scripts.inference_3d import load_model, generate_and_export_mesh
model = load_model('/content/pix2pix_generator.pth')
generate_and_export_mesh(model, output_path="final_design.stl")

# 7. İndir
from google.colab import files
files.download('final_design.stl')
```

## En İyi Sonuçlar İçin İpuçları

### 1. Dataset Kalitesi

- Girdi-hedef çiftlerinin düzgün hizalandığından emin olun
- Tutarlı görüntü boyutları kullanın (256×256)
- Temiz, yüksek kontrastlı görüntüler en iyi sonucu verir

### 2. Eğitim

- Kayıp değerlerini izleyin (yaklaşık 20 epoch sonra stabilize olmalı)
- Generator kaybı genellikle discriminator'dan daha yüksektir
- Her 10 epoch'ta checkpoint'leri kaydedin

### 3. 3D Dışa Aktarma

- İstenilen malzeme yoğunluğu için eşiği (0.65) ayarlayın
- Daha düzgün yüzeyler için gaussian smoothing kullanın
- Daha yüksek kalınlık = daha basılabilir parçalar

## Eğitim Parametrelerini Özelleştirme

`scripts/train.py` dosyasını düzenleyin:

```python
BATCH_SIZE = 16          # GPU belleğine göre ayarlayın
EPOCHS = 50             # Daha uzun eğitim için artırın
LEARNING_RATE = 0.0002  # Daha kararlı eğitim için azaltın
LAMBDA_PIXEL = 100      # L1 kaybı ağırlığı
```

## Kaynak Kullanımı

### Bellek Kullanımı (Tipik)

| İşlem | GPU Bellek | RAM |
|-------|-----------|-----|
| Eğitim (batch=16) | ~6GB | ~8GB |
| Eğitim (batch=8) | ~3GB | ~6GB |
| Çıkarım | ~2GB | ~4GB |
| 3D Rendering | ~1GB | ~6GB |

### Eğitim Süresi (T4 GPU)

- 1 epoch: ~2-3 dakika (454 örnek)
- 50 epoch: ~2-2.5 saat
- Model kayıt: ~1 saniye

## Gelişmiş Özellikler

### Checkpoint'ten Devam Etme

```python
# Eğitimi yarıda kesildiyse devam ettir
model.load_state_dict(torch.load('/content/checkpoint_epoch_20.pth'))
# Eğitimi epoch 20'den devam ettir
```

### Özel Dataset Yolu

`scripts/train.py` içinde:
```python
ZIP_PATH = '/content/my_custom_dataset.zip'
EXTRACT_PATH = '/content/my_dataset'
```

### Toplu STL Üretimi

```python
# Birden fazla rastgele tasarım oluştur
for i in range(10):
    generate_and_export_mesh(
        model, 
        thickness=30, 
        output_path=f"design_{i:03d}.stl"
    )
```

---

**Yardıma mı ihtiyacınız var?** GitHub'da issue açın: https://github.com/siracgezgin/topology-optimization-pix2pix

**Son Güncelleme:** Kasım 2025
