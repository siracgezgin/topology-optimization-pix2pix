"""
Örnek: Hızlı Başlangıç Rehberi
Tüm akışı test etmek için Google Colab'de çalıştırın
"""

# ============================================================
# ADIM 1: BAĞIMLILIKLARI YÜKLE
# ============================================================
# !pip install torch torchvision matplotlib pillow

# ============================================================
# ADIM 2: DOSYALARINIZI YÜKLEYIN
# ============================================================
# /content/ dizinine yükleyin:
# - dataset_final.zip (eğitim verileriniz)
# - pix2pix_generator.pth (eğitilmiş model, varsa)

# ============================================================
# ADIM 3: MODELİ EĞİT (sıfırdan başlıyorsanız)
# ============================================================
"""
from scripts.train import train

# Bu işlem:
# 1. Dataset'inizi çıkartır
# 2. 50 epoch eğitim yapar
# 3. Modeli /content/pix2pix_generator.pth olarak kaydeder

train()
"""

# ============================================================
# ADIM 4: ÇIKARIM YAPMA (2D)
# ============================================================
"""
from scripts.inference_2d import load_model, test_model, visualize_clean_results

# Eğitilmiş modeli yükle
model = load_model('/content/pix2pix_generator.pth')

# 5 rastgele test durumu oluştur
test_model(model, num_samples=5)

# Temiz ikili çıktılar oluştur
visualize_clean_results(model, num_samples=3)
"""

# ============================================================
# ADIM 5: 3D MODELLER OLUŞTUR
# ============================================================
"""
from scripts.inference_3d import load_model, generate_and_show_3d_part_optimized

# Model yükle
model = load_model('/content/pix2pix_generator.pth')

# 3D görselleştir (hızlı)
generate_and_show_3d_part_optimized(model, thickness=15)
"""

# ============================================================
# ADIM 6: STL'YE AKTAR (3D baskı için)
# ============================================================
"""
from scripts.inference_3d import generate_and_export_mesh

# STL dosyasını dışa aktar
generate_and_export_mesh(model, thickness=30, output_path="my_design.stl")

# .stl dosyasını Colab'ın dosya tarayıcısından indirin
"""

print("Hızlı başlangıç rehberi hazır!")
print("Çalıştırmak istediğiniz bölümlerin yorumunu kaldırın.")
