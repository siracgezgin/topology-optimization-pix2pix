"""
2D Çıkarım ve Görselleştirme Scripti
Rastgele yük senaryolarından topoloji optimize edilmiş tasarımlar oluşturur
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import GeneratorUNet


# ============================================================
# YAPILANDIRMA
# ============================================================
MODEL_PATH = '/content/pix2pix_generator.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL YÜKLE
# ============================================================
def load_model(model_path):
    """Eğitilmiş generator modelini yükle"""
    model = GeneratorUNet().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model başarıyla yüklendi: {model_path}")
    else:
        # /content dizininde herhangi bir .pth dosyası bulmaya çalış
        import glob
        pth_files = glob.glob("/content/*.pth")
        if pth_files:
            model_path = pth_files[0]
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model otomatik algılandı ve yüklendi: {model_path}")
        else:
            raise FileNotFoundError("❌ .pth model dosyası bulunamadı!")
    
    model.eval()
    return model


# ============================================================
# GİRDİ SENARYOSU OLUŞTUR
# ============================================================
def generate_random_scenario():
    """
    Rastgele bir yükleme senaryosu oluştur
    Model için hazır normalize edilmiş tensor döndürür
    """
    # Beyaz tuval oluştur
    input_tensor = torch.ones((1, 3, 256, 256))
    
    # Mavi duvar (sol taraf)
    input_tensor[:, :, :, :10] = 0
    input_tensor[:, 0, :, :10] = 0  # Kırmızı
    input_tensor[:, 1, :, :10] = 0  # Yeşil
    input_tensor[:, 2, :, :10] = 1  # Mavi
    
    # Kırmızı yük noktası (rastgele konum)
    rand_x = random.randint(50, 240)
    rand_y = random.randint(50, 240)
    
    input_tensor[:, 0, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 1  # Kırmızı
    input_tensor[:, 1, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 0  # Yeşil
    input_tensor[:, 2, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 0  # Mavi
    
    # [-1, 1] aralığına normalize et
    norm_input = (input_tensor - 0.5) / 0.5
    
    return norm_input.to(device), rand_x, rand_y


# ============================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================
def test_model(model, num_samples=5):
    """Birden fazla rastgele senaryo oluştur ve görüntüle"""
    plt.figure(figsize=(10, num_samples * 3))
    
    for i in range(num_samples):
        input_tensor, load_x, load_y = generate_random_scenario()
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Görüntülenebilir formata çevir
        inp = input_tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        out = prediction[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        
        # Girdi
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(inp)
        plt.title(f"Girdi (Yük: X={load_x}, Y={load_y})")
        plt.axis("off")
        
        # Çıktı
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(out, cmap='gray')
        plt.title("AI Tahmini")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_clean_results(model, num_samples=3):
    """Eşik işleme ile temiz ikili (siyah/beyaz) sonuçlar oluştur"""
    plt.figure(figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        # Senaryo oluştur
        input_tensor, load_x, load_y = generate_random_scenario()
        
        # Tahmin yap
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Çıktıyı işle
        pred_data = prediction[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        grayscale = np.mean(pred_data, axis=2)
        
        # Eşik: İkili çıktı (0 = yapı, 1 = boşluk)
        binary_output = np.zeros_like(grayscale)
        binary_output[grayscale < 0.65] = 0  # Yapı (siyah)
        binary_output[grayscale >= 0.65] = 1  # Boşluk (beyaz)
        
        # Görselleştir
        # 1. Girdi
        plt.subplot(num_samples, 3, i*3 + 1)
        inp = input_tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        plt.imshow(inp)
        plt.title(f"Girdi (Yük: X={load_x}, Y={load_y})")
        plt.axis('off')
        
        # 2. Ham çıktı
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(pred_data)
        plt.title("Ham AI Çıktısı")
        plt.axis('off')
        
        # 3. Temiz ikili çıktı
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(binary_output, cmap='gray')
        plt.title("İşlenmiş Tasarım")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# ANA PROGRAM
# ============================================================
if __name__ == "__main__":
    # Model yükle
    model = load_model(MODEL_PATH)
    
    # Temel testler çalıştır
    print("\n--- Temel Test (5 örnek) ---")
    test_model(model, num_samples=5)
    
    # Temiz görselleştirme çalıştır
    print("\n--- Temiz İkili Sonuçlar (3 örnek) ---")
    visualize_clean_results(model, num_samples=3)
