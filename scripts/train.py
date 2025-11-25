"""
Pix2Pix Topoloji Optimizasyonu Modeli için Eğitim Scripti
Google Colab'de ZIP dataset yüklemesini destekler
"""

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import GeneratorUNet, Discriminator
from utils import TopOptDataset


# ============================================================
# YAPILANDIRMA
# ============================================================
ZIP_PATH = '/content/dataset_final.zip'       # Yüklenmiş ZIP dosyasının yolu
EXTRACT_PATH = '/content/dataset_final'       # Çıkarım dizini
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
EPOCHS = 50
LAMBDA_PIXEL = 100  # L1 kaybı için ağırlık
DATASET_START = 0
DATASET_END = 453

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")


# ============================================================
# DATASET'İ ÇIKART
# ============================================================
def extract_dataset(zip_path, extract_path):
    """Eğer daha önce çıkarılmamışsa ZIP dataset'i çıkart"""
    if not os.path.exists(extract_path):
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset ready!")
    else:
        print("Dataset zaten çıkarılmış.")


# ============================================================
# EĞİTİM DÖNGÜSÜ
# ============================================================
def train():
    # Dataset'i çıkart
    extract_dataset(ZIP_PATH, EXTRACT_PATH)
    
    # Dataset ve dataloader oluştur
    dataset = TopOptDataset(EXTRACT_PATH, start=DATASET_START, end=DATASET_END)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Modelleri başlat
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizerlar
    opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Kaybıp fonksiyonları
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    print("\n" + "="*60)
    print("EĞİTİM BAŞLADI")
    print("="*60)
    
    for epoch in range(EPOCHS):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        
        for i, (real_A, real_B) in enumerate(loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # ---- Discriminator'ı Eğit ----
            opt_d.zero_grad()
            
            # Sahte görüntüler oluştur
            fake_B = generator(real_A)
            
            # Gerçek kaybı
            pred_real = discriminator(real_A, real_B)
            loss_d_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Sahte kaybı
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_d_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            # Toplam discriminator kaybı
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()
            
            # ---- Generator'ı Eğit ----
            opt_g.zero_grad()
            
            # Çelişmeli kaybı
            pred_fake = discriminator(real_A, fake_B)
            loss_g_gan = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # L1 piksel kaybı
            loss_g_pixel = criterion_pixel(fake_B, real_B)
            
            # Toplam generator kaybı
            loss_g = loss_g_gan + LAMBDA_PIXEL * loss_g_pixel
            loss_g.backward()
            opt_g.step()
            
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
        
        # Epoch istatistiklerini yazdır
        avg_loss_d = epoch_loss_d / len(loader)
        avg_loss_g = epoch_loss_g / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Kayip_D: {avg_loss_d:.4f}  Kayip_G: {avg_loss_g:.4f}")
        
        # Her 10 epoch'ta bir sonuçları görselleştir
        if (epoch + 1) % 10 == 0:
            visualize_results(generator, real_A, epoch + 1)
    
    # Son modeli kaydet
    save_path = '/content/pix2pix_generator.pth'
    torch.save(generator.state_dict(), save_path)
    print(f"\n{'='*60}")
    print(f"Model kaydedildi: {save_path}")
    print("="*60)


def visualize_results(generator, real_A, epoch):
    """Bir örnek tahmin görselleştir"""
    generator.eval()
    with torch.no_grad():
        sample = generator(real_A[0].unsqueeze(0)).cpu()
        img = (sample[0].permute(1, 2, 0).numpy() * 0.5) + 0.5
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Epoch {epoch}")
        plt.axis('off')
        plt.show()
    generator.train()


# ============================================================
# ANA PROGRAM
# ============================================================
if __name__ == "__main__":
    train()
