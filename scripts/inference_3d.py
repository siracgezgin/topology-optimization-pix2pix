"""
3D Görselleştirme Scripti
2D topoloji tasarımlarını 3D katı modellere dönüştürür ve STL formatında dışa aktarır
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        print(f"✅ Model yüklendi: {model_path}")
    else:
        import glob
        pth_files = glob.glob("/content/*.pth")
        if pth_files:
            model_path = pth_files[0]
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model otomatik yüklendi: {model_path}")
        else:
            raise FileNotFoundError("❌ Model dosyası bulunamadı!")
    
    model.eval()
    return model


# ============================================================
# SENARYO OLUŞTUR
# ============================================================
def generate_scenario():
    """Rastgele yükleme senaryosu oluştur"""
    input_tensor = torch.ones((1, 3, 256, 256))
    
    # Mavi duvar (sol)
    input_tensor[:, :, :, :10] = 0
    input_tensor[:, 0, :, :10] = 0
    input_tensor[:, 1, :, :10] = 0
    input_tensor[:, 2, :, :10] = 1
    
    # Kırmızı yük (rastgele)
    rand_x = random.randint(60, 220)
    rand_y = random.randint(60, 220)
    
    input_tensor[:, 0, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 1
    input_tensor[:, 1, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 0
    input_tensor[:, 2, rand_y-5:rand_y+5, rand_x-5:rand_x+5] = 0
    
    norm_input = (input_tensor - 0.5) / 0.5
    return norm_input.to(device), rand_x, rand_y


# ============================================================
# 3D GÖRSELLEŞTİRME (HIZLI VERSİYON)
# ============================================================
def generate_and_show_3d_part_optimized(model, thickness=15):
    """
    AI tahmini ile 3D katı parça oluştur
    Daha hızlı render için downsampling kullanır
    """
    print("AI tasarım oluşturuyor...")
    
    # Senaryo oluştur
    input_tensor, rand_x, rand_y = generate_scenario()
    
    # Tahmin yap
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Çıktıyı işle
    pred_data = prediction[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    grayscale = np.mean(pred_data, axis=2)
    voxel_slice = (grayscale < 0.65)
    
    # 3D render için downsampling yap (hız için 4x azaltma)
    voxel_slice_low_res = voxel_slice[::4, ::4]
    
    # Ekstruzyon ile 3D hacim oluştur
    h, w = voxel_slice_low_res.shape
    voxel_grid = np.zeros((h, w, thickness), dtype=bool)
    for z in range(thickness):
        voxel_grid[:, :, z] = voxel_slice_low_res
    
    # Görselleştir
    fig = plt.figure(figsize=(14, 6))
    
    # Sol: 2D yüksek çözünürlüklü kesit
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(voxel_slice, cmap='gray_r')
    ax1.set_title(f"2D Kesit\nYük: X={rand_x}, Y={rand_y}")
    ax1.axis('off')
    
    # Sağ: 3D model
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.voxels(voxel_grid, facecolors='#008080', alpha=0.8)
    ax2.set_title("3D Katı Model")
    ax2.set_xlabel("Genişlik")
    ax2.set_ylabel("Yükseklik")
    ax2.set_zlabel("Kalınlık")
    ax2.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 3D MESH DİŞA AKTARIMI (STL FORMAT)
# ============================================================
def generate_and_export_mesh(model, thickness=30, output_path="ai_generated_part.stl"):
    """
    3D mesh oluştur ve STL formatında dışa aktar
    Gerekli: trimesh, scikit-image
    """
    try:
        import trimesh
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("❌ Eksik bağımlılıklar. Yüklemek için:")
        print("   pip install trimesh scikit-image scipy")
        return
    
    print("1. AI tasarım oluşturuyor...")
    
    # Senaryo oluştur
    input_tensor, rand_x, rand_y = generate_scenario()
    
    # Tahmin yap
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Çıktıyı işle
    pred_data = prediction[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    grayscale = np.mean(pred_data, axis=2)
    
    # Kenarları yumuşat
    grayscale = gaussian_filter(grayscale, sigma=1)
    
    # Malzeme olasılığına çevir (ters çevrilmiş)
    material_prob = 1.0 - grayscale
    
    # 3D hacim oluştur
    volume = np.zeros((256, 256, thickness))
    for z in range(thickness):
        volume[:, :, z] = material_prob
    
    print("2. Mesh oluşturuluyor (Marching Cubes)...")
    
    # Yüzey mesh'ını çıkart
    verts, faces, normals, values = marching_cubes(volume, level=0.65)
    
    # Trimesh objesi oluştur
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    mesh.fill_holes()
    
    print(f"   - Köşe Sayısı: {len(mesh.vertices)}")
    print(f"   - Yüzey Sayısı: {len(mesh.faces)}")
    
    # STL'ye aktar
    mesh.export(output_path)
    print(f"3. Dosya kaydedildi: {output_path}")
    print("   (CAD yazılımı veya 3D görüntüleyici ile açmak için indirin)")
    
    # Colab'de görselleştirmeyi dene
    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=[600, 400], visible=True)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(plt.imread(trimesh.util.wrap_as_stream(png)))
        plt.axis('off')
        plt.title("Oluşturulan Mesh Önizleme")
        plt.show()
    except:
        print("(Önizleme bu ortamda mevcut değil)")


# ============================================================
# ANA PROGRAM
# ============================================================
if __name__ == "__main__":
    # Model yükle
    model = load_model(MODEL_PATH)
    
    print("\n" + "="*60)
    print("3D GÖRSELLEŞTİRME SEÇENEKLERİ")
    print("="*60)
    
    # Seçenek 1: Hızlı voxel görselleştirme
    print("\n--- Seçenek 1: Hızlı 3D Voxel Görselleştirme ---")
    generate_and_show_3d_part_optimized(model, thickness=15)
    
    # Seçenek 2: STL'ye aktar (kullanmak için yorumu kaldır)
    # print("\n--- Seçenek 2: STL'ye Aktar ---")
    # generate_and_export_mesh(model, thickness=30, output_path="topology_design.stl")
