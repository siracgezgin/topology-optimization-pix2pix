"""
Topoloji Optimizasyonu için Dataset sınıfı
Otomatik olarak sıfır doldurma formatını algılar (örn: input_000.png veya input_0.png)
"""

import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TopOptDataset(Dataset):
    """
    Topoloji optimizasyonu girdi-hedef çiftleri için özel dataset
    
    Parametreler:
        root_dir (str): input_*.png ve target_*.png dosyalarını içeren dizin
        start (int): Başlangıç indeksi (varsayılan: 0)
        end (int): Bitiş indeksi (varsayılan: 453)
        ext (str): Dosya uzantısı (varsayılan: '.png')
        img_size (tuple): Yeniden boyutlandırma için görüntü boyutu (varsayılan: (256, 256))
    """
    
    def __init__(self, root_dir, start=0, end=453, ext='.png', img_size=(256, 256)):
        self.root_dir = root_dir
        self.start = start
        self.end = end
        self.ext = ext

        # Dönüşümleri tanımla: Yeniden Boyutlandır -> Tensor -> [-1, 1] aralığına normalize et
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Mevcut dosyalardan sıfır doldurma uzunluğunu otomatik algıla
        all_inputs = [f for f in os.listdir(root_dir) if f.startswith('input_') and f.endswith(ext)]
        if not all_inputs:
            raise RuntimeError(f"{root_dir} dizininde input dosyası bulunamadı!")

        # Sıfır doldurma uzunluğunu çıkar (örn: input_000123.png -> pad = 6)
        m = re.search(r'input_(\d+)\.', all_inputs[0])
        self.pad_length = len(m.group(1)) if m else 0

        # Mevcut dosyaların listesini oluştur
        self.files = []
        missing = []

        for i in range(start, end + 1):
            if self.pad_length > 0:
                fname = f"input_{str(i).zfill(self.pad_length)}{ext}"
            else:
                fname = f"input_{i}{ext}"
                
            if os.path.exists(os.path.join(root_dir, fname)):
                self.files.append(fname)
            else:
                missing.append(fname)

        print(f"Dataset loaded: {len(self.files)} files found, {len(missing)} missing.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_name = self.files[idx]

        # Dosya adından sayısal ID'yi çıkar
        m = re.search(r'input_(\d+)\.', input_name)
        id_str = m.group(1)

        # Karşılık gelen hedef dosya adını oluştur
        target_name = f"target_{id_str}{self.ext}"

        # Görüntüleri yükle
        input_path = os.path.join(self.root_dir, input_name)
        target_path = os.path.join(self.root_dir, target_name)

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        # Dönüşümleri uygula
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        return input_tensor, target_tensor
