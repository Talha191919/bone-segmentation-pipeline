import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import safe_imread_grayscale, safe_load_npy

def prepare_data(image_path, mask_path):
    """Resimleri ve maskeleri ID'lerine göre gruplar."""
    print("🔍 Veriler taranıyor ve gruplanıyor...")
    all_img_paths = glob.glob(os.path.join(image_path, "**", "*.*"), recursive=True)
    all_img_paths = [f for f in all_img_paths if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_mask_paths = glob.glob(os.path.join(mask_path, "**", "*.npy"), recursive=True)

    grouped_data = {}
    for img_p in all_img_paths:
        filename = os.path.basename(img_p).split('.')[0]
        parent = os.path.basename(os.path.dirname(img_p))
        pid = parent if parent.isdigit() else filename
        
        p_masks = [m for m in all_mask_paths if (f"{os.sep}{pid}{os.sep}" in m) or (f"\\{pid}\\" in m) or (pid == os.path.basename(m).split('_')[0])]
        
        if len(p_masks) > 0:
            grouped_data[pid] = {'img': img_p, 'masks': p_masks}

    matched_list = list(grouped_data.values())
    print(f"📁 Toplam Resim: {len(all_img_paths)} | Gruplanmış Hasta: {len(matched_list)}")
    return matched_list

class BoneDataset(Dataset):
    """nnU-Net stratejilerini (Z-score, Augmentation) içeren Dataset sınıfı."""
    def __init__(self, data_list, is_train=False):
        self.data_list = data_list
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        img = safe_imread_grayscale(item['img'])
        if img is None:
            img = np.zeros((256, 256), dtype=np.float32)
        else:
            img = cv2.resize(img, (256, 256)).astype(np.float32)
            img = (img - np.mean(img)) / (np.std(img) + 1e-5) # Z-Score Norm
            
        final_mask = np.zeros((256, 256), dtype=np.float32)
        for m_path in item['masks']:
            m = safe_load_npy(m_path)
            if m is not None:
                m = cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
                final_mask = np.maximum(final_mask, m)
        
        final_mask[final_mask > 0] = 1.0

        if self.is_train:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                final_mask = np.flip(final_mask, axis=1).copy()
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((128, 128), angle, 1)
                img = cv2.warpAffine(img, M, (256, 256))
                final_mask = cv2.warpAffine(final_mask, M, (256, 256), flags=cv2.INTER_NEAREST)

        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(final_mask).unsqueeze(0)