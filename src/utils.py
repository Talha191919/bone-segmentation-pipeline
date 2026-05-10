import cv2
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

def safe_imread_grayscale(path):
    """Türkçe karakterli yollardan resmi güvenle okur."""
    try:
        with open(path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(chunk, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"⚠️ Okuma hatası: {path} -> {e}")
        return None

def safe_load_npy(path):
    """Türkçe karakterli yollardan .npy dosyasını güvenle yükler."""
    try:
        with open(path, "rb") as f:
            return np.load(f)
    except:
        return None

def get_metrics(pred, target):
    """Dice Skoru ve Hausdorff Distance (HD) hesaplar."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-7)
    
    p_np = pred.detach().cpu().numpy().squeeze()
    t_np = target.detach().cpu().numpy().squeeze()
    
    u, v = np.argwhere(p_np), np.argwhere(t_np)
    
    if len(u) == 0 or len(v) == 0: 
        hd = 0
    else: 
        hd = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return dice.item(), hd

def visualize_results(model, loader, device, fold, num_samples=1):
    """Eğitim sonrası model tahminlerini görselleştirir."""
    model.eval()
    with torch.no_grad():
        for i, (imgs, msks) in enumerate(loader):
            if i >= num_samples: break
            if torch.max(msks) == 0: continue 
            
            imgs_dev = imgs.to(device)
            preds = (torch.sigmoid(model(imgs_dev)) > 0.5).float().cpu()
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(imgs[0].squeeze(), cmap='gray')
            ax[0].set_title("Orijinal Görüntü (Z-Norm)")
            ax[1].imshow(msks[0].squeeze(), cmap='jet')
            ax[1].set_title("Gerçek Maske (Ground Truth)")
            ax[2].imshow(preds[0].squeeze(), cmap='jet')
            ax[2].set_title("Model Tahmini")
            for a in ax: a.axis('off')
            plt.suptitle(f"Fold {fold+1} Sonucu")
            plt.show()