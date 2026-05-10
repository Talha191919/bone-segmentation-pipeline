import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import numpy as np

# Kendi yazdığımız modülleri import ediyoruz (Aynı veri ve utils!)
from src.dataset import BoneDataset, prepare_data
from src.utils import get_metrics, visualize_results

# --- HYPERPARAMETERS ---
IMAGE_PATH = r"C:\Users\recep\OneDrive\Masaüstü\TUBITAK 2247-C\New_Labels-20260507T123452Z-3-001"
MASK_PATH = r"C:\Users\recep\OneDrive\Masaüstü\TUBITAK 2247-C\New_masks-20260507T123509Z-3-001"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
BATCH_SIZE = 4
LR = 0.001 # Orijinal U-Net kodundaki Learning Rate

def main():
    matched_list = prepare_data(IMAGE_PATH, MASK_PATH)
    
    if len(matched_list) < 5:
        print("❌ HATA: Eşleşen veri seti 5-Fold için yetersiz!")
        return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_stats = []

    for fold, (t_idx, v_idx) in enumerate(kf.split(matched_list)):
        print(f"\n{'='*15} BASELINE U-NET FOLD {fold+1} BAŞLADI {'='*15}")
        
        train_loader = DataLoader(BoneDataset([matched_list[i] for i in t_idx], is_train=True), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(BoneDataset([matched_list[i] for i in v_idx], is_train=False), batch_size=1)

        # Standart U-Net
        model = smp.Unet("resnet34", in_channels=1, classes=1, activation="sigmoid").to(DEVICE)
        
        # nnU-net'teki MONAI yerine, standart Binary Dice Loss
        criterion = smp.losses.DiceLoss(mode='binary')
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            for imgs, msks in train_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs), msks)
                loss.backward()
                optimizer.step()
            
            model.eval()
            d_sc, h_dst = [], []
            with torch.no_grad():
                for imgs, msks in val_loader:
                    p = model(imgs.to(DEVICE)) # Activation sigmoid olduğu için direkt model çıktısı
                    d, h = get_metrics(p, msks.to(DEVICE))
                    d_sc.append(d); h_dst.append(h)
            
            print(f"Ep {epoch+1}/{EPOCHS} | Dice: {np.mean(d_sc):.4f} | HD: {np.mean(h_dst):.2f}")

        visualize_results(model, val_loader, DEVICE, fold)
        final_stats.append((np.mean(d_sc), np.mean(h_dst)))

    print("\n🏁" + "="*30)
    print(f"STANDART U-NET ORTALAMA DICE: %{np.mean([x[0] for x in final_stats])*100:.2f}")
    print(f"STANDART U-NET ORTALAMA HAUSDORFF: {np.mean([x[1] for x in final_stats]):.2f}")

if __name__ == "__main__":
    main()