import os
import yaml
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.append('.') # Cho phép chạy script từ root directory

from src.data.dataset import DeepfakeDataset
from src.models.factory import create_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config('configs/baseline.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị: {device}")
    
    # 1. Đọc Dữ liệu Split
    csv_path = 'master_split.csv'
    if not os.path.exists(csv_path):
        print("[!] Không tìm thấy master_split.csv! Hãy chạy file identity_split.py trước.")
        return
        
    df_master = pd.read_csv(csv_path)
    
    # Cờ Debug: Đặt thành False khi muốn train trên TOÀN BỘ dữ liệu
    DEBUG_MODE = True
    
    if DEBUG_MODE:
        print("[!] ĐANG CHẠY Ở CHẾ ĐỘ DEBUG (Chỉ lấy mẫu nhỏ gọn)")
        # Lấy cân bằng mẫu của cả 2 class (Real/Fake) để tránh lỗi chỉ có 1 class
        # Dùng replace=True phòng trường hợp tập val không đủ 5 video mỗi loại
        df_train = df_master[df_master['split'] == 'train'].groupby('label').sample(n=10, replace=True, random_state=42)
        df_val = df_master[df_master['split'] == 'val'].groupby('label').sample(n=5, replace=True, random_state=42)
        df_test = df_master[df_master['split'] == 'test'].groupby('label').sample(n=5, replace=True, random_state=42)
    else:
        df_train = df_master[df_master['split'] == 'train']
        df_val = df_master[df_master['split'] == 'val']
        df_test = df_master[df_master['split'] == 'test']
    
    print("\n[+] ĐANG TRÍCH XUẤT KHUÔN MẶT - TẬP TRAIN")
    train_ds_raw = DeepfakeDataset(df_train, frames_per_video=config['frames_per_video'], device=device)
    print("\n[+] ĐANG TRÍCH XUẤT KHUÔN MẶT - TẬP VAL")
    val_ds_raw = DeepfakeDataset(df_val, frames_per_video=config['frames_per_video'], device=device)
    print("\n[+] ĐANG TRÍCH XUẤT KHUÔN MẶT - TẬP TEST")
    test_ds_raw = DeepfakeDataset(df_test, frames_per_video=config['frames_per_video'], device=device)

    # 2. Vòng lặp huấn luyện từng Model trong Config
    os.makedirs('results/checkpoints', exist_ok=True)
    
    for model_key, m_cfg in config['models'].items():
        print(f"\n{'='*50}\n🚀 ĐANG HUẤN LUYỆN: {model_key.upper()} \n{'='*50}")
        
        # Data Augmentation theo Input Size của Model
        transform_train = A.Compose([
            A.Resize(m_cfg['img_size'], m_cfg['img_size']),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(), ToTensorV2()
        ])
        transform_val = A.Compose([
            A.Resize(m_cfg['img_size'], m_cfg['img_size']),
            A.Normalize(), ToTensorV2()
        ])
        
        train_ds_raw.transform = transform_train
        val_ds_raw.transform = transform_val
        test_ds_raw.transform = transform_val
        
        train_loader = DataLoader(train_ds_raw, batch_size=m_cfg['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds_raw, batch_size=m_cfg['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds_raw, batch_size=m_cfg['batch_size'], shuffle=False)
        
        # Khởi tạo Model
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=m_cfg['lr'])
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        
        # --- TRAINING LOOP TÍCH HỢP EARLY STOPPING ---
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            for images, labels, _ in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images.to(device)).squeeze(1), labels.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    val_loss += criterion(model(images.to(device)).squeeze(1), labels.to(device)).item()
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"results/checkpoints/best_{model_key}.pth")
                print("  -> Lưu checkpoint mới nhất!")
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                print(f"  -> Kích hoạt Early Stopping tại Epoch {epoch+1}!")
                break
                
        # --- ĐÁNH GIÁ (FRAME-LEVEL & VIDEO-LEVEL) TRÊN TẬP TEST ĐỘC LẬP ---
        print(f"\n📊 Đang đánh giá {model_key} trên tập TEST...")
        model.load_state_dict(torch.load(f"results/checkpoints/best_{model_key}.pth"))
        model.eval()
        
        all_probs, all_labels, all_vids = [], [], []
        
        with torch.no_grad():
            for images, labels, vids in test_loader:
                probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_vids.extend(vids.numpy())
                
        # 1. Frame-level Metrics
        preds = (np.array(all_probs) >= 0.5).astype(int)
        labels = np.array(all_labels)
        print("--- FRAME-LEVEL ---")
        print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
        print(f"F1-Score : {f1_score(labels, preds, zero_division=0):.4f}")
        
        # 2. Video-level Metrics (Aggregation)
        df_test_preds = pd.DataFrame({'vid_id': all_vids, 'prob': all_probs, 'label': all_labels})
        # Dùng MEAN của các probability để ra điểm của Video
        vid_agg = df_test_preds.groupby('vid_id').mean()
        vid_preds = (vid_agg['prob'] >= 0.5).astype(int)
        vid_labels = vid_agg['label'].astype(int)
        
        print("--- VIDEO-LEVEL ---")
        print(f"Accuracy : {accuracy_score(vid_labels, vid_preds):.4f}")
        print(f"F1-Score : {f1_score(vid_labels, vid_preds, zero_division=0):.4f}")
        try:
            print(f"ROC-AUC  : {roc_auc_score(vid_labels, vid_agg['prob']):.4f}")
        except: pass
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
