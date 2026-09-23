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
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm

import argparse
import sys
sys.path.append('.') # Cho phép chạy script từ root directory

from src.data.dataset import DeepfakeDataset, PreExtractedFaceDataset
from src.models.factory import create_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deepfake Detection")
    parser.add_argument('--dataset_mode', type=str, default='faces', choices=['video', 'faces'], 
                        help='Chế độ: "faces" (Dataset MỚI đọc ảnh cắt sẵn) hoặc "video" (Dataset CŨ đọc mp4)')
    parser.add_argument('--faces_csv', type=str, default=None, help='Đường dẫn faces_master.csv')
    parser.add_argument('--video_csv', type=str, default=None, help='Đường dẫn master_split.csv')
    parser.add_argument('--debug', type=str, default='false', choices=['true', 'false'], 
                        help='Bật/Tắt DEBUG_MODE (mặc định "false" để train toàn bộ dữ liệu)')
    args = parser.parse_args()

    config = load_config('configs/baseline.yaml')
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị: {device}")
    
    # 1. Xác định Chế độ Dataset (Ưu tiên CLI -> Config YAML -> Mặc định 'faces')
    dataset_cfg = config.get('dataset', {})
    mode = args.dataset_mode or dataset_cfg.get('mode', 'faces')
    
    # Mặc định DEBUG_MODE = False để train all
    DEBUG_MODE = (args.debug.lower() == 'true')

    if mode == 'faces':
        # --- OPTION 1: DATASET MỚI (ẢNH ĐÃ CẮT SẴN - TRAIN SIÊU TỐC) ---
        faces_csv = args.faces_csv or dataset_cfg.get('faces_csv', 'faces_master.csv')
        
        # Tự động dò tìm đường dẫn trên Kaggle
        if not os.path.exists(faces_csv):
            candidates = [
                '/kaggle/working/ffpp_faces_c23/csv/faces_master.csv',
                '/kaggle/working/ffpp_faces_c23/faces_master.csv',
                'ffpp_faces_c23/csv/faces_master.csv',
                'ffpp_faces_c23/faces_master.csv',
                '/kaggle/input/datasets/min2k4/face-ff/kaggle/working/ffpp_faces/faces_master.csv',
                '/kaggle/input/datasets/min2k4/face-ff/ffpp_faces/faces_master.csv',
                '/kaggle/input/ffpp-faces-c23/faces_master.csv',
                '/kaggle/input/ffpp-faces-c23/ffpp_faces/faces_master.csv',
                '/kaggle/working/ffpp_faces/faces_master.csv',
                'ffpp_faces/faces_master.csv',
                'faces_master.csv'
            ]
            for c in candidates:
                if os.path.exists(c):
                    faces_csv = c
                    break

        if not os.path.exists(faces_csv):
            print(f"[!] Không tìm thấy file faces_master.csv tại: {faces_csv}")
            print("    -> Nếu chưa cắt mặt, hãy chạy extract_faces_fast.py hoặc thêm flag: --dataset_mode video")
            return

        base_dir = os.path.dirname(faces_csv)
        print(f"\n🚀 SỬ DỤNG DATASET MỚI (Ảnh đã trích xuất sẵn): {faces_csv}")
        df_faces = pd.read_csv(faces_csv)
        
        if DEBUG_MODE:
            print("[!] ĐANG CHẠY Ở CHẾ ĐỘ DEBUG (Chỉ lấy mẫu nhỏ)")
            df_train = df_faces[df_faces['split'] == 'train'].groupby('label').sample(n=50, replace=True, random_state=42)
            df_val = df_faces[df_faces['split'] == 'val'].groupby('label').sample(n=20, replace=True, random_state=42)
            df_test = df_faces[df_faces['split'] == 'test'].groupby('label').sample(n=20, replace=True, random_state=42)
        else:
            print("[★] CHẾ ĐỘ FULL DATASET: Huấn luyện trên TOÀN BỘ dữ liệu đã cắt!")
            df_train = df_faces[df_faces['split'] == 'train']
            df_val = df_faces[df_faces['split'] == 'val']
            df_test = df_faces[df_faces['split'] == 'test']

        train_ds_raw = PreExtractedFaceDataset(df_train, base_dir=base_dir)
        val_ds_raw = PreExtractedFaceDataset(df_val, base_dir=base_dir)
        test_ds_raw = PreExtractedFaceDataset(df_test, base_dir=base_dir)
        print(f"[*] Tổng mẫu nạp vào: Train ({len(train_ds_raw)} ảnh), Val ({len(val_ds_raw)} ảnh), Test ({len(test_ds_raw)} ảnh)")

    else:
        # --- OPTION 2: DATASET CŨ (ĐỌC TRỰC TIẾP TỪ VIDEO .MP4) ---
        video_csv = args.video_csv or dataset_cfg.get('video_csv', 'master_split.csv')
        if not os.path.exists(video_csv):
            print(f"[!] Không tìm thấy {video_csv}! Hãy chạy file identity_split.py trước.")
            return
            
        print(f"\n🐢 SỬ DỤNG DATASET CŨ (Trích xuất từ Video .mp4): {video_csv}")
        df_master = pd.read_csv(video_csv)
        
        if DEBUG_MODE:
            print("[!] ĐANG CHẠY Ở CHẾ ĐỘ DEBUG (Chỉ lấy mẫu nhỏ)")
            df_train = df_master[df_master['split'] == 'train'].groupby('label').sample(n=10, replace=True, random_state=42)
            df_val = df_master[df_master['split'] == 'val'].groupby('label').sample(n=5, replace=True, random_state=42)
            df_test = df_master[df_master['split'] == 'test'].groupby('label').sample(n=5, replace=True, random_state=42)
        else:
            print("[★] CHẾ ĐỘ FULL DATASET: Trích xuất và train toàn bộ video gốc!")
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
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    num_workers = 2 if os.name != 'nt' else 0
    pin_mem = (device.type == 'cuda')

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
        
        train_loader = DataLoader(train_ds_raw, batch_size=m_cfg['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        val_loader = DataLoader(val_ds_raw, batch_size=m_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        test_loader = DataLoader(test_ds_raw, batch_size=m_cfg['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        # Khởi tạo Model
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=m_cfg['lr'])
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        
        # --- TRAINING LOOP TÍCH HỢP EARLY STOPPING & PROGRESS BAR ---
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [TRAIN]", leave=False)
            for images, labels, _ in train_bar:
                optimizer.zero_grad()
                loss = criterion(model(images.to(device)).view(-1), labels.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    val_loss += criterion(model(images.to(device)).view(-1), labels.to(device)).item()
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1:02d}/{config['epochs']:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
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
                
        # Lưu Training History & Biểu đồ Loss
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(f"results/metrics/{model_key}_history.csv", index=False)
        
        plt.figure(figsize=(8, 4))
        plt.plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', marker='o')
        plt.plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', marker='s')
        plt.title(f"Training & Validation Loss ({model_key.upper()})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/figures/{model_key}_loss_curve.png", dpi=300)
        plt.close()

        # --- ĐÁNH GIÁ (FRAME-LEVEL & VIDEO-LEVEL) TRÊN TẬP TEST ĐỘC LẬP ---
        print(f"\n📊 Đang đánh giá {model_key} trên tập TEST...")
        model.load_state_dict(torch.load(f"results/checkpoints/best_{model_key}.pth", map_location=device))
        model.eval()
        
        all_probs, all_labels, all_vids = [], [], []
        test_losses = []
        test_start_time = time.time()
        
        with torch.no_grad():
            for images, labels, vids in test_loader:
                images_dev, labels_dev = images.to(device), labels.to(device)
                outputs = model(images_dev).view(-1)
                loss = criterion(outputs, labels_dev)
                test_losses.append(loss.item())
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_vids.extend(vids.numpy())
                
        test_elapsed_time = time.time() - test_start_time
        
        # 1. Frame-level Metrics (Chuẩn theo format ảnh mẫu)
        frame_preds = (np.array(all_probs) >= 0.5).astype(int)
        frame_labels = np.array(all_labels).astype(int)
        frame_probs = np.array(all_probs)
        frame_auc = roc_auc_score(frame_labels, frame_probs) if len(np.unique(frame_labels)) > 1 else 0.5
        
        frame_metrics = {
            "loss": float(np.mean(test_losses)),
            "accuracy": float(accuracy_score(frame_labels, frame_preds)),
            "precision": float(precision_score(frame_labels, frame_preds, zero_division=0)),
            "recall": float(recall_score(frame_labels, frame_preds, zero_division=0)),
            "f1": float(f1_score(frame_labels, frame_preds, zero_division=0)),
            "roc_auc": float(frame_auc),
            "time_sec": float(test_elapsed_time),
            "ms_per_sample": float(test_elapsed_time * 1000 / len(test_loader.dataset))
        }
        
        print("\n--- FRAME-LEVEL METRICS ---")
        for k, v in frame_metrics.items():
            print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
            
        # 2. Video-level Metrics (Aggregation)
        df_test_preds = pd.DataFrame({'vid_id': all_vids, 'prob': all_probs, 'label': all_labels})
        vid_agg = df_test_preds.groupby('vid_id').mean()
        vid_preds = (vid_agg['prob'] >= 0.5).astype(int)
        vid_labels = vid_agg['label'].astype(int)
        vid_probs = vid_agg['prob']
        vid_auc = roc_auc_score(vid_labels, vid_probs) if len(np.unique(vid_labels)) > 1 else 0.5
        
        video_metrics = {
            "loss": float(np.mean(test_losses)),
            "accuracy": float(accuracy_score(vid_labels, vid_preds)),
            "precision": float(precision_score(vid_labels, vid_preds, zero_division=0)),
            "recall": float(recall_score(vid_labels, vid_preds, zero_division=0)),
            "f1": float(f1_score(vid_labels, vid_preds, zero_division=0)),
            "roc_auc": float(vid_auc),
            "time_sec": float(test_elapsed_time),
            "ms_per_sample": float(test_elapsed_time * 1000 / len(vid_agg))
        }
        
        print("\n--- VIDEO-LEVEL METRICS ---")
        for k, v in video_metrics.items():
            print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
            
        # Lưu Metrics thành file JSON
        import json
        with open(f"results/metrics/{model_key}_metrics.json", "w") as jf:
            json.dump({'frame_level': frame_metrics, 'video_level': video_metrics}, jf, indent=4)
            
        # Vẽ ROC Curve & Confusion Matrix cho Video-level
        if len(np.unique(vid_labels)) > 1:
            fpr, tpr, _ = roc_curve(vid_labels, vid_probs)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {vid_auc:.4f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"ROC Curve - Video-Level ({model_key.upper()})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/figures/{model_key}_roc_curve.png", dpi=300)
            plt.close()
            
            cm = confusion_matrix(vid_labels, vid_preds)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title(f"Confusion Matrix ({model_key.upper()})")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(f"results/figures/{model_key}_confusion_matrix.png", dpi=300)
            plt.close()
        else:
            print("ROC-AUC  : N/A (Chỉ có 1 class trong tập test)")
        
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
