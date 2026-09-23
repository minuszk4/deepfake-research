import os
import torch
import yaml
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append('.')

from src.data.dataset import DeepfakeDataset, PreExtractedFaceDataset
from src.models.factory import create_model

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def evaluate_subset(model, df_subset, transform, device, batch_size=16):
    if len(df_subset) == 0:
        return {'acc': np.nan, 'auc': np.nan}
        
    ds = DeepfakeDataset(df_subset, transform=transform, device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)
    
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    except:
        auc = np.nan
        
    return {'acc': round(acc, 4), 'auc': round(auc, 4)}

from torch.utils.data import DataLoader, Subset

def run_cross_manipulation():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị: {device}")
    
    # Kiểm tra Dataset Mới hay Cũ
    dataset_cfg = config.get('dataset', {})
    mode = dataset_cfg.get('mode', 'faces')
    faces_csv = dataset_cfg.get('faces_csv', 'faces_master.csv')
    if not os.path.exists(faces_csv):
        candidates = [
            '/kaggle/input/ffpp_faces_c23/kaggle/working/ffpp_faces_c23/csv/faces_master.csv',
            '/kaggle/input/ffpp-faces-c23/kaggle/working/ffpp_faces_c23/csv/faces_master.csv',
            '/kaggle/working/ffpp_faces_c23/csv/faces_master.csv',
            'ffpp_faces_c23/csv/faces_master.csv',
            '/kaggle/input/datasets/min2k4/face-ff/kaggle/working/ffpp_faces/faces_master.csv',
            '/kaggle/working/ffpp_faces/faces_master.csv'
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                faces_csv = candidate
                break
        if not os.path.exists(faces_csv) and os.path.exists('/kaggle/input'):
            import glob
            matches = glob.glob('/kaggle/input/**/faces_master.csv', recursive=True)
            if matches:
                faces_csv = matches[0]

    if mode == 'faces' and os.path.exists(faces_csv):
        print(f"\n🚀 SỬ DỤNG DATASET MỚI (Ảnh đã cắt sẵn - Tải tức thì): {faces_csv}")
        df_faces = pd.read_csv(faces_csv)
        df_test = df_faces[df_faces['split'] == 'test']
        raw_test_ds = PreExtractedFaceDataset(df_test, base_dir=os.path.dirname(faces_csv))
    else:
        print("\n🐢 SỬ DỤNG DATASET CŨ (Trích xuất từ video .mp4)")
        df_master = pd.read_csv('master_split.csv')
        df_test = df_master[df_master['split'] == 'test']
        print("[+] ĐANG TRÍCH XUẤT KHUÔN MẶT TẬP TEST (Chỉ trích xuất 1 lần duy nhất)")
        raw_test_ds = DeepfakeDataset(df_test, transform=None, frames_per_video=config.get('frames_per_video', 3), device=device)
    
    manipulations = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    results = []

    os.makedirs('results/cross_manipulation', exist_ok=True)

    for model_key, m_cfg in config['models'].items():
        ckpt_path = f"results/checkpoints/best_{model_key}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[!] Bỏ qua {model_key}: Chưa có checkpoint tại {ckpt_path}")
            continue
            
        print(f"\n--> Đang đánh giá Cross-Manipulation cho: {model_key}")
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        transform = A.Compose([
            A.Resize(m_cfg['img_size'], m_cfg['img_size']),
            A.Normalize(), ToTensorV2()
        ])
        raw_test_ds.transform = transform
        
        row_res = {'Model': model_key}
        for manip in manipulations:
            # Lấy indices tương thích cả Dataset cũ và mới
            if hasattr(raw_test_ds, 'samples'):
                indices = [
                    i for i, s in enumerate(raw_test_ds.samples)
                    if s['label'] == 0 or s.get('manipulation') == manip
                ]
            else:
                indices = raw_test_ds.df[
                    (raw_test_ds.df['label'] == 0) | (raw_test_ds.df['manipulation'] == manip)
                ].index.tolist()
            if len(indices) == 0:
                row_res[f"{manip}_Acc"] = np.nan
                row_res[f"{manip}_AUC"] = np.nan
                continue

            subset_ds = Subset(raw_test_ds, indices)
            loader = DataLoader(subset_ds, batch_size=m_cfg['batch_size'], shuffle=False)
            
            all_probs, all_labels = [], []
            with torch.no_grad():
                for images, labels, _ in loader:
                    probs = torch.sigmoid(model(images.to(device)).view(-1)).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy())
                    
            labels = np.array(all_labels)
            probs = np.array(all_probs)
            preds = (probs >= 0.5).astype(int)
            
            acc = accuracy_score(labels, preds)
            try:
                auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
            except:
                auc = np.nan
                
            row_res[f"{manip}_Acc"] = round(acc, 4)
            row_res[f"{manip}_AUC"] = round(auc, 4)
            
        results.append(row_res)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()

    if results:
        df_res = pd.DataFrame(results)
        print("\n🏆 === MA TRẬN ĐÁNH GIÁ TỔNG QUÁT HÓA (CROSS-MANIPULATION) ===")
        print(df_res.to_string(index=False))
        df_res.to_csv('results/cross_manipulation/cross_manipulation_report.csv', index=False)
        print("\n[✔] Đã lưu báo cáo vào results/cross_manipulation/cross_manipulation_report.csv")

if __name__ == "__main__":
    run_cross_manipulation()
