import os
import time
import torch
import yaml
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append('.')
from src.models.factory import create_model
from src.data.dataset import DeepfakeDataset

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_ablation_frames():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị: {device}")
    
    df_master = pd.read_csv('master_split.csv')
    df_test = df_master[df_master['split'] == 'test']
    
    frame_options = [1, 3, 5, 10, 20] # Các mốc thời gian để test
    results = []

    os.makedirs('results/ablation', exist_ok=True)
    
    # Chọn model nhẹ nhất để test thực tế (VD: MobileNetV3)
    model_key = 'mobilenetv3'
    if model_key not in config['models']: model_key = list(config['models'].keys())[0]
    
    m_cfg = config['models'][model_key]
    ckpt_path = f"results/checkpoints/best_{model_key}.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"[!] Thiếu Checkpoint {ckpt_path}. Hãy chạy train.py trước!")
        return

    model = create_model(model_key, m_cfg, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(m_cfg['img_size'], m_cfg['img_size']),
        A.Normalize(), ToTensorV2()
    ])

    for f_count in frame_options:
        print(f"\n--> Đang thử nghiệm Ablation: Lấy {f_count} frames/video")
        
        start_time = time.time()
        ds = DeepfakeDataset(df_test, transform=transform, frames_per_video=f_count, device=device)
        loader = DataLoader(ds, batch_size=m_cfg['batch_size'], shuffle=False)
        
        all_probs, all_labels, all_vids = [], [], []
        
        with torch.no_grad():
            for images, labels, vids in loader:
                probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_vids.extend(vids.numpy())
                
        total_time = time.time() - start_time
        fps = len(ds) / total_time
        
        # 1. Khung hình (Frame-level)
        preds = (np.array(all_probs) >= 0.5).astype(int)
        labels = np.array(all_labels)
        frame_f1 = f1_score(labels, preds, zero_division=0)
        
        # 2. Cấp độ Video (Video-level)
        df_test_preds = pd.DataFrame({'vid_id': all_vids, 'prob': all_probs, 'label': all_labels})
        vid_agg = df_test_preds.groupby('vid_id').mean()
        vid_preds = (vid_agg['prob'] >= 0.5).astype(int)
        vid_labels = vid_agg['label'].astype(int)
        
        vid_f1 = f1_score(vid_labels, vid_preds, zero_division=0)
        try:
            vid_auc = roc_auc_score(vid_labels, vid_agg['prob']) if len(np.unique(vid_labels)) > 1 else np.nan
        except:
            vid_auc = np.nan
            
        results.append({
            'Frames/Video': f_count,
            'Total Extracted Frames': len(ds),
            'Frame F1': round(frame_f1, 4),
            'Video F1': round(vid_f1, 4),
            'Video AUC': round(vid_auc, 4),
            'Processing Time (s)': round(total_time, 2),
            'Extraction+Inference FPS': round(fps, 2)
        })

    df_res = pd.DataFrame(results)
    print("\n🏆 === BẢNG ABLATION: ẢNH HƯỞNG CỦA SỐ LƯỢNG KHUNG HÌNH (FRAMES/VIDEO) ===")
    print(df_res.to_string(index=False))
    df_res.to_csv('results/ablation/ablation_frames_report.csv', index=False)
    print("\n[✔] Đã lưu báo cáo vào results/ablation/ablation_frames_report.csv")

if __name__ == "__main__":
    run_ablation_frames()
