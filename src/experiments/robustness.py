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

from src.data.dataset import DeepfakeDataset
from src.models.factory import create_model

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_degraded_transform(img_size, degradation_type, intensity):
    transforms = [A.Resize(img_size, img_size)]
    
    if degradation_type == 'jpeg':
        transforms.append(A.ImageCompression(quality_lower=intensity, quality_upper=intensity, p=1.0))
    elif degradation_type == 'blur':
        transforms.append(A.GaussianBlur(blur_limit=(intensity, intensity), p=1.0))
    elif degradation_type == 'noise':
        transforms.append(A.GaussNoise(var_limit=(intensity, intensity), p=1.0))
        
    transforms.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(transforms)

def run_robustness_experiment():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị: {device}")
    
    df_master = pd.read_csv('master_split.csv')
    df_test = df_master[df_master['split'] == 'test']
    
    os.makedirs('results/robustness', exist_ok=True)
    
    tests = [
        ('Clean', 'none', 0),
        ('JPEG Q90', 'jpeg', 90),
        ('JPEG Q50', 'jpeg', 50),
        ('JPEG Q20', 'jpeg', 20),
        ('Blur (5x5)', 'blur', 5),
        ('Noise (Low)', 'noise', 20),
    ]

    all_results = []

    for model_key, m_cfg in config['models'].items():
        ckpt_path = f"results/checkpoints/best_{model_key}.pth"
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"\n--> Thử nghiệm Robustness cho: {model_key}")
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for test_name, deg_type, intensity in tests:
            transform = get_degraded_transform(m_cfg['img_size'], deg_type, intensity)
            ds = DeepfakeDataset(df_test, transform=transform, device=device)
            loader = DataLoader(ds, batch_size=m_cfg['batch_size'], shuffle=False)
            
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

            all_results.append({
                'Model': model_key,
                'Condition': test_name,
                'Accuracy': round(acc, 4),
                'ROC-AUC': round(auc, 4)
            })

        del model
        if device.type == 'cuda': torch.cuda.empty_cache()

    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n🏆 === BẢNG KẾT QUẢ ĐỘ BỀN VỮNG (ROBUSTNESS EVALUATION) ===")
        print(df_res.to_string(index=False))
        df_res.to_csv('results/robustness/robustness_report.csv', index=False)
        print("\n[✔] Đã lưu báo cáo vào results/robustness/robustness_report.csv")

if __name__ == "__main__":
    run_robustness_experiment()
