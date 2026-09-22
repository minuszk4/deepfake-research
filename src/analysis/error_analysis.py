import os
import torch
import yaml
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append('.')
from src.models.factory import create_model
from src.data.dataset import DeepfakeDataset

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_image_from_tensor(tensor, save_path):
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_original = np.clip(img_np * std + mean, 0, 1) * 255.0
    cv2.imwrite(save_path, cv2.cvtColor(img_original.astype(np.uint8), cv2.COLOR_RGB2BGR))

def run_error_analysis():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs('results/error_analysis/False_Positive', exist_ok=True)
    os.makedirs('results/error_analysis/False_Negative', exist_ok=True)
    
    df_master = pd.read_csv('master_split.csv')
    # Ở bước phân tích lỗi, ta không dùng sample, mà quét toàn bộ Test Set
    df_test = df_master[df_master['split'] == 'test']

    # Sử dụng mô hình đầu tiên trong config (thường là baseline mạnh) để khảo sát lỗi
    model_key = list(config['models'].keys())[1] # Thử MobileNetV3 hoặc Xception
    m_cfg = config['models'][model_key]
    ckpt_path = f"results/checkpoints/best_{model_key}.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"[!] Cần train model {model_key} trước khi chạy Error Analysis.")
        return

    print(f"[*] Đang thực hiện Error Analysis bằng model đại diện: {model_key}")
    model = create_model(model_key, m_cfg, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(m_cfg['img_size'], m_cfg['img_size']),
        A.Normalize(), ToTensorV2()
    ])

    ds = DeepfakeDataset(df_test, transform=transform, frames_per_video=config['frames_per_video'], device=device)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    fp_count, fn_count = 0, 0
    max_errors_to_save = 20 # Giới hạn lưu ảnh tránh tràn ổ cứng

    error_logs = []

    with torch.no_grad():
        for idx, (img_tensor, label, vid_id) in enumerate(loader):
            input_tensor = img_tensor.to(device)
            prob = torch.sigmoid(model(input_tensor)).item()
            pred = 1 if prob >= 0.5 else 0
            true_val = int(label.item())
            
            # False Positive: Đoán là Fake (1) nhưng sự thật là Real (0)
            if pred == 1 and true_val == 0:
                error_logs.append({'type': 'FP', 'vid_id': vid_id.item(), 'prob': prob})
                if fp_count < max_errors_to_save:
                    save_path = f"results/error_analysis/False_Positive/FP_prob{prob:.2f}_vid{vid_id.item()}_{idx}.jpg"
                    save_image_from_tensor(img_tensor, save_path)
                fp_count += 1
                
            # False Negative: Đoán là Real (0) nhưng sự thật là Fake (1)
            elif pred == 0 and true_val == 1:
                error_logs.append({'type': 'FN', 'vid_id': vid_id.item(), 'prob': prob})
                if fn_count < max_errors_to_save:
                    save_path = f"results/error_analysis/False_Negative/FN_prob{prob:.2f}_vid{vid_id.item()}_{idx}.jpg"
                    save_image_from_tensor(img_tensor, save_path)
                fn_count += 1

    df_errors = pd.DataFrame(error_logs)
    df_errors.to_csv("results/error_analysis/error_statistics.csv", index=False)
    
    print("\n📊 === TỔNG KẾT ERROR ANALYSIS ===")
    print(f"Tổng số False Positives (Ảnh thật bị nghi là giả): {fp_count}")
    print(f"Tổng số False Negatives (Ảnh giả qua mặt được model): {fn_count}")
    print("\n[✔] Ảnh minh họa lỗi đã được lưu vào thư mục results/error_analysis/")

if __name__ == "__main__":
    run_error_analysis()
