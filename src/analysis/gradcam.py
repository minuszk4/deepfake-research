import os
import torch
import cv2
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append('.')
from src.models.factory import create_model
from src.data.dataset import DeepfakeDataset, PreExtractedFaceDataset

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_target_layer(model, model_key):
    """
    Xác định layer cuối cùng của mô hình để vẽ Grad-CAM
    """
    if model_key == 'meso4':
        return [model.conv4]
    elif model_key == 'freq_meso':
        return [model.fuse_conv[0]]
    elif 'mobilenet' in model_key:
        return [model.conv_head]
    elif 'efficientnet' in model_key:
        return [model.conv_head]
    elif 'xception' in model_key:
        return [model.conv4]
    elif 'vit' in model_key:
        return [model.blocks[-1].norm1]
    return [list(model.children())[-2]] # Default heuristic

def run_gradcam():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị: {device}")
    
    os.makedirs('results/gradcam', exist_ok=True)
    
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
        for c in candidates:
            if os.path.exists(c):
                faces_csv = c
                break
        if not os.path.exists(faces_csv) and os.path.exists('/kaggle/input'):
            import glob
            matches = glob.glob('/kaggle/input/**/faces_master.csv', recursive=True)
            if matches:
                faces_csv = matches[0]

    use_pre_extracted = (mode == 'faces' and os.path.exists(faces_csv))

    if use_pre_extracted:
        df_faces = pd.read_csv(faces_csv)
        df_sample = pd.concat([
            df_faces[(df_faces['split'] == 'test') & (df_faces['label'] == 0)].head(2),
            df_faces[(df_faces['split'] == 'test') & (df_faces['label'] == 1)].head(2)
        ])
        base_dir = os.path.dirname(faces_csv)
    else:
        df_master = pd.read_csv('master_split.csv') if os.path.exists('master_split.csv') else pd.DataFrame()
        if df_master.empty:
            print("[!] Không tìm thấy dữ liệu mẫu cho Grad-CAM.")
            return
        df_sample = pd.concat([
            df_master[(df_master['split'] == 'test') & (df_master['label'] == 0)].head(2),
            df_master[(df_master['split'] == 'test') & (df_master['label'] == 1)].head(2)
        ])
        base_dir = None

    for model_key, m_cfg in config['models'].items():
        ckpt_path = f"results/checkpoints/best_{model_key}.pth"
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"\n--> Trực quan hóa Grad-CAM cho: {model_key}")
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        target_layers = get_target_layer(model, model_key)
        reshape_fn = reshape_transform_vit if 'vit' in model_key else None
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_fn)
        
        transform = A.Compose([
            A.Resize(m_cfg['img_size'], m_cfg['img_size']),
            A.Normalize(), ToTensorV2()
        ])
        
        if use_pre_extracted:
            ds = PreExtractedFaceDataset(df_sample, transform=transform, base_dir=base_dir)
        else:
            ds = DeepfakeDataset(df_sample, transform=transform, frames_per_video=1, device=device)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        
        fig, axes = plt.subplots(len(ds), 3, figsize=(12, 4 * max(1, len(ds))), squeeze=False)
        
        for idx, (img_tensor, label, vid_id) in enumerate(loader):
            input_tensor = img_tensor.to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            # Undo normalize để hiển thị ảnh gốc
            img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img_original = np.clip(img_np * std + mean, 0, 1)
            
            cam_image = show_cam_on_image(img_original, grayscale_cam, use_rgb=True)
            
            true_lbl = "Fake" if label.item() == 1 else "Real"
            pred_prob = torch.sigmoid(model(input_tensor)).item()
            pred_lbl = "Fake" if pred_prob >= 0.5 else "Real"
            
            axes[idx, 0].imshow(img_original); axes[idx, 0].set_title(f"Original (True: {true_lbl})"); axes[idx, 0].axis('off')
            axes[idx, 1].imshow(grayscale_cam, cmap='jet'); axes[idx, 1].set_title("Heatmap"); axes[idx, 1].axis('off')
            axes[idx, 2].imshow(cam_image); axes[idx, 2].set_title(f"Superimposed (Pred: {pred_lbl} {pred_prob:.2f})"); axes[idx, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"results/gradcam/{model_key}_gradcam.png", dpi=300)
        plt.close()
        del model
        
    print("\n[✔] Hoàn tất Grad-CAM! Bản đồ nhiệt được lưu tại results/gradcam/")

if __name__ == "__main__":
    run_gradcam()
