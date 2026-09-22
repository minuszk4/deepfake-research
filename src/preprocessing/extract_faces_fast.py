import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import sys
sys.path.append('.')

def extract_all_faces_turbo(csv_path='master_split.csv', output_dir='/kaggle/working/ffpp_faces', frames_per_video=3):
    """
    Trích xuất khuôn mặt SIÊU TỐC bằng GPU CUDA.
    Khởi tạo MTCNN 1 lần trên GPU, tối ưu đọc frame và ghi file JPG.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Kích hoạt TURBO GPU Face Extraction trên thiết bị: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Khởi tạo MTCNN duy nhất 1 lần trên GPU
    mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)
    
    all_extracted_samples = []
    
    print(f"[*] Bắt đầu xử lý {len(df)} video (Mỗi video {frames_per_video} frames)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Turbo Face Extracting"):
        vid_path = row['video_path']
        label = row['label']
        split = row['split']
        manipulation = row['manipulation']
        
        vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
        
        cap = cv2.VideoCapture(vid_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            continue
            
        frame_idxs = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
        
        # Thư mục lưu
        save_subdir = os.path.join(output_dir, split, 'fake' if label == 1 else 'real')
        os.makedirs(save_subdir, exist_ok=True)
        
        for f_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                face = mtcnn(frame_rgb)
                if face is not None:
                    # Chuyển tensor từ GPU về CPU numpy để lưu ảnh
                    face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    
                    save_filename = f"{vid_basename}_f{f_idx}.jpg"
                    save_path = os.path.join(save_subdir, save_filename)
                    cv2.imwrite(save_path, face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    all_extracted_samples.append({
                        'image_path': save_path,
                        'video_id': vid_basename,
                        'label': label,
                        'split': split,
                        'manipulation': manipulation
                    })
            except Exception:
                pass
                
        cap.release()
        
    # Xuất file metadata CSV
    df_faces = pd.DataFrame(all_extracted_samples)
    faces_csv_path = os.path.join(output_dir, 'faces_master.csv')
    df_faces.to_csv(faces_csv_path, index=False)
    
    print(f"\n[✔] HOÀN TẤT TRÍCH XUẤT TURBO!")
    print(f"    - Tổng số khuôn mặt đã trích xuất: {len(df_faces)}")
    print(f"    - File metadata: {faces_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='master_split.csv')
    parser.add_argument('--output_dir', default='/kaggle/working/ffpp_faces')
    parser.add_argument('--frames_per_video', type=int, default=3)
    args = parser.parse_args()
    
    extract_all_faces_turbo(args.csv_path, args.output_dir, args.frames_per_video)
