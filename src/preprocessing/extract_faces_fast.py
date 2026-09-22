import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append('.')

def process_single_video(args):
    """
    Xử lý đọc và cắt khuôn mặt cho 1 video
    """
    row, output_dir, frames_per_video, mtcnn_device = args
    vid_path = row['video_path']
    label = row['label']
    split = row['split']
    manipulation = row['manipulation']
    
    # Đặt tên file mặt dựa trên tên video
    vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
    
    cap = cv2.VideoCapture(vid_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return []
        
    frame_idxs = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    extracted_records = []
    
    # Khởi tạo MTCNN riêng cho thread (nếu chạy CPU) hoặc chung nếu dùng GPU
    # Để an toàn đa luồng trên CPU của Kaggle:
    mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device='cpu')
    
    for f_idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = mtcnn(frame_rgb)
            if face is not None:
                face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                
                # Lưu file ảnh JPG
                save_filename = f"{vid_basename}_frame{f_idx}.jpg"
                save_subdir = os.path.join(output_dir, split, 'fake' if label == 1 else 'real')
                os.makedirs(save_subdir, exist_ok=True)
                
                save_path = os.path.join(save_subdir, save_filename)
                cv2.imwrite(save_path, face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                extracted_records.append({
                    'image_path': save_path,
                    'video_id': vid_basename,
                    'label': label,
                    'split': split,
                    'manipulation': manipulation
                })
        except Exception:
            pass
            
    cap.release()
    return extracted_records

def extract_all_faces_fast(csv_path='master_split.csv', output_dir='/kaggle/working/ffpp_faces', frames_per_video=5, num_workers=4):
    """
    Trích xuất khuôn mặt siêu tốc dùng Đa luồng (Multi-threading CPU)
    Lưu trực tiếp thành các file ảnh .jpg và tạo file metadata 'faces_master.csv'
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"[*] Bắt đầu trích xuất đa luồng cho {len(df)} video bằng {num_workers} CPU workers...")
    
    tasks = [(row, output_dir, frames_per_video, 'cpu') for _, row in df.iterrows()]
    
    all_extracted_samples = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for records in tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Extracting Faces (Parallel)"):
            all_extracted_samples.extend(records)
            
    # Xuất file CSV định tuyến cho Dataset mới
    df_faces = pd.DataFrame(all_extracted_samples)
    faces_csv_path = os.path.join(output_dir, 'faces_master.csv')
    df_faces.to_csv(faces_csv_path, index=False)
    
    print(f"\n[✔] HOÀN TẤT TRÍCH XUẤT!")
    print(f"    - Tổng số ảnh khuôn mặt đã cắt: {len(df_faces)}")
    print(f"    - Thư mục chứa ảnh: {output_dir}")
    print(f"    - File metadata: {faces_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='master_split.csv')
    parser.add_argument('--output_dir', default='/kaggle/working/ffpp_faces')
    parser.add_argument('--frames_per_video', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    extract_all_faces_fast(args.csv_path, args.output_dir, args.frames_per_video, args.num_workers)
