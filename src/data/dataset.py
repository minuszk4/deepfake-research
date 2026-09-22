import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from tqdm import tqdm

class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None, frames_per_video=3, device='cpu'):
        self.df = df
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.device = device
        
        self.mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=self.device)
        self.samples = []
        self.error_logs = []
        
        self._extract_all_faces()
        self._print_extraction_stats()

    def _extract_all_faces(self):
        total_attempts = 0
        failed_attempts = 0
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting Faces"):
            vid_path = row['video_path']
            label = row['label']
            vid_id = row.name # Dùng index giả làm ID video để đánh giá Video-level sau này
            
            manipulation = row['manipulation'] if 'manipulation' in row else 'unknown'
            
            cap = cv2.VideoCapture(vid_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0: continue
            
            frame_idxs = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=int)
            for f_idx in frame_idxs:
                total_attempts += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        face = self.mtcnn(frame)
                        if face is not None:
                            face_img = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            self.samples.append({
                                'image': face_img, 
                                'label': label,
                                'video_id': vid_id,
                                'manipulation': manipulation
                            })
                        else:
                            failed_attempts += 1
                            self.error_logs.append({'video': vid_path, 'frame': f_idx, 'error': 'No face detected'})
                    except Exception as e:
                        failed_attempts += 1
                        self.error_logs.append({'video': vid_path, 'frame': f_idx, 'error': str(e)})
            cap.release()
            
        self.total_attempts = total_attempts
        self.failed_attempts = failed_attempts

    def _print_extraction_stats(self):
        print(f"[*] Trích xuất khuôn mặt hoàn tất.")
        print(f"    - Tổng số frames xử lý: {self.total_attempts}")
        print(f"    - Số frames bị lỗi (Failed/No face): {self.failed_attempts}")
        if self.total_attempts > 0:
            print(f"    - Tỷ lệ lỗi: {(self.failed_attempts / self.total_attempts) * 100:.2f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img, label, vid_id = sample['image'], sample['label'], sample['video_id']
        
        if self.transform:
            img = self.transform(image=img)['image']
            
        return img, torch.tensor(label, dtype=torch.float32), torch.tensor(vid_id, dtype=torch.long)

class PreExtractedFaceDataset(Dataset):
    """
    Dataset đọc trực tiếp từ các file ảnh .jpg đã cắt sẵn.
    Tốc độ nhanh gấp 50-100 lần so với đọc từ video .mp4.
    """
    def __init__(self, df, transform=None, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.base_dir = base_dir
        
        # Tạo mapping video_id dạng số để phục vụ video-level aggregation
        unique_vids = {vid: i for i, vid in enumerate(self.df['video_id'].unique())}
        self.df['vid_num_id'] = self.df['video_id'].map(unique_vids)
        
    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path):
        if os.path.exists(path):
            return path
        if self.base_dir:
            # 1. Thay thế prefix nếu ảnh được lưu từ /kaggle/working/ffpp_faces
            normalized = path.replace('\\', '/')
            if '/ffpp_faces/' in normalized:
                rel = normalized.split('/ffpp_faces/')[-1]
                candidate = os.path.join(self.base_dir, rel.replace('/', os.sep))
                if os.path.exists(candidate): return candidate
            # 2. Thử ghép cấu trúc thư mục con (split/label/filename)
            parts = normalized.split('/')
            if len(parts) >= 3:
                candidate2 = os.path.join(self.base_dir, parts[-3], parts[-2], parts[-1])
                if os.path.exists(candidate2): return candidate2
            # 3. Thử trực tiếp trong base_dir
            candidate3 = os.path.join(self.base_dir, os.path.basename(path))
            if os.path.exists(candidate3): return candidate3
        return path
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row['image_path']))
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            img = self.transform(image=img)['image']
            
        return img, torch.tensor(row['label'], dtype=torch.float32), torch.tensor(row['vid_num_id'], dtype=torch.long)

