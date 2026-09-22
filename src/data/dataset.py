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
                            face_img = face.permute(1, 2, 0).numpy().astype(np.uint8)
                            self.samples.append({
                                'image': face_img, 
                                'label': label,
                                'video_id': vid_id 
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
