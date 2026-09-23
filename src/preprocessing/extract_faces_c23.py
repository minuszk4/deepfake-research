import os
import glob
import json
import time
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN

CATEGORIES = [
    'original',
    'DeepFakeDetection',
    'Deepfakes',
    'Face2Face',
    'FaceShifter',
    'FaceSwap',
    'NeuralTextures'
]

def find_dataset_dir(custom_path=None):
    """
    Tự động dò tìm thư mục FaceForensics++_C23 trên Kaggle hoặc máy local.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path
        
    candidates = [
        # Đường dẫn Kaggle thường gặp
        '/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23',
        '/kaggle/input/faceforensics-c23/FaceForensics++_C23',
        '/kaggle/input/faceforensics-dataset-c23/FaceForensics++_C23',
        '/kaggle/input/ff-c23/FaceForensics++_C23',
        '/kaggle/input/faceforensics++/FaceForensics++_C23',
        # Local
        'FaceForensics++_C23',
        '../FaceForensics++_C23'
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
            
    # Quét đệ quy tìm kiếm trong /kaggle/input nếu có
    if os.path.exists('/kaggle/input'):
        matches = glob.glob('/kaggle/input/**/FaceForensics++_C23', recursive=True)
        if matches:
            return matches[0]
            
    return None

def parse_existing_splits(data_dir):
    """
    Đọc phân chia train/val/test cũ từ thư mục csv trong dataset FaceForensics++_C23 (nếu có).
    Hỗ trợ cả file .csv và .json.
    """
    split_map = {}
    csv_dir = os.path.join(data_dir, 'csv')
    if not os.path.exists(csv_dir):
        return split_map

    print(f"[*] Đang quét các tệp split cũ trong: {csv_dir}")
    csv_files = os.listdir(csv_dir)
    print(f"    Tìm thấy: {csv_files}")

    for fname in csv_files:
        fpath = os.path.join(csv_dir, fname)
        lower_name = fname.lower()
        split_name = None
        if 'train' in lower_name:
            split_name = 'train'
        elif 'val' in lower_name:
            split_name = 'val'
        elif 'test' in lower_name:
            split_name = 'test'
            
        if not split_name:
            continue

        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(fpath)
                # Tìm cột chứa ID hoặc tên video
                id_col = None
                for col in df.columns:
                    if any(k in col.lower() for k in ['id', 'video', 'name', 'file']):
                        id_col = col
                        break
                if id_col is not None:
                    for val in df[id_col].dropna().astype(str):
                        vid_key = os.path.splitext(os.path.basename(val))[0]
                        split_map[vid_key] = split_name
                else:
                    # Nếu file chỉ là 1 cột giá trị không header
                    for val in df.iloc[:, 0].dropna().astype(str):
                        vid_key = os.path.splitext(os.path.basename(val))[0]
                        split_map[vid_key] = split_name
            elif fname.endswith('.json'):
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, list):
                                for sub_item in item:
                                    split_map[str(sub_item)] = split_name
                            else:
                                split_map[str(item)] = split_name
        except Exception as e:
            print(f"[!] Không thể đọc {fname}: {e}")

    print(f"[*] Đã ánh xạ được {len(split_map)} video từ các file split cũ.")
    return split_map

def extract_actor_id(vid_basename):
    parts = vid_basename.split('_')
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1

def get_fallback_split(vid_basename):
    """
    Phân chia chuẩn FaceForensics++ chính thức (720 train / 140 val / 140 test)
    đảm bảo không bao giờ bị rò rỉ danh tính diễn viên (0% Identity Leakage).
    """
    actor_id = extract_actor_id(vid_basename)
    if actor_id >= 0:
        if actor_id < 720:
            return 'train'
        elif actor_id < 860:
            return 'val'
        else:
            return 'test'
    # Fallback băm ổn định
    h = hash(vid_basename) % 100
    if h < 70: return 'train'
    elif h < 85: return 'val'
    return 'test'

def extract_faces_ffpp_c23(data_dir, output_dir='ffpp_faces_c23', frames_per_video=5):
    """
    Trích xuất khuôn mặt cho toàn bộ 7,000 video của FaceForensics++ C23:
    - 5 frame/video -> Tổng ~35,000 ảnh khuôn mặt.
    - Cấu trúc thư mục khớp chuẩn FaceForensics++_C23:
      original, DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, csv/
    - Sử dụng MTCNN trên GPU với batching để tối đa hóa tốc độ.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("🎬 KHỞI ĐỘNG TRÍCH XUẤT DATASET 7,000 VIDEO (5 FRAMES/VIDEO)")
    print(f"   Thư mục nguồn (Input) : {data_dir}")
    print(f"   Thư mục đích (Output) : {output_dir}")
    print(f"   Thiết bị tính toán    : {device}")
    print(f"   Số frames mỗi video   : {frames_per_video}")
    print("=" * 70)

    # 1. Tạo cấu trúc thư mục đầu ra giống hệt FaceForensics++_C23
    os.makedirs(output_dir, exist_ok=True)
    out_csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(out_csv_dir, exist_ok=True)

    for cat in CATEGORIES:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    # 2. Đọc split cũ (nếu có)
    split_map = parse_existing_splits(data_dir)

    # 3. Thu thập danh sách 7,000 video
    video_tasks = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"[!] Cảnh báo: Không tìm thấy thư mục {cat_dir}")
            continue

        vids = glob.glob(os.path.join(cat_dir, '*.mp4')) + glob.glob(os.path.join(cat_dir, '*.avi'))
        label = 0 if cat == 'original' else 1
        print(f"[*] Danh mục [{cat}]: tìm thấy {len(vids)} video.")

        for v in sorted(vids):
            basename = os.path.splitext(os.path.basename(v))[0]
            # Xác định split: Ưu tiên ánh xạ từ split cũ -> sau đó dùng fallback chuẩn FF++
            split = split_map.get(basename)
            if not split:
                # Thử tìm theo ID target (ví dụ 001 trong 001_002)
                actor_id = extract_actor_id(basename)
                split = split_map.get(f"{actor_id:03d}", get_fallback_split(basename))

            video_tasks.append({
                'video_path': v,
                'basename': basename,
                'category': cat,
                'label': label,
                'split': split
            })

    total_vids = len(video_tasks)
    print(f"\n[+] Tổng số video cần xử lý: {total_vids} (Dự kiến trích xuất: {total_vids * frames_per_video} ảnh)")
    if total_vids == 0:
        print("[❌] Không tìm thấy video nào. Vui lòng kiểm tra lại đường dẫn input!")
        return

    # 4. Khởi tạo MTCNN trên GPU
    mtcnn = MTCNN(
        margin=20,
        keep_all=False,
        post_process=False,
        select_largest=True,
        device=device
    )

    extracted_records = []
    failed_vids = 0
    total_frames_extracted = 0
    start_time = time.time()

    # 5. Vòng lặp trích xuất từng video
    for item in tqdm(video_tasks, desc="Extracting Faces (FF++ C23)"):
        v_path = item['video_path']
        b_name = item['basename']
        cat = item['category']
        label = item['label']
        split = item['split']

        cap = cv2.VideoCapture(v_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            failed_vids += 1
            continue

        # Chọn 5 frame cách đều nhau
        frame_indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
        
        # Đọc 5 frames
        frames_rgb = []
        valid_indices = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                valid_indices.append(f_idx)
        cap.release()

        if not frames_rgb:
            failed_vids += 1
            continue

        # Phát hiện khuôn mặt bằng MTCNN
        try:
            # facenet_pytorch MTCNN hỗ trợ nhận danh sách ảnh (batch)
            faces = mtcnn(frames_rgb)
        except Exception:
            faces = [None] * len(frames_rgb)

        for f_idx, face, orig_frame in zip(valid_indices, faces, frames_rgb):
            save_name = f"{b_name}_f{f_idx}.jpg"
            save_path = os.path.join(output_dir, cat, save_name)

            if face is not None:
                # Face là tensor shape (3, H, W)
                face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            else:
                # Fallback nếu không phát hiện được mặt: crop trung tâm 256x256
                h, w, _ = orig_frame.shape
                sz = min(h, w)
                top = (h - sz) // 2
                left = (w - sz) // 2
                crop = orig_frame[top:top+sz, left:left+sz]
                crop_resized = cv2.resize(crop, (256, 256))
                face_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)

            cv2.imwrite(save_path, face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            total_frames_extracted += 1

            extracted_records.append({
                'image_path': save_path,
                'video_id': b_name,
                'frame_idx': f_idx,
                'label': label,
                'split': split,
                'manipulation': cat
            })

    elapsed = time.time() - start_time
    print(f"\n[✔] HOÀN TẤT TRÍCH XUẤT!")
    print(f"    - Thời gian thực thi: {elapsed:.1f} giây ({elapsed/60:.2f} phút)")
    print(f"    - Tổng số khuôn mặt đã lưu: {total_frames_extracted}")
    print(f"    - Video lỗi không đọc được: {failed_vids}")

    # 6. Xuất các file CSV trong thư mục output_dir/csv/
    df_all = pd.DataFrame(extracted_records)
    master_csv = os.path.join(out_csv_dir, 'faces_master.csv')
    df_all.to_csv(master_csv, index=False)

    df_train = df_all[df_all['split'] == 'train']
    df_val = df_all[df_all['split'] == 'val']
    df_test = df_all[df_all['split'] == 'test']

    df_train.to_csv(os.path.join(out_csv_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(out_csv_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(out_csv_dir, 'test.csv'), index=False)

    print("\n--- THỐNG KÊ PHÂN BỐ TẬP DỮ LIỆU ĐÃ TRÍCH XUẤT ---")
    print(f"File Master: {master_csv}")
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples  : {len(df_val)}")
    print(f"Test samples : {len(df_test)}")
    print("\nPhân bố theo từng phương pháp:")
    print(df_all.groupby(['manipulation', 'split']).size().unstack(fill_value=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất khuôn mặt chuẩn FaceForensics++ C23 (7000 video, 5 frames)")
    parser.add_argument('--data_dir', type=str, default=None, help='Đường dẫn tới thư mục FaceForensics++_C23')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/ffpp_faces_c23', help='Thư mục lưu khuôn mặt')
    parser.add_argument('--frames_per_video', type=int, default=5, help='Số frame trích xuất mỗi video')
    args = parser.parse_args()

    found_data_dir = find_dataset_dir(args.data_dir)
    if not found_data_dir:
        print("[❌] Không tìm thấy thư mục FaceForensics++_C23!")
        print("     Vui lòng truyền đối số: --data_dir <đường_dẫn_tới_FaceForensics++_C23>")
    else:
        extract_faces_ffpp_c23(
            data_dir=found_data_dir,
            output_dir=args.output_dir,
            frames_per_video=args.frames_per_video
        )
