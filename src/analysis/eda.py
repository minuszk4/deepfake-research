import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def run_comprehensive_eda(csv_path="master_split.csv", output_dir="results/eda"):
    """
    Thực hiện EDA chuẩn Research:
    - Thống kê chi tiết nhãn, phương pháp giả mạo (manipulation types)
    - Phân tích metadata video (Resolution, FPS, Frame counts)
    - Thống kê năng lượng phổ tần số cao (High-frequency FFT) giữa Real vs Fake
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"[!] Không tìm thấy file {csv_path}. Chạy identity_split.py trước.")
        return

    df = pd.read_csv(csv_path)
    print(f"[*] Tổng số video trong dataset: {len(df)}")

    # 1. PHÂN BỐ SPLIT VÀ MANIPULATION TYPES
    print("\n--- 1. Thống kê phân bố Dataset ---")
    split_summary = pd.crosstab(df['split'], df['manipulation'], margins=True)
    print(split_summary)
    split_summary.to_csv(os.path.join(output_dir, "dataset_distribution.csv"))

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='manipulation', hue='split', palette='Set2')
    plt.title("Phân bố Video theo Phương pháp Giả mạo & Split", fontsize=14)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manipulation_distribution.png"), dpi=300)
    plt.close()

    base_dir = os.path.dirname(csv_path)

    def resolve_path(p):
        if os.path.exists(p): return p
        if base_dir:
            normalized = p.replace('\\', '/')
            root_dir = os.path.dirname(base_dir) if os.path.basename(base_dir) == 'csv' else base_dir
            for prefix in ['/ffpp_faces_c23/', '/ffpp_faces/']:
                if prefix in normalized:
                    rel = normalized.split(prefix)[-1]
                    cand = os.path.join(root_dir, rel.replace('/', os.sep))
                    if os.path.exists(cand): return cand
            parts = normalized.split('/')
            if len(parts) >= 2:
                cand_cat = os.path.join(root_dir, parts[-2], parts[-1])
                if os.path.exists(cand_cat): return cand_cat
            cand2 = os.path.join(root_dir, os.path.basename(p))
            if os.path.exists(cand2): return cand2
        return p

    is_image_dataset = 'image_path' in df.columns
    path_col = 'image_path' if is_image_dataset else 'video_path'

    # 2. PHÂN TÍCH METADATA
    print(f"\n--- 2. Phân tích Metadata ({'Ảnh Khuôn mặt' if is_image_dataset else 'Video'}) ---")
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    meta_records = []

    if is_image_dataset:
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Quét Metadata Ảnh"):
            actual_path = resolve_path(str(row['image_path']))
            img = cv2.imread(actual_path)
            if img is not None:
                h, w, _ = img.shape
                meta_records.append({
                    'label': 'Real' if row['label'] == 0 else 'Fake',
                    'resolution': f"{w}x{h}",
                    'width': w,
                    'height': h
                })
        df_meta = pd.DataFrame(meta_records)
        
        # Thống kê số lượng frame trên mỗi video
        frames_per_vid = df.groupby('video_id').size().reset_index(name='frame_count')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(data=frames_per_vid, x='frame_count', bins=10, ax=axes[0], color='teal')
        axes[0].set_title("Phân bố số lượng Frame trích xuất / Video")
        
        if not df_meta.empty:
            sns.countplot(data=df_meta, x='resolution', hue='label', ax=axes[1])
            axes[1].set_title("Phân bố Độ phân giải Khuôn mặt (Cắt từ MTCNN)")
            axes[1].tick_params(axis='x', rotation=45)
    else:
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Quét Metadata Video"):
            cap = cv2.VideoCapture(str(row['video_path']))
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                meta_records.append({
                    'label': 'Real' if row['label'] == 0 else 'Fake',
                    'frames': frames,
                    'fps': fps,
                    'resolution': f"{w}x{h}"
                })
            cap.release()
        df_meta = pd.DataFrame(meta_records)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(data=df_meta, x='frames', hue='label', kde=True, ax=axes[0])
        axes[0].set_title("Phân bố số lượng Frame/Video")

        sns.countplot(data=df_meta, x='fps', hue='label', ax=axes[1])
        axes[1].set_title("Phân bố Tốc độ khung hình (FPS)")

        sns.countplot(data=df_meta, x='resolution', hue='label', ax=axes[2])
        axes[2].set_title("Phân bố Độ phân giải Video")
        axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metadata_analysis.png"), dpi=300)
    plt.close()

    # 3. THỐNG KÊ PHỔ TẦN SỐ CAO FFT (Statistical FFT Energy)
    print("\n--- 3. Thống kê năng lượng FFT tần số cao ---")
    real_sample_path = resolve_path(str(df[df['label'] == 0][path_col].iloc[0]))
    fake_sample_path = resolve_path(str(df[df['label'] == 1][path_col].iloc[0]))

    def extract_fft_spectrum(file_path, is_img):
        if is_img:
            frame = cv2.imread(file_path)
            if frame is None: return None, None
        else:
            cap = cv2.VideoCapture(file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None: return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        return gray, magnitude

    real_img, real_fft = extract_fft_spectrum(real_sample_path, is_image_dataset)
    fake_img, fake_fft = extract_fft_spectrum(fake_sample_path, is_image_dataset)

    if real_fft is not None and fake_fft is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(real_img, cmap='gray'); axes[0, 0].set_title("Real Sample (256x256)")
        axes[0, 1].imshow(real_fft, cmap='viridis'); axes[0, 1].set_title("Real FFT Spectrum")

        axes[1, 0].imshow(fake_img, cmap='gray'); axes[1, 0].set_title("Fake Sample (256x256)")
        axes[1, 1].imshow(fake_fft, cmap='viridis'); axes[1, 1].set_title("Fake FFT Spectrum")

        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fft_comparison.png"), dpi=300)
        plt.close()
        print(f"[✔] Đã lưu biểu đồ FFT tại: {os.path.join(output_dir, 'fft_comparison.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy phân tích dữ liệu EDA")
    parser.add_argument('--csv_path', type=str, default=None, help='Đường dẫn file CSV (master_split.csv hoặc faces_master.csv)')
    parser.add_argument('--output_dir', type=str, default='results/eda', help='Thư mục lưu biểu đồ')
    args = parser.parse_args()
    
    csv_target = args.csv_path
    if csv_target is None:
        candidates = [
            '/kaggle/working/ffpp_faces_c23/csv/faces_master.csv',
            '/kaggle/working/ffpp_faces_c23/faces_master.csv',
            'ffpp_faces_c23/csv/faces_master.csv',
            'ffpp_faces_c23/faces_master.csv',
            '/kaggle/input/datasets/min2k4/face-ff/kaggle/working/ffpp_faces/faces_master.csv',
            '/kaggle/working/ffpp_faces/faces_master.csv',
            'master_split.csv',
            'faces_master.csv'
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_target = c
                break
    if csv_target is None:
        csv_target = 'master_split.csv'
        
    run_comprehensive_eda(csv_path=csv_target, output_dir=args.output_dir)
