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

    # 2. PHÂN TÍCH METADATA (Lấy mẫu đại diện)
    print("\n--- 2. Phân tích Metadata Video (Lấy mẫu 100 video) ---")
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    meta_records = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Quét Metadata"):
        cap = cv2.VideoCapture(row['video_path'])
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
    axes[2].set_title("Phân bố Độ phân giải")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "video_metadata.png"), dpi=300)
    plt.close()

    # 3. THỐNG KÊ PHỔ TẦN SỐ CAO FFT (Statistical FFT Energy)
    print("\n--- 3. Thống kê năng lượng FFT tần số cao ---")
    real_sample_path = df[df['label'] == 0]['video_path'].iloc[0]
    fake_sample_path = df[df['label'] == 1]['video_path'].iloc[0]

    def extract_fft_spectrum(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        return gray, magnitude

    real_img, real_fft = extract_fft_spectrum(real_sample_path)
    fake_img, fake_fft = extract_fft_spectrum(fake_sample_path)

    if real_fft is not None and fake_fft is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(real_img, cmap='gray'); axes[0, 0].set_title("Real Frame (256x256)")
        axes[0, 1].imshow(real_fft, cmap='viridis'); axes[0, 1].set_title("Real FFT Spectrum")

        axes[1, 0].imshow(fake_img, cmap='gray'); axes[1, 0].set_title("Fake Frame (256x256)")
        axes[1, 1].imshow(fake_fft, cmap='viridis'); axes[1, 1].set_title("Fake FFT Spectrum")

        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fft_comparison.png"), dpi=300)
        plt.close()

    print(f"[✔] EDA Hoàn tất! Toàn bộ biểu đồ và CSV thống kê được lưu tại: {output_dir}")

if __name__ == "__main__":
    run_comprehensive_eda()
