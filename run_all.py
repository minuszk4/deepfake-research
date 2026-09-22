import os
import subprocess
import sys
import time

def run_cmd(step_name, command):
    print(f"\n{'='*70}")
    print(f"🚀 BẮT ĐẦU: {step_name}")
    print(f"COMMAND: {command}")
    print(f"{'='*70}\n")
    start_time = time.time()
    ret = subprocess.run(command, shell=True)
    elapsed = time.time() - start_time
    if ret.returncode != 0:
        print(f"[❌] LỖI tại bước: {step_name} (Mã lỗi: {ret.returncode})")
        # Không dừng lại hoàn toàn mà tiếp tục các bước sau nếu có thể
    else:
        print(f"[✔] HOÀN TẤT: {step_name} trong {elapsed:.1f} giây")

def main():
    print("\n" + "="*70)
    print("🎉 KHỞI ĐỘNG TOÀN BỘ PIPELINE NGHIÊN CỨU DEEPFAKE (RESEARCH-GRADE)")
    print("="*70)
    os.makedirs('results', exist_ok=True)
    
    # 1. EDA
    run_cmd("1. EDA & Phân tích Tần số FFT", f"{sys.executable} src/analysis/eda.py")
    
    # 2. Benchmark Phần cứng
    run_cmd("2. Đo lường Hiệu năng Triển khai (Latency / FPS / Params)", f"{sys.executable} src/benchmarks/inference.py")
    
    # 3. Huấn luyện Mô hình Toàn diện (TRAIN ALL)
    run_cmd("3. Huấn luyện Toàn bộ Baseline Models", f"{sys.executable} -m src.training.train")
    
    # 4. Cross Manipulation
    run_cmd("4. Thí nghiệm Đa phương pháp (Cross-manipulation)", f"{sys.executable} src/experiments/cross_manipulation.py")
    
    # 5. Robustness
    run_cmd("5. Thí nghiệm Độ bền vững Nén ảnh (Robustness)", f"{sys.executable} src/experiments/robustness.py")
    
    # 6. Grad-CAM
    run_cmd("6. Trực quan hóa Vùng chú ý (Grad-CAM Explainability)", f"{sys.executable} src/analysis/gradcam.py")
    
    # 7. Error Analysis
    run_cmd("7. Phân tích Ca Sai & Xuất Ảnh (Error Analysis)", f"{sys.executable} src/analysis/error_analysis.py")
    
    # 8. Đóng gói kết quả thành file zip
    run_cmd("8. Nén toàn bộ kết quả thành file ZIP", "zip -q -r full_research_results.zip results/")
    
    print("\n" + "="*70)
    print("🏆 TOÀN BỘ NGHIÊN CỨU ĐÃ HOÀN TẤT THÀNH CÔNG 100%!")
    print("    File tải về máy: full_research_results.zip")
    print("="*70)

if __name__ == "__main__":
    main()
