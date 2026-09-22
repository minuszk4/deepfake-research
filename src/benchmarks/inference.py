import time
import torch
import yaml
import os
import pandas as pd
import sys
sys.path.append('.')

from src.models.factory import create_model

def load_config(config_path="configs/baseline.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency_and_throughput(model, device, img_size, warmup=10, iterations=50):
    model.eval()
    
    # 1. Pure Inference Latency (Batch = 1 - Đại diện cho Web Deployment)
    dummy_input_b1 = torch.randn(1, 3, img_size, img_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input_b1)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input_b1)
            if device.type == 'cuda': torch.cuda.synchronize()
    latency_ms = ((time.time() - start_time) / iterations) * 1000

    # 2. Throughput (Batch = 16 - Đại diện cho Server Processing)
    dummy_input_b16 = torch.randn(16, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input_b16)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input_b16)
            if device.type == 'cuda': torch.cuda.synchronize()
    throughput_fps = (16 * iterations) / (time.time() - start_time)

    return latency_ms, throughput_fps

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Chạy Deployment Benchmark trên thiết bị: {device}")
    
    os.makedirs('results/benchmarks', exist_ok=True)
    benchmark_data = []

    for model_key, m_cfg in config['models'].items():
        print(f"--> Đang đo lường mô hình: {model_key}")
        model = create_model(model_key, m_cfg, num_classes=1).to(device)
        
        params = count_parameters(model)
        latency_ms, throughput_fps = measure_latency_and_throughput(
            model, device, m_cfg['img_size']
        )
        
        # Ước lượng kích thước model (MB) ở FP32
        model_size_mb = (params * 4) / (1024 * 1024)
        
        benchmark_data.append({
            'Model': model_key,
            'Input Size': m_cfg['img_size'],
            'Params (M)': round(params / 1e6, 2),
            'Model Size (MB)': round(model_size_mb, 2),
            'Latency Batch=1 (ms)': round(latency_ms, 2),
            'Throughput Batch=16 (FPS)': round(throughput_fps, 2)
        })
        
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()

    df_bench = pd.DataFrame(benchmark_data)
    print("\n🏆 === BẢNG ĐO LƯỜNG HIỆU NĂNG TÍCH HỢP (DEPLOYMENT TRADEOFF) ===")
    print(df_bench.to_string(index=False))
    
    df_bench.to_csv('results/benchmarks/hardware_benchmark.csv', index=False)
    print("\n[✔] Đã lưu bảng kết quả vào results/benchmarks/hardware_benchmark.csv")

if __name__ == "__main__":
    main()
