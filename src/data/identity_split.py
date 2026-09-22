import os
import glob
import networkx as nx
import pandas as pd
import random

def build_identity_safe_split(data_dir, output_csv="master_split.csv", train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Tạo phân chia Train/Val/Test an toàn tuyệt đối (100% Identity Leakage-free).
    Sử dụng lý thuyết Đồ thị (Graph Connected Components) để gộp các diễn viên có
    tương tác trong video fake (source_target) vào cùng một cụm.
    """
    random.seed(seed)
    
    # 1. Thu thập tất cả video
    real_videos = glob.glob(os.path.join(data_dir, 'original', '*.mp4'))
    
    fake_folders = ['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    fake_videos = []
    for folder in fake_folders:
        fake_videos.extend(glob.glob(os.path.join(data_dir, folder, '*.mp4')))
        
    print(f"[*] Đã tìm thấy {len(real_videos)} Real videos và {len(fake_videos)} Fake videos.")
    if not real_videos and not fake_videos:
        print("[!] Dataset trống. Hãy kiểm tra lại đường dẫn.")
        return
        
    # 2. Xây dựng Đồ thị Identity
    G = nx.Graph()
    
    # Hàm lấy ID (Ví dụ: 001_002.mp4 -> [1, 2], 001.mp4 -> [1])
    def extract_ids(filepath):
        basename = os.path.basename(filepath).replace('.mp4', '')
        parts = basename.split('_')
        return [int(p) for p in parts if p.isdigit()]
        
    # Thêm node (diễn viên gốc)
    for rv in real_videos:
        ids = extract_ids(rv)
        if ids: G.add_node(ids[0])
            
    # Thêm edge (quan hệ giữa source và target trong video fake)
    for fv in fake_videos:
        ids = extract_ids(fv)
        if len(ids) >= 2:
            G.add_edge(ids[0], ids[1])
        elif len(ids) == 1:
            G.add_node(ids[0])

    # 3. Trích xuất các Cụm liên thông (Connected Components)
    components = list(nx.connected_components(G))
    # Sắp xếp để đảm bảo tính tái lập (reproducibility) trước khi shuffle
    components = sorted([sorted(list(c)) for c in components])
    random.shuffle(components)
    
    print(f"[*] Tổng số diễn viên (Nodes): {G.number_of_nodes()}")
    print(f"[*] Đã phân tách thành {len(components)} cụm độc lập (Components).")

    # 4. Phân bổ các cụm vào Train/Val/Test
    train_ids, val_ids, test_ids = set(), set(), set()
    total_nodes = G.number_of_nodes()
    
    for comp in components:
        comp_size = len(comp)
        if len(train_ids) + comp_size <= total_nodes * train_ratio:
            train_ids.update(comp)
        elif len(val_ids) + comp_size <= total_nodes * val_ratio:
            val_ids.update(comp)
        else:
            test_ids.update(comp)
            
    print(f"[*] Phân bổ ID: Train ({len(train_ids)}), Val ({len(val_ids)}), Test ({len(test_ids)})")

    # 5. Gắn nhãn Split cho từng Video
    def get_split(filepath):
        ids = extract_ids(filepath)
        if not ids: return 'unknown'
        # Do ta dùng connected components, tất cả ID trong 1 video chắc chắn CÙNG 1 split.
        actor_id = ids[0]
        if actor_id in train_ids: return 'train'
        if actor_id in val_ids: return 'val'
        if actor_id in test_ids: return 'test'
        return 'unknown'
        
    records = []
    for rv in real_videos:
        records.append({'video_path': rv, 'label': 0, 'split': get_split(rv), 'manipulation': 'original'})
    for fv in fake_videos:
        manipulation = os.path.basename(os.path.dirname(fv))
        records.append({'video_path': fv, 'label': 1, 'split': get_split(fv), 'manipulation': manipulation})
        
    df = pd.DataFrame(records)
    
    # 6. Kiểm tra chéo (Sanity Check) đảm bảo Leakage = 0
    train_v = df[df['split']=='train']
    val_v = df[df['split']=='val']
    test_v = df[df['split']=='test']
    
    def get_unique_actors(df_subset):
        actors = set()
        for p in df_subset['video_path']:
            actors.update(extract_ids(p))
        return actors
        
    t_actors = get_unique_actors(train_v)
    v_actors = get_unique_actors(val_v)
    te_actors = get_unique_actors(test_v)
    
    assert len(t_actors.intersection(v_actors)) == 0, "⚠️ LEAKAGE: Train và Val trùng Actor!"
    assert len(t_actors.intersection(te_actors)) == 0, "⚠️ LEAKAGE: Train và Test trùng Actor!"
    assert len(v_actors.intersection(te_actors)) == 0, "⚠️ LEAKAGE: Val và Test trùng Actor!"
    
    print("[✔] KIỂM TRA CHÉO THÀNH CÔNG: KHÔNG TỒN TẠI IDENTITY LEAKAGE GIỮA CÁC TẬP.")
    
    # Xuất file
    df.to_csv(output_csv, index=False)
    print(f"[*] Đã xuất file {output_csv} thành công!")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))

if __name__ == "__main__":
    # Test mẫu (để chạy thử, trên Kaggle bạn sẽ trỏ data_dir vào '/kaggle/input/...')
    build_identity_safe_split(data_dir='./dummy_data', output_csv='master_split.csv')
