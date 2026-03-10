"""
Face Landmark 3D Visualizer
===========================
Script này đọc dữ liệu từ folder 'data' (JSON hoặc CSV) 
và hiển thị mesh khuôn mặt 3D tương tác.

Cách dùng:
  python visualize_3d.py                    # mặc định lấy file đầu tiên trong data/
  python visualize_3d.py --file data/image.json
  python visualize_3d.py --file data/image.csv
"""

import json
import csv
import argparse
import sys
import io

# Đảm bảo in ra đúng định dạng UTF-8 trên Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D

mp_face_mesh = mp.solutions.face_mesh

def load_data(file_path: Path):
    if not file_path.exists():
        print(f"[✗] Không tìm thấy file: {file_path}")
        return None

    suffix = file_path.suffix.lower()
    all_faces_data = []

    try:
        if suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                # Xử lý cả định dạng mới (dict) và cũ (list)
                for item in raw_data:
                    if isinstance(item, dict) and "parts" in item:
                        all_faces_data.append(item)
                    else:
                        # Convert list cũ sang format mới để đồng nhất
                        all_faces_data.append({"all": item, "parts": {}})
        elif suffix == ".csv":
            current_face_lms = []
            last_face_id = -1
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    f_id = int(row['face_id'])
                    if f_id != last_face_id:
                        if current_face_lms: 
                            all_faces_data.append({"all": current_face_lms, "parts": {}})
                        current_face_lms = []
                        last_face_id = f_id
                    current_face_lms.append({
                        "x": float(row['x']),
                        "y": float(row['y']),
                        "z": float(row['z'])
                    })
                if current_face_lms: 
                    all_faces_data.append({"all": current_face_lms, "parts": {}})
    except Exception as e:
        print(f"[✗] Lỗi khi đọc file: {e}")
        return None

    return all_faces_data

def plot_3d(all_faces, title):
    if not all_faces:
        print("[!] Không có dữ liệu để hiển thị.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    # Màu sắc tương phản mạnh
    PART_COLORS = {
        "lips": "#FF0000",          # Đỏ rực
        "left_eye": "#00FF00",      # Xanh lá
        "right_eye": "#00FF00",     # Xanh lá
        "left_eyebrow": "#FFFF00",  # Vàng
        "right_eyebrow": "#FFFF00", # Vàng
        "nose": "#00FFFF",          # Cyan
        "face_oval": "#FFFFFF",     # Trắng
        "iris": "#FF00FF",          # Hồng (Magenta)
    }

    for f_idx, face_data in enumerate(all_faces):
        landmarks = face_data["all"]
        
        # 1. Vẽ toàn bộ 468/478 điểm (Scatter)
        xs_all = [p['x'] for p in landmarks]
        ys_all = [p['y'] for p in landmarks]
        zs_all = [p['z'] for p in landmarks]
        ax.scatter(xs_all, zs_all, ys_all, c='white', s=1, alpha=0.3)

        # 2. Vẽ Khung lưới tam giác (Tesselation) - Màu xám mờ
        tess_connections = mp_face_mesh.FACEMESH_TESSELATION
        for start_idx, end_idx in tess_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1, p2 = landmarks[start_idx], landmarks[end_idx]
                ax.plot([p1['x'], p2['x']], [p1['z'], p2['z']], [p1['y'], p2['y']], 
                        c='#444444', linewidth=0.5, alpha=0.3)

        # 3. Vẽ các bộ phận chính (Contours/Parts) - Màu rực rỡ
        # Vẽ theo cặp kết nối (connections) có sẵn của MediaPipe
        PART_CONNECTIONS = {
            "lips": mp_face_mesh.FACEMESH_LIPS,
            "left_eye": mp_face_mesh.FACEMESH_LEFT_EYE,
            "right_eye": mp_face_mesh.FACEMESH_RIGHT_EYE,
            "left_eyebrow": mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            "right_eyebrow": mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            "face_oval": mp_face_mesh.FACEMESH_FACE_OVAL,
            "iris": mp_face_mesh.FACEMESH_IRISES
        }

        for part_name, connections in PART_CONNECTIONS.items():
            color = PART_COLORS.get(part_name, "#888888")
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    p1, p2 = landmarks[start_idx], landmarks[end_idx]
                    ax.plot([p1['x'], p2['x']], [p1['z'], p2['z']], [p1['y'], p2['y']], 
                            c=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Z (Depth)', color='white')
    ax.set_zlabel('Y', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    ax.set_title(f"3D Landmark Wireframe: {title}", color='cyan', fontsize=14)
    
    # Đồng bộ tọa độ: đảo ngược Y để giống không gian ảnh
    ax.set_zlim(max(ax.get_zlim()), min(ax.get_zlim()))
    
    # Không hiển thị grid để nhìn chuyên nghiệp hơn
    ax.grid(False)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.view_init(elev=-90, azim=-90)
    
    print("[*] Đang hiển thị cửa sổ 3D Wireframe rực rỡ.")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Face Landmarks in 3D Wireframe")
    parser.add_argument("--file", type=str, help="File JSON hoặc CSV")
    # parser.add_argument("--dir", type=str, default="data", help="Thư mục chứa data") # Removed as config.DATA_DIR is used
    args = parser.parse_args()

    data_dir = config.CURRENT_DATA_DIR # Use config.CURRENT_DATA_DIR
    if args.file:
        file_to_load = Path(args.file)
    else:
        if not data_dir.exists():
            print(f"[!] Không tìm thấy thư mục dữ liệu: {data_dir}")
            return
        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in [".json", ".csv"]])
        if not files:
            print(f"[!] Không tìm thấy dữ liệu trong '{data_dir}'")
            return
        file_to_load = files[0]

    print(f"[*] Đang tải: {file_to_load.name} ...")
    data = load_data(file_to_load)
    if data:
        plot_3d(data, file_to_load.name)

if __name__ == "__main__":
    main()
