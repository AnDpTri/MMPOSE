"""
MMPOSE Central Configuration
============================
Quản lý các tham số tập trung cho toàn bộ dự án.
"""

import torch
from pathlib import Path

# ─── ĐƯỜNG DẪN GỐC ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── THIẾT LẬP CHÍNH (THE HEART) ──────────────────────────────────────────────
# 'face' : Tự động chạy YOLO Face -> MediaPipe Landmarks
# 'head' : Chỉ chạy YOLO Head Detection
MODEL_TYPE = 'face' 

# 'webcam' : Chạy thời gian thực qua camera
# 'batch'  : Chạy xử lý hàng loạt ảnh trong thư mục input
RUN_MODE = 'batch'

# ─── THIẾT BỊ XỬ LÝ (CPU/GPU) ────────────────────────────────────────────────
# 'cpu' : Chạy trên CPU
# '0'   : Chạy trên GPU NVIDIA thứ nhất (nếu có CUDA)
# 'auto': Tự động tìm kiếm GPU, nếu không có sẽ dùng CPU
DEVICE = 'auto' 

# ─── CẤU HÌNH YOLOv8 ─────────────────────────────────────────────────────────
YOLO_MODEL_SIZE = 'n'        # n, s, m, l, x
YOLO_CONF_THRESHOLD = 0.50
YOLO_FACE_MODEL = BASE_DIR / "yolov8_head" / "yolov8n-face.pt"
YOLO_HEAD_MODEL = BASE_DIR / "yolov8_head" / "yolov8n-head.pt"

# ─── CẤU HÌNH MEDIAPIPE ──────────────────────────────────────────────────────
MP_MAX_FACES = 5
MP_REFINE_LANDMARKS = True
MP_STATIC_MODE = (RUN_MODE == 'batch')

# ─── THƯ MỤC DỮ LIỆU (GLOBAL ROOT) ───────────────────────────────────────────
# Tự động điều chỉnh đường dẫn dựa trên MODEL_TYPE
INPUT_DIR = BASE_DIR / "input"
OUTPUT_BASE = BASE_DIR / "output"
DATA_BASE = BASE_DIR / "data"

# Đường dẫn cụ thể cho từng loại detect
CURRENT_OUTPUT_DIR = OUTPUT_BASE / MODEL_TYPE
CURRENT_DATA_DIR = DATA_BASE / MODEL_TYPE

# Tạo thư mục nếu chưa tồn tại
CURRENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    """Trả về thiết bị xử lý chuẩn xác nhất dựa trên cấu hình."""
    target_device = DEVICE
    if isinstance(target_device, str) and target_device.isdigit():
        target_device = int(target_device)
        
    if target_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if target_device != 'cpu':
        if not torch.cuda.is_available():
            print(f"[!] Cảnh báo: Yêu cầu GPU '{target_device}' nhưng không khả dụng. Dùng 'cpu'.")
            return 'cpu'
            
    return target_device
