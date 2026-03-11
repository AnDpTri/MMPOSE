"""
Live Landmark Detection (Webcam)
================================
Xử lý thời gian thực: Camera -> YOLO -> MediaPipe -> Mesh Visualization.
"""

import cv2
import sys
import torch
import numpy as np
import time
from pathlib import Path
import mediapipe as mp

# Setup Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO
from face_landmark.landmark_model import get_face_mesh
from face_landmark.face_parts import FACE_PARTS_INDICES, FACE_PARTS_COLORS, MESH_CONNECTIONS, IRIS_CONNECTIONS

def calculate_target_point(p168, p331, p102, k=150.0):
    """Tính điểm đích của Vector pháp tuyến 3D"""
    try:
        p168, p331, p102 = np.array(p168), np.array(p331), np.array(p102)
        v1 = p331 - p168
        v2 = p102 - p168
        # v1 x v2 theo chuẩn không gian Upright Right-Hand-Rule (như trong test.py)
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm == 0: return p168
        return p168 + k * (n / norm)
    except:
        return p168

def run_live():
    # 1. Setup Environment
    device = config.get_device()
    print(f"[*] Starting Live Mode (Type: {config.MODEL_TYPE}, Device: {device})")
    
    # 2. Load Models
    model_path = config.YOLO_FACE_MODEL if config.MODEL_TYPE == 'face' else config.YOLO_HEAD_MODEL
    yolo_model = YOLO(str(model_path)).to(device)
    face_mesh = get_face_mesh(static_image_mode=False) # Video stream mode
    
    # 3. Setup Camera (Sử dụng DirectShow trên Windows để tránh lỗi kẹt frame)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Cannot open webcam.")
        return

    print("[*] Streaming! Press 'ESC' to quit.")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[!] Lỗi: Không thể lấy frame từ Camera. (Camera có thể bị ứng dụng khác chiếm dụng)")
            break
        
        orig_h, orig_w = frame.shape[:2]
        
        # --- Detection & Smart Fallback ---
        results = yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, imgsz=640, verbose=False)[0]
        
        if (results.boxes is None or len(results.boxes) == 0) and config.MODEL_TYPE == 'head':
            face_results = YOLO(str(config.YOLO_FACE_MODEL)).to(device)(frame, conf=0.3, imgsz=640, verbose=False)[0]
            if face_results.boxes is not None and len(face_results.boxes) > 0:
                results = face_results
                for box in results.boxes:
                    coords = box.xyxy[0].tolist()
                    h_box = coords[3] - coords[1]
                    coords[1] = max(0, coords[1] - h_box * 0.4)
                    box.xyxy[0] = torch.tensor(coords).to(device)

        # --- Processing & Visualization ---
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                try:
                    mp_results = face_mesh.process(rgb_crop)
                except Exception as e:
                    print(f"[!] MediaPipe Error: {e}")
                    continue
                
                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    h_c, w_c = crop.shape[:2]
                    pixel_coords = {}
                    points_3d = {}
                    
                    # Store and Draw Landmarks
                    for part_name, indices in FACE_PARTS_INDICES.items():
                        for idx in indices:
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                
                                # Tọa độ pixel để vẽ
                                px = int(lm.x * w_c + x1)
                                py = int(lm.y * h_c + y1)
                                pixel_coords[idx] = (px, py)
                                
                                # Lưu tọa độ normalize gốc của MediaPipe cho việc tính Gaze Vector
                                # (x, y ∈ [0,1] trong crop, z tương đối)
                                points_3d[idx] = (lm.x, lm.y, lm.z)
                                
                                # Highlight Iris (Gaze Estimation focus)
                                if part_name == "iris":
                                    cv2.circle(frame, (px, py), 2, FACE_PARTS_COLORS[part_name], -1)
                                else:
                                    cv2.circle(frame, (px, py), 1, FACE_PARTS_COLORS[part_name], -1)

                    # Draw Mesh Connections
                    for conn in MESH_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            color = (130, 130, 130) # Default
                            for part, p_indices in FACE_PARTS_INDICES.items():
                                if s_idx in p_indices:
                                    color = FACE_PARTS_COLORS[part]
                                    break
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], color, 1)

                    # VẼ RIÊNG CÁC KẾT NỐI IRIS
                    for conn in IRIS_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], FACE_PARTS_COLORS["iris"], 1)
                            if s_idx in [468, 473]: # Specific iris points to highlight
                                cv2.circle(frame, pixel_coords[s_idx], 2, (255, 255, 255), -1)

                    # --- GAZE VECTOR PROJECTION ---
                    if 168 in points_3d and 331 in points_3d and 102 in points_3d:
                        p168 = points_3d[168]
                        p331 = points_3d[331]
                        p102 = points_3d[102]
                        
                        # Tính tọa độ đích trong hệ tọa độ normalize của MediaPipe
                        try:
                            # k=0.3 tương đương 30% chiều rộng khuôn mặt
                            target_3d = calculate_target_point(p168, p331, p102, k=0.3)
                            
                            # Chiếu ngược về 2D pixel đơn giản:
                            # x_norm -> int(x_norm * w_c + x1)
                            # y_norm -> int(y_norm * h_c + y1)
                            target_px = int(target_3d[0] * w_c + x1)
                            target_py = int(target_3d[1] * h_c + y1)
                            start_px_168, start_py_168 = pixel_coords[168]
                            
                            # Vẽ Tam giác tham chiếu 168-331-102 (Cyan)
                            pts = np.array([pixel_coords[168], pixel_coords[331], pixel_coords[102]], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
                            
                            # Vẽ Vector Gaze (Vàng) và Điểm đích (Đỏ)
                            cv2.line(frame, (start_px_168, start_py_168), (target_px, target_py), (0, 255, 255), 2)
                            cv2.circle(frame, (target_px, target_py), 4, (0, 0, 255), -1)
                            cv2.circle(frame, (start_px_168, start_py_168), 4, (0, 255, 255), -1)
                        except Exception as e:
                            print(f"[!] Lỗi khi vẽ Vector: {e}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show Output (Phải luôn chạy bất chấp có Box hay không)
        try:
            cv2.imshow("MMPOSE: Live Gaze Ready Landmarks", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == 27: # ESC
                break
        except Exception as e:
            print(f"[!] Lỗi khi gọi imshow: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    run_live()
