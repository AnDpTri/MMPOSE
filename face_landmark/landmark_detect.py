import cv2
import sys
import time
from pathlib import Path
import mediapipe as mp

# Thêm đường dẫn gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO
from face_landmark.landmark_model import get_face_mesh

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def run_webcam():
    # 1. Tải YOLOv8 Face Model
    device = config.get_device()
    print(f"[*] Loading YOLOv8 Face Model: {config.YOLO_FACE_MODEL} (Device: {device})")
    yolo_model = YOLO(str(config.YOLO_FACE_MODEL)).to(device)
    
    # 2. Khởi tạo MediaPipe Face Mesh
    print(f"[*] Loading MediaPipe Face Mesh (max_faces={config.MP_MAX_FACES})")
    face_mesh = get_face_mesh(
        max_num_faces=config.MP_MAX_FACES, 
        refine_landmarks=config.MP_REFINE_LANDMARKS
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera.")
        return

    print("[▶] Running Face Detection + Landmarks. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # YOLO Detection
        results = yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop face and run landmarks
                face_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                  max(0, x1):min(frame.shape[1], x2)]
                
                if face_crop.size > 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    mp_results = face_mesh.process(rgb_crop)
                    
                    if mp_results.multi_face_landmarks:
                        for face_landmarks in mp_results.multi_face_landmarks:
                            # Map landmarks back to full frame
                            h_c, w_c = face_crop.shape[:2]
                            h_f, w_f = frame.shape[:2]
                            for lm in face_landmarks.landmark:
                                lm.x = (lm.x * w_c + x1) / w_f
                                lm.y = (lm.y * h_c + y1) / h_f
                            
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Face Landmark Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    run_webcam()
