import cv2
import sys
import json
from pathlib import Path
import mediapipe as mp

# Thêm đường dẫn gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO
from face_landmark.landmark_model import get_face_mesh_static

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def run_batch():
    device = config.get_device()
    input_dir = config.INPUT_DIR
    output_dir = config.CURRENT_OUTPUT_DIR
    data_dir = config.CURRENT_DATA_DIR
    
    print(f"[*] Loading Models for Batch (Type: {config.MODEL_TYPE})")
    model_path = config.YOLO_FACE_MODEL if config.MODEL_TYPE == 'face' else config.YOLO_HEAD_MODEL
    yolo_model = YOLO(str(model_path)).to(device)
    
    face_mesh = get_face_mesh_static(
        max_num_faces=config.MP_MAX_FACES, 
        refine_landmarks=config.MP_REFINE_LANDMARKS
    )
    
    images = list(input_dir.glob("*.[jJ][pP][gG]")) + list(input_dir.glob("*.[pP][nN][gG]"))
    print(f"[*] Found {len(images)} images")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        orig_h, orig_w = frame.shape[:2]
        
        results = yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
        all_landmarks = []
        
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_crop = frame[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]
                
                if face_crop.size > 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    mp_results = face_mesh.process(rgb_crop)
                    
                    if mp_results.multi_face_landmarks:
                        face_landmarks = mp_results.multi_face_landmarks[0]
                        landmarks_data = []
                        h_c, w_c = face_crop.shape[:2]
                        
                        for lm in face_landmarks.landmark:
                            # Map to original frame coordinates for JSON
                            abs_x = (lm.x * w_c + x1) / orig_w
                            abs_y = (lm.y * h_c + y1) / orig_h
                            landmarks_data.append({"x": abs_x, "y": abs_y, "z": lm.z})
                            
                            # Update landmark coordinates for drawing
                            lm.x = abs_x
                            lm.y = abs_y
                        
                        all_landmarks.append({"id": i, "landmarks": landmarks_data})
                        
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save results
        cv2.imwrite(str(output_dir / img_path.name), frame)
        if all_landmarks:
            with open(data_dir / f"{img_path.stem}.json", 'w') as f:
                json.dump(all_landmarks, f, indent=2)
        
        print(f"[✓] Processed: {img_path.name}")

    face_mesh.close()

if __name__ == "__main__":
    run_batch()
