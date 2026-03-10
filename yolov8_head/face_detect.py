import cv2
import sys
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO

def run_webcam():
    model_path = config.YOLO_FACE_MODEL
    device = config.get_device()
    
    print(f"[*] Loading Face Model: {model_path} (Device: {device})")
    model = YOLO(str(model_path)).to(device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera.")
        return

    print("[▶] Running Face Detection (Webcam). Press 'Q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
