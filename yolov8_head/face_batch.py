import cv2
import sys
import os
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO

def run_batch():
    model_path = config.YOLO_FACE_MODEL
    device = config.get_device()
    input_dir = config.INPUT_DIR
    output_dir = config.OUTPUT_BASE / "face"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Loading Face Model for Batch: {model_path}")
    model = YOLO(str(model_path)).to(device)
    
    images = list(input_dir.glob("*.[jJ][pP][gG]")) + list(input_dir.glob("*.[pP][nN][gG]"))
    print(f"[*] Found {len(images)} images in {input_dir}")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        results = model(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"[✓] Saved: {out_path}")

if __name__ == "__main__":
    run_batch()
