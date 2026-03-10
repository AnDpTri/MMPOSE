import config
import sys
import os

def main():
    print(f"--- MMPOSE System Starting ---")
    print(f"Model Type: {config.MODEL_TYPE.upper()}")
    print(f"Run Mode: {config.RUN_MODE.upper()}")
    print(f"Device: {config.get_device()}")
    print(f"------------------------------")

    if config.MODEL_TYPE == 'face':
        if config.RUN_MODE == 'webcam':
            # Chạy nhận diện mặt + landmark qua webcam
            from face_landmark import landmark_detect
            landmark_detect.run_webcam()
        elif config.RUN_MODE == 'batch':
            # Chạy nhận diện mặt + landmark hàng loạt
            from face_landmark import batch_landmark
            batch_landmark.run_batch()
            
    elif config.MODEL_TYPE == 'head':
        if config.RUN_MODE == 'webcam':
            # Chạy nhận diện đầu qua webcam
            from yolov8_head import head_detect
            head_detect.run_webcam()
        elif config.RUN_MODE == 'batch':
            # Chạy nhận diện đầu hàng loạt
            from yolov8_head import head_batch
            head_batch.run_batch()
    
    else:
        print(f"[!] Error: Unknown MODEL_TYPE '{config.MODEL_TYPE}' in config.py")

if __name__ == "__main__":
    main()
