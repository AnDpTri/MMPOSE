# GEOMETRIC GAZE ESTIMATION (Multi-Platform)
Dự án ước lượng hướng mắt (Gaze Estimation) dựa trên hình học không gian, hỗ trợ đa nền tảng (PC/CUDA & Raspberry Pi 4).

## 🚀 Tính năng nổi bật
- **Auto-Hardware Detection**: Tự động nhận diện phần cứng (PC/RPi 4) để cấu hình engine tối ưu nhất.
- **Pure ONNX Mode**: Hỗ trợ chạy trên Python 3.13 (Debian Trixie) bằng engine ONNX internally.
- **Hybrid Engine**: Tự động chuyển đổi giữa MediaPipe (Fast) và ONNX (Robust/Trixie).
- **Visualization**: Hỗ trợ xem 2D debug và 3D tương tác.

## 📁 Cấu trúc thư mục chuẩn
- `models/`: Chứa tất cả tệp trọng số (`.pt`, `.onnx`).
- `input/`: Nơi chứa ảnh/video đầu vào.
- `output/`: Kết quả xử lý (ảnh, video, dữ liệu CSV).
- `venv/`: Môi trường ảo (sau khi cài đặt).

## 🛠️ Hướng dẫn cài đặt "Clone & Run"

### 1. Trên Windows (PC/CUDA)
- Mở Terminal tại thư mục dự án.
- Chạy lệnh:
  ```cmd
  setup_pc.bat
  ```

### 2. Trên Raspberry Pi 4 (Debian 12/13/Trixie)
- Mở Terminal.
- Chạy lệnh:
  ```bash
  sudo bash setup_rpi.sh
  ```
- **Lưu ý**: Trên Debian Trixie (Python 3.13), hệ thống sẽ tự động dùng chế độ **Pure ONNX Mode** để đảm bảo không lỗi.

## 🖥️ Cách chạy chương trình
Sau khi cài đặt xong, hãy kích hoạt môi trường ảo và chạy:
```bash
# Windows
venv\Scripts\activate
python gaze_estimation.py

# Raspberry Pi / Linux
source venv/bin/activate
python3 gaze_estimation.py
```

## 📋 Yêu cầu thư viện (Requirements)
- **PC**: `requirements.txt` (Hỗ trợ GPU/CUDA).
- **Raspberry Pi**: `requirements_rpi.txt` (Tối ưu cho CPU ARM/TFLite).
