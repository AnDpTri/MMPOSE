# Hướng dẫn Cấu hình & Chạy Gaze Estimation (PC & Raspberry Pi 4)

Dự án hình học ước lượng hướng nhìn (Geometric Gaze Estimation) hỗ trợ đa nền tảng, tự động nhận diện phần cứng để tối ưu hiệu năng.

## 1. Yêu cầu Hệ thống

### Trên Máy tính (PC / Laptop)
- **HĐH**: Windows 10/11 hoặc Linux x86_64.
- **GPU**: NVIDIA (Khuyên dùng để chạy ONNX GPU).
- **Yêu cầu**: Cài đặt CUDA Toolkit 11.8+ và cuDNN nếu muốn dùng GPU.

### Trên Raspberry Pi 4
- **HĐH**: Raspberry Pi OS 64-bit (AArch64) khuyên dùng. (Môi trường 32-bit KHÔNG hỗ trợ).
- **Python**: **BẮT BUỘC dùng Python 3.10, 3.11 hoặc 3.12**. (Hiện tại Python 3.13 CHƯA hỗ trợ MediaPipe trên ARM).
- **RAM**: Tối thiểu 4GB.

---

## 2. Cài đặt nhanh trên Windows (PC)

Bạn có thể chạy tệp `setup_pc.bat` hoặc thực hiện thủ công:

1. Tạo môi trường ảo:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy chương trình:
   ```bash
   python gaze_estimation.py
   ```

---

## 3. Cài đặt trên Raspberry Pi 4

Dự án tự động nhận diện RPi để chuyển sang chế độ tiết kiệm tài nguyên (CPU-only, High FPS Mode).

1. Cài đặt thư viện hệ thống (Bắt buộc):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libopencv-dev libatlas-base-dev libhdf5-dev libqt5gui5 libqt5test5
   ```
2. Tạo môi trường ảo và cài đặt:
   Nếu máy bạn có Python 3.13 làm mặc định, hãy dùng lệnh cài Python 3.11:
   ```bash
   sudo apt install python3.11-venv
   python3.11 -m venv venv
   ```
   Sau đó cài đặt như bình thường:
   ```bash
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements_rpi.txt
   ```
3. Chạy chương trình:
   ```bash
   python gaze_estimation.py
   ```

---

## 4. Các tính năng chính
- **Tự động cấu hình (Auto-Setup)**: Tự động phát hiện RPi để bật chế độ siêu tốc.
- **Dễ dàng triển khai**: Đã có sẵn `setup_rpi.sh` để tự động hóa toàn bộ quy trình trên.

---
*Chúc bạn có trải nghiệm tốt nhất với dự án!*
