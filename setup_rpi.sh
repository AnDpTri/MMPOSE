#!/bin/bash
set -e  # Dung ngay neu co bat ky lenh nao loi

echo "========================================================="
echo "   GAZE ESTIMATION - RASPBERRY PI 4 SETUP (ARM/CPU)   "
echo "   Debian Trixie (13) / Bookworm (12) Compatible    "
echo "========================================================="

# 0. Kiem tra quyen sudo
if [ "$EUID" -ne 0 ]; then
  echo "[!] LOI: Vui long chay script bang sudo: sudo bash setup_rpi.sh"
  exit 1
fi

# 1. Thu vien he thong (Optimized for Debian Trixie/Bookworm)
echo "[*] Cap nhat bo cai dat (apt)..."
apt-get update

# Detecting Trixie for library renaming
if grep -q "trixie" /etc/os-release; then
    echo "[*] Phat hien Debian Trixie (Debian 13). Dang cau hinh thu vien moi..."
    LIB_QT="libqt5gui5t64 libqt5test5t64"
    LIB_BLAS="libopenblas-dev"
else
    echo "[*] Dang dung thu vien Debian chuan (Stable)..."
    LIB_QT="libqt5gui5 libqt5test5"
    LIB_BLAS="libatlas-base-dev"
fi

echo "[*] Dang cai dat cac thu vien he thong..."
apt-get install -y $LIB_QT $LIB_BLAS libopencv-dev libhdf5-dev \
                   python3-pip python3-venv python3-tk python3-dev \
                   libusb-1.0-0-dev

# 2. Moi truong ao (Python 3.13 / 3.12 Support)
echo "[*] Tao moi truong ao (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Cai dat thu vien Python
echo "[*] Cap nhat pip & setuptools..."
pip install --upgrade pip setuptools wheel

echo "[*] Dang cai dat cac thu vien (requirements_rpi.txt)..."
# Cap Numpy de tranh loi binary tren Python 3.13/ARM
pip install "numpy<2.0.0"
pip install -r requirements_rpi.txt

echo "========================================================="
echo "   DA THIET LAP XONG! He thong ho tro Pure ONNX Mode. "
echo "   De chay, hay dung: "
echo "   source venv/bin/activate"
echo "   python3 gaze_estimation.py"
echo "========================================================="
chmod +x setup_rpi.sh
