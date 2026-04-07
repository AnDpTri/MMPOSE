#!/bin/bash
echo "========================================================="
echo "   GAZE ESTIMATION - RASPBERRY PI 4 SETUP (ARM/CPU)   "
echo "   Debian Trixie (13) / Python 3.13 Compatible      "
echo "========================================================="

# 1. Thu vien he thong (Optimized for Debian Trixie/Bookworm)
echo "[*] Cap nhat bo cai dat (apt)..."
sudo apt-get update

# Detecting Trixie for library renaming
if grep -q "trixie" /etc/os-release; then
    echo "[*] Phat hien Debian Trixie. Dang dung thu vien moi..."
    LIB_QT="libqt5gui5t64 libqt5test5t64"
    LIB_BLAS="libopenblas-dev"
else
    echo "[*] Dang dung thu vien Debian chuan..."
    LIB_QT="libqt5gui5 libqt5test5"
    LIB_BLAS="libatlas-base-dev"
fi

sudo apt-get install -y $LIB_QT $LIB_BLAS libopencv-dev libhdf5-dev python3-pip python3-venv python3-tk

# 2. Moi truong ao (Python 3.13 / 3.12 Support)
echo "[*] Tao moi truong ao (venv)..."
python3 -m venv venv
source venv/bin/activate

# 3. Cai dat thu vien Python
echo "[*] Cap nhat pip..."
pip install --upgrade pip

echo "[*] Cai dat thu vien tu requirements_rpi.txt..."
# Cap Numpy de tranh loi binary tren Python 3.13
pip install "numpy<2.0.0"
pip install -r requirements_rpi.txt

echo "========================================================="
echo "   DA XONG! He thong da san sang chay 100% ONNX Mode. "
echo "   De chay, hay dung lenh: "
echo "   source venv/bin/activate"
echo "   python3 gaze_estimation.py"
echo "========================================================="
chmod +x setup_rpi.sh
