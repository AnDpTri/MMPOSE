#!/bin/bash
echo "========================================================="
echo "   GAZE ESTIMATION - RASPBERRY PI 4 SETUP (ARM/CPU)   "
echo "========================================================="

# 1. Thu vien he thong
echo "[*] Cap nhat bo cai dat (apt)..."
sudo apt-get update
sudo apt-get install -y libopencv-dev libatlas-base-dev libhdf5-dev libqt5gui5 libqt5test5 python3-pip python3-venv

# 2. Moi truong ao (Tu dong tim Python phu hop)
echo "[*] Kiem tra phien ban Python phu hop..."
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [ "$PY_VER" == "3.13" ]; then
    echo "[!] Canh bao: MediaPipe CHUA ho tro Python 3.13 tren RPi."
    echo "[*] Dang tim kiem Python 3.11 hoac 3.12..."
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_EXEC=python3.11
    elif command -v python3.12 >/dev/null 2>&1; then
        PYTHON_EXEC=python3.12
    else
        echo "[!] Khong tim thay Python 3.11/3.12. Dang thu cai dat python3.11..."
        sudo apt install -y python3.11-venv python3.11-dev
        PYTHON_EXEC=python3.11
    fi
else
    PYTHON_EXEC=python3
fi

echo "[*] Dang su dung luồng: $PYTHON_EXEC"
$PYTHON_EXEC -m venv venv
source venv/bin/activate

# 3. Cai dat thu vien Python
echo "[*] Cap nhat pip..."
pip install --upgrade pip

echo "[*] Cai dat thu vien tu requirements_rpi.txt..."
pip install -r requirements_rpi.txt

echo "========================================================="
echo "   DA XONG! De chay, hay dung lenh: "
echo "   source venv/bin/activate"
echo "   python3 gaze_estimation.py"
echo "========================================================="
chmod +x setup_rpi.sh
