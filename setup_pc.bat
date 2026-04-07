@echo off
setlocal
echo ===================================================
echo   GAZE ESTIMATION - WINDOWS SETUP (PC/CUDA)
echo ===================================================

:: 1. Detection of Python
echo [*] Kiem tra Python...
set PYTHON_CMD=python
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    set PYTHON_CMD=python3
    %PYTHON_CMD% --version >nul 2>&1
)

if %errorlevel% neq 0 (
    echo [!] LOI: Khong tim thay Python. Vui long cai dat tai https://www.python.org
    pause
    exit /b
)

:: 2. Version Check
for /f "tokens=2" %%v in ('%PYTHON_CMD% -c "import sys; print(sys.version_info[0]*10+sys.version_info[1])"') do set PY_VER=%%v
if %PY_VER% LSS 38 (
    echo [!] LOI: Yeu cau Python 3.8 tro len. Ban dang dung: %PY_VER%
    pause
    exit /b
)

:: 3. Virtual Environment
if not exist venv (
    echo [*] Dang tao moi truong ao (venv)...
    %PYTHON_CMD% -m venv venv
) else (
    echo [*] venv da ton tai. Dang tiep tuc...
)

:: 4. Activate and Install
echo [*] Kich hoat venv va cap nhat pip...
call venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

echo [*] Dang cai dat cac thu vien (requirements.txt)...
echo [!] Luu y: Buoc nay co the mat vai phut tuy theo toc do mang...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [!] LOI: Cai dat thu vien that bai. Vui long kiem tra ket noi mang.
    pause
    exit /b
)

:: 5. Verification
echo [*] Kiem tra trang thai CUDA (Optional)...
python -c "import torch; print('---------------------------------'); print('CUDA Available:', torch.cuda.is_available()); print('Device count :', torch.cuda.device_count()); print('---------------------------------')"

echo ===================================================
echo   DA THIET LAP XONG! 
echo   De chay: python gaze_estimation.py
echo ===================================================
pause
