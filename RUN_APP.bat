@echo off
title Stock Trend Predictor - Setup and Launch
color 0A

echo ============================================
echo    SMART STOCK TRADING SYSTEM
echo    Auto Setup and Launch
echo ============================================
echo.

set "PYTHON312=C:\Users\Subhradeep Nath\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PYTHON312%" (
    echo ERROR: Python 3.12 not found!
    echo TensorFlow does NOT support Python 3.14.
    echo Install Python 3.12 from python.org
    pause
    exit /b 1
)

echo [OK] Python 3.12 found!
echo.

if not exist "venv" (
    echo [1/4] Creating virtual environment...
    "%PYTHON312%" -m venv venv
    echo Done!
) else (
    echo [1/4] Virtual environment already exists.
)
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo [3/4] Installing packages (first run takes 5-10 mins)...
pip install tensorflow==2.18.0
pip install keras streamlit yfinance pandas-datareader neuralprophet scikit-learn matplotlib plotly
echo.

echo [4/4] Launching app at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py

pause
