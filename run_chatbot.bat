@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "VENV_PYTHON=%ROOT_DIR%.venv312\Scripts\python.exe"
set "APP_DIR=%ROOT_DIR%chatbot"

if not exist "%VENV_PYTHON%" (
    echo Virtual environment not found at:
    echo %VENV_PYTHON%
    echo.
    echo Create it first with:
    echo py -3.12 -m venv .venv312
    pause
    exit /b 1
)

if not exist "%APP_DIR%\app.py" (
    echo Could not find Streamlit app at:
    echo %APP_DIR%\app.py
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
cd /d "%APP_DIR%"

echo Starting Ayurvedic chatbot...
echo Open http://localhost:8501 in your browser if it does not open automatically.
echo Press Ctrl+C in this window to stop the app.
echo.

"%VENV_PYTHON%" -m streamlit run app.py

endlocal
