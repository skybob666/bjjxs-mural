@echo off
setlocal
cd /d "%~dp0"

rem Avoid malformed Windows proxy variables breaking DeepSeek/Stability HTTPS calls.
rem Set WALL_MURAL_USE_SYSTEM_PROXY=1 manually only when a verified proxy is required.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "WALL_MURAL_USE_SYSTEM_PROXY=0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running install.bat first...
    call install.bat
    if errorlevel 1 (
        echo Installation failed. The app will not be started.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
