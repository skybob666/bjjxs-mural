@echo off
setlocal
cd /d "%~dp0"

rem Avoid inheriting a malformed proxy value that breaks old pip/urllib3 on Windows.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "NO_PROXY=localhost,127.0.0.1,pypi.org,.pypi.org,files.pythonhosted.org"
set "no_proxy=%NO_PROXY%"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Python virtual environment creation failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m ensurepip --upgrade >nul 2>&1
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel --no-cache-dir --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
if errorlevel 1 (
    echo pip bootstrap failed. Check network or proxy settings, then run install.bat again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt --no-cache-dir --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)
echo Installation completed.
pause
