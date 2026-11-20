@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem 一条命令同时启动网页服务 + 公网隧道，结束时打印可分享的网址
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "WALL_MURAL_USE_SYSTEM_PROXY=0"

if not exist "tmp" mkdir "tmp"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 未找到虚拟环境，正在安装...
    call install.bat
    if errorlevel 1 (
        echo 安装失败，无法启动。
        pause
        exit /b 1
    )
)

if not exist "tmp\cloudflared.exe" (
    echo [2/4] 首次运行，正在下载 cloudflared 隧道工具...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'tmp\cloudflared.exe'" 
    if not exist "tmp\cloudflared.exe" (
        echo 下载失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

echo [3/4] 启动网页服务（端口 8501）...
start "壁画智护-服务" /min ".venv\Scripts\python.exe" -m streamlit run app.py

echo [4/4] 建立公网隧道，请稍候约 20 秒...
if exist "tmp\cloudflared.log" del /q "tmp\cloudflared.log"
start "壁画智护-隧道" /min "tmp\cloudflared.exe" tunnel --url http://127.0.0.1:8501 --no-autoupdate --logfile "tmp\cloudflared.log"

set "URL="
timeout /t 18 >nul
for /f "delims=" %%U in ('powershell -NoProfile -Command "$m=Select-String -Path 'tmp\cloudflared.log' -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches | Select-Object -Last 1; if($m){$m.Matches[0].Value}"') do set "URL=%%U"

echo.
echo ============================================================
if defined URL (
    echo  公网网址（发给任何设备都能打开）：
    echo.
    echo      !URL!
    echo.
    echo  同一 WiFi 下的设备也可以直接用：
    echo      http://%COMPUTERNAME%:8501
) else (
    echo  还没拿到网址，请查看窗口 "壁画智护-隧道" 里的提示。
)
echo ============================================================
echo.
echo 注意：这两个黑窗口不能关闭，关闭网址就失效。
echo 关闭时运行 停止公网服务.bat 即可。
echo.
pause
