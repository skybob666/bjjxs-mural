@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 停止网页服务和公网隧道，释放 8501 端口
echo 正在停止公网服务...
taskkill /f /im cloudflared.exe >nul 2>&1
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /f /pid %%P >nul 2>&1
echo 已停止。网址已经失效，重新运行“启动公网网址.bat”会得到一个新网址。
timeout /t 3 >nul
