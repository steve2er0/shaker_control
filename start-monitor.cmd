@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-windows.ps1" -Monitor
set "SHAKER_EXIT_CODE=%ERRORLEVEL%"
if not "%SHAKER_EXIT_CODE%"=="0" pause
exit /b %SHAKER_EXIT_CODE%
