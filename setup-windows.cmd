@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-windows.ps1" %*
set "SHAKER_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %SHAKER_EXIT_CODE%
