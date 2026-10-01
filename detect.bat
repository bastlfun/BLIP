@echo off
rem Usage: detect.bat <video file> [--no-object-detection] [options]   - or drag and drop a video onto this file
setlocal
set "HERE=%~dp0"
if not exist "%HERE%py\python.exe" (
    echo Portable Python not found. Run setup.bat once on a PC with internet first.
    pause
    exit /b 1
)
"%HERE%py\python.exe" "%HERE%detect_movements.py" %*
set RC=%ERRORLEVEL%
rem keep the window open when started by double-click / drag and drop
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b %RC%
