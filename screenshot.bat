@echo off
rem Usage: screenshot.bat <video file>   - or drag and drop a video onto this file
rem Saves the first frame as <video>_first_frame.png next to the video.
setlocal
set "HERE=%~dp0"
if not exist "%HERE%py\python.exe" (
    echo Portable Python not found. Run setup.bat once on a PC with internet first.
    pause
    exit /b 1
)
"%HERE%py\python.exe" "%HERE%screenshot.py" %*
set RC=%ERRORLEVEL%
rem keep the window open when started by double-click / drag and drop
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b %RC%
