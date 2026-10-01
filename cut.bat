@echo off
rem Usage: cut.bat video.mp4 00:12:30 00:14:00 [--copy]
setlocal
set "HERE=%~dp0"
if not exist "%HERE%py\python.exe" (
    echo Portable Python not found. Run setup.bat once on a PC with internet first.
    exit /b 1
)
"%HERE%py\python.exe" "%HERE%cut_video.py" %*
