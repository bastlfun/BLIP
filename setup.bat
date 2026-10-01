@echo off
rem One-time setup (needs internet). Afterwards the whole folder works offline.
rem   setup.bat              NVIDIA GPU build + ffmpeg (falls back to CPU at runtime)
rem   setup.bat -Cpu         smaller CPU-only build
rem   setup.bat -NoFfmpeg    skip ffmpeg (installed by default)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
pause
