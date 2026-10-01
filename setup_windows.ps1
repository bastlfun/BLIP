# Builds the portable bundle: py\ (embedded Python + packages), models\, ffmpeg\
# Run once on a Windows PC WITH internet, then copy the whole folder to the offline PC.
param(
    [switch]$Cpu,                        # CPU-only torch (~1 GB smaller, no NVIDIA needed)
    [string]$Cuda = "cu128",             # torch CUDA build: cu126 (older drivers) / cu128 (RTX 50xx needs this)
    [string]$PythonVersion = "3.12.10",
    [string[]]$Models = @("yolo11s.pt"), # add e.g. yolo11m.pt for better accuracy (slower)
    [switch]$NoFfmpeg                    # skip ffmpeg (cutting then has no audio, no ffmpeg decoder)
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
$root = $PSScriptRoot
$py = Join-Path $root "py"
$pyExe = Join-Path $py "python.exe"
$tmp = Join-Path $root "_download"
New-Item -ItemType Directory -Force $tmp, (Join-Path $root "models") | Out-Null

# 1. embedded Python
if (-not (Test-Path $pyExe)) {
    Write-Host "== Downloading Python $PythonVersion (embeddable)"
    $zip = Join-Path $tmp "python.zip"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $zip
    Expand-Archive $zip $py -Force
    # enable site-packages (disabled by default in the embeddable build)
    $pth = Get-ChildItem $py -Filter "python*._pth" | Select-Object -First 1
    (Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' | Set-Content $pth.FullName -Encoding ascii
}

# 2. pip
if (-not (Test-Path (Join-Path $py "Scripts\pip.exe"))) {
    Write-Host "== Installing pip"
    $getpip = Join-Path $tmp "get-pip.py"
    Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip
    & $pyExe $getpip --no-warn-script-location
    if ($LASTEXITCODE) { throw "pip install failed" }
}

# 3. torch first (so ultralytics does not pull the CPU build from PyPI), then the rest
$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/$Cuda" }
Write-Host "== Installing torch from $index (large download)"
& $pyExe -m pip install --no-warn-script-location torch torchvision --index-url $index
if ($LASTEXITCODE) { throw "torch install failed" }
Write-Host "== Installing requirements"
& $pyExe -m pip install --no-warn-script-location -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE) { throw "requirements install failed" }

# 4. model weights (the tool never downloads anything itself)
foreach ($m in $Models) {
    $dst = Join-Path $root "models\$m"
    if (-not (Test-Path $dst)) {
        Write-Host "== Downloading model $m"
        Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v8.3.0/$m" -OutFile $dst
    }
}

# 5. ffmpeg (static build, ends up as ffmpeg\ffmpeg.exe + ffmpeg\ffprobe.exe)
$ffDir = Join-Path $root "ffmpeg"
$ffExe = Join-Path $ffDir "ffmpeg.exe"
if (-not $NoFfmpeg -and -not (Test-Path $ffExe)) {
    $mirrors = @(
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    )
    foreach ($url in $mirrors) {
        try {
            Write-Host "== Downloading ffmpeg from $url"
            $zip = Join-Path $tmp "ffmpeg.zip"
            $unz = Join-Path $tmp "ffmpeg"
            Invoke-WebRequest $url -OutFile $zip
            Expand-Archive $zip $unz -Force
            New-Item -ItemType Directory -Force $ffDir | Out-Null
            Get-ChildItem $unz -Recurse -Include ffmpeg.exe, ffprobe.exe | Copy-Item -Destination $ffDir -Force
            if (Test-Path $ffExe) { break }
        } catch {
            Write-Warning "ffmpeg download failed: $($_.Exception.Message)"
        }
    }
    if (-not (Test-Path $ffExe)) {
        Write-Warning "ffmpeg could not be installed. The tools still work without it; cut_video.py then has no audio."
        Write-Warning "You can also copy ffmpeg.exe and ffprobe.exe into $ffDir by hand."
    }
}

Remove-Item $tmp -Recurse -Force
Write-Host "== Check"
& $pyExe -c "import torch, cv2, ultralytics; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('opencv', cv2.__version__, '| ultralytics', ultralytics.__version__)"
if (Test-Path $ffExe) { (& $ffExe -hide_banner -version | Select-Object -First 1) } else { Write-Host "ffmpeg: not installed" }
Write-Host "`nDone. Copy this whole folder to the offline PC and run:  detect.bat <video.mp4>"
