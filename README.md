# BLIP
This tool detects movements from a video file. It is designed for hours-long static security camera footage and can detect movement, categorize them by type (car, person, bicycle, ...) by default, or switch to the detection of any movement for more hits.

## Intro
* Tool works on offline PCs, but for first installation (if there is no `/py`, go to an online PC to download the packages)
* Supported file formats: `.mp4, .mov, .m4v, .avi, .mkv, .ts/.mts, .mpg, .wmv, .flv, .webm, .3gp`
* Recommended: NVIDIA GPU (cuda support)


## Quick-Start (USB-Stick given)

Open the terminal in the USB stick code folder, or drag a video file onto `detect.bat`.

```powershell
detect.bat D:\cam\cam1.mp4  # moving people/vehicles, less hits
detect.bat D:\cam\cam1.mp4 --no-object-detection # any movement, much more hits
cut.bat D:\cam\cam1.mp4 00:12:30 00:14:00 # cut a clip
```

Both detect modes can run at the same time in two terminals, and multiple videos can be run at the same time.

Useful options:

```
--sensitivity 20                      more sensitive (lower = more sensitive, default 32)
--ignore 0,0,400,40                   ignore an area, e.g. the camera clock (x,y,w,h in pixels)
--snapshot-every 10                   screenshot interval in seconds
--help                                all options
```

## Installation

If you only have the source code (no `py\` folder), do this once on a Windows PC with internet:

1. Double-click `setup.bat`. Use `setup.bat -Cpu` for PCs without an NVIDIA GPU.
2. Copy the whole folder to the USB stick.

## Technical Details

Order: the video is read at 5 frames per second, motion detection compares each frame against a background model that remembers the last ~60 s (so slow light changes are ignored), only frames with motion are passed to object detection, only detected objects that overlap the motion are kept (so parked cars are ignored), and consecutive hits are grouped into events with screenshots (with `--no-object-detection` the object detection step is skipped).

| Part | Uses | Interchangeable via |
|---|---|---|
| Runtime | Python 3.12 (Windows embeddable, in `py\`) | `setup.bat -PythonVersion 3.13.x` |
| Motion detection | OpenCV MOG2 background subtraction (CPU) | `--sensitivity`, `--min-area`, `--analyze-fps` |
| Object detection | YOLO11s (`models\yolo11s.pt`, COCO's 80 classes) via Ultralytics + PyTorch | `--model models\yolo11m.pt` (any Ultralytics `.pt`: larger/smaller YOLO, YOLOv8, custom-trained), `--classes person,car` |
| Hardware | NVIDIA GPU via CUDA, falls back to CPU automatically | `--device cpu` / `cuda:1`, `setup.bat -Cuda cu126` (older drivers) or `-Cpu` |
| Video decoding | OpenCV (built-in FFmpeg), falls back to `ffmpeg\ffmpeg.exe` | `--decoder ffmpeg`; replace `ffmpeg\` with any ffmpeg build |
| Cutting | ffmpeg (H.264 re-encode, or `--copy` for no re-encode) | without ffmpeg it uses OpenCV (no audio) |

- All logic is in `detect_movements.py` and `cut_video.py`. The `.bat` files only start them with the bundled Python.
- Nothing is downloaded at runtime; the tool works fully offline.
- Ultralytics is licensed under AGPL-3.0. 
