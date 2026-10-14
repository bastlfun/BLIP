# BLIP
This tool detects movements from a video file. It is designed for hours-long static security camera footage and can detect movement, categorize them by type (car, person, bicycle, ...) by default, or switch to the detection of any movement for more hits.

## Intro
* Tool works on offline PCs, but for first installation (if there is no `/py`, go to an online PC to download the packages)
* Supported input file formats: `.mp4, .mov, .m4v, .avi, .mkv, .ts/.mts, .mpg, .wmv, .flv, .webm, .3gp`
* Recommended: NVIDIA GPU (cuda support)


## Quick-Start (USB-Stick given)

Open the terminal in the USB stick code folder, or drag a video file onto `detect.bat`.

```powershell
# QUICK PLAY command to do object detection on videos in a folder with no displayed time 
detect.bat "D:\cam\" --start-time auto

# Detect movements of vehicles, people, dogs, ...
detect.bat "D:\cam\cam1.mp4"        # file
detect.bat "D:\cam\"                # folder

# Detect any movement, much more hits
--no-object-detection 

# If there is no clock in the video, this adds one. It auto-detects the time from the name of the file and starts an interactive cli to confirm all times by <Enter> or or you can set a time yourself
--start-time auto
--start-time "2026-06-05 23:59:59"

# Cut a clip from <start> to <end> time, creates <filename>_cut.xxx
cut.bat "D:\cam\cam1.mp4" 00:12:30 00:14:00

# Get a screenshot of the first frame; good for really large files
screenshot.bat "D:\cam\cam1.mp4"
```

Progress is saved next to the video (`<video file>.blip-named.json`/`.blip-movement.json`). After Ctrl+C, a crash or a power cut, run the **same command** again and it continues where it stopped. Changed settings start over; a finished video is skipped - delete its `.blip-*.json` to analyse it again.

Useful options:

```python
--sensitivity 20        # more sensitive (lower = more sensitive, default 32)
--ignore 0,0,400,40     # ignore an area, e.g. the camera clock (x,y,w,h in pixels)
--snapshot-every 10     # screenshot interval in seconds
--all-frames            # no keyframe pre-scan, decode every frame (much slower, old mode)
--help                  # all options
```

## Installation

If you only have the source code (no `py\` folder), do this once on a Windows PC with internet:

1. Double-click `setup.bat` on an online machine.
3. Copy the BLIP folder to the offline machine (e.g. via USB-Stick).

## Technical Details

Order: first a pre-scan decodes only the keyframes (one every ~1-2 s, the time between them is printed; warning above 4 s) and checks them for motion - only the time ranges with motion (plus 10 s before) are then analysed in detail, which is much faster on quiet footage (`--all-frames` skips the pre-scan). In detail the video is read at 5 frames per second, motion detection compares each frame against a background model that remembers the last ~60 s (so slow light changes are ignored), only frames with motion are passed to object detection, only detected objects that overlap the motion are kept (so parked cars are ignored), and consecutive hits are grouped into events with screenshots (with `--no-object-detection` the object detection step is skipped).

| Part | Uses | Interchangeable via |
|---|---|---|
| Runtime | Python 3.12 (Windows embeddable, in `py\`) | `setup.bat -PythonVersion 3.13.x` |
| Motion detection | OpenCV MOG2 background subtraction (CPU) | `--sensitivity`, `--min-area`, `--analyze-fps` |
| Object detection | YOLO11s (`models\yolo11s.pt`, COCO's 80 classes) via Ultralytics + PyTorch | `--model models\yolo11m.pt` (any Ultralytics `.pt`: larger/smaller YOLO, YOLOv8, custom-trained), `--classes person,car` |
| Hardware | NVIDIA GPU via CUDA, falls back to CPU automatically | `--device cpu` / `cuda:1`, `setup.bat -Cuda cu126` (older drivers) or `-Cpu` |
| Video decoding | OpenCV (built-in FFmpeg), falls back to `ffmpeg\ffmpeg.exe` | `--decoder ffmpeg`; replace `ffmpeg\` with any ffmpeg build |
| Cutting | ffmpeg (H.264 re-encode, or `--copy` for no re-encode) | without ffmpeg it uses OpenCV (no audio) |

- All logic is in `detect_movements.py`, `cut_video.py` and `screenshot.py`. The `.bat` files only start them with the bundled Python.
- Nothing is downloaded at runtime; the tool works fully offline.
- Ultralytics is licensed under AGPL-3.0. 
