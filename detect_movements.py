#!/usr/bin/env python3
"""
detect_movements.py - find moving people / vehicles / animals in security camera videos.

Pre-scan (default, needs ffmpeg): only the keyframes are decoded (cameras store one every ~1-2 s) and checked
for motion; the steps below then run only on the time ranges with motion. --all-frames analyses the whole video.

Pipeline per analysed frame:
  1. Motion detection (OpenCV MOG2 background subtraction) - cheap, runs on every analysed frame.
  2. Object detection (YOLO, NVIDIA GPU if available, CPU otherwise) - only on frames with motion.
  3. An object only counts if its box overlaps actual motion (parked cars etc. are ignored).
  4. Consecutive detections are grouped into events; for each event the best frame is saved
     as a JPG with red boxes around the moving objects.

With --no-object-detection steps 2+3 are skipped: every moving part is marked with a red box,
a screenshot is saved when the motion starts and then every 5 seconds while it continues.

Output (next to the video; events_movement/ with --no-object-detection):
  events_named/<video>_events.txt
  events_named/<video>_E001_00h01m23s456.jpg   (timestamp = position in the video)
  <video file>.blip-named.json                 (saved progress: run the same command again to continue
                                                after Ctrl+C / a crash; when done, a rerun only says so)

Examples:
  python detect_movements.py cam1.mp4
  python detect_movements.py cam1.mp4 --classes person --conf 0.5
  python detect_movements.py cam1.mp4 --no-object-detection  (no AI model, any motion)
  python detect_movements.py cam1.mp4 --start-time "2026-09-30 14:00:00"
  python detect_movements.py D:\exports --start-time auto  (folder + subfolders, times from the file names)
  python detect_movements.py cam1.mp4 --all-frames  (no pre-scan, every frame, much slower)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before ultralytics is imported: never try to reach the internet.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_VERBOSE", "false")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "yolo11s.pt"
DEFAULT_CLASSES = "person,bicycle,car,motorcycle,bus,truck,cat,dog"
MOTION = "motion"  # label of plain motion boxes (--no-object-detection)
RED = (0, 0, 255)  # BGR
WHITE = (255, 255, 255)


# --------------------------------------------------------------------------- helpers

def fmt_ts(seconds: float) -> str:
    """00:01:23.456"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_ts_file(seconds: float) -> str:
    """00h01m23s456 - Windows does not allow ':' in file names."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_ffmpeg(name: str = "ffmpeg") -> str | None:
    """Prefer a bundled ffmpeg (./ffmpeg/ffmpeg.exe), then PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    for cand in (HERE / "ffmpeg" / exe, HERE / "ffmpeg" / "bin" / exe):
        if cand.is_file():
            return str(cand)
    return shutil.which(name)


def start_ffmpeg(cmd: list[str], stderr) -> subprocess.Popen:
    """ffmpeg in its own process group at below-normal priority: Ctrl+C only reaches Python (which then kills
    ffmpeg at once, instead of ffmpeg ending "normally" and the scan carrying on) and the PC stays usable."""
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, creationflags=flags)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, start_new_session=True)  # nice: inherited


def stop_ffmpeg(proc: subprocess.Popen) -> None:
    proc.kill()  # first kill: closing a pipe that a reader thread is blocked on would hang
    proc.wait()
    for pipe in (proc.stdout, proc.stderr):
        if pipe:
            pipe.close()


def read_frames(proc: subprocess.Popen, size: int):
    """Raw frames of `size` bytes from ffmpeg's stdout. The blocking pipe read runs in a thread and the main
    thread only waits in short steps, so Ctrl+C works immediately (a blocked read ignores it on Windows)."""
    frames: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                buf = proc.stdout.read(size)
            except (OSError, ValueError):
                buf = b""
            item = buf if len(buf) == size else None
            while not stop.is_set():
                try:
                    frames.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass
            if item is None:
                return

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            try:
                buf = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if buf is None:
                return
            yield buf
    finally:
        stop.set()


def lower_priority() -> None:
    """Below-normal CPU priority for this process (ffmpeg children get it too): the PC stays usable."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(max(0, 10 - os.nice(0)))  # workers inherit it: do not lower twice
    except Exception:
        pass


def disable_quick_edit() -> None:
    """Windows console 'QuickEdit': one click into the window starts a selection and freezes the program at
    its next output until Enter/Esc is pressed (and Ctrl+C then copies instead of aborting). Turn it off."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, (mode.value & ~0x0040) | 0x0080)  # -QUICK_EDIT, +EXTENDED_FLAGS
    except Exception:
        pass


def parse_ignore(values: list[str]) -> list[tuple[int, int, int, int]]:
    rects = []
    for v in values or []:
        try:
            x, y, w, h = (int(p) for p in v.split(","))
        except ValueError:
            sys.exit(f"--ignore expects x,y,w,h in pixels, got '{v}'")
        rects.append((x, y, w, h))
    return rects


# --------------------------------------------------------------------------- video input

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float  # seconds, 0 if unknown


class OpenCVSource:
    """Decodes with OpenCV (bundled FFmpeg). Skips frames with grab() to hit the analysis rate."""

    def __init__(self, path: Path, analyze_fps: float):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError("OpenCV could not open the video")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        if not (1 <= fps <= 240):
            fps = 25.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.info = VideoInfo(
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps,
            n / fps if n > 0 else 0.0,
        )
        self.step = max(1, round(fps / analyze_fps)) if analyze_fps > 0 else 1

    def frames(self, start: float = 0.0, duration: float | None = None):
        idx = 0
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            idx = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        end = start + duration if duration is not None else None
        while True:
            if idx % self.step == 0:
                ok, frame = self.cap.read()
                if not ok or (end is not None and idx / self.info.fps >= end):
                    break
                yield idx / self.info.fps, frame
            elif not self.cap.grab():
                break
            idx += 1
        self.cap.release()


class FFmpegSource:
    """Decodes with an external ffmpeg (optional). Handles odd codecs, non-ASCII paths, GPU decode."""

    def __init__(self, path: Path, analyze_fps: float, ffmpeg: str):
        self.path, self.ffmpeg = path, ffmpeg
        self.info = self._probe()
        self.out_fps = analyze_fps if analyze_fps > 0 else self.info.fps
        self.hwaccel = True  # off while YOLO uses the GPU, so they do not compete for it

    def _probe(self) -> VideoInfo:
        ffprobe = find_ffmpeg("ffprobe")
        if not ffprobe:
            raise IOError("ffprobe not found (needed next to ffmpeg)")
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
               "-of", "json", str(self.path)]
        data = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
        st = data["streams"][0]

        def rate(s: str) -> float:
            num, _, den = s.partition("/")
            return float(num) / float(den or 1) if float(den or 1) else 0.0

        fps = rate(st.get("avg_frame_rate", "0/1")) or rate(st.get("r_frame_rate", "0/1")) or 25.0
        return VideoInfo(int(st["width"]), int(st["height"]), fps,
                         float(data.get("format", {}).get("duration", 0) or 0))

    def frames(self, start: float = 0.0, duration: float | None = None):
        """All frames at the analysis rate, or only `duration` seconds from `start` (fast input seek)."""
        w, h = self.info.width, self.info.height
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error"] + (["-hwaccel", "auto"] if self.hwaccel else [])
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(self.path), "-an", "-sn"]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-vf", f"fps={self.out_fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = start_ffmpeg(cmd, subprocess.DEVNULL)
        try:
            for i, buf in enumerate(read_frames(proc, w * h * 3)):
                yield start + i / self.out_fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            stop_ffmpeg(proc)


# --------------------------------------------------------------------------- keyframe pre-scan
#
# Decoding every frame of 4K footage is the slow part, even when nothing moves. The pre-scan decodes only
# the keyframes (security cameras typically store one every 1-2 s), runs motion detection on them and
# returns the time ranges with motion. Only those ranges are then decoded and analysed at full rate.

PRESCAN_WARN_KEY_GAP = 4.0  # s; with rarer keyframes short events can slip between two keyframes
PRE_ROLL = 10.0            # s analysed before the motion so the background model can settle first
PTS_RE = re.compile(rb"pts_time:\s*(-?[\d.]+)")


def keyframe_gap(video: Path) -> float | None:
    """Typical seconds between keyframes, from the first minute (reads packet headers only, no decoding)."""
    ffprobe = find_ffmpeg("ffprobe")
    if not ffprobe:
        return None
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+60",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(video)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    times = []
    for line in out.decode(errors="ignore").splitlines():
        pts, _, flags = line.strip().partition(",")
        if flags.startswith("K"):
            try:
                times.append(float(pts))
            except ValueError:
                pass
    times.sort()
    if len(times) < 3:
        return None
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    return gaps[len(gaps) // 2]


def keyframes(video: Path, info: VideoInfo, ffmpeg: str, start: float = 0.0):
    """Decodes only the keyframes (from `start`), downscaled to the motion detector's size. Yields (t, frame)."""
    w = min(info.width, MotionDetector.WORK_WIDTH)
    h = max(2, round(info.height * w / info.width / 2) * 2)
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info", "-skip_frame", "nokey"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]  # timestamps then count from `start`
    cmd += ["-i", str(video), "-an", "-sn", "-vf", f"scale={w}:{h},showinfo",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = start_ffmpeg(cmd, subprocess.PIPE)
    pts: queue.Queue = queue.Queue()

    def read_pts():  # showinfo logs the real timestamp of every frame to stderr
        try:
            for line in proc.stderr:
                m = PTS_RE.search(line)
                if m and b"showinfo" in line:
                    pts.put(float(m.group(1)))
        except (OSError, ValueError):
            pass  # pipe closed on abort

    threading.Thread(target=read_pts, daemon=True).start()
    try:
        for buf in read_frames(proc, w * h * 3):
            deadline = time.time() + 60
            t = None
            while t is None and time.time() < deadline:  # short steps: Ctrl+C stays responsive
                try:
                    t = pts.get(timeout=0.2)
                except queue.Empty:
                    pass
            if t is None:
                break
            yield start + t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        stop_ffmpeg(proc)


def add_hit(spans: list[list[float]], t: float, link: float) -> None:
    """Record a motion keyframe. Hits less than `link` s apart form one span [first, last] (keeps state small)."""
    for span in reversed(spans[-3:]):  # hits arrive in time order, after a restart slightly overlapping
        if span[0] - link <= t <= span[1] + link:
            span[0], span[1] = min(span[0], t), max(span[1], t)
            return
    spans.append([t, t])
    spans.sort()


def merge_ranges(spans: list[list[float]], before: float, after: float, duration: float) -> list[list[float]]:
    """Motion spans -> sorted, non-overlapping [start, end] ranges to analyse in detail."""
    ranges: list[list[float]] = []
    for first, last in sorted(spans):
        a, b = max(0.0, first - before), last + after
        if duration:
            b = min(b, duration)
        if ranges and a <= ranges[-1][1] + PRE_ROLL:  # close together: one range, no second warm-up
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    return ranges


def open_source(path: Path, analyze_fps: float, decoder: str):
    ffmpeg = find_ffmpeg()
    if decoder == "ffmpeg":
        if not ffmpeg:
            sys.exit("--decoder ffmpeg requested but ffmpeg was not found (put it in ./ffmpeg/ or on PATH)")
        return FFmpegSource(path, analyze_fps, ffmpeg)
    try:
        src = OpenCVSource(path, analyze_fps)
        if src.info.width > 0:
            return src
        raise IOError("video has no readable frames")
    except IOError as e:
        if decoder == "auto" and ffmpeg:
            log(f"  OpenCV failed ({e}), falling back to ffmpeg")
            return FFmpegSource(path, analyze_fps, ffmpeg)
        raise


# --------------------------------------------------------------------------- motion

class MotionDetector:
    """Background subtraction on a downscaled grey image. Returns a motion mask + motion boxes."""

    WORK_WIDTH = 640

    def __init__(self, info: VideoInfo, min_area: float, sensitivity: int, ignore: list, analyze_fps: float):
        self.scale = min(1.0, self.WORK_WIDTH / info.width)
        self.size = (max(1, int(info.width * self.scale)), max(1, int(info.height * self.scale)))
        frame_area = self.size[0] * self.size[1]
        self.min_area_px = min_area * frame_area
        self.max_area_px = 0.6 * frame_area  # more than that = light switch / IR mode / camera shake
        # history ~ 60 s of analysed frames: slow changes (sun, shadows) become background
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(max(50, 60 * analyze_fps)), varThreshold=sensitivity, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ignore_mask = np.full(self.size[::-1], 255, np.uint8)
        for x, y, w, h in ignore:
            s = self.scale
            cv2.rectangle(self.ignore_mask, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), 0, -1)
        self.warmup = max(5, int(2 * analyze_fps))  # let the background model settle
        self.n = 0

    def apply(self, frame):
        self.n += 1
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)  # grainy footage
        fg = self.bg.apply(grey)
        if self.n <= self.warmup:
            return None, []
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # drop shadows (value 127)
        mask = cv2.bitwise_and(mask, self.ignore_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        if cv2.countNonZero(mask) > self.max_area_px:
            return None, []  # global change, not an object

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / self.scale
        boxes = []
        for c in contours:
            if cv2.contourArea(c) >= self.min_area_px:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)))
        return mask, boxes

    def motion_fraction(self, mask, box) -> float:
        """Share of a full-resolution box that is covered by motion."""
        s = self.scale
        x1, y1, x2, y2 = (int(v * s) for v in box)
        roi = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        return float(cv2.countNonZero(roi)) / roi.size if roi.size else 0.0


def merge_boxes(boxes: list, pad: int) -> list:
    """Merge boxes that touch or are closer than `pad` px, so one person is one rectangle, not five."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]:
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return [tuple(b) for b in boxes]


# --------------------------------------------------------------------------- objects

@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in full resolution

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


def cuda_supported(index: int) -> bool:
    """Does the installed torch build contain kernels for this GPU? (e.g. cu128 builds dropped Pascal)"""
    import torch
    major, minor = torch.cuda.get_device_capability(index)
    cc = major * 10 + minor
    for arch in torch.cuda.get_arch_list():  # e.g. sm_75, sm_86, compute_120
        kind, _, num = arch.partition("_")
        n = int(num) if num.isdigit() else 0
        if kind == "sm" and n // 10 == major and n <= cc:  # binary code runs on same major, >= minor
            return True
        if kind == "compute" and 0 < n <= cc:  # PTX is JIT-compiled for newer GPUs
            return True
    return False


def pick_device(requested: str) -> str:
    import warnings
    import torch
    if requested == "cpu":
        return "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch's long "not compatible" warnings; we log our own
        if torch.cuda.is_available():
            device = "cuda:0" if requested in ("auto", "cuda", "gpu") else requested
            index = int(device.partition(":")[2] or 0)
            if cuda_supported(index):
                return device
            major, minor = torch.cuda.get_device_capability(index)
            log(f"  WARNING: {torch.cuda.get_device_name(index)} (compute capability {major}.{minor}) is not "
                f"supported by this PyTorch build -> using CPU. "
                f"For older GPUs rebuild py\\ with: setup.bat -Cuda cu126")
            return "cpu"
    if requested not in ("auto",):
        log(f"  WARNING: device '{requested}' requested but CUDA is not available -> using CPU")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple silicon, handy for development
    return "cpu"


class ObjectDetector:
    def __init__(self, model_path: Path, device: str, conf: float, classes: list[str], imgsz: int,
                 enhance: bool = True):
        # GPU check first: on an unsupported GPU this says so before the (slow) model load
        self.device = pick_device(device)
        from ultralytics import YOLO

        # Local contrast boost: helps a lot on dark / washed-out black-and-white IR footage.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance else None

        self.model = YOLO(str(model_path))
        self.half = False
        if self.device.startswith("cuda"):
            import torch
            # fp16 is only fast from Volta (7.x) on; on consumer Pascal (6.1) it is many times slower than fp32
            self.half = torch.cuda.get_device_capability(self.device)[0] >= 7
        self.conf, self.imgsz = conf, imgsz
        names = self.model.names  # {id: name}
        wanted = {c.strip().lower() for c in classes if c.strip()}
        unknown = wanted - {n.lower() for n in names.values()}
        if unknown:
            sys.exit(f"Unknown class(es): {', '.join(sorted(unknown))}\nAvailable: {', '.join(names.values())}")
        self.class_ids = [i for i, n in names.items() if n.lower() in wanted]
        self.names = names
        if self.device.startswith("cuda"):
            import torch
            log(f"  GPU: {torch.cuda.get_device_name(self.device)}" + ("" if self.half else " (fp32)"))
            # claim the GPU now with a test run: errors show up at once (not after the pre-scan), and YOLO
            # holds the GPU before ffmpeg could take it (GPUs in "exclusive process" mode allow only one user)
            try:
                self._predict(np.zeros((imgsz, imgsz, 3), np.uint8))
            except Exception as e:
                self._to_cpu(e)

    def _predict(self, frame):
        return self.model.predict(frame, device=self.device, conf=self.conf, classes=self.class_ids,
                                  imgsz=self.imgsz, half=self.half, verbose=False)[0]

    def _to_cpu(self, error: Exception) -> None:
        first_line = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
        log(f"\n  WARNING: GPU failed ({first_line}) -> continuing on CPU (slower). Is another program "
            f"using the GPU? Check with: nvidia-smi")
        self.device, self.half = "cpu", False
        self.model.predictor = None  # set up again for the CPU

    def __call__(self, frame) -> list[Detection]:
        if self.clahe is not None:
            frame = cv2.cvtColor(self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
        try:
            r = self._predict(frame)
        except Exception as e:
            if not self.device.startswith("cuda"):
                raise
            self._to_cpu(e)
            r = self._predict(frame)
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        return [Detection(self.names[c], float(p), tuple(b)) for b, p, c in zip(xyxy, confs, cls)]


# --------------------------------------------------------------------------- drawing

def annotate(frame, dets: list[Detection], banner: str):
    img = frame.copy()
    h, w = img.shape[:2]
    thick = max(2, round(w / 640))
    font_scale = max(0.5, w / 1600)
    for d in dets:  # just the red box, no label text (the objects are listed in the report)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), RED, thick)
    # timestamp banner, top-left
    (tw, th), bl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 16, th + bl + 16), (0, 0, 0), -1)
    cv2.putText(img, banner, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thick, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- events

@dataclass
class Snapshot:
    t: float
    jpg: bytes
    score: float


@dataclass
class Event:
    start: float
    last: float
    hits: int = 0
    max_count: dict = field(default_factory=dict)  # label -> max seen at once
    best: Snapshot | None = None
    extra: list = field(default_factory=list)  # periodic snapshots
    last_extra_t: float = -1e9
    files: list = field(default_factory=list)


class EventRecorder:
    def __init__(self, video: Path, out_dir: Path, gap: float, min_hits: int, snapshot_every: float,
                 start_clock: dt.datetime | None, jpg_quality: int):
        self.video, self.out_dir = video, out_dir
        self.gap, self.min_hits, self.snapshot_every = gap, min_hits, snapshot_every
        self.start_clock, self.q = start_clock, jpg_quality
        self.active: Event | None = None
        self.events: list[Event] = []

    def clock(self, t: float) -> str:
        return (self.start_clock + dt.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if self.start_clock else ""

    def _snap(self, t, frame, dets, event_no) -> bytes:
        banner = f"{self.video.name}  {fmt_ts(t)}"
        if self.start_clock:
            banner += f"  ({self.clock(t)})"
        banner += f"  E{event_no:03d}"
        ok, buf = cv2.imencode(".jpg", annotate(frame, dets, banner), [cv2.IMWRITE_JPEG_QUALITY, self.q])
        return buf.tobytes()

    def update(self, t: float, frame, dets: list[Detection]):
        ev = self.active
        if ev and t - ev.last > self.gap:
            self._close()
            ev = None
        if not dets:
            return
        if ev is None:
            ev = self.active = Event(start=t, last=t)
        ev.last = t
        ev.hits += 1
        counts: dict[str, int] = {}
        for d in dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        for k, v in counts.items():
            ev.max_count[k] = max(ev.max_count.get(k, 0), v)

        no = len(self.events) + 1
        # best frame: big, confident objects (people weighted up) - good for identification
        score = sum(d.conf * (d.area ** 0.5) * (1.5 if d.label == "person" else 1.0) for d in dets)
        if ev.best is None or score > ev.best.score:
            ev.best = Snapshot(t, self._snap(t, frame, dets, no), score)
        if self.snapshot_every > 0 and t - ev.last_extra_t >= self.snapshot_every:
            ev.extra.append(Snapshot(t, self._snap(t, frame, dets, no), score))
            ev.last_extra_t = t

    def _close(self):
        ev, self.active = self.active, None
        if ev is None or ev.hits < self.min_hits:
            return  # too short -> flicker / false positive
        no = len(self.events) + 1
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # with periodic snapshots (start, +N s, +2N s ...) they already cover the event;
        # otherwise save the single best frame
        snaps = ev.extra if ev.extra else [ev.best]
        for s in snaps:
            name = f"{self.video.stem}_E{no:03d}_{fmt_ts_file(s.t)}.jpg"
            (self.out_dir / name).write_bytes(s.jpg)  # write_bytes: unicode-safe on Windows
            ev.files.append(name)
        ev.best = None
        ev.extra = []
        self.events.append(ev)
        objs = ", ".join(f"{k} x{v}" for k, v in sorted(ev.max_count.items()))
        log(f"\r  E{no:03d}  {fmt_ts(ev.start)} - {fmt_ts(ev.last)}  {objs}  "
            f"({len(ev.files)} screenshot{'s' if len(ev.files) > 1 else ''})" + " " * 20)

    def finish(self):
        self._close()


def write_report(path: Path, video: Path, info: VideoInfo, rec: EventRecorder, settings: str, elapsed: float):
    lines = [
        f"Video:     {video}",
        f"Analysed:  {dt.datetime.now():%Y-%m-%d %H:%M:%S}  (took {elapsed:.0f}s)",
        f"Video:     {info.width}x{info.height}, {info.fps:.2f} fps, duration {fmt_ts(info.duration)}",
        f"Settings:  {settings}",
        f"Events:    {len(rec.events)}",
        "",
    ]
    clock = rec.start_clock is not None
    head = f"{'#':<5} {'Start':<13} {'End':<13} {'Length':>8}  "
    if clock:
        head += f"{'Clock time':<20} "
    head += "Objects"
    lines += [head, "-" * len(head)]
    for i, ev in enumerate(rec.events, 1):
        row = f"E{i:03d}  {fmt_ts(ev.start):<13} {fmt_ts(ev.last):<13} {ev.last - ev.start:>7.1f}s  "
        if clock:
            row += f"{rec.clock(ev.start):<20} "
        row += ", ".join(f"{k} x{v}" for k, v in sorted(ev.max_count.items()))
        lines.append(row)
        for name in ev.files:
            lines.append(f"{'':5} -> {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main

class Progress:
    """One status line, updated at most once per second."""

    def __init__(self, label: str, total: float, base: float = 0.0):
        self.label, self.total, self.base = label, total, base  # base: already done before a restart
        self.t0 = self.last = time.time()

    def show(self, done: float, t: float, extra: str) -> None:
        now = time.time()
        if now - self.last < 1.0:
            return
        self.last = now
        speed = (done - self.base) / (now - self.t0)
        pct = f"{100 * done / self.total:5.1f}%" if self.total else ""
        eta = f"ETA {fmt_ts((self.total - done) / speed)[:8]}" if self.total and speed > 0 else ""
        print(f"\r  {self.label} {pct} {fmt_ts(t)}  {speed:5.1f}x realtime  {eta}  {extra}   ", end="", flush=True)


# --------------------------------------------------------------------------- saved progress

STATE_VERSION = 1
SAVE_EVERY = 15.0  # s between progress saves (plus after every event and on Ctrl+C)


class State:
    """Progress file next to the video, named after it (<video file>.blip-named.json / .blip-movement.json).
    Running the same command again continues where it stopped; other settings or a changed video start over."""

    def __init__(self, video: Path, mode: str, settings: dict):
        self.path = video.with_name(f"{video.name}.blip-{mode}.json")
        self.ident = {"version": STATE_VERSION, "video_size": video.stat().st_size,
                      "settings": json.loads(json.dumps(settings))}  # JSON-normalised for comparing
        self.data: dict = {}
        self.t0, self.elapsed_before = time.time(), 0.0
        self.last_save = time.time()
        self.broken = False

    def load(self) -> str:
        """Phase to continue with ('' = start from the beginning)."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ""
        except (OSError, ValueError):
            log(f"  {self.path.name} unreadable -> starting from the beginning")
            return ""
        if data.get("version") != STATE_VERSION or data.get("video_size") != self.ident["video_size"]:
            log(f"  {self.path.name} belongs to an older version of the video -> starting from the beginning")
            return ""
        if data.get("settings") != self.ident["settings"]:
            log(f"  {self.path.name} was saved with other settings -> starting from the beginning")
            return ""
        self.data = data
        self.elapsed_before = float(data.get("elapsed", 0.0))
        return data.get("phase", "")

    def elapsed(self) -> float:
        return self.elapsed_before + time.time() - self.t0

    def due(self) -> bool:
        return time.time() - self.last_save >= SAVE_EVERY

    def save(self, **changes) -> None:
        self.data.update(changes)
        self.last_save = time.time()
        if self.broken:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:  # write + rename: a crash mid-save never leaves a half-written state file
            tmp.write_text(json.dumps({**self.data, **self.ident, "elapsed": round(self.elapsed(), 1)}),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            self.broken = True
            log(f"\n  WARNING: cannot save progress to {self.path} ({e}) - a restart will begin from the start")


def run_settings(args, objects: bool, out_dir: Path) -> dict:
    """Everything that changes the result: saved progress is only reused with the same settings."""
    s = {"mode": "objects" if objects else "motion-only", "all_frames": args.all_frames,
         "analyze_fps": args.analyze_fps, "sensitivity": args.sensitivity, "min_area": args.min_area,
         "ignore": args.ignore or [], "gap": args.gap, "min_hits": args.min_hits,
         "snapshot_every": args.snapshot_every, "jpg_quality": args.jpg_quality,
         "start_time": str(args.start_time or ""), "out": str(out_dir.resolve())}
    if objects:
        s.update(model=Path(args.model).name, classes=args.classes, conf=args.conf, imgsz=args.imgsz,
                 enhance=not args.no_enhance, min_overlap=args.min_overlap)
    return s


def event_dicts(events: list) -> list[dict]:
    return [{"start": e.start, "last": e.last, "hits": e.hits, "max_count": e.max_count, "files": e.files}
            for e in events]


def output_dir(video: Path, args, mode: str) -> Path:
    # separate folders so a named run and a motion-only run can work on the same video in parallel
    return Path(args.out) if args.out else video.parent / f"events_{mode}"


# --------------------------------------------------------------------------- folder mode

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".ts", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".flv",
              ".webm", ".3gp"}
# start time in a file name, e.g. Milestone "06_05_2026 23_59_59 (UTC+02_00)" (day first) or
# "2026-05-06 23-59-59", "20260506_235959" (year first)
NAME_TIMES = [
    ("dmy", re.compile(r"(?<!\d)(\d{2})[_.-](\d{2})[_.-](\d{4})[ _T-]+(\d{2})[_.:-](\d{2})[_.:-](\d{2})(?!\d)")),
    ("ymd", re.compile(r"(?<!\d)(\d{4})[_.-]?(\d{2})[_.-]?(\d{2})[ _T-]*(\d{2})[_.:-]?(\d{2})[_.:-]?(\d{2})(?!\d)")),
]


def time_from_name(name: str) -> dt.datetime | None:
    for order, rx in NAME_TIMES:
        for m in rx.finditer(name):
            a, b, c, hh, mm, ss = (int(g) for g in m.groups())
            day, month, year = (a, b, c) if order == "dmy" else (c, b, a)
            if 2000 <= year <= 2100:
                try:
                    return dt.datetime(year, month, day, hh, mm, ss)
                except ValueError:
                    pass
    return None


def parse_time(text: str) -> dt.datetime:
    """A typed start time: 2026-05-07 00:00:00, 07.05.2026 00:00:00 or the file name style 07_05_2026 00_00_00."""
    text = text.strip().strip('"')
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        pass
    for f in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return dt.datetime.strptime(text, f)
        except ValueError:
            pass
    t = time_from_name(text)
    if t is None:
        raise ValueError(text)
    return t


def start_time_for(video: Path) -> tuple[dt.datetime | None, str]:
    """Automatic start time of a video + where it comes from."""
    t = time_from_name(video.name)
    return t, "from the file name" if t else "not found in the file name"


def find_videos(folder: Path) -> list[Path]:
    """All videos in the folder and its subfolders (not macOS '._' files, not BLIP's own output folders)."""
    found = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).parts
        if (p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._") and p.is_file()
                and not any(part.startswith("events_") for part in rel[:-1])):
            found.append(p)
    return sorted(found, key=lambda p: str(p).lower())


def probe_duration(video: Path) -> float:
    ffprobe = find_ffmpeg("ffprobe")
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]
        try:
            return float(subprocess.run(cmd, capture_output=True, timeout=60).stdout.strip() or 0)
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0.0
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps if fps > 0 else 0.0


def saved_state(video: Path, args, mode: str) -> dict:
    """Saved progress of this video if it was made with the same settings (start time aside), else {}."""
    path = video.with_name(f"{video.name}.blip-{mode}.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        size = video.stat().st_size
    except (OSError, ValueError):
        return {}
    want = json.loads(json.dumps(run_settings(args, mode == "named", output_dir(video, args, mode))))
    have = dict(data.get("settings", {}))
    want.pop("start_time"), have.pop("start_time", None)
    ok = data.get("version") == STATE_VERSION and data.get("video_size") == size and have == want
    return data if ok else {}


def fmt_hours(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def ask_start_times(jobs: list[dict]) -> None:
    """Show the start time chosen for each video; Enter keeps it, or type the right one ('-' = none)."""
    todo = [j for j in jobs if j["status"] != "already done"]
    log(f"\nCheck the start times ({len(todo)} video{'s' if len(todo) != 1 else ''}): Enter = OK, "
        f"or type the right one (e.g. 2026-05-07 00:00:00 or 07.05.2026 00:00:00), '-' = no clock time")
    for i, job in enumerate(jobs, 1):
        log(f"\n  [{i}/{len(jobs)}] {job['rel']}")
        if job["status"] == "already done":
            log("        already analysed -> skipped")
            continue
        shown = job["start"].strftime("%Y-%m-%d %H:%M:%S") if job["start"] else "none"
        while True:
            try:
                answer = input(f"        start time: {shown}  ({job['source']})  > ").strip()
            except EOFError:  # no keyboard (input redirected): keep the proposals
                answer = ""
            if not answer:
                break
            if answer == "-":
                job["start"] = None
                break
            try:
                job["start"] = parse_time(answer)
                log(f"        -> {job['start']:%Y-%m-%d %H:%M:%S}")
                break
            except ValueError:
                log("        not understood - e.g. 2026-05-07 00:00:00 or 07.05.2026 00:00:00 ('-' = none)")


def write_summary(path: Path, folder: Path, jobs: list[dict], mode: str, t0: float) -> None:
    lines = [f"BLIP summary: {folder}",
             f"Mode:    {'objects (events_named)' if mode == 'named' else 'any motion (events_movement)'}",
             f"Updated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}  (running {fmt_hours(time.time() - t0)})", "",
             f"{'#':>3}  {'Status':<13} {'Events':>6}  {'Length':>7}  {'Took':>7}  {'Start time':<19}  Video"]
    lines.append("-" * len(lines[-1]))
    for i, j in enumerate(jobs, 1):
        start = f"{j['start']:%Y-%m-%d %H:%M:%S}" if j["start"] else "-"
        events = str(j["events"]) if j["events"] is not None else ""
        took = fmt_hours(j["elapsed"]) if j["elapsed"] else ""
        lines.append(f"{i:>3}  {j['status']:<13} {events:>6}  {fmt_hours(j['duration']):>7}  {took:>7}  "
                     f"{start:<19}  {j['rel']}")
        if j.get("error"):
            lines.append(f"{'':5}error: {j['error']}")
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        log(f"  WARNING: cannot write {path} ({e})")


def overall_line(jobs: list[dict], t0: float) -> str:
    finished = [j for j in jobs if j["status"] in ("done", "failed")]
    work = [j for j in jobs if j["status"] != "already done"]
    done_len = sum(j["duration"] for j in finished)
    total_len = sum(j["duration"] for j in work)
    elapsed = time.time() - t0
    eta = (f"  ~{fmt_hours(elapsed / done_len * (total_len - done_len))} left"
           if done_len and total_len > done_len else "")
    return (f"videos {len(finished)}/{len(work)} done ({sum(j['status'] == 'failed' for j in jobs)} failed)  "
            f"events {sum(j['events'] or 0 for j in jobs)}  video {fmt_hours(done_len)} of {fmt_hours(total_len)}  "
            f"running {fmt_hours(elapsed)}{eta}")


# --------------------------------------------------------------------------- parallel folder mode
#
# Every video runs in its own process (this script with a single video), so a crash only stops that video.
# Their output goes to <video>.blip-<mode>.log; the window shows one live status line per running video.

def parallel_count(value: str, videos: int) -> int:
    cores = os.cpu_count() or 2
    if value.strip().lower() == "auto":
        n = max(1, min(8, cores // 2))
    else:
        try:
            n = int(value)
        except ValueError:
            sys.exit(f"--parallel expects a number or auto, got '{value}'")
    return max(1, min(n, videos))


def enable_ansi() -> bool:
    """Cursor movement for the live status lines (Windows 10+ console needs it switched on)."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        return bool(k32.GetConsoleMode(handle, ctypes.byref(mode))
                    and k32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


class Board:
    """Live status lines at the bottom of the window; messages scroll above them."""

    def __init__(self, ansi: bool):
        self.ansi, self.drawn, self.last = ansi, 0, 0.0

    def _clear(self) -> None:
        if self.ansi and self.drawn:
            sys.stdout.write(f"\x1b[{self.drawn}F\x1b[J")  # up to the first status line, clear to the end
            self.drawn = 0

    def message(self, text: str) -> None:
        self._clear()
        print(text, flush=True)

    def draw(self, lines: list[str], force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < (1.0 if self.ansi else 60.0):
            return  # without cursor movement only print the block once a minute
        self.last = now
        width = shutil.get_terminal_size((120, 25)).columns - 1
        lines = [line[:width] for line in lines]  # a wrapped line would break the redraw
        self._clear()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        if self.ansi:
            self.drawn = len(lines)


def worker_argv(args, video: Path, start: dt.datetime | None) -> list[str]:
    """Command line for analysing one video in its own process, with the same settings as this run."""
    a = [sys.executable, str(Path(__file__).resolve()), str(video),
         "--model", str(args.model), "--device", args.device, "--classes", args.classes, "--conf", str(args.conf),
         "--imgsz", str(args.imgsz), "--min-overlap", str(args.min_overlap), "--analyze-fps", str(args.analyze_fps),
         "--sensitivity", str(args.sensitivity), "--min-area", str(args.min_area), "--gap", str(args.gap),
         "--min-hits", str(args.min_hits), "--snapshot-every", str(args.snapshot_every),
         "--jpg-quality", str(args.jpg_quality), "--decoder", args.decoder]
    for region in args.ignore or []:
        a += ["--ignore", region]
    for flag, on in (("--no-object-detection", args.no_object_detection), ("--no-enhance", args.no_enhance),
                     ("--all-frames", args.all_frames)):
        if on:
            a.append(flag)
    if args.out:
        a += ["--out", args.out]
    if start:
        a += ["--start-time", f"{start:%Y-%m-%d %H:%M:%S}"]
    return a


class Worker:
    """One video analysed by its own process; its output goes to <video>.blip-<mode>.log."""

    def __init__(self, job: dict, argv: list[str], mode: str, threads: int):
        self.job, self.status, self.tail = job, "starting", []
        self.log_path = job["video"].with_name(f"{job['video'].name}.blip-{mode}.log")
        self.logf = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self.logf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {subprocess.list2cmdline(argv)}\n")
        # split the CPU between the workers (YOLO on CPU would otherwise start a thread per core in each)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        # own process group: Ctrl+C reaches only this window, which then stops the workers cleanly
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
        self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env, **kw)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        """Finished lines go to the log, the latest (progress) line becomes the status."""
        buf = ""
        while True:
            chunk = self.proc.stdout.read1(65536)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            while True:
                m = re.search(r"[\r\n]", buf)
                if not m:
                    break
                seg, delim, buf = buf[:m.start()], buf[m.start()], buf[m.end():]
                if seg.strip():
                    self.status = seg.strip()
                if delim == "\n":
                    self.logf.write(seg + "\n")
                    self.logf.flush()
                    if seg.strip():
                        self.tail = (self.tail + [seg.rstrip()])[-20:]
        if buf.strip():
            self.logf.write(buf + "\n")
            self.tail.append(buf.rstrip())

    def interrupt(self) -> None:
        """Like Ctrl+C in the worker: it saves its progress and stops."""
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        except OSError:
            pass

    def close(self) -> None:
        self.reader.join(timeout=10)
        self.logf.close()

    def line(self, total: int) -> str:
        name = self.job["video"].name
        name = name if len(name) <= 40 else name[:37] + "..."
        return f"  [{self.job['index']}/{total}] {name:<40} {self.status}"


def finish_job(w: Worker, mode: str) -> str:
    """Status of a finished worker from its exit code and saved progress; returns a message for the window."""
    job, rc = w.job, w.proc.returncode
    try:
        data = json.loads(job["video"].with_name(f"{job['video'].name}.blip-{mode}.json").read_text("utf-8"))
    except (OSError, ValueError):
        data = {}
    if rc == 0 and data.get("phase") == "done":
        job.update(status="done", events=len(data.get("events", [])), elapsed=float(data.get("elapsed", 0)))
        return f"  [{job['index']}] done: {job['rel']}  ({job['events']} events, {fmt_hours(job['elapsed'])})"
    if rc == 130:
        job["status"] = "stopped"
        return f"  [{job['index']}] stopped: {job['rel']}"
    last = next((t.strip() for t in reversed(w.tail) if t.strip()), f"exit code {rc}")
    job.update(status="failed", error=last)
    tail = "\n".join(f"        {t}" for t in w.tail[-6:])
    return f"  [{job['index']}] FAILED: {job['rel']}\n{tail}\n        full output: {w.log_path}"


def run_parallel(jobs: list[dict], args, mode: str, folder: Path, summary: Path, n: int, t0: float) -> None:
    todo = [j for j in jobs if j["status"] != "already done"]
    threads = max(1, (os.cpu_count() or 2) // n)
    for i, j in enumerate(jobs, 1):
        j["index"] = i
    board = Board(enable_ansi())
    board.message(f"\nRunning {n} videos at a time (each in its own process, {threads} CPU threads each for YOLO).\n"
                  f"Output of each video: <video>.blip-{mode}.log next to it.  Ctrl+C stops all, progress is saved.\n")
    pending, running = list(todo), []
    last_summary = 0.0
    try:
        while pending or running:
            while pending and len(running) < n:
                job = pending.pop(0)
                job["status"] = "running"
                running.append(Worker(job, worker_argv(args, job["video"], job["start"]), mode, threads))
            for w in [w for w in running if w.proc.poll() is not None]:
                running.remove(w)
                w.close()
                board.message(finish_job(w, mode))
                write_summary(summary, folder, jobs, mode, t0)
            if time.time() - last_summary > 60:
                last_summary = time.time()
                write_summary(summary, folder, jobs, mode, t0)
            board.draw([f"  {overall_line(jobs, t0)}"] + [w.line(len(jobs)) for w in running])
            time.sleep(0.5)
    except KeyboardInterrupt:
        board.message("\nStopping all videos, saving progress ...")
        for w in running:
            w.interrupt()
        deadline = time.time() + 30
        for w in running:
            try:
                w.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                w.proc.kill()
                w.proc.wait()
            w.close()
            w.job["status"] = "stopped"
        raise
    board.draw([f"  {overall_line(jobs, t0)}"], force=True)


# --------------------------------------------------------------------------- main

def prescan(video: Path, info: VideoInfo, args, ignore: list, ffmpeg: str, state: State) -> list | None:
    """Time ranges that need a detailed look, or None if the pre-scan cannot be used for this video."""
    gap = state.data.get("key_gap") or keyframe_gap(video)
    if not gap or gap <= 0:
        log("  pre-scan: could not read the keyframes -> full analysis")
        return None
    log(f"  pre-scan: keyframe every {gap:.2f}s")
    if gap > PRESCAN_WARN_KEY_GAP:
        log(f"  WARNING: keyframes only every {gap:.1f}s (more than {PRESCAN_WARN_KEY_GAP:.0f}s) - motion shorter "
            f"than that can be missed. Use --all-frames to analyse every frame (much slower).")
    before, after = gap + PRE_ROLL, gap + args.gap + 1
    link = before + after + PRE_ROLL  # hits closer than this end up in the same detail range anyway
    s = min(1.0, MotionDetector.WORK_WIDTH / info.width)
    small = VideoInfo(min(info.width, MotionDetector.WORK_WIDTH), max(2, round(info.height * s / 2) * 2),
                      1 / gap, info.duration)
    motion = MotionDetector(small, args.min_area, args.sensitivity,
                            [tuple(int(v * s) for v in r) for r in ignore], 1 / gap)

    spans: list[list[float]] = state.data.get("spans", [])
    pos = float(state.data.get("scan_pos", 0.0))
    # after a restart scan a few keyframes again: the new background model is blind while it settles
    start = max(0.0, pos - (motion.warmup + 1) * gap)
    if start == 0:
        add_hit(spans, motion.warmup * gap, link)  # blind at the very start -> look at it in detail
    else:
        log(f"  pre-scan: continuing at {fmt_ts(pos)}")
    n, t0 = 0, time.time()
    progress = Progress("pre-scan", info.duration, base=start)
    try:
        for t, frame in keyframes(video, info, ffmpeg, start):
            n += 1
            if motion.apply(frame)[1]:
                add_hit(spans, t, link)
            pos = max(pos, t)
            if state.due():
                state.save(phase="prescan", key_gap=gap, scan_pos=pos, spans=spans)
            progress.show(t, t, f"motion spots: {len(spans)}")
    finally:  # also on Ctrl+C
        state.save(phase="prescan", key_gap=gap, scan_pos=pos, spans=spans)
    if n == 0 and start == 0:
        log("\r  pre-scan: no keyframes decoded -> full analysis" + " " * 60)
        return None
    if info.duration and pos < info.duration - 2 * gap - 1:
        log(f"\r  WARNING: pre-scan stopped early at {fmt_ts(pos)} (damaged or unfinished file?) "
            f"-> the rest is analysed in detail" + " " * 20)
        spans.append([pos, info.duration])
    ranges = merge_ranges(spans, before, after, info.duration)
    total = sum(b - a for a, b in ranges)
    pct = f" ({100 * total / info.duration:.1f}% of the video)" if info.duration else ""
    log(f"\r  pre-scan: done in {time.time() - t0:.0f}s, motion in {len(ranges)} "
        f"range{'s' if len(ranges) != 1 else ''}, {fmt_ts(total)[:8]} to analyse in detail{pct}" + " " * 10)
    return ranges


def process_video(video: Path, args, detector) -> None:
    log(f"\n=== {video}")
    mode = "movement" if detector is None else "named"
    out_dir = output_dir(video, args, mode)
    state = State(video, mode, run_settings(args, detector is not None, out_dir))
    phase = state.load()
    if phase == "done":
        log(f"  already analysed with these settings -> {state.data.get('report', out_dir)}\n"
            f"  (delete {state.path.name} next to the video to analyse it again)")
        return {"status": "already done", "events": len(state.data.get("events", [])),
                "elapsed": state.elapsed_before, "report": state.data.get("report", "")}

    ffmpeg = find_ffmpeg()
    src = None
    if not args.all_frames and args.decoder != "opencv":
        if ffmpeg:
            try:
                src = FFmpegSource(video, args.analyze_fps, ffmpeg)
            except (IOError, OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError) as e:
                log(f"  pre-scan: ffprobe could not read the video ({e}) -> full analysis")
        else:
            log("  pre-scan: needs ffmpeg (./ffmpeg/ or PATH) -> full analysis")
    if src is None:
        src = open_source(video, args.analyze_fps, args.decoder)
    if isinstance(src, FFmpegSource) and detector is not None and detector.device.startswith("cuda"):
        src.hwaccel = False
    info = src.info
    log(f"  {info.width}x{info.height}, {info.fps:.2f} fps, {fmt_ts(info.duration)}  [{type(src).__name__}]")

    ignore = parse_ignore(args.ignore)
    rec = EventRecorder(video, out_dir, args.gap, args.min_hits, args.snapshot_every,
                        args.start_time, args.jpg_quality)
    rec.events = [Event(start=e["start"], last=e["last"], hits=e["hits"], max_count=e["max_count"],
                        files=e["files"]) for e in state.data.get("events", [])]
    if phase:
        log(f"  continuing from {state.path.name} ({len(rec.events)} events so far)")

    if phase == "detail":
        ranges = state.data.get("ranges")
    else:
        ranges = None
        if isinstance(src, FFmpegSource) and not args.all_frames:
            ranges = prescan(video, info, args, ignore, ffmpeg, state)
        state.save(phase="detail", ranges=ranges, detail_pos=0.0)
    if ranges is None and isinstance(src, FFmpegSource) and args.decoder == "auto":
        src = open_source(video, args.analyze_fps, args.decoder)  # same decoder as without pre-scan

    whole = ranges is None
    spans = [[0.0, info.duration or float("inf")]] if whole else ranges
    total = info.duration if whole else sum(b - a for a, b in spans)
    resume = float(state.data.get("detail_pos", 0.0))  # everything before it is analysed already
    if resume > 0:
        log(f"  detail: continuing at {fmt_ts(resume)}")

    def done_at(t: float) -> float:
        return sum(max(0.0, min(b, t) - a) for a, b in spans)

    def save_progress(t: float) -> None:
        # an event still running is redone after a restart -> continue at its start, not at t
        state.save(phase="detail", detail_pos=rec.active.start if rec.active else t,
                   events=event_dicts(rec.events))

    pad = max(8, info.width // 50)
    progress = Progress("" if whole else "detail", total, base=done_at(resume))
    n, saved_events, t = 0, len(rec.events), resume
    try:
        for i, (a, b) in enumerate(spans, 1):
            if b <= resume:
                continue
            start = a if a >= resume else max(a, resume - PRE_ROLL)  # pre-roll lets the background settle
            # fresh background model per range: the video jumps, the old background would be wrong
            motion = MotionDetector(info, args.min_area, args.sensitivity, ignore, args.analyze_fps)
            part = "" if whole else f"range {i}/{len(spans)}  "
            for t, frame in src.frames(start, None if whole else b - start):
                n += 1
                mask, mboxes = motion.apply(frame)
                if t < resume:
                    continue  # analysed before the restart, only feeds the background model
                dets: list[Detection] = []
                if mboxes:
                    if detector is None:
                        dets = [Detection(MOTION, 1.0, b) for b in merge_boxes(mboxes, pad)]
                    else:
                        dets = [d for d in detector(frame) if motion.motion_fraction(mask, d.box) >= args.min_overlap]
                rec.update(t, frame, dets)
                if len(rec.events) != saved_events or state.due():
                    saved_events = len(rec.events)
                    save_progress(t)
                progress.show(done_at(t), t, f"{part}events: {len(rec.events)}")
            if not whole:
                save_progress(b)
    except BaseException:
        save_progress(t)  # Ctrl+C or error: keep what is done
        raise
    rec.finish()

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{video.stem}_events.txt"
    settings = (f"mode={'motion-only' if detector is None else 'objects'}"
                + ("" if detector is None else f" model={Path(args.model).name} device={detector.device}"
                   f" classes={args.classes} conf={args.conf}")
                + f" analyze-fps={args.analyze_fps} sensitivity={args.sensitivity} min-area={args.min_area}"
                f" gap={args.gap}s snapshot-every={args.snapshot_every}s"
                + (" pre-scan=off" if whole else f" pre-scan={len(spans)} ranges/{fmt_ts(total)[:8]}"))
    write_report(report, video, info, rec, settings, state.elapsed())
    state.save(phase="done", report=str(report), events=event_dicts(rec.events))
    log(f"\r  done: {n} frames analysed in {state.elapsed():.0f}s, {len(rec.events)} events -> {report}" + " " * 10)
    return {"status": "done", "events": len(rec.events), "elapsed": state.elapsed(), "report": str(report)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Detect moving people/objects in security camera videos and save marked screenshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", help="video file, or a folder: all videos in it and its subfolders, one after another")
    ap.add_argument("--out", help="output folder (default: <video folder>/events_named, or events_movement with --no-object-detection)")

    g = ap.add_argument_group("object detection")
    g.add_argument("--no-object-detection", "--motion-only", dest="no_object_detection", action="store_true",
                   help="no AI model: mark any moving part with a red box, "
                        "screenshot at motion start and every --snapshot-every seconds")
    g.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO .pt model file")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    g.add_argument("--classes", default=DEFAULT_CLASSES, help="comma separated COCO class names")
    g.add_argument("--conf", type=float, default=0.30,
                   help="minimum detection confidence (low-quality footage scores lower)")
    g.add_argument("--no-enhance", action="store_true",
                   help="disable contrast enhancement (CLAHE) before detection")
    g.add_argument("--imgsz", type=int, default=640, help="inference size; 960/1280 finds small/far objects (slower)")
    g.add_argument("--min-overlap", type=float, default=0.10,
                   help="share of an object box that must be moving (0 = also report still objects)")

    g = ap.add_argument_group("motion / events")
    g.add_argument("--analyze-fps", type=float, default=5, help="frames per second to analyse")
    g.add_argument("--sensitivity", type=int, default=32, help="MOG2 varThreshold; lower = more sensitive")
    g.add_argument("--min-area", type=float, default=0.001, help="min motion blob size, fraction of frame")
    g.add_argument("--ignore", action="append", metavar="X,Y,W,H",
                   help="ignore a region (pixels), e.g. the camera clock overlay; repeatable")
    g.add_argument("--gap", type=float, default=3.0, help="seconds without detection that end an event")
    g.add_argument("--min-hits", type=int, default=2, help="analysed frames with detections needed for an event")

    g = ap.add_argument_group("output")
    g.add_argument("--snapshot-every", type=float, default=None,
                   help="screenshot at event start and every N seconds while it lasts "
                        "(default: 5 with --no-object-detection, otherwise 0 = only the best frame)")
    g.add_argument("--start-time",
                   help='wall-clock time of video start, e.g. "2026-09-30 14:00:00", or "auto": read it from the '
                        'file name (e.g. "06_05_2026 23_59_59 ...mkv"); with a folder each one is shown for checking')
    g.add_argument("--jpg-quality", type=int, default=90)
    g.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto",
                   help="auto = OpenCV, falls back to ffmpeg if available")
    g.add_argument("--parallel", default="auto",
                   help="folder mode: videos analysed at the same time, each in its own process "
                        "(auto = half the CPU threads, at most 8; 1 = one after another in this window)")
    g.add_argument("--all-frames", action="store_true",
                   help="old mode: decode and analyse the whole video instead of first scanning only the "
                        "keyframes for motion (much slower; the pre-scan needs ffmpeg)")
    args = ap.parse_args(argv)
    if hasattr(signal, "SIGBREAK"):  # Windows: the folder mode stops its worker processes with Ctrl+Break
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    disable_quick_edit()
    lower_priority()

    path = Path(args.video.strip('"'))
    if not path.exists():
        sys.exit(f"Not found: {path}")
    path = path.resolve()
    auto = (args.start_time or "").strip().lower() == "auto"
    if args.start_time and not auto:
        if path.is_dir():
            sys.exit("With a folder use --start-time auto (each video gets its own start time)")
        try:
            args.start_time = parse_time(args.start_time)
        except ValueError:
            sys.exit(f'--start-time: cannot read "{args.start_time}", e.g. "2026-05-07 00:00:00" or auto')

    # the mode decides the output folder and which saved progress belongs to this run
    objects = not args.no_object_detection and Path(args.model).is_file()
    mode = "named" if objects else "movement"
    if args.snapshot_every is None:
        args.snapshot_every = 0.0 if objects else 5.0

    jobs = []
    if path.is_dir():
        videos = find_videos(path)
        if not videos:
            sys.exit(f"No videos found in {path} (or its subfolders)")
        log(f"Found {len(videos)} video{'s' if len(videos) != 1 else ''} in {path}")
        for v in videos:
            saved = saved_state(v, args, mode)
            job = {"video": v, "rel": str(v.relative_to(path)), "duration": probe_duration(v),
                   "status": "already done" if saved.get("phase") == "done" else "waiting",
                   "events": None, "elapsed": 0.0, "start": None, "source": ""}
            if saved:  # same start time as last time, or the saved progress would not be used
                st = saved["settings"].get("start_time") or ""
                job["start"] = dt.datetime.fromisoformat(st) if st else None
                job["source"] = "from the saved progress"
            elif auto:
                job["start"], job["source"] = start_time_for(v)
            else:
                job["source"] = "no --start-time auto"
            if job["status"] == "already done":
                job["events"] = len(saved.get("events", []))
            jobs.append(job)
        if auto:
            ask_start_times(jobs)
    else:
        start, source = args.start_time, "--start-time"
        if auto:
            start, source = start_time_for(path)
            log(f"Start time: {f'{start:%Y-%m-%d %H:%M:%S}' if start else 'none'} ({source})")
        jobs.append({"video": path, "rel": path.name, "duration": 0.0, "status": "waiting", "events": None,
                     "elapsed": 0.0, "start": start, "source": source})

    folder = path.is_dir()
    summary = path / f"BLIP_summary_{mode}.txt" if folder else None
    t0 = time.time()
    todo = sum(j["status"] != "already done" for j in jobs)
    n = parallel_count(args.parallel, todo) if folder and todo else 1
    if n > 1:  # every video in its own process, they load the model themselves
        try:
            run_parallel(jobs, args, mode, path, summary, n, t0)
        except KeyboardInterrupt:
            log("\nAborted. Progress is saved - run the same command again to continue.")
            return 130
        finally:
            write_summary(summary, path, jobs, mode, t0)
        log(f"\n##### finished: {overall_line(jobs, t0)}\n      summary -> {summary}")
        return 0

    detector = None
    if not args.no_object_detection:
        if not Path(args.model).is_file():
            log(f"WARNING: model not found ({args.model}) -> no object detection, motion only")
        else:
            try:
                log(f"Loading model {Path(args.model).name} ...")
                detector = ObjectDetector(Path(args.model), args.device, args.conf,
                                          args.classes.split(","), args.imgsz, not args.no_enhance)
                log(f"  device: {detector.device}")
            except ImportError as e:
                log(f"WARNING: could not load ultralytics/torch ({e}) -> no object detection, motion only")

    try:
        for i, job in enumerate(jobs, 1):
            if job["status"] == "already done":
                continue
            if folder:
                log(f"\n##### [{i}/{len(jobs)}] {overall_line(jobs, t0)}")
                write_summary(summary, path, jobs, mode, t0)
            run_args = argparse.Namespace(**vars(args))
            run_args.start_time = job["start"]
            job["status"] = "running"
            try:
                result = process_video(job["video"], run_args, detector)
                job.update(status=result["status"], events=result["events"], elapsed=result["elapsed"])
            except KeyboardInterrupt:
                job["status"] = "stopped"
                raise
            except Exception as e:
                if not folder:
                    raise
                log(f"\n  ERROR with {job['rel']}:\n{traceback.format_exc()}  -> continuing with the next video")
                job.update(status="failed", error=f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        log("\nAborted. Progress is saved - run the same command again to continue.")
        return 130
    finally:
        if folder:
            write_summary(summary, path, jobs, mode, t0)
    if folder:
        log(f"\n##### finished: {overall_line(jobs, t0)}\n      summary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
