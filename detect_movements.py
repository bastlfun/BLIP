#!/usr/bin/env python3
"""
detect_movements.py - find moving people / vehicles / animals in security camera videos.

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

Examples:
  python detect_movements.py cam1.mp4
  python detect_movements.py cam1.mp4 --classes person --conf 0.5
  python detect_movements.py cam1.mp4 --no-object-detection  (no AI model, any motion)
  python detect_movements.py cam1.mp4 --start-time "2026-09-30 14:00:00"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
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
MOTION = "motion"  # label of plain motion boxes (drawn without a text label)
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

    def frames(self):
        idx = 0
        while True:
            if idx % self.step == 0:
                ok, frame = self.cap.read()
                if not ok:
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

    def frames(self):
        w, h = self.info.width, self.info.height
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "auto",
               "-i", str(self.path), "-an", "-sn",
               "-vf", f"fps={self.out_fps},scale={w}:{h}",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
        size, i = w * h * 3, 0
        try:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield i / self.out_fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                i += 1
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()


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


def pick_device(requested: str) -> str:
    import torch
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0" if requested in ("auto", "cuda", "gpu") else requested
    if requested not in ("auto",):
        log(f"  WARNING: device '{requested}' requested but CUDA is not available -> using CPU")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple silicon, handy for development
    return "cpu"


class ObjectDetector:
    def __init__(self, model_path: Path, device: str, conf: float, classes: list[str], imgsz: int,
                 enhance: bool = True):
        from ultralytics import YOLO

        # Local contrast boost: helps a lot on dark / washed-out black-and-white IR footage.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance else None

        self.model = YOLO(str(model_path))
        self.device = pick_device(device)
        self.half = self.device.startswith("cuda")
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
            log(f"  GPU: {torch.cuda.get_device_name(0)}")

    def __call__(self, frame) -> list[Detection]:
        if self.clahe is not None:
            frame = cv2.cvtColor(self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
        r = self.model.predict(frame, device=self.device, conf=self.conf, classes=self.class_ids,
                               imgsz=self.imgsz, half=self.half, verbose=False)[0]
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
    for d in dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), RED, thick)
        if d.label == MOTION:
            continue  # plain motion: just the red box
        text = f"{d.label} {d.conf:.0%}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = y1 - 4 if y1 - th - bl - 4 > 0 else y2 + th + 4
        cv2.rectangle(img, (x1, ty - th - bl), (x1 + tw + 6, ty + bl // 2), RED, -1)
        cv2.putText(img, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thick, cv2.LINE_AA)
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

def process_video(video: Path, args, detector) -> None:
    log(f"\n=== {video}")
    src = open_source(video, args.analyze_fps, args.decoder)
    info = src.info
    log(f"  {info.width}x{info.height}, {info.fps:.2f} fps, {fmt_ts(info.duration)}  [{type(src).__name__}]")

    # separate folders so a named run and a motion-only run can work on the same video in parallel
    out_dir = Path(args.out) if args.out else video.parent / ("events_movement" if detector is None else "events_named")
    motion = MotionDetector(info, args.min_area, args.sensitivity, parse_ignore(args.ignore), args.analyze_fps)
    rec = EventRecorder(video, out_dir, args.gap, args.min_hits, args.snapshot_every,
                        args.start_time, args.jpg_quality)

    pad = max(8, info.width // 50)
    t0 = last_print = time.time()
    n = 0
    for t, frame in src.frames():
        n += 1
        mask, mboxes = motion.apply(frame)
        dets: list[Detection] = []
        if mboxes:
            if detector is None:
                dets = [Detection(MOTION, 1.0, b) for b in merge_boxes(mboxes, pad)]
            else:
                dets = [d for d in detector(frame) if motion.motion_fraction(mask, d.box) >= args.min_overlap]
        rec.update(t, frame, dets)

        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            speed = t / (now - t0)
            pct = f"{100 * t / info.duration:5.1f}%" if info.duration else ""
            eta = f"ETA {fmt_ts((info.duration - t) / speed)[:8]}" if info.duration and speed > 0 else ""
            print(f"\r  {pct} {fmt_ts(t)}  {speed:5.1f}x realtime  {eta}  events: {len(rec.events)}   ",
                  end="", flush=True)
    rec.finish()
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{video.stem}_events.txt"
    settings = (f"mode={'motion-only' if detector is None else 'objects'}"
                + ("" if detector is None else f" model={Path(args.model).name} device={detector.device}"
                   f" classes={args.classes} conf={args.conf}")
                + f" analyze-fps={args.analyze_fps} sensitivity={args.sensitivity} min-area={args.min_area}"
                f" gap={args.gap}s snapshot-every={args.snapshot_every}s")
    write_report(report, video, info, rec, settings, elapsed)
    log(f"\r  done: {n} frames analysed in {elapsed:.0f}s, {len(rec.events)} events -> {report}" + " " * 10)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Detect moving people/objects in security camera videos and save marked screenshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", help="video file (any format ffmpeg/OpenCV can read)")
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
    g.add_argument("--start-time", type=lambda s: dt.datetime.fromisoformat(s),
                   help='wall-clock time of video start, e.g. "2026-09-30 14:00:00"')
    g.add_argument("--jpg-quality", type=int, default=90)
    g.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto",
                   help="auto = OpenCV, falls back to ffmpeg if available")
    args = ap.parse_args(argv)

    video = Path(args.video.strip('"'))
    if video.is_dir():
        sys.exit(f"Please give a video file, not a folder: {video}")
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    video = video.resolve()

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
    if args.snapshot_every is None:
        args.snapshot_every = 5.0 if detector is None else 0.0

    try:
        process_video(video, args, detector)
    except KeyboardInterrupt:
        log("\nAborted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
