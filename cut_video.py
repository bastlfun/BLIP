#!/usr/bin/env python3
"""
cut_video.py - cut a part out of a video.

  python cut_video.py cam1.mp4 00:12:30 00:14:00       -> cam1_cut.mp4 (next to the video)
  python cut_video.py cam1.mp4 00:12:30 00:14:00 --copy   (instant, but starts at the nearest keyframe)

Times: HH:MM:SS, also HH:MM:SS.mmm, MM:SS or plain seconds.
Uses ffmpeg (./ffmpeg/ffmpeg.exe or PATH); without ffmpeg it falls back to OpenCV (no audio).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_time(s: str) -> float:
    try:
        parts = [float(p) for p in s.strip().split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid time '{s}', expected HH:MM:SS")
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"invalid time '{s}', expected HH:MM:SS")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def find_ffmpeg() -> str | None:
    exe = "ffmpeg" + (".exe" if os.name == "nt" else "")
    for cand in (HERE / "ffmpeg" / exe, HERE / "ffmpeg" / "bin" / exe):
        if cand.is_file():
            return str(cand)
    return shutil.which("ffmpeg")


def output_path(video: Path) -> Path:
    """cam1.mp4 -> cam1_cut.mp4, or cam1_cut2.mp4 ... if that already exists."""
    out = video.with_name(f"{video.stem}_cut{video.suffix}")
    n = 2
    while out.exists():
        out = video.with_name(f"{video.stem}_cut{n}{video.suffix}")
        n += 1
    return out


def cut_ffmpeg(ffmpeg: str, video: Path, out: Path, start: float, end: float, copy: bool) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats",
           "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}"]
    if copy:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"]
    cmd += ["-map", "0:v:0", "-map", "0:a?", str(out)]
    subprocess.run(cmd, check=True)


def cut_opencv(video: Path, out: Path, start: float, end: float) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(round((end - start) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cut a video from START to END and save it as <name>_cut.<ext>.")
    ap.add_argument("video", type=Path)
    ap.add_argument("start", type=parse_time, help="start time, HH:MM:SS")
    ap.add_argument("end", type=parse_time, help="end time, HH:MM:SS")
    ap.add_argument("--copy", action="store_true",
                    help="no re-encoding: instant and lossless, but the cut starts at the nearest keyframe")
    args = ap.parse_args(argv)

    video = args.video.resolve()
    if not video.is_file():
        sys.exit(f"Not found: {video}")
    if args.end <= args.start:
        sys.exit("End time must be after start time.")

    out = output_path(video)
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cut_ffmpeg(ffmpeg, video, out, args.start, args.end, args.copy)
    else:
        print("ffmpeg not found -> using OpenCV (no audio, lower quality)")
        out = out.with_suffix(".mp4")
        cut_opencv(video, out, args.start, args.end)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
