#!/usr/bin/env python3
"""
screenshot.py - save the first frame of a video as a PNG, e.g. to find --ignore coordinates.

  python screenshot.py cam1.mp4        -> cam1_first_frame.png (next to the video)

Opens the video the same way as detect_movements.py (OpenCV, falls back to ffmpeg).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# the embedded Python in py\ does not put the script folder on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_movements import open_source  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Save the first frame of a video as <name>_first_frame.png.")
    ap.add_argument("video", help="video file")
    ap.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto",
                    help="auto = OpenCV, falls back to ffmpeg if available")
    args = ap.parse_args(argv)

    video = Path(args.video.strip('"'))
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    video = video.resolve()

    src = open_source(video, 0, args.decoder)
    frame = next(iter(src.frames()), None)
    if frame is None:
        sys.exit(f"Could not read a frame from {video}")
    _, img = frame

    out = video.with_name(f"{video.stem}_first_frame.png")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        sys.exit("Could not encode the image")
    out.write_bytes(buf.tobytes())  # write_bytes: unicode-safe on Windows
    print(f"Saved: {out}  ({src.info.width}x{src.info.height} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
