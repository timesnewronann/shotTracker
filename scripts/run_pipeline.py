# use argparse so we can run:
# python scripts/run_pipeline.py --video .. --out .. --overlay

"""
create ArgumentParser("ShotTracker pipeline (skeleton)")
add --video (required)
add --out (required)
add --overlay (optional but recommended)
add --bootstrap-frames (int, default 30)
return parsed args
"""
import argparse
import json
import math
import os
import sys
import cv2 as cv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# parses user intent (which video? where to put results? overlay or not? how often to sample?)

# ============= CLI ============
# defines CLI and validates user intent


def get_parser():
    parser = argparse.ArgumentParser(
        description="ShotTracker pipeline (MVP). Example: python scripts/run_pipeline.py --video data/raw/game1.mp4 --out results/run1 --overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str, required=True, metavar="PATH",
                        help="Path to an input video like data/raw/game1.mp4")
    parser.add_argument("--out", type=str, required=True, metavar="DIR",
                        help="Output directory where artifacts will be written (e.g., data/processed/ or results/game1/)")
    parser.add_argument("--overlay", action="store_true",
                        help="Draw visual annotations(boxes,lines) onto frames.")
    parser.add_argument("--save-video", action="store_true",
                        help="Save an overlay .mp4 to the output directory.")
    parser.add_argument("--overlay-video", action="store_true",
                        help="Write an annotated out/overlay.mp4.")
    parser.add_argument("--save-json", action="store_true", default=True,
                        help="Save run.json and shots.json")
    parser.add_argument("--bootstrap-frames", type=int, default=30, metavar="N",
                        help="How many initial frames you'll use to \"warm up\" trackers/estimators. Validate that it's >= 1.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path/name of YOLO model. If omitted, stub detector is user.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running the pipeline.")
    parser.add_argument("--max-frames", type=int, default=100, metavar="N",
                        help="Maximum number of frames to process after warm-up.")
    parser.add_argument("--frame-stride", type=int, default=1, metavar="N",
                        help="Process every Nth frame (1 = every frame).")
    parser.add_argument("--write-frames", action="store_true",
                        help="Write sampled frames as images to out/frames/.")
    parser.add_argument("--every-seconds", type=float, metavar="S",
                        help="Process roughly one frame every S seconds (overrides --frame-stride).")
    parser.add_argument("--max-seconds", type=float, metavar="S",
                        help="Stop after about S seconds of processed video (combined with --max-frames; the smaller cap wins).")

    return parser


# Game loop - opens the game film, warms up subsystems, picks how often to sample plays, iterates through frames, draws visuals, and writes a box score at the end
def detect(video_path, out_dir, overlay, bootstrap_frames, frame_stride, max_frames, every_seconds, max_seconds, dry_run):
    """
    Run a single pass over a video with a controllable sampling policy.

    Args:
        video_path (str|Path): Input video file.
        out_dir (str|Path): Directory for artifacts (stats.json, frames, overlays).
        overlay (bool): If True, draw visual annnotations on processed frames.
        bootstrap_frames (int): Number of initial frames consumed before stats are collected.
        frame_stride (int): Process every Nth frame (default policy)
        max_frames (int): Upper bound on the number of processed frames (post-warm-up).
        every_seconds (float|None): If set, overrides frame_stride using fps; process ~one frame
        max_seconds (float|None): If set, cap the total processed time to ~max_seconds
            The time-derived frame cap is combined with max_frames; the smaller cap wins

        Notes:
        - FPS is read from the container
        - Warm-up frames are not counted toward max_frames
        - This function is model-agnostic; plug detectors/trackers/pose in the sampling branch.
    """
    # logging the plan
    print(f"[detect] opening {video_path}")
    print(f"[detect] warm-up frames: {bootstrap_frames}")
    print(f"[detect] overlays enabled: {overlay}")
    print(f"[detect] writing outputs to: {out_dir}")

    # Real video I/O: open and validate - will add live webcam and iphone camera capture later
    # Game clock
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open video: {video_path}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        fps = 30.0  # safe fallback for weird files

    print(f"[meta] width={width} height={height} fps={fps:.2f}")

    # Compute stride from --every-seconds
    if every_seconds is not None:
        # Keep around 1 frame every S seconds => Stride is about fps * S
        frame_stride = max(1, int(round(fps * every_seconds)))
        print(f"[Sampling] every ~{every_seconds}s -> frame_stride={frame_stride}")

    # -- Enforce --max-seconds as a processed-frame cap --
    cap_by_time = None
    if max_seconds is not None:
        # Only "process" every frame_stride-th frame, convert seconds to a cap
        # processed_cap ~= (fps*seconds) / frame_stride
        cap_by_time = max(1, int(math.floor((fps * max_seconds) / max(1, frame_stride))))
        print(f"[Limit] max_seconds={max_seconds:.2f}s -> processed cap={cap_by_time}")

    effective_cap = min(max_frames, cap_by_time) if cap_by_time else max_frames
    print(f"[Limit] effective processed frame cap={effective_cap} (max_frames={max_frames})")

    # Need fps and the final stride to translate seconds-> "How many processed frames"

    # warm-up: consume frames so capture position advances realistically
    actual_warmup = 0
    for _ in range(bootstrap_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # placeholder for tracker initialization
        actual_warmup += 1
    print(f"[warmup] consumed {actual_warmup} frames")

    # strided, bounded processing loop
    # Keep only every Nth frame but still move the capture forward so time moves forward
    frames_processed = 0  # only counts the ones we keep
    frame_index = 0  # counts all frames we've read (after warm-up)

    stride = frame_stride
    # loop until we've processed enough frames
    # This will call models and accumlators later
    while frames_processed < max_frames:
        ret, frame = cap.read()  # try to grab next frame
        if not ret:
            break

        # only process every Nth frame
        if frame_index % frame_stride == 0:  # true when it's time to process this frame, enforces sampling policy
            # - computer vision
            # - run detectors
            # - trackers
            # - pose estimator
            # - shot/outcome classifier
            print(f"[process] frame {frame_index}")
            if overlay:
                # draw boxes, trails, angles, etc
                print(f"[overlay] drew annotations on frame {frame_index}")
            frames_processed += 1
        frame_index += 1

    print(f"[detect] total frames processed:", frames_processed)

    # artifact write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Telemetry Write
    stats = {
        "video": str(Path(video_path).resolve()),  # absolute path string
        "out": str(out_dir.resolve()),  # the out_dir
        "width": int(width),  # video width
        "height": int(height),  # video height
        "fps": float(fps),
        "total_frames_meta": int(cap.get(cv.CAP_PROP_FRAME_COUNT) or -1),
        "overlay": bool(overlay),
        "bootstrap_frames_requested": int(bootstrap_frames),
        "bootstrap_frames_read": int(actual_warmup),  # bootstrap_frames_read = actual_warmup
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames),
        "frames_processed": int(frames_processed),
        "dry_run": bool(dry_run),
        "every_seconds": float(every_seconds) if every_seconds is not None else None,
        "max_seconds": float(max_seconds) if max_seconds is not None else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }

    # Release resources
    cap.release()

    stats_path = out_dir / "stats.jsonl"
    # with stats_path.open("w", encoding="utf-8") as f:  # append mode
    #     json.dump(stats, f, indent=2)
    with stats_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")

    print(f"[detect] wrote {stats_path}")


# Validates args, supports --dry-run and calls detect()
def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.every_seconds is not None and args.every_seconds <= 0:
        raise SystemExit("[error] --every-seconds must be > 0")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise SystemExit("[error] --max-seconds must be > 0")

    # if someone passes --bootstrap-frames 0
    if args.bootstrap_frames < 1:
        # system exit
        raise SystemExit("[error] --bootstrap-frames must be >= 1")

    # Plan + dry-run before detect()
    print(
        f"Plan: video={args.video}, out={args.out}, overlay={args.overlay}, bootstrap_frames={args.bootstrap_frames}")

    if args.dry_run:
        print("[dry-run] Plan:")
        print(f"  video={args.video}")
        print(f"  out={args.out}")
        print(f"  overlay={args.overlay}")
        print(f"  bootstrap_frames={args.bootstrap_frames}")
        print(f"  frame_stride={args.frame_stride}")
        print(f"  max_frames={args.max_frames}")
        return

    # Only run detect if not dry-run
    detect(args.video, args.out, args.overlay, args.bootstrap_frames,
           args.frame_stride, args.max_frames, args.every_seconds, args.max_seconds, args.dry_run)


if __name__ == "__main__":
    main()
