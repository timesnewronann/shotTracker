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

# ============= HELPER FUNCTIONS ============


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in s.split(",")]
    return x1, y1, x2, y2


def detect_ball_centers_stub(frame) -> List[Tuple[int, int]]:
    """
    Very barebones find the "ball" circle finder using Hough. This is just to visualize motion.
    Replace with YOLO later.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=30,
                              param1=60, param2=30, minRadius=4, maxRadius=22)

    centers = []
    if circles is not None:
        for c in circles[0, :].astype(int):
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            centers.append((x, y))

    return centers


def within_roi(x: int, y: int, roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_rim_roi(img, roi):
    x1, y1, x2, y2 = roi
    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_path(img, pts: List[Tuple[int, int]], tail: int = 40):
    pts = pts[-tail:]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        cv.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

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
    # parser.add_argument("--overlay-video", action="store_true",
    #                     help="Write an annotated out/overlay.mp4.")
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
    parser.add_argument("--rim-roi", type=str, metavar="x1,y1,x2,y2",
                        help="Manual rim ROI if your detector doesn't output a rim class.")

    return parser

# ============= CORE FUNCTION ============
# Game loop - opens the game film, warms up subsystems, picks how often to sample plays, iterates through frames, draws visuals, and writes a box score at the end


def detect(video_path, out_dir, overlay, bootstrap_frames, frame_stride, max_frames, every_seconds, max_seconds, dry_run, save_video, write_frames, rim_roi_arg):
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
    # -- Logging the plan --
    print(f"[detect] opening {video_path}")
    print(f"[detect] warm-up frames: {bootstrap_frames}")
    print(f"[detect] overlays enabled: {overlay}")
    print(f"[detect] writing outputs to: {out_dir}")

    # -- Open Video --
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open video: {video_path}")

    # -- Get the width and height of the video and fps
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        fps = 30.0  # safe fallback for weird files
    print(f"[meta] width={width} height={height} fps={fps:.2f}")

    # -- Derive Stride from every-seconds (if provided) --
    if every_seconds is not None:
        # Keep around 1 frame every S seconds => Stride is about fps * S
        frame_stride = max(1, int(round(fps * every_seconds)))
        print(f"[Sampling] every ~{every_seconds}s -> frame_stride={frame_stride}")

    # -- Enforce --max-seconds as a processed-frame cap --
    cap_by_time = None
    if max_seconds is not None:
        cap_by_time = max(1, int(math.floor((fps * max_seconds) / max(1, frame_stride))))
        print(f"[Limit] max_seconds={max_seconds:.2f}s -> processed cap={cap_by_time}")

    effective_cap = min(max_frames, cap_by_time) if cap_by_time else max_frames
    print(f"[Limit] effective processed frame cap={effective_cap} (max_frames={max_frames})")

    # -- Outputs --
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    writer = None
    if overlay and (save_video if 'args' in globals() else True):
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        # Keep playback not-too-fast when striding
        out_fps = max(5.0, fps / max(1, frame_stride))
        writer = cv.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, out_fps, (width, height))

    frames_dir = out_dir / "frames"
    if write_frames:
        ensure_dir(frames_dir)  # set through CLI later

    # -- RIM ROI Bootstrap --
    if rim_roi_arg:
        rim_roi = parse_roi(rim_roi_arg)
        print(f"[ROI] Using manual rim ROI: {rim_roi}")
    else:
        # default to upper middle band
        pad = max(20, width // 10)
        rim_roi = (pad, 0, width - pad, height // 2)
        print(f"[ROI] Default rim ROI: {rim_roi}")

    # warm-up: consume frames so capture position advances realistically
    actual_warmup = 0
    for _ in range(bootstrap_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # placeholder for tracker initialization
        actual_warmup += 1
    print(f"[warmup] consumed {actual_warmup} frames")

    # -- Processing Loop --
    recent_centers: List[Tuple[int, int]] = []

    # strided, bounded processing loop
    # Keep only every Nth frame but still move the capture forward so time moves forward
    frames_processed = 0  # only counts the ones we keep
    frame_index = 0  # counts all frames we've read (after warm-up)

    # loop until we've processed enough frames
    # This will call models and accumlators later
    while frames_processed < effective_cap:
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
            # overlay hook
            # draw = frame

            # --- Ball stub detection + overlays ---
            centers = detect_ball_centers_stub(frame)
            # choose the smallest circle or take the first one
            ball_c = centers[0] if centers else None

            # Track path history
            # if "recent_centers" not in locals():
            #     recent_centers = []

            if overlay:
                # draw boxes, trails, angles, etc
                # print(f"[overlay] drew annotations on frame {frame_index}")
                draw = frame  # reuse original
                draw_rim_roi(draw, rim_roi)
                if ball_c:
                    cv.circle(draw, ball_c, 6, (0, 0, 255), -1)
                if recent_centers:
                    draw_path(draw, recent_centers)
                cv.putText(draw, f"f={frame_index}", (10, height - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # write after the overlays
                if writer is not None:
                    writer.write(draw)

            if write_frames:
                cv.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"), frame)

            print(f"[process] frame {frame_index}")
            frames_processed += 1

        frame_index += 1

    print(f"[detect] total frames processed:", frames_processed)

    # artifact write
    # out_dir = Path(out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)

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
        "effective_cap_processed_frames": int(effective_cap),
        "dry_run": bool(dry_run),
        "every_seconds": float(every_seconds) if every_seconds is not None else None,
        "max_seconds": float(max_seconds) if max_seconds is not None else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 1
    }

    stats_path = out_dir / "stats.jsonl"
    # with stats_path.open("w", encoding="utf-8") as f:  # append mode
    #     json.dump(stats, f, indent=2)
    with stats_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")

    print(f"[detect] wrote {stats_path}")

    # Release resources
    cap.release()
    if writer is not None:
        writer.release()


# ===== MAIN ======
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
        print(f"  save_video={args.save_video}")
        print(f"  write_frames={args.write_frames}")
        print(f"  rim_roi={args.rim_roi}")
        return

    # Only run detect if not dry-run
    detect(args.video, args.out, args.overlay, args.bootstrap_frames,
           args.frame_stride, args.max_frames, args.every_seconds, args.max_seconds, args.dry_run, args.save_video, args.write_frames, args.rim_roi)


if __name__ == "__main__":
    main()
