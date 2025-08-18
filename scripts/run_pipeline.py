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

from pathlib import Path
import json
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(
        description="ShotTracker pipeline (skeleton). Example: python scripts/run_pipeline.py --video data/raw/game1.mp4 --out results/run1 --overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str, required=True, metavar="PATH",
                        help="Path to an input video like data/raw/game1.mp4")
    parser.add_argument("--out", type=str, required=True, metavar="PATH",
                        help="File or directory where artifacts go (e.g., data/processed/ or results/game1/)")
    parser.add_argument(
        "--overlay", action="store_true", help="Draw visual annotations(boxes,lines) onto frames.")
    parser.add_argument("--bootstrap-frames", type=int, default=30, metavar="N",
                        help="How many initial frames you'll use to \"warm up\" trackers/estimators. Validate that it's >= 1.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running the pipeline.")

    return parser


def detect(video_path, out_path, overlay, bootstrap_frames):
    print(f"[detect] opening {video_path}")
    print(f"[detect] warm-up frames: {bootstrap_frames}")
    print(f"[detect] overlays enabled: {overlay}")
    print(f"[detect] writing outputs to: {out_path}")

    # artifact write

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "video": str(video_path),
        "out": str(out_dir.resolve),
        "overlay": bool(overlay),
        "bootstrap_frames": int(bootstrap_frames),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 1,
    }

    stats_path = out_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[detect] wrote {stats_path}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # if someone passes --bootstrap-frames 0
    if args.bootstrap_frames < 1:
        # system exit
        raise SystemExit("[error] --bootstrap-frames must be >= 1")

    # Plan + dry-run before detect()
    print(
        f"Plan: video={args.video}, out={args.out}, overlay={args.overlay}, bootstrap_frames={args.bootstrap_frames}")

    if args.dry_run:
        print("[dry-run] Nothing executed.")
        return

    # Only run detect if not dry-run
    detect(args.video, args.out, args.overlay, args.bootstrap_frames)


if __name__ == "__main__":
    main()
