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


def get_parser():
    parser = argparse.ArgumentParser(
        description="ShotTracker pipeline (skeleton)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str, required=True,
                        help="Path to an input video like data/raw/game1.mp4")
    parser.add_argument("--out", type=str, required=True,
                        help="file or directory where artifacts go (e.g., data/processed/ or results/game1/)")
    parser.add_argument(
        "--overlay", action="store_true", help="whether to draw visual annotations(boxes,lines) onto frames.")
    parser.add_argument("--bootstrap-frames", type=int, default=30,
                        help="how many initial frames you'll use to \"warm up\" trackers/estimators. Validate that it's >= 1.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # if someone passes --bootstrap-frames 0
    
    print(f"Running {args.video} on {args.out}")


if __name__ == "__main__":
    main()
