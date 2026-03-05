import cv2 as cv

VIDEO_PATH = "data/raw/trimmedJumper.mp4"
TRAIL_LENGTH = 30


def detect_ball_center_stub(frame):
    """
    Rookie Stub:
    If we return None -> "no detection in this frame"
    I will replace this stub with YOLO or a simple shape detector
    """
    return None


def main():
    # Load the video
    capture = cv.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    centers = []

    # go through each frame
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        center = detect_ball_center_stub(frame)

        # if the center is found
        if center is not None:
            # save the center to the list
            centers.append(center)

        # draw trail (last TRAIL_LENGTH points)
        recent = centers[-TRAIL_LENGTH:]

        for (cx, cy) in recent:
            cv.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        cv.imshow("Rookie Tracker", frame)
        key = cv.waitKey(1) & 0xFF

        # if we press q quit the program
        if key == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
