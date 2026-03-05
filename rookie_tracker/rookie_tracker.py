import cv2 as cv
import numpy as np

VIDEO_PATH = "data/raw/trimmedJumper.mp4"
TRAIL_LENGTH = 30

# Basketball Color Ranges
LOW_ORANGE = (5, 100, 100)
HIGH_ORANGE = (25, 255, 255)

# Restrict the search for the ball to the hoop area/ avoid random oranges
ROI = None


def detect_ball_center_stub(frame, roi=None):
    """
    Rookie Stub:
    If we return None -> "no detection in this frame"
    1. crop to ROI (if parameter is passed)
    2. BGR -> HSV
    3. threshold orange range
    4. Clean mask
    5. Find contours
    6. Pick "best" contour by filters (area + round-ish)
    7. Return center (cv, cy) in full-frame coordinates or None
    """
    # frame size
    x1, y1, x2, y2 = (0, 0, frame.shape[1], frame.shape[0]) if roi is None else roi

    # crop to ROI
    crop = frame[y1:y2, x1:x2]

    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(LOW_ORANGE), np.array(HIGH_ORANGE))

    # Clean Noise
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # couldn't find good contours
    if not contours:
        return None

    # best contours
    best = None
    best_score = -1

    # go through the contours and find the best contoours
    for c in contours:
        area = cv.contourArea(c)
        if area < 200 or area > 20000:
            continue

        perimeter = cv.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)  # 1. 0 is a perfect circle

        if circularity < 0.2:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            best = c

    # couldn't get a best contour
    if best is None:
        return None

    M = cv.moments(best)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"] + y1)

    return (cx, cy)


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

        center = detect_ball_center_stub(frame, ROI)

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
