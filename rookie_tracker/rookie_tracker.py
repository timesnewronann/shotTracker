import cv2 as cv
import numpy as np


# CONSTANTS
VIDEO_PATH = "data/raw/trimmedJumper.mp4"
TRAIL_LENGTH = 30

# Basketball Color Ranges
LOW_ORANGE = (5, 100, 100)
HIGH_ORANGE = (25, 255, 255)

# Restrict the search for the ball to the hoop area/ avoid random oranges
ROI = (1536, 450, 3840, 1620)
SEARCH_MARGIN = 250

# Helper function to build a local ROI
"""
Starts with main ROI Boundaries
Shrink the boundaries around the previous ball center
Doesn't allow the search to go outside of the main ROI
"""


def make_search_roi(last_center, base_roi, frame_shape, margin):
    frame_height, frame_width = frame_shape[:2]
    base_x1, base_y1, base_x2, base_y2 = base_roi

    if last_center is None:
        return base_roi

    cx, cy = last_center

    x1 = max(base_x1, cx - margin)
    y1 = max(base_y1, cy - margin)
    x2 = min(base_x2, cx + margin)
    y2 = min(base_y2, cy + margin)

    return (x1, y1, x2, y2)


# Helper function to detect the ball center


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
    # Frame.shape -> (height, width)
    HEIGHT, WIDTH = frame.shape[:2]

    if roi is None:
        x1, y1, x2, y2 = 0, 0, WIDTH, HEIGHT
    else:
        x1, y1, x2, y2 = roi

    # crop to ROI
    crop = frame[y1:y2, x1:x2]

    # makes color detection easier -> detect in Hue Space which is more stable under lighting
    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(LOW_ORANGE), np.array(HIGH_ORANGE))

    # Clean Noise
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # couldn't find good contours
    if not contours:
        return None, None, mask

    # best contours
    best = None
    best_score = -1

    # go through the contours and find the best contoours
    for c in contours:
        area = cv.contourArea(c)
        if area < 200 or area > 20000:
            continue

        # ensure that the object is basketball shaped
        (x, y), radius = cv.minEnclosingCircle(c)
        if radius < 5 or radius > 80:
            continue

        # Simple score -> bigger and more compact circles
        # Try to reject reflections, clothes, and other object that aren't the ball
        score = area / (radius * radius + 1e-6)
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, None, mask

    (x, y), radius = cv.minEnclosingCircle(best)

    # Convert the coordinates back to full frame
    cx = int(x) + x1
    cy = int(y) + y1

    return (cx, cy), int(radius), mask


def main():
    # Load the video
    capture = cv.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    centers = []

    # used in helper function
    last_center = None

    # track misses
    misses = 0

    # used for toggling mask on and off
    show_mask = False

    # go through each frame
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        search_roi = make_search_roi(last_center, ROI, frame.shape, SEARCH_MARGIN)
        center, radius, mask = detect_ball_center_stub(frame, search_roi)

        # Draw the active search region for debugging purposes
        sx1, sy1, sx2, sy2 = search_roi
        cv.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

        accepted_center = None

        # if the center is found
        if center is not None:
            cx, cy = center

            # ceiling filter
            if cy >= 300:
                if len(centers) == 0:
                    accepted_center = center
                    # centers.append(center)
                else:
                    px, py = centers[-1]
                    if abs(cx - px) < 350 and abs(cy - py) < 350:
                        # draw the trail
                        # centers.append(center)
                        # last_center = center
                        accepted_center = center

            if accepted_center is not None:
                centers.append(accepted_center)
                centers = centers[-200:]
                last_center = accepted_center
                misses = 0

                cx, cy = accepted_center
                cv.circle(frame, (cx, cy), max(radius, 6), (0, 255, 0), 2)
                cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                print("detected", center, "r=", radius, "misses=", misses, "centers=", len(centers))
            else:
                misses += 1
                if misses > 10:
                    last_center = None
                    
            # centers = centers[-200:]

            # # successful center detection block
            # if center is not None and radius is not None:
            #     print("detected", center, "r=", radius, "misses=", misses, "centers=", len(centers))
            #     cx, cy = center
            #     cv.circle(frame, (cx, cy), max(radius, 6), (0, 255, 0), 2)
            #     cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            # else:
            #     misses += 1
            #     if misses > 10:
            #         last_center = None

        # draw trail (last TRAIL_LENGTH points)
        recent = centers[-TRAIL_LENGTH:]

        # Draw a shot arc line
        for i in range(1, len(recent)):
            x1, y1 = recent[i - 1]
            x2, y2 = recent[i]
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        for (cx, cy) in recent:
            cv.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)

        # Resized the 4k frame to 720 so that we can run it faster on my laptop
        display = cv.resize(frame, (1280, 720))
        cv.imshow("Rookie Tracker", display)

        if show_mask:
            mask_display = cv.resize(mask, (640, 360))
            cv.imshow("Mask", mask_display)
        key = cv.waitKey(1) & 0xFF

        # if we press q quit the program
        if key == ord("q"):
            break
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv.destroyWindow("Mask")

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
