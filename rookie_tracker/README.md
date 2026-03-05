# Rookie Tracker

## This is a simpler version of the original shot tracker

## This is intended to be the most basic version of the shot tracking app

## Rookie Tracker's psuedocode

1. Load the video (hardcoded video path)
2. For each frame

- Try to find the ball center
- Save centers to a list
- Draw the last N centers as dots
- Show the frame

3. Quit on Q

## How to run the file

Use the command

```bash
python rookie_tracker/rookie_tracker.py
```

## Current Pipeline

Frame
|
ROI crop
|
HSV Threshold
|
mask cleanup
|
contour filtering
|
ball center + radius
