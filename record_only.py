"""
Record only: clean RealSense colour video, no model, no drawings.
Use it to collect footage for training images.

Saves to recordings/rec_<date>_<time>.mp4

Keys (in the preview window):
  q - stop and save
  space - pause / resume recording (preview keeps running)

The preview window shows a red REC dot, but the saved video is clean.
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np
import pyrealsense2 as rs

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=0, help="stop automatically after N seconds (0 = run until q)")
args = ap.parse_args()

WIDTH, HEIGHT, FPS = 640, 480, 30   # same as live_detection.py

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
pipeline.start(config)
print("RealSense camera opened (colour)")

# let auto exposure / white balance settle (first frames come out blue-tinted)
WARMUP_S = 2.0
print(f"Warming up camera for {WARMUP_S:.0f} s...")
t0 = time.time()
while time.time() - t0 < WARMUP_S:
    pipeline.wait_for_frames()

os.makedirs("recordings", exist_ok=True)
path = os.path.join("recordings", time.strftime("rec_%Y%m%d_%H%M%S.mp4"))
writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
if not writer.isOpened():
    raise SystemExit("Could not open video writer for " + path)
print("Recording to", path, "- press q to stop, space to pause")

WIN = "Record only (q = stop, space = pause)"
cv2.namedWindow(WIN)
paused = False
frames_written = 0
rec_time = 0.0          # seconds actually recorded (pauses excluded)
last = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        frame = np.asanyarray(color.get_data())

        now = time.time()
        if not paused:
            writer.write(frame)          # clean frame, nothing drawn on it
            frames_written += 1
            rec_time += now - last
        last = now

        preview = frame.copy()           # drawings only on the preview
        if paused:
            cv2.putText(preview, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            cv2.circle(preview, (WIDTH - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(preview, f"{rec_time:.0f} s", (WIDTH - 90, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WIN, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            print("Paused" if paused else "Recording")
        if args.seconds and rec_time >= args.seconds:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    writer.release()

    # fix timing so the video plays at real speed, and convert to H.264
    if frames_written:
        scale = rec_time / (frames_written / FPS)
        tmp = path.replace(".mp4", "_h264.mp4")
        ok = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                             "-vf", f"setpts={scale:.5f}*PTS", "-r", str(FPS), "-c:v", "libx264",
                             "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", tmp]).returncode == 0
        if ok:
            os.replace(tmp, path)
        print(f"Video saved: {path}  ({rec_time:.0f} s)")
    else:
        os.remove(path)
        print("Nothing recorded")
