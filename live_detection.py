"""
Live blood / instrument-tip detection with RealSense depth.

Shows the 3D distance from the instrument tip to the nearest blood, and decides
"TIP IN BLOOD" using both the image AND the depth camera (so a tip hovering
above the blood is reported as "above", not "in").

Every run is logged automatically to logs/run_<date>_<time>.csv.

Also detects ArUco markers (outline + ID only, no pose yet). The dictionary is
auto-detected, or set it with --aruco-dict DICT_4X4_50. Disable with --no-aruco.

Keys:
  q - quit
  + / - - make the live window bigger / smaller (you can also drag its edge with the mouse)
  0 - reset the live window to its normal size
  f - save the current raw camera image to new_training_images/
      (use this when the model makes a mistake, then label + retrain)
"""

import argparse
import csv
import os
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="best_v7.pt", help="model file to use: a name inside weights/ (e.g. best_v7.pt) or a full path")
ap.add_argument("--no-log", action="store_true", help="do not save a CSV log for this run")
ap.add_argument("--snapshots", default=None, help="save an annotated frame every second to this folder")
ap.add_argument("--seconds", type=float, default=0, help="stop automatically after N seconds (0 = run until q)")
ap.add_argument("--aruco-dict", default="DICT_4X4_50", help="ArUco dictionary (default: DICT_4X4_50 = tray markers 0-3; pass auto to auto-detect)")
ap.add_argument("--scale", type=float, default=1.0, help="start size of the live window, e.g. 1.5 = 150%%")
ap.add_argument("--no-aruco", action="store_true", help="disable ArUco marker detection")
args = ap.parse_args()

DEVICE = 0 if torch.cuda.is_available() else "cpu"
PRECISION = 16 if DEVICE != "cpu" else None  # FP16 on GPU
print("Inference device:", torch.cuda.get_device_name(0) if DEVICE != "cpu" else "CPU")

_wdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
_wpath = args.weights if os.path.exists(args.weights) else os.path.join(_wdir, args.weights)
if not os.path.exists(_wpath):
    raise SystemExit(f"Weights not found: {args.weights}  (available: {sorted(f for f in os.listdir(_wdir) if f.endswith('.pt'))})")
print("Using weights:", _wpath)
model = YOLO(_wpath)
print("YOLO v5 loaded:", model.names)

# v5 model classes: 0 = blood, 1 = instrument (whole tool, no separate tip class).
# TIP is not predicted by the model; it is derived from the instrument mask below.
BLOOD, INSTRUMENT = 0, 1
TIP = -1               # internal id for the derived tip region
TIP_RADIUS_PX = 8      # size of the derived tip region at the instrument's far end

# ---------------- RealSense (colour + aligned depth) ----------------
WIDTH, HEIGHT, FPS = 640, 480, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)  # depth pixels -> colour pixels
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
FX, FY, CX, CY = intr.fx, intr.fy, intr.ppx, intr.ppy

print("RealSense camera opened (colour + depth)")

# ---------------- ArUco markers (detection only, works on OpenCV 4.6 and 4.7+) ----------------
aruco = cv2.aruco
ARUCO_DICTS = [n for n in [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
] if hasattr(aruco, n)]
ARUCO_SCAN_EVERY = 15   # while the dictionary is unknown, try all dictionaries every N frames
_aruco_params = aruco.DetectorParameters_create() if hasattr(aruco, "DetectorParameters_create") else aruco.DetectorParameters()
_aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
_aruco_detectors = {}


def aruco_detect(gray, name):
    """Returns (corners, ids or None) for one dictionary."""
    if hasattr(aruco, "ArucoDetector"):            # OpenCV >= 4.7
        if name not in _aruco_detectors:
            _aruco_detectors[name] = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, name)), _aruco_params)
        corners, ids, _ = _aruco_detectors[name].detectMarkers(gray)
    else:                                           # OpenCV 4.6
        corners, ids, _ = aruco.detectMarkers(gray, aruco.Dictionary_get(getattr(aruco, name)), parameters=_aruco_params)
    return corners, ids


def aruco_find_dict(gray):
    """Try every dictionary, return the one that finds the most markers (or None)."""
    best, best_n = None, 0
    for name in ARUCO_DICTS:
        _, ids = aruco_detect(gray, name)
        n = 0 if ids is None else len(ids)
        if n > best_n:              # strict '>' keeps the smaller dict on ties
            best, best_n = name, n
    return best, best_n


aruco_dict = None if args.aruco_dict == "auto" else args.aruco_dict
if aruco_dict and aruco_dict not in ARUCO_DICTS:
    raise SystemExit(f"Unknown ArUco dictionary {aruco_dict}. Options: {', '.join(ARUCO_DICTS)}")
aruco_on = not args.no_aruco
if aruco_on:
    print("ArUco detection:", aruco_dict or "auto-detecting dictionary")

# Colours: BGR
COLORS = {
    BLOOD: (0, 0, 255),        # Blood: RED
    TIP: (255, 255, 0),        # Instrument tip: CYAN
    INSTRUMENT: (0, 255, 0),   # Instrument body: GREEN
}

ALPHA = 0.45

SMOOTH = 0.9          # EMA factor: higher = steadier FPS readout
DIST_WINDOW = 9       # distance shown = median of the last N frames (removes per-frame noise)
DEADBAND_MM = 2.0     # shown distance only changes when the median moves at least this much
HOLD_FRAMES = 6       # keep the last reading this many frames if the tip/blood drops out briefly
MIN_DEPTH_M, MAX_DEPTH_M = 0.10, 2.0   # depth outside this range is treated as invalid
TOUCH_PX = 3          # tip mask grown by this many px when checking image overlap with blood
# Contact with hysteresis: enter "IN" below ENTER_MM, leave only above EXIT_MM
# (the camera sees the top of the tip, so a touching tip still reads a few mm above).
ENTER_MM = 12.0
EXIT_MM = 20.0
CONFIRM_FRAMES = 5    # a new state must hold this many frames before the display switches
LOCAL_BLOOD_PX = 20   # radius around the tip used to measure the blood surface depth
BLOOD_STEP = 2        # use every Nth blood pixel for the 3D search (speed)

TRAIN_DIR = "new_training_images"

loop_fps = None       # smoothed real FPS (camera + inference + drawing)
infer_ms = None       # smoothed inference time in ms
dist_mm_s = None      # smoothed tip->blood distance
last_time = time.perf_counter()
touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOUCH_PX + 1,) * 2)
local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * LOCAL_BLOOD_PX + 1,) * 2)

# ---------------- logging ----------------
t_start = time.time()
last_snap = 0.0
frame_idx = 0
log_f = log = None
if not args.no_log:
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", time.strftime("run_%Y%m%d_%H%M%S.csv"))
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["time_s", "frame", "tip_detected", "blood_detected", "tip_u", "tip_v",
                  "tip_depth_m", "blood_depth_at_tip_m", "height_above_blood_mm",
                  "in_blood", "raw_in_blood", "dist_mm", "dist_mm_smoothed", "status",
                  "cam_to_blood_mm", "cam_to_blood_nearest_mm", "cam_to_blood_smoothed_mm", "blood_depth_valid_pct",
                  "aruco_ids"])
    print("Logging to", log_path)
if args.snapshots:
    os.makedirs(args.snapshots, exist_ok=True)


def deproject(u, v, z):
    """Pixel (u, v) + depth z (m) -> 3D point (m) in the colour camera frame."""
    return np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], axis=-1)


def tip_point(tip_mask, depth_m):
    """Tip pixel (mask centroid) and its depth (median of valid depth in the mask)."""
    ys, xs = np.nonzero(tip_mask)
    if len(xs) == 0:
        return None, None
    u, v = int(xs.mean()), int(ys.mean())
    z = depth_m[ys, xs]
    z = z[z > 0]
    if z.size == 0:  # thin tip may have no depth -> look in a small window around it
        win = depth_m[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5]
        z = win[win > 0]
    return (u, v), (float(np.median(z)) if z.size else None)


def derive_tip_mask(inst_mask):
    """The model segments the whole instrument. Estimate the tip as the end of the
    instrument's long axis that is farther from the image border (the handle leaves
    the frame, the tip points into the scene). Returns a small mask around that end."""
    ys, xs = np.nonzero(inst_mask)
    if len(xs) < 20:
        return None
    h, w = inst_mask.shape
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    proj = (pts - mean) @ vt[0]
    ends = [pts[np.argmin(proj)], pts[np.argmax(proj)]]
    def border_dist(p):
        return min(p[0], p[1], w - 1 - p[0], h - 1 - p[1])
    end = max(ends, key=border_dist)
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - end[0]) ** 2 + (yy - end[1]) ** 2 <= TIP_RADIUS_PX ** 2
    tip = disk & inst_mask
    return tip if tip.any() else None


def blood_depth_near_tip(tip_mask, blood_mask, depth_m):
    """Depth of the blood surface right around the tip (median, m)."""
    near = (cv2.dilate(tip_mask.astype(np.uint8), local_kernel) > 0) & blood_mask & ~tip_mask
    z = depth_m[near]
    z = z[z > 0]
    return float(np.median(z)) if z.size >= 5 else None


def put(img, text, org, color, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


dist_hist = deque(maxlen=DIST_WINDOW)
cam_hist = deque(maxlen=DIST_WINDOW)   # camera -> blood readout smoothing
cam_blood_s = None


def smooth(val):
    """Rolling median + deadband so the number on screen doesn't jitter."""
    global dist_mm_s
    dist_hist.append(val)
    m = float(np.median(dist_hist))
    if dist_mm_s is None or abs(m - dist_mm_s) >= DEADBAND_MM:
        dist_mm_s = m
    return dist_mm_s


def reset_smooth():
    global dist_mm_s
    dist_hist.clear()
    dist_mm_s = None


# ---------------- extra windows: depth view + moving 64x64 foveal crop ----------------
WIN_MAIN = "YOLO V2 - Surgical Suction"
WIN_DEPTH = "Depth"
WIN_FOVEA = "Fovea 64x64 (colour | depth)"
FOVEA = 64            # crop size in pixels, centred on the tip
FOVEA_ZOOM = 6        # display enlargement (64 px -> 384 px)
FOVEA_FOLLOW = 0.5    # 0..1, how quickly the crop follows the tip (1 = instantly)
DEPTH_VIEW_RANGE = (0.15, 1.0)  # metres mapped to the colour scale in the depth window
fovea_c = None        # current crop centre (x, y), float


def crop_fixed(img, cx, cy, size=FOVEA):
    """size x size crop centred on (cx, cy); padded with black at the image border."""
    pad = size // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(round(cx)), int(round(cy))
    return padded[y0:y0 + size, x0:x0 + size]


def colorize_depth(d, lo, hi):
    """Depth (m) -> colour image; invalid depth (0) shown black."""
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[d <= 0] = 0
    return vis


def make_fovea_view(frame, depth_m, class_masks, center, tip_found):
    cx, cy = center
    z = FOVEA * FOVEA_ZOOM
    col = cv2.resize(crop_fixed(frame, cx, cy), (z, z), interpolation=cv2.INTER_NEAREST)
    # outline the detections inside the crop
    for cls in (BLOOD, INSTRUMENT, TIP):
        m = class_masks.get(cls)
        if m is None:
            continue
        mc = cv2.resize(crop_fixed(m.astype(np.uint8), cx, cy), (z, z), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(col, cnts, -1, COLORS[cls], 2)

    dcrop = crop_fixed(depth_m, cx, cy)
    valid = dcrop[dcrop > 0]
    if valid.size:
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi - lo < 0.01:  # at least 1 cm of colour range so noise isn't exaggerated
            mid = (hi + lo) / 2
            lo, hi = mid - 0.005, mid + 0.005
    else:
        lo, hi = 0, 1
    dep = cv2.resize(colorize_depth(dcrop, lo, hi), (z, z), interpolation=cv2.INTER_NEAREST)

    for img in (col, dep):  # crosshair at the crop centre (= tip)
        c = z // 2
        cv2.line(img, (c - 12, c), (c + 12, c), (255, 255, 255), 1)
        cv2.line(img, (c, c - 12), (c, c + 12), (255, 255, 255), 1)

    bar = np.zeros((44, 2 * z, 3), np.uint8)
    cz = depth_m[int(np.clip(cy, 0, HEIGHT - 1)), int(np.clip(cx, 0, WIDTH - 1))]
    txt = f"centre ({cx:.0f},{cy:.0f})  depth {cz * 1000:.0f} mm" if cz > 0 else f"centre ({cx:.0f},{cy:.0f})  depth n/a"
    txt += f"  | depth colours {lo * 1000:.0f}-{hi * 1000:.0f} mm" if valid.size else ""
    put(bar, txt, (8, 28), (255, 255, 255) if tip_found else (0, 165, 255), 0.55)
    if not tip_found:
        put(col, "TIP LOST - holding last position", (8, 24), (0, 165, 255), 0.55)
    return np.vstack([np.hstack([col, dep]), bar])


# Resizable windows: drag the edge with the mouse, or +/- keys for the live window.
# Only the display is scaled; recordings and logs keep the original 640x480.
WIN_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
SCALE_STEP, SCALE_MIN, SCALE_MAX = 1.25, 0.5, 4.0
win_scale = min(max(args.scale, SCALE_MIN), SCALE_MAX)
cv2.namedWindow(WIN_MAIN, WIN_FLAGS)
cv2.namedWindow(WIN_DEPTH, WIN_FLAGS)
cv2.namedWindow(WIN_FOVEA, WIN_FLAGS)
cv2.resizeWindow(WIN_MAIN, int(WIDTH * win_scale), int(HEIGHT * win_scale))
cv2.resizeWindow(WIN_DEPTH, WIDTH, HEIGHT)
cv2.resizeWindow(WIN_FOVEA, 2 * 64 * 6, 64 * 6 + 44)
cv2.moveWindow(WIN_MAIN, 0, 0)
cv2.moveWindow(WIN_DEPTH, int(WIDTH * win_scale) + 10, 0)
cv2.moveWindow(WIN_FOVEA, 0, HEIGHT + 60)

saved_msg_until = 0.0
lost_frames = 0
last_dist_text, last_dist_color = None, (255, 255, 255)
in_blood_state = False   # stable (displayed / logged) contact state
pending_frames = 0

# ---------------- video recording (one MP4 per run) ----------------
FOVEA_VIEW_SIZE = (2 * FOVEA * FOVEA_ZOOM, FOVEA * FOVEA_ZOOM + 44)  # (w, h)
video_path = video_writer = fovea_path = fovea_writer = raw_path = raw_writer = None
if not args.no_log:
    video_path = log_path.replace(".csv", ".mp4")
    fovea_path = log_path.replace(".csv", "_fovea.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    fovea_writer = cv2.VideoWriter(fovea_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, FOVEA_VIEW_SIZE)
    raw_path = log_path.replace(".csv", "_raw.mp4")   # clean camera image, no text/boxes (for training)
    raw_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if video_writer.isOpened() and fovea_writer.isOpened() and raw_writer.isOpened():
        print("Recording video to", video_path, fovea_path, "and", raw_path)
    else:
        print("WARNING: could not open video writer, no video will be saved")
        video_writer = fovea_writer = raw_writer = None


def finish_video(writer, path):
    """Release, fix timing to match the real run, convert to H.264."""
    writer.release()
    tmp = path.replace(".mp4", "_h264.mp4")
    real_s = run_end - t_start
    scale = real_s / (frame_idx / FPS) if frame_idx else 1.0
    ok = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                         "-vf", f"setpts={scale:.5f}*PTS", "-r", str(FPS), "-c:v", "libx264",
                         "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.DEVNULL).returncode == 0
    if ok:
        os.replace(tmp, path)
    print("Video saved:", path)


def combine_videos(main_path, fovea_path):
    """One video: main view with the fovea view small in the bottom-right corner.
    The two separate videos are deleted only if the combined one was made."""
    out = main_path.replace(".mp4", "_combined.mp4")
    pip = "[1:v]scale=256:-2:flags=neighbor,pad=iw+4:ih+4:2:2:white[pip];[0:v][pip]overlay=W-w-8:H-h-8"
    ok = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", main_path, "-i", fovea_path,
                         "-filter_complex", pip, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-pix_fmt", "yuv420p", out], stdin=subprocess.DEVNULL).returncode == 0
    if ok and os.path.exists(out) and os.path.getsize(out) > 0:
        os.remove(main_path)
        os.remove(fovea_path)
        print("Combined video saved:", out)
    else:
        print("WARNING: could not combine videos, keeping", main_path, "and", fovea_path)

MAX_CAMERA_RESTARTS = 5  # consecutive failed recoveries before giving up


def wait_for_frames_resilient():
    """Like pipeline.wait_for_frames(), but survives camera stalls (USB hiccups).

    A short stall is simply waited out; a longer one restarts the camera stream.
    Only gives up (raises) if the camera stays dead after several restarts.
    """
    for attempt in range(1, MAX_CAMERA_RESTARTS + 1):
        try:
            return pipeline.wait_for_frames(2000)
        except RuntimeError as e:
            print(f"WARNING: camera stalled ({e}), restarting stream ({attempt}/{MAX_CAMERA_RESTARTS})...")
        try:
            pipeline.stop()
        except RuntimeError:
            pass
        time.sleep(1.0)
        try:
            pipeline.start(config)
            print("Camera stream restarted")
        except RuntimeError as e:
            print("WARNING: camera restart failed:", e)
            time.sleep(2.0)
    raise RuntimeError("Camera stopped delivering frames and could not be restarted (check the USB cable/port)")


try:
    while True:
        frames = align.process(wait_for_frames_resilient())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        depth_m[(depth_m < MIN_DEPTH_M) | (depth_m > MAX_DEPTH_M)] = 0  # drop impossible depth values

        start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            device=DEVICE,
            quantize=PRECISION,
            retina_masks=True,   # full-resolution masks -> smooth edges
            verbose=False
        )

        infer = (time.perf_counter() - start) * 1000.0
        infer_ms = infer if infer_ms is None else SMOOTH * infer_ms + (1 - SMOOTH) * infer

        result = results[0]
        annotated = frame.copy()
        h, w = frame.shape[:2]
        class_masks = {}

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            # Merge all instances of the same class into one mask, so
            # overlapping detections don't stack and turn darker red.
            for cls in (INSTRUMENT, BLOOD):
                idx = np.where(classes == cls)[0]
                if len(idx) == 0:
                    continue
                if cls == INSTRUMENT:
                    idx = idx[[np.argmax(confs[idx])]]  # only the most confident instrument
                merged = masks[idx].max(axis=0)
                if merged.shape != (h, w):
                    merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = merged > 0.5
                class_masks[cls] = mask

                color = COLORS.get(cls, (255, 255, 255))
                annotated[mask] = (
                    (1 - ALPHA) * annotated[mask]
                    + ALPHA * np.array(color)
                ).astype(np.uint8)

        # derived tip: far end of the instrument mask (drawn on top in cyan)
        if class_masks.get(INSTRUMENT) is not None:
            tm = derive_tip_mask(class_masks[INSTRUMENT])
            if tm is not None:
                class_masks[TIP] = tm
                annotated[tm] = ((1 - ALPHA) * annotated[tm] + ALPHA * np.array(COLORS[TIP])).astype(np.uint8)

        # ---------------- ArUco markers (detection only) ----------------
        aruco_ids = []
        if aruco_on:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if aruco_dict is None and frame_idx % ARUCO_SCAN_EVERY == 0:
                found, n = aruco_find_dict(gray)
                if found:
                    aruco_dict = found
                    print(f"ArUco dictionary detected: {aruco_dict} ({n} markers in view)")
            if aruco_dict is not None:
                a_corners, a_ids = aruco_detect(gray, aruco_dict)
                if a_ids is not None:
                    aruco.drawDetectedMarkers(annotated, a_corners, a_ids)
                    aruco_ids = sorted(int(i) for i in a_ids.flatten())

        # ---------------- camera -> blood ----------------
        cam_blood_mm = cam_blood_near_mm = cam_valid_pct = None
        bm = class_masks.get(BLOOD)
        if bm is not None:
            bz_all = depth_m[bm]
            bz_ok = bz_all[bz_all > 0]
            cam_valid_pct = 100.0 * bz_ok.size / max(bz_all.size, 1)
            if bz_ok.size >= 10:
                cam_blood_mm = float(np.median(bz_ok)) * 1000          # typical distance to the blood
                cam_blood_near_mm = float(np.percentile(bz_ok, 5)) * 1000  # closest blood (robust min)
        if cam_blood_mm is not None:
            cam_hist.append(cam_blood_mm)
            m = float(np.median(cam_hist))
            if cam_blood_s is None or abs(m - cam_blood_s) >= DEADBAND_MM:
                cam_blood_s = m
        elif bm is None:
            cam_hist.clear()
            cam_blood_s = None

        # ---------------- tip -> blood ----------------
        dist_text, dist_color = None, (255, 255, 255)
        rec = {"tip_px": None, "tip_z": None, "blood_z": None, "height": None,
               "raw_in": 0, "dist_mm": None, "status": ""}
        tip_mask = class_masks.get(TIP)
        blood_mask = class_masks.get(BLOOD)
        raw_in = False       # this frame's own decision (before hysteresis/confirmation)
        tip_px = None

        if tip_mask is not None and blood_mask is not None:
            tip_px, tip_z = tip_point(tip_mask, depth_m)
            rec.update(tip_px=tip_px, tip_z=tip_z)
            overlaps = np.any((cv2.dilate(tip_mask.astype(np.uint8), touch_kernel) > 0) & blood_mask)

            if tip_px is not None and overlaps:
                # Tip is over the blood in the image -> use depth to tell "in" from "above"
                blood_z = blood_depth_near_tip(tip_mask, blood_mask, depth_m)
                rec["blood_z"] = blood_z
                if tip_z is not None and blood_z is not None:
                    height_mm = (blood_z - tip_z) * 1000  # + = tip closer to camera = above blood
                    rec["height"] = height_mm
                    raw_in = height_mm <= (EXIT_MM if in_blood_state else ENTER_MM)
                    rec.update(dist_mm=max(height_mm, 0.0), status="over_blood")
                    above_text = f"Tip ABOVE blood: {smooth(max(height_mm, 0.0)):.0f} mm"
                    dist_text, dist_color = above_text, (0, 165, 255)
                else:
                    # no depth to confirm -> image-only decision
                    raw_in = True
                    rec.update(dist_mm=0.0, status="over_blood_no_depth")
                    dist_text, dist_color = "Tip over blood (no depth)", (0, 165, 255)
                cv2.circle(annotated, tip_px, 8, (0, 165, 255), 2)

            elif tip_px is not None:
                bys, bxs = np.nonzero(blood_mask)
                bys, bxs = bys[::BLOOD_STEP], bxs[::BLOOD_STEP]
                bz = depth_m[bys, bxs]
                ok = bz > 0
                bys, bxs, bz = bys[ok], bxs[ok], bz[ok]

                if tip_z is not None and bz.size:
                    # true 3D distance to the nearest blood point
                    d = np.linalg.norm(deproject(bxs, bys, bz) - deproject(tip_px[0], tip_px[1], tip_z), axis=1)
                    k = int(np.argmin(d))
                    nx, ny = int(bxs[k]), int(bys[k])
                    dist_mm, label = float(d[k]) * 1000, "3D"
                else:
                    # no depth for the tip -> in-image distance scaled at blood depth
                    ally, allx = np.nonzero(blood_mask)
                    d_px = np.hypot(allx - tip_px[0], ally - tip_px[1])
                    k = int(np.argmin(d_px))
                    nx, ny = int(allx[k]), int(ally[k])
                    ref_z = float(np.median(bz)) if bz.size else None
                    dist_mm, label = (float(d_px[k]) * ref_z / FX * 1000 if ref_z else None), "2D"

                cv2.line(annotated, tip_px, (nx, ny), (0, 255, 255), 2)
                cv2.circle(annotated, (nx, ny), 4, (0, 255, 255), -1)

                rec.update(dist_mm=dist_mm, status=f"distance_{label}")
                if dist_mm is not None:
                    dist_text = f"Tip -> blood: {smooth(dist_mm):.0f} mm ({label})"
                    dist_color = (0, 255, 255)
                else:
                    dist_text = "Tip -> blood: no depth"
        else:
            rec["status"] = "no_tip" if tip_mask is None else "no_blood"
            lost_frames += 1
            if lost_frames <= HOLD_FRAMES and last_dist_text:
                dist_text, dist_color = last_dist_text, last_dist_color  # brief dropout: keep last reading
            else:
                reset_smooth()
                dist_text = "No tip detected" if tip_mask is None else "No blood detected"
        if rec["status"] not in ("no_tip", "no_blood"):
            lost_frames = 0
            last_dist_text, last_dist_color = dist_text, dist_color

        # ---------------- stable IN/OUT state (hysteresis + confirmation) ----------------
        rec["raw_in"] = int(raw_in)
        if raw_in != in_blood_state:
            pending_frames += 1
            if pending_frames >= CONFIRM_FRAMES:
                in_blood_state, pending_frames = raw_in, 0
        else:
            pending_frames = 0

        if in_blood_state:
            h_txt = f"  (depth diff {rec['height']:+.0f} mm)" if rec["height"] is not None else ""
            dist_text, dist_color = "TIP IN BLOOD" + h_txt, (0, 0, 255)
            if tip_px is not None:
                cv2.circle(annotated, tip_px, 10, (0, 0, 255), 3)

        # ---------------- overlay ----------------
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst = 1.0 / dt
            loop_fps = inst if loop_fps is None else SMOOTH * loop_fps + (1 - SMOOTH) * inst

        put(annotated, f"FPS: {loop_fps or 0:.1f}  |  Inference: {infer_ms:.0f} ms", (10, 30), (0, 255, 0))
        if dist_text:
            put(annotated, dist_text, (10, 62), dist_color, 0.8)
        if cam_blood_s is not None:
            put(annotated, f"Camera -> blood: {cam_blood_s:.0f} mm", (10, 94), (255, 200, 100), 0.7)
        elif class_masks.get(BLOOD) is not None:
            put(annotated, "Camera -> blood: no depth", (10, 94), (255, 200, 100), 0.7)
        if time.time() < saved_msg_until:
            put(annotated, "Image saved for training", (10, 126), (255, 0, 255), 0.6)
        if log:
            put(annotated, "REC", (WIDTH - 60, 30), (0, 0, 255), 0.6)
        if aruco_on:
            a_txt = (f"ArUco {len(aruco_ids)}: " + ",".join(map(str, aruco_ids))) if aruco_ids else \
                    ("ArUco: none" if aruco_dict else "ArUco: searching...")
            put(annotated, a_txt, (10, HEIGHT - 12), (0, 255, 0) if aruco_ids else (160, 160, 160), 0.6)

        # fovea follows the tip (smoothed); holds last position if the tip is lost
        if tip_px is not None:
            fovea_c = tip_px if fovea_c is None else (
                FOVEA_FOLLOW * tip_px[0] + (1 - FOVEA_FOLLOW) * fovea_c[0],
                FOVEA_FOLLOW * tip_px[1] + (1 - FOVEA_FOLLOW) * fovea_c[1])
        elif fovea_c is None:
            fovea_c = (WIDTH / 2, HEIGHT / 2)
        fx, fy = int(round(fovea_c[0])), int(round(fovea_c[1]))
        half = FOVEA // 2
        cv2.rectangle(annotated, (fx - half, fy - half), (fx + half, fy + half), (255, 0, 255), 1)

        depth_view = colorize_depth(depth_m, *DEPTH_VIEW_RANGE)
        cv2.rectangle(depth_view, (fx - half, fy - half), (fx + half, fy + half), (255, 255, 255), 2)
        if tip_px is not None:
            cv2.circle(depth_view, tip_px, 5, (255, 255, 255), -1)
        put(depth_view, f"Depth {DEPTH_VIEW_RANGE[0] * 100:.0f}-{DEPTH_VIEW_RANGE[1] * 100:.0f} cm (black = no depth)",
            (10, 30), (255, 255, 255), 0.6)

        fovea_view = make_fovea_view(frame, depth_m, class_masks, fovea_c, tip_px is not None)

        cv2.imshow(WIN_MAIN, annotated)
        cv2.imshow(WIN_DEPTH, depth_view)
        cv2.imshow(WIN_FOVEA, fovea_view)
        if video_writer:
            video_writer.write(annotated)
            fovea_writer.write(fovea_view)
            raw_writer.write(frame)

        # ---------------- log / snapshots ----------------
        frame_idx += 1
        elapsed_s = time.time() - t_start
        if log:
            tp = rec["tip_px"]
            fmt = lambda v, f: (f % v) if v is not None else ""
            log.writerow([f"{elapsed_s:.3f}", frame_idx, int(tip_mask is not None), int(blood_mask is not None),
                          tp[0] if tp else "", tp[1] if tp else "",
                          fmt(rec["tip_z"], "%.4f"), fmt(rec["blood_z"], "%.4f"), fmt(rec["height"], "%.1f"),
                          int(in_blood_state), rec["raw_in"], fmt(rec["dist_mm"], "%.1f"), fmt(dist_mm_s, "%.1f"), rec["status"],
                          fmt(cam_blood_mm, "%.1f"), fmt(cam_blood_near_mm, "%.1f"), fmt(cam_blood_s, "%.1f"),
                          fmt(cam_valid_pct, "%.0f"), " ".join(map(str, aruco_ids))])
        if args.snapshots and elapsed_s - last_snap >= 1.0:
            last_snap = elapsed_s
            cv2.imwrite(os.path.join(args.snapshots, f"snap_{int(elapsed_s):04d}.jpg"), annotated)
        if args.seconds and elapsed_s >= args.seconds:
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("+"), ord("=")) or key in (ord("-"), ord("_")) or key == ord("0"):
            if key == ord("0"):
                win_scale = 1.0
            elif key in (ord("+"), ord("=")):
                win_scale = min(win_scale * SCALE_STEP, SCALE_MAX)
            else:
                win_scale = max(win_scale / SCALE_STEP, SCALE_MIN)
            cv2.resizeWindow(WIN_MAIN, int(WIDTH * win_scale), int(HEIGHT * win_scale))
            print(f"Live window size: {win_scale * 100:.0f}%")
        if key == ord("q"):
            break
        if key == ord("f"):  # save the RAW image (no overlay) for retraining
            os.makedirs(TRAIN_DIR, exist_ok=True)
            p = os.path.join(TRAIN_DIR, time.strftime("img_%Y%m%d_%H%M%S_") + f"{frame_idx:06d}.jpg")
            cv2.imwrite(p, frame)
            print("Saved for training:", p)
            saved_msg_until = time.time() + 1.5

finally:
    run_end = time.time()
    try:
        pipeline.stop()
    except RuntimeError:
        pass  # already stopped (camera was lost)
    cv2.destroyAllWindows()
    if log_f:
        log_f.close()
        print("Log saved:", log_path)
    if video_writer:
        print("Converting videos, please wait (can take a few minutes for long runs, don't type or press Ctrl+C)...")
        finish_video(video_writer, video_path)
        finish_video(fovea_writer, fovea_path)
        finish_video(raw_writer, raw_path)
        combine_videos(video_path, fovea_path)
