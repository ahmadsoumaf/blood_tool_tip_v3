"""
Test the printed ArUco markers on the tray with the RealSense camera.

What it shows (live window):
  - every detected marker outlined, with its ID
  - which ArUco dictionary the printed markers belong to (auto-detected,
    or forced with --dict)
  - distance camera -> marker centre from the DEPTH camera
  - if --size is given (printed black-square side length in mm):
      * marker pose axes (X red, Y green, Z blue)
      * distance from the pose estimate (compare with depth = print-scale check)
      * measured edge length of each marker from depth (should match --size)
      * centre-to-centre distance between markers (measure with a ruler to check)
  - detection rate per marker ID over the run (printed at exit)

Usage:
  python test_aruco.py                       # auto-detect dictionary
  python test_aruco.py --size 40             # markers are 40 mm wide
  python test_aruco.py --dict DICT_4X4_50 --size 40

Keys:
  q - quit
  s - save current annotated frame to aruco_tests/
  r - re-scan for the dictionary (auto mode)
"""
import argparse
import itertools
import os
import time
from collections import Counter

import cv2
import numpy as np
import pyrealsense2 as rs

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default=None, help="ArUco dictionary name, e.g. DICT_4X4_50 (default: auto-detect)")
ap.add_argument("--size", type=float, default=0, help="printed marker side length in mm (black square, without white border)")
args = ap.parse_args()

WIDTH, HEIGHT, FPS = 640, 480, 30   # same as live_detection.py
MIN_DEPTH_M, MAX_DEPTH_M = 0.10, 2.0

# ---------------- ArUco helpers (works on OpenCV 4.6 and 4.7+) ----------------
aruco = cv2.aruco
DICT_NAMES = [n for n in [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
] if hasattr(aruco, n)]


def get_dict(name):
    did = getattr(aruco, name)
    return aruco.getPredefinedDictionary(did) if hasattr(aruco, "getPredefinedDictionary") else aruco.Dictionary_get(did)


def make_params():
    p = aruco.DetectorParameters_create() if hasattr(aruco, "DetectorParameters_create") else aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return p


PARAMS = make_params()
_detectors = {}


def detect(gray, name):
    """Returns (corners list, ids array or None, rejected)."""
    d = get_dict(name)
    if hasattr(aruco, "ArucoDetector"):            # OpenCV >= 4.7
        if name not in _detectors:
            _detectors[name] = aruco.ArucoDetector(d, PARAMS)
        return _detectors[name].detectMarkers(gray)
    return aruco.detectMarkers(gray, d, parameters=PARAMS)  # OpenCV 4.6


def find_dictionary(gray):
    """Try every dictionary, return the one that finds the most markers (or None)."""
    best, best_n = None, 0
    for name in DICT_NAMES:
        _, ids, _ = detect(gray, name)
        n = 0 if ids is None else len(ids)
        if n > best_n:              # strict '>' keeps the smaller dict on ties
            best, best_n = name, n
    return best, best_n


# ---------------- RealSense ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
DIST = np.array(intr.coeffs[:5], dtype=np.float64)
print("RealSense camera opened (colour + depth)")

print("Warming up camera...")
t0 = time.time()
while time.time() - t0 < 2.0:
    pipeline.wait_for_frames()


def depth_at(depth_m, u, v, r=3):
    """Median valid depth (m) in a small window around pixel (u, v), or None."""
    u, v = int(round(u)), int(round(v))
    patch = depth_m[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    vals = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
    return float(np.median(vals)) if vals.size else None


def deproject(u, v, z):
    return np.array([(u - intr.ppx) * z / intr.fx, (v - intr.ppy) * z / intr.fy, z])


# 3D corners of a marker in its own frame (same order as ArUco corners: TL, TR, BR, BL)
if args.size > 0:
    h = args.size / 2000.0  # mm -> m, half side
    OBJ = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dict_name = args.dict
if dict_name and dict_name not in DICT_NAMES:
    raise SystemExit(f"Unknown dictionary {dict_name}. Options: {', '.join(DICT_NAMES)}")

os.makedirs("aruco_tests", exist_ok=True)
WIN = "ArUco test (q = quit, s = save, r = rescan)"
cv2.namedWindow(WIN)
seen = Counter()
frames_total = 0
fps = None
last = time.time()

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            continue
        img = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = img.copy()

        if dict_name is None:
            found, n = find_dictionary(gray)
            if found:
                dict_name = found
                print(f"Dictionary detected: {dict_name} ({n} markers in view)")

        corners, ids, rejected = ([], None, []) if dict_name is None else detect(gray, dict_name)
        frames_total += 1
        centres = {}   # id -> 3D centre (m), from depth

        if ids is not None:
            aruco.drawDetectedMarkers(out, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                seen[int(mid)] += 1
                pts = c.reshape(4, 2)
                cu, cv_ = pts.mean(axis=0)
                lines = []

                z = depth_at(depth_m, cu, cv_)
                if z is not None:
                    centres[int(mid)] = deproject(cu, cv_, z)
                    lines.append(f"depth {z * 1000:.0f} mm")
                    # edge length from depth: average of the 4 edges in 3D
                    cz = [depth_at(depth_m, u, v) for u, v in pts]
                    if all(cz):
                        P = [deproject(u, v, zz) for (u, v), zz in zip(pts, cz)]
                        edge = np.mean([np.linalg.norm(P[i] - P[(i + 1) % 4]) for i in range(4)]) * 1000
                        lines.append(f"edge {edge:.1f} mm")

                if args.size > 0:
                    ok, rvec, tvec = cv2.solvePnP(OBJ, pts.astype(np.float64), K, DIST,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        cv2.drawFrameAxes(out, K, DIST, rvec, tvec, args.size / 2000.0)
                        pz = float(np.linalg.norm(tvec)) * 1000
                        lines.append(f"pose {pz:.0f} mm")

                side_px = np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)])
                lines.append(f"{side_px:.0f} px")

                x, y = int(pts[:, 0].max()) + 6, int(pts[:, 1].min()) + 12
                for i, t in enumerate(lines):
                    cv2.putText(out, t, (x, y + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(out, t, (x, y + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # centre-to-centre distances between markers (from depth)
            for a, b in itertools.combinations(sorted(centres), 2):
                d = np.linalg.norm(centres[a] - centres[b]) * 1000
                pa = corners[list(ids.flatten()).index(a)].reshape(4, 2).mean(axis=0).astype(int)
                pb = corners[list(ids.flatten()).index(b)].reshape(4, 2).mean(axis=0).astype(int)
                cv2.line(out, tuple(pa), tuple(pb), (255, 0, 255), 1)
                mid = tuple(((pa + pb) // 2).tolist())
                cv2.putText(out, f"{a}-{b}: {d:.0f} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # rejected candidates in grey: many of these = print/lighting/focus problem
        if rejected is not None and len(rejected):
            aruco.drawDetectedMarkers(out, rejected, borderColor=(120, 120, 120))

        now = time.time()
        inst = 1.0 / max(now - last, 1e-6)
        fps = inst if fps is None else 0.9 * fps + 0.1 * inst
        last = now
        n = 0 if ids is None else len(ids)
        status = f"{dict_name or 'searching for markers...'} | markers: {n} | rejected: {len(rejected) if rejected is not None else 0} | {fps:.0f} FPS"
        cv2.rectangle(out, (0, 0), (WIDTH, 24), (0, 0, 0), -1)
        cv2.putText(out, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if n:
            cv2.putText(out, "IDs: " + ", ".join(str(i) for i in sorted(ids.flatten())), (6, HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(WIN, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            p = os.path.join("aruco_tests", time.strftime("aruco_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(p, out)
            print("Saved", p)
        if key == ord("r") and not args.dict:
            dict_name = None
            seen.clear()
            frames_total = 0
            print("Re-scanning dictionary...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nDictionary: {dict_name}")
    if frames_total and seen:
        print(f"Detection rate over {frames_total} frames:")
        for mid in sorted(seen):
            print(f"  ID {mid:4d}: {100 * seen[mid] / frames_total:5.1f} %")
    else:
        print("No markers detected.")
