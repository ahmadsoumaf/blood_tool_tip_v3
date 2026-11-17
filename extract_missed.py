"""
Pull the frames where live_detection.py did NOT detect the instrument out of the
raw run videos, so they can be labelled and used for retraining.

For every run it reads logs/<run>.csv (tip_detected == 0) and grabs the matching
frames from logs/<run>_raw.mp4 (clean camera image, no overlays).

Output (a NEW folder, nothing existing is touched):
  <out>/images/      raw frames (for labelling / training)
  <out>/labels/      draft YOLO-seg labels from the current model at LOW confidence
                     (correct them in your labelling tool instead of drawing from scratch;
                      empty file = model saw nothing even at low confidence)
  <out>/preview/     same frames with the draft masks drawn, for a quick look
  <out>/summary.csv  one row per frame: run, frame, time, best instrument conf

Usage:
  python extract_missed.py run_20261113_151652 run_20261113_153439
  python extract_missed.py --last 2            # the two most recent runs
Options:
  --every N   keep every Nth missed frame inside a gap (default 3, avoids near-duplicates)
  --conf C    confidence for draft instrument labels (default 0.05)
  --blood-conf C  confidence for draft blood labels (default 0.25, same as live)
"""
import argparse
import csv
import glob
import os
import subprocess

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="run names, e.g. run_20261113_151652")
ap.add_argument("--last", type=int, default=0, help="use the N most recent runs in logs/")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--conf", type=float, default=0.05, help="min confidence for draft INSTRUMENT labels")
ap.add_argument("--blood-conf", type=float, default=0.25, help="min confidence for draft BLOOD labels (same as live)")
ap.add_argument("--out", default="missed_instrument_20261113")
args = ap.parse_args()

runs = list(args.runs)
if args.last:
    all_runs = sorted(os.path.basename(p)[:-4] for p in glob.glob("logs/run_*.csv")
                      if os.path.exists(p[:-4] + "_raw.mp4"))
    runs += all_runs[-args.last:]
if not runs:
    raise SystemExit("Give run names or --last N")

INSTRUMENT = 1
COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}
for sub in ("images", "labels", "preview"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("./weights/best.pt")


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", path], capture_output=True, text=True).stdout)


summary_path = os.path.join(args.out, "summary.csv")
new_file = not os.path.exists(summary_path)
summary_f = open(summary_path, "a", newline="")
summary = csv.writer(summary_f)
if new_file:
    summary.writerow(["image", "run", "frame", "time_s", "best_instrument_conf", "n_blood"])

for run in runs:
    rows = list(csv.DictReader(open(f"logs/{run}.csv")))
    n = len(rows)
    missed = [int(r["frame"]) for r in rows if r["tip_detected"] == "0"]

    # split into gaps of consecutive frames, keep every Nth frame of each gap (at least one)
    keep, gap = [], []
    for f in missed + [None]:
        if gap and (f is None or f != gap[-1] + 1):
            keep += gap[::args.every]
            gap = []
        if f is not None:
            gap.append(f)

    # The raw video was re-timed to real time at 30 fps (frames duplicated), so CSV frame k
    # (1-based) sits at time (k-1) * video_duration / n_csv_frames.
    raw = f"logs/{run}_raw.mp4"
    dur = duration(raw)
    cap = cv2.VideoCapture(raw)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    targets = {int(round((k - 1) * dur / n * vfps)): k for k in keep}
    print(f"{run}: {n} frames, {len(missed)} missed ({100 * len(missed) / n:.1f}%), extracting {len(keep)}")

    idx, saved = 0, 0
    last_target = max(targets) if targets else -1
    while idx <= last_target:
        ok = cap.grab()
        if not ok:
            break
        if idx in targets:
            _, frame = cap.retrieve()
            k = targets[idx]
            name = f"{run}_{k:06d}"
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), frame)

            res = model.predict(frame, imgsz=640, conf=args.conf, device=device, retina_masks=True, verbose=False)[0]
            lines, best_inst, n_blood = [], 0.0, 0
            prev = frame.copy()
            if res.masks is not None:
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for c, p, poly, m in zip(cls, conf, res.masks.xyn, res.masks.data.cpu().numpy()):
                    if len(poly) < 3 or (c != INSTRUMENT and p < args.blood_conf):
                        continue
                    lines.append(f"{c} " + " ".join(f"{v:.5f}" for v in poly.flatten()))
                    if c == INSTRUMENT:
                        best_inst = max(best_inst, float(p))
                    else:
                        n_blood += 1
                    mask = cv2.resize(m, (frame.shape[1], frame.shape[0])) > 0.5
                    prev[mask] = (0.55 * prev[mask] + 0.45 * np.array(COLORS[c])).astype(np.uint8)
                    pts = (poly * [frame.shape[1], frame.shape[0]]).astype(np.int32)
                    cv2.polylines(prev, [pts], True, COLORS[c], 1)
                    cv2.putText(prev, f"{model.names[c]} {p:.2f}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 2)
            with open(os.path.join(args.out, "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            cv2.putText(prev, f"{name}  best instrument conf {best_inst:.2f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(args.out, "preview", name + ".jpg"), prev)
            summary.writerow([name + ".jpg", run, k, rows[k - 1]["time_s"], f"{best_inst:.3f}", n_blood])
            saved += 1
        idx += 1
    cap.release()
    print(f"  saved {saved} frames")

summary_f.close()
print("Done ->", args.out)
