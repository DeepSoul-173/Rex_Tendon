"""Selective guardrail watcher for the foundation training run.

Emits ONE line only on actionable events: std divergence (the bug we fixed),
errors, curriculum promotions, first place-success milestones, and completion.
Quiet otherwise. Designed for the Monitor tool (each printed line = 1 event).
"""
import os
import re
import subprocess
import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "_train_2section.log"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 6_000_000
last_obj = None
hit = set()          # place-success milestones already reported
last_ts_seen = 0
stagnant_polls = 0


def emit(msg):
    print(msg, flush=True)


def col_val(line):
    # SB3 table row: "|    name      | value |"
    parts = line.split("|")
    if len(parts) >= 3:
        try:
            return float(parts[2].strip())
        except ValueError:
            return None
    return None


def python_alive():
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=20).stdout.lower()
        return out.count("python.exe")
    except Exception:
        return 1  # assume alive on probe failure


emit("watcher armed: guarding foundation run (std-divergence / errors / promotions / done)")
while True:
    time.sleep(120)
    if not os.path.exists(LOG):
        continue
    try:
        with open(LOG, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        continue

    text = "".join(lines[-400:])

    # crash / error signatures
    for sig in ("Traceback", "MemoryError", "CUDA error", "Killed", "RuntimeError"):
        if sig in text and sig not in hit:
            hit.add(sig)
            emit(f"ERROR signature in log: {sig}  -- run may have died")

    # latest table values
    last_std = last_place = last_obj_now = last_ts = last_rew = None
    for ln in lines[-400:]:
        if "| std " in ln or re.search(r"\|\s*std\s*\|", ln):
            v = col_val(ln);  last_std = v if v is not None else last_std
        elif "place_success_rate " in ln:
            v = col_val(ln);  last_place = v if v is not None else last_place
        elif "num_objects" in ln:
            v = col_val(ln);  last_obj_now = v if v is not None else last_obj_now
        elif "total_timesteps" in ln:
            v = col_val(ln);  last_ts = v if v is not None else last_ts
        elif "ep_rew_mean" in ln:
            v = col_val(ln);  last_rew = v if v is not None else last_rew

    # 1) divergence guard (the bug: std exploded to 8e4; healthy ~1.0)
    if last_std is not None and last_std > 5.0 and "divergence" not in hit:
        hit.add("divergence")
        emit(f"DIVERGENCE WARNING: policy std={last_std:.3g} (healthy ~1.0). "
             f"The entropy blowup may be recurring at scale.")

    # 2) curriculum promotion
    if last_obj_now is not None and last_obj is not None and last_obj_now > last_obj:
        emit(f"curriculum promoted: {int(last_obj)} -> {int(last_obj_now)} objects "
             f"(place_success crossed 90% at prev level)")
    if last_obj_now is not None:
        last_obj = last_obj_now

    # 3) place-success milestones (first time crossing each)
    if last_place is not None:
        for thr in (1, 25, 50, 75, 90):
            key = f"place{thr}"
            if last_place >= thr and key not in hit:
                hit.add(key)
                emit(f"milestone: place_success_rate reached {last_place:.0f}% "
                     f"(obj={int(last_obj_now) if last_obj_now else '?'}, "
                     f"steps={int(last_ts) if last_ts else '?'}, rew={last_rew})")

    # 4) progress stall / completion (process gone)
    if last_ts is not None:
        if last_ts <= last_ts_seen:
            stagnant_polls += 1
        else:
            stagnant_polls = 0
            last_ts_seen = last_ts

    if stagnant_polls >= 2:
        alive = python_alive()
        if alive <= 1:
            emit(f"training FINISHED or stopped: last step={int(last_ts) if last_ts else '?'}/{TOTAL:,}, "
                 f"place_success={last_place}, objects={int(last_obj_now) if last_obj_now else '?'}")
            sys.exit(0)
        else:
            stagnant_polls = 0  # still alive, just slow logging
