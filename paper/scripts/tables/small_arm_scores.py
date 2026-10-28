# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Per-epoch top-10 recall for every SMALL-scale arm that still has tfevents.

Walks each --roots directory for events.out.tfevents.* files, keeps the recall tags
needed for the Table S3 small-arm survey, and keys every value by EPOCH (the 'epoch'
scalar logged at the same step; resumed runs restart steps, so step is not a key).

Output (long CSV): root, arm, tag, epoch, step, value
"""
import argparse, bisect, csv, glob, os, re
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

KEEP = re.compile(r"^(clip_top10_recall|top10_recall)_(train|validation|ood_test|test)$"
                  r"|^(ood_test|val)_live_macro_top10_recall_[AH]$")
SKIP = re.compile(r"(^_|smoke|^cpu|^v01[78]smk)")


def arm_of(ev, root):
    rel = os.path.relpath(ev, root).split(os.sep)
    rel = rel[: rel.index("config")] if "config" in rel else rel[:-1]
    return "/".join(p for p in rel if not p.startswith("run_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for root in a.roots:
        evs = sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True))
        by_arm = defaultdict(list)
        for ev in evs:
            arm = arm_of(ev, root)
            if any(SKIP.search(p) for p in arm.split("/")):
                continue
            by_arm[arm].append(ev)
        for arm, files in sorted(by_arm.items()):
            files.sort(key=os.path.getmtime)
            for fi, ev in enumerate(files):
                try:
                    acc = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
                    acc.Reload()
                except Exception as e:  # truncated file
                    print(f"!! {arm}: {e}")
                    continue
                tags = acc.Tags().get("scalars", [])
                if "epoch" not in tags:
                    print(f"!! {arm}: no epoch scalar in {os.path.basename(ev)}")
                    continue
                ep = sorted((s.step, s.value) for s in acc.Scalars("epoch"))
                ep_steps = [s for s, _ in ep]
                n = 0
                for t in tags:
                    if not KEEP.match(t):
                        continue
                    for s in acc.Scalars(t):
                        i = bisect.bisect_right(ep_steps, s.step) - 1
                        e = int(round(ep[i][1])) if i >= 0 else 0
                        rows.append((root, arm, t, e, s.step, fi, s.value))
                        n += 1
                print(f"ok {arm} [{fi}]: {n} pts")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root", "arm", "tag", "epoch", "step", "file_idx", "value"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
