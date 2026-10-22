"""Recompute HOM features for existing demos in place.

    python -m dexagent.hom.recompute tasks/hand_demos/cube_pinch/demos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dexagent.hom.features import recompute_hom_arrays
from dexagent.skills.schema import list_demos, load_demo, save_demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demos_root", type=Path)
    ap.add_argument("--half", type=float, default=0.02, help="box half-extent (m)")
    a = ap.parse_args(argv)
    paths = list_demos(a.demos_root, success_only=False)
    # Also include failed/
    failed = a.demos_root / "failed"
    if failed.is_dir():
        paths = list(dict.fromkeys(paths + list_demos(failed, success_only=False)))
    for p in paths:
        meta, arrays = load_demo(p)
        arrays.update(
            recompute_hom_arrays(
                arrays["tip_xyz"],
                arrays["object_pose"],
                arrays["t"],
                box_half=(a.half,) * 3,
            )
        )
        save_demo(p, meta, arrays)
        print(f"recomputed HOM → {p}", flush=True)


if __name__ == "__main__":
    main()
