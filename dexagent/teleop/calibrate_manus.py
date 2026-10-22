"""Interactive Manus open/closed calibration → manus_glove_cal.json.

    python -m dexagent.teleop.calibrate_manus [--out manus_glove_cal.json]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dexagent.teleop.manus import GloveCalibration, ManusReader, features


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("manus_glove_cal.json"))
    ap.add_argument("--samples", type=int, default=45)
    a = ap.parse_args(argv)

    reader = ManusReader(calibration_file=None)
    reader.start()
    try:
        def sample_mean():
            buf = []
            for _ in range(a.samples):
                # Read raw via publisher
                pub, gid = reader._publisher, reader._gid
                data = pub.GetGloveData(gid)
                if data and (data.get("raw_nodes") or []):
                    buf.append(features(data["raw_nodes"], reader.thumb_chord_weight))
                time.sleep(0.02)
            if not buf:
                raise RuntimeError("no glove data")
            return np.mean(np.stack(buf), axis=0)

        input("1/2 — Hold hand FLAT and OPEN, then press Enter… ")
        open_v = sample_mean()
        print("open", np.round(open_v, 4).tolist())
        input("2/2 — Make a FIST, then press Enter… ")
        closed_v = sample_mean()
        print("closed", np.round(closed_v, 4).tolist())
        cal = GloveCalibration(open_v, closed_v, reader.thumb_chord_weight)
        cal.save(a.out)
        print(f"saved {a.out.resolve()}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
