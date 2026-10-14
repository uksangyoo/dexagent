#!/usr/bin/env python3
"""Import the actual hand into Newton on CPU and check forward kinematics.

This does NOT run manipulation, physics stepping, an arm, or RTX rendering.
Upstream assets are unmodified. Mesh paths are resolved in a temporary URDF.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--urdf", type=Path, default=root / "tesollo_model/dg5fs/dg5fs_right_w_mount.urdf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import warp as wp
    import newton

    wp.init()
    tree = ET.parse(args.urdf)
    for mesh in tree.findall(".//mesh"):
        ref = mesh.get("filename")
        path = args.urdf.parent / ref.removeprefix("package://")
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path.resolve()))
    expected = len([j for j in tree.findall("joint") if j.get("type") != "fixed"])
    with tempfile.TemporaryDirectory(prefix="dexagent_asset_") as tmp:
        source = Path(tmp) / args.urdf.name
        tree.write(source)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = newton.ModelBuilder()
            builder.add_urdf(str(source), floating=False, enable_self_collisions=True)
            model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        positions = state.body_q.numpy()
        finite = bool(np.isfinite(positions).all())
        if model.joint_dof_count != expected or not finite:
            raise RuntimeError("Imported DOF count or finite-transform check failed")
        value = {
            "kind": "newton_cpu_import_and_fk_only",
            "asset": str(args.urdf.resolve()),
            "versions": {n: importlib.metadata.version(n) for n in ["newton", "warp-lang", "numpy"]},
            "newton_source": newton.__file__,
            "devices": [str(d) for d in wp.get_devices()],
            "body_count": model.body_count, "shape_count": model.shape_count,
            "joint_dof_count": model.joint_dof_count,
            "finite_transforms": finite,
            "body_labels": model.body_label,
            "body_transforms_xyzw": positions.tolist(),
            "warnings": sorted({str(w.message) for w in caught}),
            "limitations": ["No dynamics steps", "No UR5e or task objects", "No RTX rendering", "No task performance measurement", "Existing Newton environment differs from IsaacLab pinned revision"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Imported {model.joint_dof_count} DOF, {model.body_count} bodies, {model.shape_count} shapes; FK finite: {finite}")
    print(args.output)


if __name__ == "__main__":
    main()
