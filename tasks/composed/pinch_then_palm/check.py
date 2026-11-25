"""Scorer for pinch_then_palm: both component scorers must pass on the same rollout.

The pinch monitor needs a held lift, a turn while held clear of the table, and a released flat rest
near the start; the palm monitor needs >= 1 s of open-palm carry, a release and a flat rest. A
policy that runs the pinch skill and then the palm skill satisfies both; neither alone does.
"""
import importlib.util
from pathlib import Path

OBJECTS = ("cube",)
_HD = Path(__file__).resolve().parents[2] / "hand_demos"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"check_{name}", _HD / name / "check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Monitor:
    def __init__(self, sim):
        self.pinch = _load("cube_pinch").Monitor(sim)
        self.palm = _load("cube_palm").Monitor(sim)

    def step(self):
        self.pinch.step()
        self.palm.step()

    def result(self):
        a, b = self.pinch.result(), self.palm.result()
        return {"success": bool(a["success"] and b["success"]),
                "metrics": {"pinch_ok": a["success"], "palm_ok": b["success"],
                            **{f"pinch_{k}": v for k, v in a["metrics"].items()},
                            **{f"palm_{k}": v for k, v in b["metrics"].items()}}}
