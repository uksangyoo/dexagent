"""Scorer for pen_writing_grasp, independent of the policy. Reads simulator state only:
the pen's pose and collider size, and pen-table / pen-cup contacts."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("pen",)
TABLE_Z = 0.75
CUP_TOP = 0.06            # cup walls are 60 mm tall
CLEAR_OF_CUP = 0.030      # strokes count only with the tip this far (xy) from where the pen stood
TIP_ON_TABLE = 0.003      # lower end within 3 mm of the table
MAX_LEAN = np.radians(60)  # pen within 60 deg of vertical: the contact is at its tip, not along the barrel
GAP = 10                  # contact gaps up to 10 steps (200 ms) do not end a stroke
MIN_PATH, MIN_EXTENT, MIN_FRACTION = 0.025, 0.020, 0.8
HELD_TIP = 0.030


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        m = sim.model
        self.pen_shapes = list(sim.scene.handles.objects["pen"].shapes)
        self.radius, self.half = (float(v) for v in m.shape_scale.numpy()[self.pen_shapes[0]][:2])
        labels = list(m.shape_label)
        self.table = [i for i, l in enumerate(labels) if l == "table"]
        self.cup = [i for i, l in enumerate(labels) if l.startswith("cup_wall")]
        p, R = self._pose()
        self.p0 = p
        self.home = self._tip(p, R)[:2]          # where the pen stood in the cup
        self.k = 0
        self.max_tip = 0.0
        self.out = False                          # tip has cleared the cup top
        self.rows = []                            # (step, inking, tip xy, lean) after leaving the cup
        self.hist = []

    def _pose(self):
        p = np.array(self.sim["pen"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def _tip(self, p, R):
        a = R[:, 2] if R[2, 2] < 0 else -R[:, 2]
        return p + a * (self.half + self.radius)

    def _touch(self, shapes):
        return bool(shapes) and bool(self.sim.contacts.in_contact(self.pen_shapes, shapes))

    def step(self):
        self.sim.contacts.update(self.sim.step_count)
        p, R = self._pose()
        tip = self._tip(p, R)
        h = tip[2] - TABLE_Z
        self.max_tip = max(self.max_tip, h)
        self.out = self.out or h > CUP_TOP + 0.002
        lean = float(np.arccos(min(1.0, abs(R[2, 2]))))
        if self.out:
            ink = (self._touch(self.table) and not self._touch(self.cup) and h < TIP_ON_TABLE
                   and lean < MAX_LEAN and np.linalg.norm(tip[:2] - self.home) > CLEAR_OF_CUP)
            self.rows.append((self.k, bool(ink), tip[:2].copy(), lean))
        self.hist.append(p.copy())
        self.k += 1

    def _strokes(self):
        ink = [r for r in self.rows if r[1]]
        segs, cur = [], []
        for r in ink:
            if cur and r[0] - cur[-1][0] > GAP:
                segs.append(cur); cur = []
            cur.append(r)
        if cur:
            segs.append(cur)
        out = []
        for s in segs:
            xy = np.array([r[2] for r in s])
            span = s[-1][0] - s[0][0] + 1
            out.append({"path": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(s) > 1 else 0.0,
                        "extent": float(np.max(np.linalg.norm(xy - xy[0], axis=1))),
                        "fraction": len(s) / span, "steps": span,
                        "lean": float(np.degrees(np.mean([r[3] for r in s])))})
        return out

    def result(self):
        self.sim.contacts.update(self.sim.step_count)
        p, R = self._pose()
        tip_h = self._tip(p, R)[2] - TABLE_Z
        back = self.hist[max(0, len(self.hist) - 25)]
        still = float(np.linalg.norm(p - back))      # pen motion over the last 0.5 s
        free = not self._touch(self.table) and not self._touch(self.cup)
        held = tip_h > HELD_TIP and free and still < 0.005
        strokes = self._strokes()
        good = [s for s in strokes if s["path"] >= MIN_PATH and s["extent"] >= MIN_EXTENT and s["fraction"] >= MIN_FRACTION]
        best = max(good or strokes, key=lambda s: s["path"], default=None)
        m = {"max_tip_height_mm": round(1000 * self.max_tip, 1), "left_cup": bool(self.out),
             "ink_steps": int(sum(r[1] for r in self.rows)), "strokes": len(strokes),
             "best_stroke_path_mm": round(1000 * best["path"], 1) if best else 0.0,
             "best_stroke_extent_mm": round(1000 * best["extent"], 1) if best else 0.0,
             "best_stroke_contact_fraction": round(best["fraction"], 3) if best else 0.0,
             "best_stroke_mean_lean_deg": round(best["lean"], 1) if best else None,
             "final_tip_height_mm": round(1000 * tip_h, 1), "final_free_of_table_and_cup": bool(free),
             "final_motion_0p5s_mm": round(1000 * still, 1), "held_at_end": bool(held)}
        return {"success": bool(good) and bool(held), "metrics": m}
