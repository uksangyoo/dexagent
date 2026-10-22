"""pen_table_slide, layered level.

Phases and layers:
  posture  joints layer: OPEN with middle/ring/little tucked (seed for the pinch IK)
  approach fingertips layer + arm: thumb/index pads open either side of the pen, palm above, then down
  close    fingertips layer, closed loop on contact force: common half-width h about a fixed palm-frame centre
  slide    fingertips layer + arm: lower palm step by step, regulate grip force, track slide exactly as check.py
  lift     fingertips layer + arm: raise palm, keep regulating grip
"""
import numpy as np

TABLE = 0.75
BLOCK_TOP = 0.81
C = np.array([0.080, 0.040, 0.118])     # thumb-index pinch centre, palm frame
EZ = np.array([0.0, 0.0, 1.0])          # pinch axis: thumb at -z, index at +z
CURL = np.array([0, 1.6, 1.3, 1.1])
OPEN = np.array([0.0, -0.3, 0.2, 0.2, 0, .25, .25, .25, 0, .25, .25, .25, 0, .25, .25, .25, 0.2, 0, .25, .25])

H_OPEN = 0.028
GRASP_BELOW_TOP = 0.012
SLIDE_GOAL = 0.021
STROKE = 0.009             # pen travel per grasp (geometry allows ~15 mm)
F_LO, F_HI = 0.45, 1.3


class Tracker:
    """Replica of check.py's slide measure (pen travel along its axis relative to the thumb pad)."""

    def __init__(self, api):
        self.api = api
        p, R = api.object_pose("pen")
        self.end = float(p[2] - TABLE - 0.0005)
        self.cum = self.cum_min = self.slip = 0.0
        self.prev = None
        self.f = np.zeros(6)
        self.fs = np.zeros(6)

    def pen(self):
        p, R = self.api.object_pose("pen")
        a = R[:, 2]
        if a[2] < 0:
            a = -a
        return p, a

    def tip_h(self):
        p, a = self.pen()
        return float(p[2] - self.end * a[2] - TABLE)

    def step(self):
        f = np.asarray(self.api.contact_forces("pen"), float)
        self.f = f
        self.fs = 0.6 * self.fs + 0.4 * f
        p, a = self.pen()
        th = float(p[2] - self.end * a[2] - TABLE)
        held = f[0] > 0.2 and f[1:5].max() > 0.2
        if held and th < 0.003:
            s = float((p - self.api.tips_world()[0]) @ a)
            if self.prev is not None:
                self.cum += s - self.prev
                self.cum_min = min(self.cum_min, self.cum)
                self.slip = max(self.slip, self.cum - self.cum_min)
            self.prev = s
        else:
            self.prev = None
        return f, th, held


def run(api):
    diag = []
    R = api.hand_down(0.7)
    tr = Tracker(api)
    p0, _ = tr.pen()
    top = p0[2] + tr.end
    z_grasp = top - GRASP_BELOW_TOP
    st = {"h": H_OPEN, "centre": C.copy(), "res": 0.0, "palm": None}

    def d(msg):
        diag.append(f"t={api.t:5.2f} {msg}")

    def tips(h, centre):
        return {"thumb": centre - h * EZ, "index": centre + h * EZ}

    def cmd(palm=None, seconds=0.02):
        r = yield from api.move_tips(xyz=palm, R=None if palm is None else R,
                                     tips=tips(st["h"], st["centre"]), seconds=seconds)
        if palm is not None:
            st["palm"] = np.asarray(palm, float).copy()
        try:
            st["res"] = max(st["res"], max(r["ik_residual_mm"].values()))
        except Exception:
            pass
        return tr.step()

    def regulate(lo=F_LO, hi=F_HI, rate=0.00004):
        fs = tr.fs
        m = min(fs[0], fs[1:5].max())
        if m < lo:
            st["h"] -= rate
        elif fs[0] > hi and fs[1:5].max() > hi:
            st["h"] += rate
        st["h"] = float(np.clip(st["h"], 0.002, H_OPEN))

    def pen_centre_palm(z_world):
        p, a = tr.pen()
        w = p + a * (z_world - p[2]) / a[2]
        P, Rp = api.measured_palm()
        return Rp.T @ (w - P)

    def close(max_s=5.0):
        pp, Rpen = api.object_in_palm("pen")
        ap = Rpen[:, 2]
        st["centre"] = pp + ap * (C[0] - pp[0]) / ap[0]
        t0 = api.t
        while api.t - t0 < max_s:
            fs = tr.fs
            m = min(fs[0], fs[1:5].max())
            if m > 0.6:
                break
            st["h"] -= 0.00015 if max(fs[0], fs[1:5].max()) < 0.1 else 0.00004
            st["h"] = max(st["h"], 0.002)
            yield from cmd()
        for _ in range(10):
            regulate()
            yield from cmd()
        d(f"closed h={st['h'] * 1000:.1f}mm f={np.round(tr.fs, 2).tolist()} centre={np.round(st['centre'] * 1000, 1).tolist()}")

    def slide(stroke_goal, max_lower=0.014, v=0.008):
        """Lower the palm while gripping; stop at the overall goal, this stroke's travel, or trouble."""
        palm = st["palm"].copy()
        z0 = palm[2]
        c0 = tr.cum
        k = lost = 0
        while True:
            pads = api.tips_world()
            lowered = z0 - palm[2]
            if tr.slip >= SLIDE_GOAL:
                why = "goal"; break
            if tr.cum - c0 >= stroke_goal:
                why = "stroke"; break
            if lowered >= max_lower:
                why = "max_lower"; break
            if min(pads[0][2], pads[1][2]) < BLOCK_TOP + 0.016:
                why = "block"; break
            if lost >= 6:
                why = "lost"; break
            palm = palm - np.array([0, 0, v * 0.02])
            regulate()
            f, th, held = yield from cmd(palm)
            lost = lost + 1 if min(tr.fs[0], tr.fs[1:5].max()) < 0.2 else 0
            if k % 25 == 0:
                d(f"slide low={lowered * 1000:.1f} slip={tr.slip * 1000:.1f} tip_h={th * 1000:.1f} h={st['h'] * 1000:.1f} "
                  f"f={np.round(f, 2).tolist()}")
            k += 1
        d(f"stroke end ({why}) lowered={(z0 - palm[2]) * 1000:.1f} stroke={(tr.cum - c0) * 1000:.1f} slip={tr.slip * 1000:.1f}")
        return why

    def axis_point(z_world):
        p, a = tr.pen()
        return p + a * (z_world - p[2]) / a[2]

    def reposition(z_pinch, h_open, seconds=0.6):
        """Open to h_open, then move the palm so the current pinch centre sits on the pen axis at z_pinch."""
        while st["h"] < h_open - 1e-6:
            st["h"] = min(h_open, st["h"] + 0.0006)
            yield from cmd()
        tgt = api.palm_for(axis_point(z_pinch), st["centre"], R)
        yield from api.move_tips(xyz=tgt, R=R, tips=tips(st["h"], st["centre"]), seconds=seconds)
        st["palm"] = tgt
        tr.step()

    def approach_from_above():
        """Open wide, hover over the pen top, descend to the grasp height (used at start and for a retry)."""
        st["h"], st["centre"] = H_OPEN, C.copy()
        p0, _ = tr.pen()
        hover = api.palm_for(np.array([p0[0], p0[1], top + 0.035]), C, R)
        r = yield from api.move_tips(xyz=hover, R=R, tips=tips(H_OPEN, C), seconds=1.5)
        d(f"hover ik={r['ik_residual_mm']}")
        down = api.palm_for(axis_point(z_grasp), C, R)
        yield from api.move_tips(xyz=down, R=R, tips=tips(H_OPEN, C), seconds=1.5)
        st["palm"] = down
        yield from api.hold(0.3)

    def lift(height=0.12, n=120):
        for _ in range(15):
            regulate(lo=0.6)
            yield from cmd()
        palm0 = st["palm"].copy()
        lost = 0
        for k in range(1, n + 1):
            u = k / n
            u = u * u * (3 - 2 * u)
            regulate(lo=0.6)
            f, th, held = yield from cmd(palm0 + np.array([0, 0, height * u]))
            lost = lost + 1 if min(tr.fs[0], tr.fs[1:5].max()) < 0.15 else 0
            if k % 30 == 0:
                d(f"lift u={u:.2f} tip_h={th * 1000:.1f} f={np.round(f, 2).tolist()} h={st['h'] * 1000:.1f}")
            if lost >= 8:
                d(f"lift lost the pen at u={u:.2f} tip_h={th * 1000:.1f}")
                return False
        for _ in range(25):
            regulate(lo=0.6)
            f, th, held = yield from cmd()
        d(f"lifted tip_h={th * 1000:.1f} held={held} f={np.round(f, 2).tolist()}")
        return th > 0.07 and held

    try:
        d(f"pen p0={np.round(p0, 4).tolist()} end={tr.end * 1000:.1f}mm top={top:.4f}")
        # posture: open with middle/ring/little tucked (joints layer)
        q = OPEN.copy()
        q[8:12] = CURL
        q[12:16] = CURL
        q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
        yield from api.move_joints(q=q, seconds=0.6)
        yield from approach_from_above()

        # short strokes: close, slide <= 9 mm, open slightly, climb back, re-centre, repeat.
        # (iter 1: one long stroke loses the pen after ~15 mm when its top meets the curled fingers)
        for stroke in range(6):
            yield from close()
            why = yield from slide(STROKE)
            if tr.slip >= SLIDE_GOAL:
                break
            h_rel = (H_OPEN if why == "lost" else st["h"] + 0.005)
            yield from reposition(z_grasp, h_rel)
            d(f"restroke {stroke + 1} slip so far {tr.slip * 1000:.1f} cum={tr.cum * 1000:.1f}")

        ok = yield from lift()
        if not ok:
            # retry: let go, come back from above, grasp and lift again (slide credit is already banked)
            yield from reposition(z_grasp + 0.06, H_OPEN, seconds=1.0)
            yield from approach_from_above()
            yield from close()
            ok = yield from lift()
        th = tr.tip_h()
        d(f"final ok={ok} tip_h={th * 1000:.1f} f={np.round(tr.f, 2).tolist()} slip={tr.slip * 1000:.1f} max_ik_res={st['res']}")
    finally:
        for line in diag[-45:]:
            api.note(line)
