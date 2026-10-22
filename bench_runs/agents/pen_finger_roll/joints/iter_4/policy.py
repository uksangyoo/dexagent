"""pen_finger_roll, joints level.

Grasp: tripod. Thumb pad on the -z side of the pen, index and middle pads on the +z side at y +-YO
(a V that locates the pen in y). Pen along palm x (vertical, palm down).
In-hand motions are rolling the pen between the pads:
  swing (about palm y): thumb pad +d along x, index/middle pads -d       (closed loop on tilt)
  roll  (about pen axis): each pad +e along its tangent x_hat x n_f        (closed loop on twist)
Finger IK is built online from api.tips(): numeric Jacobians from joint probes, DLS servo with
Broyden updates, linearised solve around the probed configuration. Falls back to a thumb-index
pinch if the middle finger cannot reach.
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
RIM = TABLE + 0.06
R_PEN = 0.00375
PAD = 0.008                       # pad centre sits ~8 mm inside the fingertip surface
X = np.array([1.0, 0, 0])
TH, IX, MI = 0, 1, 2
GRASP_BELOW_TOP = 0.022
YO = 0.009                        # index / middle lateral offset from the pen line
# Postures this policy's own online IK converged to in earlier iterations (warm start).
Q_TH = [0.36, -0.84, 0.80, 0.55]
Q_IX = [-0.24, 0.31, 0.98, 0.82]
Q_MI = [-0.35, 0.31, 0.98, 0.82]


def fmt(v, s=1000, n=1):
    return np.round(np.asarray(v, float) * s, n).tolist()


def dls(J, e, lam=0.01):
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), e)


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class Mon:
    """Mirror of the scorer's bookkeeping, from api observations."""

    def __init__(self, api, reach):
        self.api, self.reach = api, reach
        self.rest = None
        self.R_last = None
        self.twist = 0.0
        self.tw_lo = self.tw_hi = 0.0
        self.t_lo = self.t_hi = 0.0
        self.tilt = 0.0
        self.bottom = 0.0

    def step(self):
        pw, Rw = self.api.object_pose("pen")
        _, Rr = self.api.object_in_palm("pen")
        self.bottom = pw[2] - self.reach * abs(Rw[2, 2])
        if self.rest is None:
            if self.bottom > RIM + 0.01:
                self.rest = Rr[:, 2].copy()
                self.R_last = Rr
            return
        if self.bottom > RIM:
            self.twist += _twist(self.R_last, Rr)
            self.tw_lo, self.tw_hi = min(self.tw_lo, self.twist), max(self.tw_hi, self.twist)
            a = Rr[:, 2]
            self.tilt = np.degrees(_wrap(np.arctan2(a[2], a[0]) - np.arctan2(self.rest[2], self.rest[0])))
            self.t_lo, self.t_hi = min(self.t_lo, self.tilt), max(self.t_hi, self.tilt)
        self.R_last = Rr

    def summary(self):
        return (f"MON tilt lo/hi {self.t_lo:.1f}/{self.t_hi:.1f} twist range {np.degrees(self.tw_hi - self.tw_lo):.1f} "
                f"bottom-rim {1000 * (self.bottom - RIM):.0f}mm")


def run(api):
    D = []
    try:
        yield from main(api, D)
    finally:
        for line in D[-56:]:
            api.note(line)


def main(api, D):
    lim = api.joint_limits

    def clip(q):
        return np.clip(q, lim[:, 0], lim[:, 1])

    def probe(q0, fingers, h=0.04, settle=0.2):
        yield from api.move(q=q0, seconds=0.2)
        yield from api.hold(settle)
        base = api.tips()
        J = {f: np.zeros((3, 4)) for f in fingers}
        for j in range(4):
            q = q0.copy()
            hs = {}
            for f in fingers:
                k = 4 * f + j
                hh = h if q0[k] + h <= lim[k, 1] else -h
                q[k] += hh
                hs[f] = hh
            yield from api.move(q=q, seconds=0.1)
            yield from api.hold(settle)
            t = api.tips()
            for f in fingers:
                J[f][:, j] = (t[f] - base[f]) / hs[f]
        yield from api.move(q=q0, seconds=0.1)
        yield from api.hold(settle)
        return J, api.tips()

    def ik(q, targets, J, iters=14, tol=0.0008):
        for it in range(iters):
            t = api.tips()
            errs = {f: targets[f] - t[f] for f in targets}
            worst = max(np.linalg.norm(e) for e in errs.values())
            if worst < tol:
                break
            qn = q.copy()
            for f, e in errs.items():
                qn[4 * f:4 * f + 4] += np.clip(dls(J[f], e), -0.3, 0.3)
            qn = clip(qn)
            yield from api.move(q=qn, seconds=0.12)
            yield from api.hold(0.16)
            t2 = api.tips()
            for f in targets:
                dq = qn[4 * f:4 * f + 4] - q[4 * f:4 * f + 4]
                if dq @ dq > 1e-6:
                    J[f] += np.outer((t2[f] - t[f]) - J[f] @ dq, dq) / (dq @ dq)
            q = qn
        t = api.tips()
        errs = {f: float(np.linalg.norm(targets[f] - t[f])) for f in targets}
        return q, errs

    # ---------------------------------------------------------------- scene facts
    p0, Rw0 = api.object_pose("pen")
    reach = p0[2] - TABLE
    top = p0 + np.array([0, 0, reach])
    X0, R = api.palm_pose()
    mon = Mon(api, reach)
    D.append(f"pen p0 {fmt(p0)} reach {reach * 1000:.1f}mm")

    # ---------------------------------------------------------------- grasp geometry + online IK
    def layout(tripod):
        if tripod:
            C = np.array([0.080, 0.030, 0.118])
            s = YO / (R_PEN + PAD)
            c = np.sqrt(1 - s * s)
            N = {TH: np.array([0, 0, -1.0]), IX: np.array([0, s, c]), MI: np.array([0, -s, c])}
        else:
            C = np.array([0.080, 0.040, 0.118])
            N = {TH: np.array([0, 0, -1.0]), IX: np.array([0, 0, 1.0])}
        return C, N

    def pads(Cp, N, radial):
        return {f: Cp + radial * n for f, n in N.items()}

    q = api.OPEN.copy()
    q[0:4], q[4:8], q[8:12] = Q_TH, Q_IX, Q_MI
    C, N = layout(True)
    fing = tuple(N)
    J, tips = yield from probe(q, fing)
    D.append(f"guess tips {fmt(tips[:3], n=0)} limits mid {np.round(lim[8:12], 2).tolist()}")
    rad_pc = R_PEN + PAD + 0.003
    q, errs = yield from ik(q, pads(C, N, rad_pc), J, iters=14)
    D.append(f"IK tripod errs mm {({k: round(v * 1000, 1) for k, v in errs.items()})} q {np.round(q[:12], 2).tolist()}")
    if max(errs.values()) > 0.003:
        C, N = layout(False)
        fing = tuple(N)
        q[8:12] = api.OPEN[8:12]
        J, _ = yield from probe(q, fing)
        q, errs = yield from ik(q, pads(C, N, rad_pc), J, iters=10)
        D.append(f"FALLBACK pinch errs {({k: round(v * 1000, 1) for k, v in errs.items()})}")
    J, tips_pc = yield from probe(q, fing)
    q_pc = q.copy()
    D.append(f"q_pc {np.round(q_pc[:12], 3).tolist()} tips {fmt(tips_pc[:3])}")

    def solve(targets):
        qq = q_pc.copy()
        for f, x in targets.items():
            qq[4 * f:4 * f + 4] += dls(J[f], np.asarray(x) - tips_pc[f], lam=0.005)
        return clip(qq)

    q_open = solve(pads(C, N, 0.028))

    # ---------------------------------------------------------------- approach
    G = top - np.array([0, 0, GRASP_BELOW_TOP])
    X_pre = api.palm_for(G + np.array([0, 0, 0.05]), C, R)
    X_g = api.palm_for(G, C, R)
    yield from api.move(xyz=X_pre, R=R, q=q_open, seconds=2.5)
    yield from api.move(xyz=X_g, R=R, seconds=1.6)
    yield from api.hold(0.3)

    def grasp_point():
        pw, Rw = api.object_pose("pen")
        a = Rw[:, 2] if Rw[2, 2] > 0 else -Rw[:, 2]
        gw = pw + a * reach - a * GRASP_BELOW_TOP
        Pm, Rm = api.measured_palm()
        return Rm.T @ (gw - Pm)

    Gp = grasp_point()
    yield from api.move(q=solve(pads(Gp, N, rad_pc)), seconds=0.6)
    yield from api.hold(0.2)
    D.append(f"G in palm {fmt(Gp)} (C {fmt(C)}) precontact G {fmt(grasp_point())}")

    sq = 0.0065
    for k in range(6):
        Gp = grasp_point()
        yield from api.move(q=solve(pads(Gp, N, R_PEN + PAD - sq)), seconds=0.4)
        yield from api.hold(0.25)
        F = api.contact_forces("pen")
        D.append(f"squeeze {sq * 1000:.1f}mm F {np.round(F, 2).tolist()}")
        if min(F[f] for f in fing) > 1.2:
            break
        sq += 0.0015
    tips_sq = pads(Gp, N, R_PEN + PAD - sq)

    # ---------------------------------------------------------------- lift
    X_l = X_g + np.array([0, 0, 0.12])
    for _ in api.move(xyz=X_l, R=R, seconds=2.0):
        mon.step()
        yield
    for _ in api.hold(0.5):
        mon.step()
        yield
    pp, Rp = api.object_in_palm("pen")
    D.append(f"lifted t={api.t:.1f}: rest {None if mon.rest is None else np.round(mon.rest, 3).tolist()} pen {fmt(pp)} "
             f"G {fmt(grasp_point())} F {np.round(api.contact_forces('pen'), 2).tolist()} bottom-rim {1000 * (mon.bottom - RIM):.0f}mm")
    if mon.rest is None:
        D.append("pen not out of cup; abort")
        return
    sgn_a = 1.0 if mon.rest[0] > 0 else -1.0

    # ---------------------------------------------------------------- in-hand control
    LT = 0.017      # observed in iter 1: tilt ~ pad differential / 17 mm
    DMAX = 0.011
    st = {"Id": 0.0, "Ie": 0.0, "tt": 0.0, "tw": mon.twist, "d": 0.0, "e": 0.0}
    ts = []
    T = {f: np.cross(X, n) for f, n in N.items()}  # rolling tangents (pen rotation about +x)

    def stepc(tilt_target, twist_target, k=[0]):
        st["tt"] += float(np.clip(tilt_target - st["tt"], -0.25, 0.25))        # 12.5 deg/s
        st["tw"] += float(np.clip(twist_target - st["tw"], -0.007, 0.007))     # 20 deg/s
        err = np.radians(st["tt"] - mon.tilt)
        st["Id"] = float(np.clip(st["Id"] + 0.010 * LT * err, -0.008, 0.008))
        d_raw = float(np.clip(0.6 * LT * np.radians(st["tt"]) + st["Id"], -DMAX, DMAX))
        d = st["d"] + float(np.clip(d_raw - st["d"], -0.00006, 0.00006))       # <= 3 mm/s
        etw = st["tw"] - mon.twist
        st["Ie"] = float(np.clip(st["Ie"] + 0.08 * 0.004 * etw * sgn_a, -0.006, 0.006))
        e = float(np.clip(st["Ie"] + 0.004 * etw * sgn_a, -0.007, 0.007))
        st["d"], st["e"] = d, e
        tg = {}
        for f in N:
            dx = d if f == TH else -d
            tg[f] = tips_sq[f] + dx * X + e * T[f]
        yield from api.move(q=solve(tg), seconds=0.02)
        mon.step()
        k[0] += 1
        if k[0] % 10 == 0:
            F = api.contact_forces("pen")
            ts.append(f"{api.t:.1f}:{mon.tilt:.0f}/{d * 1000:.1f}/{np.degrees(mon.twist):.0f}/{e * 1000:.1f}/"
                      + "/".join(f"{F[f]:.1f}" for f in fing))

    def flush(tag):
        for j in range(0, len(ts), 8):
            D.append(f"TS[{tag}] t:tilt/d/twist/e/F " + " ".join(ts[j:j + 8]))
        ts.clear()

    tw_hold = mon.twist
    for name, tgt, cond, tmax in (("pos", 18.0, lambda: mon.tilt >= 14.0, 6.0),
                                  ("neg", -22.0, lambda: mon.tilt <= -11.0 and mon.t_hi - mon.tilt >= 31.5, 8.0)):
        t0, ok = api.t, False
        while api.t - t0 < tmax:
            yield from stepc(tgt, tw_hold)
            if cond():
                ok = True
                break
            if mon.bottom < RIM:
                break
        D.append(f"swing {name}: ok={ok} t={api.t:.2f} tilt {mon.tilt:.1f} d {st['d'] * 1000:.1f}mm")
        if mon.bottom < RIM:
            flush("swing")
            D.append("DROPPED")
            return
        if name == "pos" and mon.t_hi < 10.5:
            D.append("pos side short of 10 deg")
        # freeze here: restart the target ramp from the current tilt, keep d continuous
        st["tt"] = mon.tilt
        st["Id"] = st["d"] - 0.6 * LT * np.radians(st["tt"])
    t0, calm = api.t, 0
    while api.t - t0 < 5.0 and calm < 60:
        yield from stepc(0.0, tw_hold)
        calm = calm + 1 if abs(mon.tilt) < 2.5 else 0
    flush("swing")
    D.append(f"swing back: calm {calm} t={api.t:.2f} tilt {mon.tilt:.1f}; {mon.summary()}")
    if mon.bottom < RIM:
        D.append("DROPPED")
        return

    # roll: twist target +22 deg then -22 deg about the start, then back
    tw0 = mon.twist
    for name, off, tmax in (("plus", 22.0, 4.0), ("minus", -22.0, 5.0)):
        t0 = api.t
        while api.t - t0 < tmax:
            yield from stepc(0.0, tw0 + np.radians(off))
            if abs(np.degrees(mon.twist - tw0) - off) < 3:
                break
            if mon.bottom < RIM:
                break
        D.append(f"roll {name} t={api.t:.2f} dtw {np.degrees(mon.twist - tw0):.1f} e {st['e'] * 1000:.2f}mm")
    t0 = api.t
    while api.t - t0 < 2.5:
        yield from stepc(0.0, tw0)
    flush("roll")
    D.append(f"end t={api.t:.2f} F {np.round(api.contact_forces('pen'), 2).tolist()} {mon.summary()}")
