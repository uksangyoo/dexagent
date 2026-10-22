"""Baoding by palm gyration: the cupped fingers and thumb are walls, gravity
does the rolling. The palm is tilted (<= ~14 deg) toward a downhill direction
psi in the palm plane (psi = atan2(y, z): 0 = toward the fingers, +90 = toward
the thumb, -90 = toward the little finger) about a pivot at the palm centre.
The pair angle is tracked online exactly as the scorer does; the policy stops
as soon as the counted rotation passes the goal with margin.

Iteration 1 is also a probe: phases are logged (positions, contact forces).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DT = 0.02
ALPHA = np.radians(12.0)
GOAL = np.radians(1000.0)  # stop handled in run()
NAMES = ("thumb", "index", "middle", "ring", "little")


def held(p):
    return -0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15


class Ctl:
    def __init__(self, api):
        self.api = api
        self.P0, self.R0 = api.palm_pose()
        self.piv = np.array([0.045, 0.0, 0.075])
        self.W = self.P0 + self.R0 @ self.piv
        self.tau = np.zeros(2)            # tilt vector (palm y, z) [rad], points downhill
        self.total = 0.0
        self.last = None
        self.notes = []
        self.series = []
        self.k = 0
        self.reflex = 0
        self.lo = [9, 9]                  # min y, min z seen [m]
        self.hi = [-9, -9]
        self.obs()

    # ------------------------------------------------------------ sensing
    def obs(self):
        api = self.api
        self.pa = api.object_in_palm("ball_a")[0]
        self.pb = api.object_in_palm("ball_b")[0]
        _, Rp = api.measured_palm()
        self.tilt = float(np.degrees(np.arccos(np.clip(Rp[2, 0], -1, 1))))
        h = held(self.pa) and held(self.pb)
        d = self.pb - self.pa
        ang = float(np.arctan2(d[1], d[2]))
        self.ang = ang
        if h and self.tilt < 30 and self.last is not None and api.t > 1.0:
            self.total += (ang - self.last + np.pi) % (2 * np.pi) - np.pi
        self.last = ang if h else None
        for p in (self.pa, self.pb):
            self.lo = [min(self.lo[0], p[1]), min(self.lo[1], p[2])]
            self.hi = [max(self.hi[0], p[1]), max(self.hi[1], p[2])]
        if self.k % 25 == 0:
            self.series.append(f"{api.t:.1f}:{np.degrees(self.total):.0f}|{self.pa[1]*1e3:.0f},{self.pa[2]*1e3:.0f}|{self.pb[1]*1e3:.0f},{self.pb[2]*1e3:.0f}")
        self.k += 1

    def done(self):
        return abs(self.total) >= GOAL

    def mark(self, what):
        api = self.api
        try:
            fa = np.round(api.contact_forces("ball_a"), 2).tolist()
            fb = np.round(api.contact_forces("ball_b"), 2).tolist()
        except Exception:
            fa = fb = None
        f = lambda p: np.round(p * 1e3).astype(int).tolist()
        self.notes.append(f"MARK {what} t={api.t:.2f} tot={np.degrees(self.total):.0f} ang={np.degrees(self.ang):.0f} "
                          f"tilt={self.tilt:.1f} a={f(self.pa)} b={f(self.pb)} Fa={fa} Fb={fb}")

    def flush(self):
        api = self.api
        api.note(f"SUMMARY tot={np.degrees(self.total):.0f} reflex={self.reflex} lo(y,z)={np.round(np.array(self.lo)*1e3).tolist()} hi={np.round(np.array(self.hi)*1e3).tolist()}")
        for n in self.notes:
            api.note(n)
        s = self.series
        for i in range(0, len(s), 30):
            api.note("SERIES " + " ".join(s[i:i + 30]))

    # ------------------------------------------------------------ acting
    def pose_for(self, tau):
        a = float(np.linalg.norm(tau))
        if a < 1e-9:
            R = self.R0
        else:
            vy, vz = tau / a
            R = self.R0 @ Rot.from_rotvec(a * np.array([0.0, -vz, vy])).as_matrix()
        return self.W - R @ self.piv, R

    def safe(self, target):
        """Reflex: a ball nearing an open edge (wrist, little side below the pocket, thumb side) -> tilt away."""
        for p in (self.pa, self.pb):
            y, z = p[1], p[2]
            if not held(p):
                continue
            if z < 0.035:
                self.reflex += 1
                return tv(10)
            if y < -0.046:
                self.reflex += 1
                return tv(60)
            if y > 0.058:
                self.reflex += 1
                return tv(-30)
        return target

    def step(self, target, rate=np.radians(25.0)):
        target = self.safe(np.asarray(target, float))
        d = target - self.tau
        n = np.linalg.norm(d)
        lim = rate * DT
        self.tau = target if n <= lim else self.tau + d * lim / n
        xyz, R = self.pose_for(self.tau)
        yield from self.api.carry(xyz, R, seconds=DT)
        self.obs()


def tv(psi_deg, alpha=ALPHA):
    p = np.radians(psi_deg)
    return alpha * np.array([np.sin(p), np.cos(p)])


def dwell(c, seconds, target=None, stop=True):
    tgt = c.tau.copy() if target is None else target
    for _ in range(int(round(seconds / DT))):
        yield from c.step(tgt)
        if stop and c.done():
            return


def ramp(c, target, seconds, stop=True):
    t0 = c.tau.copy()
    n = max(1, int(round(seconds / DT)))
    for k in range(1, n + 1):
        u = k / n
        u = u * u * (3 - 2 * u)
        yield from c.step(t0 + u * (target - t0), rate=np.radians(60))
        if stop and c.done():
            return


def arc(c, psi0, psi1, dps=40.0, alpha=ALPHA, until=None):
    n = max(1, int(round(abs(psi1 - psi0) / dps / DT)))
    for k in range(1, n + 1):
        yield from c.step(tv(psi0 + (psi1 - psi0) * k / n, alpha))
        if c.done() or (until is not None and until()):
            return


def place_pad(api, finger, T, seconds=0.8, ref="ball_b"):
    """Put `finger`'s pad at palm-frame point T, expressed through approach() on a reference ball:
    the pad goes to p + u*L for u = unit(T - p), which is T exactly."""
    p, R = api.object_in_palm(ref)
    d = np.asarray(T, float) - p
    L = float(np.linalg.norm(d))
    u = d / L
    pt = R.T @ (u * 0.01)
    return (yield from api.approach(ref, {finger: pt}, standoff=L - 0.01 - api.pad_offset, seconds=seconds))


def pq(c):
    """(P, Q, names): P = the ball on the thumb (+y) side, Q the other."""
    if c.pb[1] >= c.pa[1]:
        return c.pb, c.pa
    return c.pa, c.pb


def until(c, psi, cond, tmax, alpha=ALPHA):
    t0 = c.api.t
    while c.api.t - t0 < tmax:
        yield from c.step(tv(psi, alpha))
        if cond():
            return True
    return False


def run(api):
    c = Ctl(api)
    try:
        tips = np.round(api.tips() * 1e3).astype(int).tolist()
        c.notes.append(f"TIPS home {dict(zip(NAMES, tips))}")
        # ---- thumb wall along the +y edge of the palm (done before counting starts)
        best, cands = None, []
        for x in (0.042, 0.050):
            for y in (0.050, 0.056):
                for z in (0.060, 0.070):
                    T = np.array([x, y, z])
                    r = api.reach("thumb", T)
                    cost = r + 0.3 * np.linalg.norm(T - np.array([0.045, 0.052, 0.066]))
                    cands.append((round(r * 1e3, 1), (T * 1e3).round().astype(int).tolist()))
                    if best is None or cost < best[0]:
                        best = (cost, T)
        c.notes.append(f"THUMB reach mm {cands}")
        res = yield from place_pad(api, "thumb", best[1], seconds=0.7)
        c.notes.append(f"THUMB placed at {(best[1]*1e3).round().tolist()} res={res.get('ik_residual_mm')} tips={np.round(api.tips()[0]*1e3).tolist()}")
        while api.t < 1.0:
            yield from c.step(np.zeros(2))
        c.mark("t1")
        dab = float(np.linalg.norm(c.pa - c.pb))   # ~ 2 r
        steps = 0
        while api.t < 48 and steps < 8:
            # ---- step A: from side-by-side in the pocket, send Q around P, down the thumb wall
            yield from until(c, 80, lambda: False, 0.8)

            def stacked():
                d = c.pb - c.pa
                return abs(d[1]) < 0.5 * dab and abs(d[2]) > 0.75 * dab
            ok = False
            for psi in (105, 118, 130):
                ok = yield from until(c, psi, stacked, 2.0)
                if ok:
                    break
            yield from until(c, 72, lambda: False, 0.8)
            steps += 1
            c.mark(f"A{steps} ok={ok}")
            if abs(c.total) >= np.radians(330):
                break
            # ---- step B: upper ball slides to -y along the pocket, lower ball rises in on its +y side
            low_is_a = c.pa[2] < c.pb[2]

            def side():
                lo, up = (c.pa, c.pb) if low_is_a else (c.pb, c.pa)
                return lo[1] - up[1] > 0.6 * dab and abs(lo[2] - up[2]) < 0.6 * dab
            ok = False
            for psi in (-30, -45):
                ok = yield from until(c, psi, side, 2.5)
                if ok:
                    break
            yield from until(c, -10, lambda: False, 0.8)
            steps += 1
            c.mark(f"B{steps} ok={ok}")
            if abs(c.total) >= np.radians(215) or api.t > 40:
                break
        c.mark("end")
        yield from ramp(c, np.zeros(2), 1.2, stop=False)
        yield from dwell(c, 1.0, stop=False)
        c.mark("level")
    finally:
        c.flush()
