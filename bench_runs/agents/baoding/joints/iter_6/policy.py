"""baoding, joints level: eject - pass - rise, driven by palm tilt (fingers at home).

The cupped home hand has a deep groove at the finger bases (palm z ~ 95-103):
both balls sit in it side by side and it holds them against ~13 deg of tilt,
so the groove is a 1-D channel and the flat palm below it (z < 85) is the
only passing lane. One swap (pair turns ~180 deg):
  EJECT  tilt toward the wrist (ramped) until ONE ball pops out of the groove
         onto the flat palm (E); the other (K) stays in the groove.
  PASS   tilt sideways toward K's side: E rolls under K to K's far side.
  RISE   tilt toward fingers and away from that side: K slides along the
         groove the other way, E rises into the groove on the far side.
Repeat with the same pass direction for a second swap, then level.
Guards push the downhill direction away from open edges, especially the
index-thumb gap at the +y end of the groove (the escape route seen in iter 4).
iter 6: before t = 1 s the thumb is swung toward the index (direction chosen
online from the measured thumb pad) to close that gap, ejection is steered
slightly to -y, only the right ball may be ejected (pass to the -y side, under
the left ball), and guards are capped at 6 deg.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DT = 0.02
PC = np.array([0.045, 0.0, 0.085])      # palm-frame pivot for tilting
TMAX = 18.0                             # max tilt [deg]


def held(p):
    return -0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15


def palm_R(R0, tau):
    """Palm rotation whose in-plane downhill direction (palm y, z) is tau; |tau| = tilt [rad]."""
    th = float(np.linalg.norm(tau))
    if th < 1e-9:
        return R0.copy()
    uy, uz = tau / th
    return R0 @ Rot.from_rotvec(np.array([0.0, -uz, uy]) * th).as_matrix()


def tvec(lam_deg, tilt_deg):
    """Downhill vector [deg] in palm (y, z); lam 0 = toward fingers, 90 = toward thumb (+y), 180 = wrist."""
    l = np.radians(lam_deg)
    return tilt_deg * np.array([np.sin(l), np.cos(l)])


def mm(p):
    return np.round(p * 1000).astype(int).tolist()


class Ctl:
    def __init__(self, api):
        self.api = api
        self.x0, self.R0 = api.palm_pose()
        self.W = self.x0 + self.R0 @ PC
        self.tau = np.zeros(2)           # current tilt vector [deg]
        self.q = api.q_target().copy()
        self.total = 0.0                 # replica of the scorer's signed rotation [rad]
        self.last = None
        self.trace = []
        self.events = []
        self.k = 0
        self.state = "init"
        self.lam = 0.0

    def balls(self):
        a, _ = self.api.object_in_palm("ball_a")
        b, _ = self.api.object_in_palm("ball_b")
        return a, b

    def track(self):
        a, b = self.balls()
        d = b - a
        ang = float(np.arctan2(d[1], d[2]))
        ok = held(a) and held(b)
        if ok and self.last is not None and self.api.t > 1.0:
            self.total += (ang - self.last + np.pi) % (2 * np.pi) - np.pi
        self.last = ang if ok else None
        return a, b

    @property
    def prog(self):
        return abs(np.degrees(self.total))

    def ev(self, tag):
        a, b = self.balls()
        self.events.append(f"{self.api.t:.2f} {tag}: a={mm(a)} b={mm(b)} tot={np.degrees(self.total):.0f} "
                           f"lam={self.lam:.0f} tilt={np.round(self.tau, 1).tolist()}")

    def step_q(self, g_deg, q, rate=20.0):
        self.q = np.asarray(q, float)
        yield from self.step(g_deg, rate)

    def step(self, g_deg, rate=20.0):
        """One 20 ms step: tilt vector [deg] toward g_deg, rate-limited [deg/s]; fingers keep their targets."""
        g = np.asarray(g_deg, float)
        n = np.linalg.norm(g)
        if n > TMAX:
            g = g * TMAX / n
        s = g - self.tau
        sn = np.linalg.norm(s)
        if sn > rate * DT:
            s *= rate * DT / sn
        self.tau = self.tau + s
        R = palm_R(self.R0, np.radians(self.tau))
        yield from self.api.move(xyz=self.W - R @ PC, R=R, q=self.q, seconds=DT)
        a, b = self.track()
        if self.k % 10 == 0:
            self.trace.append([round(self.api.t, 1), self.state[:4], mm(a), mm(b),
                               int(round(np.degrees(self.total))), int(round(self.lam)),
                               np.round(self.tau, 1).tolist()])
        self.k += 1


def guard(a, b):
    """Extra downhill vector [deg] pushing balls away from open edges."""
    g = np.zeros(2)
    for p in (a, b):
        if p[2] < 0.050:
            g[1] += (0.050 - p[2]) * 800
        ylim = 0.025 if p[2] > 0.082 else 0.036     # index-thumb gap at the +y end of the groove
        if p[1] > ylim:
            g[0] -= (p[1] - ylim) * 1200
        if p[1] < -0.042:
            g[0] += (-0.042 - p[1]) * 500
    n = np.linalg.norm(g)
    return g if n <= 6.0 else g * 6.0 / n


def run(api):
    c = Ctl(api)
    status = "timeout"
    # balls drop and settle; pre-load into the groove (a left, b right) before counting starts at t = 1 s
    home = api.q_target().copy()
    tip0 = api.tips()[0].copy()
    thumb = {"q": home.copy(), "dir": +1}

    def thumb_q(delta):
        q = home.copy()
        q[1] = home[1] + delta
        return q

    while api.t < 1.0:
        c.lam = 40.0
        if api.t < 0.45:
            qt = thumb_q(0.30)
        elif thumb["dir"] == +1 and api.t < 0.47:
            tip = api.tips()[0]
            c.events.append(f"thumb +0.30: tip {mm(tip0)} -> {mm(tip)}")
            if tip[1] > tip0[1] - 0.005:          # did not move toward the index: go the other way
                thumb["dir"] = -1
            qt = thumb_q(0.30 * thumb["dir"])
        else:
            qt = thumb_q(0.30 * thumb["dir"])
        yield from c.step_q(tvec(40, 9) if api.t > 0.3 else np.zeros(2), qt)
    c.events.append(f"thumb dir {thumb['dir']}: tip now {mm(api.tips()[0])}")
    c.ev("t1")

    st = {"name": "SETTLE", "t": api.t, "ok": None, "sub": 0}
    lam, tilt, rate = 40.0, 6.0, 20.0
    E = K = None          # ball indices (0 = a, 1 = b): ejected / kept
    s_first = None        # pass direction of the first swap (+1 toward +y)
    s = 0
    ej_tilt = 6.0
    ej_lam = -165.0
    swaps = 0
    tries = 0

    def go(name):
        st.update(name=name, t=api.t, ok=None, sub=0)
        c.state = name
        c.ev("-> " + name)

    def stable(cond, need=0.3):
        if cond:
            if st["ok"] is None:
                st["ok"] = api.t
            return api.t - st["ok"] >= need
        st["ok"] = None
        return False

    try:
        while api.t < 54.0:
            a, b = c.balls()
            P = (a, b)
            if not (held(a) and held(b)):
                status = "dropped"
                c.ev("DROPPED")
                break
            el = api.t - st["t"]
            name = st["name"]
            in_groove = [p[2] > 0.088 for p in P]
            rate = 20.0

            if name == "SETTLE":                 # both in the groove, side by side
                lam, tilt = 20.0, 6.0
                if stable(all(in_groove) and abs(a[1] - b[1]) > 0.028, 0.4) or el > 3.0:
                    tilt = ej_tilt = max(6.0, ej_tilt - 2.0)
                    go("EJECT")
            elif name == "EJECT":
                lam = ej_lam
                ej_tilt = min(ej_tilt + 2.5 * DT, TMAX)
                tilt = ej_tilt
                out = [p[2] < 0.084 for p in P]
                if all(out):
                    c.ev("both ejected")
                    tries += 1
                    go("RECOVER")
                elif any(out):
                    E = 0 if out[0] else 1
                    K = 1 - E
                    s = 1 if P[K][1] > P[E][1] else -1
                    if s == 1:                         # left ball out: its pass would run into the +y ramp
                        c.ev("left ball ejected; recover")
                        tries += 1
                        go("RECOVER")
                    elif s_first is not None and s != s_first:
                        c.ev("wrong ball ejected")
                        tries += 1
                        ej_lam = -165.0 - 20.0 * s_first      # bias downhill toward the side of the ball to keep
                        go("RECOVER")
                    else:
                        c.ev(f"ejected {'ab'[E]} at tilt {ej_tilt:.1f}, pass s={s}")
                        go("PASS")
                elif el > 6.0:
                    c.ev("no eject")
                    tries += 1
                    go("RECOVER")
            elif name == "PASS":
                rate = 60.0 if el < 0.6 else 25.0
                lam, tilt = s * 112.0, 8.0
                Ev, Kv = P[E], P[K]
                if s * (Ev[1] - Kv[1]) > 0.004 and Ev[2] < Kv[2] - 0.015:
                    go("RISE")
                elif Ev[2] > 0.090 and el > 1.0:
                    c.ev("E back in groove")
                    tries += 1
                    go("SETTLE")
                elif el > 6.0:
                    tries += 1
                    go("RECOVER")
            elif name == "RISE":
                lam, tilt = -s * 50.0, 8.0
                Ev, Kv = P[E], P[K]
                if stable(Ev[2] > 0.086 and Kv[2] > 0.086 and s * (Ev[1] - Kv[1]) > 0.025, 0.3):
                    swaps += 1
                    if s_first is None:
                        s_first = s
                    c.ev(f"swap {swaps} done")
                    if c.prog >= 330 or (c.prog >= 200 and api.t > 40.0):
                        status = "goal"
                        break
                    go("SETTLE")
                elif el > 6.0:
                    go("SETTLE")
            elif name == "RECOVER":              # bring both back into the groove
                lam, tilt = 0.0, 10.0
                rate = 60.0 if el < 0.6 else 20.0
                if stable(all(in_groove), 0.5) or el > 4.0:
                    go("SETTLE")
            c.lam = lam
            g = tvec(lam, tilt) + guard(a, b)
            yield from c.step(g, rate)
    except Exception as e:  # keep diagnostics if the arm IK fails
        status = f"exception {e!r}"
    c.ev(f"end loop status={status}")
    # level slowly; the natural slope keeps the balls in the groove
    c.state = "LEVEL"
    for _ in range(120):
        a, b = c.balls()
        if np.linalg.norm(c.tau) < 0.05:
            break
        yield from c.step(np.zeros(2), 12.0)
    yield from api.hold(0.3)
    c.ev("end")
    for line in c.events[-40:]:
        api.note(line)
    tr = c.trace
    for i in range(0, len(tr), 10):
        api.note("trace [t, st, a_xyz, b_xyz, tot, lam, tilt_yz] " + repr(tr[i:i + 10]))
    api.note(f"END status={status} total={np.degrees(c.total):.0f} swaps={swaps} tries={tries}")
