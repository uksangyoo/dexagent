"""Baoding by "pin and orbit" (iteration 3).

Idea: the palm is barely two balls wide and has walls only at the finger
bases, so free gravity orbiting loses balls. Instead the THUMB pins one ball
from above (pressed into the palm; iter 2 showed a ~4 N press holds a ball
still), which turns it into a post. The palm is then tilted (arm layer) so
that the free ball's downhill direction points INTO the post (radial
component keeps it in contact) plus a tangential component that walks it
around the post. Tangential tilt is closed-loop on the free ball's
tangential speed (PI). The free ball always circles in the direction that
DECREASES phi = atan2(dy, dz) of (b - a); this direction is the same
whichever ball is pinned, so roles can be swapped at will: when the free
ball's open arc ends (edge of the palm / stall), the thumb lifts, pins the
other ball, and the formerly pinned ball orbits next.

Layers: arm (palm tilt) for the orbit, fingertips (thumb pad IK + api.reach
to choose a feasible pressing direction) for the pin. Other fingers keep
their home (cupped) posture.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

C_PALM = np.array([0.040, 0.0, 0.080])   # tilt pivot (about the balls)
PAD = 0.008                               # pad centre to finger surface
DT = 0.06                                 # control chunk [s]
GOAL = 200.0                              # deg of counted rotation to aim for
REGION = (-34.0, 34.0, 26.0, 106.0)       # ball-centre y/z box (mm) where a free ball has support
NORMALS = [(0.85, 0.25, 0.45), (0.76, 0.45, 0.48), (0.9, 0.3, 0.3), (0.95, 0.3, 0.0),
           (0.85, 0.45, 0.0), (0.85, 0.0, 0.5), (0.8, 0.4, -0.25), (0.75, 0.55, 0.3),
           (0.95, 0.0, 0.3), (0.9, -0.2, 0.35)]


def tilt_R(R0, v_deg):
    """Palm rotation whose downhill direction in the palm (y, z) plane is v (deg)."""
    vy, vz = np.radians(v_deg)
    return R0 @ Rot.from_rotvec([0.0, -vz, vy]).as_matrix()


def mm(p):
    return np.round(np.asarray(p) * 1000).astype(int).tolist()


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


class Ctl:
    def __init__(self, api):
        self.api = api
        self.P0, self.R0 = api.palm_pose()
        self.cw = self.P0 + self.R0 @ C_PALM
        self.v = np.zeros(2)
        self.phi = None
        self.last = None
        self.prog = 0.0
        self.rows = []
        self.log = []
        self.k = 0
        self.th = np.asarray(api.tips()[0], float)   # current (step-limited) thumb target
        self.pos = {}
        self.vel = {"ball_a": np.zeros(3), "ball_b": np.zeros(3)}
        self.tprev = None
        self.r = 0.019
        self.tag = ""

    def obs(self):
        api = self.api
        a = api.object_in_palm("ball_a")[0]
        b = api.object_in_palm("ball_b")[0]
        if self.tprev is not None and api.t - self.tprev > 1e-4:
            dt = api.t - self.tprev
            for n, p in (("ball_a", a), ("ball_b", b)):
                v = (p - self.pos[n]) / dt
                self.vel[n] = 0.5 * self.vel[n] + 0.5 * v
        self.tprev = api.t
        self.pos = {"ball_a": a, "ball_b": b}
        Rm = api.measured_palm()[1]
        self.tilt = float(np.degrees(np.arccos(np.clip(Rm[2, 0], -1, 1))))
        d = b - a
        ang = float(np.degrees(np.arctan2(d[1], d[2])))
        held = all(-0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15 for p in (a, b))
        if self.phi is None:
            self.phi = ang
        else:
            dd = wrap(ang - self.last)
            self.phi += dd
            if held and self.tilt < 30 and api.t > 1.0:
                self.prog += dd
        self.last = ang
        self.a, self.b = a, b
        return a, b

    def step(self, g, thumb=None, extra=""):
        """One control chunk: palm downhill vector g (deg, palm y/z), thumb pad target."""
        a, b = self.obs()
        g = np.asarray(g, float).copy()
        s = np.zeros(2)
        for p in (a, b):   # edge guard: tilt balls back toward the middle
            y, z = p[1] * 1000, p[2] * 1000
            s[0] += 0.4 * (max(0, -40 - y) - max(0, y - 40))
            s[1] += 0.4 * (max(0, 18 - z) - max(0, z - 120))
        g = g + s
        n = np.linalg.norm(g)
        if n > 14:
            g *= 14 / n
        dv = g - self.v
        nd = np.linalg.norm(dv)
        if nd > 1.2:
            dv *= 1.2 / nd
        self.v = self.v + dv
        R = tilt_R(self.R0, self.v)
        xyz = self.api.palm_for(self.cw, C_PALM, R)
        if thumb is not None:
            step = np.asarray(thumb, float) - self.th
            ns = np.linalg.norm(step)
            if ns > 0.004:
                step *= 0.004 / ns
            self.th = self.th + step
        yield from self.api.move_tips(xyz, R, tips={"thumb": self.th}, seconds=DT)
        if self.k % 5 == 0:
            fa = self.api.contact_forces("ball_a")
            fb = self.api.contact_forces("ball_b")
            self.rows.append(
                f"{self.api.t:.1f}{self.tag} a{mm(a[1:])} b{mm(b[1:])} ph{self.phi:.0f} pr{self.prog:.0f} "
                f"v{np.round(self.v, 1).tolist()} T{mm(self.th)}>{mm(self.api.tips()[0])} "
                f"Fa{np.round(fa, 1).tolist()} Fb{np.round(fb, 1).tolist()}{extra}" + (" S" if np.any(s) else ""))
        self.k += 1


def press_targets(api, P, r, depth=0.006):
    """Best thumb pressing direction for a ball at P (palm frame): (press, hover, n, residual mm)."""
    best = None
    for nn in NORMALS:
        n = unit(nn)
        press = P + (r + PAD - depth) * n
        hover = P + (r + PAD + 0.018) * n
        e = 1000 * (api.reach("thumb", press) + 0.5 * api.reach("thumb", hover)) + 2.0 * (1 - n[0])
        if best is None or e < best[3]:
            best = (press, hover, n, e)
    return best


def open_arc(P, F, r):
    """How far (deg) the free ball F can circle P (decreasing angle) staying in REGION."""
    rel = (F - P)[1:]
    al = np.degrees(np.arctan2(rel[0], rel[1]))
    R = 1000 * (2 * r + 0.001)
    arc = 0
    for k in range(1, 25):
        aa = np.radians(al - 10 * k)
        y = 1000 * P[1] + R * np.sin(aa)
        z = 1000 * P[2] + R * np.cos(aa)
        if not (REGION[0] <= y <= REGION[1] and REGION[2] <= z <= REGION[3]):
            break
        arc = 10 * k
    return arc


def run(api):
    ctl = Ctl(api)
    try:
        yield from main(api, ctl)
    finally:
        log = ctl.log
        rows = [" ; ".join(ctl.rows[i:i + 3]) for i in range(0, len(ctl.rows), 3)]
        keep = log[-22:] + rows[-34:]
        keep.append(f"end t={api.t:.2f} phi={ctl.phi:.0f} prog={ctl.prog:.0f}")
        for line in keep:
            api.note(line)


def hold_g(ctl):
    """Gentle tilt toward the finger trough (the only safe direction)."""
    return np.array([0.0, 1.5])


def pin(api, ctl, name, other, depth=0.006):
    """Thumb pins ball `name` from above. Returns the pin state dict."""
    log = ctl.log
    P = ctl.pos[name]
    press, hover, n, e = press_targets(api, P, ctl.r, depth)
    log.append(f"pin {name} at {mm(P)} n={np.round(n, 2).tolist()} score={e:.1f} t={api.t:.2f}")
    # lift first if the thumb is low, then go to hover, then press
    lift = ctl.th + np.array([0.02, 0.0, 0.0]) if ctl.th[0] < 0.07 else ctl.th
    t0 = api.t
    while api.t - t0 < 0.4 and np.linalg.norm(ctl.th - lift) > 0.002:
        yield from ctl.step(hold_g(ctl), lift)
    t0 = api.t
    while api.t - t0 < 2.0:
        P = ctl.pos[name]
        hover = P + (ctl.r + PAD + 0.018) * n
        if np.linalg.norm(ctl.th - hover) < 0.002 and api.t - t0 > 0.3:
            break
        yield from ctl.step(hold_g(ctl), hover)
    t0 = api.t
    st = {"name": name, "n": n, "depth": depth, "P": ctl.pos[name].copy()}
    while api.t - t0 < 1.5:
        P = ctl.pos[name]
        st["press"] = P + (ctl.r + PAD - st["depth"]) * n
        yield from ctl.step(hold_g(ctl), st["press"])
        f = api.contact_forces(name)[0]
        if api.t - t0 > 0.6 and f > 0.8:
            break
        if api.t - t0 > 1.0 and f < 0.3 and st["depth"] < 0.010:
            st["depth"] += 0.002
    st["P"] = ctl.pos[name].copy()
    log.append(f"pinned {name} P={mm(st['P'])} F={np.round(api.contact_forces(name), 1).tolist()} "
               f"tip={mm(api.tips()[0])} other={mm(ctl.pos[other])} t={api.t:.2f}")
    return st


def orbit(api, ctl, st, free, tmax=14.0):
    """Tilt so the free ball circles the pinned one (decreasing angle). Returns the exit reason."""
    log = ctl.log
    name = st["name"]
    r = ctl.r
    vd = 0.014                 # desired tangential speed [m/s]
    ki = 1.5                   # integral tangential tilt [deg]
    t0 = api.t
    hist = []
    al_un = None
    al_prev = None
    reason = "timeout"
    while api.t - t0 < tmax:
        P, F = ctl.pos[name], ctl.pos[free]
        # keep the pin on the ball if it creeps
        if np.linalg.norm((P - st["P"])[1:]) > 0.004:
            st["P"] = P.copy()
            st["press"] = P + (r + PAD - st["depth"]) * st["n"]
        rel = (F - P)[1:]
        d = np.linalg.norm(rel)
        u = rel / (d + 1e-9)
        al = float(np.degrees(np.arctan2(u[0], u[1])))
        al_un = al if al_un is None else al_un + wrap(al - al_prev)
        al_prev = al
        tvec = np.array([-u[1], u[0]])
        vF = ctl.vel[free][1:]
        vt = float(vF @ tvec)
        e = vd - vt
        ki = float(np.clip(ki + 25.0 * e * DT * 10, 0.0, 6.0))
        kt = float(np.clip(ki + 120.0 * e, -4.0, 7.0))
        gap = 1000 * (d - 2 * r)
        kr = 2.5 + float(np.clip(0.25 * gap, 0.0, 5.0))
        if gap > 4:
            kt = min(kt, 1.5)
        g = -kr * u + kt * tvec
        hist.append((api.t, al_un))
        yield from ctl.step(g, st["press"], extra=f" al{al:.0f} gap{gap:.0f} vt{1000 * vt:.0f} kt{kt:.1f} kr{kr:.1f}")
        if abs(ctl.prog) >= GOAL:
            reason = "done"
            break
        # stop when the free ball reaches the end of its open arc
        if open_arc(P, F, r) <= 0 and api.t - t0 > 1.0:
            reason = "arc_end"
            break
        old = [h for h in hist if h[0] < api.t - 3.0]
        if old and api.t - t0 > 4.0 and old[-1][1] - al_un < 4.0 and ki > 5.5:
            reason = "stall"
            break
    log.append(f"orbit {free} about {name}: {reason} t={api.t:.2f} al_un={al_un:.0f} "
               f"a={mm(ctl.a)} b={mm(ctl.b)} prog={ctl.prog:.0f}")
    return reason


def main(api, ctl):
    log = ctl.log
    a, b = ctl.obs()
    log.append(f"start a={mm(a)} b={mm(b)} thumb={mm(api.tips()[0])}")
    park = np.array([0.075, 0.060, 0.095])
    # thumb reach probe (pure IK, no sim time): residual mm on a y/z grid at x = 60 mm
    rows = []
    for y in (-20, -10, 0, 10, 20, 30, 40):
        rows.append(f"y{y}:" + ",".join(str(int(round(1000 * api.reach('thumb', np.array([0.060, y / 1000, z / 1000])))))
                                         for z in range(40, 131, 10)))
    log.append("thumb reach x60 z40..130 " + " ".join(rows))

    # P0: let the balls drop and settle; slight tilt toward the wrist keeps a low
    while api.t < 0.98:
        yield from ctl.step(np.array([0.0, -3.0]), park)
    ctl.r = float(np.clip(ctl.b[0] - 0.021, 0.016, 0.023))
    log.append(f"t1 a={mm(ctl.a)} b={mm(ctl.b)} phi={ctl.phi:.0f} r={ctl.r * 1000:.1f}")

    st = None
    for phase in range(5):
        if abs(ctl.prog) >= GOAL or api.t > 47:
            break
        names = ("ball_a", "ball_b")
        cands = []
        for P, F in (names, names[::-1]):
            if st is not None and P == st["name"] and phase > 0:
                continue   # the previously pinned ball now orbits
            press, hover, n, e = press_targets(api, ctl.pos[P], ctl.r)
            arc = open_arc(ctl.pos[P], ctl.pos[F], ctl.r)
            cands.append((arc - 4 * max(0.0, e - 2.0), P, F, arc, e))
        if phase == 0:
            # first phase: both roles allowed
            pass
        cands.sort(reverse=True)
        log.append("roles " + " | ".join(f"pin {c[1][-1]} arc{c[3]} res{c[4]:.1f}" for c in cands))
        _, P, F, arc, e = cands[0]
        if st is not None:
            # release the old pin: lift the thumb straight off
            t0 = api.t
            up = st["press"] + 0.022 * st["n"]
            while api.t - t0 < 0.4:
                yield from ctl.step(hold_g(ctl), up)
        ctl.tag = f"P{phase}"
        st = yield from pin(api, ctl, P, F)
        reason = yield from orbit(api, ctl, st, F)
        if reason == "done":
            break

    # finish: keep the pin, free ball nestled against it, palm near level
    ctl.tag = "E"
    t0 = api.t
    while api.t - t0 < 1.5:
        if st is not None:
            F = "ball_b" if st["name"] == "ball_a" else "ball_a"
            u = unit((ctl.pos[F] - ctl.pos[st["name"]])[1:])
            yield from ctl.step(-2.0 * u, st["press"])
        else:
            yield from ctl.step(hold_g(ctl), park)
