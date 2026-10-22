"""Baoding, force level (phase 3, with memory of earlier agents).

Plan "pin-orbit, drag, roll-under":
  P0 (t<1)  balls drop; slight wrist-ward tilt keeps ball_a out of the finger groove
            (layered agent's settle).
  P1        THUMB pins the upper ball (b) from above; palm tilt walks the free ball (a)
            under it toward the thumb side, where it stalls against the thumb's links
            (layered agent's pin()/orbit(), proven 3/3 to about -80..-120 deg).
  P2        new: the thumb DRAGS the pinned ball left and down along the thumb's
            reachable band, its pad behind the ball, pressing with the grip-force
            servo (direction re-issued each chunk in palm terms); palm tilt toward +y/+z
            lets a rise into the right pocket that b vacated.
  P3        if still short of the goal: thumb lifts straight off, gentle +y tilt rolls
            b under a toward the thumb-link corner (same stall spot as P1).
  END       hold with the palm near level.
Rotation is tracked online with a replica of the scorer.

Helpers reused from bench_runs/agents/baoding/layered (tilt_R, Ctl, press_targets,
pin, orbit, open_arc), lightly adapted.
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
DRAG_PATH = [(-12.0, 96.0), (-20.0, 83.0), (-21.0, 70.0)]   # b centre waypoints (mm, palm y/z)
DRAG_F = 2.0                               # N, grip servo target while dragging
DRAG_V = 0.012                             # m/s reference speed


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
        self.thstep = 0.004
        self.dropped = False
        self.other = {}     # extra fingertip targets {finger: palm xyz} held every chunk
        self.other_cur = {}

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
        if not held:
            self.dropped = True
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
            if ns > self.thstep:
                step *= self.thstep / ns
            self.th = self.th + step
        tips = {"thumb": self.th}
        for f, tgt in self.other.items():   # rate-limited extra finger targets (6 mm per chunk)
            cur = self.other_cur.get(f)
            if cur is None:
                cur = np.asarray(self.api.tips()[FINGERS.index(f)], float)
            stp = np.asarray(tgt, float) - cur
            ns = np.linalg.norm(stp)
            if ns > 0.006:
                stp *= 0.006 / ns
            self.other_cur[f] = cur + stp
            tips[f] = self.other_cur[f]
        yield from self.api.move_tips(xyz, R, tips=tips, seconds=DT)
        if self.k % 5 == 0:
            fa = self.api.contact_forces("ball_a")
            fb = self.api.contact_forces("ball_b")
            self.rows.append(
                f"{self.api.t:.1f}{self.tag} a{mm(a[1:])} b{mm(b[1:])} pr{self.prog:.0f} "
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
        keep = log[-20:] + rows[-30:]
        keep.append(f"end t={api.t:.2f} phi={ctl.phi:.0f} prog={ctl.prog:.0f} dropped={ctl.dropped}")
        for line in keep:
            api.note(line)


def hold_g(ctl):
    """Gentle tilt toward the finger trough (the only safe direction)."""
    return np.array([0.0, 1.5])


def pin(api, ctl, name, other, depth=0.006):
    """Thumb pins ball `name` from above (position press). Returns the pin state dict."""
    log = ctl.log
    P = ctl.pos[name]
    press, hover, n, e = press_targets(api, P, ctl.r, depth)
    log.append(f"pin {name} at {mm(P)} n={np.round(n, 2).tolist()} score={e:.1f} t={api.t:.2f}")
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


def orbit(api, ctl, st, free, tmax=12.0, thumb=None, hook=None):
    """Tilt so the free ball circles the pinned one (decreasing angle). Returns the exit reason."""
    log = ctl.log
    name = st["name"]
    r = ctl.r
    vd = 0.014
    ki = 1.5
    t0 = api.t
    hist = []
    al_un = None
    al_prev = None
    reason = "timeout"
    while api.t - t0 < tmax:
        P, F = ctl.pos[name], ctl.pos[free]
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
        if hook is not None:
            yield from hook()
        yield from ctl.step(g, st["press"] if thumb is None else thumb, extra=f" al{al:.0f} gap{gap:.0f} kt{kt:.1f}")
        if abs(ctl.prog) >= GOAL:
            reason = "done"
            break
        if open_arc(P, F, r) <= 0 and api.t - t0 > 1.0:
            reason = "arc_end"
            break
        old = [h for h in hist if h[0] < api.t - 2.5]
        if old and api.t - t0 > 3.5 and old[-1][1] - al_un < 4.0 and ki > 5.5:
            reason = "stall"
            break
    log.append(f"orbit {free} about {name}: {reason} t={api.t:.2f} al_un={al_un:.0f} "
               f"a={mm(ctl.a)} b={mm(ctl.b)} prog={ctl.prog:.0f}")
    return reason


FINGER_PIN_NORMALS = [(1.0, 0.0, 0.0), (0.95, 0.0, 0.3), (0.95, 0.3, 0.0), (0.95, -0.3, 0.0), (0.9, 0.3, 0.3),
                      (0.95, 0.0, -0.3), (0.85, 0.0, 0.5)]
FPIN_N = 2.0     # grip-servo force for the finger pin [N]


def finger_pin(api, ctl, finger, name, fingers=("index", "middle")):
    """A long finger presses ball `name` into the palm from above; grip servo holds FPIN_N.
    Picks finger and pad normal by api.reach (pad + hover). Returns (finger, normal) or None."""
    log = ctl.log
    r = ctl.r
    P = ctl.pos[name]
    best = None
    for f in fingers:
        for nn in FINGER_PIN_NORMALS:
            n = unit(nn)
            e = api.reach(f, P + (r + PAD) * n) + 0.5 * api.reach(f, P + (r + PAD + 0.02) * n)
            if best is None or e < best[0]:
                best = (e, f, n)
    e, f, n = best
    log.append(f"fpin {name} at {mm(P)} with {f} n={np.round(n, 2).tolist()} res={e * 1000:.1f}")
    if e > 0.008:
        return None
    t0 = api.t
    g_hold = np.zeros(2)
    while api.t - t0 < 2.0:      # to hover
        P = ctl.pos[name]
        ctl.other[f] = P + (r + PAD + 0.02) * n
        yield from ctl.step(g_hold, ctl.th)
        if api.t - t0 > 0.5 and np.linalg.norm(ctl.other_cur[f] - ctl.other[f]) < 0.001 and \
                np.linalg.norm(api.tips()[FINGERS.index(f)] - ctl.other[f]) < 0.006:
            break
    for h in (0.014, 0.008, 0.003, 0.0):   # descend onto the top
        P = ctl.pos[name]
        ctl.other[f] = P + (r + PAD + h) * n
        yield from ctl.step(g_hold, ctl.th)
    _, Ro = api.object_in_palm(name)
    g = yield from api.grip(name, {f: FPIN_N}, directions={f: Ro.T @ (-n)}, seconds=0.5, max_travel=0.012)
    log.append(f"fpinned {name} t={api.t:.2f} grip={g} tips={mm(api.tips()[FINGERS.index(f)])} "
               f"F{name[-1]}={np.round(api.contact_forces(name), 1).tolist()}")
    return f, n


FINGERS = ("thumb", "index", "middle", "ring", "little")


def hold_until(api, ctl, g, thumb, secs, cond=None):
    t0 = api.t
    while api.t - t0 < secs:
        yield from ctl.step(np.asarray(g, float), thumb)
        if cond is not None and cond():
            break


def main(api, ctl):
    log = ctl.log
    a, b = ctl.obs()
    log.append(f"start a={mm(a)} b={mm(b)} thumb={mm(api.tips()[0])}")
    park = np.array([0.075, 0.060, 0.095])

    # P0: drop and settle; slight tilt toward the wrist keeps a low
    while api.t < 0.98:
        yield from ctl.step(np.array([0.0, -3.0]), park)
    ctl.r = float(np.clip(ctl.b[0] - 0.021, 0.016, 0.023))
    log.append(f"t1 a={mm(ctl.a)} b={mm(ctl.b)} phi={ctl.phi:.0f} r={ctl.r * 1000:.1f}")

    # P1: thumb pins the upper ball, the other orbits under it to the thumb-link corner
    upper, lower = ("ball_b", "ball_a") if ctl.b[2] >= ctl.a[2] else ("ball_a", "ball_b")
    ctl.tag = "P1"
    st = yield from pin(api, ctl, upper, lower)
    reason = yield from orbit(api, ctl, st, lower)

    # P2: a long finger pins the lower ball where it stalled; thumb lifts; the upper ball orbits round it
    P, F = ctl.pos[upper], ctl.pos[lower]
    if abs(ctl.prog) < GOAL and F[2] < P[2] - 0.015 and api.t < 40:
        ctl.tag = "P2"
        res = yield from finger_pin(api, ctl, "index", lower)
        if res is not None:
            f, n = res
            up = st["press"] + 0.022 * st["n"]
            yield from hold_until(api, ctl, np.zeros(2), up, 0.5)
            log.append(f"thumb up t={api.t:.2f} a={mm(ctl.a)} b={mm(ctl.b)} Fthumb_on_lower="
                       f"{api.contact_forces(lower)[0]:.1f} grip={api.grip_state()}")
            st2 = {"name": lower, "n": n, "depth": 0.0, "P": ctl.pos[lower].copy(), "press": up}
            cnt = {"k": 0}

            def regrip():
                cnt["k"] += 1
                if cnt["k"] % 8 == 0:
                    _, Ro = api.object_in_palm(lower)
                    yield from api.grip(lower, {f: FPIN_N}, directions={f: Ro.T @ (-n)}, seconds=0.02,
                                        max_travel=0.012)

            ctl.tag = "P2o"
            reason = yield from orbit(api, ctl, st2, upper, tmax=16.0, thumb=up, hook=regrip)
            log.append(f"P2 orbit {reason} grip={api.grip_state()}")

    # END: lean the free ball on the pinned one, palm near level
    ctl.tag = "E"
    t0 = api.t
    while api.t - t0 < 1.2:
        u = unit((ctl.pos[lower] - ctl.pos[upper])[1:])
        yield from ctl.step(1.5 * u, ctl.th)
    log.append(f"final t={api.t:.2f} a={mm(ctl.a)} b={mm(ctl.b)} prog={ctl.prog:.0f} tilt={ctl.tilt:.1f} "
               f"grip={api.grip_state()}")
