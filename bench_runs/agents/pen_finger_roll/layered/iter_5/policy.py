"""pen_finger_roll, layered level.

Plan: palm-down thumb-index pinch near the pen's top (pinch line along palm z, pen axis
along palm x), lift straight out of the cup, then with the fresh grip swing the pen about
palm y (contacts.rotate about palm -y through the grip point), closed loop on the measured
tilt: +side, -side, back to rest and hold (settles the scorer). Last, roll the pen about its
own axis by orbiting the two pads around it (contacts.rotate about the pen axis) and just
hold there (no roll-back: moving the pads back loosened the grip in iters 1-2).
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE_Z = 0.75
CUP_TOP = TABLE_Z + 0.06
PC = np.array([0.080, 0.040, 0.118])     # thumb-index pinch centre, palm frame (hand.py)
EZ = np.array([0.0, 0.0, 1.0])
AY = np.array([0.0, -1.0, 0.0])          # rotate about -y: +angle -> +tilt (rigid model)
FING = ["thumb", "index"]
R_PEN = 0.00375
CURL = np.array([0, 1.6, 1.3, 1.1])
GRIP_BELOW_TOP = 0.028
SWING_GOAL = 16.5                        # deg each side (check needs +-10, range 30)
ROLL_GOAL = 50.0                         # deg of roll range (check needs 30)


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _tilt(a):
    return float(np.arctan2(a[2], a[0]))


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class Mon:
    """Mirror of the scorer's bookkeeping, from the api's observations."""

    def __init__(self, api):
        self.api = api
        p, _ = api.object_pose("pen")
        self.reach = p[2] - TABLE_Z
        self.out = False
        self.rest = None
        self.R_last = None
        self.twist = self.tw_lo = self.tw_hi = 0.0
        self.tilt = self.t_lo = self.t_hi = 0.0
        self.calm = 0
        self.settled = False
        self.bottom = 0.0

    def update(self):
        api = self.api
        p, Rw = api.object_pose("pen")
        _, Rr = api.object_in_palm("pen")
        self.bottom = p[2] - self.reach * abs(Rw[2, 2])
        if not self.out and self.bottom > CUP_TOP + 0.01:
            self.out = True
            self.rest = Rr[:, 2].copy()
            self.R_last = Rr
            api.note(f"pen cleared the rim; rest axis in palm {np.round(self.rest, 3).tolist()}")
        if not self.out:
            return
        if self.bottom > CUP_TOP:
            self.twist += _twist(self.R_last, Rr)
            self.tw_lo, self.tw_hi = min(self.tw_lo, self.twist), max(self.tw_hi, self.twist)
            self.tilt = np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(self.rest)))
            self.t_lo, self.t_hi = min(self.t_lo, self.tilt), max(self.t_hi, self.tilt)
            if self.t_hi >= 10 and self.t_lo <= -10 and abs(self.tilt) < 5:
                self.calm += 1
                self.settled |= self.calm >= 25
            else:
                self.calm = 0
        self.R_last = Rr

    @property
    def roll(self):
        return np.degrees(self.tw_hi - self.tw_lo)

    def s(self):
        return (f"tilt={self.tilt:.1f} [{self.t_lo:.1f},{self.t_hi:.1f}] roll_range={self.roll:.1f} "
                f"bottom_above_rim={1000 * (self.bottom - CUP_TOP):.0f}mm settled={self.settled}")


def run(api):
    mon = Mon(api)

    def drive(gen, stop=None):
        """Run an api motion, updating the monitor every step; stop early when stop() is true."""
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            yield
            mon.update()
            if stop is not None and stop():
                gen.close()
                return "stopped"

    def geom(Rd):
        p, R = api.object_pose("pen")
        up = 1.0 if R[2, 2] >= 0 else -1.0
        g_obj = np.array([0.0, 0.0, up * (mon.reach - GRIP_BELOW_TOP)])
        g_world = p + R @ g_obj
        d = R.T @ Rd[:, 2]
        d[2] = 0.0
        d /= np.linalg.norm(d)
        return g_obj, g_world, d

    def beta():
        t = api.tip_targets()
        v = t[1] - t[0]
        return float(np.degrees(np.arctan2(v[0], v[2])))

    def forces():
        f = api.contact_forces("pen")
        return np.round(f, 2).tolist()

    def regrip(target=1.3, tries=3, step=0.0008):
        for _ in range(tries):
            f = api.contact_forces("pen")
            if min(f[0], f[1]) >= target:
                break
            t = api.tip_targets()
            u = (t[1] - t[0]) / np.linalg.norm(t[1] - t[0])
            yield from drive(api.move_tips(tips={"thumb": t[0] + step * u, "index": t[1] - step * u},
                                           seconds=0.3, straight=True))
            yield from drive(api.hold(0.15))
        api.note(f"regrip -> forces {forces()}")

    trace = []

    def rec():
        if int(round(api.t * 50)) % 5 == 0:
            pp, _ = api.object_in_palm("pen")
            f = api.contact_forces("pen")
            tp = np.round(api.tips()[:2] * 1000).astype(int)
            trace.append(f"{api.t:.1f}:{mon.tilt:.0f}/{f[0]:.1f},{f[1]:.1f}/{np.round(pp * 1000).astype(int).tolist()}"
                         f"/th{tp[0][[0, 2]].tolist()}ix{tp[1][[0, 2]].tolist()}/off{1000 * np.linalg.norm(grip_offset()):.1f}")
        return False

    Rd = api.hand_down()
    # ---- 1. pre-shape: thumb/index open pinch, the rest tucked; palm above the pen
    q = api.OPEN.copy()
    for i in (2, 3):
        q[4 * i:4 * i + 4] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    yield from drive(api.move_joints(q=q, seconds=0.5))
    g_obj, gw, d = geom(Rd)
    pre = api.palm_for(gw + np.array([0, 0, 0.07]), PC, Rd)
    open_tips = {"thumb": PC - 0.022 * EZ, "index": PC + 0.022 * EZ}
    r = yield from drive(api.move_tips(pre, Rd, tips=open_tips, seconds=2.0))
    api.note(f"preshape ik {r}; tips {np.round(api.tips() * 1000).astype(int).tolist()}")

    # ---- 2. descend around the pen and pinch it
    g_obj, gw, d = geom(Rd)
    yield from drive(api.carry(api.palm_for(gw, PC, Rd), Rd, seconds=1.5))
    yield from drive(api.hold(0.2))
    g_obj, gw, d = geom(Rd)
    pts = {"thumb": g_obj - R_PEN * d, "index": g_obj + R_PEN * d}
    nrm = {"thumb": -d, "index": d}
    r = yield from drive(api.grasp("pen", pts, squeeze=0.0035, seconds=1.0, normals=nrm))
    yield from drive(api.hold(0.3))
    f = forces()
    api.note(f"grasp {r}; forces now {f}")
    yield from regrip(1.3)

    # ---- 3. lift straight out of the cup
    P, _ = api.palm_pose()
    yield from drive(api.carry(P + np.array([0, 0, 0.15]), Rd, seconds=2.0))
    yield from drive(api.hold(0.5))
    pp, Rr = api.object_in_palm("pen")
    api.note(f"lifted: {mon.s()} pen in palm {np.round(pp * 1000).astype(int).tolist()} axis "
             f"{np.round(Rr[:, 2], 3).tolist()} forces {forces()} tips {np.round(api.tips() * 1000).astype(int).tolist()}")
    if not mon.out:
        api.note("pen not out of the cup; aborting in-hand phase")
        return

    g_obj = np.array([0.0, 0.0, g_obj[2]])
    # grip point in object frame, re-estimated from the pads (midpoint of the pinch)
    pp, Rr = api.object_in_palm("pen")
    mid = api.tips()[:2].mean(axis=0)
    g_obj = np.array([0.0, 0.0, float(Rr[:, 2] @ (mid - pp))])

    def lost():
        return mon.bottom < CUP_TOP + 0.02


    squeezed = {"mm": 0.0}
    shift = {"v": np.zeros(3)}

    def grip_offset():
        """Pen axis at grip height minus the measured pad midpoint, keeping only the component
        normal to both the pinch line and the pen axis (the direction a convex-convex pinch
        squirts the pen out along)."""
        pp, Rr = api.object_in_palm("pen")
        a = Rr[:, 2]
        tp = api.tips()[:2]
        mid = tp.mean(axis=0)
        u = (tp[1] - tp[0]) / np.linalg.norm(tp[1] - tp[0])
        grip = pp + a * (a @ (mid - pp))
        n = np.cross(u, a)
        n /= np.linalg.norm(n)
        return float((grip - mid) @ n) * n

    def centre_pinch(tol=0.0004, lim=0.0015):
        """Skill: re-centre the two pads on the pen (shift both pad targets by the lateral offset)."""
        e = grip_offset()
        if np.linalg.norm(e) > tol:
            e = e * min(1.0, lim / np.linalg.norm(e))
            if np.linalg.norm(shift["v"] + e) > 0.006:   # never chase the pen more than 6 mm in total
                return
            shift["v"] = shift["v"] + e
            t = api.tip_targets()
            yield from drive(api.move_tips(tips={"thumb": t[0] + e, "index": t[1] + e}, seconds=0.12,
                                           straight=True), stop=lambda: rec() or False)

    def keep_grip(target=1.0, step=0.0006, cap=0.004):
        """Force servo: re-centre the pinch first, then if either pad force sags below `target`,
        close the pinch along the (commanded) thumb-index line by `step` per finger (capped)."""
        yield from centre_pinch()
        f = api.contact_forces("pen")
        if min(f[0], f[1]) < target and squeezed["mm"] < cap * 1000 and np.linalg.norm(grip_offset()) < 0.001:
            t = api.tip_targets()
            u = (t[1] - t[0]) / np.linalg.norm(t[1] - t[0])
            yield from drive(api.move_tips(tips={"thumb": t[0] + step * u, "index": t[1] - step * u},
                                           seconds=0.2, straight=True), stop=lambda: rec() or False)
            squeezed["mm"] += step * 1000

    def hold_servo(seconds):
        """Hold while keeping the pinch centred on the pen."""
        t_end = api.t + seconds
        while api.t < t_end and not lost():
            yield from drive(api.hold(0.1), stop=lambda: rec() or False)
            yield from centre_pinch()

    def turn(sign, done, max_deg, step=2.0, sec=0.3):
        """Rotate the pinch about palm -y through the grip point in `step`-degree increments
        until done() (measured tilt), keeping the grip force up between increments."""
        total = 0.0
        while total < max_deg and not done() and not lost():
            yield from drive(api.rotate("pen", AY, np.radians(sign * step), FING, seconds=sec, centre=g_obj),
                             stop=lambda: rec() or done() or lost())
            total += step
            yield from keep_grip()
        return total

    # ---- 4. swing about palm y, closed loop on the measured tilt (fresh grip)
    def swing(sign, goal, max_deg):
        t_before, b_before = mon.tilt, beta()
        trace.clear()
        tot = yield from turn(sign, lambda: sign * mon.tilt >= goal, max_deg)
        api.note("trace t:tilt/fth,fix/pen_in_palm_mm/tips_xz " + " ".join(trace))
        api.note(f"swing {sign:+d}: cmd={tot:.0f} dbeta={beta() - b_before:.1f} dtilt={mon.tilt - t_before:.1f} "
                 f"{mon.s()} forces {forces()} squeezed={squeezed['mm']:.1f}mm shift={np.round(shift['v'] * 1000, 1).tolist()}")

    trace.clear()
    yield from centre_pinch()
    yield from centre_pinch()
    yield from regrip(1.3, tries=2, step=0.0006)
    api.note(f"pre-swing: offset {1000 * np.linalg.norm(grip_offset()):.1f}mm forces {forces()} " + " ".join(trace))
    b0 = beta()
    yield from swing(+1, SWING_GOAL, 40.0)
    if lost():
        api.note("pen lost during +swing")
        return
    yield from swing(-1, SWING_GOAL, 70.0)
    if lost():
        api.note("pen lost during -swing")
        return
    # ---- 5. back to rest: turn toward zero tilt until within 1.5 deg, then fine servo and hold
    sgn = 1 if mon.tilt < 0 else -1
    trace.clear()
    yield from turn(sgn, lambda: sgn * mon.tilt >= -1.5, 45.0)
    api.note("trace " + " ".join(trace))
    yield from hold_servo(0.3)
    api.note(f"return: {mon.s()} beta {beta() - b0:.1f} from start forces {forces()}")
    t_end = api.t + 5.0
    while api.t < t_end and not mon.settled and not lost():
        e = -mon.tilt
        if abs(e) < 2.5:
            yield from hold_servo(0.2)
            continue
        s = 1 if e > 0 else -1
        yield from turn(s, lambda: s * mon.tilt >= -1.0, min(abs(e), 6.0), step=1.0, sec=0.2)
        yield from hold_servo(0.15)
    yield from hold_servo(0.3)
    api.note(f"settle done: {mon.s()} calm={mon.calm} forces {forces()}")

    # ---- 6. roll about the pen's own axis (pads orbit the axis); no roll-back
    for ang in (35.0, -35.0):
        if mon.roll >= ROLL_GOAL or lost():
            break
        pp, Rr = api.object_in_palm("pen")
        mid = api.tips()[:2].mean(axis=0)
        c_obj = np.array([0.0, 0.0, float(Rr[:, 2] @ (mid - pp))])
        r = yield from drive(api.rotate("pen", Rr[:, 2], np.radians(ang), FING, seconds=2.0, centre=c_obj),
                             stop=lambda: mon.roll >= ROLL_GOAL or lost())
        yield from drive(api.hold(0.3))
        api.note(f"roll {ang}: {mon.s()} forces {forces()} r={r if r == 'stopped' else 'done'}")
    yield from drive(api.hold(0.8))
    api.note(f"END {mon.s()} forces {forces()}")
