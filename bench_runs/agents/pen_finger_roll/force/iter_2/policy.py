import numpy as np
from scipy.spatial.transform import Rotation

# Plan: thumb-index pinch (pinch line along palm z) near the pen top, held by the
# grip-force servo. Swing (tilt about palm y) and roll (about the pen axis) both come
# from shearing the two opposed pads in opposite directions:
#   shear along palm x (the pen axis)          -> pen rolls over the pads about palm y (swing)
#   shear along palm y (perp. to pen & pinch)  -> pen rolls about its own axis (roll)
# Each is run closed-loop on the measured pen pose (sign/gain calibrated online).

TABLE = 0.75
RIM = TABLE + 0.06
PC = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (from hand.py docs)
PAD = 0.008                            # pad centre inside the fingertip surface
R_PEN = 0.00375
GRIP_N = 1.5
EZ = np.array([0.0, 0.0, 1.0])


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _tilt(a):
    return float(np.arctan2(a[2], a[0]))


def run(api):
    D = []
    p0, _ = api.object_pose("pen")
    reach = p0[2] - TABLE

    class Mon:
        out = False
        rest = None
        R_last = None
        tw = 0.0
        tw_lo = tw_hi = 0.0
        t_lo = t_hi = 0.0
        bottom = 0.0

    mon = Mon()

    def mstep():
        p, Rw = api.object_pose("pen")
        _, Rr = api.object_in_palm("pen")
        mon.bottom = p[2] - reach * abs(Rw[2, 2])
        if not mon.out and mon.bottom > RIM + 0.01:
            mon.out = True
            mon.rest = Rr[:, 2].copy()
            mon.R_last = Rr
            D.append(f"rest captured t={api.t:.2f} axis_palm={np.round(mon.rest, 3).tolist()}")
        if not mon.out:
            return
        if mon.bottom > RIM:
            mon.tw += _twist(mon.R_last, Rr)
            mon.tw_lo, mon.tw_hi = min(mon.tw_lo, mon.tw), max(mon.tw_hi, mon.tw)
            t = np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(mon.rest)))
            mon.t_lo, mon.t_hi = min(mon.t_lo, t), max(mon.t_hi, t)
        mon.R_last = Rr

    def W(gen):
        try:
            while True:
                x = next(gen)
                mstep()
                yield x
        except StopIteration as e:
            return e.value

    def tilt_now():
        _, Rr = api.object_in_palm("pen")
        ref = mon.rest if mon.rest is not None else Rr[:, 2]
        return float(np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(ref))))

    def roll_now():
        return float(np.degrees(mon.tw))

    def pinch_world():
        p, R = api.object_pose("pen")
        up = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
        return p + (reach - 0.015) * up

    try:
        R = api.hand_down()
        # ---- pre-grasp: tuck middle/ring/little, open thumb-index pinch around the pen top
        P = pinch_world()
        pre = P - R @ PC + np.array([0, 0, 0.06])
        wide = R_PEN + PAD + 0.010
        yield from W(api.tuck(["middle", "ring", "little"], seconds=0.6))
        r = yield from W(api.move_tips(xyz=pre, R=R, tips={"thumb": PC - wide * EZ, "index": PC + wide * EZ}, seconds=2.0))
        D.append(f"pregrasp ik {r}")
        yield from W(api.hold(0.3))
        # correct palm so the measured pinch centre sits over the pen
        Pm, Rm = api.measured_palm()
        err = Rm.T @ (P + np.array([0, 0, 0.06]) - Pm) - PC
        pre = pre + Rm @ err
        D.append(f"palm correction mm {np.round(err * 1000, 1).tolist()}")
        yield from W(api.carry(xyz=pre, R=R, seconds=0.5))
        down = pre - np.array([0, 0, 0.06])
        yield from W(api.carry(xyz=down, R=R, seconds=1.2))
        yield from W(api.hold(0.3))
        # ---- re-target pads on the measured pen, close to 2 mm standoff, then force grip
        Pm, Rm = api.measured_palm()
        pp = Rm.T @ (pinch_world() - Pm)
        D.append(f"pinch pt palm mm {np.round(pp * 1000, 1).tolist()}")
        close = R_PEN + PAD + 0.002
        r = yield from W(api.move_tips(tips={"thumb": pp - close * EZ, "index": pp + close * EZ}, seconds=0.6, straight=True))
        D.append(f"close ik {r}")
        g = yield from W(api.grip("pen", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.5, max_travel=0.010))
        D.append(f"grip {g}")
        # ---- lift out of the cup
        yield from W(api.carry(xyz=down + np.array([0, 0, 0.15]), R=R, seconds=2.5))
        yield from W(api.hold(0.5))
        D.append(f"after lift t={api.t:.2f} bottom_above_rim_mm={1000 * (mon.bottom - RIM):.1f} out={mon.out} "
                 f"tilt={tilt_now():.1f} grip={api.grip_state()}")

        # ---- finger servos on the measured pen angles
        TS = []

        def sample(tag, cmd):
            gs = api.grip_state()
            TS.append(f"{tag}{api.t:.1f}:{cmd:.1f}/{tilt_now():.0f}/{np.degrees(mon.tw):.0f}/"
                      f"{gs.get('thumb', {}).get('force_N', 0):.1f},{gs.get('index', {}).get('force_N', 0):.1f}/"
                      f"{gs.get('thumb', {}).get('travel_mm', 0):.0f},{gs.get('index', {}).get('travel_mm', 0):.0f}")

        def lost():
            return mon.bottom < RIM + 0.02

        class Act:
            """Closed loop on one pen angle (deg) through a 1-D pad command u."""
            def __init__(self, name, apply, g, umax, step, measure):
                tt = api.tip_targets()
                self.bt, self.bi = tt[0].copy(), tt[1].copy()
                self.c = 0.5 * (self.bt + self.bi)
                self.name, self.apply, self.g, self.umax, self.step, self.measure = name, apply, g, umax, step, measure
                self.u = 0.0

            def set(self, u, seconds):
                self.u = float(np.clip(u, -self.umax, self.umax))
                t, i = self.apply(self, self.u)
                yield from W(api.move_tips(tips={"thumb": t, "index": i}, seconds=seconds))

            def calibrate(self, du):
                a0 = self.measure()
                yield from self.set(self.u + du, 0.4)
                yield from W(api.hold(0.2))
                d = self.measure() - a0
                obs = d / du
                if abs(d) >= 2.0:
                    self.g = float(np.sign(obs) * np.clip(abs(obs), abs(self.g) / 4, abs(self.g) * 3))
                D.append(f"{self.name} calib d={d:.1f}deg for u={du} -> g={self.g:.2f}")
                sample(self.name[0], self.u)

            def go(self, target, tol, tmax, settle=0.0, guard=None):
                t0, ok, e = api.t, None, 0.0
                while api.t - t0 < tmax and not lost():
                    a = self.measure()
                    if guard is not None and abs(tilt_now()) > guard:
                        D.append(f"{self.name} guard trip tilt={tilt_now():.1f}")
                        break
                    e = target - a
                    if abs(e) < tol:
                        ok = api.t if ok is None else ok
                        if api.t - ok >= settle:
                            break
                    else:
                        ok = None
                    yield from self.set(self.u + float(np.clip(0.4 * e / self.g, -self.step, self.step)), 0.1)
                    sample(self.name[0], self.u)
                D.append(f"{self.name} go {target:.1f}: err={e:.1f} u={self.u:.2f} in {api.t - t0:.1f}s "
                         f"tilt[{mon.t_lo:.1f},{mon.t_hi:.1f}] roll={np.degrees(mon.tw_hi - mon.tw_lo):.1f}")

        def roll_apply(self, u):          # u: mm of opposite shear along palm y
            v = np.array([0.0, 1e-3 * u, 0.0])
            return self.bt + v, self.bi - v

        def tilt_apply(self, u):          # u: deg of rigid pad rotation about palm y through the pinch centre
            Ry = Rotation.from_rotvec([0.0, np.radians(u), 0.0]).as_matrix()
            return self.c + Ry @ (self.bt - self.c), self.c + Ry @ (self.bi - self.c)

        # roll first (gentler): thumb +y / index -y predicted to give -roll
        ro = Act("roll", roll_apply, -10.0, 4.0, 0.3, roll_now)
        r0 = roll_now()
        yield from ro.calibrate(0.8)
        yield from ro.go(r0 + 20.0, 2.5, 5.0, guard=25)
        yield from ro.go(r0 - 20.0, 2.5, 6.0, guard=25)
        yield from ro.go(r0, 4.0, 3.0, guard=25)
        D.append(f"after roll grip={api.grip_state()}")
        # swing: rigid rotation of both pads about palm y; pen rotates with them, tilt ~ -u
        sw = Act("swing", tilt_apply, -1.0, 35.0, 2.0, tilt_now)
        yield from sw.calibrate(4.0)
        yield from sw.go(16.0, 2.0, 6.0)
        yield from sw.go(-16.0, 2.0, 7.0)
        yield from sw.go(0.0, 1.5, 6.0, settle=1.2)
        yield from W(api.hold(0.5))
    finally:
        D.append(f"end t={api.t:.2f} bottom_above_rim_mm={1000 * (mon.bottom - RIM):.1f} tilt[{mon.t_lo:.1f},{mon.t_hi:.1f}] "
                 f"roll_range={np.degrees(mon.tw_hi - mon.tw_lo):.1f} grip={api.grip_state()}")
        for line in D:
            api.note(line)
        try:
            for k in range(0, len(TS), 12):
                api.note("TS " + " ".join(TS[k:k + 12]))
        except NameError:
            pass
