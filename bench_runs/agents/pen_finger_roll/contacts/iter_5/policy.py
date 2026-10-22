import numpy as np
from scipy.spatial.transform import Rotation

TABLE_Z = 0.75
CUP_TOP = TABLE_Z + 0.06
PINCH = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame
R_PEN = 0.00375                           # nominal pen radius
DEPTH = 0.030                             # grasp this far below the pen's top
LIFT = 0.10
PRE = 14.0                                # in-hand pre-tilt before the rest pose is captured
LO, HI = 11.0, 20.0                       # swing targets relative to rest
TH, IX, MD = "thumb", "index", "middle"
SQ = 0.004                                # thumb/index squeeze


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _tilt(a):
    return float(np.arctan2(a[2], a[0]))


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class Mon:
    """Replica of the scorer's state, from api observations only."""

    def __init__(self, api):
        self.api = api
        p0, _ = api.object_pose("pen")
        self.reach = p0[2] - TABLE_Z
        self.out = False
        self.rest = None
        self.R_last = None
        self.twist = 0.0
        self.tw_lo = self.tw_hi = 0.0
        self.tilt = 0.0
        self.t_lo = self.t_hi = 0.0
        self.calm = 0
        self.settled = False
        self.bottom = p0[2] - self.reach

    def step(self):
        p, Rw = self.api.object_pose("pen")
        _, Rr = self.api.object_in_palm("pen")
        self.bottom = p[2] - self.reach * abs(Rw[2, 2])
        if not self.out and self.bottom > CUP_TOP + 0.01:
            self.out = True
            self.rest = Rr[:, 2].copy()
            self.R_last = Rr
            self.api.note(f"rest captured bottom={self.bottom:.3f}")
        if not self.out:
            return
        if self.bottom > CUP_TOP:
            self.twist += _twist(self.R_last, Rr)
            self.tw_lo, self.tw_hi = min(self.tw_lo, self.twist), max(self.tw_hi, self.twist)
            self.tilt = np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(self.rest)))
            self.t_lo, self.t_hi = min(self.t_lo, self.tilt), max(self.t_hi, self.tilt)
            if self.t_hi >= 10 and self.t_lo <= -10 and abs(self.tilt) < 5:
                self.calm += 1
                self.settled |= self.calm >= 27
            else:
                self.calm = 0
        self.R_last = Rr

    @property
    def roll_deg(self):
        return np.degrees(self.twist)

    @property
    def roll_range(self):
        return np.degrees(self.tw_hi - self.tw_lo)

    def summary(self):
        return (f"tilt={self.tilt:.1f} [{self.t_lo:.1f},{self.t_hi:.1f}] settled={self.settled} "
                f"roll={self.roll_deg:.1f} range={self.roll_range:.1f} bottom={self.bottom:.3f}")


def drive(mon, gen, stop=None):
    """Run a primitive step by step, updating the monitor; abandon it once stop() holds."""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        mon.step()
        yield
        if stop is not None and stop():
            gen.close()
            return None


def run(api):
    mon = Mon(api)
    p, Rw = api.object_pose("pen")
    s = 1.0 if Rw[2, 2] > 0 else -1.0               # pen +z up or down
    L2 = mon.reach                                   # centre-to-tip
    h = s * (L2 - DEPTH)                             # grasp height, pen frame
    api.note(f"pen p={np.round(p, 4).tolist()} half_len={L2:.4f} s={s}")

    R = api.hand_down()
    grasp_w = p + Rw[:, 2] * h                       # world grasp point on the axis
    palm_g = api.palm_for(grasp_w, PINCH, R)
    zt = R[:, 2]

    yield from drive(mon, api.open(0.6))
    yield from drive(mon, api.tuck(["middle", "ring", "little"], 0.6))
    yield from drive(mon, api.carry(palm_g + [0, 0, 0.10], R, 2.0))
    yield from drive(mon, api.carry(palm_g + [0, 0, 0.04], R, 1.0))

    def radial(v_world):
        """Unit vector in the pen frame, perpendicular to the pen axis, toward v."""
        _, Rw_ = api.object_pose("pen")
        v = Rw_.T @ v_world
        v[2] = 0
        return v / np.linalg.norm(v)

    yt = R[:, 1]
    cfg = {"mi": 0.0, "dm": 0.0, "mm": 0.0, "tripod": False}

    def grip(lift=0.0):
        """Tripod: thumb on the wrist side (-z); index on the fingertip side tilted toward +y at the grasp
        height; middle on the fingertip side tilted toward -y, dm lower on the pen. The two fingertip-side
        contacts cradle the pen laterally, so it cannot squirt out sideways as from a two-point pinch."""
        P_, N_ = {}, {}
        dm, mi, mm = cfg["dm"], cfg["mi"], cfg["mm"]
        spec = {TH: (-zt, 0.5 * dm), IX: (zt + mi * yt, 0.0)}
        if cfg["tripod"]:
            spec[MD] = (zt - mm * yt, dm)
        for f, (v, dh) in spec.items():
            n = radial(v)
            P_[f] = n * R_PEN + np.array([0, 0, h + s * (lift - dh)])
            N_[f] = n
        return P_, N_

    def palm_target(pt, n, sq):
        pp_, Rr_ = api.object_in_palm("pen")
        return pp_ + Rr_ @ pt + Rr_ @ n * (api.pad_offset - sq)

    def err(f, lift=0.04):
        P_, N_ = grip(lift)
        return api.reach(f, palm_target(P_[f], N_[f], SQ))

    # probe reachability at runtime (pre-grasp pose; points shifted by the 40 mm the palm will descend)
    log = []
    found = False
    for dm in (0.022, 0.026, 0.018, 0.030):
        for mm in (0.8, 0.6, 1.1):
            cfg.update(dm=dm, mm=mm, tripod=True)
            e = err(MD)
            log.append(f"m{dm*1000:.0f}/{mm}:{e*1000:.1f}")
            if e < 0.002:
                found = True
                break
        if found:
            break
    if found:
        for mi in (0.6, 0.45, 0.8, 0.3):
            cfg["mi"] = mi
            e = err(IX)
            log.append(f"i{mi}:{e*1000:.1f}")
            if e < 0.002:
                break
        log.append(f"t:{err(TH)*1000:.1f}")
        if err(IX) > 0.004 or err(TH) > 0.004:
            found = False
    if not found:
        cfg.update(mi=0.0, dm=0.0, mm=0.0, tripod=False)
    api.note(f"probe {' '.join(log)} cfg={cfg}")
    FING = [TH, IX, MD] if cfg["tripod"] else [TH, IX]

    P, N = grip(0.04)
    yield from drive(mon, api.approach("pen", P, standoff=0.016, seconds=0.9, normals=N))
    yield from drive(mon, api.carry(palm_g, R, 1.0))
    P, N = grip()
    r = yield from drive(mon, api.grasp("pen", P, squeeze=SQ, seconds=0.8, normals=N))
    yield from drive(mon, api.hold(0.3))
    pp, _ = api.object_in_palm("pen")
    api.note(f"grasp res={r['ik_residual_mm']} f={np.round(api.contact_forces('pen'), 2).tolist()} "
             f"pen_palm={np.round(pp * 1000).astype(int).tolist()} tips={np.round(api.tips()[:3] * 1000).astype(int).tolist()}")

    def forces():
        f = api.contact_forces("pen")
        return float(f[0]), float(f[1] + f[2])

    def sep():
        t = api.tips()
        return 1000 * float(np.linalg.norm(t[0] - t[1]))

    def regrip(sq=SQ, seconds=0.4):
        P_, N_ = grip()
        yield from drive(mon, api.grasp("pen", P_, squeeze=sq, seconds=seconds, normals=N_))

    C = np.array([0, 0, h])           # rotate about the grasp point

    def tilt_now():
        """Tilt in the palm x-z plane relative to the grasp-time axis (valid before rest capture too)."""
        _, Rr = api.object_in_palm("pen")
        return np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(a0)))

    def swing(target, rel, max_deg=40, seconds=2.0, tag=""):
        """Closed-loop swing: rotate thumb+index rigidly about palm y until rel() crosses target.
        +tilt is a rotation about -y. Logs samples (tilt, f_thumb, f_index, pad sep)."""
        up_ = target > rel()
        ang = -np.radians(max_deg) if up_ else np.radians(max_deg)
        log = []
        k = [0]
        fref = max(sum(forces()), 0.8)

        def stop():
            k[0] += 1
            if k[0] % 10 == 0:
                ft, fi = forces()
                log.append(f"{rel():.0f}/{ft:.1f}/{fi:.1f}/{sep():.0f}")
                if ft + fi < 0.45 * fref:
                    return True
            return rel() >= target if up_ else rel() <= target
        if not (rel() >= target if up_ else rel() <= target):
            yield from drive(mon, api.rotate("pen", [0, 1, 0], ang, FING, seconds, centre=C), stop)
        api.note(f"swing{tag}->{target:.1f} now {rel():.1f} samples(tilt/fth/fix/sep) {' '.join(log)}")

    def swing_to(target, rel, tries=3, **kw):
        for i in range(tries):
            yield from swing(target, rel, tag=f"#{i}", **kw)
            ok = rel() >= target if target > 0 else rel() <= target
            ft, fi = forces()
            if ft + fi < 0.1:
                return False
            yield from regrip()
            if ok:
                return True
        return False

    _, Rr0 = api.object_in_palm("pen")
    a0 = Rr0[:, 2].copy()
    # lift until the pen's bottom is a few mm above the rim (rest not yet captured)
    lift1 = (CUP_TOP + 0.003) - mon.bottom
    yield from drive(mon, api.carry(palm_g + [0, 0, lift1], R, 1.5))
    yield from drive(mon, api.hold(0.3))
    api.note(f"lift1 bottom={mon.bottom:.4f} out={mon.out} sep={sep():.0f} tilt={tilt_now():.1f}")
    # pre-tilt in hand (positive side swings freely), so rest sits mid-range
    yield from swing_to(PRE, tilt_now, max_deg=30, seconds=1.5)
    api.note(f"pretilt tilt={tilt_now():.1f} bottom={mon.bottom:.4f} out={mon.out}")
    yield from drive(mon, api.carry(palm_g + [0, 0, LIFT], R, 1.5))
    yield from drive(mon, api.hold(0.4))
    api.note(f"lifted {mon.summary()} graspTilt={tilt_now():.1f} f={np.round(forces(), 2).tolist()}")
    if not mon.out:
        yield from drive(mon, api.hold(0.5))
        return

    rel = lambda: mon.tilt
    yield from swing_to(-LO, rel, max_deg=25, seconds=2.0)
    api.note(f"lo: {mon.summary()}")
    yield from swing_to(HI, rel, max_deg=40, seconds=2.0)
    api.note(f"hi: {mon.summary()}")
    # back to rest and settle
    for _ in range(4):
        if abs(mon.tilt) > 1.5:
            tgt = 0.5 if mon.tilt > 0 else -0.5
            yield from swing(tgt, rel, max_deg=30, seconds=2.0, tag="rest")
        yield from drive(mon, api.hold(0.7), lambda: mon.settled)
        if mon.settled:
            break
    api.note(f"settle: {mon.summary()} f={np.round(forces(), 2).tolist()}")

    # roll about the pen's own axis, if the swing did not already give enough
    def roll_to(target, max_deg=50, seconds=2.0):
        _, Rr = api.object_in_palm("pen")
        axis = Rr[:, 2]
        up_ = target > mon.roll_deg
        ang = np.radians(max_deg) * (1 if up_ else -1)
        done = (lambda: mon.roll_deg >= target or mon.roll_range >= 40) if up_ else \
               (lambda: mon.roll_deg <= target or mon.roll_range >= 40)
        if done():
            return
        yield from drive(mon, api.rotate("pen", axis, ang, FING, seconds, centre=C), done)

    r0 = mon.roll_deg
    for target in (r0 + 22, r0 - 22):
        if mon.roll_range >= 40:
            break
        for _ in range(2):
            yield from roll_to(target)
            yield from regrip()
            if mon.roll_range >= 40 or (target > r0 and mon.roll_deg >= target) or (target < r0 and mon.roll_deg <= target):
                break
        api.note(f"roll->{target:.0f}: {mon.summary()}")
    yield from drive(mon, api.hold(0.5))
    api.note(f"end: {mon.summary()} f={np.round(forces(), 2).tolist()}")
