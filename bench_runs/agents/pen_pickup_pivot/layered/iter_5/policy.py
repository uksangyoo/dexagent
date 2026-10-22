"""pen_pickup_pivot, layered level.

Plan: thumb-index pinch ~45 mm from the pen centre (toward the thumb-side end),
pinch line horizontal and perpendicular to the pen. The pen has no torsional
friction, so the pinch is a free hinge. Lifting the hinge drags the long end on
the table until the pen stands on it, then the pen hangs vertically below the
pinch (pendulum, centre of mass below the hinge) and is lifted clear.

iter 3 change: the pinch force is set from the MEASURED pad positions
(targets = measured pads + ~0.6 mm inward), not from the overshooting
commanded targets, so the squeeze is ~0.1-0.6 N instead of 2-5 N. The wedge
(pads sit above the pen's equator on the table) is then held by friction, and
the pinch improves as the pen tilts up. Lift through the bad low-angle zone
briskly. Release gently on failure (no open() flick).

Layers: arm via carry; fingertip layer (move_tips) for the pinch, table-touch
probing and force-trimmed squeezing; contacts layer `tuck` for unused fingers.
"""
import numpy as np

C = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame
S = 0.045                              # hinge offset from pen centre toward the thumb-side end [m]
HW_OPEN = 0.022
THOFF = 0.0145                         # thumb pad parked this far from the pen axis (palm z); it never moves in z
CP = C + np.array([0, 0, -HW_OPEN + THOFF])   # palm-frame point placed over the hinge point
DIRS = {"thumb": np.array([1.0, 0, 0.3]) / np.hypot(1, 0.3),
        "index": np.array([1.0, 0, -0.3]) / np.hypot(1, 0.3)}
TABLE = 0.75
TH, IX = 0, 1
F_LO, F_HI = 0.08, 0.9                 # (unused) old force band [N]
F_T = 0.25                             # force-servo target per pinch finger [N]
KF, DZ, MAXPEN = 0.002, 0.0003, 0.0025  # servo gain [m/N], max step per 0.1 s [m], max commanded penetration [m]


BUF = []


def note(api, s):
    """api.note plus a buffer re-emitted at the end (the bench keeps only the last 60 log lines)."""
    api.note(s)
    BUF.append(f"{api.t:.2f} {s}")


def _until(gen, cond):
    """Run a motion generator, stop early when cond() is true. Returns True if stopped by cond."""
    hit = False
    try:
        for _ in gen:
            yield
            if cond():
                hit = True
                break
    finally:
        gen.close()
    return hit


def pen(api):
    p, R = api.object_pose("pen")
    a = R[:, 2]
    if a[2] > 0:
        a = -a
    tilt = float(np.degrees(np.arccos(min(1.0, abs(a[2])))))
    return p, a, tilt


def dev(api):
    return api.tips() - api.tip_targets()


def forces(api):
    try:
        return np.asarray(api.contact_forces("pen"), float)
    except Exception:
        return np.zeros(6)


def fmt(v, k=1000, d=1):
    return np.round(np.asarray(v) * k, d).tolist()


def set_pinch(api, gap_th, gap_ix, seconds=0.3):
    """Pinch targets = measured pads, each pushed `gap` m toward the other along palm z (force by offset)."""
    m = api.tips()
    tt = api.tip_targets()
    th = m[TH].copy()
    ix = m[IX].copy()
    th[2] += gap_th
    ix[2] -= gap_ix
    th[0] = 0.5 * (th[0] + tt[TH][0])          # keep the height roughly where it is commanded
    ix[0] = 0.5 * (ix[0] + tt[IX][0])
    yield from api.move_tips(tips={"thumb": th, "index": ix}, seconds=seconds, straight=True, directions=DIRS)


def release(api):
    """Gentle release: pads apart along the pinch line, then palm up."""
    try:
        tt = api.tip_targets()
        yield from api.move_tips(tips={"thumb": tt[TH] + [-0.004, 0, -0.012], "index": tt[IX] + [-0.004, 0, 0.012]},
                                 seconds=0.6, directions=DIRS)
        P, _ = api.palm_pose()
        yield from api.carry(xyz=P + [0, 0, 0.06], seconds=1.0)
        yield from api.hold(0.6)
    except ValueError as e:
        note(api, f"release error {e}")


def attempt(api, k, geo):
    p, a, tilt = pen(api)
    u = np.array([a[0], a[1], 0.0])
    u /= np.linalg.norm(u) + 1e-12
    yaw = float(np.arctan2(-u[0], u[1]))
    while yaw - api.DEFAULT_YAW > np.pi / 2:
        yaw -= np.pi
    while yaw - api.DEFAULT_YAW < -np.pi / 2:
        yaw += np.pi
    R = api.hand_down(yaw)
    uy = R[:, 1]
    H = p + S * uy
    H[2] = p[2]
    note(api, f"att{k} pen p={fmt(p)} tilt={tilt:.0f} yaw={yaw:.3f}")

    thumb = C + np.array([0, 0, -HW_OPEN])
    index = C + np.array([0, 0, HW_OPEN])
    yield from api.tuck(["middle", "ring", "little"], seconds=0.5)
    yield from api.move_tips(xyz=api.palm_for(H + [0, 0, 0.09], CP, R), R=R,
                             tips={"thumb": thumb, "index": index}, seconds=1.8, directions=DIRS)
    yield from api.carry(xyz=api.palm_for(H + [0, 0, 0.032], CP, R), R=R, seconds=0.9)
    yield from api.hold(0.3)
    base = dev(api)

    def changed(i):
        return float(np.linalg.norm(dev(api)[i] - base[i]))

    # --- descend until a pinch finger touches the table (deflection sensing)
    yield from _until(api.carry(xyz=api.palm_for(H + [0, 0, -0.008], CP, R), R=R, seconds=4.0),
                      lambda: changed(TH) > 0.0015 or changed(IX) > 0.0015 or forces(api)[:2].sum() > 0.08)
    yield from api.hold(0.1)
    first = TH if changed(TH) >= changed(IX) else IX
    other = IX if first == TH else TH
    tt = api.tip_targets()
    name = "thumb" if other == TH else "index"
    yield from _until(api.move_tips(tips={name: tt[other] + np.array([0.014, 0, 0])}, seconds=1.6,
                                    straight=True, directions=DIRS),
                      lambda: changed(other) > 0.0015)
    yield from api.hold(0.1)
    # --- pads just above their table-touch heights (0.4 mm clear: no table friction while closing)
    m = api.tips()
    tt = api.tip_targets()
    new = {}
    for i, nm in ((TH, "thumb"), (IX, "index")):
        q = tt[i].copy()
        q[0] = m[i][0] - 0.0004
        new[nm] = q
    yield from api.move_tips(tips=new, seconds=0.4, directions=DIRS)
    yield from api.hold(0.15)
    tw = api.tips_world()
    note(api, f"pads over table=({(tw[TH][2]-TABLE)*1000:.1f},{(tw[IX][2]-TABLE)*1000:.1f})mm")

    # --- close: thumb parked THOFF from the pen axis as a fixed stop; the index closes until both touch
    pp, Rp = api.object_in_palm("pen")
    ap = Rp[:, 2]
    tt = api.tip_targets()
    m = api.tips()
    y_pin = 0.5 * (tt[TH][1] + tt[IX][1])
    s = ((y_pin - pp[1]) / ap[1]) if abs(ap[1]) > 0.3 else 0.0
    zc = pp[2] + s * ap[2]                        # pen axis at the pinch, palm z
    xc = pp[0] + s * ap[0]                        # pen axis depth, palm x (down)
    r = float(np.clip(pen(api)[0][2] - TABLE, 0.003, 0.0045))
    D = {}
    for i in (TH, IX):
        rho = float(np.clip(tw[i][2] - TABLE + 0.0004, 0.008, 0.0105))   # pad 'radius' = its table-touch height
        v = xc - m[i][0]                          # pad centre above the pen axis
        D[i] = float(np.sqrt(max((rho + r) ** 2 - v ** 2, 1e-6)))
    hit = yield from _until(api.move_tips(tips={"index": np.array([tt[IX][0], tt[IX][1], zc + D[IX] - 0.0025])},
                                          seconds=2.2, straight=True, directions=DIRS),
                            lambda: (lambda f: f[TH] > 0.015 and f[IX] > 0.015)(forces(api)))
    yield from api.hold(0.15)
    # --- per-finger force servo: each pad's target is nudged along the pinch line every 0.1 s so its
    #     contact force tracks F_T (independent of finger stiffness; balanced forces at lift-off)
    m = api.tips()
    tt = api.tip_targets()
    yield from api.move_tips(tips={"index": np.array([tt[IX][0], tt[IX][1], m[IX][2] - 0.0003])}, seconds=0.2,
                             straight=True, directions=DIRS)
    diag = {"fmin": 9.0, "n": 0}

    def servo(palm=None, seconds=0.1, ft=F_T):
        f = forces(api)
        tt = api.tip_targets()
        m = api.tips()
        th, ix = tt[TH].copy(), tt[IX].copy()
        fe = 0.5 * (f[TH] + f[IX])                # thumb is a passive stop; the index regulates the pinch force
        ix[2] -= float(np.clip(KF * (ft - fe), -DZ, DZ))
        ix[2] = max(ix[2], m[IX][2] - MAXPEN)     # never command more than MAXPEN into the contact
        diag["fmin"] = min(diag["fmin"], f[TH], f[IX])
        diag["n"] += 1
        yield from api.move_tips(xyz=palm, R=R if palm is not None else None,
                                 tips={"index": ix}, seconds=seconds, directions=DIRS)

    for _ in range(10):
        yield from servo(seconds=0.1)
    f = forces(api)
    p, a, _ = pen(api)
    pp, _ = api.object_in_palm("pen")
    m = api.tips()
    ang = [float(np.degrees(np.arctan2(pp[0] - m[i][0], abs(m[i][2] - pp[2])))) for i in (TH, IX)]
    note(api, f"close hit={hit} D=({D[TH]*1000:.1f},{D[IX]*1000:.1f}) dz=({(pp[2]-m[TH][2])*1000:.1f},{(m[IX][2]-pp[2])*1000:.1f})mm "
             f"ang=({ang[0]:.0f},{ang[1]:.0f})deg f={fmt(f[:2], 1, 2)}")

    def pinch_dist():
        p, a, _ = pen(api)
        tw = api.tips_world()
        h = 0.5 * (tw[TH] + tw[IX]) - p
        return float(np.linalg.norm(h - (h @ a) * a))

    d0 = pinch_dist()
    lost = {"n": 0}

    def lost_now():
        f = forces(api)
        ok = f[TH] + f[IX] > 0.004 and pinch_dist() < d0 + 0.008
        lost["n"] = 0 if ok else lost["n"] + 1
        return lost["n"] > 3            # checked every 0.1 s segment

    def lift(dz, T):
        """Raise the palm by dz over T seconds in 0.1 s servo segments (smooth overall profile)."""
        P0, _ = api.palm_pose()
        n = max(1, int(round(T / 0.1)))
        for k in range(1, n + 1):
            u = k / n
            yield from servo(palm=P0 + [0, 0, dz * u * u * (3 - 2 * u)], seconds=0.1)
            if lost_now():
                return False
        return True

    ok = yield from lift(0.008, 0.6)
    p, a, tilt = pen(api)
    note(api, f"lift8 tilt={tilt:.1f} pz={(p[2]-TABLE)*1000:.1f}mm f={fmt(forces(api)[:2], 1, 2)} ok={ok}")
    if ok:
        ok = yield from lift(0.062, 1.6)
    p, a, tilt = pen(api)
    f = forces(api)
    note(api, f"liftA tilt={tilt:.1f} pz={(p[2]-TABLE)*1000:.0f}mm f={fmt(f[:2], 1, 2)} ok={ok} fmin={diag['fmin']:.2f}")
    if not ok or tilt > 80:
        return "lost"
    ok = yield from lift(0.105, 2.6)
    for _ in range(10):
        yield from servo(seconds=0.1)
    p, a, tilt = pen(api)
    f = forces(api)
    note(api, f"liftB tilt={tilt:.1f} pz={(p[2]-TABLE)*1000:.0f}mm f={fmt(f[:2], 1, 2)} ok={ok}")
    if not ok and f[TH] + f[IX] < 0.004:
        return "lost"
    return "ok" if tilt < 12 else "tilted"


def restand(api):
    """Pen hangs tilted: set its lower end back on the table under the hinge, then lift again."""
    p, a, tilt = pen(api)
    tw = api.tips_world()
    hinge = 0.5 * (tw[TH] + tw[IX])
    L = float(np.linalg.norm(hinge - p)) + 0.07       # rough hinge-to-lower-end distance
    low = hinge[2] - L * abs(a[2]) - TABLE
    P, _ = api.palm_pose()
    yield from api.carry(xyz=P + [0, 0, -(low - 0.002)], seconds=1.5)
    yield from api.hold(0.3)
    p, a, tilt = pen(api)
    bottom = p + 0.07 * a
    tw = api.tips_world()
    d = bottom - 0.5 * (tw[TH] + tw[IX])
    d[2] = 0
    if np.linalg.norm(d) < 0.06:
        P, _ = api.palm_pose()
        yield from api.carry(xyz=P + d, seconds=1.5)
        yield from api.hold(0.4)
    note(api, f"restand tilt={pen(api)[2]:.1f}")
    P, _ = api.palm_pose()
    yield from api.carry(xyz=P + [0, 0, 0.08], seconds=2.5)
    yield from api.hold(1.0)
    note(api, f"restand clear tilt={pen(api)[2]:.1f}")


def run(api):
    BUF.clear()
    try:
        yield from _run(api)
    finally:
        for s in BUF[-18:]:
            api.note("recap " + s)


def _run(api):
    geo = {}
    for k in range(4):
        p, _, tilt = pen(api)
        if np.hypot(p[0] - 0.515, p[1] + 0.07) > 0.08 or p[2] < TABLE or p[2] > TABLE + 0.02:
            note(api, f"pen out of reach at {fmt(p)} tilt={tilt:.0f}; stopping")
            return
        if api.t > 42:
            return
        try:
            res = yield from attempt(api, k, geo)
        except ValueError as e:
            note(api, f"attempt {k} error {e}")
            yield from release(api)
            continue
        note(api, f"attempt {k}: {res} t={api.t:.1f}")
        if res == "ok":
            yield from api.hold(1.0)
            return
        if res == "tilted":
            try:
                yield from restand(api)
            except ValueError as e:
                note(api, f"restand error {e}")
            return
        yield from release(api)
