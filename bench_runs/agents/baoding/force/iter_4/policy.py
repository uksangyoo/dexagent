"""baoding, force level.

"Pinned pivot + gravity orbit". A finger presses ball_a onto the palm with the
closed-loop grip-force servo (pressing straight down), so ball_a is a post.
The palm tilts (<= ~20 deg from vertical, arm `carry`, pivot at ball_a) so the
in-palm downhill direction points from ball_b to ball_a with a rate-controlled
lead: ball_b leans on ball_a and orbits it. Half an orbit = exchanged places.

iter2: pin ball_a right after the drop, before it rolls to the palm's low spot
next to ball_b (iter1: it rolled ~35 mm to the little-finger edge, where no orbit
fits); radius from the initial gap; explicit downward servo direction; safe-box
and pin-drift guards; orbit direction chosen by a geometric path check.
"""
import numpy as np
from scipy.spatial.transform import Rotation

DT = 0.1
ALPHA0 = np.radians(13)
ALPHA_MAX = np.radians(24)
PIN_N = 1.5
GOAL = np.radians(205)
W_D = np.radians(18)
T_END = 46.0
FACE_X = 0.021              # palm surface height under the balls (iter1: centres at x = r + 21 mm)
SAFE = dict(y=0.042, z_lo=0.018, z_hi=0.120)


def _mm(v):
    return "[" + ",".join(f"{x * 1000:.0f}" for x in v) + "]"


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def pair(api):
    a, _ = api.object_in_palm("ball_a")
    b, _ = api.object_in_palm("ball_b")
    d = b - a
    return a, b, float(np.arctan2(d[1], d[2])), float(np.hypot(d[1], d[2]))


def held(p):
    return -0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15


def tilt_R(R0, theta, alpha):
    """Palm rotation whose in-palm downhill direction is (0, sin th, cos th)."""
    d = np.array([0.0, np.sin(theta), np.cos(theta)])
    k = np.array([0.0, -d[2], d[1]])
    return R0 @ Rotation.from_rotvec(k * alpha).as_matrix()


def update(api, st):
    a, b, phi, _ = pair(api)
    _, Rp = api.measured_palm()
    tilt = np.degrees(np.arccos(np.clip(Rp[2, 0], -1, 1)))
    ok = held(a) and held(b)
    if ok and tilt < 30 and st["last"] is not None and api.t > 1.0:
        st["total"] += _wrap(phi - st["last"])
    st["last"] = phi if ok else None
    st["max_tilt"] = max(st["max_tilt"], tilt)
    if not ok:
        st["dropped"] = True


def tracked(api, st, gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        update(api, st)
        yield


def outside(p):
    """How far a palm-frame (y, z) point is outside the safe box [m]."""
    y, z = p
    return max(abs(y) - SAFE["y"], SAFE["z_lo"] - z, z - SAFE["z_hi"], 0.0)


def run(api):
    notes = []
    try:
        yield from _run(api, notes)
    finally:
        for i in range(0, len(notes), 6):
            api.note(" | ".join(notes[i:i + 6]))


def pin_ball(api, st, notes, r, fingers, tag, limit=0.008):
    """Chase ball_a's top with a pad (velocity-predicted) and engage a straight-down force servo."""
    a, _, _, _ = pair(api)
    res = {f: api.reach(f, a + np.array([r + 0.012, 0, 0])) for f in fingers}
    pin = next((f for f in fingers if res[f] < limit), min(fingers, key=lambda f: res[f]))
    notes.append(f"{tag} t={api.t:.2f} a={_mm(a)} reach_mm={ {f: round(v * 1000, 1) for f, v in res.items()} } pin={pin}")
    yield from tracked(api, st, api.move_tips(tips={pin: a + np.array([r + 0.016, 0, 0])}, seconds=0.2))
    _, Rp = api.measured_palm()
    for h in np.linspace(0.013, 0.006, 4):
        a, _, _, _ = pair(api)
        try:
            v = api.object_velocity("ball_a")
            vp = np.zeros(3) if v is None else Rp.T @ np.asarray(v[0], float)
        except Exception:
            vp = np.zeros(3)
        vp[0] = 0.0
        if not np.all(np.isfinite(vp)) or np.linalg.norm(vp) > 0.15:
            vp = np.zeros(3)
        yield from tracked(api, st, api.move_tips(tips={pin: a + 0.1 * vp + np.array([r + h, 0, 0])}, seconds=0.04))
    a, Ra = api.object_in_palm("ball_a")
    down = Ra.T @ np.array([-1.0, 0, 0])
    g = yield from tracked(api, st, api.grip("ball_a", {pin: PIN_N}, directions={pin: down}, seconds=0.4))
    a2, _, _, _ = pair(api)
    notes.append(f"{tag} gripped t={api.t:.2f} a={_mm(a2)} moved={np.linalg.norm(a2 - a) * 1000:.1f} grip={g} "
                 f"pad={_mm(api.tips()[['thumb', 'index', 'middle', 'ring', 'little'].index(pin)])}")
    return pin


def _run(api, notes):
    st = {"total": 0.0, "last": None, "max_tilt": 0.0, "dropped": False}
    P0, R0 = api.palm_pose()
    a, b, phi, dist = pair(api)
    r = float(np.clip((dist - 0.005) / 2, 0.016, 0.022))
    notes.append(f"t0 a={_mm(a)} b={_mm(b)} phi={np.degrees(phi):.0f} dist={dist * 1000:.1f} r={r * 1000:.1f}")
    pin = yield from pin_ball(api, st, notes, r, ["little", "ring", "thumb"], "pin1", limit=0.012)
    yield from tracked(api, st, api.hold(max(0.02, 1.3 - api.t)))
    gs = api.grip_state().get(pin, {})
    a, b, phi, dist = pair(api)
    notes.append(f"pre t={api.t:.2f} a={_mm(a)} b={_mm(b)} phi={np.degrees(phi):.0f} d={dist * 1000:.1f} grip={gs} "
                 f"tips={';'.join(_mm(t) for t in api.tips())} Fb={np.round(api.contact_forces('ball_b'), 2).tolist()}")

    # ---- orbit direction: the path of ball_b around ball_a that stays in the palm
    tip_home = {"little": (-0.048, 0.149), "ring": (-0.023, 0.169), "thumb": (0.101, 0.074)}[pin]
    sector = np.arctan2(tip_home[0] - a[1], tip_home[1] - a[2])
    cost = {}
    for sg in (+1, -1):
        c = 0.0
        for u in np.radians(np.arange(0, 215, 5)):
            ang = phi + sg * u
            c += outside(a[1:] + 2 * r * np.array([np.sin(ang), np.cos(ang)]))
            if abs(_wrap(ang - sector)) < np.radians(40):
                c += 0.01
        cost[sg] = c
    s = min(cost, key=cost.get)
    notes.append(f"dir costs={ {k: round(v * 1000, 1) for k, v in cost.items()} } s={s} sector={np.degrees(sector):.0f}")

    if s > 0:   # thumb as an outer wall on the +y side of the orbit (ball_b escaped there in iters 2-3)
        guard = np.array([a[0] + 0.004, a[1] + 3 * r + 0.010, a[2] + 0.005])
        rg = api.reach("thumb", guard)
        th = api.tips()[0]
        notes.append(f"guard={_mm(guard)} reach={rg * 1000:.1f} thumb={_mm(th)}")
        if rg < 0.010 and th[1] > guard[1]:
            yield from tracked(api, st, api.move_tips(tips={"thumb": guard}, seconds=0.5))
            notes.append(f"guarded t={api.t:.2f} thumb={_mm(api.tips()[0])}")
    a, b, phi, dist = pair(api)
    a_palm0 = a.copy()
    a_world0 = api.object_pose("ball_a")[0]

    def tick(theta, alpha):
        R = tilt_R(R0, theta, alpha)
        yield from tracked(api, st, api.carry(api.palm_for(a_world0, a_palm0, R), R, seconds=DT))

    theta_cmd = phi + np.pi
    for k in range(12):
        a, b, phi, dist = pair(api)
        theta_cmd = phi + np.pi
        yield from tick(theta_cmd, ALPHA0 * (k + 1) / 12)

    prev = st["total"]
    w_f = 0.0
    alpha = ALPHA0
    n = 0
    stall = 0.0
    repinned = False
    ref = st["total"]
    while s * st["total"] < GOAL and api.t < T_END:
        a, b, phi, dist = pair(api)
        w = s * (st["total"] - prev) / DT
        prev = st["total"]
        w_f = 0.5 * w_f + 0.5 * w
        # position reference running ahead at W_D, never more than 20 deg ahead
        ref = st["total"] + s * min(s * (ref - st["total"]) + W_D * DT, np.radians(20))
        e = s * (ref - st["total"])
        lead = float(np.clip(np.radians(28) + 1.2 * e - 0.8 * w_f, np.radians(-35), np.radians(50)))
        gap = dist - 2 * r
        if gap > 0.003:
            lead *= max(0.0, 1 - (gap - 0.003) / 0.006)
        if outside(b[1:]) > 0:
            lead = min(lead, 0.0)
        target = phi + np.pi - s * lead
        drift = np.linalg.norm(a[1:] - a_palm0[1:])
        if drift > 0.012:
            notes.append(f"PIN LOST t={api.t:.2f} a={_mm(a)} b={_mm(b)} grip={api.grip_state()}")
            # level the palm, re-pin where the ball is, continue
            for k in range(8):
                yield from tick(theta_cmd, alpha * (1 - (k + 1) / 8))
            if repinned:
                break
            repinned = True
            yield from tracked(api, st, api.ungrip())
            pin = yield from pin_ball(api, st, notes, r, [pin, "little", "ring", "thumb"], "pin2")
            a, b, phi, dist = pair(api)
            a_palm0 = a.copy()
            a_world0 = api.object_pose("ball_a")[0]
            alpha = np.radians(10)
            continue
        if drift > 0.004:
            alpha = max(np.radians(7), alpha - np.radians(1))
        elif stall > 1.5:
            alpha = min(ALPHA_MAX, alpha + np.radians(0.5))
        elif w_f > np.radians(8):   # moving: kinetic friction is lower, drop back to avoid a runaway
            alpha = max(ALPHA0, alpha - np.radians(2))
        theta_cmd += float(np.clip(_wrap(target - theta_cmd), -np.radians(10), np.radians(10)))
        stall = stall + DT if w_f < np.radians(5) else 0.0
        try:
            yield from tick(theta_cmd, alpha)
        except ValueError:
            alpha = max(np.radians(6), alpha - np.radians(4))
            notes.append(f"armIK fail t={api.t:.2f}")
        if n % 5 == 0:
            gs = api.grip_state().get(pin, {})
            notes.append(f"t={api.t:.1f} tot={np.degrees(st['total']):.0f} phi={np.degrees(phi):.0f} d={dist * 1000:.0f} "
                         f"a={_mm(a[1:])} b={_mm(b)} lead={np.degrees(lead):.0f} th={np.degrees(theta_cmd):.0f} "
                         f"al={np.degrees(alpha):.0f} F={gs.get('force_N')} tr={gs.get('travel_mm')}")
        n += 1

    for k in range(15):
        a, b, phi, dist = pair(api)
        theta_cmd += float(np.clip(_wrap(phi + np.pi - theta_cmd), -np.radians(10), np.radians(10)))
        yield from tick(theta_cmd, alpha * (1 - (k + 1) / 15))
    yield from tracked(api, st, api.hold(0.5))
    a, b, phi, dist = pair(api)
    notes.append(f"end t={api.t:.2f} tot={np.degrees(st['total']):.0f} a={_mm(a)} b={_mm(b)} d={dist * 1000:.0f} "
                 f"dropped={st['dropped']} max_tilt={st['max_tilt']:.1f} grip={api.grip_state()}")
