# pen_table_slide at level force
#
# Plan: thumb-index pinch across the pen near its top (pinch line along the palm z axis,
# palm facing down), force-servoed grip (api.grip). Before sliding, push the pinch sideways
# (palm -y, perpendicular to the pinch line) until the pen leans against the guide bore, so
# the downward drag of the sliding pads presses the pen into the wall instead of letting it
# fall out of the pinch sideways (iter 1 failure). Lower the palm in small chunks, watching
# the pinch forces; regrasp and repeat if the held slip is short. Then firm the grip, pull
# the pen out along its own axis and lift.
import numpy as np

TABLE = 0.75
BLOCK_TOP = 0.81
C = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (hand.py PINCH_CENTRE)
Z = np.array([0.0, 0.0, 1.0])         # pinch line: thumb at -z, index at +z
Y = np.array([0.0, 1.0, 0.0])
R_PEN = 0.00375
PAD = 0.008                           # pad centre sits ~8 mm inside the finger surface
HW_OPEN = 0.028
STROKE = 0.024
CHUNK = 0.002
SLIP_GOAL = 0.020
F_GRASP = 1.0
F_SLIDE = 0.7
F_LIFT = 1.6
LEAN_SIGN = -1.0                      # push the pinch toward palm -y


def pen_axis(api):
    p, R = api.object_pose("pen")
    a = R[:, 2]
    if a[2] < 0:
        a = -a
    return p, a


def axis_point_world(api, g):
    p, a = pen_axis(api)
    return p + a * (g - p[2]) / a[2]


def to_palm(api, x):
    P, Rp = api.measured_palm()
    return Rp.T @ (np.asarray(x) - P)


def rel(api):
    """Pen position along its axis relative to the thumb pad (what the scorer integrates)."""
    p, a = pen_axis(api)
    return float((p - api.tips_world()[0]) @ a)


def lateral(api):
    """Palm-y offset of the pen axis from the measured pad midpoint (palm frame) [m]."""
    tp = api.tips()
    mid = 0.5 * (tp[0] + tp[1])
    p, R = api.object_in_palm("pen")
    a = R[:, 2]
    q = p + a * ((mid - p) @ a)
    return float((q - mid)[1])


def tilt(api):
    _, a = pen_axis(api)
    return float(np.degrees(np.arccos(min(1.0, abs(a[2])))))


def fmt(v):
    return np.round(np.asarray(v, float), 3).tolist()


def held(api):
    f = api.contact_forces("pen")
    return f[0] > 0.22 and f[1:5].max() > 0.22, f


def close_targets(api, g, squeeze=0.002):
    xp = to_palm(api, axis_point_world(api, g))
    d = R_PEN + PAD - squeeze
    return {"thumb": xp - d * Z, "index": xp + d * Z}


def grasp_at(api, g, cm, R):
    """Open pinch centred on the pen axis at world height g (approach from above if needed), close, grip."""
    X = axis_point_world(api, g)
    P_g = api.palm_for(X, cm, R)
    P_now, _ = api.palm_pose()
    if P_now[2] > P_g[2] + 0.03:      # first approach: hover above, then come straight down
        yield from api.carry(xyz=P_g + np.array([0, 0, 0.05]), R=R, seconds=2.5)
        yield from api.hold(0.3)
        yield from api.carry(xyz=P_g, R=R, seconds=1.5)
    else:
        yield from api.carry(xyz=P_g, R=R, seconds=1.0)
    yield from api.hold(0.3)
    tgt = close_targets(api, g)
    r = yield from api.move_tips(tips=tgt, seconds=0.6, straight=True)
    res = yield from api.grip("pen", {"thumb": F_GRASP, "index": F_GRASP}, seconds=1.0)
    api.note(f"grasp at z={g:.4f} ik={r.get('ik_residual_mm')} grip={res} lat={lateral(api) * 1000:.1f}mm")


def lean(api):
    """Push the pinch sideways until the pen stops following (it rests against the bore)."""
    _, Rp = api.measured_palm()
    ydir = LEAN_SIGN * Rp[:, 1]
    e0 = lateral(api)
    pushed = 0.0
    for k in range(12):
        P, _ = api.palm_pose()
        yield from api.carry(xyz=P + 0.001 * ydir, seconds=0.12)
        pushed += 0.001
        e = (lateral(api) - e0) * LEAN_SIGN   # >0: pen lags behind the pads (stopped by the wall)
        if e < -0.0015 or e > 0.0015:
            break
    # push 1.5 mm further so the pen is pressed against its stop
    P, _ = api.palm_pose()
    yield from api.carry(xyz=P + 0.0015 * ydir, seconds=0.2)
    yield from api.hold(0.2)
    api.note(f"lean: pushed {pushed * 1000:.0f}+1.5 mm, lag {(lateral(api) - e0) * 1000:.1f} mm, tilt {tilt(api):.1f} deg, "
             f"forces={fmt(api.contact_forces('pen'))}")
    return pushed + 0.0015


def stroke(api, total):
    """Lower the palm in chunks while the pads slide down the pen; returns held slip [m]."""
    res = yield from api.grip("pen", {"thumb": F_SLIDE, "index": F_SLIDE}, seconds=0.4)
    slip, prev, bad, moved = 0.0, None, 0, 0.0
    while moved < STROKE - 1e-9:
        p, a = pen_axis(api)
        d = np.array([0, 0, -1.0]) - a          # half way between straight down and down the pen axis
        d = d / np.linalg.norm(d)
        d = d * CHUNK / -d[2]                    # CHUNK of vertical travel
        P, _ = api.palm_pose()
        yield from api.carry(xyz=P + d, seconds=0.12)
        moved += CHUNK
        ok, f = held(api)
        if ok:
            s = rel(api)
            if prev is not None:
                slip += s - prev
            prev = s
            bad = 0
        else:
            prev = None
            bad += 1
        gs = api.grip_state()
        trav = max(v["travel_mm"] for v in gs.values())
        if bad >= 3 or trav > 9.0:
            api.note(f"stroke abort at {moved * 1000:.0f} mm: forces={fmt(f)} grip={gs} lat={lateral(api) * 1000:.1f}mm")
            break
        if total + slip >= SLIP_GOAL + 0.004:
            break
    yield from api.hold(0.2)
    ok, f = held(api)
    if ok and prev is not None:
        slip += rel(api) - prev
    api.note(f"stroke: moved {moved * 1000:.0f} mm, held slip {slip * 1000:.1f} mm, forces={fmt(f)}, "
             f"tilt {tilt(api):.1f} deg, lat={lateral(api) * 1000:.1f}mm grip={api.grip_state()}")
    return slip, ok


def run(api):
    R = api.hand_down()
    p0, a0 = pen_axis(api)
    half = p0[2] - TABLE - 0.0005
    top = p0[2] + half
    g = max(top - 0.016, BLOCK_TOP + 0.044)
    api.note(f"pen p={fmt(p0)} half={half:.4f} top={top:.4f} grasp_z={g:.4f}")

    # Preshape at the start pose: open pinch, other fingers tucked.
    open_t = {"thumb": C - HW_OPEN * Z, "index": C + HW_OPEN * Z}
    yield from api.tuck(["middle", "ring", "little"], seconds=0.6)
    r = yield from api.move_tips(tips=open_t, seconds=0.8)
    yield from api.hold(0.3)
    tp = api.tips()
    cm = 0.5 * (tp[0] + tp[1])
    api.note(f"open ik={r.get('ik_residual_mm')} mid={fmt(cm)}")

    total, attempts = 0.0, 0
    while total < SLIP_GOAL and attempts < 4 and api.t < 40:
        attempts += 1
        yield from grasp_at(api, g, cm, R)
        yield from lean(api)
        s, ok = yield from stroke(api, total)
        total += max(0.0, s)
        api.note(f"attempt {attempts}: slip total {total * 1000:.1f} mm")
        if total >= SLIP_GOAL:
            break
        # regrasp: let go, open, go back up (grasp_at re-centres on the pen)
        yield from api.ungrip()
        tt = api.tip_targets()
        yield from api.move_tips(tips={"thumb": tt[0] - 0.008 * Z, "index": tt[1] + 0.008 * Z}, seconds=0.4, straight=True)
        P, _ = api.palm_pose()
        yield from api.carry(xyz=P + np.array([0, 0, 0.01]), seconds=0.4)
        yield from api.move_tips(tips={"thumb": C - HW_OPEN * Z, "index": C + HW_OPEN * Z}, seconds=0.4)

    # Firm grip, pull the pen out along its own axis, then lift straight up.
    res = yield from api.grip("pen", {"thumb": F_LIFT, "index": F_LIFT}, seconds=0.8)
    api.note(f"lift grip {res}")
    p, a = pen_axis(api)
    P, _ = api.palm_pose()
    yield from api.carry(xyz=P + 0.075 * a, seconds=2.0)
    P, _ = api.palm_pose()
    yield from api.carry(xyz=P + np.array([0, 0, 0.06]), seconds=1.5)
    yield from api.hold(0.5)
    p, a = pen_axis(api)
    api.note(f"lifted tip_h={(p[2] - half * a[2] - TABLE) * 1000:.1f}mm tilt {tilt(api):.1f} forces={fmt(api.contact_forces('pen'))} grip={api.grip_state()}")
