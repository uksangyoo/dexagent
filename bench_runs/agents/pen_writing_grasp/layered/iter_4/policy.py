import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
RIM = TABLE + 0.060
R_PEN = 0.00375                      # nominal pen radius (+-10 %)
PAD = 0.008                          # pad centre sits ~8 mm inside the fingertip surface
YAW = 0.7
# Three-finger chuck with the pen along palm y (hand rolled thumb-down so palm y = world down):
# thumb pad on the wrist side (-z), index and middle pads on the fingertip side (+z), 22 mm apart
# along the pen, thumb between them -> 3-point grip, stiff against tip forces along palm z.
X0, Z0 = 0.080, 0.118
Y_T, Y_I, Y_M = 0.029, 0.040, 0.018
STROKE = 0.060
# Pen lean during the stroke, tip LEADING the motion. With table friction mu = 1 the sliding
# reaction sits 45 deg from vertical; at 50 deg lean it points almost along the pen axis, so it
# pushes the pen axially into the grip (friction-held, strong) instead of twisting it out of the
# fingers, and the tip can never jam (reaction outside the friction cone). Fallbacks if arm IK fails.
LEANS = (50.0, 43.0, 36.0)
D_PRESS = 0.0005     # target axial push of the pen into the grip [m]: grip compliance used as a force sensor

LOG = []


def note(api, s):
    LOG.append(f"{api.t:6.2f} {s}")


def pen(api):
    p, R = api.object_pose("pen")
    a = R[:, 2]
    up = a if a[2] > 0 else -a
    return p, up


def lowest(api, H):
    p, up = pen(api)
    return p[2] - up[2] * H - R_PEN - TABLE


def tip_w(api, H):
    p, up = pen(api)
    return p - up * (H + R_PEN)


def lean_deg(api):
    _, up = pen(api)
    return float(np.degrees(np.arccos(min(1.0, up[2]))))


def axial(api):
    """Pen centre position along the pen's upward axis, measured in the palm frame."""
    pp, Rp = api.object_in_palm("pen")
    _, Rm = api.measured_palm()
    _, up = pen(api)
    return float(pp @ (Rm.T @ up))


def align(a, b):
    """Rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / s * np.arctan2(s, np.dot(a, b))).as_matrix()


def chuck_targets(api, hw, y_t):
    """Pad targets around the measured pen axis (palm frame)."""
    pp, Rp = api.object_in_palm("pen")
    ap = Rp[:, 2]
    ap = ap if ap[1] > 0 else -ap          # along +palm y
    at = lambda y: pp + ap * (y - pp[1]) / ap[1]
    ez = np.array([0, 0, 1.0])
    return {"thumb": at(y_t) - ez * hw, "index": at(Y_I) + ez * hw, "middle": at(Y_M) + ez * hw}


def safe_carry(api, P, R, seconds):
    """carry(); on arm IK failure snap the command back to the measured palm and report False."""
    try:
        yield from api.carry(P, R, seconds)
        return True
    except ValueError as e:
        note(api, f"ARM IK fail: {e}")
        Pm, Rm = api.measured_palm()
        yield from api.carry(Pm, Rm, 0.02)
        yield from api.hold(0.2)
        return False


def run(api):
    try:
        yield from _run(api)
    finally:
        for s in LOG:
            api.note(s)


def _run(api):
    R0 = api.hand_down(YAW)
    Rs = R0 @ Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()   # thumb down, palm y = world down
    fz = Rs[:, 2].copy()
    d = -fz[:2] / np.linalg.norm(fz[:2])           # stroke direction (toward the wrist)

    y_t = Y_T
    yield from api.tuck(["ring", "little"], seconds=0.5)
    wide = 0.028
    pre = {"thumb": np.array([X0, Y_T, Z0 - wide]), "index": np.array([X0, Y_I, Z0 + wide]),
           "middle": np.array([X0, Y_M, Z0 + wide])}
    r = yield from api.move_tips(tips=pre, seconds=0.6)
    note(api, f"preshape ik {r}")
    if api.reach("thumb", [X0, Y_T, Z0 - 0.012]) > 0.003:
        y_t = 0.034
        note(api, f"thumb reach fallback y={y_t}")

    p, up = pen(api)
    H = (p[2] - TABLE - R_PEN) / up[2]
    home = (p - up * (H + R_PEN))[:2]
    top = p[2] + up[2] * (H + R_PEN)
    note(api, f"pen half={H*1000:.1f}mm lean={lean_deg(api):.1f} home={np.round(home, 4).tolist()} top={top:.4f}")

    # pen top ~20 mm above the middle contact (clear of the tucked ring finger), index >= 18 mm above the rim
    zc = max(top - 0.020 - (y_t - Y_M), RIM + 0.018 + (Y_I - y_t))
    gc = p + up * ((zc - p[2]) / up[2])
    Rg = align(np.array([0, 0, -1.0]), -up) @ Rs
    C = np.array([X0, y_t, Z0])
    Pg = api.palm_for(gc, C, Rg)
    # turn the hand high above the pen first, then come straight down along the pen axis
    yield from api.carry(Pg + up * 0.16 + np.array([0, 0, 0.03]), Rg, 2.0)
    yield from api.carry(Pg + up * 0.16, Rg, 0.6)
    yield from api.carry(Pg, Rg, 1.6)
    yield from api.hold(0.2)
    note(api, f"at grasp: pen lean {lean_deg(api):.1f}")
    for k in range(2):   # re-centre the palm on the measured pen axis
        pp, Rp = api.object_in_palm("pen")
        ap = Rp[:, 2] if Rp[1, 2] > 0 else -Rp[:, 2]
        e = pp + ap * (y_t - pp[1]) / ap[1] - C
        note(api, f"pen in palm {np.round(pp, 4).tolist()} axis {np.round(ap, 3).tolist()} err {np.round(e * 1000, 1).tolist()}")
        if abs(e[0]) < 0.003 and abs(e[2]) < 0.003:
            break
        P, _ = api.palm_pose()
        yield from api.carry(P + Rg @ np.array([e[0], 0, e[2]]), Rg, 0.5)
        yield from api.hold(0.1)

    hw0 = R_PEN + PAD
    r = yield from api.move_tips(tips=chuck_targets(api, hw0 + 0.006, y_t), seconds=0.5)
    note(api, f"approach ik {r}")
    for sq in (0.005, 0.008, 0.011):
        r = yield from api.move_tips(tips=chuck_targets(api, hw0 - sq, y_t), seconds=0.6, straight=True)
        yield from api.hold(0.3)
        f = api.contact_forces("pen")
        note(api, f"squeeze {sq*1000:.0f}mm forces {np.round(f, 2).tolist()} ik {r}")
        if f[0] > 1.2 and f[1] > 0.3 and f[2] > 0.3:
            break

    # lift along the pen axis, then straight up
    P, _ = api.palm_pose()
    yield from api.carry(P + up * 0.075, Rg, 1.5)
    yield from api.carry(P + up * 0.075 + [0, 0, 0.035], Rg, 0.7)
    note(api, f"lifted: lowest {lowest(api, H)*1000:.1f}mm lean {lean_deg(api):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # move over the stroke start (tip ~25 mm up), hand on the far side from the cup
    py = np.array([-np.sin(YAW), np.cos(YAW)])
    start = home + 0.070 * py
    t = tip_w(api, H)
    P, R = api.palm_pose()
    yield from api.carry(P + np.r_[start - t[:2], 0.0], R, 1.5)

    # tilt the pen (closed loop on its measured axis) about the grip: tip leads along d
    lean_used = None
    for lean in LEANS:
        th = np.radians(lean)
        u_des = np.r_[d * np.sin(th), -np.cos(th)]          # tip direction from the grip
        ok = True
        for k in range(3):
            _, upm = pen(api)
            Rc = align(-upm, u_des)
            ang = np.degrees(np.arccos(np.clip(np.dot(-upm, u_des), -1, 1)))
            if k > 0 and ang < 1.5:
                break
            P, R = api.palm_pose()
            g = P + R @ C
            Rn = Rc @ R
            ok = yield from safe_carry(api, g - Rn @ C, Rn, 1.2 if k == 0 else 0.5)
            if not ok:
                break
            yield from api.hold(0.2)
        if ok:
            lean_used = lean
            break
    note(api, f"tilt target {lean_used} -> lean {lean_deg(api):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # hover the tip 10 mm above the stroke start
    for k in range(2):
        t = tip_w(api, H)
        low = lowest(api, H)
        P, R = api.palm_pose()
        yield from api.carry(P + np.r_[start - t[:2], 0.010 - low], R, 1.2 if k == 0 else 0.5)
        yield from api.hold(0.3)
    a0 = axial(api)
    note(api, f"above start: lowest {lowest(api, H)*1000:.1f}mm lean {lean_deg(api):.1f} tip {np.round(tip_w(api, H), 4).tolist()} axial0 {a0*1000:.2f}mm")

    # touch down slowly, closed loop on the pen's lowest point
    P, Rs = api.palm_pose()
    for i in range(400):
        low = lowest(api, H)
        if low < 0.0002:
            break
        P = P - [0, 0, 0.00005 if low < 0.002 else (0.0001 if low < 0.005 else 0.0003)]
        yield from api.carry(P.copy(), Rs, 0.02)

    z_floor = [P[2] - 0.003]

    def press_step(P):
        """Height control: descend while the tip is clear, else regulate the pen's axial push into the grip."""
        low = lowest(api, H)
        ds = axial(api) - a0
        if low > 0.0004:
            dz = -min(0.0001, 0.00003 + 0.2 * low)
        elif low < -0.0008:            # pressing into the table: back off
            dz = 0.00004
        else:
            dz = float(np.clip(0.15 * (ds - D_PRESS), -0.00004, 0.00004))
        P[2] = max(P[2] + dz, z_floor[0])   # never push more than 3 mm below the touchdown height
        return low, ds

    for i in range(25):    # settle the press before moving
        press_step(P)
        yield from api.carry(P.copy(), Rs, 0.02)
    note(api, f"touchdown: lowest {lowest(api, H)*1000:.2f}mm push {(axial(api)-a0)*1000:.2f}mm lean {lean_deg(api):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # stroke along d (tip leading), height regulated
    n = 200
    lows, pushes = [], []
    t0 = tip_w(api, H)[:2]
    for i in range(n):
        low, ds = press_step(P)
        lows.append(low)
        pushes.append(ds)
        P[:2] = P[:2] + d * STROKE / n
        yield from api.carry(P.copy(), Rs, 0.02)
        if i % 40 == 0:
            note(api, f"stroke {i}: lowest {low*1000:.2f}mm push {ds*1000:.2f}mm lean {lean_deg(api):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")
    for i in range(10):
        press_step(P)
        yield from api.carry(P.copy(), Rs, 0.02)
    lows = np.array(lows)
    note(api, f"stroke done: frac<0.5mm {np.mean(lows < 0.0005):.2f} push mean {np.mean(pushes)*1000:.2f} max {np.max(pushes)*1000:.2f}mm "
              f"lean {lean_deg(api):.1f} tip moved {np.linalg.norm(tip_w(api, H)[:2] - t0)*1000:.1f}mm")

    # lift and hold
    yield from api.carry(P + [0, 0, 0.07], Rs, 1.2)
    yield from api.hold(1.5)
    note(api, f"final: lowest {lowest(api, H)*1000:.1f}mm lean {lean_deg(api):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")
