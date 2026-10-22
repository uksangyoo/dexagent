"""pen_writing_grasp, level force (phase 3: reuses earlier agents' helpers).

Grasp: three-finger chuck (layered agent's design): hand rolled thumb-down (palm y = world down),
pen along palm y, thumb pad on the wrist side opposing index + middle pads 22 mm apart along the pen.
Force layer: the thumb alone is force-servoed (index/middle are position anchors, so the grip keeps a
positional stiffness against tip torques instead of drifting), which keeps the squeeze constant through
the lift, the re-orientation and the stroke.
Stroke: pen tilted ~30 deg with the tip LEADING the motion (table reaction nearly along the pen),
in the chuck's stiff plane (palm y-z), toward the arm base; height regulated on the hand-pen compression.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TABLE = 0.75
RIM = TABLE + 0.060
R_PEN = 0.00375
PAD = 0.008
YAW = 0.7
X0, Z0 = 0.080, 0.118
Y_T, Y_I, Y_M = 0.029, 0.040, 0.018
F_THUMB = 3.5
LEANS = (30.0, 22.0, 14.0)      # stroke lean (tip leading); fallbacks if the arm cannot reach
START_X, START_D = 0.055, 0.060   # stroke start (tip) from the cup: along palm x and along d [m]
STROKE = 0.050
HOVER = 0.008
PUSH_T = 0.0004                 # target hand-pen compression during the stroke [m]

LOG = []


def note(api, s):
    LOG.append(f"{api.t:6.2f} {s}")


def smooth(u):
    return u * u * (3 - 2 * u)


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


def align(a, b):
    """Rotation taking unit vector a to unit vector b (layered agent's helper)."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / s * np.arctan2(s, np.dot(a, b))).as_matrix()


def chuck_targets(api, hw_t, hw_f, y_t):
    """Chuck pad targets around the measured pen axis (palm frame); thumb -z side, index/middle +z side."""
    pp, Rp = api.object_in_palm("pen")
    ap = Rp[:, 2]
    ap = ap if ap[1] > 0 else -ap
    at = lambda y: pp + ap * (y - pp[1]) / ap[1]
    ez = np.array([0, 0, 1.0])
    return {"thumb": at(y_t) - ez * hw_t, "index": at(Y_I) + ez * hw_f, "middle": at(Y_M) + ez * hw_f}


def thumb_dir(api):
    """Thumb closing direction (palm +z, made perpendicular to the pen) in the pen's object frame."""
    _, Rp = api.object_in_palm("pen")
    a = Rp[:, 2]
    v = np.array([0, 0, 1.0])
    v = v - a * (a @ v)
    v /= np.linalg.norm(v) + 1e-9
    return Rp.T @ v


def regrip(api, tag):
    r = yield from api.grip("pen", {"thumb": F_THUMB}, directions={"thumb": thumb_dir(api)}, seconds=0.02, max_travel=0.012)
    note(api, f"regrip {tag}: {np.round(api.contact_forces('pen'), 2).tolist()} {api.grip_state()}")
    return r


def tip_in_palm(api, H):
    Pm, Rm = api.measured_palm()
    return Rm.T @ (tip_w(api, H) - Pm)


def compression(api, H, ref):
    """How far the pen tip has been pushed up relative to the hand since `ref` [m]."""
    _, Rm = api.measured_palm()
    return float((tip_in_palm(api, H) - ref) @ (Rm.T @ np.array([0, 0, 1.0])))


ST = {"on": False, "k": 0, "yt": Y_T, "diag": None}
MAXT = 0.007
ONE3, XONLY = np.ones(3), np.array([1.0, 0.0, 0.0])


def tick(api, P=None, R=None):
    """One 20 ms control step. Every 3rd step re-issues the thumb grip with its closing direction recomputed
    from the palm frame (object-frame directions rotate with the pen: a pen that spins or tilts in the grip
    would otherwise be pushed sideways/axially by its own servo)."""
    ST["k"] += 1
    if ST["on"] and ST["k"] % 3 == 0:
        yield from api.grip("pen", {"thumb": F_THUMB}, directions={"thumb": thumb_dir(api)}, seconds=0.02, max_travel=MAXT)
    if ST["diag"] is not None and ST["k"] % 25 == 0:
        note(api, "  " + ST["diag"] + " " + state(api))
    if ST.get("anchor") is not None:
        # anchor servo: integral loop keeping the index/middle pads' MEASURED positions (palm frame) where they were
        # when the grip settled, so finger sag under the changing gravity direction cannot let the pen pivot out
        # thumb: same loop on its palm-x coordinate only (z belongs to the force servo): a laterally compliant
        # thumb lets the pen roll/squirt sideways over the domed index/middle pads (seen in iter 3: x 82 -> 74 mm)
        meas = api.tips()
        for f, i, mask in (("index", 1, ONE3), ("middle", 2, ONE3), ("thumb", 0, XONLY)):
            c = ST["cmd"][f] + mask * np.clip(0.15 * (ST["anchor"][f] - meas[i]), -0.0002, 0.0002)
            off = c - ST["cmd0"][f]
            n = np.linalg.norm(off)
            ST["cmd"][f] = ST["cmd0"][f] + (off if n < 0.008 else off * 0.008 / n)
        yield from api.move_tips(xyz=P, R=R, tips={f: ST["cmd"][f].copy() for f in ("thumb", "index", "middle")}, seconds=0.02)
    elif P is None:
        yield from api.hold(0.02)
    else:
        yield from api.carry(P, R, 0.02)


def hold_t(api, seconds):
    for _ in range(max(1, round(seconds * 50))):
        yield from tick(api)


def state(api):
    """Pen axis point at thumb height in the palm frame (mm), axis tilt in palm (deg), pad forces, thumb travel."""
    pp, Rp = api.object_in_palm("pen")
    ap = Rp[:, 2] if Rp[1, 2] > 0 else -Rp[:, 2]
    q = pp + ap * (ST["yt"] - pp[1]) / ap[1]
    f = api.contact_forces("pen")
    gs = api.grip_state().get("thumb", {})
    return (f"ax@thumb x{q[0]*1000:.1f} z{q[2]*1000:.1f} tilt_yz {np.degrees(np.arctan2(ap[2], ap[1])):.1f} "
            f"tilt_x {np.degrees(np.arctan2(ap[0], ap[1])):.1f} f {np.round(f[:3], 2).tolist()} palm {f[5]:.2f} tr {gs.get('travel_mm')} lean {lean_deg(api):.1f}"
            + ("" if ST.get("anchor") is None else
               f" anc_off_mm {[round(float(np.linalg.norm(ST['cmd'][k] - ST['cmd0'][k])) * 1000, 1) for k in ('thumb', 'index', 'middle')]}"
               f" pad_dx_mm {[round(float(api.tips()[i][0] - q[0]) * 1000, 1) for i in (0, 1, 2)]}"
               f" thumb_d_mm {np.round((api.tips()[0] - ST['anchor']['thumb']) * 1000, 1).tolist()}"))


def move_grip(api, C, g1, R1, seconds):
    """Per-step arm motion carrying palm point C along a straight world line to g1 while slerping to R1.
    Returns True, or False after an arm IK failure (command snapped back to the measured palm)."""
    P0, R0 = api.palm_pose()
    g0 = P0 + R0 @ C
    sl = Slerp([0, 1], Rotation.from_matrix([R0, R1]))
    n = max(1, round(seconds * 50))
    for k in range(1, n + 1):
        u = smooth(k / n)
        R = sl(u).as_matrix()
        g = (1 - u) * g0 + u * np.asarray(g1, float)
        try:
            yield from tick(api, g - R @ C, R)
        except ValueError as e:
            note(api, f"ARM IK fail u={u:.2f}: {e}")
            Pm, Rm = api.measured_palm()
            yield from api.carry(Pm, Rm, 0.02)
            yield from hold_t(api, 0.2)
            return False
    return True


def lost(api):
    f = api.contact_forces("pen")
    return lean_deg(api) > 75 or (f[0] < 0.05 and f[1] < 0.05 and f[2] < 0.05)


def run(api):
    try:
        yield from _run(api)
    finally:
        for s in LOG:
            api.note(s)


def _run(api):
    R0 = api.hand_down(YAW)
    Rs = R0 @ Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()   # thumb down, palm y = world down
    xs = Rs[:2, 0] / np.linalg.norm(Rs[:2, 0])
    d = -Rs[:2, 2] / np.linalg.norm(Rs[:2, 2])                        # stroke direction: toward the wrist/base

    # ---- preshape (layered agent's chuck)
    y_t = Y_T
    yield from api.tuck(["ring", "little"], seconds=0.5)
    wide = 0.028
    pre = {"thumb": np.array([X0, Y_T, Z0 - wide]), "index": np.array([X0, Y_I, Z0 + wide]),
           "middle": np.array([X0, Y_M, Z0 + wide])}
    r = yield from api.move_tips(tips=pre, seconds=0.6)
    if api.reach("thumb", [X0, Y_T, Z0 - 0.012]) > 0.003:
        y_t = 0.034
        note(api, f"thumb reach fallback y={y_t}")
    yield from api.hold(0.2)

    p, up = pen(api)
    H = (p[2] - TABLE - R_PEN) / up[2]
    home = (p - up * (H + R_PEN))[:2]
    top = p[2] + up[2] * (H + R_PEN)
    note(api, f"pen H={H*1000:.1f}mm lean={lean_deg(api):.1f} home={np.round(home, 4).tolist()} top={top:.4f} low={lowest(api, H)*1000:.2f}")

    # ---- approach: turn high above the pen, then come down along its axis (layered iter 4)
    zc = max(top - 0.020 - (y_t - Y_M), RIM + 0.018 + (Y_I - y_t))
    gc = p + up * ((zc - p[2]) / up[2])
    Rg = align(np.array([0, 0, -1.0]), -up) @ Rs
    C = np.array([X0, y_t, Z0])
    Pg = api.palm_for(gc, C, Rg)
    yield from api.carry(Pg + up * 0.16 + np.array([0, 0, 0.03]), Rg, 2.0)
    yield from api.carry(Pg + up * 0.16, Rg, 0.6)
    yield from api.carry(Pg, Rg, 1.6)
    yield from api.hold(0.2)
    for k in range(2):
        pp, Rp = api.object_in_palm("pen")
        ap = Rp[:, 2] if Rp[1, 2] > 0 else -Rp[:, 2]
        e = pp + ap * (y_t - pp[1]) / ap[1] - C
        note(api, f"pen in palm {np.round(pp, 4).tolist()} err {np.round(e * 1000, 1).tolist()}")
        if abs(e[0]) < 0.003 and abs(e[2]) < 0.003:
            break
        P, _ = api.palm_pose()
        yield from api.carry(P + Rg @ np.array([e[0], 0, e[2]]), Rg, 0.5)
        yield from api.hold(0.1)

    # ---- close: index/middle as position anchors 4 mm into the pen, thumb force-servoed
    hw0 = R_PEN + PAD
    yield from api.move_tips(tips=chuck_targets(api, hw0 + 0.006, hw0 + 0.006, y_t), seconds=0.5)
    r = yield from api.move_tips(tips=chuck_targets(api, hw0 - 0.002, hw0 - 0.004, y_t), seconds=0.6, straight=True)
    yield from api.hold(0.2)
    note(api, f"closed: forces {np.round(api.contact_forces('pen'), 2).tolist()} ik {r.get('ik_residual_mm')}")
    r = yield from api.grip("pen", {"thumb": F_THUMB}, directions={"thumb": thumb_dir(api)}, seconds=1.0, max_travel=MAXT)
    ST["on"], ST["yt"] = True, y_t
    meas, cmd = api.tips(), api.tip_targets()
    ST["anchor"] = {"thumb": meas[0].copy(), "index": meas[1].copy(), "middle": meas[2].copy()}
    ST["cmd"] = {"thumb": cmd[0].copy(), "index": cmd[1].copy(), "middle": cmd[2].copy()}
    ST["cmd0"] = {"thumb": cmd[0].copy(), "index": cmd[1].copy(), "middle": cmd[2].copy()}
    note(api, f"grip {r} | {state(api)}")
    Z3 = np.zeros(3)

    # ---- lift out along the pen axis, then up until the tip is ~80 mm high
    P, _ = api.palm_pose()
    ST["diag"] = "lift"
    yield from move_grip(api, Z3, P + up * 0.075, Rg, 1.5)
    P, _ = api.palm_pose()
    yield from move_grip(api, Z3, P + [0, 0, max(0.0, 0.082 - lowest(api, H))], Rg, 0.6)
    yield from hold_t(api, 0.2)
    note(api, f"lifted: low {lowest(api, H)*1000:.1f}mm | {state(api)}")
    if lost(api):
        note(api, "pen lost after lift")
        yield from hold_t(api, 1.0)
        return

    # ---- tilt (tip leading along d) while carrying the tip over the stroke start
    S = np.r_[home + START_X * xs + START_D * d, TABLE + R_PEN + HOVER + 0.010]
    lean_used = None
    ST["diag"] = "tilt"
    for lean in LEANS:
        th = np.radians(lean)
        u_des = np.r_[d * np.sin(th), -np.cos(th)]
        P, R = api.palm_pose()
        _, upm = pen(api)
        R1 = align(-upm, u_des) @ R
        tp = tip_in_palm(api, H)
        P1 = S - R1 @ tp
        ok = yield from move_grip(api, C, P1 + R1 @ C, R1, 3.0)
        if ok:
            lean_used = lean
            break
    yield from hold_t(api, 0.3)
    note(api, f"tilt {lean_used}: low {lowest(api, H)*1000:.1f} | {state(api)}")
    if lost(api):
        note(api, "pen lost in tilt")
        yield from hold_t(api, 1.0)
        return
    if lean_used is None:
        lean_used = 0.0

    # closed-loop lean correction + hover over the start
    th = np.radians(lean_used)
    u_des = np.r_[d * np.sin(th), -np.cos(th)]
    ST["diag"] = "hover"
    for k in range(2):
        _, upm = pen(api)
        err = np.degrees(np.arccos(np.clip(-upm @ u_des, -1, 1)))
        P, R = api.palm_pose()
        R1 = align(-upm, u_des) @ R if err > 2.0 else R
        tp = tip_in_palm(api, H)
        # tip point target such that the pen's lowest point is HOVER above the table
        S_h = np.r_[home + START_X * xs + START_D * d, TABLE + HOVER + R_PEN * (1 - np.cos(th))]
        P1 = S_h - R1 @ tp
        yield from move_grip(api, C, P1 + R1 @ C, R1, 1.0 if k == 0 else 0.6)
        yield from hold_t(api, 0.3)
    ref = tip_in_palm(api, H)
    note(api, f"hover: low {lowest(api, H)*1000:.2f}mm tip {np.round(tip_w(api, H), 4).tolist()} dist_home {np.linalg.norm(tip_w(api, H)[:2]-home)*1000:.0f}mm | {state(api)}")

    # ---- touch down slowly on the measured tip height / hand-pen compression
    P, R = api.palm_pose()
    ST["diag"] = None
    for i in range(500):
        low = lowest(api, H)
        c = compression(api, H, ref)
        if (low < 0.0015 and c > 0.00015) or low < -0.0003:
            break
        P[2] -= 0.00006 if low > 0.0015 else 0.00003
        yield from tick(api, P.copy(), R)
    z_td = P[2] - max(0.0, lowest(api, H))
    note(api, f"touchdown i={i}: low {lowest(api, H)*1000:.2f}mm comp {compression(api, H, ref)*1000:.2f}mm | {state(api)}")

    def press_step(P):
        low = lowest(api, H)
        c = compression(api, H, ref)
        if low > 0.0004:                      # tip off the table: come down
            dz = -0.00004
        elif c > 0.0009 or low < -0.0006:     # pressing too hard: back off
            dz = 0.00003
        elif c < 0.0004:
            dz = -0.00001
        else:
            dz = 0.0
        P[2] = float(np.clip(P[2] + dz, z_td - 0.0025, z_td + 0.002))
        return low, c

    for i in range(20):
        press_step(P)
        yield from tick(api, P.copy(), R)

    # ---- stroke along d, tip leading
    n = 175
    t0 = tip_w(api, H)[:2].copy()
    lows, comps = [], []
    ST["diag"] = "stroke"
    for i in range(n):
        low, c = press_step(P)
        lows.append(low); comps.append(c)
        P[:2] = P[:2] + d * STROKE / n
        yield from tick(api, P.copy(), R)
    for i in range(10):
        press_step(P)
        yield from tick(api, P.copy(), R)
    lows = np.array(lows)
    note(api, f"stroke done: frac(low<1mm) {np.mean(lows < 0.001):.2f} comp mean {np.mean(comps)*1000:.2f} max {np.max(comps)*1000:.2f} "
              f"tip moved {np.linalg.norm(tip_w(api, H)[:2] - t0)*1000:.1f}mm dist_home {np.linalg.norm(tip_w(api, H)[:2]-home)*1000:.0f} | {state(api)}")

    # ---- lift and hold
    ST["diag"] = "final"
    yield from move_grip(api, Z3, P + [0, 0, 0.065], R, 1.5)
    yield from hold_t(api, 1.2)
    note(api, f"final: low {lowest(api, H)*1000:.1f}mm | {state(api)}")
