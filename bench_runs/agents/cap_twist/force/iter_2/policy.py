"""cap_twist, force level: press index+middle pads down on the lid top (force servo),
sweep them about the jar axis (rotate primitive, world frame), lift, return, repeat
until the measured lid angle passes 360 deg + margin. Palm fixed during all strokes."""
import numpy as np
from scipy.spatial.transform import Rotation

LID_TOP = 0.123          # lid top face above jar base [m]
PAD = 0.008              # pad centre sits this far inside the fingertip surface
R_PAD = 0.020            # pad radius from the lid axis (inside the smallest lid, 31.5 mm)
SPREAD = np.radians(35)  # index / middle angular offset either side of the finger direction
HOVER = 0.004            # pad surface above the lid top before pressing
LIFT = 0.025             # pad surface above the lid top while returning
F0 = 2.5                 # pressing force per pad [N]
GOAL = 2 * np.pi + np.radians(20)
T_END = 53.0
WORK = ("index", "middle")
DOWN = {"index": (0, 0, -1.0), "middle": (0, 0, -1.0)}


def pads(top, a, phi, h):
    """Palm-frame pad targets for index/middle at sweep angle phi, pad surface h above the top."""
    u = np.array([0, 0, 1.0]) - a * a[2]
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    Rm = Rotation.from_rotvec(a * phi).as_matrix()
    out = {}
    for k, f in enumerate(WORK):
        s = 1 if k == 0 else -1
        off = R_PAD * (s * np.sin(SPREAD) * v + np.cos(SPREAD) * u)
        out[f] = top + Rm @ off + a * (PAD + h)
    return out


def check(api, top, a, alpha, limit=0.0025):
    worst = 0.0
    for h, phi in [(0.0, 0.0), (0.0, -alpha), (0.0, alpha), (-0.008, -alpha), (-0.008, alpha),
                   (LIFT, -alpha), (LIFT, alpha), (LIFT, 0.0)]:
        for f, x in pads(top, a, phi, h).items():
            worst = max(worst, api.reach(f, x))
            if worst > limit:
                return worst
    return worst


def run(api):
    info, strokes = [], []
    try:
        yield from _run(api, info, strokes)
    finally:
        api.note("INFO " + " | ".join(info))
        for i in range(0, len(strokes), 6):
            api.note("STROKES " + " ; ".join(strokes[i:i + 6]))


def _run(api, info, strokes):
    R0 = api.hand_down(0.7)
    jp, _ = api.object_pose("jar")
    top_w = jp + np.array([0, 0, LID_TOP])
    lid0 = api.object_joint("lid_twist")

    # idle fingers straight (up at palm level, clear of the lid)
    yield from api.move_joints(q={"ring": [0, 0, 0, 0], "little": [0.2, 0, 0, 0]}, seconds=0.4)
    tips = api.tips()
    info.append("tips0 " + str(np.round(tips * 1000).astype(int).tolist()))
    yc = 0.5 * (tips[1, 1] + tips[2, 1])
    a_nom = np.array([-1.0, 0, 0])

    best = None
    for alpha in np.radians([32, 25, 18]):
        feas = []
        for D in np.arange(0.05, 0.111, 0.01):
            for zc in np.arange(0.07, 0.171, 0.01):
                w = check(api, np.array([D, yc, zc]), a_nom, alpha)
                if w < 0.0025:
                    feas.append((D, zc, w))
        if feas:
            F = np.array(feas)
            cen = F[:, :2].mean(0)
            j = int(np.argmin(np.linalg.norm((F[:, :2] - cen) / [1.0, 1.0], axis=1)))
            best = (alpha, F[j, 0], F[j, 1], F[j, 2], len(feas))
            info.append(f"feasible alpha{np.degrees(alpha):.0f} n={len(feas)} " +
                        str([(round(d * 1000), round(z * 1000)) for d, z, _ in feas]))
            break
    if best is None:
        alpha, D, zc = np.radians(18), 0.08, 0.12
        info.append("NO FEASIBLE; default")
    else:
        alpha, D, zc = best[0], best[1], best[2]
    info.append(f"chosen alpha={np.degrees(alpha):.0f} D={D * 1000:.0f} yc={yc * 1000:.0f} zc={zc * 1000:.0f}")
    top_p = np.array([D, yc, zc])
    # thumb clearance
    th = tips[0]
    dh = np.hypot(th[1] - yc, th[2] - zc)
    info.append(f"thumb pad {np.round(th * 1000).astype(int).tolist()} dist {dh * 1000:.0f}mm")

    P = api.palm_for(top_w, top_p, R0)
    pre = P + np.array([0, 0, 0.07])
    yield from api.move_tips(xyz=pre, R=R0, tips=pads(top_p, a_nom, -alpha, LIFT), seconds=2.0)
    yield from api.carry(xyz=P, R=R0, seconds=1.2)
    yield from api.hold(0.3)

    p, Rj = api.object_in_palm("jar")
    top = p + Rj @ np.array([0, 0, LID_TOP])
    a = Rj @ np.array([0, 0, 1.0])
    info.append(f"meas top {np.round(top * 1000, 1).tolist()} axis {np.round(a, 3).tolist()}")
    yield from api.move_tips(tips=pads(top, a, -alpha, LIFT), seconds=0.3)

    F = F0
    n = 0
    while api.object_joint("lid_twist") - lid0 < GOAL and api.t < T_END:
        n += 1
        yield from api.move_tips(tips=pads(top, a, -alpha, HOVER), seconds=0.3, straight=True)
        g = yield from api.grip("jar", {f: F for f in WORK}, directions=DOWN, seconds=0.6)
        a0 = api.object_joint("lid_twist")
        yield from api.rotate("jar", (0, 0, 1.0), 2 * alpha, list(WORK), seconds=1.2, frame="world",
                              centre=(0, 0, LID_TOP))
        st = api.grip_state()
        a1 = api.object_joint("lid_twist")
        yield from api.ungrip()
        yield from api.move_tips(tips=pads(top, a, alpha, LIFT), seconds=0.3, straight=True)
        yield from api.rotate("jar", (0, 0, 1.0), -2 * alpha, list(WORK), seconds=0.6, frame="world",
                              centre=(0, 0, LID_TOP))
        a2 = api.object_joint("lid_twist")
        gain = a1 - a0
        strokes.append(f"{n} t{api.t:.1f} F{F:.1f} g{[round(x, 1) for x in g['forces'][1:3]]} "
                       f"sw{np.degrees(gain):.0f} back{np.degrees(a2 - a1):.0f} "
                       f"f{[st[f]['force_N'] for f in WORK]} tr{[st[f]['travel_mm'] for f in WORK]} "
                       f"tot{np.degrees(a2 - lid0):.0f}")
        if gain < 0.6 * 2 * alpha:
            F = min(F + 1.0, 4.5)
    info.append(f"done strokes={n} turn={np.degrees(api.object_joint('lid_twist') - lid0):.0f} t={api.t:.1f}")
    yield from api.hold(0.2)
