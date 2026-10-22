import numpy as np

TABLE = 0.75
RIM = TABLE + 0.060
PINCH_C = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame
AX = np.array([0.0, 0.0, -1.0])             # pinch axis, centre -> thumb
GRIP_N = 2.5


def smooth(u):
    return u * u * (3 - 2 * u)


def pen_axis(api):
    """Pen centre, unit vector toward its lower end, lean [rad]."""
    p, R = api.object_pose("pen")
    a = R[:, 2] if R[2, 2] < 0 else -R[:, 2]
    return p, a, float(np.arccos(min(1.0, abs(R[2, 2]))))


def tip_state(api, ell):
    p, a, lean = pen_axis(api)
    tip = p + a * ell
    return tip, tip[2] - TABLE, lean


def pen_in_palm_axis(api):
    _, R = api.object_in_palm("pen")
    return R[:, 2].copy()


def pivot(a0, a1):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a0 @ a1))))))


def in_contact(h, lean):
    return h < 0.0042 * (1 - np.cos(lean)) + 0.00025


def run(api):
    diag = []
    try:
        yield from _run(api, diag)
    finally:
        for line in diag:
            api.note(line)


def _run(api, diag):
    yield from api.hold(0.4)
    p, a, lean0 = pen_axis(api)
    ell = (p[2] - TABLE) / max(0.5, -a[2])          # centre -> lower end (scorer's tip point)
    home = (p + a * ell)[:2].copy()
    top_z = p[2] - a[2] * ell
    zg = max(RIM + 0.030, top_z - 0.022)            # grasp height on the pen
    s_g = (zg - p[2]) / a[2]
    g = p + a * s_g                                  # pen axis point at grasp height
    diag.append(f"ell={ell*1000:.1f}mm lean0={np.degrees(lean0):.1f} top={top_z:.4f} zg={zg:.4f} home={np.round(home,4).tolist()}")

    R0 = api.hand_down()
    # 1) tuck middle/ring/little, open thumb-index pinch, go above the pen
    yield from api.tuck(["middle", "ring", "little"], seconds=0.5)
    open_tips = {"thumb": PINCH_C + 0.030 * AX, "index": PINCH_C - 0.030 * AX}
    r = yield from api.move_tips(xyz=api.palm_for(g + [0, 0, 0.07], PINCH_C, R0), R=R0, tips=open_tips, seconds=2.0)
    diag.append(f"open ik={r['ik_residual_mm']}")
    # 2) descend around the pen
    yield from api.carry(xyz=api.palm_for(g, PINCH_C, R0), R=R0, seconds=1.3)
    yield from api.hold(0.2)
    # 3) centre the pads on the actual pen point (measured palm frame), close, force-grip
    Pm, Rm = api.measured_palm()
    p, a, _ = pen_axis(api)
    g = p + a * ((zg - p[2]) / a[2])
    gp = Rm.T @ (g - Pm)
    diag.append(f"pen point in palm={np.round(gp*1000,1).tolist()} (pinch centre {(PINCH_C*1000).tolist()})")
    yield from api.move_tips(tips={"thumb": gp + 0.020 * AX, "index": gp - 0.020 * AX}, seconds=0.4, straight=True)
    r = yield from api.move_tips(tips={"thumb": gp + 0.0135 * AX, "index": gp - 0.0135 * AX}, seconds=0.5, straight=True)
    diag.append(f"close ik={r['ik_residual_mm']}")
    r = yield from api.grip("pen", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.2, max_travel=0.02)
    diag.append(f"grip {r}")

    # 4) lift straight up out of the cup
    cmd = api.palm_pose()[0].copy()
    yield from api.carry(xyz=cmd + [0, 0, 0.085], seconds=1.6)
    yield from api.hold(0.2)
    tip, h, lean = tip_state(api, ell)
    if h < 0.070:
        cmd = api.palm_pose()[0].copy()
        yield from api.carry(xyz=cmd + [0, 0, 0.075 - h], seconds=0.6)
        tip, h, lean = tip_state(api, ell)
    diag.append(f"lifted tip_h={h*1000:.1f}mm lean={np.degrees(lean):.1f} gs={api.grip_state()}")

    # 5) carry to the stroke start: sideways, then down in stages, settling the arm each time
    d = R0[:, 1].copy(); d[2] = 0; d /= np.linalg.norm(d)          # drag toward palm +y (thumb side)
    f = R0[:, 2].copy(); f[2] = 0; f /= np.linalg.norm(f)          # finger direction
    L = 0.055
    start = home - 0.080 * f[:2] - 0.030 * d[:2]
    tip, h, _ = tip_state(api, ell)
    cmd = api.palm_pose()[0].copy()
    cmd[:2] += start - tip[:2]
    yield from api.carry(xyz=cmd.copy(), seconds=2.0)
    yield from api.hold(0.3)
    for above, secs in ((0.025, 1.8), (0.007, 1.2)):
        tip, h, _ = tip_state(api, ell)
        cmd = api.palm_pose()[0].copy()
        cmd[:2] += start - tip[:2]
        cmd[2] += above - h
        yield from api.carry(xyz=cmd.copy(), seconds=secs)
        yield from api.hold(0.5)
        tip, h, lean = tip_state(api, ell)
        diag.append(f"stage {above*1000:.0f}mm: tip_h={h*1000:.1f} lean={np.degrees(lean):.1f} arm_err_z={(api.palm_pose()[0][2]-api.measured_palm()[0][2])*1000:.1f}mm")
    ax_ref = pen_in_palm_axis(api)

    # 6) touch down slowly (3 mm/s); on contact, park the command where the arm actually is
    zmin = TABLE + 0.09
    cmd = api.palm_pose()[0].copy()
    off = cmd[2] - api.measured_palm()[0][2]          # static arm offset (cmd - measured)
    for k in range(200):
        tip, h, lean = tip_state(api, ell)
        if in_contact(h, lean):
            break
        cmd[2] = max(zmin, cmd[2] - 0.00006)
        yield from api.carry(xyz=cmd.copy(), seconds=0.02)
    cmd[2] = api.measured_palm()[0][2] + off
    piv = pivot(ax_ref, pen_in_palm_axis(api))
    diag.append(f"touchdown t={api.t:.2f} h={h*1000:.2f}mm lean={np.degrees(lean):.1f} pivot={piv:.2f}deg steps={k}")
    yield from api.carry(xyz=cmd.copy(), seconds=0.1)

    # 7) stroke: drag along d; lower slowly when contact is lost, back off when the pen pivots in the grip
    n = 200
    xy0 = cmd[:2].copy()
    ink, lmax, pmax, t0 = 0, 0.0, 0.0, api.t
    tip0 = tip_state(api, ell)[0]
    piv_prev = pivot(ax_ref, pen_in_palm_axis(api))
    miss = 0
    for k in range(1, n + 1):
        tip, h, lean = tip_state(api, ell)
        piv = pivot(ax_ref, pen_in_palm_axis(api))
        lmax, pmax = max(lmax, lean), max(pmax, piv)
        if in_contact(h, lean):
            ink += 1; miss = 0
        else:
            miss += 1
            if miss >= 2:
                cmd[2] = max(zmin, cmd[2] - 0.00005)
        if piv - piv_prev > 0.04 or lean > np.radians(40):
            cmd[2] += 0.00015
        piv_prev = piv
        cmd[:2] = xy0 + smooth(k / n) * L * d[:2]
        if k % 25 == 0:
            diag.append(f"  k={k} h={h*1000:.2f} lean={np.degrees(lean):.1f} pivot={piv:.1f} gs={[v['force_N'] for v in api.grip_state().values()]}")
        yield from api.carry(xyz=cmd.copy(), seconds=0.02)
    tip, h, lean = tip_state(api, ell)
    diag.append(f"stroke {t0:.2f}-{api.t:.2f}: contact {ink}/{n} max_lean={np.degrees(lmax):.1f} max_pivot={pmax:.1f} "
                f"tip moved {np.linalg.norm(tip[:2]-tip0[:2])*1000:.1f}mm, dist from home {np.linalg.norm(tip[:2]-home)*1000:.0f}mm")

    # 8) lift and hold
    cmd = api.palm_pose()[0].copy()
    yield from api.carry(xyz=cmd + [0, 0, 0.07], seconds=1.2)
    yield from api.hold(0.8)
    tip, h, lean = tip_state(api, ell)
    diag.append(f"final tip_h={h*1000:.1f}mm lean={np.degrees(lean):.1f} gs={api.grip_state()}")
    yield from api.hold(0.5)
