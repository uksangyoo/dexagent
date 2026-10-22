# pen_gravity_pivot at level layered
#
# Plan: thumb + one finger pinch the pen off-centre with the pinch line along
# palm z (horizontal), which is a free hinge for the pen (no torsional
# friction).  A neighbouring finger presses the short end from above so the
# pen stays level for the lift.  Then the hand lowers the pen beside the rest,
# the press finger backs off so the long end settles on the table, and the
# hand moves the hinge on an arc over the resting tip: the pen turns in the
# fingers quasi-statically until it hangs vertical, then the hand lifts.

import numpy as np

TABLE = 0.75
PAD = 0.008
RPEN = 0.00375
FN = ("thumb", "index", "middle", "ring", "little")
EZ = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rot(axis, ang):
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def run(api):
    diag = []
    try:
        yield from _policy(api, diag)
    except Exception as e:
        diag.append(f"EXC {type(e).__name__}: {str(e)[:300]}")
        raise
    finally:
        for d in diag[-40:]:
            api.note(d)


def _policy(api, diag):
    def pen():
        p, R = api.object_pose("pen")
        return p, R[:, 2].copy()

    p0, a = pen()
    if a[1] < 0:
        a = -a
    psi = float(np.arctan2(a[1], a[0]) - np.pi / 2)
    Rh = api.hand_down(psi)
    try:
        pr, _ = api.object_pose("pen_rest")
        s_rest = float((pr - p0) @ a)
    except Exception as e:  # noqa
        s_rest = None
    cfg = "P" if (s_rest is None or s_rest > 0) else "M"
    if cfg == "P":   # pinch thumb+index at the -a end, long part toward palm +y, middle presses
        pf, sf, sgn, ys = 1, 2, +1, (0.028, 0.034, 0.040)
    else:            # pinch thumb+middle at the +a end, long part toward palm -y, index presses
        pf, sf, sgn, ys = 2, 1, -1, (0.004, 0.010, 0.016)
    sA = -sgn * 0.030
    W = p0 + sA * a
    diag.append(f"cfg={cfg} s_rest={s_rest} psi={psi:.3f} p0={np.round(p0, 4).tolist()}")

    ref = {}

    def snap(tag):
        p, an = pen()
        P, Rm = api.measured_palm()
        ap = Rm.T @ an
        f = api.contact_forces("pen")
        th = float(np.degrees(np.arcsin(min(1.0, abs(an[2])))))
        piv = ""
        if "ap" in ref:
            piv = f" piv={np.degrees(np.arccos(min(1.0, abs(float(ap @ ref['ap']))))):.0f}"
        tw = api.tips_world()
        mid = 0.5 * (tw[0] + tw[pf])
        sh = float((mid - p) @ an)
        dh = float(np.linalg.norm(mid - p - sh * an))
        diag.append(f"{tag} t={api.t:.2f} lift={1000 * (p[2] - p0[2]):.0f} tiltH={th:.1f}{piv} "
                    f"pinch_s={1000 * sh:.0f} pinch_d={1000 * dh:.0f} "
                    f"ap={np.round(ap, 2).tolist()} F={np.round(f, 2).tolist()}")
        return th, ap

    def hinge():
        tw = api.tips_world()
        mid = 0.5 * (tw[0] + tw[pf])
        p, an = pen()
        return p + an * float((mid - p) @ an), p, an

    # ---- approach above the pinch point, hand open
    A_nom = np.array([0.080, ys[1], 0.120])
    P_up = api.palm_for(W, A_nom, Rh) + np.array([0, 0, 0.07])
    yield from api.move_joints(xyz=P_up, R=Rh, q=api.OPEN, seconds=2.0)

    # ---- choose pinch centre / press direction by fingertip reachability
    hw = RPEN + PAD - 0.0022
    rs = 0.022
    ey = np.array([0, 1.0, 0])
    best, rows = None, []
    for ax in (0.07, 0.08, 0.09):
        for ay in ys:
            for az in (0.112, 0.122, 0.132):
                A = np.array([ax, ay, az])
                rt = api.reach("thumb", A - hw * EZ)
                rp = api.reach(FN[pf], A + hw * EZ)
                if rt > 0.003 or rp > 0.003:
                    rows.append(f"{ax:.2f},{ay:.3f},{az:.3f}:{rt*1e3:.0f}/{rp*1e3:.0f}")
                    continue
                rsf, kk = 1.0, 0
                for kk, phi in enumerate((0.35, 0.8, 1.2)):
                    n = np.array([-np.cos(phi), 0, np.sin(phi)])
                    rsf = api.reach(FN[sf], A - sgn * rs * ey + hw * n)
                    if rsf < 0.002:
                        break
                rows.append(f"{ax:.2f},{ay:.3f},{az:.3f}:{rt*1e3:.0f}/{rp*1e3:.0f}/{rsf*1e3:.0f}k{kk}")
                score = 1000 * (rt + rp + rsf) + kk + 0.3 * abs(ax - 0.08) / 0.01 + 0.3 * abs(az - 0.122) / 0.01
                if best is None or score < best[0]:
                    best = (score, A, (0.35, 0.8, 1.2)[kk])
    diag.append("reach " + " ".join(rows))
    if best is None:
        best = (0, np.array([0.08, ys[1], 0.118]), 1.2)
    _, A, phi = best
    diag.append(f"chosen A={A.tolist()} phi={phi}")

    n0 = np.array([-np.cos(phi), 0, np.sin(phi)])
    yield from api.move_tips(tips={"thumb": A - 0.026 * EZ, FN[pf]: A + 0.026 * EZ,
                                   FN[sf]: A - sgn * rs * ey + 0.03 * n0}, seconds=0.8)
    P_grasp = api.palm_for(W, A, Rh)
    yield from api.carry(xyz=P_grasp, seconds=1.4)
    yield from api.hold(0.2)

    # ---- pinch around the measured pen
    P, Rm = api.measured_palm()
    p, an = pen()
    if an @ a < 0:
        an = -an
    a_pf = Rm.T @ an
    qA = Rm.T @ (p + sA * an - P)
    l_pf = sgn * a_pf                                  # pinch -> long part (palm frame)
    n_z = _unit(EZ - (EZ @ a_pf) * a_pf)
    ex = np.array([1.0, 0, 0])
    d_x = _unit(ex - (ex @ a_pf) * a_pf - (ex @ n_z) * n_z)
    nS = _unit(-np.cos(phi) * d_x + np.sin(phi) * n_z)  # pen axis -> press pad
    qS = qA - rs * l_pf
    diag.append(f"qA={np.round(qA, 4).tolist()} a_pf={np.round(a_pf, 3).tolist()}")
    yield from api.move_tips(tips={"thumb": qA - 0.024 * n_z, FN[pf]: qA + 0.024 * n_z,
                                   FN[sf]: qS + 0.03 * nS}, seconds=0.4)
    r = yield from api.move_tips(tips={"thumb": qA - hw * n_z, FN[pf]: qA + hw * n_z}, seconds=0.8, straight=True)
    yield from api.hold(0.3)
    diag.append(f"pinch ik={r.get('ik_residual_mm')}")
    snap("pinch")
    r = yield from api.move_tips(tips={FN[sf]: qS + (RPEN + PAD - 0.002) * nS}, seconds=0.6, straight=True)
    yield from api.hold(0.3)
    diag.append(f"press ik={r.get('ik_residual_mm')}")
    snap("press")

    # ---- lift level
    Pc, _ = api.palm_pose()
    yield from api.carry(xyz=Pc + np.array([0, 0, 0.11]), seconds=1.6)
    yield from api.hold(0.5)
    th, ap = snap("lift")
    yield from api.hold(0.5)
    th, ap = snap("hold")
    ref["ap"] = ap

    # ---- shift away from the rest and lower beside it
    H, _, _ = hinge()
    Pc, Rc = api.palm_pose()
    shift = -0.05 * Rc[:, 2]
    shift[2] = 0.0
    yield from api.carry(xyz=Pc + shift + np.array([0, 0, 0.795 - H[2]]), seconds=2.0)
    yield from api.hold(0.3)
    snap("low")

    # ---- back the press off: long end settles on the table
    tt = api.tip_targets()
    yield from api.move_tips(tips={FN[sf]: tt[sf] + 0.014 * nS}, seconds=1.0, straight=True)
    yield from api.hold(0.5)
    th, _ = snap("rel")
    for _ in range(3):
        if th > 8.0:
            break
        Pc, _ = api.palm_pose()
        yield from api.carry(xyz=Pc - np.array([0, 0, 0.006]), seconds=0.3)
        yield from api.hold(0.3)
        th, _ = snap("rel+")
    yield from api.move_joints(q={FN[sf]: api.OPEN[4 * sf:4 * sf + 4]}, seconds=0.6)
    yield from api.hold(0.3)
    snap("retract")

    # ---- arc the hinge over the resting tip
    H, p, an = hinge()
    u = an if float((p - H) @ an) > 0 else -an
    h = 0.0662
    if -u[2] > 0.12:
        h = float(np.clip((p[2] - TABLE - RPEN) / (-u[2]), 0.056, 0.076))
    C = p + u * h
    v = H - C
    rho = float(np.linalg.norm(v))
    hor = v.copy()
    hor[2] = 0.0
    w = _unit(hor)
    b0 = float(np.arctan2(v[2], np.linalg.norm(hor)))
    diag.append(f"arc h={h * 1000:.1f} rho={rho * 1000:.1f} b0={np.degrees(b0):.1f} C={np.round(C, 4).tolist()}")
    P0, _ = api.palm_pose()
    N = 24
    for k in range(1, N + 1):
        b = b0 + (np.pi / 2 - b0) * k / N
        tgt = C + rho * (np.cos(b) * w + np.sin(b) * EZ)
        yield from api.carry(xyz=P0 + (tgt - H), seconds=0.12)
    yield from api.hold(0.3)
    snap("arc")

    # ---- lift only as far as the rule needs (pen centre > start + 50 mm; aim +68 mm).
    #      iter_2: every extra squeeze or long lift made the vertical pen tip out of the
    #      two-pad pinch, so no re-squeeze and a short, slow lift.
    p, _ = pen()
    dz = float(np.clip(p0[2] + 0.068 - p[2], 0.02, 0.05))
    Pc, _ = api.palm_pose()
    yield from api.carry(xyz=Pc + np.array([0, 0, dz]), seconds=1.5)
    yield from api.hold(0.3)
    snap("up")

    tl = []
    for _ in range(10):
        yield from api.hold(0.1)
        _, an = pen()
        tl.append(round(float(np.degrees(np.arccos(min(1.0, abs(an[2]))))), 1))
    diag.append(f"final tilt-from-vertical samples {tl}")
    snap("end")
