# pen_gravity_pivot at level force (phase 3, skills).  Phases 1-4 (pinch, press,
# level lift, table-assisted arc pivot) are the layered agent's iter_3 policy
# (bench_runs/agents/pen_gravity_pivot/layered, 2/3 dev, 3/6 held-out), reused
# nearly verbatim.  Its failure was the two-pad hinge bleeding grip after the
# pivot; here the standing pen is re-grasped in a 3-contact V-cage (thumb vs
# index+middle, from pen_table_slide/contacts) with a thumb force servo, then lifted.
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
    # pen radius from its resting height (rest is fixed in z; nominal centre z 0.7843 at r 3.75 mm)
    rp = float(np.clip(RPEN + (p0[2] - 0.7843), 0.0032, 0.0044))
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
        # iter_3: thumb+MIDDLE pinch, RING presses the stub, index parked: after the arc the pen sits
        # on the +y side of the pinch line, where the free index can close the cage.
        pf, sf, sgn, ys = 2, 3, +1, (0.010, 0.004, 0.016, 0.022)
    else:            # pinch thumb+middle at the +a end, long part toward palm -y, index presses
        pf, sf, sgn, ys = 2, 1, -1, (0.004, 0.010, 0.016)
    sA = -sgn * 0.030
    W = p0 + sA * a
    diag.append(f"rp={rp*1000:.2f}mm cfg={cfg} s_rest={s_rest} psi={psi:.3f} p0={np.round(p0, 4).tolist()}")

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
        # pad offsets from the pen axis, palm frame [mm] (thumb, index, middle, ring)
        offs = []
        for j in range(4):
            w = tw[j] - p
            w = w - (w @ an) * an
            offs.append(np.round(1000 * (Rm.T @ w)).astype(int).tolist())
        gs = api.grip_state()
        gtxt = " G=" + str({kk: (vv["force_N"], vv["travel_mm"]) for kk, vv in gs.items()}) if gs else ""
        diag.append(f"{tag} t={api.t:.2f} lift={1000 * (p[2] - p0[2]):.0f} tiltH={th:.1f}{piv} "
                    f"pinch_s={1000 * sh:.0f} pinch_d={1000 * dh:.0f} "
                    f"ap={np.round(ap, 2).tolist()} F={np.round(f, 2).tolist()} offs={offs}{gtxt}")
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
    t_open = api.tips()
    diag.append(f"OPEN tips palm mm {np.round(1000 * t_open).astype(int).tolist()}")
    if pf != 1 and sf != 1 and t_open[1][0] > 0.045:      # park the index well above the pen plane
        park = np.array([0.035, t_open[1][1], t_open[1][2]])
        diag.append(f"park index reach {1000 * api.reach('index', park):.1f}mm")
        yield from api.move_tips(tips={"index": park}, seconds=0.5)

    # ---- choose pinch centre / press direction by fingertip reachability
    hw = rp + PAD - 0.0022
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
    r = yield from api.move_tips(tips={FN[sf]: qS + (rp + PAD - 0.002) * nS}, seconds=0.6, straight=True)
    yield from api.hold(0.3)
    diag.append(f"press ik={r.get('ik_residual_mm')}")
    snap("press")

    # ---- lift level
    Pc, _ = api.palm_pose()
    yield from api.carry(xyz=Pc + np.array([0, 0, 0.11]), seconds=1.6)
    yield from api.hold(0.5)
    th, ap = snap("lift")
    # ---- skill level(): closed-loop levelling with the press finger (iter_1: the press left the
    #      long part 17 deg up in all 3 cases, so the level hold never counted). The pinch is a free
    #      hinge, so moving the press pad along its normal by rs*tan(elev) sets the elevation.
    for i in range(4):
        _, an = pen()
        if an @ a < 0:
            an = -an
        elev = float(np.arcsin(np.clip(sgn * an[2], -1, 1)))      # long part above horizontal (+)
        if abs(np.degrees(elev)) < 4.0:
            break
        tt = api.tip_targets()
        step = float(np.clip(0.8 * rs * np.tan(elev), -0.004, 0.006))
        yield from api.move_tips(tips={FN[sf]: tt[sf] + step * nS}, seconds=0.4, straight=True)
        yield from api.hold(0.4)
        snap(f"level{i} e={np.degrees(elev):.1f} step={1000 * step:.1f}")
    yield from api.hold(0.6)
    th, ap = snap("hold")
    ref["ap"] = ap
    # ---- force layer, phase A: keep the thumb/pinch-finger hinge at 0.8 N from here through lowering,
    #      release and arc (iter_2 sev 0.5 and iter_4 sev 0.5 dropped the pen in the arc after the pinch
    #      bled to 0.5-0.6 N; in iter_4 it bled while lowering and the servo hit its 3 mm cap).
    #      Default closing direction = toward the other pad (palm frame, along the hinge line).
    g = yield from api.grip("pen", {"thumb": 0.8, FN[pf]: 0.8}, seconds=0.3, max_travel=0.005)
    diag.append(f"hinge grip {g}")

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
    # iter_6: skill arc_cl(): closed-loop arc. Each step re-measures the pen's lower end T (cap centre,
    # h along the axis from the centre) and moves the commanded hinge on the circle about the CURRENT T
    # with the CURRENT radius, so a sliding tip does not make the hand drag the pen sideways through the
    # two-pad hinge (open-loop arcs ended with a ~9 deg lean and dropped the pen in 3 of 12 dev cases).
    Pc, _ = api.palm_pose()
    off = H - Pc                                   # commanded palm -> hinge (fixed)
    db = (np.pi / 2 - b0) / 24
    trace = []
    for kk in range(40):
        _, p, an = hinge()
        Pc, _ = api.palm_pose()
        Hc = Pc + off
        uu = an if float((p - Hc) @ an) > 0 else -an
        T = p + uu * h
        vv = Hc - T
        hor = vv.copy()
        hor[2] = 0.0
        if np.linalg.norm(hor) < 1e-4:
            break
        rr = float(np.linalg.norm(vv))
        ww = _unit(hor)
        b = float(np.arctan2(vv[2], np.linalg.norm(hor)))
        if kk % 4 == 0:
            f = api.contact_forces("pen")
            trace.append((round(np.degrees(b)), round(1000 * rr), round(1000 * T[2] - 750), round(float(f[0]), 2), round(float(f[pf]), 2)))
        if b >= np.radians(88.0):
            break
        bn = min(b + db, np.radians(89.0))
        tgt = T + rr * (np.cos(bn) * ww + np.sin(bn) * EZ)
        yield from api.carry(xyz=Pc + (tgt - Hc), seconds=0.12)
    diag.append(f"arc steps={kk} trace(b,rho,Tz-750mm,Ft,Fp)={trace}")
    yield from api.hold(0.3)
    snap("arc")



    # ================= new in phase 3 (skills agent) =================
    # ---- skill cage(): close a 3-contact cage on the standing pen (tip still on the table).
    #      iter_1/2: after the arc the pen sits ~6 mm toward palm +y of the thumb/pinch-finger line
    #      (open gap on the +y side). With the thumb+middle hinge, the parked index comes down on
    #      that +y/+z side, a bit lower on the pen, so thumb / middle / index span the circle.
    #      Angles phi about the pen axis: pos = c + rho*(cos(phi) u + sin(phi) v), u = palm z, v ~ palm +y.
    k = rp + PAD          # (hinge servo keeps running while the index comes in)

    def cframe():
        P, Rm = api.measured_palm()
        tw = api.tips_world()
        p, an = pen()
        if an @ (Rm[:, 0]) < 0:
            an = -an                                  # a_p points down the pen (palm +x)
        mid = 0.5 * (tw[0] + tw[pf])
        ax = p + an * float((mid - p) @ an)
        c = Rm.T @ (ax - P)
        a_p = Rm.T @ an
        u = _unit(EZ - (EZ @ a_p) * a_p)
        v = _unit(np.cross(a_p, u))
        if v[1] < 0:
            v = -v
        tp = np.array([Rm.T @ (tw[j] - P) for j in range(5)])
        return c, a_p, u, v, tp

    def at(c, a_p, u, v, phi, rho, dx=0.0):
        return c + dx * a_p + rho * (np.cos(phi) * u + np.sin(phi) * v)

    def ang(c, u, v, x):
        w = x - c
        return float(np.arctan2(w @ v, w @ u))

    c, a_p, u, v, tp = cframe()
    ph_t, ph_m = ang(c, u, v, tp[0]), ang(c, u, v, tp[pf])
    best = None
    for dxi in (0.014, 0.010, 0.018, 0.022):
        for dphi in (85, 75, 95, 65, 105):
            phi_i = np.radians(np.clip(np.degrees(ph_m) + dphi, 30, 85))
            e1 = api.reach("index", at(c, a_p, u, v, phi_i, k - 0.0015, dxi))
            e2 = api.reach("index", at(c, a_p, u, v, phi_i, k + 0.004, dxi))
            sc = e1 + e2 + 0.0003 * abs(dphi - 85) / 10 + 0.0003 * abs(dxi - 0.014) / 0.004
            if best is None or sc < best[0]:
                best = (sc, dxi, phi_i, e1, e2)
    _, dxi, phi_i, e1, e2 = best
    diag.append(f"cage c={np.round(1000 * c, 1).tolist()} phi thumb={np.degrees(ph_t):.0f} {FN[pf]}={np.degrees(ph_m):.0f} "
                f"-> index phi={np.degrees(phi_i):.0f} dx={1000 * dxi:.0f} reach {1000 * e1:.1f}/{1000 * e2:.1f}")
    # index: waypoint out on its side and lower, then pre-position, then light contact
    wp = at(c, a_p, u, v, phi_i, k + 0.018, dxi + 0.010)
    yield from api.move_tips(tips={"index": wp}, seconds=0.6)
    c, a_p, u, v, tp = cframe()
    yield from api.move_tips(tips={"index": at(c, a_p, u, v, phi_i, k + 0.004, dxi)}, seconds=0.4, straight=True)
    snap("idx_pre")
    c, a_p, u, v, tp = cframe()
    yield from api.move_tips(tips={"index": at(c, a_p, u, v, phi_i, k - 0.0015, dxi)}, seconds=0.4, straight=True)
    yield from api.hold(0.2)
    snap("idx_on")
    yield from api.ungrip([FN[pf]])        # pinch finger becomes a position anchor; thumb re-gripped below
    # iter_5: no position firm-up of all three (in iters 3-4 it spiked forces to ~3 N and tipped the
    # standing pen from ~2 to 20+ deg, leaving it at 11-15 deg); the thumb force servo alone closes the cage.
    c, a_p, u, v, tp = cframe()
    diag.append(f"cage phis t/{FN[pf]}/i={[round(np.degrees(ang(c, u, v, tp[j]))) for j in (0, pf, 1)]}")

    # ---- force layer, phase B: thumb servo presses the pen into the index/middle side of the cage.
    #      Direction (object frame) = thumb pad -> pen axis, refreshed from measurement each chunk.
    FT = {"v": 0.9}

    def regrip(seconds=0.02):
        p, Rp = api.object_pose("pen")
        an = Rp[:, 2]
        w = api.tips_world()[0] - p
        w = w - (w @ an) * an
        d = Rp.T @ _unit(-w)
        res = yield from api.grip("pen", {"thumb": FT["v"]}, directions={"thumb": d}, seconds=seconds,
                                  max_travel=0.007)
        return res

    res = yield from regrip(0.6)
    diag.append(f"grip {res}")
    snap("gripped")

    # ---- lift the caged pen in small steps (pen centre to ~ start + 82 mm), lag check
    p, _ = pen()
    dz = float(np.clip(p0[2] + 0.082 - p[2], 0.025, 0.065))
    nst = int(np.ceil(dz / 0.008))
    zp0 = p[2]
    for i in range(nst):
        Pc, _ = api.palm_pose()
        yield from api.carry(xyz=Pc + np.array([0, 0, dz / nst]), seconds=0.3)
        yield from regrip()
        p, _ = pen()
        lag = (i + 1) * dz / nst - (p[2] - zp0)
        if lag > 0.005 and FT["v"] < 1.8:
            FT["v"] += 0.3
            diag.append(f"lift lag {1000 * lag:.1f}mm -> thumb {FT['v']:.1f} N")
    yield from api.hold(0.2)
    snap("lifted")

    tl = []
    for i in range(8):
        yield from api.hold(0.18)
        yield from regrip()
        _, an = pen()
        tl.append(round(float(np.degrees(np.arccos(min(1.0, abs(an[2]))))), 1))
    diag.append(f"final tilt-from-vertical samples {tl}")
    snap("end")
