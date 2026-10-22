"""scissors, phase 3 (force level + skills from earlier agents).

Plan (built on scissors/joints iter 4, which got furthest, and scissors/contacts, whose lift was reliable):
 1. Index pad deep in hole A, ring pad deep in hole B, closing squeeze with the force servo
    (index pushes A's inner wall -y, ring pushes B's inner wall +y; the closed stop carries it).
 2. Lift high. In the air: thumb outside A's outer wall at mid-wall height, servoed -y.
    The thumb and ring then squeeze the whole scissors through the closed stop, so the index can
    leave A's inner wall and cross hole A to A's outer wall (+y): thumb+index pinch part A alone.
 3. The ring alone drives B: centre it in hole B (no contact), then a force servo pushes B's outer
    wall (-y, opens) or inner wall (+y, closes), stopped by hinge-angle feedback. The servo measures
    |force| regardless of which wall, so the ring is always re-centred (zero force) before it is
    re-gripped in the other direction.
 4. Lower onto the table with the ring holding B closed, release, palm up.
Skills reused: contacts' reach placement search + pad measure-and-correct, pads_obj/status/clearance,
joints' little-finger parking and in-air sequence, size from start pivot height (6.8 mm * s).
"""
import numpy as np

OBJ = "scissors"
HINGE = "scissors_hinge"
TABLE = 0.75
FI = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
CURL = np.array([0.0, 1.6, 1.3, 1.1])
LITTLE_PARK = np.array([0.3, 1.4, 1.3, 1.1])   # contacts tuck pose; joints' [0.8,-0.6,..] blocked the ring (iter 1)
_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]
YM, YP = np.array([0, -1.0, 0]), np.array([0, 1.0, 0])
YMD, YPD = np.array([0, -0.96, -0.29]), np.array([0, 0.96, -0.29])   # lateral push with a little down: keeps pads deep

F_LIFT = 5.0      # N per pad, closing squeeze (lateral force comes from abduction only: <= ~6 N)
F_THUMB = 4.5     # N, thumb on A's outer face (A pinch)
F_RING = 3.0      # N, ring push on B (pivot friction needs ~0.6-1 N at the loop)
OPEN_DEG, CLOSED_DEG = 15.0, 1.0


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def clearance(p, R, a):
    pts = np.vstack([_A, _B @ Rz(a).T]) @ R.T + p
    return float(pts[:, 2].min() - abs(R[2, 2]) * 0.0055 - TABLE)


def run(api):
    D = []
    try:
        yield from main(api, D)
    except Exception as e:  # keep the diagnostics if something raises
        D.append(f"EXC {type(e).__name__}: {e}")
        for line in D[-50:]:
            api.note(line)
        raise
    for line in D[-50:]:
        api.note(line)


def main(api, D):
    def mm(v):
        return np.round(np.asarray(v, float) * 1000, 1).tolist()

    def hinge():
        return api.object_joint(HINGE)

    def pads_obj():
        p, R = api.object_in_palm(OBJ)
        return (api.tips() - p) @ R

    def L2P(q):
        p, R = api.object_in_palm(OBJ)
        return p + R @ np.asarray(q, float)

    def clr():
        p, R = api.object_pose(OBJ)
        return clearance(p, R, hinge())

    def st(tag):
        p, R = api.object_pose(OBJ)
        a = hinge()
        q = pads_obj()
        gs = {k[:2]: (v["force_N"], v["travel_mm"]) for k, v in api.grip_state().items()}
        D.append(f"{api.t:5.1f} {tag}: h={np.degrees(a):.1f} c={1000 * clearance(p, R, a):.0f} "
                 f"pz={1000 * (p[2] - TABLE):.1f} zR={R[2, 2]:.2f} f={np.round(api.contact_forces(OBJ), 1).tolist()} "
                 f"t={mm(q[0])} i={mm(q[1])} r={mm(q[3])} g={gs}")
        return p, R, a

    # ------------------------------------------------------------ size, pose, plan
    p_init, _ = api.object_pose(OBJ)
    s = float(np.clip((p_init[2] - TABLE) / 0.0068, 0.88, 1.12))
    yield from api.hold(0.4)
    p0, R0 = api.object_pose(OBJ)
    psi = float(np.arctan2(R0[1, 0], R0[0, 0]))
    h0 = p0[2] - TABLE
    yaw = 0.7 + float(np.clip(psi - 0.7, -0.3, 0.3))
    Rp = api.hand_down(yaw)
    M = Rp.T @ R0                      # flat scissors frame -> palm frame
    XP = -0.065 * s                    # pad x in the holes
    zlo, zhi = -h0 + 0.009, -h0 + 0.026
    T = {
        "i_low": ("index", [XP, 0.024 * s, zlo]), "i_high": ("index", [XP, 0.024 * s, zhi]),
        "i_in": ("index", [XP, 0.016 * s, zlo]), "i_mid": ("index", [XP, 0.024 * s, 0.0]),
        "i_out": ("index", [XP, 0.033 * s, 0.0]),
        "r_low": ("ring", [XP, -0.024 * s, zlo]), "r_high": ("ring", [XP, -0.024 * s, zhi]),
        "r_in": ("ring", [XP, -0.016 * s, zlo]), "r_mid": ("ring", [XP, -0.024 * s, 0.0]),
        "r_open": ("ring", list(Rz(0.3) @ np.array([XP, -0.033 * s, 0.0]))),
        "t_park": ("thumb", [-0.065 * s, 0.0425 * s + 0.022, 0.028]),
        "t_pre": ("thumb", [-0.065 * s, 0.0425 * s + 0.022, 0.0]),
        "t_press": ("thumb", [-0.065 * s, 0.0425 * s + 0.005, 0.0]),
    }
    W = {"t_press": 0.5, "r_open": 0.7, "i_out": 0.8}
    best = None
    for cx in (0.08, 0.09, 0.10, 0.11, 0.12):
        for cy in (-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02):
            for cz in (0.13, 0.15, 0.17, 0.19, 0.21):
                c = np.array([cx, cy, cz])
                worst, tot = 0.0, 0.0
                for k, (f, q) in T.items():
                    e = W.get(k, 1.0) * api.reach(f, c + M @ np.asarray(q))
                    worst, tot = max(worst, e), tot + e
                    if best is not None and worst > best[0]:
                        break
                else:
                    score = worst + 0.3 * tot / len(T)
                    if best is None or score < best[0]:
                        best = (score, c)
    c = best[1]
    res = {k: round(1000 * float(api.reach(f, c + M @ np.asarray(q))), 1) for k, (f, q) in T.items()}
    D.append(f"abd limits {np.round(api.joint_limits[[0, 4, 8, 12, 16]], 2).tolist()}")
    D.append(f"s={s:.3f} h0={1000 * h0:.2f} psi={psi:.2f} yaw={yaw:.2f} c={mm(c)} res={res}")
    X = p0 - Rp @ c                    # grasp palm position

    def P(k, corr=None):
        f, q = T[k]
        v = c + M @ np.asarray(q)
        return v if corr is None else v + corr

    # ------------------------------------------------------------ preshape, approach, place
    qt = api.q_target()
    qt[8:12], qt[16:20] = CURL, LITTLE_PARK
    yield from api.move_joints(q=qt, seconds=0.5)
    up = np.array([0, 0, 0.05])
    yield from api.move_tips(X + up, Rp, tips={"index": P("i_high"), "ring": P("r_high"),
                                               "thumb": P("t_park")}, seconds=2.0)
    yield from api.carry(X, Rp, seconds=1.0)
    yield from api.hold(0.2)
    Xm, _ = api.measured_palm()
    D.append(f"palm err {mm(Xm - X)}")
    corr = {"index": np.zeros(3), "ring": np.zeros(3)}
    for k in range(2):
        q = pads_obj()
        for f, key in (("index", "i_high"), ("ring", "r_high")):
            e = q[FI[f]] - np.asarray(T[key][1])
            corr[f] -= M @ np.array([e[0], e[1], 0.0])
        D.append(f"place{k} err i={mm(q[1] - T['i_high'][1])} r={mm(q[3] - T['r_high'][1])}")
        yield from api.move_tips(tips={"index": P("i_high", corr["index"]), "ring": P("r_high", corr["ring"])},
                                 seconds=0.4)
        yield from api.hold(0.1)

    # into the holes
    for attempt in range(2):
        yield from api.move_tips(tips={"index": P("i_low", corr["index"]), "ring": P("r_low", corr["ring"])},
                                 seconds=0.8, straight=True)
        yield from api.hold(0.25)
        st(f"in{attempt}")
        q = pads_obj()
        bad = [f for f in ("index", "ring") if q[FI[f]][2] > 0.0115 * s]
        if not bad:
            break
        D.append(f"pads on a wall: {bad}")
        yield from api.move_tips(tips={"index": P("i_high", corr["index"]), "ring": P("r_high", corr["ring"])},
                                 seconds=0.5, straight=True)
        yield from api.hold(0.1)
        q = pads_obj()
        for f, key in (("index", "i_high"), ("ring", "r_high")):
            e = q[FI[f]] - np.asarray(T[key][1])
            corr[f] -= M @ np.array([e[0], e[1], 0.0])

    # ------------------------------------------------------------ closing squeeze (position, like joints iter 4) and lift
    lifted = False
    for attempt, yin in enumerate((0.016, 0.013)):
        sq = {"index": M @ np.array([XP, yin * s, zlo]) + c + corr["index"],
              "ring": M @ np.array([XP, -yin * s, zlo]) + c + corr["ring"]}
        r = yield from api.move_tips(tips=sq, seconds=0.6, straight=True)
        yield from api.hold(0.3)
        st(f"sq{attempt} ik={r.get('ik_residual_mm')}")
        Xc, _ = api.palm_pose()
        yield from api.carry(Xc + np.array([0, 0, 0.015]), Rp, seconds=0.6)
        yield from api.hold(0.2)
        p, R, a = st(f"ltest{attempt}")
        if p[2] - p0[2] > 0.006:
            lifted = True
            break
        yield from api.carry(Xc, Rp, seconds=0.5)
    if lifted:   # contacts are clean lateral ones now: hand the squeeze to the force servo for the big lift
        r = yield from api.grip(OBJ, {"index": F_LIFT, "ring": F_LIFT}, directions={"index": YMD, "ring": YPD},
                                seconds=0.6, max_travel=0.02)
        st("servo")
    # only a low lift for the hand-over: the drooping blade tips may rest on the table, which limits the pitch
    # (iter 4: at +20 cm the pitched scissors were knocked out by the thumb moving in)
    Xc, _ = api.palm_pose()
    yield from api.carry(Xc + np.array([0, 0, 0.035]), Rp, seconds=1.0)
    yield from api.hold(0.3)
    p, R, a = st("low")
    low_ok = lifted and p[2] - p0[2] > 0.015

    # ------------------------------------------------------------ thumb onto A's outer face
    # Targets stay force-consistent (ring = index + thumb along y). Before the thumb touches, index 2.5 / ring 5
    # settle at ~3.7 N each and drift A toward the thumb, so contact happens even if the thumb falls short.
    thumb_ok = False
    if low_ok:
        yield from api.move_tips(tips={"thumb": L2P(T["t_park"][1])}, seconds=0.4)
        r = yield from api.move_tips(tips={"thumb": L2P(T["t_pre"][1])}, seconds=0.6, straight=True)
        yield from api.hold(0.1)
        if abs(pads_obj()[0][2] - T["t_pre"][1][2]) > 0.008:
            D.append(f"thumb pre off {mm(pads_obj()[0])}; retry")
            yield from api.move_tips(tips={"thumb": L2P(T["t_pre"][1])}, seconds=0.5)
            yield from api.hold(0.1)
        D.append(f"thumb pre ik={r.get('ik_residual_mm')} meas={mm(pads_obj()[0])} want={mm(T['t_pre'][1])}")
        yield from api.grip(OBJ, {"thumb": 2.5, "index": 2.5, "ring": 5.0},
                            directions={"thumb": YM, "index": YMD, "ring": YPD}, seconds=0.1, max_travel=0.03)
        t0 = api.t
        while api.t - t0 < 1.6:
            yield from api.hold(0.1)
            if api.contact_forces(OBJ)[0] > 1.5 and api.t - t0 > 0.6:
                thumb_ok = True
                break
        st("thumb")
        if not thumb_ok:
            yield from api.grip(OBJ, {"thumb": 0.5, "index": F_LIFT, "ring": F_LIFT},
                                directions={"thumb": YM, "index": YMD, "ring": YPD}, seconds=0.5)
            D.append("thumb no contact: back to the squeeze, skip cycles")

    # ------------------------------------------------------------ index across hole A to the outer wall
    cycles_ok = False
    if thumb_ok:
        yield from api.ungrip(["index"])
        yield from api.grip(OBJ, {"thumb": 4.0, "ring": 4.0}, directions={"thumb": YM, "ring": YPD},
                            seconds=0.4, max_travel=0.03)
        q = pads_obj()[1]
        yield from api.move_tips(tips={"index": L2P([q[0], 0.0375 * s - 0.0115, 0.0])}, seconds=0.5, straight=True)
        st("ix_moved")
        yield from api.grip(OBJ, {"thumb": 6.0, "index": 3.0, "ring": 3.0},
                            directions={"thumb": YM, "index": YP, "ring": YPD}, seconds=1.2, max_travel=0.03)
        st("A_pinch3")
        fz = api.contact_forces(OBJ)
        if fz[1] > 1.5 and fz[0] > 1.5:
            yield from api.ungrip(["ring"])
            yield from api.grip(OBJ, {"thumb": F_THUMB, "index": F_THUMB}, directions={"thumb": YM, "index": YP},
                                seconds=0.5, max_travel=0.03)
            Xc, _ = api.palm_pose()
            yield from api.carry(Xc + np.array([0, 0, 0.19]), Rp, seconds=3.0)
            yield from api.hold(0.3)
            st("lifted")
            yield from api.ungrip(["index"])
            cycles_ok = clr() > 0.03
        else:
            yield from api.grip(OBJ, {"thumb": 4.0, "ring": 4.0}, directions={"thumb": YM, "ring": YPD},
                                seconds=0.4, max_travel=0.03)
            D.append("A pinch not confirmed; keep thumb+ring squeeze, skip cycles")

    # ------------------------------------------------------------ open / close cycles with the ring
    def ring_center():
        a = hinge()
        qb = Rz(-a) @ pads_obj()[3]
        tgt = Rz(a) @ np.array([qb[0], -0.024 * s, 0.0])
        yield from api.move_tips(tips={"ring": L2P(tgt)}, seconds=0.35, straight=True)

    def drive(direction, until, Fr, tmax):
        yield from ring_center()
        yield from api.grip(OBJ, {"ring": Fr}, directions={"ring": direction}, seconds=0.02, max_travel=0.035)
        t0, a0, boosted, peak = api.t, hinge(), False, 0.0
        while api.t - t0 < tmax:
            yield from api.hold(0.06)
            a = np.degrees(hinge())
            peak = max(peak, a)
            if until(a):
                return True, peak
            if not boosted and api.t - t0 > 1.3 and abs(a - np.degrees(a0)) < 4:
                yield from api.grip(OBJ, {"ring": Fr * 1.7}, directions={"ring": direction}, seconds=0.02,
                                    max_travel=0.035)
                boosted = True
        return False, peak

    done, tries = 0, 0
    ring_gripped = False
    while cycles_ok and done < 2 and tries < 4 and api.t < 44:
        tries += 1
        if clr() < 0.025:
            D.append("too low / dropped; stop cycles")
            break
        if ring_gripped:
            yield from api.ungrip(["ring"])
        ok_o, pk = yield from drive(YM, lambda a: a >= OPEN_DEG, F_RING, 3.0)
        yield from api.ungrip(["ring"])
        p, R, a = st(f"open{tries}")
        air_o = clr() > 0.022
        ok_c, _ = yield from drive(YP, lambda a: a <= CLOSED_DEG, F_RING, 3.5)
        ring_gripped = True
        yield from api.hold(0.1)
        p, R, a = st(f"close{tries}")
        if pk > 10.5 and np.degrees(a) < 2.5 and air_o and clr() > 0.022:
            done += 1
        D.append(f"cycle {tries}: peak={pk:.1f} ok_o={ok_o} ok_c={ok_c} counted={done}")

    # ------------------------------------------------------------ keep closed, set down, release
    if cycles_ok:
        if ring_gripped:
            yield from api.grip(OBJ, {"ring": 2.0}, directions={"ring": YP}, seconds=0.02, max_travel=0.035)
        elif np.degrees(hinge()) > 1.5:
            yield from drive(YP, lambda a: a <= CLOSED_DEG, F_RING, 3.0)
    st("pre_down")
    Xs = X + np.array([0, 0, 0.005])
    for k in range(30):
        Xc, _ = api.palm_pose()
        if Xc[2] <= Xs[2] + 1e-4 or api.t > 55:
            break
        p, R = api.object_pose(OBJ)
        cl = clearance(p, R, hinge())
        tw = api.tips_world()
        if cl < 0.004 and abs(R[2, 2]) > 0.95:
            break
        if tw[[0, 1, 3], 2].min() < TABLE + 0.013:
            break
        dz = min(0.03, Xc[2] - Xs[2], max(0.006, cl - 0.002))
        yield from api.carry(Xc - np.array([0, 0, dz]), Rp, seconds=max(0.15, dz * 12))
    yield from api.hold(0.3)
    st("down")
    yield from api.ungrip()
    q = pads_obj()
    a = hinge()
    qb = Rz(-a) @ q[3]
    tg = {"thumb": L2P(q[0] + np.array([0, 0.015, 0.006])),
          "index": L2P([q[1][0], 0.024 * s, q[1][2] + 0.003]),
          "ring": L2P(Rz(a) @ np.array([qb[0], -0.024 * s, qb[2] + 0.003]))}
    yield from api.move_tips(tips=tg, seconds=0.5, straight=True)
    Xc, _ = api.palm_pose()
    yield from api.carry(Xc + np.array([0, 0, 0.03]), Rp, seconds=0.6)
    Xc, _ = api.palm_pose()
    yield from api.carry(Xc + np.array([0, 0, 0.06]), Rp, seconds=0.6)
    yield from api.hold(0.3)
    st("released")
