"""pen_pickup_pivot, force level, phase 3 ("skills").

Idea: every earlier agent lost the pen at lift-off because a thumb-index pinch on a pen lying
on the table touches it above its equator (thick round tips), a wedge that squirts the pen
out downward. Fix: a stub ledge.
  1. Palm down, pen along palm y. Thumb (near side) + index (far side) pinch the pen 33 mm from
     its centre toward its -y end A (force servo, palm-z directions). The middle hovers on the
     far side of the stub (the 29-45 mm of pen beyond the pinch, toward A), at table level.
  2. Lift the pinch ~10 mm (pen pivots on its far end B, still on the table) while the middle
     stays at table level; the stub rises above the middle's tip.
  3. Hook the middle under the stub (distal link pointing down-and-back, pad facing up-back)
     and force-servo it up into the stub. Pinch + stub ledge + B on the table is a stable
     beam: the wedge can no longer push the pinched end down.
  4. Pivot: rotate the whole hand rigidly about the line through B's cap centre (along palm z)
     so the pen turns about B with no motion relative to the fingers, until it stands on B.
  5. Firm the grip, lift 45 mm, level the pen by rotating about the pinch, hold.
Helpers reused/adapted: pose/axis + obj_dir (pen_gravity_pivot/force), _rotvec/_rotm and the
measured-palm re-centring grasp (pen_pickup_pivot/force), buffered notes + print diagnostics.
"""
import numpy as np

TABLE = 0.75
C = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (thumb -z, index +z)
PAD = 0.008                           # pad centre to finger surface
H_NOM = 0.0662
S_PINCH = 0.033                       # pinch this far from the pen centre toward end A (palm -y)
DM = 0.022                            # middle pad this much further toward A
PADZ = 0.0105                         # pinch pad centres this high above the table
F_PINCH = 0.8
F_HOLD = 1.1
F_LEDGE = 0.3
LIFT0 = 0.018
GAMMA = np.radians(35)                # middle pad normal: 0 = facing straight up, 90 = facing the thumb
STEP = np.radians(2.5)


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def rotm(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rotvec(b, t):
    """Rotation vector turning unit b onto unit t."""
    ax = np.cross(b, t)
    s, c = np.linalg.norm(ax), float(b @ t)
    if s < 1e-9:
        return np.zeros(3)
    return ax / s * np.arctan2(s, c)


def r3(v, k=1000):
    return np.round(np.asarray(v, float) * k, 1).tolist()


def run(api):
    log = []

    def say(s):
        s = f"{api.t:6.2f} {s}"
        log.append(s)
        print("POLICY", s, flush=True)

    try:
        yield from _run(api, say)
    except Exception as ex:
        say(f"EXC {type(ex).__name__}: {ex}")
        raise
    finally:
        for s in log[-45:]:
            api.note(s)


def _run(api, say):
    st = {"r": 0.00375, "h": H_NOM, "u": None, "stand": 90.0}
    Z = np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------ observation helpers
    def pen():
        """centre p, axis a_B pointing to end B (palm +y side at grasp), elevation of A [deg], low [m]."""
        p, R = api.object_pose("pen")
        a = R[:, 2]
        ref = st.get("a_prev", st["u"])
        if ref is not None and a @ ref < 0:
            a = -a
        st["a_prev"] = a
        elev = float(np.degrees(np.arcsin(np.clip(-a[2], -1, 1))))     # A above B
        low = float(p[2] - st["h"] * abs(a[2]) - st["r"] - TABLE)
        tilt = 90.0 - abs(elev)
        if low < 0.003:
            st["stand"] = min(st["stand"], tilt)
        return p, a, elev, low

    def forces():
        return np.round(api.contact_forces("pen"), 2).tolist()

    def pen_palm():
        pp, Rp = api.object_in_palm("pen")
        a = Rp[:, 2]
        if a[1] < 0:
            a = -a          # toward palm +y (end B)
        return pp, a, Rp

    def axis_at_y(y):
        pp, a, _ = pen_palm()
        return pp + a * (y - pp[1]) / a[1]

    def obj_dir(d_palm):
        _, _, Rp = pen_palm()
        return Rp.T @ unit(d_palm)

    def x_table():
        Pm, Rm = api.measured_palm()
        return float((Pm[2] - TABLE) / max(-Rm[2, 0], 0.5))

    def geom(tag):
        tp = api.tips()
        pt = axis_at_y(tp[0][1])
        d = [r3(tp[i] - axis_at_y(tp[i][1])) for i in range(3)]
        tw = api.tips_world()
        say(f"{tag}: pad-axis (palm x down,y,z) th={d[0]} ix={d[1]} md={d[2]} padz={r3(tw[:3, 2] - TABLE)} F={forces()} gs={api.grip_state()}")
        return pt

    def safe(X, R=None, seconds=0.5):
        try:
            yield from api.carry(xyz=X, R=R, seconds=seconds)
            return True
        except ValueError as e:
            say(f"IK fail {str(e)[:60]}")
            return False

    # ------------------------------------------------------------ phases
    def plan():
        p0, Rpen0 = api.object_pose("pen")
        st["r"] = float(np.clip(p0[2] - TABLE, 0.003, 0.0045))
        a = Rpen0[:, 2]
        ah = unit([a[0], a[1], 0.0])
        best = None
        for s in (1.0, -1.0):
            u = s * ah
            psi = np.arctan2(u[1], u[0]) - np.pi / 2
            d = (psi - 0.7 + np.pi) % (2 * np.pi) - np.pi
            if best is None or abs(d) < abs(best[0]):
                best = (d, u)
        d, u = best
        st["u"] = u
        st.pop("a_prev", None)
        R = api.hand_down(0.7 + d)          # palm y = u (toward end B), palm x down
        Q = p0 - S_PINCH * u
        Q[2] = TABLE + PADZ
        X = api.palm_for(Q, C, R)
        say(f"plan p0={r3(p0)} r={st['r']*1000:.2f} yaw={0.7 + d:.3f} X={r3(X)}")
        return X, R

    def approach(X, R):
        wo = st["r"] + PAD + 0.012
        q = api.OPEN.copy()
        yield from api.move_joints(q={"ring": q[12:16], "little": q[16:20]}, seconds=0.3)
        tips = {"thumb": C + [0, 0, -wo], "index": C + [0, 0, wo],
                "middle": C + np.array([-0.015, -DM, st["r"] + PAD + 0.006])}
        res = yield from api.move_tips(xyz=X + [0, 0, 0.05], R=R, tips=tips, seconds=2.0)
        say(f"preshape ik={res.get('ik_residual_mm')} tips={r3(api.tips()[:3])}")
        ok = yield from safe(X, R, 1.2)
        yield from api.hold(0.2)
        return ok

    def trace(gen, tag):
        """Run a motion generator, printing pen-vs-pinch state every step (log_*.txt only)."""
        k = 0
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            k += 1
            if k % 2 == 0:
                try:
                    tp = api.tips()
                    pa = axis_at_y(C[1])
                    p, a, elev, low = pen()
                    print(f"TR {api.t:6.2f} {tag} elev={elev:5.1f} axis@pinch(palm)={r3(pa)} th={r3(tp[0] - pa)} "
                          f"ix={r3(tp[1] - pa)} md={r3(tp[2] - axis_at_y(tp[2][1]))} F={forces()}", flush=True)
                except Exception:
                    pass
            yield

    NH = np.array([np.cos(GAMMA), 0, np.sin(GAMMA)])     # stub axis -> middle pad
    DIRM = np.array([np.sin(GAMMA), 0, -np.cos(GAMMA)])  # middle distal link: down and back

    def close():
        pz = api.object_pose("pen")[0][2]
        say(f"settled pen z={pz*1000:.2f} (plan r={st['r']*1000:.2f})")
        st["r"] = float(np.clip(pz - TABLE, 0.003, 0.0045))
        r = st["r"]
        xt = x_table()
        pt = axis_at_y(C[1])
        w = r + PAD + 0.001
        tgt = {"thumb": np.array([xt - PADZ, C[1], pt[2] - w]), "index": np.array([xt - PADZ, C[1], pt[2] + w])}
        ptm = axis_at_y(C[1] - DM)
        # middle pre-curled (pad facing up-back), hovering on the far side of the stub, 13 mm above the table
        tgt["middle"] = np.array([xt - 0.013, C[1] - DM, ptm[2] + 0.024])
        res = yield from api.move_tips(tips=tgt, seconds=0.5, directions={"middle": DIRM})
        ik = res.get("ik_residual_mm") or {}
        say(f"close ik={ik} xt={xt*1000:.1f} md_got={r3(api.tips()[2] - ptm)}")
        if max(ik.get("thumb", 0), ik.get("index", 0)) > 5:
            return False, tgt["middle"]
        g = yield from api.grip("pen", {"thumb": F_PINCH, "index": F_PINCH},
                                directions={"thumb": obj_dir([0, 0, 1]), "index": obj_dir([0, 0, -1])}, seconds=1.2)
        say(f"grip {g}")
        geom("pinch")
        F = forces()
        return min(F[0], F[1]) > 0.3, tgt["middle"]

    def lift_and_hook(X, R, mid_t):
        """Lift the pinch fast while the middle stays at table level, then push the middle under the stub."""
        lifted = LIFT0
        X = X + np.array([0, 0, LIFT0])
        try:
            yield from trace(api.move_tips(xyz=X, R=R, tips={"middle": mid_t + np.array([lifted, 0, 0])},
                                           directions={"middle": DIRM}, seconds=0.14), "lift")
        except ValueError as e:
            say(f"lift IK fail {e}")
            return X, False
        p, a, elev, low = pen()
        stub = axis_at_y(C[1] - DM)
        stub_h = x_table() - stub[0]
        say(f"lift +{lifted*1000:.0f}mm elev={elev:.1f} stub_h={stub_h*1000:.1f} F={forces()}")
        if elev < 2.0:
            say("pen did not rise with the pinch")
            return X, False
        if -a[2] > np.sin(np.radians(3)):
            st["h"] = float(np.clip((p[2] - TABLE - st["r"]) / (-a[2]), 0.055, 0.078))
        hook = stub + (st["r"] + PAD - 0.0015) * NH          # 1.5 mm into the stub: it must touch
        below = np.array([stub[0] + st["r"] + PAD + 0.002, stub[1], hook[2]])   # pad top 2 mm under the stub
        res = yield from trace(api.move_tips(tips={"middle": below}, directions={"middle": DIRM}, seconds=0.12,
                                             straight=True), "under")
        say(f"under ik={res.get('ik_residual_mm')} stub_h={(x_table() - stub[0])*1000:.1f}")
        res = yield from trace(api.move_tips(tips={"middle": hook}, directions={"middle": DIRM}, seconds=0.06,
                                             straight=True), "hook")
        g = yield from trace(api.grip("pen", {"middle": F_LEDGE}, directions={"middle": obj_dir(-NH)}, seconds=0.06),
                             "ledge")
        for k in range(10):
            if forces()[2] > 0.12:
                break
            yield from trace(api.hold(0.02), "wait")
        say(f"hook ik={res.get('ik_residual_mm')} h_est={st['h']*1000:.1f} target={r3(hook)} got={r3(api.tips()[2])}")
        geom("hooked")
        return X, True

    def palm_err():
        Pm, Rm = api.measured_palm()
        Pc, Rc = api.palm_pose()
        return r3(Pm - Pc), round(float(np.degrees(np.linalg.norm(rotvec(Rc[:, 2], Rm[:, 2])))), 2)

    def middle_follow(pen_m=0.0015):
        """Middle hook target under the stub for the pen's current pose in the palm frame."""
        pp, ap, _ = pen_palm()
        pinch = axis_at_y(C[1])
        stub = pinch - DM * ap
        dperp = unit(np.cross(ap, [0, 0, 1.0]))                          # pen "down" in the palm x-y plane
        nh = np.cos(GAMMA) * dperp + np.sin(GAMMA) * np.array([0, 0, 1.0])
        dm = np.sin(GAMMA) * dperp - np.cos(GAMMA) * np.array([0, 0, 1.0])
        return stub + (st["r"] + PAD - pen_m) * nh, dm

    def middle_nh():
        pp, ap, _ = pen_palm()
        dperp = unit(np.cross(ap, [0, 0, 1.0]))
        return np.cos(GAMMA) * dperp + np.sin(GAMMA) * np.array([0, 0, 1.0])

    def pivot(X, R):
        """Hinge pivot: move the pinch on an arc about B's cap centre with the palm orientation fixed
        (pure translations, like the lift that worked); the pen turns in the thumb-index hinge while the
        middle ledge keeps following under the stub. The ledge is dropped when it can no longer reach."""
        f = R[:, 2].copy()
        p, a, elev0, low = pen()
        ledge = True
        soft = False
        k = 0
        while k < 80:
            k += 1
            p, a, elev, low = pen()
            if elev > 86.0:
                break
            if elev < 1.0 or (k > 3 and max(forces()[:2]) < 0.05):
                say(f"pivot lost at k={k}: elev={elev:.1f} F={forces()} palm_err={palm_err()}")
                return X, R, False
            if 12 < elev < 45:
                st["h"] = float(np.clip((p[2] - TABLE - st["r"]) / max(-a[2], 0.2), 0.055, 0.078))
            B = p + st["h"] * a
            exc = B[2] - (TABLE + st["r"])
            tw = api.tips_world()
            P = 0.5 * (tw[0] + tw[1])
            step = STEP if elev < 60 else STEP * 0.6
            Pn = B + rotm(-step * f) @ (P - B)
            Xn = X + (Pn - P)
            if exc > 0.006:
                Xn = Xn - np.array([0, 0, exc - 0.004])
            tips, dirs = {}, {}
            if ledge and elev > 72:
                say(f"drop ledge at elev {elev:.1f} F={forces()} gs={api.grip_state()}")
                yield from api.ungrip(["middle"])
                tt = api.tip_targets()
                yield from api.move_tips(tips={"middle": tt[2] + np.array([-0.012, 0, 0.010])}, seconds=0.15)
                ledge = False
            if ledge and elev > 40 and not soft:
                yield from api.grip("pen", {"middle": 0.15}, directions={"middle": obj_dir(-middle_nh())}, seconds=0.02)
                soft = True
            if ledge:
                hook, dm = middle_follow(0.0015 if elev < 30 else (0.0005 if elev < 40 else -0.001))
                tips, dirs = {"middle": hook}, {"middle": dm}
            try:
                mv = api.move_tips(xyz=Xn, R=R, tips=tips or None, directions=dirs or None, seconds=0.08)
                res = yield from (trace(mv, "piv") if k <= 10 else mv)
            except ValueError as e:
                say(f"pivot IK fail at elev {elev:.1f}: {str(e)[:50]}")
                return X, R, elev > 55
            X = Xn
            if ledge and ((res or {}).get("ik_residual_mm") or {}).get("middle", 0) > 3.0:
                say(f"ledge out of reach at elev {elev:.1f}: {res.get('ik_residual_mm')} -> drop middle")
                yield from api.ungrip(["middle"])
                tt = api.tip_targets()
                yield from api.move_tips(tips={"middle": tt[2] + np.array([-0.015, 0, 0.012])}, seconds=0.15)
                ledge = False
            if k % 4 == 0:
                say(f"piv{k} elev={elev:.1f} lowB={exc*1000:.1f} F={forces()} gs={api.grip_state()} palm_err={palm_err()}")
        yield from api.hold(0.15)
        p, a, elev, low = pen()
        say(f"pivot end elev={elev:.1f} low={low*1000:.1f} stand={st['stand']:.1f} F={forces()}")
        return X, R, elev > 55

    def level(X, R, about="pinch", rounds=2, lim=np.radians(12)):
        for j in range(rounds):
            p, a, elev, low = pen()
            up = -a                          # pen direction toward A (should point up)
            rv = rotvec(unit(up), Z)
            ang = np.linalg.norm(rv)
            if np.degrees(ang) < 2.0:
                break
            if ang > lim:
                rv = rv * lim / ang
            Rc = rotm(rv)
            Pc = (p + st["h"] * a) if about == "B" else 0.5 * (api.tips_world()[0] + api.tips_world()[1])
            Xn, Rn = Pc + Rc @ (X - Pc), Rc @ R
            ok = yield from safe(Xn, Rn, 0.5)
            if not ok:
                break
            X, R = Xn, Rn
            yield from api.hold(0.15)
            say(f"level{about}{j} tilt was {90 - elev:.1f} rot {np.degrees(np.linalg.norm(rv)):.1f}")
        return X, R

    def finish(X, R):
        X, R = yield from level(X, R, about="B", rounds=2)
        p, a, elev, low = pen()
        say(f"standing: tilt={90 - elev:.1f} low={low*1000:.1f} stand={st['stand']:.1f}")
        _, _, Rp = pen_palm()
        yield from api.grip("pen", {"thumb": F_HOLD, "index": F_HOLD},
                            directions={"thumb": obj_dir([0, 0, 1]), "index": obj_dir([0, 0, -1])}, seconds=0.3)
        for k in range(4):
            p, a, elev, low = pen()
            if low > 0.040:
                break
            need = 0.047 - low
            ok = yield from safe(X + np.array([0, 0, need]), R, 0.6 + 8 * need)
            if not ok:
                break
            X = X + np.array([0, 0, need])
        yield from api.hold(0.2)
        X, R = yield from level(X, R, about="pinch", rounds=2)
        p, a, elev, low = pen()
        if low < 0.035:
            ok = yield from safe(X + np.array([0, 0, 0.04 - low]), R, 0.4)
            if ok:
                X = X + np.array([0, 0, 0.04 - low])
        yield from api.hold(0.3)
        p, a, elev, low = pen()
        say(f"final tilt={90 - elev:.1f} low={low*1000:.1f} stand={st['stand']:.1f} F={forces()} gs={api.grip_state()}")
        return X, R

    def retreat(X, R):
        yield from api.ungrip()
        tt = api.tip_targets()
        yield from api.move_tips(tips={"thumb": tt[0] + [0, 0, -0.012], "index": tt[1] + [0, 0, 0.012],
                                       "middle": tt[2] + [-0.02, 0, 0.012]}, seconds=0.4)
        yield from safe(X + np.array([0, 0, 0.06]), R, 0.8)
        yield from safe(X + np.array([0, 0, 0.06]), api.hand_down(0.7), 1.0)
        yield from api.hold(0.4)

    # ------------------------------------------------------------ main
    for attempt in range(3):
        if api.t > 40:
            break
        p0, _ = api.object_pose("pen")
        if np.linalg.norm(p0[:2] - np.array([0.515, -0.07])) > 0.09 or p0[2] > TABLE + 0.006:
            say(f"pen out of place {r3(p0)}; stop")
            break
        X, R = plan()
        ok = yield from approach(X, R)
        if not ok:
            break
        good, mid_t = yield from close()
        if not good:
            say("pinch failed")
            yield from retreat(X, R)
            continue
        X, up = yield from lift_and_hook(X, R, mid_t)
        if not up:
            yield from retreat(X, R)
            continue
        X, R, stood = yield from pivot(X, R)
        if not stood:
            yield from retreat(X, R)
            continue
        X, R = yield from finish(X, R)
        break
