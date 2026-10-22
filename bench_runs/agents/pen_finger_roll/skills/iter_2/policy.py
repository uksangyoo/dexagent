import numpy as np
from scipy.spatial.transform import Rotation

# pen_finger_roll, force level, phase 3.
#
# Idea: a scissor grip. The pen sits in the slot between the index and middle fingers and is
# squeezed by their abduction (grip-force servo), so the pinch line runs along palm y, the swing
# axis. A two-point pinch is a hinge about its pinch line, so the swing (rotation about palm y)
# is the free direction: the earlier agents all pinched along palm z and tried to swing about y,
# which squeezes the pen out.
#
# Swing ("hybrid"): the palm rotates about the index/middle MCP axis (palm y through the MCP
# knuckles) while both MCP joints counter-rotate by the same angle, so the two fingers, and the
# pen they hold, stay still in the world; only the palm turns around them. This changes the
# pen-in-palm tilt whether the pinch is a free hinge (pen hangs by gravity) or a rigid one.
# Roll: shear the two pads in opposite directions perpendicular to the pen (index pad back
# toward the palm, middle pad forward), so the pen rolls between the finger sides.

TABLE = 0.75
RIM = TABLE + 0.06
DEPTH = 0.030                  # contact point below the pen top
BETA = -0.40                   # palm pitch about palm y: keeps the bent fingers' distal links near level
FLEX = (0.40, 0.20, 0.20)      # index/middle MCP, PIP, DIP for the slot
SPREAD = 0.017                 # index pad moved toward the thumb to open the slot
L_C = 0.015                    # slot contact this far proximal of the pad centre along the distal link
F_GRIP = 1.5
LIFT = 0.11
# MCP flex axes from the hand model (dg5fs_right.xml): index (14.3, 28, 107.8) mm, middle (14.3, 5, 112.8) mm.
MCP_I = np.array([0.0143, 0.028, 0.1078])
MCP_M = np.array([0.0143, 0.005, 0.1128])
MCP = 0.5 * (MCP_I + MCP_M)
SEG = (0.0334, 0.0334, 0.0272)  # proximal, middle, MCP-straight-to-pad remainder


def Ry(a):
    return Rotation.from_rotvec([0.0, a, 0.0]).as_matrix()


def u(a):
    return np.array([np.sin(a), 0.0, np.cos(a)])


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _tilt(a):
    return float(np.arctan2(a[2], a[0]))


def run(api):
    D, TS = [], []
    p0, Rp0 = api.object_pose("pen")
    reach = p0[2] - TABLE

    # ---------------------------------------------------------------- scorer replica
    class Mon:
        out = False
        rest = None
        R_last = None
        tw = 0.0
        tw_lo = tw_hi = 0.0
        t_lo = t_hi = 0.0
        bottom = 0.0
        calm = 0
        settled = False

    mon = Mon()

    def mstep():
        p, Rw = api.object_pose("pen")
        _, Rr = api.object_in_palm("pen")
        mon.bottom = p[2] - reach * abs(Rw[2, 2])
        if not mon.out and mon.bottom > RIM + 0.01:
            mon.out = True
            mon.rest = Rr[:, 2].copy()
            mon.R_last = Rr
            D.append(f"rest t={api.t:.2f} axis={np.round(mon.rest, 3).tolist()}")
        if not mon.out:
            return
        if mon.bottom > RIM:
            mon.tw += _twist(mon.R_last, Rr)
            mon.tw_lo, mon.tw_hi = min(mon.tw_lo, mon.tw), max(mon.tw_hi, mon.tw)
            t = np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(mon.rest)))
            mon.t_lo, mon.t_hi = min(mon.t_lo, t), max(mon.t_hi, t)
            if mon.t_hi >= 10 and mon.t_lo <= -10 and abs(t) < 5:
                mon.calm += 1
                mon.settled |= mon.calm >= 25
            else:
                mon.calm = 0
        mon.R_last = Rr

    def W(gen):
        try:
            while True:
                x = next(gen)
                mstep()
                yield x
        except StopIteration as e:
            return e.value

    def tilt_now():
        _, Rr = api.object_in_palm("pen")
        ref = mon.rest if mon.rest is not None else Rr[:, 2]
        return float(np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(ref))))

    def roll_range():
        return float(np.degrees(mon.tw_hi - mon.tw_lo))

    def lean_world():
        _, Rw = api.object_pose("pen")
        return float(np.degrees(np.arccos(min(1.0, abs(Rw[2, 2])))))

    def forces():
        f = api.contact_forces("pen")
        return np.round(f, 2).tolist()

    def lost():
        return mon.out and mon.bottom < RIM + 0.012

    def pen_top():
        p, Rw = api.object_pose("pen")
        up = Rw[:, 2] if Rw[2, 2] > 0 else -Rw[:, 2]
        return p + reach * up

    def pad_model(mcp, m, p, d):
        return mcp + SEG[0] * u(m) + SEG[1] * u(m + p) + SEG[2] * u(m + p + d) + np.array([0.0017, 0, 0])

    def sample(tag, extra=""):
        gs = api.grip_state()
        fi = gs.get("index", {})
        fm = gs.get("middle", {})
        TS.append(f"{tag}{api.t:.1f}:{tilt_now():.1f}/{lean_world():.0f}/{roll_range():.0f}/"
                  f"{fi.get('force_N', 0):.1f},{fm.get('force_N', 0):.1f}/{fi.get('travel_mm', 0):.0f},{fm.get('travel_mm', 0):.0f}{extra}")

    try:
        R0 = api.hand_down() @ Ry(BETA)
        q0 = api.q_target()
        tp0 = api.tips()
        D.append(f"q0 idx={np.round(q0[4:8], 2).tolist()} mid={np.round(q0[8:12], 2).tolist()} "
                 f"tips0 mm={np.round(tp0 * 1000).astype(int).tolist()}")
        m, pj, dj = FLEX
        a_tot = m + pj + dj
        qi = q0[4:8].copy()
        qi[1:] = FLEX
        qm = q0[8:12].copy()
        qm[1:] = FLEX
        ud = u(a_tot)
        c_pred = 0.5 * (pad_model(MCP_I, m, pj, dj) + pad_model(MCP_M, m, pj, dj)) - L_C * ud
        c_pred[1] += 0.5 * SPREAD * 0.85
        top = pen_top()
        Wc = top - np.array([0, 0, DEPTH])
        hi = Wc + np.array([0, 0, DEPTH + 0.07])
        # ---- preshape the slot while moving high above the pen
        yield from W(api.move_joints(xyz=hi - R0 @ c_pred, R=R0, q={"index": qi, "middle": qm}, seconds=2.5))
        tt = api.tip_targets()
        r = yield from W(api.move_tips(tips={"index": tt[1] + np.array([0, SPREAD, 0])}, seconds=0.6))
        yield from W(api.hold(0.3))
        tp = api.tips()
        c = 0.5 * (tp[1] + tp[2]) - L_C * ud
        D.append(f"pads mm idx={np.round(tp[1] * 1000, 1).tolist()} mid={np.round(tp[2] * 1000, 1).tolist()} "
                 f"model idx={np.round(pad_model(MCP_I, m, pj, dj) * 1000, 1).tolist()} spread_ik={r} "
                 f"c={np.round(c * 1000, 1).tolist()} c_pred={np.round(c_pred * 1000, 1).tolist()} q_idx={np.round(api.q()[4:8], 2).tolist()}")

        def approach(c):
            top = pen_top()
            Wc = top - np.array([0, 0, DEPTH])
            above = Wc + np.array([0, 0, DEPTH + 0.015])
            xyz = above - R0 @ c
            yield from W(api.carry(xyz=xyz, R=R0, seconds=1.2))
            yield from W(api.hold(0.4))
            Pm, Rm = api.measured_palm()
            err = above - (Pm + Rm @ c)
            xyz = xyz + np.array([err[0], err[1], 0.0])
            yield from W(api.carry(xyz=xyz, R=R0, seconds=0.4))
            down = xyz - np.array([0, 0, DEPTH + 0.015])
            yield from W(api.carry(xyz=down, R=R0, seconds=1.5))
            yield from W(api.hold(0.3))
            # where is the pen relative to the slot centre (palm frame)?
            pp, Rr = api.object_in_palm("pen")
            a = Rr[:, 2]
            w = c - pp
            off = w - a * (w @ a)
            D.append(f"descent: corr_mm={np.round(err * 1000, 1).tolist()} pen_off_from_slot_mm={np.round(off * 1000, 1).tolist()} "
                     f"lean={lean_world():.1f} F={forces()}")
            return down

        down = yield from approach(c)
        g = yield from W(api.grip("pen", {"index": F_GRIP, "middle": F_GRIP}, seconds=2.0))
        D.append(f"grip {g} lean={lean_world():.1f}")
        fz = api.contact_forces("pen")
        if fz[1] < 0.4 or fz[2] < 0.4:
            # retry once: open, back up, re-measure, re-approach
            D.append("grip weak -> retry")
            yield from W(api.ungrip())
            yield from W(api.move_joints(q={"index": qi, "middle": qm}, seconds=0.5))
            tt = api.tip_targets()
            yield from W(api.move_tips(tips={"index": tt[1] + np.array([0, SPREAD, 0])}, seconds=0.5))
            yield from W(api.carry(xyz=down + np.array([0, 0, DEPTH + 0.03]), R=R0, seconds=1.0))
            tp = api.tips()
            c = 0.5 * (tp[1] + tp[2]) - L_C * ud
            down = yield from approach(c)
            g = yield from W(api.grip("pen", {"index": F_GRIP, "middle": F_GRIP}, seconds=2.0))
            D.append(f"grip2 {g}")

        # ---- lift straight up
        yield from W(api.carry(xyz=down + np.array([0, 0, LIFT]), R=R0, seconds=3.0))
        yield from W(api.hold(0.5))
        sample("L")
        D.append(f"lifted t={api.t:.1f} bottom_rim_mm={1000 * (mon.bottom - RIM):.0f} out={mon.out} tilt={tilt_now():.1f} "
                 f"lean={lean_world():.1f} F={forces()} grip={api.grip_state()}")

        # ---- swing: palm turns about the MCP axis, MCPs counter-rotate (fingers + pen stay put)
        P_ref, R_ref = api.palm_pose()
        W_M = P_ref + R_ref @ MCP
        qt = api.q_target()
        mi0, mm0 = float(qt[5]), float(qt[9])
        st = {"th": 0.0, "k": 0, "sign": 1.0, "chk": False, "t0": tilt_now()}

        def cmd(th, secs=0.1):
            R = R_ref @ Ry(th)
            xyz = W_M - R @ MCP
            q = api.q_target()
            a, b = q[4:8].copy(), q[8:12].copy()
            a[1] = mi0 - th
            b[1] = mm0 - th
            yield from W(api.move_joints(xyz=xyz, R=R, q={"index": a, "middle": b}, seconds=secs))

        def palm_err_deg():
            _, Rm = api.measured_palm()
            _, Rc = api.palm_pose()
            return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rc.T @ Rm).as_rotvec())))

        def seg(T, done, tmax, rate=0.016):
            t0 = api.t
            while api.t - t0 < tmax and not lost():
                e = T - tilt_now()
                if T == 0.0 and abs(e) < 0.8:
                    step = 0.0
                else:
                    step = float(np.clip(np.radians(e) * 0.6 * st["sign"], -rate, rate))
                st["th"] = float(np.clip(st["th"] + step, -0.38, 0.38))
                try:
                    yield from cmd(st["th"])
                except ValueError as ex:
                    D.append(f"arm IK fail th={st['th']:.2f}: {ex}")
                    st["th"] -= step
                    break
                st["k"] += 1
                if not st["chk"] and abs(st["th"]) > 0.10:
                    st["chk"] = True
                    dt = tilt_now() - st["t0"]
                    D.append(f"swing response: th={np.degrees(st['th']):.1f}deg dtilt={dt:.1f}deg")
                    if dt * st["th"] < 0 and abs(dt) > 2.0:
                        st["sign"] = -1.0
                        D.append("swing sign flipped")
                if st["k"] % 4 == 0:
                    sample("s", f"/{np.degrees(st['th']):.0f}")
                if done():
                    break
            D.append(f"swing to {T}: tilt={tilt_now():.1f} th={np.degrees(st['th']):.1f} in {api.t - t0:.1f}s "
                     f"rng[{mon.t_lo:.1f},{mon.t_hi:.1f}] lean={lean_world():.1f} palm_err={palm_err_deg():.1f} F={forces()}")

        yield from seg(17.0, lambda: tilt_now() >= 15.5, 5.0)
        yield from seg(-17.0, lambda: tilt_now() <= -15.5 and mon.t_hi - mon.t_lo >= 31.5, 8.0)
        yield from seg(0.0, lambda: mon.settled and mon.calm >= 40, 6.0)
        D.append(f"after swing settled={mon.settled} calm={mon.calm} grip={api.grip_state()}")

        # ---- roll: index pad back toward the palm, middle pad forward, perpendicular to the pen
        _, Rr = api.object_in_palm("pen")
        a = Rr[:, 2]
        s = np.cross(np.array([0.0, 1.0, 0.0]), a)
        s = s / (np.linalg.norm(s) + 1e-9)
        if s[2] > 0:
            s = -s
        tt = api.tip_targets()
        bi, bm = tt[1].copy(), tt[2].copy()
        d, t0 = 0.0, api.t
        while api.t - t0 < 8.0 and not lost() and roll_range() < 40.0 and d < 0.010:
            d += 0.0002
            r = yield from W(api.move_tips(tips={"index": bi + d * s, "middle": bm - 0.4 * d * s}, seconds=0.1))
            st["k"] += 1
            if st["k"] % 4 == 0:
                sample("r", f"/{d * 1000:.1f}")
        D.append(f"roll: d={d * 1000:.1f}mm s={np.round(s, 2).tolist()} range={roll_range():.1f} tilt={tilt_now():.1f} "
                 f"in {api.t - t0:.1f}s ik={r if d > 0 else '-'} F={forces()}")
        yield from W(api.hold(0.8))
    finally:
        D.append(f"end t={api.t:.2f} bottom_rim_mm={1000 * (mon.bottom - RIM):.1f} tilt[{mon.t_lo:.1f},{mon.t_hi:.1f}] "
                 f"settled={mon.settled} roll={roll_range():.1f} grip={api.grip_state()}")
        for line in D:
            api.note(line)
        for k in range(0, len(TS), 10):
            api.note("TS " + " ".join(TS[k:k + 10]))
