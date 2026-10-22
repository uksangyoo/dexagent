# pen_gravity_pivot at level synergies
#
# Thumb-index pinch near one end of the pen, pinch line horizontal and
# perpendicular to the pen (along the palm's finger axis), long part toward
# the thumb side; the middle finger presses on top of the short part so the
# pen stays level when lifted.  Then roll the hand about the pinch line with
# the pen clamped until the pen hangs ~vertical, lift the middle off (pen
# contacts have no torsional friction -> the pinch is a free hinge) and roll
# the hand back: gravity keeps the pen vertical while it turns in the fingers.
#
# Synergy postures: a linear model tips(s) = T0 + J s (palm frame, measured in
# iteration 1 in the air: mean posture and +1 sd per synergy), solved by
# weighted least squares and refined in the policy by measured Gauss-Newton steps.
import numpy as np

Z = np.array([0.0, 0.0, 1.0])
TH, IX, MI, RI, LI = 0, 1, 2, 3, 4
# Measured in iter 1 [mm]: pads at s = 0, and pad displacement per +1 sd of each synergy.
T0_MEAS = [[82.7, 66.0, 92.3], [83.8, 35.7, 160.0], [93.1, 8.5, 140.1], [83.2, -18.0, 79.9], [88.1, -55.6, 89.2]]
J_MEAS = [[[4.4, -7.2, 3.7], [6.3, -3.1, -41.7], [-9.5, -3.0, -47.5], [-52.1, 0.3, -25.9], [-9.5, 23.1, -20.9]],
          [[-10.8, 9.6, 11.3], [-12.2, -2.4, 14.8], [-14.4, 4.7, 32.4], [2.3, 0.0, 4.1], [-4.9, 16.6, -15.8]],
          [[-8.7, 30.5, -10.1], [4.1, -3.2, -6.7], [0.8, -2.1, -12.7], [-5.7, -0.0, -5.4], [-4.0, -20.9, 19.2]],
          [[0.3, -29.6, 9.9], [-9.2, 0.0, 6.5], [-6.0, 4.3, -1.8], [-1.5, 0.0, 2.3], [-0.1, -7.0, 7.0]],
          [[-9.8, -1.4, 6.8], [4.1, -1.0, -11.8], [2.1, 1.9, -7.0], [3.8, -0.0, 3.1], [-4.7, 6.0, -0.5]],
          [[2.2, -3.1, 2.0], [-4.9, -0.7, 8.2], [2.3, -3.6, -22.1], [5.8, 0.0, 11.6], [0.2, 4.0, -4.2]]]
C0 = np.array([0.085, 0.035, 0.115])     # pinch centre prior, palm frame
HW = 0.0080                              # closed half gap of pad centres (pen r 3.75 + pad ~8 mm -> ~3 mm squeeze each)
G_OFF = 0.032                            # grasp point from the pen centre [m]
LIFT = 0.14
ROLL_DEG = 88.0
BACK_DEG = 78.0
S_C_KNOWN = [1.08, 0.14, 0.74, 1.72, -0.42, -1.25]
PRESS = 0.0015                           # middle pad pressed this far into the pen top


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def rotm(axis, ang):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def line_ang(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(float(unit(u) @ unit(v))), 0, 1))))


def r3(v):
    return np.round(np.asarray(v, float) * 1000, 1).tolist()


def run(api):
    notes, trace = [], []
    try:
        yield from _run(api, notes, trace)
    finally:
        api.note(" | ".join(notes))
        api.note(" | ".join(trace))


def _run(api, notes, trace):
    k = api.k
    T0 = np.array(T0_MEAS) / 1000.0
    J = np.array(J_MEAS).transpose(1, 2, 0) / 1000.0      # (5, 3, k)
    P0, Rpen = api.object_pose("pen")
    a_pen = Rpen[:, 2].copy()
    a_pen[2] = 0
    a_pen = unit(a_pen)
    try:
        Prest, _ = api.object_pose("pen_rest")
        rest_off = float((Prest - P0) @ a_pen)
    except Exception as e:
        rest_off = 0.012
        notes.append(f"no rest pose: {e}")
    notes.append(f"pen {r3(P0)} axis {np.round(a_pen, 3).tolist()} rest_off_mm {rest_off * 1000:.1f}")

    def go(s, sec=0.3, xyz=None, R=None):
        yield from api.move(xyz=xyz, R=R, s=np.asarray(s, float), seconds=sec)

    def lin_solve(Tm, sm, spec, c_prior, wc, s_prior, ws):
        A_rows, b_rows = [], []
        for f, off, w in spec:
            A = np.zeros((3, k + 3))
            A[:, :k] = J[f]
            A[:, k:] = -np.eye(3)
            b = -(Tm[f] - J[f] @ sm - off)
            A_rows.append(A * np.asarray(w)[:, None])
            b_rows.append(b * np.asarray(w))
        A = np.zeros((3, k + 3))
        A[:, k:] = np.eye(3)
        A_rows.append(wc * A)
        b_rows.append(wc * c_prior)
        A = np.zeros((k, k + 3))
        A[:, :k] = np.eye(k)
        A_rows.append(ws * A)
        b_rows.append(ws * s_prior)
        x = np.linalg.lstsq(np.vstack(A_rows), np.concatenate(b_rows), rcond=None)[0]
        return np.clip(x[:k], -2.5, 2.5), x[k:]

    def solve(spec, c_prior, wc, s_prior, ws, rounds=2, label=""):
        # first guess from the stored model, then measured refinement
        s, c = lin_solve(T0, np.zeros(k), spec, c_prior, wc, s_prior, ws)
        for _ in range(rounds):
            yield from go(s, 0.35)
            yield from api.hold(0.15)
            Tm = api.tips()
            s, c = lin_solve(Tm, s, spec, c_prior, wc, s_prior, ws)
        yield from go(s, 0.35)
        yield from api.hold(0.15)
        Tm = api.tips()
        err = {f: r3(Tm[f] - c - off) for f, off, w in spec}
        notes.append(f"{label} s={np.round(s, 2).tolist()} c={r3(c)} err_mm={err} tips={[r3(t) for t in Tm]}")
        return s, c, Tm

    W_T = np.array([2.0, 1.0, 2.0])
    # middle well above the pen in the in-air closed posture; it is pressed on later from the measured pen pose
    spec_c = [(TH, np.array([0, 0, -HW]), W_T), (IX, np.array([0, 0, HW]), W_T),
              (MI, np.array([-0.020, -0.020, 0.0]), np.array([1.0, 0.5, 0.5]))]
    # Iterated measured Gauss-Newton with the mean-posture Jacobian diverged (iter 4); use the closed posture
    # that pinched the pen well in iter 2 and measure its pads in the air.
    s_c = np.array(S_C_KNOWN)
    yield from go(s_c, 0.5)
    yield from api.hold(0.2)
    T_c = api.tips()
    notes.append(f"closed s={s_c.tolist()} tips={[r3(t) for t in T_c]}")

    def local(targets, s_ref, T_ref, ws=0.002):
        """Linear solve around a measured posture: {finger: (palm target, weight3)} -> s."""
        A_rows, b_rows = [], []
        for f, (tgt, w) in targets.items():
            w = np.asarray(w, float)
            W = np.diag(w) if w.ndim == 1 else w
            A_rows.append(W @ J[f])
            b_rows.append(W @ (np.asarray(tgt) - T_ref[f]))
        A_rows.append(ws * np.eye(k))
        b_rows.append(np.zeros(k))
        ds = np.linalg.lstsq(np.vstack(A_rows), np.concatenate(b_rows), rcond=None)[0]
        s = np.clip(s_ref + ds, -2.5, 2.5)
        return s, T_ref + np.einsum("fij,j->fi", J, s - s_ref)

    KEEP = np.array([3.0, 3.0, 3.0])
    # tighter pinch than iter 5 (its pinch force decayed from ~2.3 N to <0.6 N while held): pads 2 mm closer each
    u_c = unit(T_c[IX] - T_c[TH])
    s_c, _ = local({TH: (T_c[TH] + 0.002 * u_c, KEEP), IX: (T_c[IX] - 0.002 * u_c, KEEP),
                    MI: (T_c[MI], np.array([1.0, 1.0, 1.0]))}, s_c, T_c, ws=0.003)
    yield from go(s_c, 0.35)
    yield from api.hold(0.2)
    T_c = api.tips()
    notes.append(f"closed2 s={np.round(s_c, 2).tolist()} tips={[r3(t) for t in T_c[:3]]} gap={np.linalg.norm(T_c[IX] - T_c[TH]) * 1000:.1f}")
    s_o, P_o = local({TH: (T_c[TH] - np.array([0, 0, 0.008]), KEEP), IX: (T_c[IX] + np.array([0, 0, 0.008]), KEEP),
                      MI: (T_c[MI] - np.array([0.006, 0, 0]), np.array([1.0, 0.3, 0.3]))}, s_c, T_c, ws=0.004)
    yield from go(s_o, 0.35)
    yield from api.hold(0.15)
    T_o = api.tips()
    notes.append(f"open s={np.round(s_o, 2).tolist()} pred={[r3(t) for t in P_o[:3]]} meas={[r3(t) for t in T_o[:3]]}")

    # ------------------------------------------------ palm orientation for the grasp
    m = 0.5 * (T_c[TH] + T_c[IX])
    u_p = unit(T_c[IX] - T_c[TH])
    y_p = unit(np.array([0, 1.0, 0]) - u_p[1] * u_p)
    w_p = y_p                                 # long part toward the thumb side
    n_p = np.cross(u_p, w_p)
    Fp = np.column_stack([u_p, w_p, n_p])
    sgn = 1.0 if rest_off >= 0 else -1.0
    w_w = sgn * a_pen                         # long part points toward the rest's side
    best = None
    for nz in (-1.0, 1.0):
        n_w = nz * Z
        Fw = np.column_stack([np.cross(w_w, n_w), w_w, n_w])
        R = Fw @ Fp.T
        if best is None or R[2, 0] < best[2, 0]:
            best = R
    Rg = best
    G = P0 - G_OFF * w_w
    G[2] = P0[2]
    notes.append(f"grasp G {r3(G)} fingers {np.round(Rg[:, 2], 2).tolist()} palmx {np.round(Rg[:, 0], 2).tolist()} m {r3(m)} u_p {np.round(u_p, 2).tolist()}")

    def rec(tag):
        p, Rp = api.object_pose("pen")
        a = Rp[:, 2]
        _, Rin = api.object_in_palm("pen")
        f = api.contact_forces("pen")
        trace.append(f"{tag} t{api.t:.2f} dz{(p[2] - P0[2]) * 1000:.0f} tv{line_ang(a, Z):.0f} "
                     f"ap{np.round(Rin[:, 2], 2).tolist()} f{np.round(f[:3], 2).tolist()}")

    def hold_rec(sec, tag, every=10):
        n = max(1, int(round(sec * 50)))
        for i in range(n):
            yield from api.hold(0.02)
            if i % every == 0:
                rec(tag)

    # ------------------------------------------------ per-step control helpers (iter 7)
    # st["s"] is the finger command applied at every step; st["P"] the linear-model prediction of the commanded pads.
    st = {"s": np.array(s_c, float), "P": None, "lvl": 0, "lvl_best": 0, "tight": 0.0, "ori": 0.0}

    def smooth(x):
        return x * x * (3 - 2 * x)

    def monitor():
        p, Rp = api.object_pose("pen")
        lift = p[2] - P0[2]
        tilt_h = 90.0 - line_ang(Rp[:, 2], Z)
        st["lvl"] = st["lvl"] + 1 if (lift > 0.075 and tilt_h < 12.0) else 0
        st["lvl_best"] = max(st["lvl_best"], st["lvl"])

    def regrip(lo=1.6, goal=2.4, cap=0.004):
        """Pinch-force servo: if the thumb/index force decayed (creep), close both pads along the pinch line."""
        if st["P"] is None or st["tight"] >= cap:
            return
        f = api.contact_forces("pen")
        fm = 0.5 * (f[TH] + f[IX])
        if fm >= lo:
            return
        d = float(np.clip((goal - fm) / 600.0, 0.0003, 0.0012))
        d = min(d, cap - st["tight"])
        tm = api.tips()
        uu = unit(tm[IX] - tm[TH])
        P = st["P"]
        st["s"], st["P"] = local({TH: (P[TH] + d * uu, KEEP), IX: (P[IX] - d * uu, KEEP),
                                  MI: (P[MI], np.array([1.0, 1.0, 1.0]))}, st["s"], P)
        st["tight"] += d
        trace.append(f"regrip t{api.t:.2f} f{fm:.2f} d{d * 1000:.1f} tot{st['tight'] * 1000:.1f}")

    def wrap(gen, every=10):
        i = 0
        for x in gen:
            yield x
            i += 1
            monitor()
            if every and i % every == 0:
                regrip()

    def hold_s(sec, tag=None, until_lvl=None):
        n = max(1, int(round(sec * 50)))
        for i in range(n):
            yield from api.move(s=st["s"], seconds=0.02)
            monitor()
            if i % 10 == 9:
                regrip()
            if tag and i % 15 == 0:
                rec(tag)
            if until_lvl and st["lvl"] >= until_lvl:
                return

    def rot_err():
        _, Rm = api.measured_palm()
        _, Rc = api.palm_pose()
        c = (np.trace(Rc.T @ Rm) - 1) / 2
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def arc(ang, sec, tag):
        """Rotate the palm about the measured pinch line by `ang`, exactly on the circle at every 20 ms step."""
        tw = api.tips_world()
        piv = 0.5 * (tw[TH] + tw[IX])
        ax = unit(tw[IX] - tw[TH])
        X0, R0 = api.palm_pose()
        n = max(1, int(round(sec * 50)))
        for i in range(1, n + 1):
            Q = rotm(ax, ang * smooth(i / n))
            try:
                yield from api.move(xyz=piv + Q @ (X0 - piv), R=Q @ R0, s=st["s"], seconds=0.02)
            except ValueError as e:
                notes.append(f"{tag} IK fail at step {i}/{n}: {e}")
                return
            monitor()
            if i % 10 == 0:
                st["ori"] = max(st["ori"], rot_err())
                regrip()
            if i % 25 == 0:
                rec(tag)
        notes.append(f"{tag} done, max palm R track err {st['ori']:.1f} deg")

    # ------------------------------------------------ approach, grasp, lift
    X_g = api.palm_for(G, m, Rg)
    yield from api.move(xyz=X_g + np.array([0, 0, 0.08]), R=Rg, s=s_o, seconds=1.6)
    yield from api.hold(0.2)
    yield from api.move(xyz=X_g, seconds=0.8)
    yield from api.hold(0.2)
    yield from go(s_c, 0.6)
    yield from api.hold(0.3)
    rec("grasp")
    # middle onto the top of the short part, 20 mm from the pinch, from the measured pen pose
    p_pc, R_in = api.object_in_palm("pen")
    a_in = R_in[:, 2]
    tm = api.tips()
    mm = 0.5 * (tm[TH] + tm[IX])
    q = p_pc + a_in * float((mm - p_pc) @ a_in)
    d_short = unit(q - p_pc)
    u_in = unit(tm[IX] - tm[TH])
    n_up = unit(np.cross(u_in, a_in))
    if n_up[0] > 0:
        n_up = -n_up
    mid_t = q + 0.020 * d_short + n_up * (0.00375 + 0.008 - PRESS)
    Wm = 4.0 * np.outer(n_up, n_up) + 1.0 * np.outer(u_in, u_in) + 0.15 * np.outer(a_in, a_in)
    s_p, P_p = local({TH: (T_c[TH], KEEP), IX: (T_c[IX], KEEP), MI: (mid_t, Wm)}, s_c, T_c)
    notes.append(f"press mid_t {r3(mid_t)} pred {r3(P_p[MI])} s={np.round(s_p, 2).tolist()} q {r3(q)} n_up {np.round(n_up, 2).tolist()}")
    yield from go(s_p, 0.5)
    yield from api.hold(0.3)
    rec("press")
    st["s"], st["P"] = s_p, P_p
    # faster lift than iter 5 (less time with the pen horizontal, where the pinch force creeps away)
    yield from wrap(api.move(xyz=X_g + np.array([0, 0, LIFT]), s=st["s"], seconds=1.0))
    rec("lifted")
    # closed-loop levelling: move the middle along n_up by the long part's elevation error times its lever
    for it in range(4):
        p_pc, R_in = api.object_in_palm("pen")
        tw = api.tips_world()
        pw, _ = api.object_pose("pen")
        pinch_w = 0.5 * (tw[TH] + tw[IX])
        Lw = unit(pw - pinch_w)
        elev = float(np.arcsin(np.clip(Lw[2], -1, 1)))
        a_in = R_in[:, 2]
        n_up = unit(np.cross(u_in, a_in))
        if n_up[0] > 0:
            n_up = -n_up
        d = float(np.clip((elev - np.radians(2.0)) * 0.022, -0.005, 0.005))
        trace.append(f"lev{it} t{api.t:.2f} elev{np.degrees(elev):.1f} d{d * 1000:.1f} lvl{st['lvl']}")
        if abs(elev - np.radians(2.0)) < np.radians(3.0):
            break
        Wm = 4.0 * np.outer(n_up, n_up) + 1.0 * np.outer(u_in, u_in) + 0.15 * np.outer(a_in, a_in)
        P = st["P"]
        s_n, P_n = local({TH: (P[TH], KEEP), IX: (P[IX], KEEP), MI: (P[MI] + d * n_up, Wm)}, st["s"], P)
        yield from wrap(api.move(s=s_n, seconds=0.16), every=0)
        st["s"], st["P"] = s_n, P_n
        yield from hold_s(0.1)
    # hold until the pen has been level (own, stricter check) for 0.56 s, at most 1.6 s
    yield from hold_s(1.6, "lvl", until_lvl=28)
    notes.append(f"level run {st['lvl']} best {st['lvl_best']} steps at t{api.t:.2f}")

    # ------------------------------------------------ clamped roll to vertical (long part down)
    tw = api.tips_world()
    ax = unit(tw[IX] - tw[TH])
    X, R = api.palm_pose()
    wcur = R @ w_p
    sign = 1.0 if (np.cross(ax, wcur) @ Z) < 0 else -1.0
    yield from arc(sign * np.radians(ROLL_DEG), 2.0, "roll")
    yield from hold_s(0.2, "vert")

    # ------------------------------------------------ lift the middle off: the pen hangs on the pinch
    P = st["P"]
    mid_l = P[MI] + 0.022 * u_in + 0.006 * n_up
    s_l, P_l = local({TH: (P[TH], KEEP), IX: (P[IX], KEEP), MI: (mid_l, np.array([1.5, 1.5, 1.5]))}, st["s"], P)
    notes.append(f"loose pred {[r3(t) for t in P_l[:3]]} s={np.round(s_l, 2).tolist()}")
    yield from wrap(api.move(s=s_l, seconds=0.4), every=0)
    st["s"], st["P"] = s_l, P_l
    yield from hold_s(0.4, "free")

    # ------------------------------------------------ roll back: the pen turns in the fingers
    yield from arc(-sign * np.radians(BACK_DEG), 2.2, "back")
    yield from hold_s(1.0, "end")
    rec("final")
    notes.append(f"regrip total {st['tight'] * 1000:.1f} mm, level best {st['lvl_best']} steps")
