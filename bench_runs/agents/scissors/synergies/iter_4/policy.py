import numpy as np

# Plan (iter 3, replaces the untested iter-2 two-loop squeeze, which loses its grip whenever it
# switches from squeezing the inner walls to spreading the outer walls):
# Default palm-down yaw (fingers along the blades, thumb over loop A, as at the start).
#  * ring pad goes into hole B next to B's outer wall, little pad outside B's outer wall;
#    little moves toward ring -> pinches B's outer wall (rigid hold of part B, independent of the hinge).
#  * thumb pad goes into hole A. Synergy s3 is ~pure thumb sideways motion (palm y, -26 mm/sd):
#    s3 down -> thumb pushes A's outer wall out -> blades open; s3 up -> pushes A's inner wall -> closed.
#    Cycles are run as closed-loop steps on s3 with the measured hinge angle.
#  * curled finger links sit over the loops' far ends, which stops the blades from drooping.
# Home-made pad IK on the synergies: linear model pads(s) = t + J ds with J measured in iter 1
# (1 sd finite differences, palm down, in the air), damped least squares with active-set
# inequality rows (keep middle/index above the pads), refined on measured pads in the air.

TABLE = 0.75
TH, IX, MI, RI, LI = 0, 1, 2, 3, 4
REST_Z = 4.7         # mm, pivot frame above table when lying flat (nominal size, measured iter 1)
PAD_Z = 9.5          # mm, pad-centre height above the table when inserted (walls 11 mm tall)
PAD_R = 10.0         # mm, effective fingertip radius (iter 3: pads 8.5 mm from a wall landed on it)

T0 = np.array([[84, 57, 93], [82, 35, 162], [92, 10, 140], [86, -18, 84], [88, -57, 92]], float)  # mm
J0 = np.array([[5, -14, 3, 6, -3, -43, -12, -5, -49, -55, 0, -31, -8, 23, -23],
               [-1, -2, -11, 11, 1, -19, 0, -5, -34, -3, 0, -5, -7, -10, 25],
               [-9, -19, -2, -1, 5, 3, -2, 0, 14, 5, 0, 6, -8, 20, -17],
               [-1, -26, 3, -10, 1, 10, -6, 3, 2, -4, 0, 0, 0, -4, 5],
               [-10, -13, 9, -1, 0, 1, 2, -2, -16, 3, 0, 7, -4, 6, -1],
               [-7, 2, 2, 5, 1, -13, -4, 6, 12, -3, 0, -7, -3, 2, 4]], float).T  # (15, 6) mm/sd

_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]


def hinge(api):
    return api.object_joint("scissors_hinge")


def clearance(api):
    p, R = api.object_pose("scissors")
    a = hinge(api)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    pts = np.vstack([_A, _B @ Rz.T]) @ R.T + p
    return float(pts[:, 2].min() - abs(R[2, 2]) * 0.0055 - TABLE)


def fc(api):
    return np.round(api.contact_forces("scissors"), 2).tolist()


# ---------------------------------------------------------------- synergy pad IK (skill)
def dvec(i, j, ax):
    a = np.zeros(15)
    a[3 * i + ax] += 1
    a[3 * j + ax] -= 1
    return a


def grasp_spec(sepTR, sepRL):
    """thumb/ring/little pads at one depth; thumb-ring and ring-little sideways separations (mm);
    middle/index pads kept shallower than the ring pad; thumb/little roughly level with ring along fingers."""
    # iter 4: fewer, softer demands so s stays within ~1 sd where the linear model holds
    # (iter 3 asked middle 12 mm above ring -> s saturated at +-3.2, thumb pad retracted 43 mm).
    eq = [(dvec(TH, RI, 0), 0.0, 0.7), (dvec(LI, RI, 0), 0.0, 0.7),
          (dvec(TH, RI, 1), sepTR, 1.5), (dvec(RI, LI, 1), sepRL, 1.5)]
    ineq = [(dvec(MI, RI, 0), 0.0, 1.0),
            (dvec(TH, RI, 2), 14.0, 0.7), (-dvec(TH, RI, 2), 6.0, 0.7),
            (dvec(LI, RI, 2), 15.0, 0.7), (-dvec(LI, RI, 2), 15.0, 0.7)]
    return eq, ineq


def solve(J, t, s_ref, eq, ineq, lam=6.0, smax=2.0, step=0.8):
    tf = np.asarray(t, float).ravel()
    k = J.shape[1]
    rows = [w * (a @ J) for a, v, w in eq]
    rhs = [w * (v - a @ tf) for a, v, w in eq]
    active = []
    s_new = s_ref.copy()
    for _ in range(6):
        A = np.vstack(rows + [ineq[i][2] * (ineq[i][0] @ J) for i in active] + [lam * np.eye(k)])
        b = np.concatenate([rhs, [ineq[i][2] * (ineq[i][1] - 1.0 - ineq[i][0] @ tf) for i in active], np.zeros(k)])
        ds = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.abs(ds).max() > step:  # trust region
            ds = ds * step / np.abs(ds).max()
        s_new = np.clip(s_ref + ds, -smax, smax)
        p = tf + J @ (s_new - s_ref)
        add = [i for i, (a, ub, w) in enumerate(ineq) if i not in active and a @ p > ub + 0.5]
        if not add:
            break
        active += add
    return s_new


def spec_err(t, eq, ineq):
    tf = np.asarray(t, float).ravel()
    e = [abs(a @ tf - v) for a, v, w in eq] + [max(0.0, a @ tf - ub) for a, ub, w in ineq]
    return float(max(e))


def settle(api, s, move_s=0.25, hold_s=0.3):
    yield from api.move(s=s, seconds=move_s)
    yield from api.hold(hold_s)
    return api.tips() * 1000.0


def refine(api, J, s, eq, ineq, n=5, tol=2.0, tag=""):
    """measured-feedback LS; J (mm/sd) is Broyden-updated in place from each measured step."""
    t = yield from settle(api, s)
    for _ in range(n):
        if spec_err(t, eq, ineq) < tol:
            break
        s1 = solve(J, t, s, eq, ineq)
        t1 = yield from settle(api, s1)
        ds = s1 - s
        if ds @ ds > 1e-4:
            J += np.outer((t1 - t).ravel() - J @ ds, ds) / (ds @ ds)
        s, t = s1, t1
    api.note(f"refine {tag}: err={spec_err(t, eq, ineq):.1f}mm s={np.round(s, 2).tolist()} "
             f"pads mm={np.round(t).astype(int).tolist()}")
    return s, t


# ---------------------------------------------------------------- policy
def run(api):
    J = J0.copy()
    k = api.k
    P0, _ = api.palm_pose()
    s0 = np.zeros(k)
    yield from api.move(s=s0, seconds=0.6)
    yield from api.hold(0.4)
    p, R = api.object_pose("scissors")
    psi = float(np.arctan2(R[1, 0], R[0, 0]))
    sc = float(np.clip((p[2] - TABLE) * 1000.0 / REST_Z, 0.88, 1.12))
    api.note(f"k={k} scissors p={np.round(p, 4).tolist()} psi={psi:.3f} sc={sc:.3f}")
    if k != J0.shape[1]:  # fallback: measure the Jacobian
        t0 = yield from settle(api, s0)
        J = np.zeros((15, k))
        for j in range(k):
            e = s0.copy(); e[j] = 1.0
            t = yield from settle(api, e)
            J[:, j] = (t - t0).ravel()
            yield from settle(api, s0, 0.2, 0.1)

    Rh = api.hand_down(psi)
    # stage high above the loops with the scissors' yaw
    loops = p + R @ np.array([-0.065 * sc, 0.0, 0.0])
    stage = api.palm_for(loops + np.array([0, 0, 0.13]), np.array([0.085, 0.02, 0.088]), Rh)
    yield from api.move(stage, Rh, seconds=1.5)

    # separations (mm, palm y): ring in hole B near its outer wall, thumb mid hole A, little outside B
    # (iter 4) thumb and ring at the hole centres (y = +-24 sc), little outside B's outer wall
    sepTR_ins = 48.0 * sc
    sepRL_ins = 18.5 * sc + PAD_R + 3.5
    sepRL_touch = 18.5 * sc + PAD_R
    sepRL_pinch = 5.0 * sc + 2 * PAD_R - 12.0
    s_ins, t_ins = yield from refine(api, J, s0.copy(), *grasp_spec(sepTR_ins, sepRL_ins), tag="insert")
    s_touch, t_touch = yield from refine(api, J, s_ins.copy(), *grasp_spec(sepTR_ins + 3.0, sepRL_touch), n=3, tag="touch")
    s_pinch = solve(J, t_touch, s_touch, *grasp_spec(sepTR_ins + 3.0, sepRL_pinch), step=1.5)
    api.note(f"s_pinch={np.round(s_pinch, 2).tolist()}")
    yield from settle(api, s_ins)

    ok = False
    down = stage
    for attempt in range(2):
        p, R = api.object_pose("scissors")
        psi = float(np.arctan2(R[1, 0], R[0, 0]))
        Rh = api.hand_down(psi)
        t_m = api.tips() * 1000.0
        dz = t_m[TH, 2] - t_m[RI, 2]  # thumb further along the fingers = further toward the pivot
        x_th = -65.0 * sc + 0.5 * float(np.clip(dz, -14, 14))  # both pads as central in the holes as dz allows
        x_ri = x_th - dz
        y_ri = -0.5 * (t_m[TH, 1] - t_m[RI, 1])   # holes are symmetric about y = 0: split any sep error
        y_th = -y_ri
        tgt = {TH: np.array([x_th, y_th, 0.0]) / 1000, RI: np.array([x_ri, y_ri, 0.0]) / 1000}
        W = {i: p + R @ v for i, v in tgt.items()}
        for i in W:
            W[i][2] = TABLE + PAD_Z / 1000
        mid_w = 0.5 * (W[TH] + W[RI])
        mid_p = 0.5 * (t_m[TH] + t_m[RI]) / 1000
        above = api.palm_for(mid_w + np.array([0, 0, 0.06]), mid_p, Rh)
        yield from api.move(above, Rh, s=s_ins, seconds=1.5)
        yield from api.hold(0.5)
        for c in range(2):  # correct the arm's placement error with the measured pads
            tw = api.tips_world()
            err = 0.5 * ((W[TH] - tw[TH]) + (W[RI] - tw[RI])) + np.array([0, 0, 0.06])
            api.note(f"att{attempt} place corr{c}: err mm={np.round(err * 1000, 1).tolist()}")
            above = above + np.array([err[0], err[1], 0.0])
            if np.linalg.norm(err[:2]) < 0.0015:
                break
            yield from api.move(above, Rh, seconds=0.4)
            yield from api.hold(0.4)
        tw = api.tips_world()
        zpad = 0.5 * (tw[TH, 2] + tw[RI, 2])
        down = above - np.array([0, 0, zpad - (TABLE + PAD_Z / 1000)])
        # guarded descent
        for f in (0.5, 0.75, 0.9, 1.0):
            yield from api.move(above + f * (down - above), Rh, seconds=0.45)
            fcs = api.contact_forces("scissors")
            if fcs[[TH, RI, LI]].max() > 6.0:  # middle may rest on the shank (compliant), ignore it
                api.note(f"descent blocked at f={f} forces={np.round(fcs, 2).tolist()}")
                down = above + max(f - 0.08, 0.0) * (down - above)
                yield from api.move(down, Rh, seconds=0.2)
                break
        yield from api.hold(0.2)
        tw = api.tips_world()
        p, R = api.object_pose("scissors")
        lp = lambda w: np.round((R.T @ (w - p)) * 1000, 1).tolist()
        api.note(f"att{attempt} inserted: thumb={lp(tw[TH])} ring={lp(tw[RI])} little={lp(tw[LI])} "
                 f"(scissors frame mm) pad z mm={np.round((tw[:, 2] - TABLE) * 1000, 1).tolist()} forces={fc(api)}")
        # pinch B's outer wall between ring and little
        yield from api.move(s=s_pinch, seconds=0.6)
        yield from api.hold(0.4)
        api.note(f"pinched: hinge={np.degrees(hinge(api)):.1f} forces={fc(api)}")
        yield from api.move(down + np.array([0, 0, 0.05]), Rh, seconds=0.8)
        yield from api.hold(0.3)
        cl = clearance(api)
        api.note(f"lift5: clearance={cl * 1000:.1f}mm pivot dz={(api.object_pose('scissors')[0][2] - p[2]) * 1000:.1f} forces={fc(api)}")
        if cl > 0.015:
            ok = True
            break
        # failed: put down, release, retry from the new pose
        yield from api.move(down + np.array([0, 0, 0.003]), Rh, seconds=0.6)
        yield from api.move(s=s_ins, seconds=0.4)
        yield from api.move(down + np.array([0, 0, 0.06]), Rh, seconds=0.6)
        yield from api.hold(0.3)

    up = down + np.array([0, 0, 0.23])
    yield from api.move(up, Rh, seconds=1.6)
    yield from api.hold(0.4)
    cl = clearance(api)
    _, Rs = api.object_pose("scissors")
    api.note(f"lifted: clearance={cl * 1000:.0f}mm Rzz={Rs[2, 2]:.3f} hinge={np.degrees(hinge(api)):.1f} forces={fc(api)}")

    # ---- open / close cycles in the air: closed-loop on s3 (thumb sideways)
    s_cur = s_pinch.copy()
    s3_open, s3_close = None, None
    cycles = 0
    for n in range(4):
        if clearance(api) < 0.025:
            api.note(f"cycle{n}: not in the air (clearance {clearance(api) * 1000:.0f}mm), stop")
            break
        # open: thumb outward (s3 down)
        if s3_open is not None:
            s_cur[3] = s3_open + 0.15
            yield from api.move(s=s_cur, seconds=0.5)
            yield from api.hold(0.2)
        start = s_cur[3]
        while hinge(api) < 0.27 and s_cur[3] > s_pinch[3] - 3.0:
            s_cur[3] -= 0.2
            yield from api.move(s=s_cur, seconds=0.25)
            yield from api.hold(0.1)
        yield from api.hold(0.2)
        a_open = hinge(api)
        s3_open = s_cur[3] if a_open > 0.2 else None
        # close: thumb inward (s3 up)
        if s3_close is not None:
            s_cur[3] = s3_close - 0.15
            yield from api.move(s=s_cur, seconds=0.5)
            yield from api.hold(0.2)
        while hinge(api) > 0.02 and s_cur[3] < s_pinch[3] + 2.5:
            s_cur[3] += 0.2
            yield from api.move(s=s_cur, seconds=0.25)
            yield from api.hold(0.1)
        s_cur[3] += 0.2   # press the inner wall a bit: blades against the closed stop
        yield from api.move(s=s_cur, seconds=0.25)
        yield from api.hold(0.3)
        a_close = hinge(api)
        s3_close = s_cur[3]
        cl = clearance(api)
        api.note(f"cycle{n}: open {np.degrees(a_open):.1f}deg (s3 {start:.2f}->{s3_open}) close {np.degrees(a_close):.1f}deg "
                 f"(s3 {s3_close:.2f}) clear {cl * 1000:.0f}mm forces={fc(api)}")
        if a_open > 0.19 and a_close < 0.045 and cl > 0.02:
            cycles += 1
        if cycles >= 3:
            break

    # ---- set down (blades held closed by the thumb), release, retreat
    yield from api.move(down + np.array([0, 0, 0.004]), Rh, seconds=2.5)
    yield from api.hold(0.4)
    p, R = api.object_pose("scissors")
    api.note(f"set down: hinge={np.degrees(hinge(api)):.1f} pivot z={(p[2] - TABLE) * 1000:.1f}mm Rzz={R[2, 2]:.3f}")
    for _ in range(3):
        p, R = api.object_pose("scissors")
        if p[2] - TABLE < 0.010 and R[2, 2] > 0.97:
            break
        down = down - np.array([0, 0, 0.003])
        yield from api.move(down, Rh, seconds=0.4)
        yield from api.hold(0.3)
    s_rel = s_ins.copy()
    s_rel[3] = 0.5 * (s_ins[3] + s_cur[3])   # thumb toward the hole centre, still near the inner wall side
    yield from api.move(s=s_rel, seconds=0.5)
    yield from api.hold(0.3)
    yield from api.move(down + np.array([0, 0, 0.12]), Rh, seconds=1.4)
    yield from api.hold(0.4)
    p, R = api.object_pose("scissors")
    api.note(f"final: hinge={np.degrees(hinge(api)):.1f} pivot z={(p[2] - TABLE) * 1000:.1f}mm Rzz={R[2, 2]:.3f}")
