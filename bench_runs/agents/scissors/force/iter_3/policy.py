# scissors at level force
#
# Strategy: hold part A by an external thumb/index pinch ALONG the length of
# loop A (thumb on the back face, index on the front face, force servo), so the
# pinch has a 45 mm lever against the blades drooping.  A third finger (middle
# or ring, chosen at runtime by IK residual) sits inside loop B and swings part
# B open/closed under closed-loop hinge-angle feedback (position control).
import numpy as np

TABLE = 0.75
NAMES = ("thumb", "index", "middle", "ring", "little")
F_PINCH = 7.0          # N per pinch pad (servo regulates |contact force|)
R_PINCH = 0.008        # pad centre to pad surface (pinch pads)
R_SIDE = 0.009         # pad centre to side surface (finger in loop B)
PEN = 0.0            # how far B finger targets press into a wall
A_OPEN = np.radians(22.0)
A_CLOSED = np.radians(0.8)
LEAD = np.radians(12.0)
F_B = 1.5              # N, B-finger push on loop B (pivot friction ~0.6-0.9 N at the loop)

# check.py extreme points (nominal), for my own clearance estimate
_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def clearance(p, R, a):
    pts = np.vstack([_A, _B @ Rz(a).T]) @ R.T + p
    return float(pts[:, 2].min() - abs(R[2, 2]) * 0.0055 - TABLE)


def run(api):
    diag = []

    def D(s):
        diag.append(f"{api.t:6.2f} {s}")

    def f3(v):
        return np.round(np.asarray(v, float) * 1000, 1).tolist()

    try:
        yield from _run(api, D, f3)
    finally:
        for line in diag[-45:]:
            api.note(line)


def _run(api, D, f3):
    p_init, _ = api.object_pose("scissors")
    s = float(np.clip((p_init[2] - TABLE) / 0.0068, 0.88, 1.12))   # start pivot height = 6.8 mm * scale
    others_all = ("middle", "ring", "little")
    yield from api.hold(0.4)                                          # let the scissors settle
    p0, R0 = api.object_pose("scissors")
    psi = float(np.arctan2(R0[1, 0], R0[0, 0]))
    D(f"start pivot={f3(p_init)} settled_z={(p0[2] - TABLE) * 1000:.2f}mm yaw={np.degrees(psi):.1f} size_est={s:.3f}")
    Rp = api.hand_down(psi)
    M = Rp.T @ R0

    H_PINCH = 0.006                   # commanded pad-centre height above table (tips then rest on the table)
    zp = H_PINCH - (p0[2] - TABLE)
    zB = 0.008 - (p0[2] - TABLE)
    G = np.array([-0.065 * s, 0.024 * s, zp])
    half = 0.0225 * s + R_PINCH
    clr = 0.004

    def B_pt(a, mode, z=None):
        y = {"centre": -0.024 * s, "open": -0.0375 * s + R_SIDE - PEN, "close": -0.0105 * s - R_SIDE + PEN}[mode]
        return Rz(a) @ np.array([-0.065 * s, y, zB if z is None else z])

    def plan(C, pt):
        return C + M @ (np.asarray(pt) - G)

    best = None
    for cx in (0.07, 0.08, 0.09):
        for cy in (0.025, 0.04, 0.055):
            for cz in (0.10, 0.115, 0.13):
                C = np.array([cx, cy, cz])
                e_t = max(api.reach("thumb", plan(C, G + [-(half + clr), 0, 0])),
                          api.reach("thumb", plan(C, G + [-(half - 0.002), 0, 0])))
                e_i = max(api.reach("index", plan(C, G + [half + clr, 0, 0])),
                          api.reach("index", plan(C, G + [half - 0.002, 0, 0])))
                if max(e_t, e_i) > 0.006:
                    cand = [(max(e_t, e_i) + 0.01, "middle")]
                else:
                    cand = []
                    for fb in ("middle", "ring"):
                        e_b = max(api.reach(fb, plan(C, B_pt(0, "centre"))),
                                  api.reach(fb, plan(C, B_pt(0, "close"))),
                                  api.reach(fb, plan(C, B_pt(A_OPEN + 0.1, "open"))))
                        cand.append((max(e_t, e_i, e_b), fb))
                for e, fb in cand:
                    score = e + 0.02 * np.linalg.norm(C - [0.08, 0.04, 0.118])
                    if best is None or score < best[0]:
                        best = (score, e, C, fb)
    _, err, C, FB = best
    iB = NAMES.index(FB)
    D(f"plan C={f3(C)} Bfinger={FB} max_resid_mm={err * 1000:.1f}")
    others = [f for f in others_all if f != FB]
    yield from api.tuck(others, seconds=0.6)

    F = F_PINCH
    lifted = False
    for attempt in range(2):
        p0, R0 = api.object_pose("scissors")
        M = Rp.T @ R0
        W_G = p0 + R0 @ G
        xyz_g = W_G - Rp @ C
        down = M @ np.array([0, 0, -1.0])
        dirs = {"thumb": down, "index": down, FB: down}
        targets = {"thumb": plan(C, G + [-(half + clr), 0, 0]),
                   "index": plan(C, G + [half + clr, 0, 0]),
                   FB: plan(C, B_pt(0, "centre"))}
        r = yield from api.move_tips(xyz_g + [0, 0, 0.05], Rp, tips=targets, seconds=1.8, directions=dirs)
        D(f"try{attempt} preshape ik_mm={r.get('ik_residual_mm')}")
        yield from api.carry(xyz_g, Rp, seconds=1.3)
        yield from api.hold(0.3)
        # touch-down calibration: fingers blocked by the table get their target raised to just above contact
        meas = api.tips()
        _, Rm = api.measured_palm()
        up = Rm.T @ np.array([0, 0, 1.0])
        tt = api.tip_targets()
        new, blocked = {}, {}
        for f in ("thumb", "index", FB):
            i = NAMES.index(f)
            b = float((meas[i] - tt[i]) @ up)
            blocked[f] = round(b * 1000, 1)
            lift = 0.003 if f == FB else -0.0015      # pinch pads: small preload into the table; B finger: hover
            new[f] = tt[i] + up * (max(b, 0.0) + lift)
        r = yield from api.move_tips(tips=new, seconds=0.4, directions=dirs)
        yield from api.hold(0.2)
        tw = api.tips_world()
        zB_cal = float((R0.T @ (tw[iB] - p0))[2])
        D(f"try{attempt} blocked_mm={blocked} pad_z_above_table_mm thumb={(tw[0, 2] - TABLE) * 1000:.1f} "
          f"index={(tw[1, 2] - TABLE) * 1000:.1f} {FB}={(tw[iB, 2] - TABLE) * 1000:.1f} zB_cal={zB_cal * 1000:.1f} forces={np.round(api.contact_forces('scissors'), 2).tolist()}")

        r = yield from api.grip("scissors", {"thumb": F, "index": F},
                                directions={"thumb": [1.0, 0, 0], "index": [-1.0, 0, 0]},
                                seconds=1.5, max_travel=0.03)
        D(f"try{attempt} grip {r}")
        yield from api.carry(xyz_g + [0, 0, 0.03], Rp, seconds=1.2)
        yield from api.hold(0.2)
        p, R = api.object_pose("scissors")
        rise = p[2] - p0[2]
        D(f"try{attempt} test lift rise={rise * 1000:.1f}mm tiltz={R[2, 2]:.3f} grip={api.grip_state()}")
        if rise > 0.015:
            lifted = True
            break
        # failed: put back, open the pinch, retry lower and harder
        yield from api.carry(xyz_g, Rp, seconds=0.8)
        yield from api.ungrip()
        cur = api.tip_targets()
        yield from api.move_tips(tips={"thumb": cur[0] + M @ [-0.008, 0, 0], "index": cur[1] + M @ [0.008, 0, 0]}, seconds=0.4)
        yield from api.carry(xyz_g + [0, 0, 0.05], Rp, seconds=0.8)
        G = G + [0, 0, -0.0015]
        F = F + 1.5

    # ---- lift
    yield from api.carry(xyz_g + [0, 0, 0.09], Rp, seconds=2.0)
    yield from api.hold(0.3)
    p, R = api.object_pose("scissors")
    a = api.object_joint("scissors_hinge")
    D(f"lifted clear={clearance(p, R, a) * 1000:.1f}mm tiltz={R[2, 2]:.3f} hinge={np.degrees(a):.1f} grip={api.grip_state()}")

    def B_target(a_cmd, mode):
        po, Ro = api.object_in_palm("scissors")
        return po + Ro @ B_pt(a_cmd, mode, z=zB_cal + 0.001), Ro @ np.array([0, 0, -1.0])

    def drive(goal_open):
        # B finger on the force servo: its nominal target is planted on the loop-B wall at the
        # current angle, then the servo pushes along the wall normal (object frame) with a bounded
        # force (above the pivot friction) until the measured hinge angle reaches the goal.
        a = api.object_joint("scissors_hinge")
        mode = "open" if goal_open else "close"
        tgt, dn = B_target(a, mode)
        yield from api.move_tips(tips={FB: tgt}, seconds=0.25, directions={FB: dn})
        if goal_open:
            n = Rz(0.19) @ np.array([0, -1.0, 0])
            yield from api.grip("scissors", {FB: F_B}, directions={FB: n}, seconds=0.04, max_travel=0.045)
        else:
            n = Rz(max(a, 0.0) * 0.5) @ np.array([0, 1.0, 0])
            yield from api.grip("scissors", {FB: F_B + 0.5}, directions={FB: n}, seconds=0.04, max_travel=0.045)
        t_end = api.t + 5.0
        peak, minclear = 0.0, 1.0
        while api.t < t_end:
            a = api.object_joint("scissors_hinge")
            p, R = api.object_pose("scissors")
            minclear = min(minclear, clearance(p, R, a))
            peak = max(peak, a)
            if goal_open and a >= A_OPEN:
                break
            if not goal_open and a <= A_CLOSED:
                break
            yield from api.hold(0.04)
        st = api.grip_state().get(FB)
        if not goal_open:
            yield from api.hold(0.3)
        yield from api.ungrip([FB])
        D(f"drive open={goal_open} a={np.degrees(api.object_joint('scissors_hinge')):.1f} B={st}")
        return api.object_joint("scissors_hinge"), peak, minclear

    done = 0
    for k in range(4):
        if done >= 2:
            break
        a1, pk, mc1 = yield from drive(True)
        yield from api.hold(0.15)
        a2, _, mc2 = yield from drive(False)
        a3 = api.object_joint("scissors_hinge")
        ok = np.degrees(pk) > 11 and np.degrees(a3) < 2.5 and min(mc1, mc2) > 0.022
        done += int(ok)
        D(f"cycle {k}: open->{np.degrees(a1):.1f} peak {np.degrees(pk):.1f} closed->{np.degrees(a3):.1f} "
          f"minclear={min(mc1, mc2) * 1000:.1f}mm ok={ok} grip={api.grip_state()}")

    # ---- set down, keep B pressed closed
    yield from api.carry(xyz_g + [0, 0, 0.004], Rp, seconds=2.5)
    yield from api.hold(0.3)
    p, R = api.object_pose("scissors")
    D(f"set down: rest={(p[2] - TABLE) * 1000:.1f}mm flat={R[2, 2]:.3f} hinge={np.degrees(api.object_joint('scissors_hinge')):.1f}")

    # ---- release: B finger to hole centre and up, then open the pinch, then palm up
    tgt, dn = B_target(0.0, "centre")
    yield from api.move_tips(tips={FB: tgt}, seconds=0.3, directions={FB: dn})
    yield from api.move_tips(tips={FB: tgt + np.array([-0.035, 0, 0])}, seconds=0.6)
    yield from api.ungrip()
    cur = api.tip_targets()
    yield from api.move_tips(tips={"thumb": cur[0] + [0, 0, -0.012], "index": cur[1] + [0, 0, 0.012]}, seconds=0.5)
    yield from api.carry(api.palm_pose()[0] + [0, 0, 0.08], Rp, seconds=1.0)
    yield from api.hold(0.3)
    p, R = api.object_pose("scissors")
    D(f"end: rest={(p[2] - TABLE) * 1000:.1f}mm flat={R[2, 2]:.3f} hinge={np.degrees(api.object_joint('scissors_hinge')):.1f}")
