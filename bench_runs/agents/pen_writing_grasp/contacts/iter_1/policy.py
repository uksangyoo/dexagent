"""pen_writing_grasp, contacts level.

Plan: hand rolled 90 deg about the fingers so the thumb is down and the palm faces
sideways; the pen then runs along palm -y (up) in front of the palm at the
thumb-index pinch spot (palm x 0.08, z 0.118), so the pen's upper part passes
freely in front of the fingers.  Three-point grip: index (lower) and middle
(upper) pads on the +z side, thumb on the -z side between them, so tilt of
the pen is braced by normal forces.  Lift along the pen axis, straighten the
pen vertical, put the tip down away from the cup with a closed-loop descent,
draw a 50 mm stroke while regulating the tip's measured deflection in the
hand (light contact), lift and hold.
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
R_NOM = 0.00375
C_PEN = np.array([0.080, 0.040, 0.118])   # palm-frame pen axis point at the index contact
YAW = 0.7
SQUEEZE = 0.005


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def pen_axes(api):
    p, R = api.object_pose("pen")
    sgn = 1.0 if R[2, 2] >= 0 else -1.0
    up = sgn * R[:, 2]
    return p, R, up, sgn


def run(api):
    diag = {}
    try:
        yield from _run(api, diag)
    finally:
        for k, v in diag.items():
            api.note(f"DIAG {k}: {v}")


def _run(api, diag):
    yield from api.hold(0.3)
    p, R, up, sgn = pen_axes(api)
    lean0 = float(np.degrees(np.arccos(min(1, up[2]))))
    cosl = up[2]
    E = (p[2] - TABLE - R_NOM * (1 - cosl)) / cosl          # centre -> lower pole distance (half + r)
    home = p - up * E
    diag["start"] = f"p={np.round(p, 4).tolist()} lean={lean0:.1f} E={E * 1000:.1f}mm home={np.round(home[:2], 4).tolist()}"

    def tip_state():
        p, R, up, _ = pen_axes(api)
        tip = p - up * E
        lean = np.arccos(min(1.0, up[2]))
        h = tip[2] - TABLE - R_NOM * (1 - np.cos(lean))
        return tip, h, np.degrees(lean)

    R_side = api.hand_down(YAW) @ Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()  # thumb down
    f_dir = R_side[:, 2]

    # ---- choose finger layout by fingertip IK reach (palm-frame pad targets at the grasp)
    off = R_NOM + api.pad_offset - SQUEEZE

    def pad(y, side):
        return np.array([C_PEN[0], y, C_PEN[2] + side * off])
    res_i = api.reach("index", pad(0.040, +1))
    mids = [(api.reach("middle", pad(y, +1)), y) for y in (0.018, 0.014, 0.022, 0.026)]
    okm = [m for m in mids if m[0] < 0.003]
    best_m = okm[0] if okm else min(mids)
    use_mid = bool(okm)
    y_m = best_m[1]
    cands = (0.029, 0.033, 0.040) if use_mid else (0.040,)
    th = [(api.reach("thumb", pad(y, -1)), y) for y in cands]
    ok = [t for t in th if t[0] < 0.003]
    y_t = ok[0][1] if ok else min(th)[1]
    diag["layout"] = (f"res index {res_i * 1000:.1f} middle {best_m[0] * 1000:.1f}@{y_m} thumb "
                      f"{[(round(a * 1000, 1), b) for a, b in th]} -> use_mid={use_mid} y_t={y_t}")
    fingers = {"index": (0.040, +1), "thumb": (y_t, -1)}
    if use_mid:
        fingers["middle"] = (y_m, +1)

    W_I = 0.832  # world height of the index contact at the grasp (clears the 0.81 rim)

    def grasp_frame():
        p, R, up, sgn = pen_axes(api)
        Rg = rot_between(np.array([0, 0, 1.0]), up) @ R_side
        w_i = (W_I - p[2]) / up[2]
        return p, R, up, Rg, w_i

    def contact_points(R, up, Rg, w_i, lift=0.0):
        pts, nrm = {}, {}
        for fn, (y, side) in fingers.items():
            n_w = Rg @ np.array([0, 0, side])
            w = w_i + (0.040 - y) + lift
            pts[fn] = R.T @ (up * w + R_NOM * n_w)
            nrm[fn] = R.T @ n_w
        return pts, nrm

    # ---- pre-grasp: above the pen, aligned with its lean
    p, R, up, Rg, w_i = grasp_frame()
    Pg = p + up * w_i - Rg @ C_PEN
    PRE = 0.13
    yield from api.carry(Pg + up * PRE + np.array([0, 0, 0.03]), Rg, seconds=2.5)
    yield from api.carry(Pg + up * PRE, Rg, seconds=0.5)
    pts, nrm = contact_points(R, up, Rg, w_i, lift=PRE)
    r = yield from api.approach("pen", pts, standoff=0.015, seconds=1.0, normals=nrm)
    diag["approach"] = r.get("ik_residual_mm")
    yield from api.carry(Pg, Rg, seconds=1.8)
    p, R, up, Rg2, w_i = grasp_frame()
    pts, nrm = contact_points(R, up, Rg, w_i)
    r = yield from api.grasp("pen", pts, squeeze=SQUEEZE, seconds=0.8, normals=nrm)
    yield from api.hold(0.3)
    diag["grasp"] = f"res={r.get('ik_residual_mm')} F={np.round(api.contact_forces('pen'), 2).tolist()}"

    # ---- lift out along the pen axis, then straight up
    P0, R0 = api.palm_pose()
    _, _, up, _ = pen_axes(api)
    yield from api.carry(P0 + up * 0.075, R0, seconds=1.6)
    yield from api.carry(P0 + up * 0.075 + np.array([0, 0, 0.03]), R0, seconds=0.6)
    tip, h, lean = tip_state()
    diag["lifted"] = f"tip_h={h * 1000:.1f}mm lean={lean:.1f} F={np.round(api.contact_forces('pen'), 2).tolist()}"

    # ---- straighten the pen and carry it above the stroke start
    T0 = home + np.array([0.0, -0.060, 0.0])
    u = -f_dir.copy(); u[2] = 0; u /= np.linalg.norm(u)
    STROKE = 0.050

    def palm_for_tip(tip_target, vertical=True):
        Pm, Rm = api.measured_palm()
        p, R, up, _ = pen_axes(api)
        tip = p - up * E
        Rc = rot_between(up, np.array([0, 0, 1.0])) if vertical else np.eye(3)
        Rn = Rc @ Rm
        t_pal = Rm.T @ (tip - Pm)
        return tip_target - Rn @ t_pal, Rn

    Pn, Rn = palm_for_tip(np.array([T0[0], T0[1], TABLE + 0.10]))
    yield from api.carry(Pn, Rn, seconds=2.0)
    Pn, Rn = palm_for_tip(np.array([T0[0], T0[1], TABLE + 0.012]))
    yield from api.carry(Pn, Rn, seconds=1.5)
    for _ in range(2):  # correct for arm tracking error
        tip, h, lean = tip_state()
        Pc, Rcmd = api.palm_pose()
        err = np.array([T0[0], T0[1], TABLE + 0.010]) - tip
        yield from api.carry(Pc + err, Rcmd, seconds=0.4)
    tip, h, lean = tip_state()
    diag["above_start"] = f"tip={np.round(tip, 4).tolist()} h={h * 1000:.1f}mm lean={lean:.1f}"

    # ---- closed-loop descent and stroke
    Pc, Rcmd = api.palm_pose()
    Pc = Pc.copy()
    ref = None
    touched = False
    moved = 0.0
    v = 0.012 * 0.02      # 12 mm/s
    D_STAR = 0.00025
    trace = []
    air = 0
    k = 0
    contact_steps = 0
    t_touch = None
    h_bias = 0.0
    while k < 700:
        tip, h, lean = tip_state()
        Pm, Rm = api.measured_palm()
        tp = Rm.T @ (tip - Pm)
        if not touched:
            if ref is None or h > 0.002:
                ref = tp.copy()
            Dz = (Rm @ (tp - ref))[2]
            if h > 0.00025 and Dz < D_STAR:
                Pc[2] -= np.clip(0.1 * h, 0.00004, 0.0003)
            else:
                touched = True
                t_touch = k
                h_bias = max(0.0, h)
                diag["touchdown"] = f"k={k} h={h * 1000:.2f}mm Dz={Dz * 1000:.2f}mm lean={lean:.1f}"
        else:
            D = Rm @ (tp - ref)
            hc = h - h_bias
            if hc > 0.0005:
                air += 1
                Pc[2] -= np.clip(0.1 * hc, 0.00003, 0.00015)
                if air > 4 and D[2] > D_STAR:
                    ref = tp.copy()       # pen slipped in the grip: re-reference
            else:
                air = 0
                contact_steps += 1
                Pc[2] += np.clip(0.15 * (D[2] - D_STAR), -0.00005, 0.0001)
            if k - t_touch > 15 and moved < STROKE:
                ramp = min(1.0, (k - t_touch - 15) / 15)
                step = v * ramp
                Pc[:2] += u[:2] * step
                moved += step
            if moved >= STROKE:
                break
            if lean > 40:
                diag["abort"] = f"lean {lean:.1f} at k={k}"
                break
            if k % 10 == 0:
                trace.append((k, round(float(hc) * 1000, 2), round(float(D[2]) * 1000, 2), round(float(lean), 1),
                              round(moved * 1000, 1), round(float(Pc[2] - Pm[2]) * 1000, 2)))
        yield from api.carry(Pc, Rcmd, seconds=0.02)
        k += 1
    diag["stroke"] = (f"touch_k={t_touch} steps={k} contact_steps={contact_steps} moved={moved * 1000:.0f}mm "
                      f"F={np.round(api.contact_forces('pen'), 2).tolist()}")
    diag["trace(k,h,Dz,lean,moved,zcmd-zmeas)"] = trace

    # ---- lift and hold
    Pc, Rcmd = api.palm_pose()
    yield from api.carry(Pc + np.array([0, 0, 0.06]), Rcmd, seconds=1.2)
    yield from api.hold(1.0)
    tip, h, lean = tip_state()
    diag["end"] = f"tip_h={h * 1000:.1f}mm lean={lean:.1f} F={np.round(api.contact_forces('pen'), 2).tolist()}"
