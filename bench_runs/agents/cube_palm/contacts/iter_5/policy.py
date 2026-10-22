import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
OBJ = "cube"
F3 = ["thumb", "index", "middle"]
FI = {"thumb": 0, "index": 1, "middle": 2}
C_GRASP = np.array([0.086, 0.025, 0.118])  # cube centre in palm frame at the first pinch
PAD_UP = -0.006                             # pads sit this far toward the palm from the cube centre when grasping
TILT = 0.3                                  # palm-up tilt (fingers raised): gravity pulls the cube toward the wrist
PALM_X = 0.0225                             # palm face height (palm-frame x) over z 0.06..0.10 (palm mesh bbox)
MAX_RES = 0.0025                            # pad IK residual allowed along the transfer path [m]

S = {}  # compact run summary, emitted at the end (the bench keeps only the last 60 log lines)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def Rz(a):
    return Rotation.from_rotvec([0, 0, a]).as_matrix()


def palm_up_val(api):
    return float(api.measured_palm()[1][2, 0])


def face_axis(Ro):
    """Object-frame unit normal of the cube face whose normal is closest to palm +z."""
    j = int(np.argmax(np.abs(Ro[2, :])))
    n = np.zeros(3)
    n[j] = np.sign(Ro[2, j])
    return n


def pinch_points(api, h, x_off=PAD_UP, dy=0.011):
    """Contact points/normals (object frame) for thumb on the -z face, index+middle on the +z face."""
    p, Ro = api.object_in_palm(OBJ)
    n = face_axis(Ro)
    pts = {"thumb": -h * n + Ro.T @ np.array([x_off, 0, 0]),
           "index": h * n + Ro.T @ np.array([x_off, dy, 0]),
           "middle": h * n + Ro.T @ np.array([x_off, -dy, 0])}
    nrm = {"thumb": -n, "index": n, "middle": n}
    return pts, nrm


def st(api):
    p, _ = api.object_in_palm(OBJ)
    f = api.contact_forces(OBJ)
    return np.round(p * 1000).astype(int).tolist(), np.round(f, 2).tolist()


def note_state(api, tag):
    p, f = st(api)
    S[tag] = [round(api.t, 2), p, f]
    api.note(f"{tag}: cube_palm={p}mm forces={f} up={palm_up_val(api):.2f}")


def on_palm_ok(api, z0):
    p, _ = api.object_in_palm(OBJ)
    pw, _ = api.object_pose(OBJ)
    f = api.contact_forces(OBJ)
    return (palm_up_val(api) > 0.72 and pw[2] - z0 > 0.105 and f[5] > 0.03 and f[:5].sum() < 0.4
            and p[0] < 0.058 and abs(p[1]) < 0.042)


# ---------------------------------------------------------------- reach probing (fingertip IK queries, no motion)
def path_res(api, base, d, fingers=F3, n=4):
    """Worst pad IK residual [m] along a straight rigid pad shift by d (palm frame)."""
    worst = 0.0
    for s in np.linspace(1.0 / n, 1.0, n):
        for f in fingers:
            worst = max(worst, api.reach(f, base[FI[f]] + s * d))
    return worst


def plan_transfer(api, h, p_up):
    """Pick the deepest (toward the wrist) pinch-carried cube position whose pad path stays reachable."""
    base = api.tip_targets()
    dx = (PALM_X + h + 0.004) - p_up[0]          # stop 4 mm above the expected palm contact
    # map for the notes: worst residual (mm) over a dx/dz grid
    grid = []
    for ddz in (0, -0.01, -0.02, -0.03):
        row = []
        for ddx in (0, -0.01, -0.02, -0.03, -0.04):
            row.append(round(1000 * max(api.reach(f, base[FI[f]] + np.array([ddx, 0, ddz])) for f in F3), 1))
        grid.append(row)
    S["reach_grid_dz0..-30_dx0..-40"] = grid
    best = None
    dx_full = dx
    for dz, frac in [(z, fr) for z in (-0.030, -0.026, -0.022, -0.018, -0.014) for fr in (1.0, 0.75, 0.5)]:
        dx = frac * dx_full
        for kind in ("diag", "x_then_z", "z_then_x"):
            if kind == "diag":
                r = path_res(api, base, np.array([dx, 0, dz]))
            elif kind == "x_then_z":
                r = max(path_res(api, base, np.array([dx, 0, 0])),
                        path_res(api, base + np.array([dx, 0, 0]), np.array([0, 0, dz])))
            else:
                r = max(path_res(api, base, np.array([0, 0, dz])),
                        path_res(api, base + np.array([0, 0, dz]), np.array([dx, 0, 0])))
            if r < MAX_RES:
                best = (dz, kind, r, dx)
                break
        if best:
            break
    S["plan"] = None if best is None else [round(dx * 1000), round(best[0] * 1000), best[1], round(best[2] * 1000, 1)]
    return dx, best


def run(api):
    try:
        yield from main(api)
    finally:
        keys = list(S.keys())
        for i in range(0, len(keys), 4):
            api.note("SUMMARY " + "; ".join(f"{k}={S[k]}" for k in keys[i:i + 4]))


def square_yaw(api, R0):
    k = int(np.argmax(np.abs(R0[2, :])))
    a = R0[:, (k + 1) % 3]
    cy = np.arctan2(a[1], a[0])
    psi = min((cy + m * np.pi / 2 for m in range(-4, 5)), key=lambda v: abs(wrap(v - api.DEFAULT_YAW)))
    return api.DEFAULT_YAW + wrap(psi - api.DEFAULT_YAW)


def pick(api, h, tag):
    """Palm-down pinch of the cube resting on the table: thumb on the -z face, index+middle on the +z face."""
    p, R0 = api.object_pose(OBJ)
    psi = square_yaw(api, R0)
    Rd = api.hand_down(psi)
    grasp_palm = api.palm_for(p, C_GRASP, Rd)
    hover = grasp_palm + np.array([0, 0, 0.05])
    yield from api.carry(hover + np.array([0, 0, 0.05]), Rd, seconds=1.5)
    yield from api.carry(hover, Rd, seconds=0.8)
    yield from api.hold(0.3)
    pcur, Ro = api.object_in_palm(OBJ)
    n = face_axis(Ro)
    want = {"thumb": C_GRASP + np.array([PAD_UP, 0, -h]),
            "index": C_GRASP + np.array([PAD_UP, 0.011, h]),
            "middle": C_GRASP + np.array([PAD_UP, -0.011, h])}
    pts = {f: Ro.T @ (v - pcur) for f, v in want.items()}
    nrm = {"thumb": -n, "index": n, "middle": n}
    yield from api.approach(OBJ, pts, standoff=0.006, seconds=1.0, normals=nrm)
    yield from api.carry(grasp_palm, Rd, seconds=1.2)
    yield from api.hold(0.3)
    pts, nrm = pinch_points(api, h)
    r = yield from api.grasp(OBJ, pts, squeeze=0.004, seconds=0.8, normals=nrm)
    S[tag] = r["forces"]
    yield from api.hold(0.3)
    return psi, Rd, grasp_palm


def place_down(api, p0, h, Rd):
    """Palm-down hand holding the cube: set it on the table over the start and let go."""
    pc, _ = api.object_in_palm(OBJ)
    above = api.palm_for(np.array([p0[0], p0[1], TABLE + h + 0.04]), pc, Rd)
    yield from api.carry(above, Rd, seconds=1.5)
    down = api.palm_for(np.array([p0[0], p0[1], TABLE + h + 0.003]), pc, Rd)
    yield from api.carry(down, Rd, seconds=1.2)
    yield from api.hold(0.3)
    yield from api.release(F3, OBJ, clear=0.015, seconds=0.5)
    yield from api.carry(down + np.array([0, 0, 0.12]), Rd, seconds=1.2)


def fallen(api, p0):
    pw, _ = api.object_pose(OBJ)
    return pw[2] < p0[2] + 0.05


def palm_cycle(api, p0, h, z0, good0, tag):
    """Pick the cube from the table, turn palm up, set it on the palm, carry, regrasp, turn back and place.
    Returns (good steps on the open palm, placed: True / False (still in hand) / None (dropped))."""
    psi, Rd, grasp_palm = yield from pick(api, h, f"grasp{tag}")
    f = np.array(api.contact_forces(OBJ))
    if not (f[0] > 0.3 and f[1] + f[2] > 0.3):
        yield from api.release(F3, OBJ, clear=0.015, seconds=0.4)
        yield from api.carry(grasp_palm + np.array([0, 0, 0.08]), Rd, seconds=0.8)
        return 0, None
    # ---- lift and turn palm up (roll about the finger axis, palm origin fixed)
    yield from api.carry(grasp_palm + np.array([0, 0, 0.16]), Rd, seconds=1.5)
    yield from api.hold(0.3)
    Rm = Rd @ Rz(np.pi / 2)
    Ru = api.hand_up(psi, TILT)
    yield from api.carry(None, Rm, seconds=1.2)
    yield from api.carry(None, Ru, seconds=1.2)
    yield from api.hold(0.4)
    note_state(api, f"palm_up{tag}")
    p_up, _ = api.object_in_palm(OBJ)
    if fallen(api, p0):
        return 0, None

    # ---- transfer: carry the pinched cube toward the wrist to just above the palm, then seek contact gently
    dx, plan = plan_transfer(api, h, p_up)
    on_palm = False
    if plan is not None:
        dz, kind, _, dx = plan
        dx = min(0.0, max(dx, 0.060 - p_up[0]))  # stop above the support (it is met at cube x ~50 mm)
        segs = {"diag": [np.array([dx, 0, dz])], "x_then_z": [np.array([dx, 0, 0]), np.array([0, 0, dz])],
                "z_then_x": [np.array([0, 0, dz]), np.array([dx, 0, 0])]}[kind]
        for d in segs:
            yield from api.translate(OBJ, d, F3, seconds=max(0.6, np.linalg.norm(d) / 0.03))
        yield from api.hold(0.2)
        note_state(api, f"transferred{tag}")
        for i in range(10):
            if api.contact_forces(OBJ)[5] > 0.03:
                break
            pa, _ = api.object_in_palm(OBJ)
            r = yield from api.translate(OBJ, np.array([-0.002, 0, 0]), F3, seconds=0.25)
            pb, _ = api.object_in_palm(OBJ)
            if max(r["ik_residual_mm"].values()) > 4 or pa[0] - pb[0] < 0.0006:
                break
        note_state(api, f"seek{tag}")
        # loosen gently: small radial back-offs until the fingers carry (almost) nothing
        for i in range(6):
            yield from api.squeeze(F3, OBJ, -0.002, seconds=0.3)
            yield from api.hold(0.1)
            if sum(api.contact_forces(OBJ)[:5]) < 0.2:
                break
        f = api.contact_forces(OBJ)
        if sum(f[:5]) >= 0.2:
            busy = [g for g in F3 if f[FI[g]] > 0.06]
            if busy:
                yield from api.translate(OBJ, np.array([0.006, 0, 0]), busy, seconds=0.3)
        yield from api.hold(0.3)
        note_state(api, f"loosened{tag}")
        f = api.contact_forces(OBJ)
        on_palm = f[5] > 0.02 and sum(f[:5]) < 0.4
    if fallen(api, p0):
        return 0, None
    pushed = False
    if not on_palm:
        # fallback: open the pinch, park the thumb high, push the cube toward the wrist with index+middle
        yield from api.release(F3, OBJ, clear=0.012, seconds=0.5)
        yield from api.release(["thumb"], OBJ, clear=0.01, lift=(0.03, 0.02, 0.0), seconds=0.6)
        yield from api.hold(0.3)
        pts, nrm = pinch_points(api, h, x_off=0.010)
        sel = ("index", "middle")
        yield from api.approach(OBJ, {g: pts[g] for g in sel}, standoff=0.002, seconds=0.5,
                                normals={g: nrm[g] for g in sel})
        for i in range(6):
            f = api.contact_forces(OBJ)
            pc, _ = api.object_in_palm(OBJ)
            if f[5] > 0.03 and pc[2] < 0.105:
                break
            r = yield from api.translate(OBJ, np.array([0, 0, -0.008]), list(sel), seconds=0.4)
            if max(r["ik_residual_mm"].values()) > 6:
                break
        yield from api.release(list(sel), OBJ, clear=0.012, lift=(0.02, 0, 0), seconds=0.5)
        yield from api.hold(0.3)
        note_state(api, f"pushed{tag}")
        pushed = True
        if fallen(api, p0):
            return 0, None

    # ---- carry on the open palm, counting our own estimate of the success rule
    good = 0
    for i in range(150):
        yield from api.hold(0.02)
        good += int(on_palm_ok(api, z0))
        if good0 + good >= 70 or (i > 60 and good == 0):
            break
    S[f"carry{tag}"] = [good, i + 1]
    if fallen(api, p0):
        return good, None

    # ---- regrasp on the palm (pad height chosen by reach queries), restore the pick pinch, turn back, place
    p, Ro = api.object_in_palm(OBJ)
    best = None
    for xo in (-0.004, 0.0, 0.004, 0.008, 0.012, 0.016):
        pts, nrm = pinch_points(api, h, x_off=xo)
        res = max(api.reach(g, p + Ro @ pts[g] + Ro @ nrm[g] * (api.pad_offset - 0.004)) for g in F3)
        if best is None or res < best[1] - 1e-4:
            best = (xo, res)
        if res < 0.002:
            break
    S[f"regrasp_xoff{tag}"] = [round(best[0] * 1000), round(best[1] * 1000, 1)]
    pts, nrm = pinch_points(api, h, x_off=best[0])
    if pushed:
        yield from api.approach(OBJ, pts, standoff=0.008, seconds=0.6, normals=nrm)
    r = yield from api.grasp(OBJ, pts, squeeze=0.004, seconds=0.6, normals=nrm)
    S[f"regrasp{tag}"] = r["forces"]
    yield from api.hold(0.2)
    f = np.array(api.contact_forces(OBJ))
    if not (f[0] > 0.3 and f[1] + f[2] > 0.3):
        return good, False
    pc, _ = api.object_in_palm(OBJ)
    d = p_up - pc
    yield from api.translate(OBJ, np.array([0.5 * d[0], 0, 0]), F3, seconds=0.5)
    pc, _ = api.object_in_palm(OBJ)
    yield from api.translate(OBJ, p_up - pc, F3, seconds=0.8)
    yield from api.hold(0.2)
    # re-seat the pads at the pick-pinch points (the regrasp points sat high on the faces)
    pts, nrm = pinch_points(api, h)
    yield from api.grasp(OBJ, pts, squeeze=0.004, seconds=0.6, normals=nrm)
    yield from api.hold(0.3)
    note_state(api, f"regrasped{tag}")
    f = np.array(api.contact_forces(OBJ))
    if not (f[0] > 0.3 and f[1] + f[2] > 0.3):
        return good, False
    yield from api.carry(None, Rm, seconds=1.5)
    yield from api.carry(None, Rd, seconds=1.5)
    yield from api.hold(0.3)
    note_state(api, f"palm_down{tag}")
    f = np.array(api.contact_forces(OBJ))
    pw, _ = api.object_pose(OBJ)
    if f[0] > 0.1 and pw[2] > TABLE + h + 0.03:
        yield from place_down(api, p0, h, Rd)
        return good, True
    return good, None


def resting(api, p0):
    pw, Rw = api.object_pose(OBJ)
    return abs(pw[2] - p0[2]) < 0.006 and np.abs(Rw[2]).max() > 0.99


def reset_hand(api):
    Pp, Rp = api.palm_pose()
    yield from api.carry(Pp + np.array([0, 0, 0.10]), None, seconds=0.8)
    yield from api.open(seconds=0.6)


def main(api):
    p0, R0 = api.object_pose(OBJ)
    h = float(p0[2] - TABLE)                      # half edge from the resting height
    z0 = p0[2]
    S["start"] = [np.round(p0, 4).tolist(), round(h, 4)]
    good, placed = 0, None
    for attempt in range(2):
        if attempt > 0:
            if good >= 55 or api.t > 26 or not resting(api, p0):
                break
            S["retry_t"] = round(api.t, 1)
            yield from reset_hand(api)
        try:
            g, placed = yield from palm_cycle(api, p0, h, z0, good, str(attempt))
        except ValueError as e:
            S[f"cycle_err{attempt}"] = str(e)[:80]
            g, placed = 0, None
        good += g
        if placed is False:
            break
    S["good_total"] = good
    S["placed"] = placed
    if placed is False:
        # still on the palm: pour it off low over the start (re-picked below if it lands far away)
        yield from api.release(F3, OBJ, clear=0.012, seconds=0.4)
        try:
            pc, _ = api.object_in_palm(OBJ)
            Pp, Rp = api.palm_pose()
            tgt = api.palm_for(np.array([p0[0], p0[1], TABLE + 0.10]), pc + np.array([0, -0.05, 0]), Rp)
            yield from api.carry(tgt, Rp, seconds=1.5)
            yield from api.carry(None, Rp @ Rz(-np.pi / 2), seconds=1.5)
            yield from api.hold(0.8)
            Pp, Rp = api.palm_pose()
            yield from api.carry(Pp + np.array([0, 0, 0.12]), Rp, seconds=1.0)
        except ValueError as e:
            S["pour_err"] = str(e)[:80]
    # ---- recovery: the cube lies on the table away from the start -> plain pick and place
    for attempt in range(2):
        yield from api.hold(0.5)
        pw, _ = api.object_pose(OBJ)
        off = float(np.linalg.norm(pw[:2] - p0[:2]))
        if (off < 0.03 and resting(api, p0)) or api.t > 45 or not resting(api, p0):
            break
        S[f"recover{attempt}"] = [round(api.t, 1), round(off * 1000)]
        try:
            yield from reset_hand(api)
            _, Rd2, gp = yield from pick(api, h, f"repick{attempt}")
            f = np.array(api.contact_forces(OBJ))
            if f[0] > 0.3 and f[1] + f[2] > 0.3:
                yield from api.carry(gp + np.array([0, 0, 0.05]), Rd2, seconds=0.8)
                yield from place_down(api, p0, h, Rd2)
            else:
                yield from api.release(F3, OBJ, clear=0.015, seconds=0.4)
                yield from api.carry(gp + np.array([0, 0, 0.10]), Rd2, seconds=0.8)
        except ValueError as e:
            S[f"recover_err{attempt}"] = str(e)[:80]
            break
    yield from api.hold(1.0)
    pw, _ = api.object_pose(OBJ)
    S["end"] = [np.round(pw, 3).tolist(), round(float(np.linalg.norm(pw[:2] - p0[:2])) * 1000, 1)]
