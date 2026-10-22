"""card_pickup, synergies level.

Plan: fingers pointing straight down, palm facing +y (the book), thumb toward +x,
so the thumb-index pinch axis (palm z) is vertical.
 1. In the air: calibrate a linear model tips(s) (palm-frame pads vs synergy
    coefficients) by central differences, then solve/refine (measured feedback +
    Broyden updates) two postures: an open pinch (index pad 45 mm below thumb pad)
    and a closed pinch, with the other fingers kept above / behind the index.
 2. Drag: with the open posture the index is the lowest finger; press it on the
    tile centre and slide the tile toward -y until its near end overhangs the
    book's near edge (centre stays on the book).
 3. Pinch: index pad under the overhang, thumb pad above, close along the
    synergy line, lift.
"""
import numpy as np
from scipy.optimize import least_squares

R_FD = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])   # palm x=+y, y=+x, z=-z (world)
PC = np.array([0.080, 0.040, 0.118])                     # nominal pinch centre, palm frame
R_PINCH = 0.034      # pinch point distance from tile centre along its long axis
Y_CLEAR = 0.015      # pinch point this far beyond the book's near edge
SB = 2.5             # coefficient bound

LOG = {}


def _measure(api, s, secs=0.3, hold=0.2):
    yield from api.move(s=s, seconds=secs)
    yield from api.hold(hold)
    return api.tips().copy()


def _predict(J, s0, t0, s):
    return (t0.reshape(-1) + J @ (s - s0)).reshape(5, 3)


def _solve(J, s0, t0, resid, s_ref, lam=0.004):
    def f(s):
        return np.concatenate([resid(_predict(J, s0, t0, s)), lam * (s - s_ref)])
    x0 = np.clip(s0, -SB + 1e-3, SB - 1e-3)
    r = least_squares(f, x0, bounds=(-SB, SB))
    return r.x


def _frame(n):
    """Rotation (palm->world) putting palm vector n at world +z and the palm-x direction
    (orthogonalised) at world +y."""
    n = n / np.linalg.norm(n)
    b = np.array([1.0, 0, 0]) - n[0] * n
    b /= np.linalg.norm(b)
    ex = np.cross(b, n)
    return np.column_stack([ex, b, n]).T


def joint_resid(J, sA, tA, D_c=0.022, travel=0.014, s_ref=None):
    """Residual over x = [s_open, s_close]: thumb static, index travels straight toward the
    thumb pad, middle level with the index (co-support), ring/little kept up and back."""
    k = len(sA)
    b0 = np.array([1.0, 0, 0])

    def f(x):
        so, sc = x[:k], x[k:]
        to, tc = _predict(J, sA, tA, so), _predict(J, sA, tA, sc)
        d = tc[0] - tc[1]
        L = np.linalg.norm(d) + 1e-9
        n = d / L
        b = b0 - n[0] * n
        b /= np.linalg.norm(b)
        u = tc[1] - to[1]
        r = [[(L - D_c) * 1.0], (u - (u @ n) * n) * 1.0, [(u @ n - travel) * 1.0],
             (tc[0] - to[0]) * 0.5, (n - np.array([0, 0, -1.0])) * 0.03]
        for t in (to, tc):
            m = t[2] - t[1]
            r.append(np.array([m @ n, m @ b]) * 0.5)
            for i in (3, 4):
                q = t[i] - t[1]
                r.append(np.array([max(0.0, 0.012 - q @ n), max(0.0, q @ b + 0.005)]))
        if s_ref is not None:
            r.append(0.003 * (x - s_ref))
        return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in r])
    return f


def _refine_pair(api, J, so, sc, rounds=3):
    """Jointly solve open/close postures on the linear model, measure both, Broyden-update J."""
    k = len(so)
    s_k = np.zeros(k)
    t_k = None
    hist = []
    to = yield from _measure(api, so)
    s_k, t_k = so.copy(), to
    x = np.r_[so, sc]
    for _ in range(rounds):
        f = joint_resid(J, s_k, t_k, s_ref=x.copy())
        x = least_squares(f, np.clip(x, -SB + 1e-3, SB - 1e-3), bounds=(-SB, SB)).x
        for s_new in (x[:k], x[k:]):
            t_new = yield from _measure(api, s_new)
            ds = s_new - s_k
            if ds @ ds > 1e-8:
                J = J + np.outer((t_new - t_k).reshape(-1) - J @ ds, ds) / (ds @ ds)
            s_k, t_k = s_new, t_new
        hist.append(float(np.linalg.norm(f(x))))
    to = yield from _measure(api, x[:k])
    tc = yield from _measure(api, x[k:])
    return x[:k], x[k:], to, tc, J, hist


def _tile_geom(api):
    c, R = api.object_pose("tile")
    a = R[:, 1].copy()
    a[2] = 0
    a /= np.linalg.norm(a)
    if a[1] > 0:
        a = -a
    return c, R, a          # a: unit xy direction of the long axis toward the near (-y) end


def _tilt(R):
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1))))


def _move_until(api, xyz, seconds, cond):
    """Palm to xyz, stopping early when cond() is true. Returns True if stopped early."""
    for _ in api.move(xyz=xyz, seconds=seconds):
        yield
        if cond():
            return True
    return False


def run(api):
    try:
        yield from _run(api)
    finally:
        api.note("SUMMARY " + repr({k: _clean(v) for k, v in LOG.items()}))


def _clean(v):
    if isinstance(v, np.ndarray):
        return np.round(v, 4).tolist()
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def _run(api):
    k = api.k
    book_p, _ = api.object_pose("book")
    y_edge = book_p[1] - 0.110

    # ---- 1. turn fingers down, calibrate a linear tips(s) model in the air
    yield from api.move(xyz=np.array([0.52, -0.24, 1.02]), R=R_FD, s=np.zeros(k), seconds=2.0)
    yield from api.hold(0.3)
    s_c = np.zeros(k)
    J = np.zeros((15, k))
    d = 0.8
    for j in range(k):
        e = np.zeros(k); e[j] = d
        tp = yield from _measure(api, s_c + e)
        tm = yield from _measure(api, s_c - e)
        J[:, j] = (tp - tm).reshape(-1) / (2 * d)
    # seeds: iter-1 open / closed pinch postures
    so0 = np.array([0.37, -0.68, 0.52, 1.31, -0.30, 1.32])
    sc0 = np.array([0.81, -1.03, -0.11, 0.58, -0.38, 0.91])
    so, sc, to, tc, J, hist = yield from _refine_pair(api, J, so0, sc0)
    n = tc[0] - tc[1]
    Rp = _frame(n)
    LOG["so"], LOG["sc"], LOG["to"], LOG["tc"], LOG["fit_hist"] = so, sc, to, tc, hist
    LOG["n_palm"] = n / np.linalg.norm(n)
    D_o = float(np.linalg.norm(to[0] - to[1])); D_c = float(np.linalg.norm(tc[0] - tc[1]))
    LOG["D_o"], LOG["D_c"] = D_o, D_c
    LOG["index_path_world"] = Rp @ (tc[1] - to[1])
    LOG["thumb_path_world"] = Rp @ (tc[0] - to[0])
    LOG["others_rel_index_world_open"] = (to[2:] - to[1]) @ Rp.T
    lam_max = float(np.clip((D_o - 0.006) / max(D_o - D_c, 1e-3), 1.0, 3.0))
    s_close = so + lam_max * (sc - so)
    LOG["lam_max"] = lam_max
    yield from api.move(R=Rp, seconds=1.0)

    # ---- 2. drag the tile toward the near edge with index + middle
    c, R, a = _tile_geom(api)
    lat = np.array([-a[1], a[0], 0.0])
    if lat[0] < 0:
        lat = -lat
    top = c[2] + 0.0015
    cos_psi = -a[1]
    target_cy = y_edge - Y_CLEAR + R_PINCH * cos_psi
    LOG["tile0"], LOG["target_cy"] = c, target_cy
    press = c - 0.030 * a + 0.012 * lat
    press[2] = top + 0.035
    above = api.palm_for(press, to[1], Rp)
    yield from api.move(xyz=above + np.array([0, 0, 0.04]), s=so, seconds=1.5)
    yield from api.move(xyz=above, seconds=0.6)
    yield from api.hold(0.2)
    down = above - np.array([0, 0, 0.07])
    hit = yield from _move_until(api, down, 2.0,
                                 lambda: api.contact_forces("tile").sum() > 0.15 or api.contact_forces("book").sum() > 0.3)
    LOG["drag_contact"] = [hit, np.round(api.contact_forces("tile"), 2).tolist(), np.round(api.contact_forces("book"), 2).tolist()]
    P, _ = api.palm_pose()
    yield from api.move(xyz=P - np.array([0, 0, 0.002]), seconds=0.25)
    yield from api.hold(0.2)
    LOG["press_force"] = np.round(api.contact_forces("tile"), 2).tolist()
    for attempt in range(3):
        c, R, a = _tile_geom(api)
        dy = target_cy - c[1]
        if abs(dy) < 0.003:
            break
        P, _ = api.palm_pose()
        yield from api.move(xyz=P + np.array([0, dy, 0]), seconds=max(0.6, abs(dy) / 0.05))
        yield from api.hold(0.2)
        if attempt < 2:
            P, _ = api.palm_pose()
            yield from api.move(xyz=P - np.array([0, 0, 0.0015]), seconds=0.15)
    c, R, a = _tile_geom(api)
    LOG["after_drag"] = [c, _tilt(R), np.round(api.contact_forces("tile"), 2).tolist()]
    P, _ = api.palm_pose()
    yield from api.move(xyz=P + np.array([0, 0, 0.035]), seconds=0.5)
    yield from api.hold(0.3)

    # ---- 3. pinch the overhang: thumb pad above Q, index (+middle) hooked under
    c, R, a = _tile_geom(api)
    lat = np.array([-a[1], a[0], 0.0])
    if lat[0] < 0:
        lat = -lat
    Q = c + R_PINCH * a + 0.012 * lat
    LOG["tile_pre_pinch"], LOG["Q"], LOG["overhang_Q_mm"] = [c, _tilt(R)], Q, round(1000 * (y_edge - Q[1]), 1)
    T_star = Q + np.array([0, 0, 0.0015 + 0.008 + 0.003])
    lift6 = np.array([0, 0, 0.006])
    pre = api.palm_for(T_star + lift6 + np.array([0, -0.065, 0.045]), to[0], Rp)
    yield from api.move(xyz=pre, seconds=1.0)
    yield from api.move(xyz=pre - np.array([0, 0, 0.045]), seconds=0.8)
    yield from api.move(xyz=api.palm_for(T_star + lift6, to[0], Rp), seconds=1.2)
    yield from api.hold(0.3)
    c2, R2, _ = _tile_geom(api)
    LOG["tile_after_slide"] = [c2, _tilt(R2), np.round(api.contact_forces("tile"), 2).tolist(),
                               np.round(api.contact_forces("book"), 2).tolist()]
    # correct the palm from the measured thumb pad (and keep the index off the book face)
    tw = api.tips_world()
    err = tw[0] - (T_star + lift6)
    corr = -err
    corr[1] = min(corr[1], (y_edge - 0.011) - tw[1, 1])
    corr = np.clip(corr, -0.012, 0.012)
    LOG["pad_err_mm"] = [np.round(err * 1000, 1).tolist(), np.round((tw[1] - (Q - np.array([0, 0, D_o - 0.0125]))) * 1000, 1).tolist()]
    P, _ = api.palm_pose()
    yield from api.move(xyz=P + corr - lift6, seconds=0.5)
    yield from api.hold(0.2)
    # close: index rises toward the thumb along the synergy line
    yield from api.move(s=sc, seconds=0.6)
    yield from api.move(s=s_close, seconds=0.8)
    yield from api.hold(0.4)
    f = api.contact_forces("tile")
    c3, R3, _ = _tile_geom(api)
    LOG["pinch"] = [np.round(f, 2).tolist(), c3, _tilt(R3), np.round(api.contact_forces("book"), 2).tolist()]

    # ---- 4. lift
    P, _ = api.palm_pose()
    yield from api.move(xyz=P + np.array([0, -0.01, 0.16]), seconds=2.5)
    yield from api.hold(0.5)
    c, R, a = _tile_geom(api)
    LOG["final_tile"], LOG["final_tilt"] = c, _tilt(R)
    LOG["final_force"] = np.round(api.contact_forces("tile"), 2).tolist()
