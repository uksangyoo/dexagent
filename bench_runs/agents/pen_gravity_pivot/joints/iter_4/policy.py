"""pen_gravity_pivot, joints level (v3: tripod grasp, table-assisted pivot).

Plan: (1) calibrate a product-of-exponentials FK model of each finger from
single-joint sweeps in the air (api.tips() on measured q), giving my own
fingertip IK; (2) thumb-index (or thumb-middle) pinch near one end, with a
third finger pressing on top of the short end as a lever so the pen stays
level while lifted; (3) set the pen's far end on the table, lift the presser
(the pinch is a free hinge: no torsional friction), then raise the hand
straight up so the pen turns in the pinch about the table contact until it
hangs vertical, without any free swing.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

TABLE = 0.75
R_PEN = 0.00375
PAD = 0.008          # pad centre to pad surface (guess)
HALF = 0.070         # nominal half length incl. caps
NAMES = ("thumb", "index", "middle", "ring", "little")


def _rot(w, th):
    x, y, z = w
    c, s = np.cos(th), np.sin(th)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class Finger:
    def __init__(self, W, C, p0, qref, order=(0, 1, 2, 3)):
        self.W = np.asarray(W, float)
        self.C = np.asarray(C, float)
        self.p0 = np.asarray(p0, float)
        self.qref = np.asarray(qref, float)
        self.order = tuple(order)

    def chain(self, q):
        TR, Tt, last = np.eye(3), np.zeros(3), None
        for idx, k in enumerate(self.order):
            w = self.W[k] / (np.linalg.norm(self.W[k]) + 1e-12)
            c = self.C[k]
            if idx == 3:
                last = (TR @ w, TR @ c + Tt)
            R = _rot(w, q[k] - self.qref[k])
            Tt = TR @ (c - R @ c) + Tt
            TR = TR @ R
        return TR @ self.p0 + Tt, last

    def fk(self, q):
        return self.chain(q)[0]

    def paddir(self, q):
        tip, (w, c) = self.chain(q)
        v = np.cross(w, tip - c)
        return v / (np.linalg.norm(v) + 1e-12)


def _circle(p0, pp, pm, dth):
    a, b = pm - p0, pp - p0
    n = np.cross(b, a)
    nn = np.linalg.norm(n)
    if nn < 1e-9 or np.linalg.norm(a) < 5e-4 or np.linalg.norm(b) < 5e-4:
        return None
    n = n / nn
    axb = np.cross(a, b)
    c = p0 + np.cross(np.dot(a, a) * b - np.dot(b, b) * a, axb) / (2 * np.dot(axb, axb))
    best = None
    for w in (n, -n):
        e = np.linalg.norm(c + _rot(w, dth) @ (p0 - c) - pp)
        if best is None or e < best[0]:
            best = (e, w)
    return best[1], c


def fit_finger(Q, P, qref, p0, sweeps):
    """Q (n,4) measured q, P (n,3) tips; sweeps: per joint (i_plus, i_minus) sample indices."""
    W0, C0 = np.zeros((4, 3)), np.zeros((4, 3))
    for k in range(4):
        ip, im = sweeps[k]
        r = _circle(p0, P[ip], P[im], Q[ip][k] - qref[k])
        if r is None:
            W0[k], C0[k] = np.array([0, 1.0, 0]), p0 + np.array([0, 0, -0.03])
        else:
            W0[k], C0[k] = r
    best = None
    orders = [(0, 1, 2, 3)]
    import itertools
    for attempt in range(2):
        for order in orders:
            x0 = np.r_[W0.ravel(), C0.ravel(), p0]

            def res(x, order=order):
                W = x[:12].reshape(4, 3)
                C = x[12:24].reshape(4, 3)
                f = Finger(W, C, x[24:27], qref, order)
                r = [f.fk(q) - p for q, p in zip(Q, P)]
                r.append(0.01 * (np.linalg.norm(W, axis=1) - 1))
                r.append(1e-3 * (C - C0).ravel())
                return np.concatenate([np.ravel(v) for v in r])
            try:
                sol = least_squares(res, x0, max_nfev=400 if attempt == 0 else 120)
            except Exception:
                continue
            x = sol.x
            f = Finger(x[:12].reshape(4, 3), x[12:24].reshape(4, 3), x[24:27], qref, order)
            rms = float(np.sqrt(np.mean([np.sum((f.fk(q) - p) ** 2) for q, p in zip(Q, P)])))
            if best is None or rms < best[1]:
                best = (f, rms)
        if best is not None and best[1] < 2e-5:
            break
        orders = [o for o in itertools.permutations(range(4)) if o != (0, 1, 2, 3)]
    return best


def ik(f, target, lim4, seeds, pad=None, wpad=0.012, reg=0.002):
    best = None
    for s in seeds:
        s = np.clip(np.asarray(s, float), lim4[:, 0] + 1e-4, lim4[:, 1] - 1e-4)

        def res(q, s=s):
            tip, (w, c) = f.chain(q)
            r = [tip - target]
            if pad is not None:
                v = np.cross(w, tip - c)
                v = v / (np.linalg.norm(v) + 1e-12)
                r.append(wpad * (v - pad))
            r.append(reg * (q - s))
            return np.concatenate(r)
        try:
            sol = least_squares(res, s, bounds=(lim4[:, 0], lim4[:, 1]), max_nfev=200)
        except Exception:
            continue
        err = float(np.linalg.norm(f.fk(sol.x) - target))
        score = err + (0.0 if pad is None else 0.004 * float(np.linalg.norm(f.paddir(sol.x) - pad)))
        if best is None or score < best[2]:
            best = (sol.x, err, score)
    return best[0], best[1]




def fmt(v, k=4):
    return " ".join(f"{x:.{k}f}" for x in np.ravel(v))


# Finger FK models (product of exponentials) fitted in iteration 1 from my own
# calibration sweep (measured q -> api.tips()); fit residual was ~0.
MODELS_JSON = [{"W": [1.0, 0.0, -0.0, -0.0, -0.0001, 1.0, 0.9239, -0.3825, -0.0, 0.9239, -0.3825, -0.0], "C": [0.0684, 0.018, 0.0298, 0.0279, 0.018, 0.0511, 0.0421, 0.0533, 0.0298, 0.055, 0.0835, 0.0364], "p0": [0.0683, 0.1157, 0.0511], "qref": [0.0001, -0.3926, 0.1994, 0.1997]}, {"W": [1.0, 0.0, 0.0, -0.0, 0.993, -0.1179, -0.0, 0.993, -0.1179, -0.0, 0.993, -0.1179], "C": [0.068, 0.028, 0.0837, 0.0143, 0.0309, 0.1076, 0.0269, 0.0345, 0.1383, 0.0469, 0.0377, 0.1649], "p0": [0.0679, 0.0396, 0.1817], "qref": [-0.1182, 0.3884, 0.2531, 0.2505]}, {"W": [1.0, 0.0, 0.0, -0.0, 1.0, 0.0, -0.0, 1.0, 0.0, -0.0, 1.0, 0.0], "C": [0.0449, 0.005, 0.0886, 0.0143, 0.005, 0.1128, 0.0269, 0.005, 0.1437, 0.0469, 0.005, 0.1705], "p0": [0.0679, 0.005, 0.1874], "qref": [0.0, 0.3884, 0.2531, 0.2505]}, {"W": [1.0, 0.0, 0.0, -0.0, 0.9983, 0.0586, -0.0, 0.9983, 0.0586, -0.0, 0.9983, 0.0586], "C": [-0.0291, -0.018, 0.0806, 0.0139, 0.0067, 0.1063, 0.0263, -0.0368, 0.1348, 0.0458, -0.0473, 0.1613], "p0": [0.0665, -0.0238, 0.18], "qref": [0.0587, 0.3804, 0.2443, 0.2505]}, {"W": [0.0, -0.0, 1.0, 0.9427, 0.3337, -0.0, -0.3318, 0.9372, 0.1074, -0.3318, 0.9372, 0.1074], "C": [0.013, -0.0181, 0.1596, 0.0029, -0.0475, 0.0671, 0.0234, -0.0476, 0.0937, 0.0317, -0.048, 0.1261], "p0": [0.048, -0.0422, 0.16], "qref": [0.3402, 0.1076, 0.2292, 0.2454]}]
MODELS = [Finger(np.reshape(m["W"], (4, 3)), np.reshape(m["C"], (4, 3)), m["p0"], m["qref"]) for m in MODELS_JSON]

BETA = np.radians(20.0)      # pen direction in the palm y-z plane, turned toward +z (away from the thumb)
X0 = 0.100                    # pen axis depth below the palm
CP = (0.030, 0.090)           # pinch point (palm y, z)
SM = -0.024                   # middle contact offset along the pen (toward the short end)
G_OFF = 0.040                 # grip point from the short end (nominal length)
HW = R_PEN + PAD
SQ_ROT = 0.0055               # thumb-index squeeze while the pen turns in the pinch
SQ_HINGE = 0.008              # thumb-index squeeze once the pen hangs near vertical


def seeds_for(f, extra=()):
    m = MODELS[f]
    if f == 0:
        return [m.qref, np.array([1.0, -0.6, 0.6, -1.0]), np.array([0.3, -1.3, 1.3, -1.2])] + list(extra)
    return [m.qref, np.array([-0.2, 0.5, 1.5, 0.3]), np.array([0.0, 1.2, 0.8, 0.4])] + list(extra)


def finger_lim(api, f):
    L = api.joint_limits[4 * f:4 * f + 4].copy()
    if f > 0:
        L[2:, 0] = np.maximum(L[2:, 0], -0.05)
    return L


def tripod(api, open_=0.0, sq=0.004, sq_mid=None, extra_seeds=None):
    """Thumb on one side, index (opposite the thumb) and middle (further along) on the other."""
    u = np.array([0, np.cos(BETA), np.sin(BETA)])
    n = np.array([0, -np.sin(BETA), np.cos(BETA)])
    C = np.array([X0, CP[0], CP[1]])
    sq_mid = sq if sq_mid is None else sq_mid
    tg = {0: (C - (HW - sq + open_) * n, n), 1: (C + (HW - sq + open_) * n, -n),
          2: (C + SM * u + (HW - sq_mid + open_) * n, -n)}
    out, errs = {}, {}
    for f, (t, pd) in tg.items():
        ex = [] if extra_seeds is None else [extra_seeds[f]]
        q, e = ik(MODELS[f], t, finger_lim(api, f), seeds_for(f, ex), pad=pd)
        out[f], errs[f] = q, e
    return out, errs, u, n, C


def run(api):
    try:
        yield from attempt(api)
    except Exception as e:  # keep the diagnostics
        import traceback
        api.note("EXC " + repr(e)[:300] + " | " + traceback.format_exc(limit=4).replace("\n", " / ")[-700:])
    yield from api.hold(0.1)


def attempt(api):
    pen_p, pen_R = api.object_pose("pen")
    a = pen_R[:, 2].copy()
    a[2] = 0.0
    a /= np.linalg.norm(a)
    psi = np.arctan2(-a[0], a[1])
    if abs((psi - 0.15 + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2:
        a = -a
        psi = np.arctan2(-a[0], a[1])
    rest_p, _ = api.object_pose("pen_rest")
    o = float((rest_p - pen_p) @ a)
    side = 1.0 if o > 0 else -1.0          # the rest sits toward +side*a; grip the other end
    A = pen_p - side * HALF * a
    G = A + side * G_OFF * a
    G[2] = pen_p[2]
    yaw = psi + BETA
    R = api.hand_down(yaw)
    api.note(f"GEOM pen {fmt(pen_p)} rest_off {o * 1000:.1f}mm psi {psi:.3f} yaw {yaw:.3f} side {side}")

    qg, eg, u, n, C = tripod(api)
    qo, eo, _, _, _ = tripod(api, open_=0.015, extra_seeds=qg)
    api.note("IK err mm grip " + str({NAMES[f]: round(e * 1000, 1) for f, e in eg.items()}) +
             " open " + str({NAMES[f]: round(e * 1000, 1) for f, e in eo.items()}))
    OPENq = api.OPEN.copy()
    q_pre, q_grip = OPENq.copy(), OPENq.copy()
    for f in (0, 1, 2):
        q_pre[4 * f:4 * f + 4] = qo[f]
        q_grip[4 * f:4 * f + 4] = qg[f]

    n_w = R @ n                              # pinch normal in the world (horizontal)
    above = G + np.array([0, 0, 0.08])
    yield from api.move(xyz=api.palm_for(above, C, R), R=R, q=q_pre, seconds=2.0)
    yield from api.move(xyz=api.palm_for(G, C, R), R=R, seconds=1.2)
    yield from api.hold(0.2)
    api.note("pre tips " + fmt(api.tips()[:3], 3) + " pen_in_palm " + fmt(api.object_in_palm("pen")[0], 3))
    yield from api.move(q=q_grip, seconds=0.7)
    yield from api.hold(0.3)
    api.note("grip forces " + fmt(api.contact_forces("pen"), 2) + " pen_in_palm " + fmt(api.object_in_palm("pen")[0], 3) + " tips " + fmt(api.tips()[:3], 3))

    def tilt():
        _, Rn = api.object_pose("pen")
        ax = Rn[:, 2] * np.sign(Rn[:, 2] @ (side * a) + 1e-9)
        return float(np.degrees(np.arcsin(np.clip(-ax[2], -1, 1))))  # + = long end down

    Gup = G + np.array([0, 0, 0.10])
    yield from api.move(xyz=api.palm_for(Gup, C, R), seconds=1.5)
    sqm = 0.004
    for k in range(9):
        yield from api.hold(0.1)
        if tilt() > 5.0 and sqm < 0.008:
            sqm += 0.001
            qq, _, _, _, _ = tripod(api, sq_mid=sqm, extra_seeds=qg)
            q_grip[8:12] = qq[2]
            yield from api.move(q=q_grip, seconds=0.1)
    pp, _ = api.object_pose("pen")
    api.note(f"lifted tilt {tilt():.1f} lift {1000 * (pp[2] - pen_p[2]):.0f}mm sqm {sqm * 1000:.0f} forces " + fmt(api.contact_forces("pen"), 2))

    # carry clear of the rest, then down close to the table
    Glow = G - 0.06 * n_w
    Glow[2] = TABLE + 0.040
    yield from api.move(xyz=api.palm_for(Glow + np.array([0, 0, 0.04]), C, R), seconds=1.5)
    yield from api.move(xyz=api.palm_for(Glow, C, R), seconds=0.8)
    yield from api.hold(0.2)
    api.note(f"low tilt {tilt():.1f} forces " + fmt(api.contact_forces("pen"), 2))
    # release the middle: the thumb-index pinch becomes a free hinge, the far end drops onto the table
    # squeeze the thumb-index pinch harder at the same time (it alone holds the pen from here on)
    qh, eh, _, _, _ = tripod(api, sq=SQ_ROT, extra_seeds=qg)
    qhh, _, _, _, _ = tripod(api, sq=SQ_HINGE, extra_seeds=qg)
    q_rel = q_grip.copy()
    q_rel[0:8] = np.r_[qh[0], qh[1]]
    q_rel[8:12] = qo[2]
    yield from api.move(q=q_rel, seconds=0.8)
    q_rel[8:12] = OPENq[8:12]
    yield from api.move(q=q_rel, seconds=0.5)
    yield from api.hold(0.3)
    api.note(f"released hinge_ik_mm {eh[0] * 1000:.1f},{eh[1] * 1000:.1f} tilt {tilt():.1f} forces " + fmt(api.contact_forces("pen"), 2))
    # raise straight up in small steps: the far end slides in on the table and the pen turns in
    # the pinch; once it is near vertical, squeeze the pinch hard and keep lifting
    def vert():
        _, Rn = api.object_pose("pen")
        return float(np.degrees(np.arccos(min(1.0, abs(Rn[2, 2])))))
    squeezed = False
    for k in range(1, 35):
        yield from api.move(xyz=api.palm_for(Glow + np.array([0, 0, 0.005 * k]), C, R), seconds=0.12)
        if k == 12:
            api.note(f"mid vert {vert():.1f} forces " + fmt(api.contact_forces("pen"), 2))
        if not squeezed and (vert() < 12.0 or k == 30):
            q_sq = q_rel.copy()
            q_sq[0:8] = np.r_[qhh[0], qhh[1]]
            yield from api.move(q=q_sq, seconds=0.3)
            squeezed = True
            api.note(f"squeeze at k {k} vert {vert():.1f} forces " + fmt(api.contact_forces("pen"), 2))
    yield from api.hold(0.6)
    pp, Rn = api.object_pose("pen")
    _, Rpal = api.object_in_palm("pen")
    api.note(f"final tilt_from_vert {np.degrees(np.arccos(abs(Rn[2, 2]))):.1f} pen_axis_palm {fmt(Rpal[:, 2], 3)} lift {1000 * (pp[2] - pen_p[2]):.0f} forces " + fmt(api.contact_forces("pen"), 2))
    yield from api.hold(0.4)
