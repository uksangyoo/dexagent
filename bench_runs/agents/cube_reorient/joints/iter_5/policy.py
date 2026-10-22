import json
import numpy as np
from scipy.optimize import least_squares

# Joint-level policy. Finger kinematics: a product-of-exponentials model per finger
# (4 revolute screw axes + pad point, palm frame, metres), fitted offline to
# (api.q(), api.tips()) pairs logged in iteration 1 (fit rms 0.006 mm).
# On top of it: my own position IK (least squares) and straight-line pad paths.
MODEL = {"0": {"qr": [0.0, -0.3, 0.2, 0.2], "P": [1.0, 4e-06, -3e-06, -0.105606, 0.017999, 0.0298, 1e-05, -4e-06, 1.0, 0.027898, 0.018001, -0.180899, 0.955336, -0.295521, -8e-06, 0.435629, -0.068244, 0.029797, 0.95534, -0.295508, 0.0, 0.079397, 0.076216, 0.036436, 0.059135, 0.118976, 0.051176]}, "1": {"qr": [0.0, 0.25, 0.25, 0.25], "P": [1.0, -2.6e-05, -6.1e-05, 0.524784, 0.027987, 0.083624, -1.2e-05, 1.0, -2.1e-05, 0.014289, 0.893536, 0.107782, 4e-06, 1.0, 2.2e-05, 0.022559, -0.893911, 0.140141, 1e-06, 1.0, 0.000123, 0.038575, 0.352949, 0.169513, 0.056945, 0.028001, 0.189192]}, "2": {"qr": [0.0, 0.25, 0.25, 0.25], "P": [1.0, 8e-06, 6e-06, 0.081191, 0.005001, 0.088649, -2.1e-05, 1.0, -2e-05, 0.014305, -0.236982, 0.112805, -1.1e-05, 1.0, -5e-06, 0.022566, -0.237673, 0.145163, -4e-06, 1.0, 0.000179, 0.038575, 0.155808, 0.1745, 0.056945, 0.005001, 0.194192]}, "3": {"qr": [0.0, 0.25, 0.25, 0.25], "P": [1.0, -4e-06, 3.2e-05, 0.108315, -0.018, 0.08065, -2.3e-05, 1.0, -3.2e-05, 0.013894, 0.254541, 0.104791, -8e-06, 1.0, -1.1e-05, 0.022161, 0.242065, 0.137159, -7e-06, 1.0, 0.000186, 0.03817, 0.760872, 0.166618, 0.056546, -0.017999, 0.186192]}, "4": {"qr": [0.2, 0.0, 0.25, 0.25], "P": [-1.4e-05, -2.3e-05, 1.0, 0.012999, -0.018054, 0.267736, 0.980066, 0.198673, -1.3e-05, 0.054856, -0.034512, 0.067099, -0.198675, 0.980065, 4e-06, -0.021729, 0.153269, 0.094301, -0.198662, 0.980068, 5.4e-05, -0.048773, 0.328292, 0.126682, 0.043741, -0.036765, 0.159878]}}
PAR = {int(i): (np.array(v["P"]), np.array(v["qr"])) for i, v in MODEL.items()}
X = np.array([1.0, 0, 0])


def rod(w, a):
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def fk(i, q):
    """Pad centre of finger i (palm frame, m) at its 4 joint angles q."""
    P, qr = PAR[i]
    p = P[24:27].copy()
    for j in (3, 2, 1, 0):
        w, c = P[6 * j:6 * j + 3], P[6 * j + 3:6 * j + 6]
        p = rod(w, q[j] - qr[j]) @ (p - c) + c
    return p


# Thumb skeleton: knuckle points on the fitted joint-2 and joint-3 axes (closest points between
# successive axes / the pad), carried through the same screw chain. Used to reject pushes whose
# thumb links would lie over or in the cube.
TJ = [np.array([0.0394, 0.0546, 0.0298]), np.array([0.0486, 0.0857, 0.0364])]


def thumb_skeleton(q):
    P, qr = PAR[0]
    out = []
    for k, p in enumerate(TJ + [P[24:27]]):
        p = p.copy()
        for j in range(k + 1, -1, -1):
            w, c = P[6 * j:6 * j + 3], P[6 * j + 3:6 * j + 6]
            p = rod(w, q[j] - qr[j]) @ (p - c) + c
        out.append(p)
    return out


def thumb_hits(q, c, R, h, margin=0.008):
    J2, J3, tip = thumb_skeleton(q)
    pts = [J2 + u * (J3 - J2) for u in np.linspace(0, 1, 6)]
    L = np.linalg.norm(tip - J3)
    pts += [J3 + u * (tip - J3) for u in np.linspace(0, 1, 8) if (1 - u) * L > 0.016]
    for p in pts:
        r = R.T @ (p - c)
        if np.all(np.abs(r) < h / 2 + margin):
            return True
    return False


class IK:
    def __init__(self, limits):
        self.lim = limits

    def solve(self, i, target, seed, tries=0, wpost=0.003):
        l = self.lim[4 * i:4 * i + 4]
        seed = np.clip(np.asarray(seed, float), l[:, 0] + 1e-6, l[:, 1] - 1e-6)
        rng = np.random.default_rng(len(target))
        best = None
        for k in range(tries + 1):
            s = seed if k == 0 else l[:, 0] + rng.uniform(.1, .9, 4) * (l[:, 1] - l[:, 0])
            f = least_squares(lambda q: np.r_[fk(i, q) - target, wpost * (q - seed)], s, bounds=(l[:, 0], l[:, 1]))
            e = float(np.linalg.norm(fk(i, f.x) - target))
            if best is None or e < best[1] - 1e-4:
                best = (f.x, e)
            if best[1] < 5e-4:
                break
        return best


def yaw_of(R):
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2)), float(np.abs(R[0]).max())


def run(api):
    log = []
    ik = IK(api.joint_limits)
    TH = 0                      # the thumb does all the pushing
    st = {"last": None, "rot": 0.0}

    def track():
        _, R = api.object_in_palm("cube")
        y, flat = yaw_of(R)
        if st["last"] is not None:
            # check.py's yaw decreases for a positive rotation about palm +x
            st["rot"] -= (y - st["last"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        st["last"] = y
        return st["rot"]

    def step_q(i, q, n=1):
        for _ in api.move(q={i: q}, seconds=0.02 * n):
            track()
            yield

    def line(i, a, b, speed=0.05, q0=None, stop=None):
        """Pad of finger i along a straight palm-frame line a->b (speed m/s), IK per step."""
        q = api.q_target()[4 * i:4 * i + 4] if q0 is None else q0
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / (speed * 0.02))))
        emax = 0.0
        for k in range(1, n + 1):
            q, e = ik.solve(i, a + (b - a) * k / n, q)
            if e > 0.002:
                q, e = ik.solve(i, a + (b - a) * k / n, q, tries=3)
            emax = max(emax, e)
            yield from step_q(i, q)
            if stop is not None and stop():
                break
        return q, emax

    def cube():
        p, R = api.object_in_palm("cube")
        return p, R

    def side_faces(R):
        F = []
        for j in range(3):
            if abs(R[0, j]) < 0.5:
                n = R[:, j].copy(); n[0] = 0; n /= np.linalg.norm(n)
                F += [n, -n]
        return F

    def orbit(a, b, c, rad, speed=0.15, q0=None):
        """Pad from a to b around the cube centre c (palm y-z plane) on the +y/+z side, radius >= rad."""
        va, vb = a[1:] - c[1:], b[1:] - c[1:]
        fa, fb = np.arctan2(va[1], va[0]), np.arctan2(vb[1], vb[0])
        ra, rb = max(np.linalg.norm(va), rad), max(np.linalg.norm(vb), rad)
        k = max(2, int(abs(fb - fa) / np.radians(15)) + 1)
        pts = [a]
        if np.linalg.norm(va) < rad:
            pts.append(np.r_[a[0], c[1:] + ra * np.array([np.cos(fa), np.sin(fa)])])
        for u in np.linspace(0, 1, k + 1)[1:-1]:
            f, r = fa + u * (fb - fa), ra + u * (rb - ra)
            pts.append(np.r_[a[0] + u * (b[0] - a[0]), c[1:] + r * np.array([np.cos(f), np.sin(f)])])
        if np.linalg.norm(vb) < rad:
            pts.append(np.r_[b[0], c[1:] + rb * np.array([np.cos(fb), np.sin(fb)])])
        pts.append(b)
        q = q0
        for p1, p2 in zip(pts[:-1], pts[1:]):
            q, _ = yield from line(TH, p1, p2, speed, q)
        return q

    yield from api.hold(0.2)
    # little finger spread out of the way (its base otherwise acts as a wall beside the cube)
    yield from api.move(q={4: [0.0, 0.6, 0.2, 0.2]}, seconds=0.4)
    track()
    p0, R0 = cube()
    h = 2 * (p0[0] - 0.0235) if p0[0] > 0.03 else 0.04   # edge from resting height (palm face ~23.5 mm)
    h = float(np.clip(h, 0.034, 0.046))
    log.append(f"start c={np.round(p0 * 1000, 1).tolist()} h={h * 1000:.1f}")
    GOAL, STOP = np.radians(90), np.radians(88)
    BOX_LO, BOX_HI = np.array([-0.024, 0.052]), np.array([0.024, 0.092])   # where the cube centre should stay (y, z)
    qth = api.q_target()[0:4]
    pad = fk(TH, api.q()[0:4])
    RAD = h / np.sqrt(2) + 0.017
    prev = None

    for stroke in range(20):
        if api.t > 50:
            break
        rot = track()
        rem = GOAL - rot
        if abs(rem) < np.radians(5):
            break
        d = 1.0 if rem > 0 else -1.0       # torque sense needed about palm +x
        c, R = cube()
        best = None
        for relax in (0, 1):
          if best is not None:
              break
          ylo, zlo, fmin = (-0.036, 0.036, 0.3) if not relax else (-0.046, 0.030, 0.1)
          for n in side_faces(R):
              if n[1] < fmin and n[2] < fmin:      # the thumb only works faces looking +y or toward the fingers
                  continue
              t = np.cross(X, n) * d             # this tangent gives the wanted torque sense
              for off, dx in ((0.4 * h, -0.006), (0.4 * h, 0.0), (0.25 * h, -0.006), (0.25 * h, 0.004), (0.1 * h, -0.006)):
                  cp = c + n * (h / 2) + t * off
                  cp[0] = c[0] + dx
                  kap = 0.4 if n[1] > 0.7 else 0.15   # drag the face along t too: more torque, and
                  pre = cp + n * (0.008 + 0.010)       # on the +y face it walks the cube toward the fingers
                  end = cp + n * (0.008 - 0.016) + kap * 0.026 * t
                  q1, e1 = ik.solve(TH, pre, qth, tries=4)
                  q2, e2 = ik.solve(TH, end, q1)
                  if max(e1, e2) > 0.002 or thumb_hits(q1, c, R, h):
                      continue
                  pred = c[1:] - 0.010 * n[1:] + 0.010 * kap * t[1:]   # a push drifts the cube along -n (+ drag)
                  if pred[0] < ylo or pred[1] < zlo or pred[1] > 0.106:
                      continue
                  out = np.maximum(BOX_LO - pred, 0) + np.maximum(pred - BOX_HI, 0)
                  score = -np.linalg.norm(out) / 0.002 - np.linalg.norm(pred - (BOX_LO + BOX_HI) / 2) / 0.01 \
                      - 2 * (e1 + e2) / 0.001 - 20 * np.linalg.norm(pre - pad) + off / h \
                      + 1.5 * max(0.0, -n[2])        # wrist-side faces push the cube toward +z, clear of the knuckle wall
                  if prev is not None and n @ prev[0] > 0.9 and prev[1] < np.radians(3):
                      score -= 8.0 * prev[2]          # that face stalled last time: try another
                  if best is None or score > best[0]:
                      best = (score, n, cp, pre, end, q1)
                  break
        if best is None:
            log.append(f"s{stroke} no feasible push rot={np.degrees(rot):.0f}")
            break
        _, n, cp, pre, end, q1 = best
        qth = yield from orbit(pad, pre, c, RAD, 0.15, qth)
        r0 = track(); cs, _ = cube()

        def stop():
            r = track()
            cc, _ = cube()
            return d * (r - GOAL) >= -(GOAL - STOP) or np.linalg.norm(cc[1:] - cs[1:]) > 0.012

        qth, emax = yield from line(TH, pre, end, 0.03 if abs(GOAL - r0) > np.radians(25) else 0.015, qth, stop)
        yield from api.hold(0.06)
        r1 = track(); ce, _ = cube()
        gain = d * (r1 - r0)
        stalls = (prev[2] + 1) if (prev is not None and n @ prev[0] > 0.9 and gain < np.radians(3)) else 1
        prev = (n.copy(), gain, stalls)
        back = fk(TH, qth) + n * 0.012
        qth, _ = yield from line(TH, fk(TH, qth), back, 0.08, qth)
        pad = back
        log.append(f"s{stroke} n={np.round(n[1:], 2).tolist()} rot {np.degrees(r0):.0f}->{np.degrees(r1):.0f}"
                   f" c {np.round(cs[1:] * 1000).tolist()}->{np.round(ce[1:] * 1000).tolist()} e={emax * 1000:.1f}"
                   f" f={np.round(api.contact_forces('cube'), 2).tolist()} t={api.t:.1f}")

    # release: thumb back round to the +y side, up, and out to its open posture
    c, _ = cube()
    side = np.r_[pad[0], c[1] + RAD + 0.01, c[2]]
    qth = yield from orbit(pad, side, c, RAD, 0.12, qth)
    up = side.copy(); up[0] += 0.015
    qth, _ = yield from line(TH, side, up, 0.08, qth)
    yield from api.move(q={TH: api.OPEN[0:4]}, seconds=0.8)
    yield from api.hold(0.3)
    track()
    c, R = cube()
    log.append(f"end rot={np.degrees(st['rot']):.1f} c={np.round(c * 1000, 1).tolist()} flat={yaw_of(R)[1]:.3f} t={api.t:.1f}")
    api.note(" | ".join(log))
