# cube_pinch at level synergies
#
# Plan: (1) in the air, learn how the synergy coefficients move the thumb and
# index pads (finite-difference Jacobian from measured api.tips(), refined by
# Broyden updates) and solve for an "open" and a "closed" thumb-index pinch
# along the palm's finger axis; (2) align palm yaw to the cube, descend with the
# pinch open around it, close; (3) lift 110 mm, turn the palm about the vertical
# axis through the cube by ~42 deg; (4) lower back to the start pose using the
# measured cube pose, open, retreat.
import numpy as np

PAD = 0.008                                  # pad centre sits ~8 mm inside the fingertip surface
M_TARGET = np.array([0.080, 0.040, 0.118])   # preferred pinch midpoint (palm frame)
CLEAR = 0.012                                # open clearance per side
SQUEEZE = 0.006                              # closing overshoot per side
LIFT = 0.11
TURN = np.radians(42)


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


class SynIK:
    """Online synergy -> pad-position model: measured finite differences + Broyden."""

    def __init__(self, api):
        self.api, self.k = api, api.k
        self.J = None
        self.s_last = self.t_last = None

    def measure(self, s, T=0.3, settle=0.2):
        s = np.clip(np.asarray(s, float), -3, 3)
        yield from self.api.move(s=s, seconds=T)
        yield from self.api.hold(settle)
        t = self.api.tips().copy()
        if self.J is not None and self.s_last is not None:
            ds = s - self.s_last
            if ds @ ds > 1e-8:
                dt = (t - self.t_last).ravel()
                self.J += np.outer(dt - self.J @ ds, ds) / (ds @ ds)
        self.s_last, self.t_last = s.copy(), t.copy()
        return t

    def probe(self, s0, step=1.0):
        s0 = np.asarray(s0, float)
        t0 = yield from self.measure(s0, 0.5, 0.3)
        J = np.zeros((15, self.k))
        for j in range(self.k):
            e = s0.copy(); e[j] += step
            tj = yield from self.measure(e)
            J[:, j] = (tj - t0).ravel() / step
        self.J = J
        self.s_last, self.t_last = None, None
        t0 = yield from self.measure(s0)
        return t0

    def residual(self, t, s, gap, s_ref, w_mid=0.3, w_reg=0.5):
        rel = t[1] - t[0]
        mid = 0.5 * (t[0] + t[1])
        g = np.array([0, 0, gap])
        r = np.r_[1000 * (rel - g), w_mid * 1000 * (mid - M_TARGET), w_reg * (s - s_ref)]
        Jt = self.J.reshape(5, 3, self.k)
        A = np.vstack([1000 * (Jt[1] - Jt[0]), w_mid * 1000 * 0.5 * (Jt[0] + Jt[1]), w_reg * np.eye(self.k)])
        return r, A

    def solve(self, gap, s_init, s_ref, iters=5):
        s = np.asarray(s_init, float)
        t = yield from self.measure(s)
        best = None
        for it in range(iters + 1):
            r, A = self.residual(t, s, gap, s_ref)
            relerr = np.linalg.norm(r[:3])
            cost = r @ r
            if best is None or cost < best[0]:
                best = (cost, s.copy(), t.copy(), relerr)
            if it == iters or relerr < 1.5:
                break
            ds = np.linalg.lstsq(A, -r, rcond=None)[0]
            n = np.linalg.norm(ds)
            if n > 1.5:
                ds *= 1.5 / n
            s = np.clip(s + ds, -3, 3)
            t = yield from self.measure(s)
        return best


def fmt(v):
    return np.round(np.asarray(v) * 1000, 1).tolist()


def run(api):
    keep = []
    _note = api.note

    def note(text):
        keep.append(text)
        _note(text)
    try:
        yield from _run(api, note)
    finally:
        _note("SUMMARY >>>")
        for k in keep:
            _note(k)


def _run(api, note):
    note("k=%d open_coeffs=%s" % (api.k, np.round(api.open_coeffs, 2).tolist()))
    note("describe:\n" + api.describe())
    p0, R0 = api.object_pose("cube")
    note(f"cube p0={fmt(p0)}")

    # ---- edge estimate: cube rests on table, centre at half edge above z=0.75
    h = float(np.clip(p0[2] - 0.75, 0.015, 0.025))
    g_contact = 2 * h + 2 * PAD
    g_open = g_contact + 2 * CLEAR
    g_close = g_contact - 2 * SQUEEZE

    # ---- learn synergy -> pads in the air
    ik = SynIK(api)
    yield from ik.probe(np.zeros(api.k))
    note("J (mm/sd) thumb:%s index:%s" % (np.round(ik.J[0:3] * 1000).astype(int).tolist(),
                                             np.round(ik.J[3:6] * 1000).astype(int).tolist()))
    cost, s_open, t_open, e_open = yield from ik.solve(g_open, np.zeros(api.k), np.zeros(api.k))
    note(f"s_open={np.round(s_open, 2).tolist()} relerr={e_open:.1f}mm gap={1000 * (t_open[1, 2] - t_open[0, 2]):.1f} "
             f"want={1000 * g_open:.1f} tips={fmt(t_open)}")
    cost, s_close, t_close, e_close = yield from ik.solve(g_close, s_open, s_open)
    note(f"s_close={np.round(s_close, 2).tolist()} relerr={e_close:.1f}mm gap={1000 * (t_close[1, 2] - t_close[0, 2]):.1f} "
             f"want={1000 * g_close:.1f} tips={fmt(t_close)}")
    m_open = 0.5 * (t_open[0] + t_open[1])
    m_close = 0.5 * (t_close[0] + t_close[1])

    # ---- align with the cube and grasp (retry once if it does not lift)
    for attempt in range(2):
        c, Rc = api.object_pose("cube")
        d = wrap(yaw_of(Rc) - api.DEFAULT_YAW, np.pi / 2)
        R = api.hand_down(api.DEFAULT_YAW + d)
        above = c + np.array([0, 0, 0.07]) - R @ m_open
        yield from api.move(above, R, s=s_open, seconds=2.0)
        c, _ = api.object_pose("cube")
        yield from api.move(c - R @ m_open, R, s=s_open, seconds=1.5)
        yield from api.hold(0.2)
        c, _ = api.object_pose("cube")
        P_grasp = c - R @ m_close
        yield from api.move(P_grasp, R, s=s_close, seconds=1.0)
        yield from api.hold(0.4)
        f = api.contact_forces("cube")
        pin, _ = api.object_in_palm("cube")
        note(f"grasp{attempt}: forces={np.round(f, 2).tolist()} cube_in_palm={fmt(pin)} tips={fmt(api.tips()[:2])}")
        # test lift
        yield from api.move(P_grasp + np.array([0, 0, 0.03]), R, seconds=0.8)
        yield from api.hold(0.2)
        c1, _ = api.object_pose("cube")
        note(f"test lift dz={1000 * (c1[2] - p0[2]):.1f}mm")
        if c1[2] - p0[2] > 0.02:
            break
        # release and retry
        P, _ = api.palm_pose()
        yield from api.move(P + R @ (m_close - m_open), R, s=s_open, seconds=0.8)
        yield from api.move(P + R @ (m_close - m_open) + np.array([0, 0, 0.06]), R, seconds=0.8)

    # ---- lift
    P, R = api.palm_pose()
    yield from api.move(P + np.array([0, 0, LIFT - 0.03]), R, seconds=1.5)
    yield from api.hold(0.3)
    c, Rc = api.object_pose("cube")
    note(f"lifted dz={1000 * (c[2] - p0[2]):.1f}mm forces={np.round(api.contact_forces('cube'), 2).tolist()}")

    # ---- turn about the vertical through the cube, toward the default yaw
    d = wrap(yaw_of(R0) - api.DEFAULT_YAW, np.pi / 2)
    sgn = -np.sign(d) if abs(d) > 1e-3 else 1.0
    for i in range(4):
        c, _ = api.object_pose("cube")
        P, R = api.palm_pose()
        Rr = Rz(sgn * TURN / 4)
        yield from api.move(c + Rr @ (P - c), Rr @ R, seconds=0.6)
    yield from api.hold(0.3)
    c, Rc = api.object_pose("cube")
    note(f"turned: cube yaw change={np.degrees(wrap(yaw_of(Rc) - yaw_of(R0), np.pi / 2)):.1f}deg dz={1000 * (c[2] - p0[2]):.1f}mm")

    # ---- put it back where it started
    for target_dz, T in ((0.03, 1.5), (0.001, 1.2)):
        c, _ = api.object_pose("cube")
        P, R = api.palm_pose()
        yield from api.move(P + (p0 + np.array([0, 0, target_dz]) - c), R, seconds=T)
        yield from api.hold(0.3)
    c, Rc = api.object_pose("cube")
    note(f"placed: offset={fmt(c - p0)} R22={Rc[2, 2]:.3f}")

    # ---- release and retreat
    P, R = api.palm_pose()
    P_rel = P + R @ (m_close - m_open)
    yield from api.move(P_rel, R, s=s_open, seconds=0.8)
    yield from api.hold(0.2)
    yield from api.move(P_rel + np.array([0, 0, 0.08]), R, seconds=1.0)
    yield from api.hold(0.3)
    c, Rc = api.object_pose("cube")
    note(f"final: offset={fmt(c - p0)} R22={Rc[2, 2]:.3f} t={api.t:.1f}")
