"""pen_finger_roll, synergies level.

Strategy: thumb-index pinch with the pinch axis along palm z (thumb toward the
wrist, index toward the fingertips), pen axis along palm x (palm down, pen vertical).
  roll  (rotation about pen axis) = differential pad shear along palm y (closed loop on twist)
  swing (rotation about palm y) = pivot the pen in the pinch with its lower tip on free
        table (palm moves along palm z, closed loop on in-palm tilt), lift to read the tilt
        in the air (only tilt above the rim is scored), compensate spring-back, repeat
        for +18, -18, 0 deg
Only the RELATIVE pad vector rel = index_pad - thumb_pad matters; its z part is the
squeeze. Synergy-space "IK" is built in-policy from measured pads (api.tips()):
a linear model tips(s) ~ T_b + J (s - s_b), J measured by finite differences in
free air, refined by Gauss-Newton + Broyden. Swing and roll are closed-loop on
the measured pen pose (same formulas as the scorer).
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
RIM = TABLE + 0.06
PEN_R = 0.00375
PAD = 0.008
C_STAR = np.array([80.0, 40.0, 118.0])  # mm, palm frame, thumb-index pinch centre (hand.py)
REL_MEAS = 24.0     # mm pad-centre separation for the free-air local Jacobian
REL_CLOSE = 2 * (PEN_R * 1000 + PAD * 1000) - 2 * 3.0   # 3 mm squeeze per side
REL_OPEN = 56.0


def _tilt(a):
    return float(np.arctan2(a[2], a[0]))


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def _twist(R0, R):
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


class Track:
    """Replica of the scorer's bookkeeping, updated every control step."""

    def __init__(self, api):
        self.api = api
        p, _ = api.object_pose("pen")
        self.reach = p[2] - TABLE
        self.out = False
        self.rest = None
        self.R_last = None
        self.twist = 0.0
        self.tw_lo = self.tw_hi = 0.0
        self.ti_lo = self.ti_hi = 0.0
        self.tilt = 0.0
        self.calm = 0
        self.settled = False
        self.bottom = -1.0
        self.t_out = None

    def step(self):
        api = self.api
        p, Rw = api.object_pose("pen")
        _, Rr = api.object_in_palm("pen")
        self.bottom = p[2] - self.reach * abs(Rw[2, 2])
        if not self.out and self.bottom > RIM + 0.01:
            self.out = True
            self.rest = _tilt(Rr[:, 2])
            self.R_last = Rr
            self.t_out = api.t
        if not self.out:
            return
        if self.bottom > RIM:
            self.twist += np.degrees(_twist(self.R_last, Rr))
            self.tw_lo, self.tw_hi = min(self.tw_lo, self.twist), max(self.tw_hi, self.twist)
            t = np.degrees(_wrap(_tilt(Rr[:, 2]) - self.rest))
            self.tilt = t
            self.ti_lo, self.ti_hi = min(self.ti_lo, t), max(self.ti_hi, t)
            if self.ti_hi >= 10 and self.ti_lo <= -10 and abs(t) < 5:
                self.calm += 1
                self.settled |= self.calm >= 25
            else:
                self.calm = 0
        self.R_last = Rr

    def summary(self):
        return (f"out={self.out} bottom_above_rim_mm={1000 * (self.bottom - RIM):.1f} tilt_lo/hi={self.ti_lo:.1f}/{self.ti_hi:.1f} "
                f"settled={self.settled} twist_range={self.tw_hi - self.tw_lo:.1f}")


class Model:
    """tips (15, mm) ~ Tb + J (s - sb)."""

    def __init__(self, sb, Tb, J):
        self.sb, self.Tb, self.J = np.array(sb, float), np.array(Tb, float), np.array(J, float)

    def copy(self):
        return Model(self.sb, self.Tb, self.J)

    def predict(self, s):
        return self.Tb + self.J @ (np.asarray(s) - self.sb)

    def update(self, s, T):
        s = np.asarray(s, float)
        ds = s - self.sb
        if ds @ ds > 1e-8:
            self.J = self.J + np.outer(T - self.predict(s), ds) / (ds @ ds)
        self.sb, self.Tb = s.copy(), np.asarray(T, float).copy()


def rel_cen(T):
    T = np.asarray(T).reshape(5, 3)
    return T[1] - T[0], 0.5 * (T[1] + T[0])


def solve(model, rel_t, cen_t, w_c=0.3, lam=0.3, mu=1.0, smax=3.0):
    k = model.J.shape[1]
    J = model.J.reshape(5, 3, k)
    rb, cb = rel_cen(model.Tb)
    A = [J[1] - J[0], w_c * 0.5 * (J[1] + J[0]), lam * np.eye(k), mu * np.eye(k)]
    b = [rel_t - rb, w_c * (cen_t - cb), -lam * model.sb, np.zeros(k)]
    ds = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    return np.clip(model.sb + ds, -smax, smax)


class P:
    def __init__(self, api):
        self.api = api
        self.trk = Track(api)
        self.log = []

    def go(self, gen):
        for _ in gen:
            self.trk.step()
            yield

    def L(self, msg):
        self.log.append(f"t={self.api.t:.2f} {msg}")

    def measure(self, s, tm=0.16, th=0.08):
        yield from self.go(self.api.move(s=np.asarray(s, float), seconds=tm))
        yield from self.go(self.api.hold(th))
        return self.api.tips().reshape(-1) * 1000.0

    def cmd(self, seconds=0.06):
        s = self.s_c + self.V @ self.u
        yield from self.go(self.api.move(s=s, seconds=seconds))


def gn(p, model, rel_t, cen_t, iters, tag):
    s = solve(model, rel_t, cen_t)
    for i in range(iters):
        T = yield from p.measure(s)
        model.update(s, T)
        r, c = rel_cen(T)
        p.L(f"GN[{tag}] {i} rel={np.round(r, 1).tolist()} cen={np.round(c, 1).tolist()}")
        s = solve(model, rel_t, cen_t)
    return s


def tilt_now(p):
    _, Rr = p.api.object_in_palm("pen")
    return np.degrees(_wrap(_tilt(Rr[:, 2]) - p.trk.rest))


def main(p):
    api = p.api
    k = api.k
    p.L(f"k={k}")
    R_dn = api.hand_down()

    # ---------- 1. global Jacobian at the mean posture (free air, high above the table)
    s0 = np.zeros(k)
    T0 = yield from p.measure(s0, tm=0.4, th=0.15)
    J0 = np.zeros((15, k))
    d = 0.6
    for j in range(k):
        e = np.zeros(k); e[j] = d
        Tj = yield from p.measure(e)
        J0[:, j] = (Tj - T0) / d
    p.L(f"T0 (mm)={np.round(T0.reshape(5, 3), 0).astype(int).tolist()}")
    model = Model(s0, T0, J0)

    # ---------- 2. pinch posture with a small gap: Gauss-Newton on measured pads
    rel_m = np.array([0.0, 0.0, REL_MEAS])
    s_m = yield from gn(p, model, rel_m, C_STAR, 5, "meas")
    Tm = yield from p.measure(s_m, tm=0.2, th=0.1)
    model.update(s_m, Tm)
    # local Jacobian there
    Jl = np.zeros((15, k))
    dl = 0.3
    for j in range(k):
        e = s_m.copy(); e[j] += dl if s_m[j] <= 0 else -dl
        Tj = yield from p.measure(e)
        Jl[:, j] = (Tj - Tm) / (e[j] - s_m[j])
    local = Model(s_m, Tm, Jl)
    Jx = Jl.reshape(5, 3, k)
    Jr = Jx[1] - Jx[0]
    Jc = 0.5 * (Jx[1] + Jx[0])
    sv = np.linalg.svd(Jr, compute_uv=False)
    r_m, c_m = rel_cen(Tm)
    p.L(f"s_meas={np.round(s_m, 2).tolist()} rel={np.round(r_m, 1).tolist()} cen={np.round(c_m, 1).tolist()} Jr_sv={np.round(sv, 2).tolist()}")

    # directions in s for unit (mm) changes of rel x / y / z, light centre-motion penalty
    V = np.zeros((k, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A = np.vstack([Jr, 0.3 * Jc, 0.05 * np.eye(k)])
        b = np.concatenate([e, np.zeros(3), np.zeros(k)])
        V[:, i] = np.linalg.lstsq(A, b, rcond=None)[0]
    p.L(f"V norms={np.round(np.linalg.norm(V, axis=0), 3).tolist()} Jr@V={np.round(Jr @ V, 2).tolist()}")
    # closed posture by local extrapolation
    s_c = s_m + V @ np.array([-r_m[0], -r_m[1], REL_CLOSE - r_m[2]])
    pred_c = local.predict(s_c)
    r_c, c_c = rel_cen(pred_c)
    p.L(f"s_close={np.round(s_c, 2).tolist()} pred rel={np.round(r_c, 1).tolist()} cen={np.round(c_c, 1).tolist()} others={np.round(pred_c.reshape(5, 3)[2:], 0).tolist()}")

    # open posture, same centre
    om = local.copy()
    s_o = yield from gn(p, om, np.array([0.0, 0.0, REL_OPEN]), c_m, 3, "open")
    To = yield from p.measure(s_o, tm=0.2, th=0.1)
    r_o, c_o = rel_cen(To)
    p.L(f"s_open={np.round(s_o, 2).tolist()} rel={np.round(r_o, 1).tolist()} cen={np.round(c_o, 1).tolist()}")

    p.s_c, p.V, p.u = s_c, V, np.zeros(3)

    # ---------- 3. approach
    pen, Rpen = api.object_pose("pen")
    L = 2 * (pen[2] - TABLE)
    top = TABLE + L
    z_g = max(top - 0.042, RIM + 0.03)
    C = c_m / 1000.0   # pinch centre (palm frame, m) used for placement
    p.C = C
    p.L(f"pen L={L * 1000:.0f}mm top={top:.3f} z_g={z_g:.3f}")
    hover = api.palm_for(np.array([pen[0], pen[1], z_g + 0.08]), C, R_dn)
    yield from p.go(api.move(hover, R_dn, s=s_o, seconds=2.0))
    grasp = api.palm_for(np.array([pen[0], pen[1], z_g]), C, R_dn)
    yield from p.go(api.move(grasp, R_dn, seconds=1.5))
    # lateral correction: bring the pen axis through the pinch centre
    for _ in range(2):
        pp, Rp = api.object_in_palm("pen")
        a = Rp[:, 2]
        q = pp + a * ((C - pp) @ a)
        _, Rm = api.measured_palm()
        delta = Rm @ (q - C)
        p.L(f"align err_mm={np.round((q - C) * 1000, 1).tolist()}")
        xyz, _ = api.palm_pose()
        yield from p.go(api.move(xyz + delta, R_dn, seconds=0.4))
        yield from p.go(api.hold(0.1))
    # close
    yield from p.go(api.move(s=s_c, seconds=1.0))
    yield from p.go(api.hold(0.3))
    F = api.contact_forces("pen")
    p.L(f"close forces={np.round(F, 2).tolist()}")
    for _ in range(3):
        if F[0] > 0.15 and F[1] > 0.15:
            break
        p.u[2] -= 2.0
        yield from p.cmd(0.3)
        yield from p.go(api.hold(0.2))
        F = api.contact_forces("pen")
        p.L(f"squeeze u_z={p.u[2]} forces={np.round(F, 2).tolist()}")

    # ---------- 4. lift (a little extra squeeze first)
    p.u[2] -= 1.0
    yield from p.cmd(0.3)
    xyz, _ = api.palm_pose()
    yield from p.go(api.move(xyz + np.array([0, 0, 0.13]), R_dn, seconds=2.5))
    yield from p.go(api.hold(0.3))
    F = api.contact_forces("pen")
    p.L(f"lifted: {p.trk.summary()} forces={np.round(F, 2).tolist()}")
    if not p.trk.out:
        return

    # carry to a free spot of table, away from the cup toward the start pose
    pen_now, _ = api.object_pose("pen")
    away = np.array([0.45, -0.30]) - pen_now[:2]
    away = away / (np.linalg.norm(away) + 1e-9)
    xyz, _ = api.palm_pose()
    yield from p.go(api.move(xyz + 0.12 * np.array([away[0], away[1], 0.0]), R_dn, seconds=1.5))
    yield from p.go(api.hold(0.2))

    def keep_grip():
        F = api.contact_forces("pen")
        if min(F[0], F[1]) < 0.3 and p.u[2] > -6.0:
            p.u[2] -= 0.3
        return F

    def servo(target, axis, g, done, tmax, get, umax=8.0, k_i=0.3, rate=0.3):
        t_end = api.t + tmax
        while api.t < t_end and api.t < 40.0:
            if p.trk.bottom < RIM + 0.005:
                p.L("pen dropping")
                return False
            keep_grip()
            err = target - get()
            du = float(np.clip(k_i * err / g, -rate, rate))
            p.u[axis] = float(np.clip(p.u[axis] + du, -umax, umax))
            yield from p.cmd(0.06)
            if done():
                return True
        return False

    # ---------- 5. roll first (rel y = differential pad shear across the pen), closed loop on twist
    w0 = p.trk.twist
    p.u[1] += 1.5
    yield from p.cmd(0.3)
    yield from p.go(api.hold(0.3))
    gy = (p.trk.twist - w0) / 1.5
    p.L(f"probe y: dtwist={p.trk.twist - w0:.2f}deg gy={gy:.2f}deg/mm forces={np.round(api.contact_forces('pen'), 2).tolist()}")
    if abs(gy) < 0.5:
        gy = 5.0 if gy >= 0 else -5.0
    for tgt in (w0 + 21.0, w0 - 17.0):
        ok = yield from servo(tgt, 1, gy, (lambda tg=tgt: (p.trk.tw_hi - p.trk.tw_lo) >= 37.0 or abs(p.trk.twist - tg) < 1.0),
                              5.0, lambda: p.trk.twist)
        p.L(f"roll to {tgt:.1f}: ok={ok} twist={p.trk.twist:.1f} range={p.trk.tw_hi - p.trk.tw_lo:.1f} u={np.round(p.u, 2).tolist()} F={np.round(api.contact_forces('pen'), 2).tolist()}")
        if p.trk.tw_hi - p.trk.tw_lo >= 37.0 or p.trk.bottom < RIM + 0.005:
            break
    # return the shear to the grasp posture, then re-tighten until both pads press >= 0.5 N
    p.u[1] = 0.0
    yield from p.cmd(0.4)
    yield from p.go(api.hold(0.15))
    for _ in range(8):
        F = api.contact_forces("pen")
        if min(F[0], F[1]) >= 0.5 or p.u[2] <= -5.0:
            break
        p.u[2] -= 0.5
        yield from p.cmd(0.15)
        yield from p.go(api.hold(0.1))
    p.L(f"after roll: {p.trk.summary()} u={np.round(p.u, 2).tolist()} F={np.round(api.contact_forces('pen'), 2).tolist()}")
    if p.trk.bottom < RIM + 0.005:
        return

    # ---------- 6. swing: pivot the pen in the pinch about its lower tip resting on the table.
    # The pinch centre is driven along the arc of radius d about the tip (rigid-pen geometry),
    # so if the pen rotates in the pinch more than commanded its tip lifts off instead of
    # buckling (iter 3: following the *measured* tilt down made the pen buckle flat).
    zdir = R_dn[:, 2]          # palm +z in world, horizontal; pinch toward -z => tilt +
    up = np.array([0.0, 0.0, 1.0])
    C = p.C

    def geom():
        pc, Rw = api.object_pose("pen")
        a = Rw[:, 2] * (-1.0 if Rw[2, 2] > 0 else 1.0)       # axis pointing down
        B = pc + a * (p.trk.reach - 0.0035)                   # bottom hemisphere centre
        P, Rm = api.measured_palm()
        Cw = P + Rm @ C
        Q = pc + a * ((Cw - pc) @ a)
        return B, Cw, Q

    def lower():
        t_end = api.t + 4.0
        while api.t < t_end:
            b = p.trk.bottom
            if b <= TABLE + 0.0008:
                break
            xyz, _ = api.palm_pose()
            if b > TABLE + 0.02:
                yield from p.go(api.move(xyz - np.array([0, 0, b - TABLE - 0.012]), R_dn, seconds=max(0.3, (b - TABLE) / 0.08)))
                continue
            yield from p.go(api.move(xyz - np.array([0, 0, min(b - TABLE + 0.0005, 0.001)]), R_dn, seconds=0.06))

    def pivot(goal, tmax=3.0):
        B, Cw0, Q0 = geom()
        xyz0, _ = api.palm_pose()
        v = Q0 - B
        d = float(np.linalg.norm(v))
        vperp = v - (v @ up) * up - (v @ zdir) * zdir
        off = Cw0 - Q0
        phi0 = float(np.arctan2(-(v @ zdir), v @ up))
        tc = tilt_now(p)
        phi = phi0
        phi_goal = phi0 + np.radians(goal - tc)
        good = 0
        t_end = api.t + tmax
        while api.t < t_end:
            tl = tilt_now(p)
            err = goal - tl
            if abs(err) < 1.0:
                good += 1
                if good >= 3:
                    break
            else:
                good = 0
            dphi = phi_goal - phi
            if abs(dphi) < 1e-4:
                phi_goal += np.radians(np.clip(0.2 * err, -1.0, 1.0))
                phi_goal = float(np.clip(phi_goal, phi0 - np.radians(50), phi0 + np.radians(50)))
            phi += float(np.clip(phi_goal - phi, -np.radians(0.8), np.radians(0.8)))
            tgt = B + d * (np.cos(phi) * up - np.sin(phi) * zdir) + vperp + off
            yield from p.go(api.move(xyz0 + (tgt - Cw0) - np.array([0, 0, 0.001]), R_dn, seconds=0.06))
        p.L(f"pivot goal={goal:.1f} from {tc:.1f}: tilt={tilt_now(p):.1f} phi-phi0={np.degrees(phi - phi0):.1f} d={d * 1000:.0f}mm")
        return d

    def lift(hold):
        xyz, _ = api.palm_pose()
        dz = (RIM + 0.022) - p.trk.bottom
        yield from p.go(api.move(xyz + np.array([0, 0, dz]), R_dn, seconds=0.6))
        yield from p.go(api.hold(hold))

    sb = 4.0    # spring-back estimate [deg]: air tilt falls short of the table tilt by this
    hist = []

    def cycle(goal):
        nonlocal sb
        cur = tilt_now(p)
        dsg = np.sign(goal - cur) if abs(goal - cur) > 0.5 else 1.0
        tgt = float(np.clip(goal + dsg * sb, -40, 40))
        yield from lower()
        dg = yield from pivot(tgt)
        t_tab = tilt_now(p)
        yield from lift(0.35)
        t_air = tilt_now(p)
        if abs(t_tab - cur) > 3:
            sb = float(np.clip(0.5 * sb + 0.5 * dsg * (t_tab - t_air), 0.0, 20.0))
        hist.append((goal, round(tgt, 1), round(float(t_tab), 1), round(float(t_air), 1)))
        p.L(f"cycle goal={goal:.1f} tgt={tgt:.1f} table={t_tab:.1f} air={t_air:.1f} sb={sb:.1f} d={dg * 1000:.0f}mm {p.trk.summary()} F={np.round(api.contact_forces('pen'), 2).tolist()}")
        return t_air

    def lost():
        return p.trk.bottom < RIM

    # side 1
    for _ in range(3):
        if lost() or api.t > 46:
            break
        yield from cycle(18.0)
        if p.trk.ti_hi >= 14.0:
            break
    # side 2
    for _ in range(3):
        if lost() or api.t > 48:
            break
        g2 = -max(18.0, 33.0 - p.trk.ti_hi)
        yield from cycle(g2)
        if p.trk.ti_lo <= -11.0 and p.trk.ti_hi - p.trk.ti_lo >= 31.0:
            break
    # back to rest
    for _ in range(3):
        if lost() or api.t > 49:
            break
        if abs(tilt_now(p)) < 3.0:
            break
        yield from cycle(0.0)
    if not lost():
        yield from p.go(api.hold(0.8))
    p.L(f"end: {p.trk.summary()} hist={hist} forces={np.round(api.contact_forces('pen'), 2).tolist()}")


def run(api):
    p = P(api)
    try:
        yield from main(p)
    except Exception as e:
        p.L(f"EXCEPTION {type(e).__name__}: {e}")
        raise
    finally:
        for i in range(0, len(p.log), 3):
            api.note(" || ".join(p.log[i:i + 3]))
