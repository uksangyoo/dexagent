"""pen_pickup_pivot, force level (v2).

Thumb-index pinch near one end of the pen (pinch line horizontal, across the pen),
held by the grip-force servo. Instead of relying on the pinch to act as a free
hinge (iter 1: it did not; the pads peeled off), pivot actively: move the pinch
along an arc about the pen's far end while rolling the palm about the pinch line
by the same angle, so the grasp never has to turn relative to the pen. At 90 deg
the pen is vertical under the pinch. Level any residual tilt by rotating the palm
about the pinch, lower until the pen's end touches the table (contact detected
from the pen moving up relative to the pads), then lift it clear and level again.
"""
import numpy as np

TABLE = 0.75
PC = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (thumb -z, index +z)
S_PINCH = 0.045                        # pinch this far from the pen centre toward the top end [m]
PADZ = 0.0105                          # pad centres this high above the table (they rest at ~10 mm)
GRIP_N = 0.9
PAD = 0.008                            # pad centre to finger surface
H_MAX = 0.0662 * 1.12                  # longest possible cylinder half-length
CLEAR = 0.060                          # target height of the lower end when held


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _rotvec(b, t):
    ax = np.cross(b, t)
    s, c = np.linalg.norm(ax), float(b @ t)
    if s < 1e-9:
        return np.zeros(3)
    return ax / s * np.arctan2(s, c)


def _rotm(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def run(api):
    log = []
    st = {"h": 0.0662, "stand": 90.0, "sgn": 1.0, "r": 0.00375}

    def pen():
        p, R = api.object_pose("pen")
        b = -st["sgn"] * R[:, 2]            # toward the bottom end
        tilt = float(np.degrees(np.arccos(np.clip(-b[2], -1, 1))))
        low = float(p[2] - st["h"] * abs(b[2]) - st["r"] - TABLE)
        if low < 0.002:
            st["stand"] = min(st["stand"], tilt)
        return p, b, tilt, low

    def pinch_mid():
        tw = api.tips_world()
        return 0.5 * (tw[0] + tw[1])

    def forces():
        return np.round(api.contact_forces("pen")[:2], 2).tolist()

    def level(X, R, rounds=3, tag="", full=True):
        """Rotate the palm about the pinch so the pen axis points straight down."""
        for k in range(rounds):
            p, b, tilt, low = pen()
            rv = _rotvec(b, np.array([0, 0, -1.0]))
            if not full:
                fw = R[:, 2]
                rv = rv - (rv @ fw) * fw
            ang = np.linalg.norm(rv)
            log.append(f"lvl{tag}{k} tilt={tilt:.1f} low={low*1000:.1f} rot={np.degrees(ang):.1f} F={forces()}")
            if np.degrees(ang) < 2.0:
                break
            if ang > 0.4:
                rv = rv * 0.4 / ang
            Rc = _rotm(rv)
            P = pinch_mid()
            Xn, Rn = P + Rc @ (X - P), Rc @ R
            try:
                yield from api.carry(xyz=Xn, R=Rn, seconds=0.8)
            except ValueError as e:
                log.append(f"lvl IK fail {str(e)[:60]}")
                break
            X, R = Xn, Rn
            yield from api.hold(0.3)
        return X, R

    def grasp(e):
        """Approach and pinch the pen near its end e*u (e = -1: little-finger side). Returns (X, R, u)."""
        p0, Rpen0 = api.object_pose("pen")
        r = float(np.clip(p0[2] - TABLE, 0.003, 0.0045))
        st["r"] = r
        a = Rpen0[:, 2]
        ah = _unit(np.array([a[0], a[1], 0.0]))
        best = None
        for s in (1.0, -1.0):
            u = s * ah
            psi = np.arctan2(u[1], u[0]) - np.pi / 2
            d = (psi - 0.7 + np.pi) % (2 * np.pi) - np.pi
            if best is None or abs(d) < abs(best[0]):
                best = (d, u)
        d, u = best
        Rp = api.hand_down(0.7 + d)
        f = Rp[:, 2]
        st["sgn"] = 1.0 if Rpen0[:, 2] @ (e * u) > 0 else -1.0   # sgn * pen z -> pinched (top) end
        # pads hover just above the table (iter 1-3: commanding them lower pressed the tips onto the
        # table, where table friction stalled the index's closing)
        Q = p0 + e * S_PINCH * u
        Q[2] = TABLE + PADZ
        X = api.palm_for(Q, PC, Rp)
        wo = r + PAD + 0.016
        pads = {"thumb": PC + [0, 0, -wo], "index": PC + [0, 0, wo]}
        res = yield from api.move_tips(xyz=X + [0, 0, 0.045], R=Rp, tips=pads, seconds=1.4)
        e1 = res.get("ik_residual_mm")
        yield from safe(X, Rp, 0.8)
        g = {}
        for tries in range(2):
            # centre the pads on the pen's current position, in the measured palm frame
            p, Rpen = api.object_pose("pen")
            ax = _unit(np.array([Rpen[0, 2], Rpen[1, 2], 0.0]))
            ax = ax if ax @ u > 0 else -ax
            Qw = p + e * S_PINCH * ax
            Qw[2] = TABLE + PADZ
            Pm, Rm = api.measured_palm()
            wc = r + PAD + 0.001
            tgt = {"thumb": Rm.T @ (Qw - wc * f - Pm), "index": Rm.T @ (Qw + wc * f - Pm)}
            res = yield from api.move_tips(tips=tgt, seconds=0.5)
            g = yield from api.grip("pen", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.6)
            F = (g.get("forces") or [0, 0])[:2]
            tw = api.tips_world()
            log.append(f"grasp e={e:+.0f} try{tries} r={r*1000:.2f} psi={0.7+d:.3f} ik={e1}/{res.get('ik_residual_mm')} "
                       f"F={F} tr={g.get('travel_mm')} padz={np.round((tw[:2, 2]-TABLE)*1000, 1).tolist()}")
            if min(F) >= 0.35:
                break
            yield from api.ungrip()
            op = {"thumb": Rm.T @ (Qw - (wc + 0.012) * f - Pm), "index": Rm.T @ (Qw + (wc + 0.012) * f - Pm)}
            yield from api.move_tips(tips=op, seconds=0.4)
            yield from api.hold(0.3)
        return X, Rp, u

    def pivot(X, Rp, u, e):
        """Arc the pinch about the far end while rolling the palm about the pinch line."""
        f = Rp[:, 2]
        P0 = pinch_mid()
        Pl = Rp.T @ (P0 - X)                      # pinch point in the palm frame
        dd = H_MAX + S_PINCH + 0.002              # arc radius (>= true: the far end may lift a little early)
        B = P0 - e * dd * u
        b0 = -e * u
        sign = np.sign(np.cross(b0, [0, 0, -1.0]) @ f) or 1.0
        n = 24
        Rroll, rolling = Rp, True
        for k in range(1, n + 1):
            th = 0.5 * np.pi * k / n
            P = B + dd * (e * np.cos(th) * u + np.array([0, 0, np.sin(th)]))
            R = _rotm(sign * th * f) @ Rp if rolling else Rroll
            Xk = P - R @ Pl
            try:
                yield from api.carry(xyz=Xk, R=R, seconds=0.2 if k > 2 else 0.4)
            except ValueError as ex:
                if not rolling:
                    log.append(f"pivot IK fail (no roll) at {np.degrees(th):.0f}deg")
                    return X, Rroll, False
                log.append(f"roll IK fail at {np.degrees(th):.0f}deg; continue arc with hinge")
                rolling = False
                R = Rroll
                Xk = P - R @ Pl
                ok = yield from safe(Xk, R, 0.3)
                if not ok:
                    return X, Rroll, False
            X, Rroll = Xk, R
            if k in (3, 12, 24):
                p, b, tilt, low = pen()
                log.append(f"piv{k} tilt={tilt:.1f} pz={(p[2]-TABLE)*1000:.1f} F={forces()}")
        return X, Rroll, True

    def safe(X, R=None, seconds=0.5):
        try:
            yield from api.carry(xyz=X, R=R, seconds=seconds)
            return True
        except ValueError as e:
            log.append(f"IK fail {str(e)[:50]}")
            return False

    try:
        q = api.OPEN.copy()
        q[8:12] = [0.0, 0.1, 0.15, 0.15]
        q[12:16] = [0.0, 0.1, 0.15, 0.15]
        yield from api.move_joints(q=q, seconds=0.4)
        ok = False
        for attempt, e in enumerate((-1.0, -1.0, 1.0)):
            if api.t > 34:
                break
            X, Rp, u = yield from grasp(e)
            X, Rp, done = yield from pivot(X, Rp, u, e)
            yield from api.hold(0.3)
            p, b, tilt, low = pen()
            F = forces()
            log.append(f"after pivot {attempt}: tilt={tilt:.1f} pz={(p[2]-TABLE)*1000:.1f} F={F}")
            if tilt < 45 and max(F) > 0.2:
                ok = True
                break
            # failed: let go, back off, return to palm-down and try again from the pen's new pose
            yield from api.ungrip()
            yield from api.move_tips(tips={"thumb": PC + [0, 0, -0.03], "index": PC + [0, 0, 0.03]}, seconds=0.4)
            Xu = X + np.array([0, 0, 0.06])
            yield from safe(Xu, None, 0.8)
            yield from safe(Xu, api.hand_down(0.7), 1.0)
            yield from api.hold(0.5)
        if ok:
            X, Rp = yield from level(X, Rp, rounds=3, tag="A")
            # lower until the pen's end touches the table
            p, b, tilt, low = pen()
            dz0 = p[2] - api.measured_palm()[0][2]
            touched = False
            for k in range(70):
                p, b, tilt, low = pen()
                dz = p[2] - api.measured_palm()[0][2]
                if k > 2 and dz - dz0 > 0.0012:
                    touched = True
                    break
                if pinch_mid()[2] - TABLE < 0.095:
                    break
                if k == 2:
                    dz0 = dz      # baseline once the arm is moving (tracking lag included)
                X = X - np.array([0, 0, 0.002])
                if not (yield from safe(X, None, 0.1)):
                    break
            yield from api.hold(0.2)
            p, b, tilt, low = pen()
            hm = (p[2] - TABLE - st["r"]) / max(abs(b[2]), 0.5)
            if touched and 0.05 < hm < 0.085:
                st["h"] = float(hm)
            p, b, tilt, low = pen()
            log.append(f"touch={touched} k={k} tilt={tilt:.1f} h={st['h']*1000:.1f} low={low*1000:.1f} stand={st['stand']:.1f} F={forces()}")
            # lift clear
            for k in range(60):
                p, b, tilt, low = pen()
                if low >= CLEAR:
                    break
                X = X + np.array([0, 0, 0.005])
                if not (yield from safe(X, None, 0.15)):
                    break
            yield from api.hold(0.3)
            X, Rp = yield from level(X, Rp, rounds=3, tag="B")
            yield from api.hold(0.4)
        p, b, tilt, low = pen()
        log.append(f"final tilt={tilt:.1f} low={low*1000:.1f} stand={st['stand']:.1f} F={forces()} gs={api.grip_state()}")
    finally:
        for line in log:
            api.note(line)
