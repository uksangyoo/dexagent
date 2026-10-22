import numpy as np
from scipy.spatial.transform import Rotation

# Iteration 2: cup the hand with synergies (thumb + little in, slight curl), then
# gravity-driven baoding: the arm tilts the palm (base 10 deg) with the downhill
# direction psi fed back from the measured pair angle (psi = pair angle - 90 + lead,
# rate limited), plus a PD "virtual wall" tilt that keeps each ball inside a safe box.

PIVOT = np.array([0.045, 0.0, 0.09])   # palm-frame point the tilt rotates about
TILT = np.radians(10.0)
TILT_MAX = np.radians(20.0)
LEAD = np.radians(35.0)
RATE = np.radians(20.0)                # max psi rate [rad/s]
TARGET = np.radians(215.0)
BOX_Y = (-0.036, 0.026)
BOX_Z = (0.055, 0.125)
KP, KD = 12.0, 1.2                     # rad per m, rad per m/s
CUP_DELTA = np.array([0.5, 0.0, -1.2, 0.0, 0.0, 0.0])


class Tracker:
    """Same signed cumulative rotation as the scorer."""
    def __init__(self):
        self.last, self.total = None, 0.0

    def update(self, api):
        a, _ = api.object_in_palm("ball_a")
        b, _ = api.object_in_palm("ball_b")
        _, Rp = api.measured_palm()
        tilt = np.degrees(np.arccos(np.clip(Rp[2, 0], -1, 1)))
        held = all(-0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15 for p in (a, b))
        d = b - a
        ang = float(np.arctan2(d[1], d[2]))
        if held and tilt < 30 and self.last is not None and api.t > 1.0:
            self.total += (ang - self.last + np.pi) % (2 * np.pi) - np.pi
        self.last = ang if held else None
        return a, b, ang, tilt, held, Rp


def tilt_pose(R0, W, vec):
    """Palm pose tilted so palm-frame direction vec (y,z; |vec| = angle rad) points downhill."""
    th = float(np.linalg.norm(vec))
    if th < 1e-6:
        R = R0
    else:
        wd = R0[:, 1] * vec[0] + R0[:, 2] * vec[1]
        k = np.cross(R0[:, 0], wd)
        k /= np.linalg.norm(k)
        R = Rotation.from_rotvec(k * th).as_matrix() @ R0
    return W - R @ PIVOT, R


def contain(api, Rp, balls):
    """PD virtual-wall tilt (palm y,z) pushing balls back into the safe box."""
    out = np.zeros(2)
    for name, p in balls:
        v = api.object_velocity(name)
        v = np.zeros(3) if v is None else Rp.T @ v[0]
        for j, (lo, hi) in ((0, BOX_Y), (1, BOX_Z)):
            q, vq = p[j + 1], v[j + 1]
            if q > hi - 0.006:
                out[j] -= KP * max(q - hi, 0) + KD * max(vq, 0)
            if q < lo + 0.006:
                out[j] += KP * max(lo - q, 0) + KD * max(-vq, 0)
    return out


def run(api):
    P0, R0 = api.palm_pose()
    W = P0 + R0 @ PIVOT
    home_s = api.coeffs()
    head = [f"home_coeffs={np.round(home_s, 2).tolist()}"]
    tr = Tracker()
    log = []
    for _ in range(25):
        yield from api.hold(0.02)
        tr.update(api)
    # cup the hand while level (balls settle meanwhile)
    s_cup = home_s + CUP_DELTA
    yield from api.move(s=s_cup, seconds=1.0)
    for _ in range(10):
        yield from api.hold(0.02)
        a, b, ang, tilt, held, Rp = tr.update(api)
    head.append(f"cup s={np.round(s_cup, 2).tolist()} tips_mm={np.round(api.tips() * 1000).astype(int).tolist()} "
                f"a={np.round(a*1000).astype(int).tolist()} b={np.round(b*1000).astype(int).tolist()} ang={np.degrees(ang):.0f} tot={np.degrees(tr.total):.0f}")
    psi = ang - np.pi / 2
    t_start = api.t
    cur = np.zeros(2)
    step = 0
    while abs(tr.total) < TARGET and api.t < 50:
        a, b, ang, tilt, held, Rp = tr.update(api)
        ramp = min(1.0, (api.t - t_start) / 1.0)
        want = ang - np.pi / 2 + LEAD
        dpsi = (want - psi + np.pi) % (2 * np.pi) - np.pi
        psi += float(np.clip(dpsi, -RATE * 0.02, RATE * 0.02))
        vec = TILT * ramp * np.array([np.sin(psi), np.cos(psi)])
        push = contain(api, Rp, (("ball_a", a), ("ball_b", b)))
        vec = vec + push
        n = np.linalg.norm(vec)
        if n > TILT_MAX:
            vec *= TILT_MAX / n
        cur = 0.7 * cur + 0.3 * vec
        xyz, R = tilt_pose(R0, W, cur)
        try:
            yield from api.move(xyz, R, seconds=0.02)
        except ValueError:
            yield from api.hold(0.02)
        if step % 25 == 0:
            f = np.round(api.contact_forces("ball_a") + api.contact_forces("ball_b"), 1)
            log.append(f"t={api.t:.1f} psi={np.degrees(psi):.0f} cur={np.round(np.degrees(cur)).astype(int).tolist()} "
                       f"push={np.round(np.degrees(push)).astype(int).tolist()} ang={np.degrees(ang):.0f} tot={np.degrees(tr.total):.0f} "
                       f"a={np.round(a[1:]*1000).astype(int).tolist()} b={np.round(b[1:]*1000).astype(int).tolist()} F={f.tolist()}")
        step += 1
    head.append(f"loop end t={api.t:.1f} total={np.degrees(tr.total):.0f}")
    c0 = cur.copy()
    for i in range(1, 76):
        u = i / 75
        u = u * u * (3 - 2 * u)
        xyz, R = tilt_pose(R0, W, (1 - u) * c0)
        try:
            yield from api.move(xyz, R, seconds=0.02)
        except ValueError:
            yield from api.hold(0.02)
        tr.update(api)
    for _ in range(25):
        yield from api.hold(0.02)
        a, b, ang, tilt, held, Rp = tr.update(api)
    api.note(f"final total={np.degrees(tr.total):.0f} a={np.round(a*1000).astype(int).tolist()} b={np.round(b*1000).astype(int).tolist()} held={held}")
    for h in head:
        api.note(h)
    for i in range(0, len(log), 3):
        api.note(" | ".join(log[i:i + 3]))
