import numpy as np

TABLE_Z = 0.75
BLOCK_TOP = 0.81
PINCH = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (hand.py constant)
R_PEN = 0.00375                            # nominal pen radius
GRASP_BELOW_TOP = 0.016                    # first contact this far below the pen's upper end
SLIP_TARGET = 0.021                        # stop lowering after this much measured slide
LIFT = 0.135


def pen_axis(api):
    p, R = api.object_pose("pen")
    a = R[:, 2]
    return p, R, (a if a[2] >= 0 else -a)


def to_obj(api, x_w, n_w=None):
    """World point (and direction) -> pen object frame."""
    p, R = api.object_pose("pen")
    pt = R.T @ (np.asarray(x_w) - p)
    return pt if n_w is None else (pt, R.T @ np.asarray(n_w))


V_ANG = np.radians(55)                      # index/middle contacts at +-55 deg either side of the thumb's opposite


def frame_uv(api, Rpalm):
    _, _, a = pen_axis(api)
    u = Rpalm[:, 2] - (Rpalm[:, 2] @ a) * a     # palm +z (finger direction) ⟂ pen: thumb -> index
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    if v @ Rpalm[:, 1] < 0:                     # v toward the thumb side (palm +y)
        v = -v
    return a, u, v


def pinch_points(api, axis_world_pt, r, Rpalm):
    """Thumb on one side of the pen; index and middle form a V on the other side (index toward palm +y).
    Object-frame points and outward normals, so the pen sits in a groove and cannot roll out sideways."""
    a, u, v = frame_uv(api, Rpalm)
    dirs = {"thumb": -u,
            "index": np.cos(V_ANG) * u + np.sin(V_ANG) * v,
            "middle": np.cos(V_ANG) * u - np.sin(V_ANG) * v}
    pts, nrm = {}, {}
    for f, d in dirs.items():
        pts[f], nrm[f] = to_obj(api, axis_world_pt + r * d, d)
    return pts, nrm


def palm_dirs(cy):
    c = np.array([PINCH[0], cy, PINCH[2]])
    k = R_PEN + 0.008
    z, y = np.array([0, 0, 1.]), np.array([0, 1., 0])
    return c, {"thumb": c - k * z,
               "index": c + k * (np.cos(V_ANG) * z + np.sin(V_ANG) * y),
               "middle": c + k * (np.cos(V_ANG) * z - np.sin(V_ANG) * y)}


class Tracker:
    """Live copy of the scorer's slide metric (pen travel along its axis relative to the thumb pad,
    counted only while thumb + another finger press > 0.2 N and the pen's lower end is on the table)."""

    def __init__(self, api, end):
        self.api, self.end = api, end
        self.cum = self.cmin = self.slip = 0.0
        self.prev = None

    def update(self):
        p, _, a = pen_axis(self.api)
        tip_h = p[2] - self.end * a[2] - TABLE_Z
        f = self.api.contact_forces("pen")
        held = f[0] > 0.25 and f[1:5].max() > 0.25
        if held and tip_h < 0.0025:
            s = float((p - self.api.tips_world()[0]) @ a)
            if self.prev is not None:
                self.cum += s - self.prev
                self.cmin = min(self.cmin, self.cum)
                self.slip = max(self.slip, self.cum - self.cmin)
            self.prev = s
        else:
            self.prev = None
        return f, tip_h


def lean(api):
    """Pen axis in the measured palm frame (x down, y thumb side, z fingers) x1000."""
    _, Rp = api.measured_palm()
    _, _, a = pen_axis(api)
    return np.round(Rp.T @ a * 1000).astype(int).tolist()


def run(api):
    Rd = api.hand_down()
    p, R, a = pen_axis(api)
    end = float(p[2] - TABLE_Z - 0.0005)              # centre-to-end half length
    top = p + end * a
    api.note(f"pen centre {np.round(p, 4).tolist()} half-length {end:.4f} top z {top[2]:.4f}")

    # 0. pick the palm-frame grasp centre where thumb/index/middle all reach their V-grasp pads
    best = None
    for cy in (0.030, 0.034, 0.026, 0.038, 0.022):
        c, tg = palm_dirs(cy)
        res = {f: api.reach(f, x) for f, x in tg.items()}
        worst = max(res.values())
        api.note(f"reach cy={cy}: " + str({f: round(e * 1000, 1) for f, e in res.items()}))
        if best is None or worst < best[0] - 1e-4:
            best = (worst, c)
        if worst < 0.0015:
            break
    CTR = best[1]
    api.note(f"grasp centre palm {np.round(CTR, 3).tolist()} worst reach {best[0]*1000:.1f}mm")

    # 1. open, go above the pen with the grasp centre 45 mm above its upper end
    yield from api.open(0.6)
    pre = api.palm_for(top + np.array([0, 0, 0.045]), CTR, Rd)
    yield from api.carry(pre, Rd, 2.0)
    yield from api.hold(0.3)

    # 2. pre-shape thumb/index/middle around the pen axis (extended upward) at the centre height
    P, Rp = api.measured_palm()
    c_w = P + Rp @ CTR
    p, R, a = pen_axis(api)
    axis_pt = p + ((c_w - p) @ a) * a
    pts, nrm = pinch_points(api, axis_pt, R_PEN, Rp)
    r = yield from api.approach("pen", pts, standoff=0.014, seconds=0.8, normals=nrm)
    api.note(f"approach ik {r.get('ik_residual_mm')} tips_palm {np.round(api.tips()[:3] * 1000).tolist()}")

    # 3. descend so the open pads straddle the grasp height
    p, R, a = pen_axis(api)
    top = p + end * a
    g_w = top - GRASP_BELOW_TOP * a
    drop = float((axis_pt - g_w) @ a)
    cmd, _ = api.palm_pose()
    yield from api.carry(cmd - drop * a, Rd, 1.2)
    yield from api.hold(0.2)

    # 4. close thumb against the index/middle V lightly (slide-friendly)
    sq = 0.002

    def regrasp(squeeze, seconds=0.5):
        tw = api.tips_world()
        p, R, a = pen_axis(api)
        h = (tw[0] + tw[1] + tw[2]) / 3
        ax = p + ((h - p) @ a) * a
        _, Rp = api.measured_palm()
        pts, nrm = pinch_points(api, ax, R_PEN, Rp)
        res = yield from api.grasp("pen", pts, squeeze=squeeze, seconds=seconds, normals=nrm)
        return res

    res = yield from regrasp(sq, 0.8)
    yield from api.hold(0.3)
    f = api.contact_forces("pen")
    api.note(f"grasp sq={sq} forces {np.round(f, 2).tolist()} ik {res.get('ik_residual_mm')} lean {lean(api)}")
    for _ in range(4):
        f = api.contact_forces("pen")
        if f[0] > 0.4 and f[1:3].max() > 0.4:
            break
        sq += 0.001
        yield from regrasp(sq, 0.4)
        yield from api.hold(0.2)
        api.note(f"tighten sq={sq} forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # 5. lower the hand: the table pushes the pen up through the grasp
    tr = Tracker(api, end)
    tr.update()
    lowered = 0.0
    while tr.slip < SLIP_TARGET and lowered < 0.040:
        tw = api.tips_world()
        if tw[:3, 2].min() < BLOCK_TOP + 0.012:
            api.note(f"stop: pads near block top {np.round(tw[:3, 2], 4).tolist()}")
            break
        cmd, _ = api.palm_pose()
        for _ in api.carry(cmd - np.array([0, 0, 0.002]), Rd, 0.1):
            f, tip_h = tr.update()
            yield
        lowered += 0.002
        f, tip_h = tr.update()
        if f[0] < 0.35 or f[1:5].max() < 0.35:
            api.note(f"lowered {lowered*1000:.0f}mm slip {tr.slip*1000:.1f} LOST f {np.round(f, 2).tolist()} lean {lean(api)}")
            sq = min(sq + 0.0007, 0.005)
            yield from regrasp(sq, 0.2)
        elif round(lowered * 1000) % 4 == 0:
            api.note(f"lowered {lowered*1000:.0f}mm slip {tr.slip*1000:.1f} tip_h {tip_h*1000:.1f} f {np.round(f, 2).tolist()} lean {lean(api)}")
    for _ in range(10):
        tr.update()
        yield
    api.note(f"slide done: lowered {lowered*1000:.0f}mm tracked slip {tr.slip*1000:.1f}mm")

    # 6. firm up and lift the pen out of the guide, checking it follows
    sq = sq + 0.001
    yield from regrasp(sq, 0.4)
    yield from api.hold(0.2)
    api.note(f"pre-lift forces {np.round(api.contact_forces('pen'), 2).tolist()} lean {lean(api)}")
    lifted = 0.0
    p, R, a = pen_axis(api)
    h0 = p[2]
    while lifted < LIFT - 1e-6:
        cmd, _ = api.palm_pose()
        yield from api.carry(cmd + np.array([0, 0, 0.01]), Rd, 0.25)
        lifted += 0.01
        p, R, a = pen_axis(api)
        f = api.contact_forces("pen")
        lag = lifted - (p[2] - h0)
        api.note(f"lift {lifted*1000:.0f}mm pen rose {(p[2]-h0)*1000:.1f} f {np.round(f, 2).tolist()} lean {lean(api)}")
        if lag > 0.004 and f[0] > 0.1 and sq < 0.006:
            sq += 0.001
            yield from regrasp(sq, 0.25)
    yield from api.hold(0.5)
    p, R, a = pen_axis(api)
    api.note(f"lifted: tip_h {(p[2] - end * a[2] - TABLE_Z)*1000:.1f}mm forces {np.round(api.contact_forces('pen'), 2).tolist()}")
