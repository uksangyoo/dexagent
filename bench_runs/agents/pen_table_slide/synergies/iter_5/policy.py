"""pen_table_slide, synergies level.

Plan: calibrate the synergy -> fingertip map at runtime (finite differences of
measured pads, then Gauss-Newton/Broyden) to find a thumb-index pinch whose
pads oppose along palm z at the same depth; approach the pen top from above
open, close along the pre->tight synergy line until both pads press, lower
the palm so the table pushes the pen up through the pinch, then lift.
"""
import numpy as np

TABLE = 0.75
PEN_R = 0.00375          # nominal radius
PAD = 0.008              # pad centre to finger surface (approx.)
C_DES = np.array([0.080, 0.040, 0.118])   # palm-frame pinch centre to aim for
H_TIGHT = 0.0095         # half distance between pad centres, slightly squeezing the pen
H_PRE = 0.026            # open, pads well clear of the pen
LOG = []


def note(api, s):
    LOG.append(s)
    api.note(s)


def probe(api, s, t_move=0.3, t_hold=0.2):
    yield from api.move(s=s, seconds=t_move)
    yield from api.hold(t_hold)
    return api.tips()


def residual(T, J, s, pair, h, lam=0.004):
    a, b = pair
    Ja, Jb = J[3 * a:3 * a + 3], J[3 * b:3 * b + 3]
    rows, jac = [], []
    W1 = np.diag([2.0, 1.0, 1.0])
    d = T[a] - T[b]
    rows.append(W1 @ (d - np.array([0, 0, -2 * h])))
    jac.append(W1 @ (Ja - Jb))
    m = 0.5 * (T[a] + T[b])
    W2 = np.diag([0.3, 0.1, 0.1])
    rows.append(W2 @ (m - C_DES))
    jac.append(W2 @ (0.5 * (Ja + Jb)))
    for f in range(5):
        if f in pair:
            continue
        lim = m[0] - 0.02
        if T[f, 0] > lim:   # keep the other fingers well above the pinch (clear of the block)
            rows.append(np.array([0.5 * (T[f, 0] - lim)]))
            jac.append(0.5 * (J[3 * f:3 * f + 1] - 0.5 * (Ja[0:1] + Jb[0:1])))
    rows.append(lam * s)
    jac.append(lam * np.eye(len(s)))
    return np.concatenate(rows), np.vstack(jac)


def gn_step(T, J, s, pair, h, maxstep=1.0):
    r, Jr = residual(T, J, s, pair, h)
    ds = np.linalg.lstsq(Jr, -r, rcond=None)[0]
    n = np.abs(ds).max()
    if n > maxstep:
        ds *= maxstep / n
    return ds, float(r @ r)


def cost(T, J, s, pair, h):
    r, _ = residual(T, J, s, pair, h)
    return float(r @ r)


def solve(api, T, J, s, pair, h, iters):
    """Gauss-Newton with Broyden Jacobian updates on the measured pads."""
    best = (cost(T, J, s, pair, h), s.copy(), T.copy())
    for _ in range(iters):
        ds, _ = gn_step(T, J, s, pair, h)
        s1 = s + ds
        T1 = yield from probe(api, s1)
        dT = (T1 - T).ravel()
        J = J + np.outer(dT - J @ ds, ds) / max(ds @ ds, 1e-9)
        s, T = s1, T1
        c = cost(T, J, s, pair, h)
        if c < best[0]:
            best = (c, s.copy(), T.copy())
    return best[1], best[2], J


def fmt(v, k=1000, nd=1):
    return np.round(np.asarray(v) * k, nd).tolist()


def run(api):
    try:
        yield from _run(api)
    finally:
        api.note("SUMMARY | " + " | ".join(LOG))


def _run(api):
    note(api, f"k={api.k}")
    try:
        api.note("DESCRIBE\n" + api.describe())
    except Exception as e:
        api.note(f"describe failed {e}")
    R = api.hand_down()
    k = api.k

    # ---------------- calibrate synergies -> pads (palm frame), in the air
    s0 = np.zeros(k)
    T0 = yield from probe(api, s0, 0.6, 0.3)
    D = 0.5
    J = np.zeros((15, k))
    for j in range(k):
        e = s0.copy()
        e[j] += D
        Tj = yield from probe(api, e)
        J[:, j] = (Tj - T0).ravel() / D
    note(api, f"T0 mm {fmt(T0, nd=0)}")
    # choose the opposing finger by the linear model
    best = None
    for pair in ((0, 1), (0, 2)):
        ds, _ = gn_step(T0, J, s0, pair, H_TIGHT, maxstep=3.0)
        Tp = T0 + (J @ ds).reshape(5, 3)
        c = cost(Tp, J, s0 + ds, pair, H_TIGHT)
        note(api, f"pair {pair} lin cost {c:.2e}")
        if best is None or c < best[0]:
            best = (c, pair)
    pair = best[1]
    s_t, T_t, J = yield from solve(api, T0, J, s0, pair, H_TIGHT, 4)
    note(api, f"tight s {np.round(s_t, 2).tolist()} pads mm {fmt(T_t, nd=0)} ap {fmt(T_t[pair[0]] - T_t[pair[1]])}")
    s_p, T_p, J = yield from solve(api, T_t, J, s_t, pair, H_PRE, 2)
    note(api, f"pre s {np.round(s_p, 2).tolist()} ap {fmt(T_p[pair[0]] - T_p[pair[1]])}")
    mid = 0.5 * (T_t[pair[0]] + T_t[pair[1]])
    other = pair[1]

    # ---------------- task phase (iter_5: slide grasp, then a fresh grasp for the lift)
    p0, _ = api.object_pose("pen")
    half = p0[2] - (TABLE + 0.0005)
    bore = p0[:2].copy()                       # pen starts centred in the bore
    UMAX = 1.4                                 # beyond ~1.4x the calibrated squeeze the line stops squeezing
    st = dict(u=0.0, cum=0.0, cmin=0.0, slip=0.0, prev=None, off=np.zeros(2))

    def s_of(u):
        return s_p + u * (s_t - s_p)

    def pen():
        pp, RR = api.object_pose("pen")
        a = RR[:, 2] * (1 if RR[2, 2] >= 0 else -1)
        return pp, a, float(pp[2] - half * a[2] - TABLE)

    def forces():
        f = api.contact_forces("pen")
        return f, min(f[0], f[other])

    def track():
        """Mirror of the scorer's slide count (held = thumb and partner > 0.25 N, tip on table)."""
        pp, a, tip_h = pen()
        f, fmin = forces()
        if fmin > 0.25 and tip_h < 0.0025:
            s = float((pp - api.tips_world()[0]) @ a)
            if st["prev"] is not None:
                st["cum"] += s - st["prev"]
                st["cmin"] = min(st["cmin"], st["cum"])
                st["slip"] = max(st["slip"], st["cum"] - st["cmin"])
            st["prev"] = s
        else:
            st["prev"] = None

    def tracked(gen):
        for _ in gen:
            yield
            track()

    def axis_at(z):
        pp, a, _ = pen()
        return pp + a * (z - pp[2]) / max(a[2], 0.5)

    def centre_step(gain, lim):
        """While held: nudge the palm so the pen axis at pad height sits over the bore centre."""
        _, fmin = forces()
        if fmin < 0.3:
            return
        tw = api.tips_world()
        e = bore - axis_at(0.5 * (tw[0, 2] + tw[other, 2]))[:2]
        st["off"] += np.clip(gain * e, -lim, lim)
        n = np.linalg.norm(st["off"])
        if n > 0.008:
            st["off"] *= 0.008 / n

    def tilt():
        _, a, _ = pen()
        return float(np.degrees(np.arccos(min(1.0, a[2]))))

    def grasp(zg):
        """Open, put the pinch on the pen's actual axis at height zg, close until both pads press."""
        target = axis_at(zg)
        target[2] = zg
        palm_g = api.palm_for(target, mid, R)
        cur, _ = api.palm_pose()
        if cur[2] < palm_g[2] + 0.06:
            yield from tracked(api.move(s=s_p, seconds=0.5))
            cur, _ = api.palm_pose()
            yield from tracked(api.move(xyz=np.r_[cur[:2], palm_g[2] + 0.07], seconds=0.8))
        yield from tracked(api.move(xyz=palm_g + [0, 0, 0.07], R=R, s=s_p, seconds=1.8))
        yield from tracked(api.move(xyz=palm_g, R=R, seconds=1.0))
        yield from tracked(api.hold(0.2))
        U_END, N = 1.6, 100
        n, x = 0, 1.0
        for _ in api.move(s=s_of(U_END), seconds=N / 50):
            yield
            n += 1
            f, fmin = forces()
            if fmin > 0.7 or max(f[0], f[other]) > 4.0:
                x = n / N
                break
        st["u"] = U_END * x * x * (3 - 2 * x)
        yield from tracked(api.hold(0.3))
        st["off"] = np.zeros(2)
        for _ in range(3):        # straighten the pen in the bore
            centre_step(0.8, 0.004)
            yield from tracked(api.move(xyz=palm_g + np.r_[st["off"], 0], seconds=0.2))
        f, _ = forces()
        note(api, f"grasp zg {zg * 1000:.0f} u {st['u']:.2f} F {np.round(f, 2).tolist()} tilt {tilt():.1f} off {fmt(st['off'])} t {api.t:.1f}")
        return palm_g

    def squeeze_to(fmin_goal, du=0.05, tries=8):
        for _ in range(tries):
            f, fmin = forces()
            if fmin >= fmin_goal or st["u"] >= UMAX:
                break
            st["u"] = min(UMAX, st["u"] + du)
            yield from tracked(api.move(s=s_of(st["u"]), seconds=0.12))

    # ---------------- slide strokes: lower the palm, the table pushes the pen up through the pinch
    SLIDE_GOAL = 0.021
    for stroke in range(2):
        pp, a, _ = pen()
        zg = pp[2] + half * a[2] - 0.020
        palm_g = yield from grasp(zg)
        max_low = min(0.042, zg - (TABLE + 0.060 + 0.012))
        lowered, step = 0.0, 0.0015
        slip_start = st["slip"]
        while lowered < max_low and st["slip"] < SLIDE_GOAL:
            f, fmin = forces()
            if f[5] > 0.3:
                note(api, f"pen touches palm F {np.round(f, 2).tolist()}")
                break
            if fmin < 0.5:
                yield from squeeze_to(0.6, du=0.04, tries=2)
            elif fmin > 2.5:
                st["u"] -= 0.03
                yield from tracked(api.move(s=s_of(st["u"]), seconds=0.06))
            centre_step(0.4, 0.0015)
            lowered += step
            yield from tracked(api.move(xyz=palm_g + np.r_[st["off"], -lowered], seconds=0.1))
        yield from tracked(api.hold(0.2))
        f, _ = forces()
        _, a, tip_h = pen()
        note(api, f"stroke {stroke}: slip {st['slip'] * 1000:.1f}mm (+{(st['slip'] - slip_start) * 1000:.1f}) lowered {lowered * 1000:.0f} tip_h {tip_h * 1000:.1f} u {st['u']:.2f} F {np.round(f, 2).tolist()} tilt {tilt():.1f} t {api.t:.1f}")
        if st["slip"] >= 0.018 or api.t > 30:
            break

    # ---------------- fresh grasp lower on the pen, lift out, hold
    for attempt in range(2):
        pp, a, tip_h = pen()
        if tip_h > 0.01 and a[2] < 0.9:
            note(api, f"pen not standing (tip_h {tip_h * 1000:.0f} tilt {tilt():.0f}); giving up")
            break
        zg = pp[2] + half * a[2] - 0.035
        zg = max(zg, TABLE + 0.060 + 0.020)
        yield from grasp(zg)
        yield from squeeze_to(1.0, du=0.05, tries=8)
        yield from tracked(api.hold(0.2))
        xyzL, _ = api.palm_pose()
        off0 = st["off"].copy()
        lift = 0.0
        while lift < 0.13:
            lift += 0.004
            _, _, tip_h = pen()
            if tip_h < 0.07:
                centre_step(0.4, 0.0015)
            yield from tracked(api.move(xyz=xyzL + np.r_[st["off"] - off0, lift], seconds=0.08))
            f, fmin = forces()
            if fmin < 0.6:
                yield from squeeze_to(0.8, du=0.04, tries=2)
            _, _, tip_h = pen()
            if fmin < 0.05 and tip_h < 0.003:
                break
        for _ in range(5):       # settle; keep both pads pressing
            yield from tracked(api.hold(0.1))
            f, fmin = forces()
            if fmin < 0.7:
                yield from squeeze_to(0.9, du=0.04, tries=2)
        f, fmin = forces()
        pp, a, tip_h = pen()
        note(api, f"lift {attempt}: tip_h {tip_h * 1000:.1f} F {np.round(f, 2).tolist()} u {st['u']:.2f} tilt {tilt():.1f} t {api.t:.1f}")
        if tip_h > 0.065 or api.t > 42:
            break
