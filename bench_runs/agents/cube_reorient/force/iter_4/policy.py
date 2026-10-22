"""cube_reorient, force level.

Strategy: two-pad "twirl" gait. Thumb on one cube face, an opposing finger on the
antipodal point of the opposite face (pinch line through the cube centre), both
held by the grip-force servo; the pads are swept rigidly about the palm normal
through the cube centre (api.rotate), then released, and the step is repeated
until the tracked yaw reaches 90 deg. Each step is planned at run time from
api.reach() probes (fingertip IK residuals, cached on a 3 mm grid): which face
pair, which contact offset on the face, which opposing finger, which pad heights
and how far the pads can sweep.
"""
import numpy as np

FN = ("thumb", "index", "middle", "ring", "little")
OPP = ("index", "middle", "ring", "little")
PAIRS = [("thumb", "index"), ("thumb", "middle"), ("thumb", "ring")]
# where each pad may START a twirl (palm angle around the cube, deg; 0 = wrist side, 90 = thumb side, 180 = finger
# side): the thumb tucks in on the wrist / little-finger side, the finger hooks onto the opposite corner from the
# finger side. iter 1-2: these are the grasps that turned the cube without pressing it into the palm.
START = {"thumb": (270, 335), "index": (98, 148), "middle": (135, 190), "ring": (165, 220)}
DIRECTION = -1          # thumb sweeps from the wrist side toward the little-finger side
# rough finger base (MCP / thumb root) positions in palm (y, z) [m], for a collision heuristic:
# the straight line from base to pad must not cross the cube footprint (else the finger hooks over the cube)
BASE = {"thumb": (0.028, 0.030), "index": (0.028, 0.105), "middle": (0.005, 0.105),
        "ring": (-0.018, 0.105), "little": (-0.040, 0.095)}
TOL = 0.003            # reach residual accepted [m]
GRIP_N = 0.7           # pad force during the twirl [N]
PAD = 0.008            # pad centre to surface
INSET = -0.001         # pads stop 1 mm short of the surface; the grip servo closes the gap
STEP = np.radians(7.5)
US = np.arange(-0.012, 0.0121, 0.004)   # tangential contact offsets on a face [m]


def yaw_of(R):
    flat = float(np.abs(R[0]).max())
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2)), flat


def d2(a):            # palm (y, z) unit vector at angle a, measured from -z toward +y
    return np.array([np.sin(a), -np.cos(a)])


def rot2(v, th):      # rotation about palm +x by th, acting on (y, z)
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def face_angles(R):
    k = int(np.argmax(np.abs(R[0])))
    out = []
    for j in range(3):
        if j == k:
            continue
        for sg in (1, -1):
            v = sg * R[1:, j]
            out.append(float(np.arctan2(v[0], -v[1])))
    return out


def run(api):
    st = {"total": 0.0, "last": None}
    logs = []

    def note(s):
        logs.append(f"{api.t:.2f} {s}")

    def upd():
        _, R = api.object_in_palm("cube")
        y, _ = yaw_of(R)
        if st["last"] is None:
            st["last"] = y
        st["total"] += (y - st["last"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        st["last"] = y

    def T(gen):
        try:
            while True:
                next(gen)
                upd()
                yield
        except StopIteration as e:
            return e.value

    cache = {}

    def reach(f, p):
        key = (f,) + tuple(int(v) for v in np.round(np.asarray(p, float) / 0.003))
        if key not in cache:
            cache[key] = float(api.reach(f, np.array(key[1:], float) * 0.003))
        return cache[key]

    def padrel(h, an, u, inset=INSET):
        return (h + PAD - inset) * d2(an) + u * d2(an + np.pi / 2)

    def crosses(f, c, pos, an_now, h):
        """True if the straight line from finger f's base to its pad crosses the cube footprint."""
        if f == "thumb":   # the thumb wraps around the cube without trouble (iter 1); only check fingers
            return False
        b = np.asarray(BASE[f])
        v = pos - b
        L = np.linalg.norm(v)
        e1, e2 = d2(an_now), d2(an_now + np.pi / 2)
        for lam in np.arange(0.0, max(L - 0.012, 0.0), 0.003) / max(L, 1e-9):
            r = b + lam * v - c
            if abs(r @ e1) < h - 0.001 and abs(r @ e2) < h - 0.001:
                return True
        return False

    def sweep(f, c3, h, an, u, s, cap, heights):
        """Largest sweep [rad] finger f can follow holding face `an` at offset u, and the pad height used."""
        best = (-1.0, None)
        c = c3[1:]
        base = padrel(h, an, u)
        ths = list(np.arange(0, cap, STEP)) + [cap]
        a0 = np.degrees(np.arctan2(base[0], -base[1])) % 360
        lo, hi = START[f]
        if not lo <= a0 <= hi:
            return best
        for xh in heights:
            st0 = c + base + 0.010 * d2(an)
            if reach(f, [xh, *st0]) > 0.008 or crosses(f, c, st0, an, h):
                continue
            dm = -1.0
            for th in ths:
                q = c + rot2(base, s * th)
                if reach(f, [xh, *q]) > TOL or crosses(f, c, q, an + s * th, h):
                    break
                dm = th
            if dm > best[0] + 1e-6:
                best = (dm, xh)
        return best

    def plan(c3, h, alphas, s, rem):
        cap = min(rem + np.radians(15), np.radians(100))
        H = [c3[0], c3[0] + 0.007, c3[0] + 0.014]
        memo = {}

        def sw(f, an, u):
            k = (f, round(an, 4), round(u, 4))
            if k not in memo:
                memo[k] = sweep(f, c3, h, an, u, s, cap, H)
            return memo[k]
        best = None
        for an in alphas:
            for fa, fb in PAIRS:
                for ua in US:
                    da, xa = sw(fa, an, ua)
                    if da < np.radians(10):
                        continue
                    for ub in US:
                        if abs(ua - ub) > 0.85 * h:
                            continue
                        db, xb = sw(fb, an + np.pi, ub)
                        dl = min(da, db)
                        if dl < np.radians(10):
                            continue
                        key = (round(min(dl, rem), 4), round(dl, 4), -round(abs(ua - ub), 4), -round(abs(ua) + abs(ub), 4))
                        if best is None or key > best[0]:
                            best = (key, dict(fa=fa, fb=fb, an=an, ua=ua, ub=ub, xa=xa, xb=xb,
                                              delta=min(dl, rem), dmax=dl, s=s))
        return None if best is None else best[1]

    def desc(q):
        return (f"{q['fa']}+{q['fb']} an={np.degrees(q['an']):.0f} u={q['ua'] * 1000:.0f}/{q['ub'] * 1000:.0f} "
                f"x={q['xa'] * 1000:.0f}/{q['xb'] * 1000:.0f} d={np.degrees(q['delta']):.0f}/{np.degrees(q['dmax']):.0f}")

    def simulate(c3, h, alphas, s):
        prog, steps = 0.0, []
        al = list(alphas)
        for _ in range(5):
            rem = np.radians(90) - prog
            if rem < np.radians(5):
                break
            p = plan(c3, h, al, s, rem)
            if p is None:
                break
            steps.append(p)
            prog += p["delta"]
            al = [a + s * p["delta"] for a in al]
        return prog, steps

    def lift_clear(fingers, top):
        """Pads of `fingers` straight up to palm height `top` (above the cube)."""
        if not fingers:
            return
        tt = api.tip_targets()
        up = {g: np.r_[max(top, tt[FN.index(g)][0]), tt[FN.index(g)][1:]] for g in fingers}
        yield from T(api.move_tips(tips=up, seconds=0.35, straight=True))

    def recenter(limit):
        """If the cube has drifted toward the finger bases (palm z > limit), push its finger-side face back
        toward the palm centre with the middle and ring pads side by side (position-controlled push)."""
        p, R = api.object_in_palm("cube")
        if p[2] <= limit:
            return
        k = int(np.argmax(np.abs(R[0])))
        ext = h * sum(abs(R[2, j]) for j in range(3) if j != k)      # cube half-extent along palm z
        ys = {"middle": p[1] + 0.009, "ring": p[1] - 0.009}
        xh = p[0] + 0.004
        zs = p[2] + ext + PAD + 0.010
        zt = 0.066 + ext + PAD - 0.001
        yield from T(api.move_tips(tips={g: np.r_[top, y, zs] for g, y in ys.items()}, seconds=0.6))
        yield from T(api.move_tips(tips={g: np.r_[xh, y, zs] for g, y in ys.items()}, seconds=0.4, straight=True))
        r = yield from T(api.move_tips(tips={g: np.r_[xh, y, zt] for g, y in ys.items()}, seconds=1.0, straight=True))
        yield from T(api.move_tips(tips={g: np.r_[xh, y, zt + 0.012] for g, y in ys.items()}, seconds=0.3, straight=True))
        yield from T(api.move_tips(tips={g: np.r_[top, y, zt + 0.012] for g, y in ys.items()}, seconds=0.3, straight=True))
        yield from T(api.move_joints(q={g: home[4 * FN.index(g):4 * FN.index(g) + 4] for g in ys}, seconds=0.5))
        pe, Re = api.object_in_palm("cube")
        note(f"recenter z {p[2] * 1000:.0f}->{pe[2] * 1000:.0f} ik={r.get('ik_residual_mm')} yaw {np.degrees(-DIRECTION * st['total']):.1f} flat={yaw_of(Re)[1]:.3f}")

    try:
        home = api.q_target().copy()
        yield from T(api.hold(0.4))
        p0, R0 = api.object_in_palm("cube")
        h = float(np.clip(p0[0] - 0.0235, 0.0176, 0.0224))
        top = p0[0] + h + 0.012
        note(f"start cube_mm={np.round(p0 * 1000, 1).tolist()} yaw={np.degrees(yaw_of(R0)[0]):.1f} h_est={h * 1000:.1f}")
        alphas = face_angles(R0)
        sims = {s: simulate(p0, h, alphas, s) for s in (DIRECTION,)}
        for s, (prog, steps) in sims.items():
            note(f"sim s={s}: prog={np.degrees(prog):.0f} " + "; ".join(desc(q) for q in steps))
        s = max(sims, key=lambda k: (round(min(sims[k][0], np.radians(90)), 3), -len(sims[k][1]),
                                     sims[k][1][0]["dmax"] if sims[k][1] else 0))
        note(f"direction s={s}")

        used = set()
        for k in range(7):
            if api.t > 47:
                note("time guard")
                break
            if k > 0:
                pc, _ = api.object_in_palm("cube")
                if pc[2] > 0.084:
                    back = [g for g in used if g in ("middle", "ring")]
                    if back:
                        yield from lift_clear(back, top)
                    yield from recenter(0.084)
            p, R = api.object_in_palm("cube")
            prog = -s * st["total"]
            rem = np.radians(90) - prog
            if abs(rem) < np.radians(4):
                break
            sd = s if rem > 0 else -s
            pl = plan(p, h, face_angles(R), sd, abs(rem))
            if pl is None:
                note(f"step {k}: no plan (prog {np.degrees(prog):.1f}) cube_mm={np.round(p * 1000, 1).tolist()}")
                break
            fa, fb = pl["fa"], pl["fb"]
            # fingers used before but not now: lift clear and go home
            back = [g for g in used if g not in (fa, fb)]
            if back:
                yield from lift_clear(back, top)
                yield from T(api.move_joints(q={g: home[4 * FN.index(g):4 * FN.index(g) + 4] for g in back}, seconds=0.5))
                used -= set(back)
            used |= {fa, fb}
            c, an = p[1:], pl["an"]
            ba, bb = c + padrel(h, an, pl["ua"]), c + padrel(h, an + np.pi, pl["ub"])
            na, nb = d2(an), d2(an + np.pi)
            xa, xb = pl["xa"], pl["xb"]
            hover = {fa: np.r_[max(xa + 0.022, top), ba + 0.018 * na], fb: np.r_[max(xb + 0.022, top), bb + 0.018 * nb]}
            stand = {fa: np.r_[xa, ba + 0.010 * na], fb: np.r_[xb, bb + 0.010 * nb]}
            touch = {fa: np.r_[xa, ba], fb: np.r_[xb, bb]}
            r1 = yield from T(api.move_tips(tips=hover, seconds=0.8))
            yield from T(api.move_tips(tips=stand, seconds=0.4, straight=True))
            r2 = yield from T(api.move_tips(tips=touch, seconds=0.3, straight=True))
            g = yield from T(api.grip("cube", {fa: GRIP_N, fb: GRIP_N}, seconds=0.8))
            y0 = -s * st["total"]
            dl = pl["delta"]
            rr = yield from T(api.rotate("cube", [1, 0, 0], sd * dl, [fa, fb], seconds=max(0.8, dl / np.radians(45) * 1.4)))
            gs = api.grip_state()
            y1 = -s * st["total"]
            yield from T(api.release([fa, fb], "cube", clear=0.010, lift=(0.010, 0, 0), seconds=0.5))
            yield from lift_clear([fa, fb], top)
            yield from T(api.hold(0.2))
            pe, Re = api.object_in_palm("cube")
            note(f"step {k}: sd={sd} {desc(pl)} ik={r1.get('ik_residual_mm')}/{r2.get('ik_residual_mm')} "
                 f"grip={g} rotIK={rr.get('ik_residual_mm')} rotF={rr.get('forces')} gs={gs} "
                 f"yaw {np.degrees(y0):.1f}->{np.degrees(y1):.1f}->{np.degrees(-s * st['total']):.1f} "
                 f"cube_mm={np.round(pe * 1000, 1).tolist()} flat={yaw_of(Re)[1]:.3f}")
        # retreat: pads already above the cube; push them a bit further out from the cube, then home
        if used & {"middle", "ring"}:
            yield from lift_clear(list(used & {"middle", "ring"}), top)
            yield from T(api.move_joints(q={g: home[4 * FN.index(g):4 * FN.index(g) + 4] for g in used & {"middle", "ring"}}, seconds=0.5))
            used -= {"middle", "ring"}
        yield from recenter(0.078)
        pe, _ = api.object_in_palm("cube")
        tt = api.tip_targets()
        out = {}
        for g in used:
            v = tt[FN.index(g)][1:] - pe[1:]
            out[g] = np.r_[tt[FN.index(g)][0], tt[FN.index(g)][1:] + 0.015 * v / (np.linalg.norm(v) + 1e-9)]
        if out:
            yield from T(api.move_tips(tips=out, seconds=0.4, straight=True))
        if "thumb" in used:   # carry the thumb high over to its own (+y) side before going home
            tt = api.tip_targets()
            yield from T(api.move_tips(tips={"thumb": np.r_[top + 0.015, max(tt[0][1], pe[1] + 0.05), tt[0][2]]},
                                       seconds=0.6, straight=True))
        yield from T(api.move_joints(q=home, seconds=1.2))
        yield from T(api.hold(0.4))
        pe, Re = api.object_in_palm("cube")
        note(f"end yaw_rel={np.degrees(-s * st['total']):.1f} cube_mm={np.round(pe * 1000, 1).tolist()} flat={yaw_of(Re)[1]:.3f} "
             f"forces={np.round(api.contact_forces('cube'), 2).tolist()} reach_cache={len(cache)}")
    finally:
        for line in logs:
            api.note(line)
