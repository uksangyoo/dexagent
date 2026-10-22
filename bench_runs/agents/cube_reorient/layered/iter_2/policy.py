"""cube_reorient, layered level.

Strategy: twist gait. Two fingers squeeze the cube on opposite side faces and
orbit together about the cube's vertical axis (pure couple, no net force), then
release, lift, and regrasp on the faces that are now best placed; repeat with
yaw feedback until the cube has turned 90 deg. The finger pair, faces, contact
height and twist direction are chosen at runtime from fingertip-IK reach tests
(api.reach). Fallback: single-finger corner pushes.

Layers: fingertips (move_tips) for every finger motion; reach() for planning.
"""
import numpy as np

FN = ("thumb", "index", "middle", "ring", "little")
EX = np.array([1.0, 0.0, 0.0])
PAD = 0.008          # pad centre sits ~8 mm inside the fingertip surface
ABOVE = 0.03         # pre-grasp height above the contact point
STAND = 0.008        # pre-grasp standoff from the face
D = []               # diagnostics, dumped as notes at the end


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def yaw_of(R):
    flat = float(np.abs(R[0]).max())
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2)), flat


def mm(v):
    return np.round(np.asarray(v) * 1000).astype(int).tolist()


def normals(R):
    k = int(np.argmax(np.abs(R[0])))
    out = []
    for j in range(3):
        if j == k:
            continue
        v = R[:, j].copy()
        v[0] = 0.0
        v /= np.linalg.norm(v)
        out += [v, -v]
    return out


class Ctx:
    def __init__(self, api):
        self.api = api
        p, R = api.object_in_palm("cube")
        self.last, self.flat = yaw_of(R)
        self.p = p
        self.total = 0.0
        self.k = 0
        self.trace = []
        self.cache = {}

    @property
    def rot(self):  # rotation about +palm x [deg]; the checker's yaw is minus this
        return -np.degrees(self.total)

    def update(self):
        p, R = self.api.object_in_palm("cube")
        y, flat = yaw_of(R)
        self.total += (y - self.last + np.pi / 4) % (np.pi / 2) - np.pi / 4
        self.last, self.flat, self.p = y, flat, p
        self.k += 1
        if self.k % 15 == 0:
            self.trace.append(f"{self.api.t:.1f}:{self.rot:.0f}/" + ",".join(map(str, mm(p))) + f"/{flat:.2f}")

    def run(self, gen):
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            self.update()
            yield

    def reach(self, f, x):
        key = (f, tuple(np.round(np.asarray(x) * 1000).astype(int)))
        if key not in self.cache:
            self.cache[key] = float(self.api.reach(f, np.asarray(x, float)))
        return self.cache[key]


def probe(ctx):
    ys = np.arange(-0.05, 0.0601, 0.01)
    zs = np.arange(0.01, 0.1301, 0.01)
    for f in FN:
        parts = []
        for xl in (0.03, 0.045, 0.06):
            rows = []
            for z in zs:
                s = ""
                for y in ys:
                    r = ctx.reach(f, [xl, y, z])
                    s += "#" if r < 0.002 else ("+" if r < 0.005 else ".")
                rows.append(s)
            parts.append(f"x{int(round(xl * 1000))}:" + "|".join(rows))
        D.append(f"MAP {f} rows z=10..130/10 cols y=-50..60/10 " + " ".join(parts))


def approach_mode(ctx, f, t, n, a, h):
    """How a pad reaches contact point t on the face with normal n: 'A' from above the cube, 'B' from the side."""
    side = t + STAND * n
    if ctx.reach(f, side) > 0.004:
        return None
    above = side + (a - h + 0.016) * EX
    return "A" if ctx.reach(f, above) < 0.005 else "B"


def plan_twist(ctx, c, a, R, sign, only=None, hs=(0.0, 0.008), degs=range(5, 55, 5)):
    best = None
    stats = {}
    pairs = [(FN[i], FN[j]) for i in range(5) for j in range(i + 1, 5)] if only is None else [only]
    for fa, fb in pairs:
        for n in normals(R):
            for h in hs:
                ta = c + (a + PAD) * n + h * EX
                tb = c - (a + PAD) * n + h * EX
                if ctx.reach(fa, ta) > 0.003 or ctx.reach(fb, tb) > 0.003:
                    continue
                ma, mb = approach_mode(ctx, fa, ta, n, a, h), approach_mode(ctx, fb, tb, -n, a, h)
                if ma is None or mb is None:
                    stats[f"{fa}/{fb}"] = "noapproach"
                    continue
                dmax = 0
                for d in degs:
                    M = Rx(sign * np.radians(d))
                    if ctx.reach(fa, c + M @ (ta - c)) < 0.0035 and ctx.reach(fb, c + M @ (tb - c)) < 0.0035:
                        dmax = d
                    else:
                        break
                stats[f"{fa}/{fb}"] = max(dmax, stats.get(f"{fa}/{fb}", 0) if isinstance(stats.get(f"{fa}/{fb}"), int) else 0)
                score = dmax - 3 * (h > 0) + 2 * (ma == "A") + 2 * (mb == "A")
                if best is None or score > best["score"]:
                    best = dict(score=score, dmax=dmax, fa=fa, fb=fb, n=n, h=h, ma=ma, mb=mb)
    if best is not None:
        best["stats"] = stats
    else:
        D.append(f"PLAN none sign={sign} stats={stats}")
    return best


def twist(ctx, plan, sign, amt, a):
    api = ctx.api
    fa, fb, n, h = plan["fa"], plan["fb"], plan["n"], plan["h"]
    ia, ib = FN.index(fa), FN.index(fb)
    c, _ = api.object_in_palm("cube")
    ta, tb = c + (a + PAD) * n + h * EX, c - (a + PAD) * n + h * EX
    sa, sb = ta + STAND * n, tb - STAND * n
    up = (a - h + 0.016) * EX
    first = {fa: sa + up if plan["ma"] == "A" else sa, fb: sb + up if plan["mb"] == "A" else sb}
    yield from ctx.run(api.move_tips(tips=first, seconds=0.9))
    yield from ctx.run(api.move_tips(tips={fa: sa, fb: sb}, seconds=0.4, straight=True))
    c, _ = api.object_in_palm("cube")
    ta, tb = c + (a + PAD) * n + h * EX, c - (a + PAD) * n + h * EX
    yield from ctx.run(api.move_tips(tips={fa: ta, fb: tb}, seconds=0.3, straight=True))
    sq = 0.0
    for _ in range(6):
        f = api.contact_forces("cube")
        if f[ia] > 0.2 and f[ib] > 0.2:
            break
        sq += 0.001
        yield from ctx.run(api.move_tips(tips={fa: ta - sq * n, fb: tb + sq * n}, seconds=0.08, straight=True))
    sq += 0.0015
    ta, tb = ta - sq * n, tb + sq * n
    yield from ctx.run(api.move_tips(tips={fa: ta, fb: tb}, seconds=0.1, straight=True))
    fg = np.round(api.contact_forces("cube"), 2).tolist()
    r0 = ctx.rot
    nseg = max(1, int(np.ceil(amt / 5)))
    for k in range(1, nseg + 1):
        M = Rx(np.radians(sign * amt * k / nseg))
        yield from ctx.run(api.move_tips(tips={fa: c + M @ (ta - c), fb: c + M @ (tb - c)}, seconds=0.12, straight=True))
    yield from ctx.run(api.hold(0.15))
    ft = np.round(api.contact_forces("cube"), 2).tolist()
    M = Rx(np.radians(sign * amt))
    m = M @ n
    A, B = c + M @ (ta - c) + (sq + 0.005) * m, c + M @ (tb - c) - (sq + 0.005) * m
    yield from ctx.run(api.move_tips(tips={fa: A, fb: B}, seconds=0.25, straight=True))
    yield from ctx.run(api.move_tips(tips={fa: A + up, fb: B + up}, seconds=0.35, straight=True))
    D.append(f"TWIST t={api.t:.1f} {fa}{plan['ma']}/{fb}{plan['mb']} n={np.round(n, 2).tolist()} h={h} dmax={plan['dmax']} cmd={sign * amt:.0f} "
             f"got={ctx.rot - r0:.1f} rot={ctx.rot:.1f} sq={sq * 1000:.1f}mm fgrasp={fg} ftwist={ft} p={mm(ctx.p)} flat={ctx.flat:.3f}")


def plan_push(ctx, c, a, R, sign, c0, hs=(0.0, 0.01)):
    best = None
    for f in FN:
        for n in normals(R):
            t = np.array([0.0, -n[2], n[1]])
            for e in (1, -1):
                r = a * n + e * (a - 0.006) * t
                if np.cross(r, -n)[0] * sign <= 0:
                    continue
                for h in hs:
                    base = c + r + h * EX
                    pts = [base + PAD * n, base + (PAD - 0.012) * n, base + (PAD + STAND) * n,
                           base + (PAD + STAND) * n + ABOVE * EX]
                    if all(ctx.reach(f, x) < 0.003 for x in pts):
                        score = float(np.dot(-n, c0 - c)) * 100 - 0.3 * (h > 0)
                        if best is None or score > best["score"]:
                            best = dict(score=score, f=f, n=n, r=r, h=h)
    return best


def push(ctx, plan, a):
    api = ctx.api
    f, n, r, h = plan["f"], plan["n"], plan["r"], plan["h"]
    c, _ = api.object_in_palm("cube")
    base = c + r + h * EX
    r0 = ctx.rot
    yield from ctx.run(api.move_tips(tips={f: base + (PAD + STAND) * n + ABOVE * EX}, seconds=0.8))
    yield from ctx.run(api.move_tips(tips={f: base + (PAD + STAND) * n}, seconds=0.4, straight=True))
    yield from ctx.run(api.move_tips(tips={f: base + (PAD - 0.012) * n}, seconds=0.8, straight=True))
    yield from ctx.run(api.move_tips(tips={f: base + (PAD + STAND) * n}, seconds=0.3, straight=True))
    yield from ctx.run(api.move_tips(tips={f: base + (PAD + STAND) * n + 0.02 * EX}, seconds=0.3, straight=True))
    D.append(f"PUSH t={api.t:.1f} {f} n={np.round(n, 2).tolist()} r={mm(r)} got={ctx.rot - r0:.1f} rot={ctx.rot:.1f} p={mm(ctx.p)} flat={ctx.flat:.3f}")


def run(api):
    ctx = Ctx(api)
    try:
        yield from ctx.run(api.hold(0.6))
        p0, R0 = api.object_in_palm("cube")
        a = float(np.clip(p0[0] - 0.0238, 0.0172, 0.0228))
        D.append(f"START p={mm(p0)} a={a * 1000:.1f} yaw={np.degrees(yaw_of(R0)[0]):.1f} tips={mm(api.tips())} "
                 f"forces={np.round(api.contact_forces('cube'), 2).tolist()}")
        D.append(f"LIMITS {np.round(api.joint_limits, 2).tolist()} qt={np.round(api.q_target(), 2).tolist()}")
        plans = {s: plan_twist(ctx, p0, a, R0, s) for s in (1, -1)}
        D.append("PLANS " + "; ".join(f"{s}: " + ("none" if p is None else f"{p['fa']}{p['ma']}/{p['fb']}{p['mb']} n={np.round(p['n'], 2).tolist()} h={p['h']} dmax={p['dmax']} stats={p['stats']}")
                                      for s, p in plans.items()))
        ok = {s: p for s, p in plans.items() if p is not None and p["dmax"] >= 10}
        if ok:
            sign = max(ok, key=lambda s: ok[s]["dmax"])
            pair = (ok[sign]["fa"], ok[sign]["fb"])
            pair = tuple(sorted(pair, key=FN.index))
            goal = sign * 90.0
            weak = 0
            for _ in range(10):
                if api.t > 42:
                    break
                rem = goal - ctx.rot
                if abs(rem) < 4 or ctx.flat < 0.95:
                    break
                d = 1 if rem > 0 else -1
                p, R = api.object_in_palm("cube")
                ctx.cache.clear()
                pl = plan_twist(ctx, p, a, R, d, only=pair)
                if pl is None or pl["dmax"] < 10:
                    pl = plan_twist(ctx, p, a, R, d)
                if pl is None or pl["dmax"] < 5:
                    D.append(f"NO TWIST PLAN t={api.t:.1f} d={d} p={mm(p)}")
                    break
                pair = tuple(sorted((pl["fa"], pl["fb"]), key=FN.index))
                amt = min(abs(rem) * 1.1, pl["dmax"])
                r0 = ctx.rot
                yield from twist(ctx, pl, d, amt, a)
                weak = weak + 1 if abs(ctx.rot - r0) < 0.3 * amt else 0
                if weak >= 2:
                    D.append("TWIST INEFFECTIVE -> push mode")
                    ok = None
                    break
        if not ok:
            goal = 90.0 if ctx.rot >= 0 else -90.0
            for _ in range(12):
                if api.t > 42:
                    break
                rem = goal - ctx.rot
                if abs(rem) < 5 or ctx.flat < 0.95:
                    break
                d = 1 if rem > 0 else -1
                p, R = api.object_in_palm("cube")
                ctx.cache.clear()
                pl = plan_push(ctx, p, a, R, d, p0)
                if pl is None:
                    D.append(f"NO PUSH PLAN t={api.t:.1f} d={d} p={mm(p)}")
                    break
                yield from push(ctx, pl, a)
        yield from ctx.run(api.hold(0.4))
        D.append(f"END t={api.t:.1f} rot={ctx.rot:.1f} p={mm(ctx.p)} flat={ctx.flat:.3f} forces={np.round(api.contact_forces('cube'), 2).tolist()}")
    finally:
        tr = ctx.trace
        for i in range(0, len(tr), 45):
            D.append("TRACE t:rot/p_mm/flat " + " ".join(tr[i:i + 45]))
        for line in D[-55:]:
            api.note(line)
