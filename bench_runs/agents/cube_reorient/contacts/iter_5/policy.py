"""cube_reorient, contacts level (iter 5).

Antipodal *offset* pinch: a +y-side finger (index) and a -y-side finger
(ring or little, both hang down beside the cube without crossing it) press
opposite side faces at points shifted along the faces by phi (toward the
trailing corner for the chosen turn sense). api.rotate about the palm normal
through the cube centre then *pushes* the faces (normal force x lever arm)
instead of relying only on pad friction, which lost to palm friction before.
Light squeeze (1.5 mm) so the fingers do not crush the cube into the palm.

A runtime planner (api.reach, the level's own fingertip IK) picks
pair / face / phi / pad height / turn sense to maximise the reachable sweep
toward a net yaw of +-90 deg, then approach -> grasp -> rotate in small
chunks with measured progress -> release -> open -> replan.
alpha = palm-frame angle from +z toward +y; the scorer's yaw has the same sense.
"""
import numpy as np

PAD = 0.008                  # api.pad_offset
SQ = 0.0015                  # squeeze into the face [m]
GOAL = np.radians(90)
PAIRS = (("index", "ring"), ("index", "little"), ("thumb", "ring"), ("thumb", "little"))
FI = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
PHIS = np.radians([36.0, 28.0, 20.0, 13.0])
HS = (0.007, 0.002)          # pad height above the cube centre (palm +x) [m]
STEP = np.radians(5)
EX = np.array([1.0, 0.0, 0.0])
DIAG = []


def yaw_of(R):
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2))


class Track:
    """Unwrapped cube yaw about the palm normal, as the scorer measures it."""

    def __init__(self, api):
        self.api = api
        _, R = api.object_in_palm("cube")
        self.last = yaw_of(R)
        self.total = 0.0

    def update(self):
        _, R = self.api.object_in_palm("cube")
        y = yaw_of(R)
        self.total += (y - self.last + np.pi / 4) % (np.pi / 2) - np.pi / 4
        self.last = y
        return self.total


def tracked(gen, tr):
    """Run a primitive, updating the yaw tracker every control step."""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        tr.update()
        yield


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def cube_state(api):
    p, R = api.object_in_palm("cube")
    k = int(np.argmax(np.abs(R[0])))
    half = float(np.clip(p[0] - 0.0235, 0.0172, 0.0228))
    faces = []
    for j in range(3):
        if j == k:
            continue
        for sg in (1.0, -1.0):
            n = sg * R[:, j]
            n = n - EX * n[0]
            n /= np.linalg.norm(n)
            faces.append((float(np.arctan2(n[1], n[2])), n))
    return p, R, half, faces


def pads_for(p, n, half, phi, sig, h, out):
    """Palm-frame pad centres for A (face n) and B (face -n); contact offset along the
    face by sig*phi (alpha sense); pad centre `out` m outside the face plane."""
    tau = np.cross(n, EX)                       # alpha-increasing tangent of face n
    t = half * np.tan(phi) * sig
    cA = p + half * n + t * tau + h * EX
    cB = p - half * n - t * tau + h * EX
    return cA + n * out, cB - n * out


def rot_about(p, x, ang):
    return p + Rx(ang) @ (x - p)


def plan(api, targets, okg=0.003, oks=0.004):
    p, R, half, faces = cube_state(api)
    Y = TR.total
    best = None
    for al, n in faces:
        if not (np.radians(15) <= al <= np.radians(165)):
            continue
        for A, B in PAIRS:
            for h in HS:
                for phi in PHIS:
                    for T in targets:
                        sig = 1.0 if T > Y else -1.0
                        rem = abs(T - Y)
                        if rem < np.radians(3):
                            continue
                        gA, gB = pads_for(p, n, half, phi, sig, h, PAD - SQ)
                        if api.reach(A, gA) > okg or api.reach(B, gB) > okg:
                            continue
                        aA, aB = pads_for(p, n, half, phi, sig, h, PAD + 0.005)
                        if max(api.reach(A, aA), api.reach(B, aB)) > 0.005:
                            continue
                        lim = min(rem, np.radians(60))
                        ds = list(np.arange(STEP, lim - 1e-6, STEP)) + [lim]
                        D = 0.0
                        for d in ds:
                            ang = -sig * d       # rotate() angle about palm +x
                            if max(api.reach(A, rot_about(p, gA, ang)), api.reach(B, rot_about(p, gB, ang))) > oks:
                                break
                            D = d
                        if D < min(rem, np.radians(8)) - 1e-6:
                            continue
                        left = rem - D
                        key = -np.degrees(left) + 0.15 * np.degrees(phi) + (1.0 if h > 0.004 else 0.0) - (4.0 if A == "thumb" else 0.0)
                        if best is None or key > best["key"]:
                            best = dict(key=key, A=A, B=B, n=n, al=al, h=h, phi=phi, sig=sig, T=T, D=D,
                                        rem=rem, half=half, p=p)
    return best


def goto(api, tr, pads, normals, seconds, straight):
    """Put pad centres at palm-frame points via the contact primitives (object frame)."""
    p, R = api.object_in_palm("cube")
    pts = {f: R.T @ (pads[f] - normals[f] * PAD - p) for f in pads}
    nr = {f: R.T @ normals[f] for f in pads}
    if straight:
        r = yield from tracked(api.grasp("cube", pts, squeeze=0.0, seconds=seconds, normals=nr), tr)
    else:
        r = yield from tracked(api.approach("cube", pts, standoff=0.0, seconds=seconds, normals=nr), tr)
    return r


def execute(api, tr, P):
    A, B, n, sig = P["A"], P["B"], P["n"], P["sig"]
    iA, iB = FI[A], FI[B]
    log = [f"plan {A}+{B} face={np.degrees(P['al']):.0f} phi={np.degrees(P['phi']):.0f} h={P['h'] * 1000:.0f} "
           f"sig={sig:+.0f} T={np.degrees(P['T']):.0f} D={np.degrees(P['D']):.0f} rem={np.degrees(P['rem']):.0f} "
           f"half={P['half'] * 1000:.1f} p={np.round(P['p'] * 1000, 1).tolist()} Y={np.degrees(tr.total):.1f}"]
    p, half, phi, h = P["p"], P["half"], P["phi"], P["h"]
    nrm = {A: n, B: -n}

    def pads(out, up=0.0):
        a, b = pads_for(p, n, half, phi, sig, h + up, out)
        return {A: a, B: b}
    # 1. hover outside the faces (and above if reachable), joint-interpolated
    hv = pads(PAD + 0.012, 0.010)
    if max(api.reach(A, hv[A]), api.reach(B, hv[B])) > 0.004:
        hv = pads(PAD + 0.010)
    if max(api.reach(A, hv[A]), api.reach(B, hv[B])) < 0.006:
        yield from goto(api, tr, hv, nrm, 0.7, False)
    # 2. straight in to just outside, then onto the faces
    yield from goto(api, tr, pads(PAD + 0.004), nrm, 0.4, False)
    r = yield from goto(api, tr, pads(PAD - SQ), nrm, 0.4, True)
    yield from tracked(api.hold(0.2), tr)
    fz = api.contact_forces("cube")
    if fz[iA] < 0.3 or fz[iB] < 0.3:
        yield from tracked(api.squeeze([A, B], "cube", 0.0015, seconds=0.3), tr)
        yield from tracked(api.hold(0.1), tr)
        fz = api.contact_forces("cube")
    log.append(f"grasp ik={r.get('ik_residual_mm')} F={np.round(fz, 1).tolist()} Y={np.degrees(tr.total):.1f}")
    # 3. rotate in chunks, measured
    done, stall, extra = 0.0, 0, False
    T = P["T"]
    while True:
        need = sig * (T - tr.total)
        if need < np.radians(1.5) or done > P["D"] - np.radians(0.5):
            break
        ch = min(np.radians(6), P["D"] - done, need)
        before = tr.total
        r = yield from tracked(api.rotate("cube", [1, 0, 0], -sig * ch, [A, B], seconds=0.35), tr)
        yield from tracked(api.hold(0.08), tr)
        done += ch
        worst = max(r["ik_residual_mm"].values())
        fz = api.contact_forces("cube")
        _, Rn = api.object_in_palm("cube")
        gain = sig * (tr.total - before)
        log.append(f"{np.degrees(tr.total):.0f}/{worst:.0f}/{fz[iA]:.1f},{fz[iB]:.1f},{fz[5]:.1f}")
        if worst > 6 or np.abs(Rn[0]).max() < 0.95:
            log.append("stop:ik/tilt")
            break
        stall = stall + 1 if gain < 0.4 * ch else 0
        if stall >= 2 and not extra:
            extra = True
            yield from tracked(api.squeeze([A, B], "cube", 0.001, seconds=0.25), tr)
            log.append("sq+1")
        elif stall >= 3:
            log.append("stall")
            break
    # 4. release and open
    yield from tracked(api.release([A, B], "cube", clear=0.008, lift=(0.012, 0, 0), seconds=0.6), tr)
    yield from tracked(api.open(0.6), tr)
    yield from tracked(api.hold(0.3), tr)
    p2, R2 = api.object_in_palm("cube")
    log.append(f"rel Y={np.degrees(tr.total):.1f} p={np.round(p2 * 1000, 1).tolist()} flat={np.abs(R2[0]).max():.3f} "
               f"F={np.round(api.contact_forces('cube'), 2).tolist()}")
    DIAG.append(" ".join(log))


TR = None


def run(api):
    global TR
    DIAG.clear()
    yield from api.hold(0.5)
    TR = tr = Track(api)
    try:
        p, R, half, faces = cube_state(api)
        DIAG.append(f"start p={np.round(p * 1000, 1).tolist()} half={half * 1000:.1f} "
                    f"faces={[round(float(np.degrees(a))) for a, _ in faces]}")
        lock, switched = None, False
        for cyc in range(7):
            if api.t > 46:
                DIAG.append("time")
                break
            tr.update()
            Y = tr.total
            if min(abs(GOAL - Y), abs(-GOAL - Y)) < np.radians(5):
                break
            # keep the turn sense once it is established; switch it at most once, early
            if lock is None and abs(Y) > np.radians(10):
                lock = float(np.sign(Y))
            targets = [lock * GOAL] if lock else [GOAL, -GOAL]
            P = plan(api, targets)
            if P is None:
                P = plan(api, targets, okg=0.005, oks=0.006)
            if P is None and lock and not switched and abs(Y) < np.radians(25):
                switched, lock = True, -lock
                P = plan(api, [lock * GOAL])
            if P is None:
                _, _, _, fc = cube_state(api)
                DIAG.append(f"no plan Y={np.degrees(Y):.1f} faces={[round(float(np.degrees(a))) for a, _ in fc]}")
                break
            yield from execute(api, tr, P)
            p, R = api.object_in_palm("cube")
            if np.abs(R[0]).max() < 0.95 or abs(p[1]) > 0.05:
                DIAG.append("cube tipped/off-centre, stop")
                break
        yield from tracked(api.hold(0.3), tr)
        if api.contact_forces("cube")[:5].sum() > 0.2:
            yield from tracked(api.open(0.8), tr)
            yield from tracked(api.hold(0.3), tr)
        p, R = api.object_in_palm("cube")
        DIAG.append(f"final Y={np.degrees(tr.total):.1f} p={np.round(p * 1000, 1).tolist()} "
                    f"F={np.round(api.contact_forces('cube'), 2).tolist()}")
    finally:
        for d in DIAG:
            api.note(d)
