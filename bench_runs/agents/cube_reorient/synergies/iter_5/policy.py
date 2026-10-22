import numpy as np
from scipy.optimize import lsq_linear

# Iteration 5: PIP-led far-edge pulls by one finger set at a time; yaw direction adopted from the first clear rotation, set switched when it stalls or turns the wrong way.
# The fingers on one side of the cube (index+middle on +y, or
# off-centre pull turns the cube about the palm normal. The finger posture is specified in joint
# space (curl level u of the pushing fingers, all others open) and mapped to synergy coefficients
# by bounded weighted least squares on api.mean/api.basis (helper `Syn.solve`). Closed loop on the
# tracked yaw: ramp u slowly, stop at ~90 deg, retract along the same path, open.

OPEN = np.array([0.0, -0.3, 0.2, 0.2, 0, .25, .25, .25, 0, .25, .25, .25, 0, .25, .25, .25, 0.2, 0, .25, .25])
MCP = {1: 5, 2: 9, 3: 13, 4: 17}
FY = {1: 33.0, 2: 12.0, 3: -19.0, 4: -52.0}   # rough finger lateral positions (palm y, mm)


def yaw_of(R):
    flat = float(np.abs(R[0]).max())
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2)), flat


def r1(x):
    return np.round(np.asarray(x, float), 1).tolist()


class Syn:
    """Joint-space posture -> synergy coefficients (bounded weighted least squares)."""

    def __init__(self, api, amp=4.0):
        self.M = np.asarray(api.mean, float)
        self.B = np.asarray(api.basis, float)
        self.k = self.B.shape[1]
        self.so = np.asarray(api.open_coeffs, float)
        self.amp = amp
        hi = np.full(20, -9.0)
        lo = np.full(20, 9.0)
        for j in range(self.k):
            for sg in (-1, 1):
                e = np.zeros(self.k)
                e[j] = 60 * sg
                q = np.asarray(api.posture(e), float)
                hi = np.maximum(hi, q)
                lo = np.minimum(lo, q)
        self.lo, self.hi = lo, hi

    def solve(self, qd, w, s_prev=None, reg=0.05):
        qd = np.clip(qd, self.lo + 0.02, self.hi - 0.02)
        W = np.asarray(w, float)
        A = [W[:, None] * self.B, reg * np.eye(self.k)]
        b = [W * (qd - self.M), reg * self.so]
        if s_prev is not None:
            A.append(0.2 * np.eye(self.k))
            b.append(0.2 * s_prev)
        r = lsq_linear(np.vstack(A), np.concatenate(b), bounds=(-self.amp, self.amp))
        return r.x


def hook_target(fingers, u, side):
    """Joint targets for a far-edge pull by the listed fingers, others open.
    u 0..0.3: PIP/DIP close to a hook (1.6/1.3) with MCP low; u 0.3..1.3: MCP 0.6 -> 1.9,
    which drags the hooked tips/middle phalanges toward the wrist (posture class that pulled the
    cube 28 mm in iter 2)."""
    q = OPEN.copy()
    w = np.full(20, 1.0)
    w[:4] = 0.8
    a = min(u, 0.3) / 0.3
    b = max(0.0, u - 0.3)
    for f in fingers:
        m = MCP[f]
        q[m] = 0.25 + 0.35 * a + 1.3 * b
        q[m + 1] = 0.25 + 1.35 * a
        q[m + 2] = 0.25 + 1.05 * a
        w[m:m + 3] = 4.0
        w[m - 1] = 2.0
    if side > 0:
        q[8] = -0.4      # middle spread toward +y
    for f in (1, 2, 3, 4):
        if f not in fingers:
            w[MCP[f]] = 2.5
    return q, w


def run(api):
    syn = Syn(api)
    p, R = api.object_in_palm("cube")
    p0 = p * 1000
    last, flat = yaw_of(R)
    st = {"total": 0.0, "last": last, "flat": flat, "p": p0}
    trace = []

    def update():
        p, R = api.object_in_palm("cube")
        y, f = yaw_of(R)
        st["total"] += (y - st["last"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        st["last"], st["flat"], st["p"] = y, f, p * 1000
        st["F"] = api.contact_forces("cube")

    s = np.asarray(api.coeffs(), float)
    s_open = syn.solve(OPEN, np.ones(20))
    # settle
    yield from api.move(s=s_open, seconds=0.6)
    yield from api.hold(0.3)
    update()
    base = st["total"]
    SETS = {"IM": ((1, 2), 1.0), "RL": ((3, 4), -1.0)}
    name = "IM" if st["p"][1] <= 9.0 else "RL"
    gain = {"IM": None, "RL": None}   # last yaw gain (in the chosen direction) per finger set
    sign = 0.0                        # yaw direction, adopted from the first clear rotation
    target = 87.0
    s = s_open.copy()
    u = 0.0
    k = 0
    pushes = 0
    events = []
    done = False
    while api.t < 51 and pushes < 8 and not done:
        pushes += 1
        fingers, side = SETS[name]
        y_start = np.degrees(st["total"] - base)
        events.append(f"push{pushes} set={name} sign={sign} t={api.t:.1f} cube={r1(st['p'])}")
        umax = 1.3
        contact_u = None
        wrong = False
        while api.t < 51:
            yaw = np.degrees(st["total"] - base)
            if sign == 0 and abs(yaw) > 6:
                sign = float(np.sign(yaw))
                events.append(f"adopt sign={sign} t={api.t:.1f}")
            F = st["F"]
            fp = float(sum(F[f] for f in fingers))
            if sign != 0 and sign * yaw >= target:
                done = True
                break
            if st["flat"] < 0.85 or st["p"][2] < 12 or abs(st["p"][1]) > 40:
                events.append(f"guard t={api.t:.1f} flat={st['flat']:.3f} cube={r1(st['p'])}")
                done = True
                break
            if contact_u is None and fp > 0.3:
                contact_u = u
            if sign != 0 and sign * (yaw - y_start) < -8:
                events.append(f"wrong-way t={api.t:.1f} yaw={yaw:.1f}")
                wrong = True
                break
            du = 0.012 if contact_u is None else (0.008 if fp < 4 else 0.004)
            u = min(umax, u + du)
            if u >= umax and fp < 0.2:
                break
            q, w = hook_target(fingers, u, side)
            s = syn.solve(q, w, s)
            yield from api.move(s=s, seconds=0.02)
            update()
            k += 1
            if k % 25 == 0:
                trace.append(f"{api.t:.1f}|{name}|u{u:.2f}|y{np.degrees(st['total'] - base):.1f}|c{r1(st['p'])}|f{st['flat']:.3f}|F{r1(st['F'])}")
                if u >= umax and len(trace) > 3 and all(f"u{umax:.2f}" in tr for tr in trace[-3:]):
                    break
        yaw = np.degrees(st["total"] - base)
        events.append(f"end push{pushes} t={api.t:.1f} u={u:.2f} yaw={yaw:.1f} cube={r1(st['p'])} F={r1(st['F'])}")
        # retract along the same path
        while u > 0.05:
            u = max(0.0, u - 0.025)
            q, w = hook_target(fingers, u, side)
            s = syn.solve(q, w, s)
            yield from api.move(s=s, seconds=0.02)
            update()
        if done:
            break
        if sign != 0:
            gain[name] = sign * (np.degrees(st["total"] - base) - y_start)
        other = "RL" if name == "IM" else "IM"
        if wrong or (gain[name] is not None and gain[name] < 3 and (gain[other] is None or gain[other] > gain[name])):
            name = other
        u = 0.0
    yield from api.move(s=s_open, seconds=0.8)
    yield from api.hold(0.6)
    update()
    events.append(f"final t={api.t:.1f} yaw={np.degrees(st['total'] - base):.1f} cube={r1(st['p'])} "
                  f"flat={st['flat']:.3f} F={r1(st['F'])}")
    api.note("LIM lo=" + str(r1(syn.lo)) + " hi=" + str(r1(syn.hi)) + " so=" + str(r1(syn.so)) + " p0=" + str(r1(p0)))
    for e in events:
        api.note("EV " + e)
    for i in range(0, len(trace), 8):
        api.note("TR " + " ; ".join(trace[i:i + 8]))
