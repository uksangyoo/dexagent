"""cap_twist, synergies level: "scratch" gait on the lid top with the little pad.

Palm held down over the jar. The flexion synergy s[0] sweeps the index..little
pads in a downward arc toward the wrist (-z palm). The lid centre is placed
16 mm on the far little-finger side (-y palm) of the little pad, so the pad
dragging toward the wrist turns the lid counter-clockwise from above (the
other fingers sit beyond the lid rim). The palm height puts the lid top a few
mm above the lowest point of the little pad's arc, so each sweep presses in,
drags, and leaves again. Between sweeps (lid at rest) the palm lifts, the
fingers re-extend, and the palm comes back down to the same pose.
"""
import numpy as np

GOAL = np.radians(400.0)     # stop once the net turn exceeds this
YAW = 0.7
R_OFF = 0.008                # pad-centre site to pad surface [m]
LIFT = 0.016                 # palm lift for the return sweep [m]
LEVER = 0.016                # little pad offset from the lid axis [m]
DEPTH = 0.007                # commanded arc depth below the lid top [m]


def _interp(tg, T, t):
    out = np.empty((5, 3))
    for f in range(5):
        for c in range(3):
            out[f, c] = np.interp(t, tg, T[:, f, c])
    return out


def _wait_still(api, max_s=0.6):
    prev = api.object_joint("lid_twist")
    calm = 0
    for _ in range(int(max_s * 50)):
        yield from api.hold(0.02)
        a = api.object_joint("lid_twist")
        calm = calm + 1 if abs(np.degrees(a - prev)) < 0.03 else 0
        prev = a
        if calm >= 4:
            return


def run(api):
    R = api.hand_down(YAW)
    base = np.zeros(api.k)
    summary = []

    def s_of(t):
        s = base.copy()
        s[0] = t
        return s

    # ---------------- calibration in free air (palm at start, far from jar)
    tg = np.arange(-3.0, 2.01, 0.5)
    T = []
    yield from api.move(s=s_of(tg[0]), seconds=0.5)
    for t in tg:
        yield from api.move(s=s_of(t), seconds=0.12)
        yield from api.hold(0.18)
        T.append(api.tips())
    T = np.array(T)
    fine = np.linspace(tg[0], tg[-1], 501)
    X = np.array([_interp(tg, T, t) for t in fine])
    kpk = {f: int(np.argmax(X[:, f, 0])) for f in range(5)}
    peak = {f: X[kpk[f], f] for f in range(5)}
    summary.append("peaks_mm " + " ".join(f"{n}:{np.round(peak[f] * 1000).astype(int).tolist()}@t{fine[kpk[f]]:.2f}"
                                          for f, n in enumerate("TIMRL")))

    L = 4
    y_c = peak[L][1] - LEVER
    z_c = float(peak[L][2])
    x_peak = float(peak[L][0])
    others = [f for f in range(0, 4)]

    def stroke_range(depth):
        site_plane = x_peak - depth
        iA = kpk[L]
        while iA > 0 and X[iA, L, 0] >= site_plane - 0.005:
            iA -= 1
        iB = kpk[L]
        while iB < len(fine) - 1 and X[iB, L, 0] >= site_plane - 0.003:
            iB += 1
        return float(fine[iA]), float(fine[iB]), iB == len(fine) - 1

    p_jar, _ = api.object_pose("jar")
    lid_top = p_jar + np.array([0, 0, 0.123])

    def palm_at(lift=0.0):
        P = np.array([x_peak - DEPTH + R_OFF, y_c, z_c])
        return api.palm_for(lid_top, P, R) + np.array([0, 0, lift])

    tA, tB, noexit = stroke_range(DEPTH)
    pk = X[kpk[L]]
    summary.append(f"stroke t {tA:.2f}->{tB:.2f} noexit={noexit}; others rel lid (a_y, b_z, dx below top) mm: " + " ".join(
        f"{f}:{[round((pk[f][1] - y_c) * 1000), round((pk[f][2] - z_c) * 1000), round((pk[f][0] - x_peak + DEPTH - R_OFF) * 1000)]}"
        for f in others))
    api.note(summary[-2]); api.note(summary[-1])

    # ---------------- approach
    yield from api.move(s=s_of(tA), seconds=0.4)
    yield from api.move(xyz=palm_at(0.06), R=R, seconds=1.3)
    yield from api.move(xyz=palm_at(LIFT), R=R, seconds=0.5)
    yield from api.move(xyz=palm_at(), R=R, seconds=0.4)
    yield from api.hold(0.15)

    a0 = api.object_joint("lid_twist")
    n = 0
    shifted = False
    log = []
    while api.t < 55.0:
        n += 1
        th0 = api.object_joint("lid_twist")
        fmax = np.zeros(6)
        for k in range(3):
            yield from api.move(s=s_of(tA + (tB - tA) * (k + 1) / 3), seconds=0.3)
            fmax = np.maximum(fmax, api.contact_forces("lid"))
        yield from _wait_still(api)
        th1 = api.object_joint("lid_twist")
        d = np.degrees(th1 - th0)
        log.append(f"{d:.0f}/{fmax[4]:.0f}N")
        if th1 - a0 >= GOAL:
            break
        if d < -5 and not shifted:     # pad on the wrong side of the axis: move the lid centre further out
            shifted = True
            y_c -= 0.006
            api.note(f"stroke {n} dtheta {d:.1f}: shift lid centre 6 mm")
        # return: lift, re-extend, lower
        th2 = api.object_joint("lid_twist")
        yield from api.move(xyz=palm_at(LIFT), R=R, seconds=0.25)
        yield from api.move(s=s_of(tA), seconds=0.35)
        yield from api.move(xyz=palm_at(), R=R, seconds=0.3)
        yield from api.hold(0.06)
        dr = np.degrees(api.object_joint("lid_twist") - th2)
        if abs(dr) > 1:
            log.append(f"ret{dr:.0f}")

    api.note("strokes dtheta/littleN: " + " ".join(log))
    api.note(f"done: n={n} total={np.degrees(api.object_joint('lid_twist') - a0):.1f}deg")
    yield from api.move(xyz=palm_at(0.04), R=R, seconds=0.5)
    yield from api.hold(0.2)
    for s in summary:
        api.note(s)
