"""cube_pinch, force level.

Plan: thumb-index pinch along the palm's finger axis (palm z), squared to the
cube faces by choosing the palm yaw = cube yaw (mod 90 deg) nearest 0.7.
Pads are placed by fingertip IK (move_tips), the grip is held by the force
servo (api.grip), the arm lifts, turns the palm ~40 deg about the vertical axis
through the cube centre in 4 slices, carries the cube back over its start xy
with closed-loop palm corrections, lowers it, ungrips, opens and lifts away.
"""
import numpy as np

TABLE = 0.75
PAD = 0.008                         # pad centre to fingertip surface
PX, PY, PC = 0.080, 0.040, 0.118    # nominal thumb-index pinch centre (palm frame), axis along palm z
GRIP_N = 1.5
TURN = np.radians(40)


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def run(api):
    p0, R0 = api.object_pose("cube")
    p0 = p0.copy()
    yaw0 = _yaw(R0)
    half = float(p0[2] - TABLE)
    st = {"turn": 0.0, "lift": 0.0}

    def upd():
        p, R = api.object_pose("cube")
        lift = p[2] - p0[2]
        st["lift"] = max(st["lift"], lift)
        if lift > 0.062:
            st["turn"] = max(st["turn"], abs(_wrap90(_yaw(R) - yaw0)))

    def track(gen):
        r = None
        try:
            while True:
                next(gen)
                upd()
                yield
        except StopIteration as e:
            r = e.value
        return r

    def fmt(v):
        return np.round(np.asarray(v, float) * 1000, 1).tolist()

    api.note(f"cube p0={fmt(p0)}mm edge={2000 * half:.1f}mm yaw0={np.degrees(yaw0):.1f}")

    # Open hand first (good IK seed), palm high above the cube.
    p, R = api.object_pose("cube")
    psi = _yaw(R) + round((0.7 - _yaw(R)) / (np.pi / 2)) * (np.pi / 2)
    Rh = api.hand_down(psi)
    hp = max(half + 0.003, 0.022)   # pinch height above the table
    G = np.array([p[0], p[1], TABLE + hp])
    yield from track(api.move_joints(api.palm_for(G + [0, 0, 0.10], [PX, PY, PC], Rh), Rh, api.OPEN, seconds=2.0))
    yield from track(api.tuck(["middle", "ring", "little"], seconds=0.6))

    # Choose the pinch centre along palm z that leaves the widest reachable aperture.
    best = None
    for c in (0.118, 0.113, 0.123, 0.108, 0.128, 0.133):
        lim = {}
        for f, sgn in (("thumb", -1), ("index", +1)):
            lim[f] = 0.0
            for hw in np.arange(half + PAD + 0.012, half + PAD - 0.004, -0.001):
                if api.reach(f, [PX, PY, c + sgn * hw]) < 0.0015:
                    lim[f] = hw
                    break
        m = min(lim.values())
        if best is None or m > best[1] + 1e-4:
            best = (c, m, lim)
    c, hwmax, lim = best
    hw_clear = max(min(half + PAD + 0.006, hwmax), half + PAD + 0.001)
    hw_grip = half + PAD - 0.0015
    api.note(f"pinch centre z={c * 1000:.0f}mm hw_max={fmt([lim['thumb'], lim['index']])} clear={hw_clear * 1000:.1f} grip={hw_grip * 1000:.1f} half={half * 1000:.1f}")
    pc = np.array([PX, PY, c])

    def tips(hw):
        return {"thumb": [PX, PY, c - hw], "index": [PX, PY, c + hw]}

    attempt = 0
    while True:
        attempt += 1
        p, R = api.object_pose("cube")
        psi = _yaw(R) + round((0.7 - _yaw(R)) / (np.pi / 2)) * (np.pi / 2)
        Rh = api.hand_down(psi)
        G = np.array([p[0], p[1], TABLE + hp])
        P = api.palm_for(G, pc, Rh)
        r = yield from track(api.move_tips(P + [0, 0, 0.06], Rh, tips(hw_clear), seconds=1.2))
        api.note(f"attempt {attempt}: pre-pinch residuals {r.get('ik_residual_mm') if r else None}")
        yield from track(api.hold(0.3))
        # Closed-loop palm correction: put the measured palm where it should be.
        for _ in range(2):
            Pm, _R = api.measured_palm()
            Pc, _R2 = api.palm_pose()
            Pd = P + [0, 0, 0.06]
            yield from track(api.carry(Pc + (Pd - Pm), Rh, seconds=0.4))
        Pm, _R = api.measured_palm()
        Pc, _R2 = api.palm_pose()
        yield from track(api.carry(P + (Pc - Pm), Rh, seconds=1.5))
        yield from track(api.hold(0.2))
        pcube, _ = api.object_in_palm("cube")
        api.note(f"cube in palm before close {fmt(pcube)} want {fmt([PX, PY, c])} forces {np.round(api.contact_forces('cube'), 2).tolist()}")
        # Close: straight pads onto the faces, then force servo.
        yield from track(api.move_tips(tips=tips(hw_grip), seconds=0.6, straight=True))
        g = yield from track(api.grip("cube", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.0))
        api.note(f"grip {g} state {api.grip_state()}")
        # Lift.
        Pc, _ = api.palm_pose()
        yield from track(api.carry(Pc + [0, 0, 0.125], Rh, seconds=1.5))
        yield from track(api.hold(0.3))
        pw, _ = api.object_pose("cube")
        api.note(f"after lift cube z rise {1000 * (pw[2] - p0[2]):.1f}mm grip {api.grip_state()} forces {np.round(api.contact_forces('cube'), 2).tolist()}")
        if pw[2] - p0[2] > 0.07 or attempt >= 3:
            break
        # Failed grasp: let go, back off, retry from the cube's current pose.
        yield from track(api.ungrip())
        yield from track(api.move_tips(tips=tips(hw_clear), seconds=0.5, straight=True))
        Pc, _ = api.palm_pose()
        yield from track(api.carry(Pc + [0, 0, 0.04], Rh, seconds=0.6))

    # Turn about the vertical axis through the cube centre, in slices.
    p, R = api.object_pose("cube")
    sgn = 1.0 if psi <= 0.75 else -1.0
    Rcur = Rh
    n = 4
    for k in range(n):
        pcube, _ = api.object_in_palm("cube")
        pw, _ = api.object_pose("cube")
        Pm, _R = api.measured_palm()
        Pc, _R2 = api.palm_pose()
        Rn = _Rz(sgn * TURN / n) @ Rcur
        Pn = pw - Rn @ pcube + (Pc - Pm)
        yield from track(api.carry(Pn, Rn, seconds=0.7))
        Rcur = Rn
    yield from track(api.hold(0.3))
    api.note(f"after turn: tracked turn {np.degrees(st['turn']):.1f}deg lift {1000 * st['lift']:.1f}mm grip {api.grip_state()}")
    if st["turn"] < np.radians(30):
        pcube, _ = api.object_in_palm("cube")
        pw, _ = api.object_pose("cube")
        Rn = _Rz(sgn * np.radians(4)) @ Rcur
        yield from track(api.carry(pw - Rn @ pcube, Rn, seconds=0.5))
        Rcur = Rn
        api.note(f"extra turn: {np.degrees(st['turn']):.1f}deg")

    # Carry back over the start, with closed-loop correction on the cube position.
    def place_at(target, seconds):
        pcube, _ = api.object_in_palm("cube")
        Pm, Rm = api.measured_palm()
        Pc, _ = api.palm_pose()
        cube_now = Pm + Rm @ pcube
        yield from track(api.carry(Pc + (target - cube_now), Rcur, seconds=seconds))

    above = p0 + [0, 0, 0.03]
    yield from place_at(above, 1.5)
    yield from track(api.hold(0.2))
    for _ in range(2):
        yield from place_at(above, 0.4)
        yield from track(api.hold(0.1))
    pw, _ = api.object_pose("cube")
    api.note(f"above start: cube err {fmt(pw - above)}")
    yield from place_at(p0 + [0, 0, 0.002], 1.5)
    yield from track(api.hold(0.3))
    pw, _ = api.object_pose("cube")
    api.note(f"placed: cube err {fmt(pw - p0)} grip {api.grip_state()}")

    # Release.
    yield from track(api.ungrip())
    yield from track(api.move_tips(tips=tips(hw_clear), seconds=0.6, straight=True))
    yield from track(api.hold(0.2))
    Pc, _ = api.palm_pose()
    yield from track(api.carry(Pc + [0, 0, 0.08], Rcur, seconds=1.0))
    yield from track(api.hold(0.3))
    pw, Rw = api.object_pose("cube")
    api.note(f"final: cube err {fmt(pw - p0)} Rzz {Rw[2, 2]:.3f} tracked turn {np.degrees(st['turn']):.1f} lift {1000 * st['lift']:.1f}")
