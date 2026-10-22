import numpy as np

# cube_palm, synergies level (v4).
# Grasp/lift/roll exactly as iter 2-3 (that part is reproducible: palm-up the
# cube always sits pinched by thumb+middle at palm-frame ~(73, 10, 103) mm).
# New transfer: tilt the palm-up hand so the fingers are raised (palm slopes
# toward the wrist, gravity points to the palm centre), then relax the grip
# SLOWLY in small synergy steps along S_GRASP -> open, checking the scorer's
# on-palm condition after every step and freezing the posture as soon as it
# holds (fast opening in iter 2/3 flicked the cube off the hand).
# Then re-close, roll back palm-down, set down at the start, release.

PALM_FACE = 0.0245   # palm face height along palm +x
GAP = 0.003          # palm face to cube top while grasping
CZ = 0.085           # cube centre along the fingers (palm z)
LIFT = 0.17
TABLE = 0.75
TILT = 0.6           # rad, fingers raised while palm-up (normal z = 0.83 > 0.7)
S_GRASP = np.array([0.06, 3.0, 1.53, -1.46, -3.0, -0.77])
S_CAGE = np.array([-0.25, 2.0, 0.73, -2.22, -2.06, -1.38])


def Rzl(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def cube_yaw(R):
    up = int(np.argmax(np.abs(R[2])))
    ax = R[:, (up + 1) % 3]
    ang = np.arctan2(ax[1], ax[0])
    q = np.pi / 2
    return 0.7 + ((ang - 0.7 + q / 2) % q - q / 2)


def fr(api):
    return np.round(api.contact_forces("cube"), 2).tolist()


def on_palm(api, z0):
    """Own (slightly stricter) copy of the scorer's carried-on-open-palm test."""
    f = api.contact_forces("cube")
    _, Rp = api.measured_palm()
    p, _ = api.object_pose("cube")
    pc, _ = api.object_in_palm("cube")
    ok = (Rp[2, 0] > 0.72 and p[2] - z0 > 0.105 and f[5] > 0.03 and f[:5].sum() < 0.4
          and pc[0] < 0.058 and abs(pc[1]) < 0.042)
    return ok, f, pc, p


def run(api):
    p0, R0 = api.object_pose("cube")
    e = float(np.clip(2 * (p0[2] - TABLE), 0.03, 0.05))
    h = e / 2
    yaw = cube_yaw(R0)
    Rd = api.hand_down(yaw)
    s_open = np.asarray(api.open_coeffs, float)
    api.note(f"start cube={np.round(p0, 4).tolist()} edge={e:.4f} yaw={yaw:.3f}")
    c_palm = np.array([PALM_FACE + GAP + h, 0.0, CZ])

    # ---------------- approach open, descend, close through the cage to the grasp
    above = api.palm_for(p0 + np.array([0, 0, 0.07]), c_palm, Rd)
    yield from api.move(above, Rd, s=s_open, seconds=1.8)
    yield from api.move(api.palm_for(p0, c_palm, Rd), Rd, seconds=1.0)
    yield from api.move(s=S_CAGE, seconds=0.6)
    yield from api.move(s=S_GRASP, seconds=0.6)
    yield from api.hold(0.3)
    pc, _ = api.object_in_palm("cube")
    api.note(f"grasped f={fr(api)} cube_in_palm={np.round(pc, 4).tolist()}")

    # ---------------- lift, roll palm-up about the cube centre
    Pc = p0 + np.array([0, 0, LIFT])
    yield from api.move(api.palm_for(Pc, pc, Rd), Rd, seconds=1.2)
    pc, _ = api.object_in_palm("cube")
    for kk in range(1, 5):
        R = Rd @ Rzl(kk * np.pi / 4)
        yield from api.move(api.palm_for(Pc, pc, R), R, seconds=0.5)
    Ru = Rd @ Rzl(np.pi)
    yield from api.hold(0.2)
    pc, _ = api.object_in_palm("cube")
    api.note(f"palm up f={fr(api)} cube_in_palm={np.round(pc, 4).tolist()}")
    Pw, _ = api.object_pose("cube")
    if Pw[2] < p0[2] + 0.08 or np.linalg.norm(pc) > 0.2:
        api.note("cube lost before transfer")
        return

    # ---------------- tilt: fingers up, palm slopes toward the wrist
    Rt = api.hand_up(yaw, TILT)
    yield from api.move(api.palm_for(Pw, pc, Rt), Rt, seconds=0.8)
    yield from api.hold(0.2)
    pc, _ = api.object_in_palm("cube")
    api.note(f"tilted f={fr(api)} cube_in_palm={np.round(pc, 4).tolist()}")

    # ---------------- slow relaxation with feedback; freeze when the palm carries it
    a, on, lost = 0.0, 0, False
    t_start = api.t
    last_note = -1.0
    while api.t - t_start < 14.0:
        ok, f, pcur, p = on_palm(api, p0[2])
        if p[2] < p0[2] + 0.06 or np.linalg.norm(pcur) > 0.2:
            lost = True
            break
        if ok:
            on += 1
            if on >= 80:
                break
            yield from api.hold(0.02)
            continue
        if f.sum() < 1e-3 and a > 0:          # free fall: let it land first
            yield from api.hold(0.02)
            continue
        if a < 1.0:
            a = min(1.0, a + 0.02)
            yield from api.move(s=S_GRASP + a * (s_open - S_GRASP), seconds=0.1)
            yield from api.hold(0.04)
            if api.t - last_note > 0.6:
                last_note = api.t
                api.note(f"relax a={a:.2f} f={np.round(f, 2).tolist()} pc={np.round(pcur * 1000).astype(int).tolist()} on={on}")
        else:
            yield from api.hold(0.02)
    ok, f, pcur, p = on_palm(api, p0[2])
    api.note(f"carry done a={a:.2f} on={on} lost={lost} f={np.round(f, 2).tolist()} pc={np.round(pcur * 1000).astype(int).tolist()}")
    if lost:
        return
    s_rel = S_GRASP + a * (s_open - S_GRASP)

    # ---------------- re-grasp, untilt, roll back palm-down
    yield from api.move(s=S_GRASP, seconds=1.0)
    yield from api.hold(0.3)
    pc, _ = api.object_in_palm("cube")
    api.note(f"regrasp f={fr(api)} cube_in_palm={np.round(pc, 4).tolist()}")
    Pc, _ = api.object_pose("cube")
    yield from api.move(api.palm_for(Pc, pc, Ru), Ru, seconds=0.6)
    for kk in range(3, -1, -1):
        R = Rd @ Rzl(kk * np.pi / 4)
        yield from api.move(api.palm_for(Pc, pc, R), R, seconds=0.5)
    yield from api.hold(0.2)
    pc, _ = api.object_in_palm("cube")
    api.note(f"palm down f={fr(api)} cube_in_palm={np.round(pc, 4).tolist()}")

    # ---------------- set down at the start, release, lift away
    hover = np.array([p0[0], p0[1], p0[2] + 0.04])
    yield from api.move(api.palm_for(hover, pc, Rd), Rd, seconds=1.2)
    pc, _ = api.object_in_palm("cube")
    place = np.array([p0[0], p0[1], p0[2] + 0.004])
    yield from api.move(api.palm_for(place, pc, Rd), Rd, seconds=1.0)
    yield from api.hold(0.2)
    yield from api.move(s=S_CAGE, seconds=0.5)
    yield from api.hold(0.3)
    xyz, _ = api.palm_pose()
    yield from api.move(xyz + np.array([0, 0, 0.12]), Rd, seconds=1.0)
    yield from api.move(s=s_open, seconds=0.5)
    p, R = api.object_pose("cube")
    api.note(f"end cube={np.round(p, 4).tolist()} off_mm={1000 * np.linalg.norm(p[:2] - p0[:2]):.1f} f={fr(api)} s_rel={np.round(s_rel, 2).tolist()}")
