import numpy as np

TABLE = 0.75
CURL = [0.0, 1.6, 1.3, 1.1]           # finger tucked out of the way (pickup)
LCURL = [0.3, 1.4, 1.3, 1.1]          # little finger tucked (cup, flex, ...)
FLAT = [0.0, 0.15, 0.15, 0.15]        # finger nearly straight, a flat support
LFLAT = [0.0, 0.1, 0.15, 0.15]
RAKE = [0.0, 1.0, 0.35, 0.2]          # MCP flex: pushes a cube on the fingers toward the palm
LRAKE = [0.1, 1.0, 0.35, 0.2]


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mm(v):
    return np.round(np.asarray(v, float) * 1000).astype(int).tolist()


def run(api):
    OPEN_THUMB = list(api.OPEN[0:4])
    zhat = np.array([0.0, 0.0, 1.0])
    p0, R0 = api.object_pose("cube")
    h = float(np.clip(p0[2] - TABLE, 0.016, 0.024))          # half edge from resting height
    # palm yaw: the cube face direction closest to the default yaw
    k = int(np.argmax(np.abs(R0[2])))
    a = R0[:, (k + 1) % 3]
    cands = np.arctan2(a[1], a[0]) + np.pi / 2 * np.arange(-4, 5)
    yaw = float(cands[np.argmin(np.abs(cands - api.DEFAULT_YAW))])
    Rg = api.hand_down(yaw)
    api.note(f"start h={h*1000:.1f}mm yaw={yaw:.3f} tipsOPEN={mm(api.tips())}")

    # ---- pinch centre (palm frame): the hand's measured thumb-index pinch
    c = np.array([0.08, 0.04, 0.118])

    # ---- approach and pinch (tips layer for preshape, contacts layer for the grasp)
    pw = p0 + np.array([0, 0, 0.003])
    Pg = api.palm_for(pw, c, Rg)
    yield from api.move_joints(q={"middle": CURL, "ring": CURL, "little": LCURL}, seconds=0.6)
    gap = h + 0.008 + 0.018
    r = yield from api.move_tips(xyz=Pg + np.array([0, 0, 0.06]), R=Rg,
                                 tips={"thumb": c - gap * zhat, "index": c + gap * zhat}, seconds=2.0)
    yield from api.carry(xyz=Pg, seconds=1.2)
    p, R = api.object_pose("cube")
    ax = R.T @ Rg[:, 2]
    j = int(np.argmax(np.abs(ax)))
    aobj = np.zeros(3)
    aobj[j] = np.sign(ax[j])
    up = R.T @ np.array([0, 0, 0.003])
    r = yield from api.grasp("cube", {"thumb": -h * aobj + up, "index": h * aobj + up}, squeeze=0.004,
                             seconds=0.8, normals={"thumb": -aobj, "index": aobj})
    yield from api.hold(0.3)
    f = api.contact_forces("cube")
    rel, _ = api.object_in_palm("cube")
    api.note(f"grasped f={np.round(f, 2).tolist()} rel={mm(rel)} ik={r.get('ik_residual_mm')}")
    if f[0] < 0.05 or f[1] < 0.05:
        yield from api.squeeze(["thumb", "index"], "cube", 0.003)

    # ---- lift and turn palm up (supination via a palm-sideways keyframe), cube kept at a carry point
    yield from api.carry(xyz=Pg + np.array([0, 0, 0.10]), seconds=1.0)
    rel, _ = api.object_in_palm("cube")
    Wc = p0 + np.array([-0.02, -0.05, 0.18])
    Rm = Rg @ rz(np.pi / 2)
    yield from api.carry(xyz=api.palm_for(Wc, rel, Rm), R=Rm, seconds=1.5)
    tilt = 0.3
    Ru = api.hand_up(yaw, tilt)
    yield from api.move_joints(xyz=api.palm_for(Wc, rel, Ru), R=Ru,
                               q={"middle": FLAT, "ring": FLAT, "little": LFLAT}, seconds=1.5)
    yield from api.hold(0.3)
    rel, _ = api.object_in_palm("cube")
    f = api.contact_forces("cube")
    api.note(f"palm up rel={mm(rel)} f={np.round(f, 2).tolist()} z={api.object_pose('cube')[0][2]:.3f}")

    # ---- transfer (contacts layer): translate the held pinch toward the palm centre as far as the
    # fingertip IK allows (searched with api.reach), then release both pads straight off the cube.
    cur = api.tip_targets()
    P, Rp = api.measured_palm()
    g = Rp.T @ np.array([0, 0, -1.0])
    table = []
    best, bs = np.zeros(3), 1e9
    for dx in (0.0, -0.01, -0.015, -0.02, -0.025, -0.03, -0.035):
        for dy in (0.0, -0.01, -0.015, -0.02, -0.025, -0.03):
            for dz in (0.0, -0.01, -0.02, -0.03):
                d = np.array([dx, dy, dz])
                e = max(api.reach("thumb", cur[0] + d), api.reach("index", cur[1] + d))
                if e < 0.003:
                    table.append([int(round(v * 1000)) for v in d])
                    cn = rel + d
                    # first get the cube down onto the palm, then toward y~+10 mm, then toward the wrist
                    s = max(cn[0] - 0.047, 0) + 0.7 * max(cn[1] - 0.015, 0) + 0.3 * max(cn[2] - 0.10, 0)
                    if s < bs - 1e-6:
                        best, bs = d, s
    api.note(f"{len(table)} reachable pinch shifts; with dx<=-25mm: {[t for t in table if t[0] <= -25]}")
    api.note(f"chosen shift {mm(best)} score {bs*1000:.0f}")
    if np.linalg.norm(best) > 0:
        r = yield from api.translate("cube", best, ["thumb", "index"], seconds=1.5)
        yield from api.hold(0.2)
    rel, _ = api.object_in_palm("cube")
    f = api.contact_forces("cube")
    api.note(f"shifted rel={mm(rel)} f={np.round(f, 2).tolist()}")
    # no roll (iter 2: rolling 0.6 rad after the shift threw the cube off the little-finger edge)
    api.note(f"drop {max(rel[0] - 0.047, 0)*1000:.0f}mm")
    # release the index first (it can back off along +z), then unload the thumb, lift its pad off the
    # palm (+x, over the cube) before swinging it open: iter 3 the thumb, still pressing, dragged the cube
    # toward +y off the palm edge when it went straight to OPEN.
    yield from api.release(["index"], "cube", clear=0.015, seconds=0.4)
    yield from api.hold(0.2)
    yield from api.release(["thumb"], "cube", clear=0.01, seconds=0.4)
    yield from api.hold(0.2)
    th = api.tip_targets()[0]
    lift = None
    for d in ([0.03, 0.0, -0.005], [0.03, 0.01, 0.0], [0.025, 0.015, -0.01], [0.02, 0.02, 0.0]):
        if api.reach("thumb", th + np.array(d)) < 0.004:
            lift = np.array(d)
            break
    api.note(f"thumb lift {None if lift is None else mm(lift)} f={np.round(api.contact_forces('cube'), 2).tolist()}")
    if lift is not None:
        yield from api.move_tips(tips={"thumb": th + lift}, seconds=0.5, straight=True)
    yield from api.move_joints(q={"thumb": OPEN_THUMB}, seconds=0.6)
    yield from api.move_joints(q={"index": FLAT}, seconds=0.5)
    yield from api.hold(0.2)
    rel, _ = api.object_in_palm("cube")
    f = api.contact_forces("cube")
    api.note(f"dropped rel={mm(rel)} f={np.round(f, 2).tolist()}")
    yield from api.hold(0.3)
    rel, _ = api.object_in_palm("cube")
    f = api.contact_forces("cube")
    api.note(f"levelled rel={mm(rel)} f={np.round(f, 2).tolist()} tips={mm(api.tips())}")

    def rake():
        yield from api.move_joints(q={"index": RAKE, "middle": RAKE, "ring": RAKE, "little": LRAKE}, seconds=0.7)
        yield from api.hold(0.2)
        yield from api.move_joints(q={"index": FLAT, "middle": FLAT, "ring": FLAT, "little": LFLAT}, seconds=0.7)
        yield from api.hold(0.3)

    if rel[2] > 0.09 or f[5] < 0.03:
        yield from rake()
        rel, _ = api.object_in_palm("cube")
        api.note(f"raked rel={mm(rel)} f={np.round(api.contact_forces('cube'), 2).tolist()}")

    # ---- carry on the open palm, counting the steps that meet the rule
    def on_palm():
        P, Rp = api.measured_palm()
        pc, _ = api.object_pose("cube")
        rl = Rp.T @ (pc - P)
        ff = api.contact_forces("cube")
        return (Rp[2, 0] > 0.72 and pc[2] - p0[2] > 0.105 and ff[5] > 0.03 and ff[:5].sum() < 0.4
                and rl[0] < 0.058 and abs(rl[1]) < 0.043)

    n, raked = 0, False
    for i in range(175):
        yield from api.hold(0.02)
        n += bool(on_palm())
        if n >= 70:
            break
        if i == 50 and n < 10 and not raked:
            raked = True
            yield from rake()
    rel, _ = api.object_in_palm("cube")
    api.note(f"on-palm steps {n} after {i + 1} rel={mm(rel)} f={np.round(api.contact_forces('cube'), 2).tolist()}")

    # ---- set down: lower palm-up, then pronate so the cube slides off the little-finger edge
    Rf = api.hand_up(yaw, 0.0)
    v = Rf @ np.array([0.0, -0.04, rel[2]])     # iter 2-3: cube lands 0-72 mm beyond the -75 mm guess
    Ppre = np.array([p0[0] - v[0], p0[1] - v[1], TABLE + 0.09])
    yield from api.move_joints(xyz=Ppre, R=Rf, q={"index": FLAT, "middle": FLAT, "ring": FLAT,
                                                   "little": [0.0, 0.0, 0.1, 0.1], "thumb": OPEN_THUMB},
                               seconds=2.5)
    yield from api.hold(0.3)
    phi = 1.15
    Rp = Rg @ rz(np.pi - phi)
    yield from api.carry(xyz=Ppre, R=Rp, seconds=3.0)
    yield from api.hold(0.4)
    pc, Rc = api.object_pose("cube")
    api.note(f"poured cube={mm(pc)} f={np.round(api.contact_forces('cube'), 2).tolist()}")
    Paway = Ppre + 0.05 * Rp[:, 1] + np.array([0, 0, 0.12])
    yield from api.carry(xyz=Paway, seconds=1.0)
    yield from api.hold(0.6)
    pc, Rc = api.object_pose("cube")
    api.note(f"final cube off={mm(pc[:2] - p0[:2])} dz={(pc[2]-p0[2])*1000:.1f} upz={np.abs(Rc[2]).max():.3f}")
