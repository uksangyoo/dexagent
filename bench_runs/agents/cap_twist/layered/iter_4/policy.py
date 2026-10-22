"""cap_twist, layered level.

Strategy: palm held fixed over the lid (palm down). Two "worker" fingers
(default index + ring, on either side of the lid axis) press their pads down
on the lid's TOP face (the hinge carries the normal load, so no opposition is
needed) and stroke along arcs about the jar axis (contacts layer `rotate`),
then lift, swing back in the air and press again (fingertips layer). Closed
loop on the lid angle until > 360 deg.  The palm placement, stroke radius and
stroke angle are chosen at run time by an IK-feasibility search with
`api.reach` (fingertip layer IK), so no offline kinematics is used.
"""
import numpy as np

LID_TOP = 0.123            # lid top face above the jar base [m]
GOAL = 2 * np.pi + 0.30    # target net turn [rad]
T_END = 54.0               # stop stroking after this sim time
LIFT = 0.016               # pad lift between strokes [m]
NAMES = ("thumb", "index", "middle", "ring", "little")


def run(api):
    notes = []
    try:
        yield from _run(api, notes)
    finally:
        for s in notes[-90:]:
            api.note(s)


def _run(api, notes):
    N = notes.append
    pj, _ = api.object_pose("jar")
    T = pj + np.array([0.0, 0.0, LID_TOP])      # lid top centre, world
    a0 = api.object_joint("lid_twist")
    lim = api.joint_limits
    N(f"jar {np.round(pj, 4).tolist()} lid0 {a0:.3f}")
    N("limits lo " + str(np.round(lim[:, 0], 2).tolist()))
    N("limits hi " + str(np.round(lim[:, 1], 2).tolist()))

    # ---------------------------------------------------------- 1. flat hand: finger lanes, thumb out of the way
    flat = api.OPEN.copy()
    for i in (1, 2, 3, 4):
        flat[4 * i:4 * i + 4] = np.clip([0, 0, 0, 0], lim[4 * i:4 * i + 4, 0], lim[4 * i:4 * i + 4, 1])
    best_th, best_x = None, 1e9
    for cand in (api.OPEN[0:4], np.zeros(4), np.array([0.0, -0.6, 0.0, 0.0]), np.array([0.3, -0.3, 0.0, 0.0])):
        q = flat.copy()
        q[0:4] = np.clip(cand, lim[0:4, 0], lim[0:4, 1])
        yield from api.move_joints(q=q, seconds=0.1)
        tt = api.tip_targets()
        if tt[0][0] < best_x:
            best_x, best_th = tt[0][0], q[0:4].copy()
    flat[0:4] = best_th
    yield from api.move_joints(q=flat, seconds=0.5)
    tt = api.tip_targets()
    N("flat tips " + str(np.round(tt * 1000).astype(int).tolist()))
    lanes = tt[:, 1]

    # ---------------------------------------------------------- 2. preshape (IK seed) and feasibility search
    pre = flat.copy()
    for i in (1, 2, 3):
        pre[4 * i:4 * i + 4] = np.clip([0, 0.7, 0.6, 0.4], lim[4 * i:4 * i + 4, 0], lim[4 * i:4 * i + 4, 1])
    yield from api.move_joints(q=pre, seconds=0.5)

    pairs = []
    for (a, b, c) in ((1, 3, None), (2, 3, None), (1, 2, None)):
        yc = lanes[2] if (a, b) == (1, 3) else 0.5 * (lanes[a] + lanes[b])
        rc = float(np.clip(0.5 * abs(lanes[a] - lanes[b]), 0.015, 0.024))
        pairs.append(((a, b), yc, rc))

    def pad(phi, x, yc, zc, rc):
        return np.array([x, yc + rc * np.sin(phi), zc + rc * np.cos(phi)])

    def ok(f, p, tol=0.0015):
        return api.reach(f, p) < tol

    plan = None
    for (workers, yc, rc) in pairs:
        phic = {f: (np.pi / 2 if lanes[f] > yc else -np.pi / 2) for f in workers}
        cand = []
        for xp in np.arange(0.050, 0.1001, 0.01):
            for zc in np.arange(0.05, 0.2001, 0.01):
                if all(ok(f, pad(phic[f], xp + 0.006, yc, zc, rc)) and ok(f, pad(phic[f], xp - LIFT, yc, zc, rc))
                       for f in workers):
                    cand.append((abs(xp - 0.075), xp, zc))
        cand.sort()
        N(f"workers {workers} yc {yc*1000:.1f} rc {rc*1000:.1f}: {len(cand)} centre-feasible (xp,zc): "
          + str([(round(c[1]*1000), round(c[2]*1000)) for c in cand[:12]]))
        for alpha in np.radians([90, 75, 60, 45, 30]):
            for _, xp, zc in cand:
                if all(ok(f, pad(phic[f] + s * alpha / 2, xp + 0.006, yc, zc, rc))
                       for f in workers for s in (-1, 1)):
                    plan = dict(workers=workers, yc=yc, rc=rc, xp=xp, zc=zc, alpha=alpha, phic=phic)
                    break
            if plan:
                break
        if plan:
            break
    if plan is None:
        N("NO FEASIBLE PLAN; fallback index+ring guess")
        plan = dict(workers=(1, 3), yc=lanes[2], rc=0.02, xp=0.075, zc=0.13, alpha=np.radians(30),
                    phic={1: np.pi / 2, 3: -np.pi / 2})
    W = plan["workers"]
    yc, zc, rc, xp, alpha, phic = plan["yc"], plan["zc"], plan["rc"], plan["xp"], plan["alpha"], plan["phic"]
    N(f"PLAN workers {W} xp {xp*1000:.0f} yc {yc*1000:.1f} zc {zc*1000:.0f} rc {rc*1000:.1f} alpha {np.degrees(alpha):.0f}")

    # ---------------------------------------------------------- 3. palm placement over the lid
    Rd = api.hand_down(api.DEFAULT_YAW)
    lid_in_palm = np.array([xp + 0.008, yc, zc])          # where the lid top centre sits in the palm frame
    palm = T - Rd @ lid_in_palm
    N(f"palm target {np.round(palm, 4).tolist()}")
    ph0 = {f: phic[f] - alpha / 2 for f in W}
    ph1 = {f: phic[f] + alpha / 2 for f in W}
    lifted0 = {f: pad(ph0[f], xp - LIFT, yc, zc, rc) for f in W}
    yield from api.move_tips(xyz=palm + np.array([0, 0, 0.06]), R=Rd, tips=lifted0, seconds=2.0)
    yield from api.carry(palm, Rd, seconds=1.2)
    yield from api.hold(0.3)
    Pm, Rm = api.measured_palm()
    lp = Rm.T @ (T - Pm)
    N(f"lid top in measured palm {np.round(lp * 1000, 1).tolist()} (planned {np.round(lid_in_palm*1000,1).tolist()})")
    yc, zc = lp[1], lp[2]
    xs = lp[0] - 0.008                                   # nominal pad-centre depth at touch

    # ---------------------------------------------------------- 4. guarded press onto the lid top
    depth = {f: xs - LIFT for f in W}
    touch = {}
    while len(touch) < len(W):
        for f in W:
            if f not in touch:
                depth[f] += 0.001
        yield from api.move_tips(tips={f: pad(ph0[f], depth[f], yc, zc, rc) for f in W}, seconds=0.06)
        fc = api.contact_forces("lid")
        for f in W:
            if f not in touch and (fc[f] > 0.25 or depth[f] >= xs + 0.014):
                touch[f] = depth[f]
    fc = api.contact_forces("lid")
    N(f"touch depth rel nominal mm {[round((touch[f]-xs)*1000, 1) for f in W]} forces {np.round(fc, 2).tolist()}")
    # Constant press (no escalation unless the time budget demands it): the palm's load stays the same
    # stroke after stroke, so the arm's integral loop holds it at one place (palm-still rule).
    press = 0.005
    px = {f: touch[f] + press for f in W}
    yield from api.move_tips(tips={f: pad(ph0[f], px[f], yc, zc, rc) for f in W}, seconds=0.3, straight=True)
    yield from api.hold(0.5)
    N(f"pressed forces lid {np.round(api.contact_forces('lid'), 2).tolist()} jar {np.round(api.contact_forces('jar'), 2).tolist()}")

    # ---------------------------------------------------------- 5. stroke cycles, closed loop on the lid angle
    # Stroke = contacts-layer `rotate` about the world-vertical jar axis, issued one control step at a time
    # (smoothstep profile split into per-step increments). Every control step of every phase is tracked
    # (lid angle + measured palm) the same way the scorer does, labelled by phase.
    names = [NAMES[f] for f in W]
    ts = 1.0 + 0.4 * (alpha > np.radians(60))
    n = int(round(ts * 50))
    prof = np.array([0.0] + [(k / n) ** 2 * (3 - 2 * k / n) for k in range(1, n + 1)])
    dth = alpha * np.diff(prof)
    trk = {"prev": api.object_joint("lid_twist"), "P": [], "ph": []}

    def check(phase):
        a = api.object_joint("lid_twist")
        if abs(a - trk["prev"]) > np.radians(0.1):
            trk["P"].append(api.measured_palm()[0])
            trk["ph"].append(phase)
        trk["prev"] = a

    def tracked(gen, phase):
        r = None
        try:
            while True:
                next(gen)
                yield
                check(phase)
        except StopIteration as e:
            r = e.value
        return r

    gains = []

    def stroke():
        fsum, fmax = np.zeros(6), np.zeros(6)
        for d in dth:
            yield from api.rotate("jar", [0, 0, 1], float(d), names, seconds=0.02, frame="world")
            check("stroke")
            fc = api.contact_forces("lid")
            fsum += fc
            fmax = np.maximum(fmax, fc)
        return fsum / len(dth), fmax

    def unload_lift(phi, secs_unload, secs_lift):
        # slow unload to touch depth first (palm load fades gradually, lid recoil stays slow), then lift
        yield from tracked(api.move_tips(tips={f: pad(phi[f], touch[f] - 0.001, yc, zc, rc) for f in W},
                                         seconds=secs_unload, straight=True), "unload")
        yield from tracked(api.move_tips(tips={f: pad(phi[f], touch[f] - LIFT, yc, zc, rc) for f in W},
                                         seconds=secs_lift, straight=True), "lift")

    cyc = 0
    while api.object_joint("lid_twist") - a0 < GOAL and api.t < T_END:
        b, t0 = api.object_joint("lid_twist"), api.t
        fmean, fmax = yield from stroke()
        g = api.object_joint("lid_twist") - b
        gains.append(g)
        cyc += 1
        tot = api.object_joint("lid_twist") - a0
        Pnow = api.measured_palm()[0]
        N(f"cyc {cyc} t {api.t:.1f} gain {np.degrees(g):.0f}/{np.degrees(alpha):.0f} total {np.degrees(tot):.0f} "
          f"fmean {[round(float(fmean[f]), 1) for f in W]} fmax {[round(float(fmax[f]), 1) for f in W]} "
          f"press {press*1000:.0f} palm dz {1000 * (Pnow[2] - Pm[2]):.1f} mm")
        if tot >= GOAL:
            break
        # unload, lift, swing back in the air, press again
        a_s = api.object_joint("lid_twist")
        yield from unload_lift(ph1, 0.4, 0.2)
        a_l = api.object_joint("lid_twist")
        # time check: escalate press only if the remaining turn cannot be finished at the current rate
        rate = np.mean(gains[-3:])
        cyc_time = api.t - t0 + 1.35
        need = (GOAL - (a_l - a0)) / max(rate, 1e-3) * cyc_time
        if need > (T_END - api.t) and press < 0.0075:
            press += 0.001
            N(f"cyc {cyc} rate {np.degrees(rate):.0f} deg/cyc too slow -> press {press*1000:.0f} mm")
        px = {f: touch[f] + press for f in W}
        yield from tracked(api.move_tips(tips={f: pad(ph0[f], touch[f] - LIFT, yc, zc, rc) for f in W},
                                         seconds=0.6), "swing")
        a_w = api.object_joint("lid_twist")
        yield from tracked(api.move_tips(tips={f: pad(ph0[f], px[f], yc, zc, rc) for f in W},
                                         seconds=0.3, straight=True), "press")
        yield from tracked(api.hold(0.2), "settle")
        if cyc <= 4:
            a_p = api.object_joint("lid_twist")
            N(f"cyc {cyc} backdrag deg: unload+lift {np.degrees(a_l - a_s):.1f} swing {np.degrees(a_w - a_l):.1f} "
              f"press {np.degrees(a_p - a_w):.1f}")

    # ---------------------------------------------------------- 6. let go (slow unload, then lift clear)
    yield from tracked(api.hold(0.2), "endhold")
    yield from unload_lift({f: ph0[f] + alpha for f in W}, 0.8, 0.4)
    yield from tracked(api.hold(0.3), "end")
    if trk["P"]:
        P = np.array(trk["P"])
        mu = P.mean(0)
        dist = np.linalg.norm(P - mu, axis=1)
        N(f"own palm spread while turning {2000 * dist.max():.1f} mm over {len(P)} steps, ptp xyz mm "
          f"{np.round(1000 * np.ptp(P, axis=0), 1).tolist()}")
        for phs in sorted(set(trk["ph"])):
            m = np.array([p == phs for p in trk["ph"]])
            N(f"  phase {phs}: {m.sum()} turning steps, max dist from mean {1000 * dist[m].max():.1f} mm")
    pm, _ = api.measured_palm()
    N(f"END t {api.t:.1f} cycles {cyc} turn {np.degrees(api.object_joint('lid_twist') - a0):.0f} deg "
      f"palm drift mm {np.round((pm - Pm) * 1000, 1).tolist()}")
