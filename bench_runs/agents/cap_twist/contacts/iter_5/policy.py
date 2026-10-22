"""cap_twist, contacts level.

Strategy: palm held still, palm-down over the lid. Several fingertips press DOWN
on the lid's top face at radius R_PAD around the axis (the lid is on a hinge, so
no force closure is needed, only friction). Each stroke: grasp (press) ->
rotate the held pads about world +z (CCW) -> release (lift) -> rotate the pads
back in the air -> repeat, closed-loop on the lid angle until >= GOAL.

Placement of the palm relative to the lid, and each finger's arc, are chosen at
runtime (zero sim time) by a grid search with api.reach (fingertip IK residual).
"""
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring"]
FIDX = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
PRIORITY = ["middle", "index", "thumb", "ring"]
# preferred pad direction around the lid centre, palm y-z plane, deg from palm +z toward +y
PREF_BETA = {"thumb": 160.0, "index": 50.0, "middle": 0.0, "ring": -50.0}
R_PAD = 0.021        # pad radius on the lid top (lid radius >= 31.5 mm)
LID_TOP = 0.123      # lid top above the jar base
INSET = 0.002        # grasp squeeze (iter4: 4 mm + extra -> 15-20 N -> palm spread 17 mm)
GOAL = np.radians(372.0)
MAX_EXTRA = 0.003   # cap on extra press depth (iter 3: 15 mm jammed fingers at 40 N and pushed the palm 25 mm)
T_END = 53.0
YAW = 0.7


def _world_angle(R, beta_deg):
    b = np.radians(beta_deg)
    v = R @ np.array([0.0, np.sin(b), np.cos(b)])
    return float(np.arctan2(v[1], v[0]))


def _pad_offset_world(alpha, h):
    return np.array([R_PAD * np.cos(alpha), R_PAD * np.sin(alpha), h])


LIFT = 0.020         # release clearance (mostly straight up)


def _ok(api, f, cp, R, alpha, h_press):
    """Pad reachable both pressing at `alpha` and lifted above it."""
    pp = cp + R.T @ _pad_offset_world(alpha, h_press)
    if api.reach(f, pp) > 0.0015:
        return False
    lifted = cp + R.T @ (_pad_offset_world(alpha, h_press + LIFT) * np.array([1.17, 1.17, 1.0]))
    return api.reach(f, lifted) < 0.002


def plan(api, R):
    """Coarse-to-fine search of the palm-frame point that coincides with the
    lid-top centre, and per-finger start angles; requires every pad to be
    reachable pressed AND lifted LIFT above, across the whole stroke."""
    step = 15
    rel = np.arange(-75, 76, step)
    h = 0.008 - INSET
    placements = [np.array([d, yc, zc]) for d in (0.07, 0.085, 0.10)
                  for yc in (0.0, 0.02) for zc in (0.095, 0.115, 0.135)]
    coarse = []
    for cp in placements:
        n = 0
        for f in FINGERS:
            a0 = _world_angle(R, PREF_BETA[f])
            n += sum(_ok(api, f, cp, R, a0 + np.radians(da), h) for da in (-30, 0, 30))
        coarse.append(n)
    order = np.argsort(coarse)[::-1][:3]
    best = None
    for j in order:
        cp = placements[j]
        rows = {}
        for f in FINGERS:
            a0 = _world_angle(R, PREF_BETA[f])
            rows[f] = (a0, np.array([_ok(api, f, cp, R, a0 + np.radians(da), h) for da in rel]))
        for dth in (60, 45, 30):
            k = dth // step
            starts = {}
            for f in FINGERS:
                a0, ok = rows[f]
                cand = [i for i in range(len(rel) - k) if ok[i:i + k + 1].all()]
                if not cand:
                    continue
                mid = len(rel) // 2
                i = min(cand, key=lambda i: abs(i + k / 2 - mid))
                starts[f] = a0 + np.radians(rel[i])
            chosen = {}
            for f in PRIORITY:
                if f not in starts:
                    continue
                pf = _pad_offset_world(starts[f], 0)
                if all(np.linalg.norm(pf - _pad_offset_world(a, 0)) > 0.019 for a in chosen.values()):
                    chosen[f] = starts[f]
            n = len(chosen)
            if n < 2:
                continue
            score = n * min(dth, 45) + 0.1 * dth
            if best is None or score > best["score"]:
                best = {"score": score, "cp": cp, "dth": dth, "start": chosen, "coarse": coarse[j],
                        "maps": {f: "".join("#" if o else "." for o in rows[f][1]) for f in FINGERS}}
    return best


def run(api):
    p0, R0 = api.object_pose("jar")
    top = p0 + R0 @ np.array([0.0, 0.0, LID_TOP])
    up_obj = R0.T @ np.array([0.0, 0.0, 1.0])
    R = api.hand_down(YAW)

    pl = plan(api, R)
    if pl is None:
        api.note("PLAN FAILED, fallback")
        plan_txt = "fallback"
        pl = {"cp": np.array([0.07, 0.015, 0.115]), "dth": 30,
              "start": {f: _world_angle(R, PREF_BETA[f]) - np.radians(15) for f in ("index", "middle", "thumb")}}
    else:
        plan_txt = f"cp={np.round(pl['cp'], 3).tolist()} dth={pl['dth']} fingers={list(pl['start'])} maps={pl['maps']}"
        api.note(f"plan cp={np.round(pl['cp'], 3).tolist()} dth={pl['dth']} fingers={list(pl['start'])} maps={pl['maps']}")
    cp, dth = pl["cp"], np.radians(pl["dth"])
    fingers = list(pl["start"])

    P = api.palm_for(top, cp, R)
    yield from api.carry(P + np.array([0, 0, 0.05]), R, 1.5)
    yield from api.carry(P, R, 1.0)
    yield from api.hold(0.6)
    nrm = {f: up_obj for f in fingers}

    def points(sgn):
        out = {}
        for f, a in pl["start"].items():
            w = top + _pad_offset_world(a + (0.0 if sgn > 0 else dth), 0.0)
            out[f] = R0.T @ (w - p0)
        return out

    sgn = 1.0
    pts = points(sgn)
    yield from api.approach("jar", pts, standoff=0.015, seconds=1.0, normals=nrm)
    yield from api.hold(0.2)

    extra = {f: 0.0 for f in fingers}
    a_start = api.object_joint("lid_twist")
    deg = lambda: np.degrees(api.object_joint("lid_twist") - a_start)
    cycle, flips = 0, 0
    while deg() < np.degrees(GOAL) and api.t < T_END:
        cycle += 1
        a0 = deg()
        press = {f: pts[f] - extra[f] * nrm[f] for f in fingers}
        yield from api.grasp("jar", press, squeeze=INSET, seconds=0.4, normals=nrm)
        for _ in range(2):
            F = api.contact_forces("lid")
            weak = [f for f in fingers if F[FIDX[f]] < 0.5 and extra[f] < MAX_EXTRA]
            if not weak:
                break
            yield from api.squeeze(weak, "jar", 0.003, 0.25)
            for f in weak:
                extra[f] += 0.003
        F = api.contact_forces("lid")
        a1 = deg()
        Pm, _ = api.measured_palm()
        yield from api.rotate("jar", [0, 0, 1], sgn * dth, fingers, seconds=0.7, frame="world")
        a2 = deg()
        yield from api.release(fingers, "jar", clear=LIFT + max(extra.values()), seconds=0.3)
        Fr = api.contact_forces("lid")
        stuck = [f for f in fingers if Fr[FIDX[f]] > 0.3]
        if stuck:
            yield from api.release(stuck, "jar", clear=0.01, seconds=0.2)
        a3 = deg()
        if a3 >= np.degrees(GOAL):
            api.note(f"c{cycle} s{sgn:+.0f} grasp{a1 - a0:+.0f} rot{a2 - a1:+.0f} rel{a3 - a2:+.0f} F={np.round(F, 1).tolist()} stuck={stuck}")
            break
        yield from api.rotate("jar", [0, 0, 1], -sgn * dth, fingers, seconds=0.5, frame="world")
        a4 = deg()
        api.note(f"c{cycle} s{sgn:+.0f} grasp{a1 - a0:+.0f} rot{a2 - a1:+.0f} rel{a3 - a2:+.0f} ret{a4 - a3:+.0f} "
                 f"tot{a4:.0f} dP={np.round((Pm - P) * 1000, 1).tolist()} F={np.round(F, 1).tolist()} ex={[round(e * 1000) for e in extra.values()]} stuck={stuck}")
        if a4 - a0 < -5 and flips < 2:   # this cycle turned the lid the wrong way net: mirror the stroke
            sgn, flips = -sgn, flips + 1
            pts = points(sgn)
            yield from api.rotate("jar", [0, 0, 1], -sgn * dth, fingers, seconds=0.5, frame="world")
            api.note(f"flip stroke sign -> {sgn:+.0f}")
    api.note(f"PLAN {plan_txt}")
    api.note(f"done total={np.degrees(api.object_joint('lid_twist') - a_start):.1f} t={api.t:.1f}")
    yield from api.hold(0.3)
