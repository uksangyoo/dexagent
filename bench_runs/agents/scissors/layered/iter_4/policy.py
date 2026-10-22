"""scissors, layered level.

Grasp: a wide thumb/index pinch across loop A along the blade axis (thumb
behind the back wall, index in front of the front wall), so the blade weight
(CoM near the pivot, ~85 mm ahead of the grasp) is resisted by a 45 mm lever.
Actuation: a third finger (middle or ring, chosen by IK reach) sits inside
hole B and is swung about the hinge axis with the contacts-layer `rotate`
primitive, closed-loop on the hinge angle.
Layers: joints (pre-shape), fingertips (grasp placement, squeeze, release),
contacts (rotate about the hinge, tuck), arm (carry).
"""
import numpy as np

TABLE = 0.75
NOM_H = 0.0068            # start pivot height above the table = 6.8 mm * size (measured iter 1)
M = np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])  # scissors local (X,Y,Z) -> palm (down, left, fwd)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])




YG = 0.035   # thumb/index pinch line on loop A (local y). Film frames (iter 1) show each shank is a
             # ~9 mm capsule from about y = +-18 mm on the loop's FRONT wall to the pivot, so the free
             # face in front of loop A is its OUTER part (y > ~29 mm); iters 2-3 (y = 18, 12) hit the shank.


def _local_targets(s, Z, st, si, sq, open_a=0.45, ZB=None):
    ZB = Z if ZB is None else ZB
    L = {}
    L["A"] = np.array([-0.065 * s, 0.024 * s, Z])
    L["thumb_pre"] = np.array([-0.0875 * s - 0.008 - st, YG * s, Z])
    L["thumb_sq"] = np.array([-0.0875 * s - 0.008 + sq, YG * s, Z])
    L["index_pre"] = np.array([-0.0425 * s + 0.008 + si, YG * s, Z])
    L["index_sq"] = np.array([-0.0425 * s + 0.008 - sq, YG * s, Z])
    b0 = np.array([-0.065 * s, -0.024 * s, ZB])
    L["B0"] = b0
    L["Bopen"] = _rz(open_a) @ b0
    return L


def _to_palm(L, a, Rrel):
    return {k: a + Rrel @ (v - L["A"]) for k, v in L.items()}


def _search(api, s, Z, Rrel, D):
    """Pick the palm-frame location `a` of loop A's centre and the hole-B finger by IK reach.
    Among feasible anchors prefer the highest palm (large a_x) so tucked fingers and the thumb base clear the table."""
    cands = []
    for st, si in ((0.006, 0.004), (0.003, 0.002)):
        L = _local_targets(s, Z, st, si, 0.005)
        for ax in (0.08, 0.09, 0.10, 0.11):
            for ay in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
                for az in (0.10, 0.115, 0.13, 0.145):
                    a = np.array([ax, ay, az])
                    P = _to_palm(L, a, Rrel)
                    e_ti = max(api.reach("thumb", P["thumb_pre"]), api.reach("thumb", P["thumb_sq"]),
                               api.reach("index", P["index_pre"]), api.reach("index", P["index_sq"]))
                    if e_ti > 0.02:
                        continue
                    for fB in ("ring", "middle"):
                        e = max(e_ti, api.reach(fB, P["B0"]), 0.5 * api.reach(fB, P["Bopen"]))
                        cands.append((e, a, fB, st, si))
        if cands and min(c[0] for c in cands) < 0.002:
            break
    emin = min(c[0] for c in cands)
    ok = [c for c in cands if c[0] <= max(0.0015, emin + 0.0005)]
    best = min(ok, key=lambda c: (-round(c[1][0], 3), abs(c[1][2] - 0.12) + abs(c[1][1] - 0.02)))
    D.append(f"search n={len(cands)} emin={emin*1000:.1f}mm best err={best[0]*1000:.1f}mm a={np.round(best[1], 3).tolist()} "
             f"fB={best[2]} st={best[3]} si={best[4]}")
    return best


def _diag(api, D, tag):
    p, R = api.object_pose("scissors")
    try:
        f = np.round(api.contact_forces("scissors"), 2).tolist()
    except Exception:
        f = None
    try:
        ft = np.round(api.contact_forces("table"), 1).tolist()
    except Exception:
        ft = None
    tp = np.round(1000 * api.tips(), 0).astype(int).tolist()
    D.append(f"{tag}: t={api.t:.2f} Ftable={ft} tips_palm_mm={tp} piv={np.round(p, 4).tolist()} Rzz={R[2, 2]:.3f} Rxz={R[2, 0]:.3f} Ryz={R[2, 1]:.3f} "
             f"hinge={np.degrees(api.object_joint('scissors_hinge')):.1f}deg F={f}")
    return p, R


def run(api):
    D = []
    try:
        yield from _main(api, D)
    except Exception as e:
        D.append(f"EXCEPTION {type(e).__name__}: {e}")
        raise
    finally:
        for line in D[-40:]:
            api.note(line)


def _main(api, D):
    p0, R0 = api.object_pose("scissors")
    psi = float(np.arctan2(R0[1, 0], R0[0, 0]))
    s_z = (p0[2] - TABLE) / NOM_H
    s = float(np.clip(s_z, 0.9, 1.1))
    D.append(f"psi_p={0.7 + 0.35 * (psi - 0.7):.3f} s={s:.3f}")
    D.append(f"start piv={np.round(p0, 5).tolist()} psi={psi:.3f} s_z={s_z:.3f} Rzz={R0[2, 2]:.4f} "
             f"hinge={api.object_joint('scissors_hinge'):.4f}")
    psi_p = 0.7 + 0.35 * (psi - 0.7)          # palm yaw only partly follows the scissors (arm reach near the base)
    Rp = api.hand_down(yaw=psi_p)
    Rrel = Rp.T @ R0                           # scissors local -> palm frame
    # joints layer: open pre-shape with the little finger tucked, as the IK seed
    seed = api.OPEN.copy()
    seed[16:20] = [0.3, 1.4, 1.3, 1.1]
    yield from api.move_joints(q=seed, seconds=0.5)

    h = float(np.clip(0.0095 * s, 0.008, 0.0095))   # pinch pad centres below the wall top (~10 mm * s)
    hB = 0.009                                  # hole-B finger pad height
    Z = TABLE + h - p0[2]
    err, a, fB, st, si = _search(api, s, Z, Rrel, D)
    other = "ring" if fB == "middle" else "middle"
    yield from api.tuck([other, "little"], seconds=0.4)
    iB = 2 if fB == "middle" else 3

    lifted = False
    for attempt in range(3):
        sq = 0.005 + 0.003 * attempt
        p0, R0 = api.object_pose("scissors")
        Rrel = Rp.T @ R0
        Z = TABLE + h - p0[2]
        L = _local_targets(s, Z, st, si, sq, ZB=TABLE + hB - p0[2])
        P = _to_palm(L, a, Rrel)
        WA = p0 + R0 @ L["A"]
        Pg = WA - Rp @ a                        # palm position at grasp height
        above = Pg + np.array([0, 0, 0.06])
        r = yield from api.move_tips(xyz=above, R=Rp, seconds=2.0 if attempt == 0 else 1.2,
                                     tips={"thumb": P["thumb_pre"], "index": P["index_pre"], fB: P["B0"]})
        D.append(f"[{attempt}] pre ik={r.get('ik_residual_mm')} Pg={np.round(Pg, 3).tolist()} sq={sq}")
        yield from api.hold(0.2)
        # descend vertically
        yield from api.carry(xyz=Pg, seconds=1.2)
        yield from api.hold(0.3)
        tw = api.tips_world()
        D.append(f"[{attempt}] down tip z mm " + str(np.round(1000 * (tw[:, 2] - TABLE), 1).tolist()))
        # iters 2-3: free pads sat ~2.5-3.5 mm above their targets; correct the palm height with the
        # (still free) thumb so the pads meet the wall faces below the wall top, not its edge
        bias = float(np.clip(tw[0, 2] - (TABLE + h), 0.0, 0.005))
        if bias > 0.001:
            Pg = Pg - np.array([0, 0, bias])
            yield from api.carry(xyz=Pg, seconds=0.4)
            yield from api.hold(0.2)
            tw = api.tips_world()
            D.append(f"[{attempt}] bias {bias*1000:.1f}mm -> tip z mm " + str(np.round(1000 * (tw[:, 2] - TABLE), 1).tolist()))
        _diag(api, D, f"[{attempt}] down")
        # squeeze thumb and index toward each other across loop A
        yield from api.move_tips(tips={"thumb": P["thumb_sq"], "index": P["index_sq"]}, seconds=0.8, straight=True)
        yield from api.hold(0.4)
        _diag(api, D, f"[{attempt}] squeezed")
        yield from api.carry(xyz=Pg + np.array([0, 0, 0.03]), seconds=1.0)
        yield from api.hold(0.3)
        p1, R1 = _diag(api, D, f"[{attempt}] lift30")
        riseA = (p1 + R1 @ L["A"])[2] - (p0 + R0 @ L["A"])[2]
        D.append(f"[{attempt}] loop A rise {riseA*1000:.1f}mm pivot rise {(p1[2]-p0[2])*1000:.1f}mm")
        if riseA > 0.015 or p1[2] - p0[2] > 0.012:
            lifted = True
            break
        D.append(f"[{attempt}] LIFT FAILED")
        # open the pinch, raise, and retry from the fresh pose
        yield from api.move_tips(tips={"thumb": P["thumb_pre"], "index": P["index_pre"]}, seconds=0.5, straight=True)
        yield from api.carry(xyz=Pg + np.array([0, 0, 0.06]), seconds=0.8)
        yield from api.hold(0.3)
    if not lifted:
        return
    yield from api.carry(xyz=Pg + np.array([0, 0, 0.15]), seconds=1.8)
    yield from api.hold(0.4)
    _diag(api, D, "lifted")

    # cycles: contacts layer rotate of the hole-B finger about the hinge axis
    def axis():
        _, Ro = api.object_in_palm("scissors")
        return Ro[:, 2]

    for c in range(2):
        turned = 0.0
        while api.object_joint("scissors_hinge") < 0.33 and turned < 1.0:
            yield from api.rotate("scissors", axis(), 0.06, [fB], seconds=0.12)
            turned += 0.06
        yield from api.hold(0.2)
        _diag(api, D, f"open{c} turned={turned:.2f}")
        back = 0.0
        while api.object_joint("scissors_hinge") > 0.025 and back < turned + 0.6:
            yield from api.rotate("scissors", axis(), -0.06, [fB], seconds=0.12)
            back += 0.06
        yield from api.hold(0.2)
        _diag(api, D, f"closed{c} back={back:.2f}")

    # set down and release
    yield from api.carry(xyz=Pg + np.array([0, 0, 0.004]), seconds=2.0)
    yield from api.hold(0.3)
    _diag(api, D, "set down")
    cur = api.tip_targets()
    yield from api.move_tips(tips={"thumb": cur[0] + np.array([0, 0, -0.012]),
                                   "index": cur[1] + np.array([0, 0, 0.012])}, seconds=0.6, straight=True)
    yield from api.carry(xyz=Pg + np.array([0, 0, 0.09]), seconds=1.2)
    yield from api.hold(0.5)
    _diag(api, D, "released")
