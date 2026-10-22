"""pen_pickup_pivot, contacts level.

Plan: thumb-index pinch across the pen near one end (pinch line horizontal,
across the pen); the pinch is a free hinge (the pen has no torsional friction).
Swing the hinge on a closed-loop arc about the pen's far end (which stays on
the table) until the pen stands vertical on that end, then lift straight up so
it hangs vertically from the hinge. The first pinch must sit above the pen's
equator (fingertip shells vs table); once the end is raised, the contacts are
moved down onto the equator (re-grasp) for a secure hinge.
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
EZ = np.array([0.0, 0.0, 1.0])
EX_P, EZ_P = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
PINCH_C = np.array([0.080, 0.040, 0.118])  # comfortable thumb-index pinch centre (palm frame)
S_PINCH = 0.045           # pinch this far from the pen centre toward the thumb-side end
ATTEMPTS = [(25, 0.003), (15, 0.004), (32, 0.004)]   # (contact elevation deg, squeeze m)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _fmt(v):
    return np.round(np.asarray(v) * 1000, 1).tolist()


def _F(api):
    return np.round(api.contact_forces("pen"), 2)


def plan(api, elev_deg, sq, r):
    """Pinch geometry from the pen's current pose: palm R, palm xyz, object-frame points/normals."""
    c, Ro = api.object_pose("pen")
    a = Ro[:, 2].copy(); a[2] = 0; a /= np.linalg.norm(a)
    best = None
    for sgn in (1, -1):
        y = _wrap(np.arctan2(sgn * a[1], sgn * a[0]) - np.pi / 2)
        if best is None or abs(_wrap(y - 0.7)) < abs(_wrap(best[0] - 0.7)):
            best = (y, sgn)
    yaw, sgn = best
    a = sgn * a                       # toward the thumb-side end (A) = palm +y
    R = api.hand_down(yaw)
    f = R[:, 2] - (R[:, 2] @ a) * a; f /= np.linalg.norm(f)
    E = np.radians(elev_deg)
    a_o, f_o, up_o = Ro.T @ a, Ro.T @ f, Ro.T @ EZ
    n_th = -np.cos(E) * f_o + np.sin(E) * up_o
    n_ix = np.cos(E) * f_o + np.sin(E) * up_o
    pts = {"thumb": S_PINCH * a_o + r * n_th, "index": S_PINCH * a_o + r * n_ix}
    nrm = {"thumb": n_th, "index": n_ix}
    # palm-frame pinch location the two fingers reach best
    d = r + api.pad_offset - sq
    bestp = None
    for dx in (0.0, -0.01, 0.01):
        for dz in (0.0, -0.008, 0.008, -0.016):
            for dy in (0.0, -0.01, 0.01):
                Pp = PINCH_C + np.array([d * np.sin(E) + dx, dy, dz])
                th = Pp + d * (-np.cos(E) * EZ_P - np.sin(E) * EX_P)
                ix = Pp + d * (np.cos(E) * EZ_P - np.sin(E) * EX_P)
                e = max(api.reach("thumb", th), api.reach("index", ix))
                score = e + 0.02 * np.linalg.norm([dx, dy, dz])
                if bestp is None or score < bestp[0]:
                    bestp = (score, Pp, e)
    _, Pp, e = bestp
    palm = api.palm_for(c + S_PINCH * a, Pp, R)
    api.note(f"plan elev={elev_deg} sq={sq} yaw={yaw:.3f} pen c={_fmt(c)} Pp={_fmt(Pp)} reach={e*1000:.1f}mm")
    return dict(R=R, palm=palm, pts=pts, nrm=nrm, u=a, a_o=a_o, f_o=f_o)


def run(api):
    c0, _ = api.object_pose("pen")
    r = float(np.clip(c0[2] - TABLE, 0.003, 0.0045))
    yield from api.open(0.8)
    h_nom = 0.0662

    for attempt, (elev, sq) in enumerate(ATTEMPTS):
        if api.t > 40:
            break
        g = plan(api, elev, sq, r)
        R, u = g["R"], g["u"]
        pc, _ = api.palm_pose()
        if attempt == 0:
            yield from api.carry(g["palm"] + [0, 0, 0.05], R, seconds=2.0)
        else:
            yield from api.carry(pc + [0, 0, 0.04], None, seconds=0.6)
            yield from api.open(0.6)
            yield from api.carry(g["palm"] + [0, 0, 0.05], R, seconds=1.5)
        yield from api.carry(g["palm"], R, seconds=1.0)
        res = yield from api.approach("pen", g["pts"], standoff=0.015, seconds=0.8, normals=g["nrm"])
        res = yield from api.grasp("pen", g["pts"], squeeze=sq, seconds=0.8, normals=g["nrm"])
        api.note(f"grasp {res}")
        yield from api.hold(0.3)

        # ---- pivot about the far (B) end, closed loop on the pen's elevation angle
        k = np.cross(u, EZ)               # rotation axis: positive raises A
        h_samples = []
        regrasped = False

        def state():
            c, Rc = api.object_pose("pen")
            ax = Rc[:, 2].copy()
            tw = api.tips_world()
            P = 0.5 * (tw[0] + tw[1])
            if (P - c) @ ax < 0:
                ax = -ax
            return c, ax, P, np.arctan2(ax[2], ax @ u)

        def step_to(tgt_deg, seconds):
            c, ax, P, th = state()
            if np.radians(15) < th < np.radians(50):
                h_samples.append((c[2] - TABLE - r) / ax[2])
            h = float(np.clip(np.median(h_samples), 0.05, 0.085)) if h_samples else h_nom
            E = c - h * ax
            Rk = Rotation.from_rotvec(k * (np.radians(tgt_deg) - th)).as_matrix()
            P2 = E + Rk @ (P - E)
            pm, _ = api.measured_palm()
            yield from api.carry(pm + (P2 - P), R, seconds=seconds)
            return th, h

        nseg, ok = 20, True
        for kseg in range(1, nseg + 1):
            tgt = 90.0 * kseg / nseg
            try:
                th, h = yield from step_to(tgt, 0.35)
            except ValueError:
                ok = False
                break
            F = _F(api)
            c, ax, P, th = state()
            api.note(f"seg {kseg} tgt={tgt:.0f} th={np.degrees(th):.1f} h={h*1000:.1f} F={F[:2].tolist()}")
            if (F[0] < 0.03 or F[1] < 0.03) and np.degrees(th) < tgt - 8:
                ok = False
                break
            if not regrasped and np.degrees(th) > 14:
                # A end is up: slide the contacts down onto the equator for a secure hinge
                pts = {"thumb": S_PINCH * g["a_o"] - r * g["f_o"], "index": S_PINCH * g["a_o"] + r * g["f_o"]}
                nrm = {"thumb": -g["f_o"], "index": g["f_o"]}
                res = yield from api.grasp("pen", pts, squeeze=0.003, seconds=0.5, normals=nrm)
                api.note(f"regrasp equator {res}")
                regrasped = True
        if not ok:
            api.note(f"attempt {attempt} lost the pen")
            continue

        for _ in range(4):   # settle at vertical
            c, ax, P, th = state()
            if abs(np.degrees(th) - 90) < 2:
                break
            yield from step_to(90.0, 0.4)
        yield from api.hold(0.4)
        c, ax, P, th = state()
        h = float(np.clip(np.median(h_samples), 0.05, 0.085)) if h_samples else h_nom
        api.note(f"stood? th={np.degrees(th):.1f} h={h*1000:.1f} low_est={(c[2]-h*abs(ax[2])-r-TABLE)*1000:.1f}mm")

        # ---- lift clear, hanging from the hinge
        pc, _ = api.palm_pose()
        yield from api.carry(pc + [0, 0, 0.07], R, seconds=2.0)
        yield from api.hold(1.0)
        c, ax, P, th = state()
        api.note(f"final th={np.degrees(th):.1f} c={_fmt(c)} F={_F(api).tolist()}")
        return
