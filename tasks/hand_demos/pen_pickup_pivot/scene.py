"""IMG_4773 / IMG_4780: pick a pen up off the table and bring it upright in the fingers, then stand it on its tip.

The pivot is extrinsic, as in IMG_4780: thumb and index pinch the pen 50 mm
from its centre, and as the hand rises the far end drags on the table, so the
pen turns in the pinch until it hangs upright. It is then lowered until its
tip stands on the table, and lifted clear again.

The DG5F-S fingertips are 16-20 mm thick and the pen is 7.5 mm across, so the
pads cannot reach the pen's equator on a flat table. The pinch is a tweezer:
the rounded fingertip ends, tilted 50 deg inward on a palm pitched 0.3 rad,
meet the pen about 25 deg above its equator, inside the friction cone (0.9).
The pen's contacts have no torsional friction (`grip.smooth_pen`: its geom
takes priority over the hand's condim 4), so a two-pad pinch cannot stop the
pen turning about the pinch axis: that freedom is what lets it pivot upright.

Assumed numbers (not measured): pen 5 g, 7.5 mm x 140 mm; pad friction 0.9.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, YAW
from dexagent.hand import Controller
from dexagent.objects import pen, PEN_RADIUS, PEN_HALF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pen_gravity_pivot"))
import grip as fingers  # noqa: E402  (fingertip IK helpers shared with pen_gravity_pivot)

CLIPS = [HERE.parents[2] / f"reference/hand_demos/IMG_{n}.mov" for n in ("4773_pen_pickup_pivot", "4780_pen_pickup_stand")]
ACROSS = np.array([-np.sin(YAW), np.cos(YAW), 0.0])   # across the fingers
OFFSET = 0.03
# Placed where the pen lies in IMG_4780; its direction is turned to lie across the fingers.
START = WORK + np.array([-0.035, -0.07, PEN_RADIUS + 0.0005])
PITCH, BETA = 0.3, np.radians(50)
CENTRE = np.array([0.11, 0.03, 0.10])
TIGHT, OPEN_W = PEN_RADIUS - 0.00425, 0.025
STOP = 0.015      # middle-finger stop beside the index, across the fingers [m]
ENTER = 0.015     # the middle comes in this much further out, then closes to STOP [m]

scene = table_scene("pen_pickup_pivot")
scene.add(pen(at=tuple(START), direction=ACROSS, friction=0.9))   # the pad friction: the pen pair uses the pen's own


def tilt_from_vertical(axis):
    return float(np.degrees(np.arccos(min(1.0, abs(axis[2])))))


def pickup_pivot(sim, session=None, tight=TIGHT, lift_pad=0.0, offset=OFFSET, rise=3.0, stand_gap=0.004,
                 stop=STOP, enter=ENTER, regrip=None, use_middle=True):
    fingers.smooth_pen(sim)
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, R0 = c.pose("pen")
        axis0 = R0[:, 2]
        side = np.sign(axis0 @ ACROSS) or 1.0
        grip = p0 - side * offset * axis0
        Ry = Rotation.from_rotvec([0, PITCH, 0]).as_matrix()
        R = hand_down() @ Ry
        axis, down = Ry.T @ np.array([0, 0, -1.0]), Ry.T @ np.array([1.0, 0, 0])
        pinch = lambda h: fingers.tweezer(c.kin, CENTRE, axis, down, h, BETA)[0]
        at = lambda dz: c.palm_for(grip + [0, 0, lift_pad + dz], centre=CENTRE, R=R)
        yield from c.move(at(.08), R, pinch(OPEN_W), 1.5, "approach")
        yield from c.hold(0.5)
        yield from c.move(at(0), R, seconds=1.0, phase="descend")
        yield from c.hold(0.4)
        # Correct the few mm of arm sag from the measured pad positions.
        err = p0[2] + lift_pad - (fingers.pads_world(c)[:2, 2].mean() - fingers.TIP_END * np.cos(BETA))
        yield from c.move(c.xyz + [0, 0, err], R, seconds=0.3, phase="correct")
        yield from c.hold(0.2)
        # Close in task space; a joint-space blend swings the thumb through the table.
        yield from c.move(hand=lambda u: pinch(OPEN_W + u * (tight - OPEN_W)), seconds=1.0, phase="pinch")
        yield from c.hold(0.3)
        base = c.xyz.copy()
        # Rise until the pen stands upright on its far tip (IMG_4780): the far
        # end drags on the table and the pen turns in the pinch. The table still
        # carries the tip, so the pen does not swing.
        reach = offset + PEN_HALF + PEN_RADIUS
        yield from c.move(base + [0, 0, reach - stand_gap], R, seconds=rise, phase="pivot up")
        yield from c.hold(0.5, "standing")
        p, Rp = c.pose("pen")
        metrics["stand_tilt_from_vertical_deg"] = round(tilt_from_vertical(Rp[:, 2]), 1)
        metrics["stand_tip_height_mm"] = round(1000 * (p[2] - (PEN_HALF + PEN_RADIUS) * abs(Rp[2, 2]) - (p0[2] - PEN_RADIUS)), 1)
        # While the table still carries the tip, bring the middle finger in
        # beside the index as a lateral stop: two rounded pads alone cannot
        # cage a cylinder sideways, and the pen otherwise rolls out across the
        # fingers as soon as it hangs.
        pp, Rpp = c.in_palm("pen")
        d = Rpp[:, 2]
        centre_v = pp + d * ((CENTRE - pp) @ d)
        if regrip is not None:
            # Re-centre the tweezer on the pen where it now passes the pinch.
            q1 = fingers.tweezer(c.kin, centre_v, axis, down, regrip, BETA, base=c.hand)[0]
            yield from c.move(hand=q1, seconds=0.6, phase="regrip")
            yield from c.hold(0.2)
        dm = np.cos(BETA) * down + np.sin(BETA) * axis
        def middle(back, out=0.0):
            q = c.hand.copy()
            end = centre_v - (tight + back) * axis + np.array([0.0, -stop - out, 0.0])
            q[8:12] = fingers.solve(c.kin, 2, end - fingers.TIP_END * dm, dm, q[8:12], 0.02)[0]
            return q
        if use_middle:
            yield from c.move(hand=middle(0.02, enter), seconds=0.6, phase="middle in")
            yield from c.move(hand=lambda u: middle(0.02 * (1 - u), enter * (1 - u)), seconds=0.6, phase="middle stop")
        yield from c.hold(0.3)
        yield from c.move(c.xyz + [0, 0, .10], R, seconds=1.5, phase="lift clear")
        yield from c.hold(1.0, "hold")
        p, Rp = c.pose("pen")
        metrics.update(final_tilt_from_vertical_deg=round(tilt_from_vertical(Rp[:, 2]), 1),
                       final_height_mm=round(1000 * (p[2] - p0[2]), 1),
                       pivot_deg=round(float(np.degrees(np.arccos(min(1.0, abs(Rp[:, 2] @ axis0))))), 1))
        held = p[2] - p0[2] > .1
        upright = metrics["final_tilt_from_vertical_deg"] < 15 and metrics["stand_tilt_from_vertical_deg"] < 15
        outcome = "success" if held and upright else ("dropped" if not held else "not_upright")
    finally:
        c.finish("pen_pickup_pivot", outcome, metrics, HERE / "results")


routines = {"pick up and pivot upright": pickup_pivot}
