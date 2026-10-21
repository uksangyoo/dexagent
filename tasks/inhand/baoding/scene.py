"""Baoding balls: orbit two balls around each other in the upturned, cupped hand.

Two 38 mm balls of 50 g each (hollow chiming balls; solid steel ones of this
size are 200+ g, beyond what the 0.65 N m finger motors could cage). They
start 3 mm above the cupped palm and fall in under gravity.

Method: middle, ring and an abducted index stand up from the knuckles as the
distal wall; the little finger walls part of the far side; the palm normal is
tilted 11.5 deg and the tilt precesses around a cone, so "downhill" circles
the palm and the balls roll around after it. The thumb is a moving wall that
follows downhill around the open proximal half of the palm (`thumb_wall`).
The orbit stops as soon as the measured exchange angle reaches 200 deg.
Only the arm and finger joint targets are commanded.

Score: cumulative rotation of the ball-to-ball vector about the palm normal
(180 deg = the balls have exchanged places), and both balls still in the hand
after leveling and a 0.6 s hold.

Status: NOT WORKING. Every tested setting loses a ball. The DG5F-S palm has
no wall at about 80 mm from the wrist on either side (between the thumb tip
and the index knuckle, and along the little-finger metacarpal), and the balls
roll out there. The best run reached 211 deg before a ball fell. Ball
friction 0.5 against the default hand friction 0.9 (nominal, not measured).
"""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio.specs import CameraSpec
from dexagent.workcell import Kinematics, TABLE_Z
from dexagent.hand import Controller
from rig import NAME, palm_up, palm_up_scene

HERE = Path(__file__).resolve().parent
RADIUS, MASS = 0.019, 0.05
PALM_XYZ = np.array([0.5, -0.1, TABLE_Z + 0.25])
# Cupped hand: fingers curl up into a wall at the distal end, thumb and little finger raise the sides.
# Middle, ring and index stand up from the knuckles as the distal wall (index spread
# toward the thumb side); the little finger walls the far side. The thumb is a
# moving wall: it follows the ball pair around the open proximal half of the palm.
CUP = np.array([-0.028, -0.948, 0.812, 1.006,  -0.6, 1.1, 0.4, 0.3,  0.0, 1.1, 0.4, 0.3,
                0.1, 1.1, 0.4, 0.3,  0.0, 0.1, 0.9, 0.5])
BOWL = np.array([0.0, 0.068])     # bowl centre (palm y, z)
# Ball centres in the palm frame (+x out of the palm face at x = .026, +z toward the fingers).
SPOTS = (np.array([0.026 + RADIUS + 0.003, -0.006, 0.062]), np.array([0.026 + RADIUS + 0.003, 0.01, 0.102]))

scene = palm_up_scene("baoding", PALM_XYZ, CUP)
_kin = Kinematics()
_arm = np.array([j for j in __import__("robosimstudio").robots.registry.EMBODIMENTS.get(NAME).arms[0].home_q])
_P, _R = _kin.palm_pose(_arm)
for name, spot, color in (("ball_a", SPOTS[0], (0.72, 0.55, 0.18)), ("ball_b", SPOTS[1], (0.25, 0.45, 0.7))):
    scene.add(rss.sphere(name=name, radius=RADIUS, at=tuple(_P + _R @ spot), mass=MASS, friction=0.5, color=color))
scene.add(CameraSpec(name="palm", width=1280, height=720, fov_y=38,
                     eye=tuple(_P + np.array([0.28, 0.16, 0.22])), look_at=tuple(_P + _R @ np.array([0.03, 0, 0.06]))))


def thumb_wall(c, psi, u, sense=1.0, radius=0.047, open_deg=30, height=0.05):
    """Thumb joint targets that wall the downhill side of the bowl.

    Downhill toward the fingers (|psi| < open_deg) is held by the finger
    wall, so the thumb lifts clear and waits on the side the precession sends
    the balls to next (`sense` is the direction psi turns).
    """
    lift = np.clip((np.radians(open_deg + 25) - abs(psi)) / np.radians(25), 0, 1)
    side = psi if abs(psi) >= np.radians(open_deg) else -sense * np.radians(90)
    yz = BOWL + radius * np.array([np.sin(side), np.cos(side)])
    target = np.array([height + 0.05 * lift, yz[0], yz[1]])
    q, _ = c.kin.solve_finger(0, target, c.hand[:4])
    return (1 - u) * CUP[:4] + u * q


def pair_angle(c):
    """Angle of ball_a -> ball_b about the palm normal, in the measured palm frame."""
    a, _ = c.in_palm("ball_a")
    b, _ = c.in_palm("ball_b")
    d = b - a
    return float(np.arctan2(d[1], d[2])), a, b


def in_hand(c):
    """Both balls within the palm's footprint and within 45 mm of its face."""
    ok = []
    for name in ("ball_a", "ball_b"):
        p, _ = c.in_palm(name)
        ok.append(-0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15)
    return all(ok)


def orbit(sim, session=None, turns=2.0, tilt=0.2, period=6.0, sense=-1.0, open_deg=65, goal_deg=200):
    c = Controller(sim, session, objects=("ball_a", "ball_b"))
    c.hand = CUP.copy()
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(1.0, "settle")
        if not in_hand(c):
            outcome = "balls_missed_the_hand"; return
        last, _, _ = pair_angle(c)
        total, phase0 = 0.0, None
        n = round(turns * period * 50)
        # Ease the tilt in, first toward the finger wall (phi = 90 deg tips the palm down toward +x).
        c.phase = "orbit"
        for k in range(n + 25):
            u = min(1.0, k / 25)
            phi = np.pi / 2 + sense * 2 * np.pi * max(0, k - 25) / (period * 50)
            c.R = palm_up(0.0, (tilt * u * np.cos(phi), tilt * u * np.sin(phi)))
            c.arm = c.kin.solve_arm(c.xyz, c.R, c.arm, tol=0.004)
            # Downhill direction in the palm plane (palm y, z): psi = 0 toward the fingers.
            g = c.R.T @ np.array([0, 0, -1.0])
            psi = float(np.arctan2(g[1], g[2]))
            c.hand = CUP.copy()
            c.hand[:4] = thumb_wall(c, psi, u, sense, open_deg=open_deg)
            c.step()
            a, _, _ = pair_angle(c)
            total += (a - last + np.pi) % (2 * np.pi) - np.pi
            last = a
            if abs(total) >= np.radians(goal_deg):
                break   # the balls have exchanged places; level out
            if k % 25 == 0 and not in_hand(c):
                outcome = "dropped"
                metrics["swap_deg"] = round(float(np.degrees(abs(total))), 1)
                return
            yield
        yield from c.move(R=palm_up(0.0), hand=CUP, seconds=0.8, phase="level")
        yield from c.hold(0.6, "hold")
        a, _, _ = pair_angle(c)
        total += (a - last + np.pi) % (2 * np.pi) - np.pi
        metrics["swap_deg"] = round(float(np.degrees(abs(total))), 1)
        metrics["both_in_hand"] = in_hand(c)
        outcome = "success" if metrics["both_in_hand"] and abs(total) >= np.pi else (
            "dropped" if not metrics["both_in_hand"] else "under_rotated")
    finally:
        c.finish("baoding", outcome, metrics, HERE / "results")


routines = {"orbit the balls": orbit}
