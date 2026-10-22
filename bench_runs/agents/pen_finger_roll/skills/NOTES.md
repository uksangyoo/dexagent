# pen_finger_roll, phase 3 (force level + cross-agent memory, 6 iterations)


## Plan (from reading the 5 earlier pen_finger_roll agents + pen_gravity_pivot/layered)
All five earlier agents pinched thumb-index with the pinch line along palm z and tried to swing about palm y,
which is not the pinch's free axis, so the pen squirted out at 10-18 deg. pen_gravity_pivot/layered found a
two-pad pinch is a near-free hinge about its pinch line (pen swings as a pendulum). So:
- Grip: scissor (cigarette) grip. Pen in the slot between the index and middle fingers (pen ~ palm x), squeezed
  by abduction with `grip(pen, {index, middle})`; pinch line = palm y = the swing axis. Finger sides are straight
  along the finger, so the pen can't squirt along palm z the way it does between two domed pads.
- Swing ("hybrid"): palm turns about the index/middle MCP axis while both MCPs counter-rotate by the same angle,
  so fingers + pen stay still in the world and only the palm turns. Works whether the pinch is a free hinge or
  rigid. Closed loop on a replica of the scorer's tilt.
- Roll: index pad back toward the palm, middle pad forward (perpendicular to the pen) -> pen rolls between them.
- Layers: joints (slot preshape, hybrid MCP counter-rotation), fingertips (index spread, roll shear), force (grip),
  arm (carry / palm rotation). No contacts primitives.
- Geometry read once from the hand model file (dg5fs_right.xml + link mesh bounds, not the CPU kinematics):
  MCP axis points (index 14.3/28/107.8 mm, middle 14.3/5/112.8 mm), link widths 19.6 mm across, 16.2 mm thick,
  so the natural index-middle gap is ~3.4 mm and the index must spread to admit the pen.
- Reused: `Mon`/`W(gen)` scorer replica + per-step wrapper and the TS compact time-series logging from
  pen_finger_roll/force; mock-api dry run idea from several agents (scratch/mock.py).

## iter 1: 3/3 (sev 0, 0.5, 1 all PASS; filmed sev 1 overview)
- Policy as planned above. Spread index pad +17 mm (IK abduction -0.17), slot gap ~18 mm; descent put the pen within
  0.6 mm of the slot centre; grip servo closed 10-11 mm travel to 1.3 N, held 1.3-1.8 N through everything.
- Swing (hybrid palm-about-MCP + MCP counter-rotation): tilt tracked the palm angle ~1:1 on the + side
  (th 15.6 deg -> tilt 15.9); on the - side the pen leaned 6-7 deg in the world (th -20 for tilt -16), i.e. the pinch is
  neither a perfectly free nor a rigid hinge, so the closed loop on the scorer replica mattered. Range 32 deg, settled.
- Roll: index pad 3-5.6 mm back toward the palm, middle 0.4x forward -> roll 41-48 deg in 1.5-2.8 s, ~8 deg/mm
  (close to pure rolling, d*1.4/(2r)). Tilt drifted +6 deg during the roll (harmless, settled already latched).
- Force control: essential to the grasp. One target (1.5 N) closed an 18 mm slot on pens 6.7-8.4 mm thick with no pad-offset
  model, and kept force within 1.2-1.8 N while the MCPs swung 0.6 rad.

## iter 2: 3/3 (same policy, new seeds) -> stop rule met
- Swing ranges 31.7-32.0 deg (min side 15.8), roll 46-48 deg, clearance 49-51 mm, settled in all.
- The swing range margin is only ~2 deg above the 30 deg rule. It is set by the closed-loop stops at +-15.5 deg,
  so it is consistent rather than lucky. Widening the targets to +-18 would be the first change if more margin were needed.
- FINAL = 2 (identical to iter 1).

## Skills written (reusable)
- `approach(c)`: put a palm-frame slot point on a world point above the pen, correct with the measured palm, descend
  vertically so the pen slides up into the slot.
- `cmd(th)` / `seg(T, done, tmax)`: the hybrid swing. Palm rotated by th about the MCP axis (fixed in world) with
  MCP = MCP0 - th on the holding fingers, so the held object stays put and only its in-palm angle changes. Closed loop on
  the scorer replica, with a sign check from the first response.
- Roll shear: pads moved in opposite directions along s = y x pen_axis, closed loop on the replica's twist range.
- `pad_model`: analytic planar finger FK from model-file link lengths (to predict the slot before moving; matched
  measured pads to ~1 mm).
