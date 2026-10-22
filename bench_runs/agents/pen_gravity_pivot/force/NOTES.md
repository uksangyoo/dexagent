# pen_gravity_pivot at level force


## iter 1: 0/3 (all: pen knocked off the rest at grasp, never lifted)
- Plan: fingers-DOWN hand, index+middle curled under the pen overhang as a ledge (position), thumb on top
  (force servo 1 N, explicit object-frame direction) -> vertical pinch holds level by friction; then clamped
  90deg tilt, release middle, roll hand about the frictionless thumb-index hinge (pen stays vertical by gravity).
- Film: ledge pads under the pen OK (measured gap 4 mm = target), but the thumb came down off-axis and the
  convex fingertip ledge has no lateral hold; the pen rolled off sideways before the thumb force built (0 N).
- Learned: rest is at +12 mm along the pen axis in every case (free overhang is the -axis end, toward the robot);
  fingertip site positions match IK targets within ~0.5 mm; convex pads + cylinder = no lateral stability without opposition.

## iter 2: 0/3 (grasp + level hold now work in all 3: level 2.2-2.6 s, lift ~160 mm; pen dropped at the start of the clamped roll)
- Switched to palm-down horizontal thumb-index pinch at the hand's natural pinch point (palm (80,40,118) mm), pen 32 mm
  off-centre toward its free end; middle 22 mm further out on the index side. Force layer: grip thumb 1.5 N + middle 0.25 N
  (explicit object-frame directions, palm +-z), index = position anchor. Middle friction (22 mm lever) holds it level (tilt 2-9 deg).
- Failure: thumb servo saturated at max travel (15 mm) with only 0.5-1 N measured; at roll start the logged thumb joint
  targets jump between IK branches (unreachable servo target + object-frame direction that moves with the pen) -> pen dropped.
- Learned: object-frame grip directions feed back pen motion into the servo direction; the thumb runs out of reach ~6 mm past
  the pinch point, so the servo needs headroom (pinch point chosen with api.reach) and a smaller max_travel.

## iter 3: 0/3 (level hold 2.8-2.9 s in all; pen still lost within the first ~10 deg of the clamped roll)
- Added: pinch point chosen by api.reach scan (nominal point was fine), grip directions refreshed from the palm frame every
  5 steps during rolls (no more servo IK branch jumps), thumb max_travel 10 mm, roll 4 s.
- Result: contact forces all 0 already at u=0.2 of the roll. Pinch force was only thumb 0.45-0.5 N (servo saturated at
  max travel), index 0.3, middle 0.18: the pinch is about as strong as the fingers' own weight, so re-orienting the hand
  (finger gravity loads change) opens it and the pen swings out of the hinge and drops.
- Learned: with kp 3 fingers, a 6 mm commanded squeeze gives only ~0.5 N; the force servo needs much more travel
  headroom (or an opposing preloaded finger) to reach 1.5 N. Force servo saturation is the real limit, not friction.

## iter 4: 0/3 (budget exhausted). Level hold 3.3-3.4 s and ~160 mm lift in all 3; pen lost 10-30 deg into the clamped roll
- Changed: index position target preloaded 1.5 mm into the pen (an anchor that pushes back), thumb servo 1.8 N with
  22 mm max travel, middle 0.4 N, clamped roll 5 s, wait-for-swing-to-settle loop after releasing the middle.
- Forces now reach target (thumb 1.8, index 1.5, middle 0.35 N; travel 8 mm) and hold through the first ~10 deg of roll,
  but between u=0.2 and 0.4 the pen pivots about the thumb-index hinge (middle friction lost), swings down and then
  slides out of the pinch along its own axis (film). Root cause not isolated: the middle's friction lever (22 mm) is
  not enough once the hand re-orients, and a 2-pad pinch on a 7.5 mm cylinder does not stop axial slip when it swings.
- Next step I would try: no clamped roll at all; release the middle slowly in the palm-down pose (squeeze ramp-down
  with grip()) so the swing is braked by the middle's sliding friction, then roll back only after it settles; and
  pinch nearer the pen centre so the stub is longer than the pads are wide.

## Summary / force-layer usage
- Layers used: contacts (tuck), fingertips (move_tips for preshape, middle release), arm (carry, per-step pivoting about
  the pinch point), force (grip/regrip: thumb clamp + middle friction support; ungrip to "loosen").
- Force control mattered: it made the grasp of a pen whose diameter varies +-10 % work without per-case tuning, and the
  thumb force target was the knob that fixed the iter-3 drop; but saturation of servo travel (fingers kp 3) was invisible
  until I read grip_state, and object-frame grip directions need refreshing when the object turns.
- Helpers written (in policy.py): rot_about() rotates the palm about a world point fixed at a palm-frame point, one arm
  step per control step, with monitor/refresh hooks; regrip() re-issues grip() with palm-frame directions converted to the
  object frame; pinch-point reach scan with api.reach.
