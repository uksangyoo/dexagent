# card_pickup at level joints


## iter 1: 0/3
- Built in-run pad FK from measured (api.q, api.tips) pairs: 5 s of sinusoidal finger sweeps, exact multilinear model
  (product over 4 joints of [1,cos,sin] = 81 features/finger, lstsq). Held-out error 0.0 mm. IK = scipy least_squares on it.
  (Skill: `Hand.calibrate/fk/ik` in policy.py -- joint-level IK built from api observations only.)
- Strategy: palm-down, index+middle pads press tile centre and drag it -y until COM is 10 mm inside the book's near edge (works
  in all 3 cases, ~0 tilt); then fingers-down hand facing the book, index under / thumb over the overhang, close, lift.
- Failure: squeeze only ~0.6 N per pad (index lifted tile instead of compressing) and a 2-point pinch is a free hinge for
  the tile's droop; tile pivoted in the pinch while far end dragged on book and slipped out (max lift 35 mm).

## iter 2: 3/3
- Lever grip: index under 4 mm deeper (toward COM), thumb on top 4 mm shallower, so the pinch resists the tile's droop
  torque instead of being a hinge; pinch midpoint 31 mm from tile centre, auto-pushed out if index would be < 17 mm from
  the book face. COM margin 7 mm. Index IK bounded to no PIP/DIP hyperextension.
- Force-closed squeeze: after the straight-line close, step each pad 0.3 mm/step until its contact force >= 2 N
  (skill: closed-loop squeeze on api.contact_forces). Lift 15 mm straight up first, then up+away from the book.
- Result: all pass, lift ~133 mm, tile 25 mm from pinch midpoint, tile tilts ~33 deg in the grip, no book contacts.

## iter 3: 3/3 (unchanged policy, confirmation run)
- All pass again (lift 123-134 mm, tile 25-28 mm from pinch midpoint). Two cases tilted ~57 deg in the grip (vs ~33 deg in
  iter 2), so the grip's droop margin is the weakest point, but it held. Stop rule met (3/3 twice); FINAL = 3.
