# card_pickup at level force


## iter 1: 3/3 (sev 0, 0.5, 1 all PASS; lift ~140 mm, tile-to-pinch ~28 mm)
- Plan: (1) palm-down, index+middle pads pressed on tile top by force servo `grip(tile, {index:1.2, middle:1.2}, directions=-z)`; carry palm toward robot in a closed loop on measured tile y until tile centre is 8 mm behind book's near edge (overhang ~30-45 mm, still stable). (2) ungrip, raise, rotate hand to fingers-down/palm-facing-book (R_hook), so hand.py's thumb-index pinch axis (palm z) is vertical: thumb above, index hooked below. Open pinch (hw 24 mm) comes in from robot side (guarded on book/tile contact), closes to hw 12.5 mm (move_tips straight), then `grip(tile,{thumb:2,index:2})` along pinch line. (3) pull 40 mm off, lift 150 mm with servo on.
- Layers: joints (OPEN reset), contacts (tuck), fingertips (move_tips for press shape and pinch shape), arm (carry), force (grip/ungrip both phases). Force control used for the press-drag (constant 1.2 N normal regardless of pad/tile height) and the pinch hold (servo keeps ~2 N through lift; travel 4-8 mm). It removed all height/thickness tuning: book thickness variation was absorbed by the servo.
- Learned: initial thumb pad hangs ~59 mm below palm in OPEN, so I lift it (runtime reach() grid search, picked [0.03,0.05,0.07]) -- a "skill" built from api.reach. Drag converges in 2-3 closed-loop carries (finger lag ~2 mm). Hook pinch IK residual 0.1 mm. Tile droops ~15 deg in pinch, harmless.
- Helpers: `guarded(api, gen, stop)` runs a motion step-by-step and stops on a condition (contact). `R_hook()` fingers-down palm-toward-+y rotation.

## iter 2: 3/3 (policy unchanged; sev-1 case had tile yaw 17.5 deg, size -7 %, book +4.5 % thick)
- Same behaviour as iter 1: drag converges to centre 8 mm behind edge, pinch forces ~2.1/1.6 N at close, servo holds ~2 N through lift, final lift ~141 mm, tile-to-pinch ~28 mm. Stop rule met (two consecutive 3/3). FINAL = 2.
- Force control mattered most in the press-drag (constant normal force, no height tuning, tile followed pads with ~2 mm lag) and in the pinch (closing by force rather than a guessed squeeze depth; holds through the dynamic lift). Guarded approach never triggered.
