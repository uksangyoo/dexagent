# pen_table_slide at level force


## iter 1: 0/3 (slip 7.7-9.6 mm counted, never lifted)
- Policy: hand-down thumb-index pinch (pinch line along palm z, hand.py PINCH_CENTRE) 16 mm below pen top, force grip 0.9 N (api.grip), lower palm 22 mm in one stroke, then grip 1.6 N and lift 130 mm.
- Grasp/approach worked in all cases (IK residual <0.3 mm, forces ~0.95 N each after grip). During the slide the pen escaped the pinch sideways (perpendicular to the pinch line): index force -> 0, both servos travelled ~8 mm toward each other, pen ended leaning ~6-7 deg in the bore (its max lean). My own slip estimate (23 mm) ignored the held gate; scorer counted ~9 mm before contact was lost.
- Lesson: while sliding axially the friction cone is saturated, so a two-point pinch has ~no lateral stiffness and the axially compressed pen is an unstable column; it falls sideways out of the pinch. Force servo along the pinch line cannot fix that direction.

## iter 2 (submitted): lean preload + chunked, monitored slide + regrasp loop
- After the force grip (1.0 N, api.grip), push the palm along palm -y (perpendicular to the pinch line) in 1 mm steps until the pen stops following (it rests on the bore), +1.5 mm; then slide with grip 0.7 N, palm moving half straight down / half down the pen axis, 2 mm chunks, checking thumb+index > 0.22 N and servo travel; abort/regrasp (ungrip, open, re-centre at the same height) if contact is lost or held slip < 20 mm. Lift: grip 1.6 N, pull 75 mm along the pen axis, then 60 mm up.
- Skills: `lateral(api)` palm-y offset of pen axis from the pad midpoint; `rel(api)` pen-vs-thumb-pad axial coordinate (mirrors the scorer); `held(api)`; `grasp_at(...)` centred pinch from pen pose. Force control used for every grasp/slide/lift phase (servo keeps the pinch closing as the pen moves).
- Result iter 2: 3/3 (slip 22.7-23.7 mm, final lower end ~132 mm up, held 1.5 N each). The lean push never detected a stop (pen followed the full 13.5 mm, tilt ~9 deg) but the pre-lean alone stabilised the slide: no contact loss, one stroke each, forces stayed ~0.5-0.6 N. Force servo kept the pinch at target through the lean, slide and lift (travel <3.5 mm); it mattered mainly as an automatic pinch that tracks the moving pen, the lateral stability came from the lean geometry.

## iter 3: 3/3 (policy unchanged from iter 2, confirmation run)
- Slip 23.1-23.4 mm, final lower end 132-134 mm, held 1.5-1.6 N, incl. block yaw -23 deg / dy -19 mm at severity 1. Stop rule met (3/3 twice in a row). FINAL = 3.
- Level usage: contacts layer (tuck) for the idle fingers, fingertip layer (move_tips) for preshape/closing in the palm frame, arm (carry) for the approach, lean, slide and lift, and the force layer (grip/ungrip) for every holding phase. No joint-level or synergy-level calls; I wrote no custom IK.
