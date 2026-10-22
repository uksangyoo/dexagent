# pen_pickup_pivot, phase 3 (force level + cross-agent memory, 6 iterations)


Prior work read: pen_pickup_pivot/{contacts,force,joints,layered,synergies} NOTES+policies (all 0/3: pen squirts out of the
thumb-index wedge within 5-15 deg), pen_gravity_pivot/{layered,force} (hinge/arc pivot about the table tip works once the pen
stands; clamped roll fails), pen_table_slide/force+contacts (vertical pen held by a force-servo pinch / V-cage), QUALITATIVE.md,
REPORT.md (expert only 4/10 on this task).

Plan (before iter 1): "stub ledge". Pinch thumb (near) + index (far) 33 mm from the pen centre toward the -y end A, force servo
0.8 N with palm-z directions given in the object frame. Lift the pinch 10 mm while the middle stays at table level (pen pivots on
end B, the 29-45 mm stub beyond the pinch rises ~12 mm), then hook the middle under the stub (distal link down-and-back, pad
facing up-back, force servo 0.3 N). Pinch + stub ledge + B on the table is a statically stable beam, so the wedge can no longer
push the pinched end down. Then rotate the whole hand rigidly about B's cap centre (axis -palm z, "thumb-down" roll) so the pen
turns about B with no motion relative to the fingers until it stands; then firm the grip to 1.1 N, lift 45 mm, level, hold.
Reused: rotvec/rotm + measured-palm re-centred grasp and arc idea (pen_pickup_pivot/force), obj_dir + palm-frame grip directions
(pen_gravity_pivot/force), buffered notes + print() diagnostics to log_*.txt (pen_pickup_pivot/joints), mock api (scratch/mock.py,
adapted from pen_pickup_pivot/force/scratch/mock.py).

## iter 1: 0/3 (all three identical)
Grasp good (thumb 0.77 / index 0.58 N, pads 10.4/11.7 mm above the table). The fast 14 mm wedge lift WORKED: pen end up
(elev 7.9 deg, stub axis 20 mm above the table, forces steady 0.6-0.65 N) - the first lift-off any agent logged with forces
holding. The middle reached the hook pose (IK 0.6 mm) but stopped ~1 mm short of the stub (0 N, and grip() returned at once
because |0 - 0.3| <= 0.3 counts as settled). Pen then squirted out violently (flung ~85 mm) during the first 2.5 deg pivot step,
~0.5 s after lift-off. Pad-centre geometry: thumb 7.4 mm above / 10.8 mm beside the axis, index 9.1 above / 8.0 beside (steep).
Retry re-planned onto a pen 90 mm away (thumb IK residual 94 mm) and swept it off: now refused (residual > 5 mm).
Change for iter 2: middle pre-curled (pad facing up-back) during the grasp so hooking is one short straight push 1.5 mm INTO
the stub, wait for middle force > 0.12 N before pivoting; lift 14 mm in one 0.14 s move; per-step trace prints (skill `trace`).

## iter 2: 0/3 (all three identical)
The pre-curled middle reached the stub and loaded it (0.47-0.5 N; thumb rose to 1.0 N, index 0.6 N): a 3-finger hold with
the end up 9 deg. But the first 2.5 deg rigid-roll pivot step lost it within 0.04 s: trace shows the stub slid DOWN past the
middle on the thumb side (middle pad went from 6.3 mm below / 10.7 mm beyond the stub axis to level with it) and the pen was
flung 300 mm. The ledge contact at gamma=60 deg (30 deg below the equator on the far side) leaves the near-bottom quadrant open
(thumb ~150 deg, middle ~330 deg: a 177 deg gap), so the pen squirts down-and-toward-the-thumb.
Change for iter 3: ledge further under the pen (gamma 35 deg: contact 55 deg below the equator, gap ~155 deg), 18 mm lift so
the curled tip clears the table, two-stage hook (slide under 2 mm below the stub, then up into it), and the pivot is now a
hinge arc (pure palm translations, as in the lift that held) with the middle ledge following under the stub every step
(skill `middle_follow`), dropped when its IK residual exceeds 3 mm. Trace now logs measured-vs-commanded palm.

## iter 3: 0/3, but max pivot 67-72 deg in all three (min tilt on the table 21-24 deg, stood needs < 20)
The gamma-35 ledge + hinge arc works: lift-off, hook (middle 1.3-1.6 N at first, then ~0.4 N) and the pivot climbed smoothly
from 9 to ~63 deg elevation with the arm tracking within 1 mm (so iter 2's loss was the ledge geometry, not the arm). Failure,
identical in all cases: at ~55-63 deg the index lost the pen (index force -> 0, its servo travel crept 5 -> 9 mm) while the
middle force rose to 0.7 N: the following ledge's push has a component toward the thumb that walks the pen out of the index.
Also my B estimate used r from the unsettled start pose (4.25 mm vs 3.75): h_est 3-4 mm short, so the "B lifted" correction
pushed the pen into the table late in the arc.
Change for iter 4: r from the settled pose before the grasp, h re-estimated during the arc (12-45 deg), push-down only if B
seems > 6 mm up; ledge penetration 1.5 mm -> 0.5 mm (>30 deg) -> -1 mm gap with 0.15 N servo (>40 deg), ledge dropped at 72 deg;
smaller arc steps above 60 deg; pivot runs to 86 deg.

## iter 4: 0/3, but sev 1 stood_on_end=true (min tilt on table 17.9 deg); max pivot 69-74 deg in all
r/h estimates now right (h_est 66.2 mm on the nominal pen, lowB ~0). Ledge went soft (0 N) above 40 deg as intended. Still lost
at 63-70 deg elevation in all three: the index (or both pads together) loses force even when the middle carries nothing. In the
hinge arc the stub (25-40 mm of pen beyond the pinch) rotates up toward the palm, between the thumb and index links; by ~60 deg it
is where the index's middle phalanx is, so it levers the pinch open. Any hinge pivot with this pinch has a stub-clearance limit.
Change for iter 5: hinge arc only to 52 deg, then a rigid roll of the whole hand about B (axis perpendicular to the pen through its
table end, recomputed each step from the pen pose), so the pen no longer moves relative to the fingers; ledge dropped at 42 deg.

## iter 5: 0/3 (regression: max pivot 48-55 deg)
Hinge arc + ledge identical to iter 4 up to 53 deg, then the first rigid-roll step (2 deg about B, arm tracking error 0.1 deg /
<1 mm) lost the pen in all three cases within 0.1 s - the same instant loss as iter 2's roll. So rotating the hand while the pen is
pinched breaks the pinch here, although the same pinch displacement done as a pure translation (hinge arc) holds; I could not
isolate why (finger targets only change after the loss starts). Lesson: keep the palm orientation fixed with this pinch.
Change for iter 6 (last): no roll at all (hinge arc all the way, no rotation about B when leveling); when the ledge is dropped at
42 deg, re-aim the index's distal link (move_tips direction, tip end tilted back 19 deg) so its middle phalanx leans away from where
the rising stub goes; level about the pinch only if the final tilt > 12 deg; full per-step trace.

## iter 6: 0/3 (budget used)
Re-aiming the index while gripping (at 42 deg) dropped the pen in 2/3 cases (forces 0 right after the 0.3 s re-aim); the third
survived the re-aim but lost the pen again at 62.5 deg, so tilting the index distal link 19 deg did not clear the stub either.

## Summary / FINAL = 4
Dev results: 1: 0/3, 2: 0/3, 3: 0/3, 4: 0/3, 5: 0/3, 6: 0/3. FINAL = 4: the most progress (max pivot 69-74 deg in all three cases,
stood_on_end in the severity-1 case at 17.9 deg, correct length estimate, no crashes). Every iteration still ends 0/3.
Main failure modes:
- (solved) Lift-off: a fast 14-18 mm pure-translation lift of the thumb-index wedge holds for ~0.3 s; a curled middle hooked 55 deg
  below the equator under the stub, plus the pen's far end on the table, then holds the pen indefinitely (iter 3+).
- Any hand rotation with the pen pinched (rigid roll about the table end, iter 2 and 5; leveling) loses it within 0.1 s.
- Hinge arc stalls at 60-70 deg: the stub beyond the pinch swings up into the index's links and levers the pinch open.
What I'd try next: pinch closer to end A (shorter stub, ledge at ~15 mm, so the arc can reach 85+ deg), or pinch with the index
coming from the A side so its links are not above the pinch line; and understand why a pinched pen cannot survive a palm rotation.
Skills written (policy.py): `trace` (wrap any motion generator, print per-step pen-vs-pad geometry and forces to log_*.txt),
`axis_at_y`/`pen_palm` (pen axis in the measured palm frame), `middle_follow`/`middle_nh` (ledge pad target and distal-link
direction under the stub for the current pen pose), `pivot` (closed-loop hinge arc about the estimated table-end cap centre with
on-line half-length estimate), `palm_err` (commanded vs measured palm), `x_table` (table height in the palm frame).
Force control: used for the thumb-index pinch (0.8 N, palm-z directions in the object frame, which stay valid through the hinge
rotation), the middle ledge (0.3 N, direction toward the stub axis) and the hold (1.1 N). It mattered: the servo found contact on a
pen of any radius and the ledge servo kept the middle loaded while its target moved; but grip()'s settle test treats 0 N as settled
for a 0.3 N target (|err| <= 0.3), which hid a missed contact in iter 1.
