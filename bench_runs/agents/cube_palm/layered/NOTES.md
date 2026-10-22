# cube_palm at level layered


Pre-iteration geometry (static, from the MJCF/mesh bounds, not kinematics): palm hull x in [-19, 26] mm, y +-41 mm, z 0..103 mm;
index MCP ~(14, 28, 108) mm. Conclusion: an index pad can never press the cube's +z face while the cube sits on the palm,
so the transfer must be a drop (open the pinch) plus optional finger rake, not a placed pinch.

Plan v1 layers: joints (tuck/flatten/rake/thumb open: whole-finger postures), fingertips (pinch preshape + reach() search
for a pinch centre nearer the palm), contacts (grasp from the measured cube pose), arm carry for rotations/pour.
Put-down: palm-up near the table, pronate ~65 deg so the cube slides off the little-finger edge, move the hand away.

## Resume (2026-10-22, restart). Earlier iter_1 was killed mid-run (no summary.json, not counted); its one finished
case (sev 0.5) showed pinch + palm-up work (thumb/index 2.6 N each, cube kept), but opening index then thumb dropped
the cube off the +y (thumb) edge: the default pinch centre sits at palm y=40 mm, the palm's edge. Folder renamed? no:
the iterate counter will start at iter_2.
Plan v2 transfer (contacts layer): search api.reach over rigid shifts of the held thumb+index pinch toward the palm
centre (-x,-y,-z), translate() there, roll thumb-side up if the remaining drop would still land off-centre, then
release() both pads straight off the faces; thumb -> OPEN and index -> FLAT by joints. Rake (joints, MCP flex of the
four fingers) if the cube ended on the fingers. Put-down unchanged: lower palm-up, pronate 66 deg, cube slides off.

### iter 2 (first counted): 1/3 (sev1 PASS, 8.7 s on palm, set down 1.7 mm off; sev0, sev0.5 fail)
Pinch shift search works: reachable shifts reach dx=-20..-30 mm, dy=-20 mm, dz=-10 mm (the cube even touches the palm
at dx=-30). Sev1 (shift -30,-20,-10, no roll): drop ~0, cube at z=105 on palm/finger boundary, rake (MCP flex 1.0)
slid it to z=69, 70 counted steps at once; pour put-down landed it 2 mm from start. Sev0/0.5: shift only -20 in x, my
roll rule then rolled 0.6 rad (drop 6 mm!) and the cube slid off the little-finger edge (once during release, once
during the levelling carry). Lesson: with friction 0.9 the cube stays where it lands on a near-level palm; never roll.
Change for iter 3: no roll/levelling; finer shift grid (5 mm) with score = drop + 0.7*(y-15mm)+ + 0.3*(z-100mm)+.
Skill: pinch-shift search = grid of rigid pad shifts checked with api.reach (fingertips IK), executed with translate().

### iter 3: 2/3 (sev0 PASS 7.6 s on palm, offset 71 mm; sev1 PASS 7.9 s, offset 47 mm; sev0.5 fail)
Shift [-25,-25,-10] chosen every case, cube touching the palm before release (drop 0-2 mm), lands at y 20-27, z 94-102,
70 counted steps at once. Sev0.5 failure: during release/opening the cube was carried to palm y=+108 (off the thumb
edge). Thumb release IK residual 12 mm (it cannot back off along -z), so the still-pressing thumb dragged the cube when
swung to OPEN (sev0 also showed +12 mm y drag). Pour put-down overshoots toward the little side by 0-72 mm (mean ~37).
Change for iter 4: release index first, then thumb, then lift thumb pad +x over the cube (reach-checked move_tips)
before the OPEN joint move; put-down aim point -40 mm instead of -75 mm, pronation slowed 2.3 -> 3.0 s.

### iter 4: 2/3 (sev0 PASS 9.2 s on palm, offset 35 mm; sev1 PASS 9.4 s, offset 61 mm; sev0.5 fail)
The index-first release + thumb lift worked when the pinch was intact (no y drag: cube at y 16-19, raked to z 71-83).
Sev0.5 failure was earlier: during the single 1.5 s translate the thumb lost contact and the index wedged the cube
onto the palm (index 9 N, palm 9 N); when the index released, the cube sprang off the thumb edge.
Put-down: landing distance out along palm -y is 74-147 mm (mean ~100) across all runs, so aim -100 mm next.
Change for iter 5 (last): shift executed in 8 translate() steps with force feedback (stop on palm touch >0.08 N or a
pad <0.4 N), slower index release (0.8 s), put-down aim -100 mm.
Skill: stepped_translate = contacts-layer translate in small steps checking contact_forces after each.

### iter 5 (last): 1/3 (sev0 PASS 9.2 s, offset 26 mm with aim -100; sev0.5, sev1 fail)
Stepped shift did not help: the force check comes too late (one 3 mm step already gives palm 4-6 N) and both failures
were again the wedged-cube spring-off on release. Pattern across iters 3-5: every failure has edge >= 1.03 (41-45 mm),
every pass edge <= 1.0. The shift target ignores cube size, so a bigger cube is driven into the palm, tilts, and is
wedged by the index; releasing it flings it off the thumb edge. Fix not tested (budget out): stop the shift at
cube-centre x = 26 mm + h + 3 mm (computed from h), i.e. dx = clip(26+h+3 - rel_x), plus finer steps.
FINAL = 4 (2/3, same as iter 3 but with the thumb-drag fix; put-down offsets 35/61 mm).

Layers used: joints (tuck/flat/rake/thumb-open postures and the put-down hand shape: whole-finger shapes where
joint angles are the natural description), fingertips (pinch preshape; api.reach as an in-policy IK oracle to search
reachable pinch shifts; thumb lift), contacts (grasp from measured cube pose, translate the held pinch toward the palm,
release pads along the face normals), arm carry for all wrist rotations and the pour. Synergies not used.
