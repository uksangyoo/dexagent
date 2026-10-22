# baoding at level contacts


Approach: palm gyration. Fingers stay in the home cup and act as walls with the thumb; the arm tilts the palm
(~12 deg, pivot at the palm centre) toward a downhill direction psi in the palm plane, and gravity rolls the balls.
The pair angle is tracked online the same way check.py does, and the policy stops once it passes the goal.
Skills (policy.py): `Ctl.step(tau)` is a per-step tilt servo with a rate limit and an edge reflex that tilts away
when a ball nears the open little/wrist edges; `arc/ramp/dwell` shape the tilt; `Ctl.obs` is an online copy of the scorer.

## iter 1: 0/3 (exploratory gyration sweep, thumb->fingers->little, 12 deg tilt)
- Learned geometry: at level the balls roll into a "pocket" at the finger bases (z~90-103), side by side (a on the little side, b on the middle), angle ~70 deg at t=1. The thumb at home is splayed wide (pad y=+101), so the palm's +y side below the index is OPEN. The little side is open below the little finger.
- Sweep T->F->L just slid the pair along the pocket (rotation <10 deg). In 2/3 cases ball a went out over the open thumb side when tilted +y; in the nominal case it went off the little side at psi=-115. The closeup camera shows only the table.
- Next: close the +y side with the thumb (pad placed as a wall at y~+52, z~65), then run a ratchet. psi~+100 sends the -y ball around the +y ball and down the thumb wall. psi~-30 slides the upper ball along the pocket to -y so the lower ball rises into the pocket on its +y side. Each step turns the pair -90 deg and none of them reverses.

## iter 2: 0/3 (thumb wall + ratchet)
- The thumb wall works. place_pad(thumb, (42,50,70)mm) has 0.1 mm IK residual (fingertip IK built from approach(): the pad is aimed at an absolute palm point through a reference ball). Step A (psi 105) worked in 3/3 cases: the -y ball rolled around the +y ball and down the thumb wall, -30..-58 deg.
- Step B (psi -30, 12 deg) failed in 3/3. The lower ball did not rise into the pocket. It raced to -y under the upper ball, which undid step A, and then hopped over the low little finger and dropped.
- Next: in step B the thumb pins the lower ball from above (grasp with 4 mm squeeze). A gentle 8 deg tilt to -y slides the upper ball along the pocket. Then unpin, tilt toward the fingers, and the freed ball rolls up into the pocket on the +y side. The little-side reflex is now at y < -41 mm.
- Skills added: `place_pad` (absolute pad placement through approach), `pin`/`unpin` (thumb pins a ball, picking the best-reaching contact direction by api.reach).

## iter 3: launched (thumb-pin step B); I was forced to hand back while it was still queued, so the result is unknown. FINAL=2 for now. Switch FINAL to 3 if iter_3 scores >0.
