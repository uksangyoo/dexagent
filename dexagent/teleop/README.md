# Teleop skill collection (Vive + Manus → dexagent)

Self-contained clutch teleop: **no rio-hw**. Drivers talk to `openvr` and `manus_glove` directly.

## See the hand (recommended)

```bash
# SteamVR running (null HMD), tracker on, Manus plugged in
tasks/live tasks/hand_demos/cube_pinch up
```

Open the printed URL, press **Run** on **`vive manus collect`**.

| Key | Action |
| --- | --- |
| `c` | Clutch engage / disengage (arm follows tracker only while engaged) |
| `n` | New episode (reset cube + hand) |
| `s` | Save demo (`success` if cube lifted ≥ 80 mm) |
| `q` | Quit teleop |

Focus can be anywhere — keys are OS-wide via pynput. Watch the purple cube and DG5F-S in the viewer while you pinch-lift.

Demos: `tasks/hand_demos/cube_pinch/demos/<stamp>/`

Then:

```bash
$PY -m dexagent.skills.pack tasks/hand_demos/cube_pinch --n 5
# Run "skill replay" on the live page, or:
$PY -m dexagent.skills.replay_skill tasks/hand_demos/cube_pinch
```

## Dependencies

```bash
pip install openvr pynput
pip install 'manus_glove @ git+https://github.com/etaoxing/manus_glove.git'
```

SteamVR setup (null HMD, tracker pairing): same as usual Vive-on-Linux docs.

Optional calibration files next to the scene (or cwd):

- `manus_glove_cal.json` — open/closed glove ranges (compatible with prior rio cal files)
- `vive_yaw_cal.json` — `{"yaw_deg": …}` SteamVR↔table yaw offset

## CLI (headless)

```bash
$PY -m dexagent.teleop.collect tasks/hand_demos/cube_pinch --source hw
# scripted demos without hardware:
$PY -m dexagent.teleop.collect tasks/hand_demos/cube_pinch --source routine --episodes 5
```
