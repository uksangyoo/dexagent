# Goal → policy code/graph → hand abstraction → 20 + 6 motors

This package follows Graph-as-Policy's general structure without its harness. An authoring agent consumes a language goal and sampled video frames, emits an observation-conditioned graph, and receives execution feedback for revision. The compiler emits executable Python node functions from the validated graph. The runner executes those functions at 20 Hz. A selected hand adapter lowers each typed command to 26 joint references and bounded motor efforts at 500 Hz; Drake SAP integrates contact dynamics at 4 kHz on CPU.

The examples in `artifacts/agentic_workflow` were authored by Codex in this conversation after inspecting the human reference. They are not independent model samples or an evaluation of a configured model API. The external-generator protocol is implemented and tested with a stub; a live model provider is not configured. The website inspects recorded executions and exports requests; it does not run an LLM or physics in the browser.

## Run locally

From the repository root, use the installed Drake environment. GPU-capable commands must select physical GPU 1. This physics backend uses CPU.

```bash
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
PY=.venv-drake/bin/python
$PY -m dexagent.agentic.workflow prepare --goal 'From a prepared grasp, turn the screw clockwise 30 degrees, then hold within 5 degrees for 0.3 seconds.' --video screwdriver_site/dist/assets/human-reference.mp4 --output /tmp/new-goal
# An agent reads request.json and its sampled images and writes candidate.json.
$PY -m dexagent.agentic.workflow compile --request /tmp/new-goal/request.json --candidate /tmp/candidate.json --output /tmp/new-policy --author 'Name and provenance of actual authoring agent'
$PY -m dexagent.agentic.run --bundle /tmp/new-policy --name unique-trial-name --duration 4
$PY -m dexagent.agentic.workflow feedback --result artifacts/hom_revision/unique-trial-name.json --output /tmp/repair-request.json
```

Alternatively, `workflow generate --request ... --command '/path/to/provider-adapter' --output /tmp/new-policy` invokes a configured program. Stdin contains request_sha256, request_directory, request, and absolute image_inputs. The program must actually load the images for visual reasoning and return only a candidate graph JSON on stdout. It receives CUDA_VISIBLE_DEVICES=1. There is no canned policy fallback. Credentials belong in the provider's environment, not the graph or site. A revision uses the original request plus the saved repair packet and must retain the goal and physical acceptance criteria.

## Bounded agent revision loop

`python -m dexagent.agentic.loop --request path/request.json --goal-contract artifacts/agentic_workflow/goal-contract.json --generator-command /path/to/provider-adapter --output /tmp/unique-campaign --max-attempts 3` runs author → compile → execute → feedback until success or the attempt budget. Subsequent generator inputs include execution_feedback and previous_graph. The independently supplied goal contract is compared exactly on every attempt, so an agent cannot improve its score by loosening the target or hold requirement. No live-provider campaign has been tested; the displayed revision was authored in this conversation from recorded feedback.

## Distinct contracts

| Contract | Agent operation | Adapter controls |
|---|---|---|
| Fingertip paths | `finger_paths` | World-frame fingertip arcs |
| Rigid object twist | `object_twist` | Tool twist mapped to fingertip translation/orientation |
| Relative contact (HOM-inspired) | `relative_contact` | Object rotation and relative contact sliding, with two rolling coordinates released |

All three expose bounded speed and stroke size, `acquire`, and `hold`. Contact regulation, drive/reset microstates, inverse kinematics, and motor feedback stay below the graph. These are distinct controllers behind a common experiment interface, not three different names for the same controller. The agent cannot change friction, assign object trajectories, apply object torques, or alter actuator limits. Twelve hand joints participate in this pinch; eight other hand joints and six arm joints hold posture. All 26 receive commands.

Graphs use typed observations: measured screw angle/depth, contact readiness/loss, tilt, unexpected force, seat contact, sustained hold, and elapsed time. Failure transitions are explicit. Graph code and source hashes, initial configuration, commands, measurements, and transitions are saved. A graph's success terminal is independently checked against measured angle and a continuous physical hold; declaring success does not make the trial pass.

## Scope and evidence

The runnable goal is a prepared-grasp rotation-and-hold subgoal. Full pickup, pivot, cap acquisition, insertion and seating are not supported as one successful execution; the runner rejects a full-task initial condition rather than silently teleporting into a grasp. Physics parameters remain uncertain and are not hardware-calibrated. See the separate frozen nine-run manipulation study for initial-condition sensitivity. Do not infer a general abstraction ranking from this single goal/initial state.

The three v1 graphs share the same guards, initial state and physics. Their only intentional policy difference is the selected abstraction. Any later repaired graph is a development result and must be shown separately from that matched comparison. Full raw motor traces are at 500 Hz; website traces are sampled for inspection. RTX replays use recorded states at native time on GPU 1; they do not drive dynamics.

Run `python -m unittest tests.test_agentic_workflow` for contract, code-generation, routing and provider-protocol tests. The protocol stub is not a live model test.

## Physical pickup and cube development runners

The prepared-screw runner above retains its original limited contract. Two
additional runners now execute continuous physical pickup sequences:

- `tasks.screwdriver.sequence.pickup --assembly --policy <bundle>` executes
  pickup, loaded pivot, cap support, transport, engagement, drive and hold.
  A full sequence is supported in code but has not yet passed all stages.
- `tasks.cube.run --policy <bundle>` executes acquire, lift, in-hand
  rotation and hold. Graph commands carry bounded pressure, speed, pusher
  offset and angular contact mode. The chosen adapter maps these into all 26
  bounded motor commands. `--resume` is a prepared-grasp diagnostic and is
  permanently excluded from full pickup credit.

`python -m tasks.cube.workflow prepare --goal 'Pick up the 40 mm cube, rotate it 30 degrees about the pinch axis without turning the palm, then hold.' --output /tmp/cube-goal`
creates a request with immutable acceptance criteria. An agent supplies a
candidate graph; `cube.workflow compile --candidate candidate.json --request
/tmp/cube-goal/request.json --output /tmp/cube-policy --author 'actual author'`
validates and generates executable Python. The runner requires matching goal,
axis, object size and abstraction. Physics edits and unknown command arguments
are rejected. The supplied bundle is accepted with `cube.run --policy
/tmp/cube-policy --goal 30 --axis y --edge .04 --balanced --pulp --interface hom
--initial-from cube40-hom-pulp-v1 --name unique-name`.

`--initial-from` reuses only a recorded time-zero, clear-finger scene, with zero
initial velocity; it does not begin from a lifted grasp. `--yaw` perturbs only
the initial cube and rejects penetration. Full measurements, source snapshots,
graph decisions and failures remain in `artifacts/cube_revision`. The current
matched 30-degree study includes a complete pickup and stable-hold success for
relative-contact control (`cube40-pitch30-hom-full-v12`). The two other adapters
collide from the same starting state. Quarter-turn and perturbed-start success
remain unproven; these are not general agent success rates.

The cube compiler also accepts optional `tangent_kp` (N/m) and `tangent_kd`
(N s/m) per graph node. These select the bounded robot motor impedance; they
do not change the contact material or apply an object wrench. Omission retains
the historical joint-impedance inner servo. Phase-specific changes and repaired
graphs must be reported separately from a frozen matched-controller comparison.

`python -m tasks.cube.workflow feedback --result <trial.json>
--output <feedback.json>` returns the immutable goal, measured final state,
continuous hold, near-goal angular-speed distribution, unexpected force, all 26
peak motor efforts, source-record hashes and previous graph to the authoring
agent. The current conversation uses this evidence to author revisions; no
independent live-provider benchmark is implied.

The full screwdriver runner has its own goal compiler:
`python -m dexagent.planning.workflow prepare --goal 'Pick up, pivot
through loaded slip, and drive the screw to its seat' --video
reference/screwdriver-five_hands-compressed.mp4 --output /tmp/sequence-goal`.
Its `compile` and `feedback` subcommands use the same request-binding pattern.
The assembly node chooses `paired` or `alternating` drive/reset, side pressures,
and angular speed through a validated hand command. These parameters configure
the motor controller, never the contact solver. The `sequence-balanced-ramp-v7`
bundle records the failed abrupt-release trial that motivated its revision.
Prepared drive diagnostics remain excluded from complete-task credit.
