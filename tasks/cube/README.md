# Cube

Pick up a 40 mm cube, rotate it about the pinch axis without turning the palm, then hold. It uses the same workcell, hand abstractions and planning stack as the screwdriver.

```bash
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 PY=.venv-drake/bin/python
$PY -m tasks.cube.workflow prepare --goal 'Pick up the 40 mm cube, rotate it 30 degrees about the pinch axis without turning the palm, then hold.' --output /tmp/cube-goal
# An agent writes candidate.json for /tmp/cube-goal/request.json.
$PY -m tasks.cube.workflow compile --candidate candidate.json --request /tmp/cube-goal/request.json --output /tmp/cube-policy --author 'actual author'
$PY -m tasks.cube.run --policy /tmp/cube-policy --goal 30 --axis y --edge .04 --balanced --pulp --interface hom --initial-from cube40-hom-pulp-v1 --name unique-name
$PY -m tasks.cube.workflow feedback --result artifacts/cube_revision/unique-name.json --output feedback.json
```

`model` builds the cube scene, `policy` compiles graph commands, `metrics` scores orientation and hold, and `arm_setup` plans the arm corridor. `export` and `site_report` prepare RTX bundles and website data. Contract details and limitations are in [dexagent/agentic/README.md](../../dexagent/agentic/README.md#physical-pickup-and-cube-development-runners). Outputs go to `artifacts/cube_revision`.
