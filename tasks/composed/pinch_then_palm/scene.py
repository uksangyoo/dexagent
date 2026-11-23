"""Long-horizon task (RQ3): the cube_pinch scene, to be solved by chaining the cube_pinch and
cube_palm skills. The scene, its object and its initial-condition variation are cube_pinch's."""
import runpy
from pathlib import Path

_pinch = runpy.run_path(str(Path(__file__).resolve().parents[2] / "hand_demos/cube_pinch/scene.py"))
scene = _pinch["scene"]
routines = {}  # no expert routine: the point is to compose learned skills
