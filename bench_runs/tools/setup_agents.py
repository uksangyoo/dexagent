"""Create bench_runs/agents/<task>/<level>/ workdirs."""
import json, re, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("joints", "fingertips", "synergies", "contacts", "layered", "force")
MAX_ITERS = 4
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
LV = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set(LEVELS)
for scene in sorted(ROOT.glob("tasks/*/*/scene.py")):
    t = scene.parent
    if ONLY and t.name not in ONLY:
        continue
    if not (t / "TASK.md").exists() or not (t / "check.py").exists():
        print("skip (no harness)", t.name); continue
    cams = ["reference", "overview", "closeup"] + re.findall(r'CameraSpec\(\s*name="([^"]+)"', scene.read_text())
    for lv in [l for l in LEVELS if l in LV]:
        w = ROOT / "bench_runs/agents" / t.name / lv
        if w.exists():
            continue
        w.mkdir(parents=True)
        shutil.copy(t / "TASK.md", w / "TASK.md")
        (w / "agent.json").write_text(json.dumps({"task": str(t.relative_to(ROOT)), "level": lv,
                                                  "max_iters": MAX_ITERS, "cameras": sorted(set(cams))}, indent=1))
        (w / "policy.py").write_text("import numpy as np\n\n\ndef run(api):\n    yield from api.hold(0.5)\n")
        (w / "NOTES.md").write_text(f"# {t.name} at level {lv}\n\n")
        print("created", w.relative_to(ROOT))
