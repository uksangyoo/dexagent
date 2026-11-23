"""Editor tools for the coding-agent loop: edit policy.py, run dexagent.score iterate, finish.

The model may only touch files inside its workdir. `write_policy` compiles the source and
scans for `api.<name>` calls against the forced level's public surface; failures are returned
as tool errors and do not consume an iteration. `run_iterate` shells out to
`python -m dexagent.score iterate` so budget and seed uniqueness stay in that module.
"""
from __future__ import annotations

import ast
import base64
import inspect
import json
import os
import subprocess
from pathlib import Path

from dexagent import levels as L

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")

_CLASSES = {
    "joints": L.Joints,
    "fingertips": L.Fingertips,
    "synergies": L.Synergies,
    "contacts": L.Contacts,
    "layered": L.Layered,
    "force": L.Force,
}

# Instance attributes that are not methods but that policies legitimately read.
_EXTRA_ATTRS = {
    "objects", "t", "level", "k", "open_coeffs", "pad_offset", "joint_limits",
    "OPEN", "DEFAULT_YAW", "GAIN",
}


def allowed_api_names(level: str, extra=()) -> set[str]:
    cls = _CLASSES[level]
    names = set(_EXTRA_ATTRS) | set(extra)
    for n, v in inspect.getmembers(cls):
        if n.startswith("_"):
            continue
        names.add(n)
    # Contacts.move exists only to raise AttributeError; policies must use carry + primitives.
    if level == "contacts":
        names.discard("move")
    return names


def api_doc(level: str) -> str:
    """Human-readable allowed surface for prompts: common API + level-specific methods with docs."""
    cls = _CLASSES[level]
    common = inspect.getdoc(L) or ""
    # Keep the module docstring header up through Common API.
    lines = [f"Forced representation: '{level}'.", "", common.strip(), "", f"## Level-specific API (`{cls.__name__}`)"]
    # Prefer the class docstring, then each public method's signature + first paragraph.
    if inspect.getdoc(cls):
        lines += ["", inspect.getdoc(cls).strip()]
    lines.append("")
    for name in sorted(allowed_api_names(level)):
        if name in _EXTRA_ATTRS and not hasattr(cls, name):
            continue
        attr = getattr(cls, name, None)
        if attr is None or name in ("level",):
            continue
        if callable(attr) and not isinstance(attr, type):
            try:
                sig = str(inspect.signature(attr)).replace("(self, ", "(").replace("(self)", "()")
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(attr) or "").strip().split("\n\n")[0].replace("\n", " ")
            lines.append(f"- api.{name}{sig}" + (f"  — {doc}" if doc else ""))
        elif isinstance(attr, property):
            doc = (inspect.getdoc(attr) or "").strip().split("\n\n")[0].replace("\n", " ")
            lines.append(f"- api.{name}" + (f"  — {doc}" if doc else "  — property"))
    if level == "contacts":
        lines.append("\nDo not call api.move on the contacts level; use carry() for the palm and the contact primitives for fingers.")
    return "\n".join(lines)


def scan_policy(source: str, level: str, extra=()) -> list[str]:
    """Return a list of problems (empty if the policy looks usable)."""
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    has_run = any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body)
    if not has_run:
        problems.append("policy must define a top-level function run(api)")
    # Ban importing level classes or the live agent tools.
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", None) or ""
            names = [a.name for a in getattr(n, "names", [])]
            joined = ".".join([mod] + names)
            if "dexagent.levels" in (mod or "") or any(x in names for x in _CLASSES):
                problems.append(f"do not import dexagent.levels classes; use only the api argument ({joined})")
            if "dexagent.agent" in (mod or ""):
                problems.append("do not import dexagent.agent from a policy")
    allowed = allowed_api_names(level, extra)
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "api":
            used.add(n.attr)
            if n.attr.startswith("_"):
                problems.append(f"api.{n.attr} is private; do not use it")
            elif n.attr not in allowed:
                problems.append(f"api.{n.attr} is not available at level '{level}'; "
                                f"allowed: {sorted(allowed)}")
    return problems


def compile_policy(source: str, level: str, extra=()) -> list[str]:
    problems = scan_policy(source, level, extra)
    if problems:
        return problems
    try:
        compile(source, "policy.py", "exec")
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    return []


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file inside the workdir (policy.py, NOTES.md, TASK.md, progress.jsonl, or an iter_*/case_*.json).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path inside the workdir, e.g. policy.py or iter_1/case_20000_0.json"},
        }, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "write_policy",
        "description": "Replace policy.py with a complete Python source that defines run(api). Compiled and API-scanned before saving; errors do not consume an iteration.",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "Full contents of policy.py"},
        }, "required": ["content"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "append_notes",
        "description": "Append a short lab-notebook entry to NOTES.md (what you tried, what failed, what you learned).",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
        }, "required": ["text"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_iterate",
        "description": ("Snapshot the current policy.py and score it on 3 fresh cases (severity 0, 0.5, 1). "
                        "Returns successes/3, per-case metrics, api_log tails, errors, and a contact-sheet image. "
                        "This consumes one iteration of the budget."),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the editing loop. Prefer the best-performing iteration; omit final_iter to let the harness pick the best.",
        "parameters": {"type": "object", "properties": {
            "final_iter": {"type": "integer", "description": "Iteration number to submit (optional)"},
            "reason": {"type": "string"},
        }, "additionalProperties": False}}},
]


class Workbench:
    """Stateful tool host for one (task, level) cell."""

    def __init__(self, workdir: Path, level: str, film: str = "closeup,overview", timeout_s: int = 3600,
                 extra_api=()):
        self.workdir = Path(workdir).resolve()
        self.level = level
        self.extra_api = tuple(extra_api)  # e.g. ('run_skill',) when composing
        self.film = film
        self.timeout_s = timeout_s
        self.finished = False
        self.final_iter = None
        self.finish_reason = None
        self.stats = {"write_rejects": 0, "writes": 0, "iterates": 0, "notes": 0}
        self.last_sheet: Path | None = None
        self.last_result_text: str | None = None

    def _resolve(self, path: str) -> Path:
        p = (self.workdir / path).resolve()
        if not str(p).startswith(str(self.workdir)):
            raise ValueError(f"path escapes workdir: {path}")
        return p

    def read_file(self, path: str) -> dict:
        p = self._resolve(path)
        if not p.exists():
            return {"error": f"not found: {path}"}
        if p.stat().st_size > 400_000:
            return {"error": f"file too large ({p.stat().st_size} bytes); read a smaller one"}
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".npy"}:
            return {"error": f"binary file {path}; use the sheet attached to run_iterate results instead"}
        text = p.read_text(errors="replace")
        if len(text) > 80_000:
            text = text[:80_000] + "\n... [truncated]"
        return {"path": path, "content": text}

    def write_policy(self, content: str) -> dict:
        problems = compile_policy(content, self.level, self.extra_api)
        if problems:
            self.stats["write_rejects"] += 1
            return {"error": "policy rejected", "problems": problems}
        (self.workdir / "policy.py").write_text(content if content.endswith("\n") else content + "\n")
        self.stats["writes"] += 1
        return {"ok": True, "bytes": len(content), "lines": content.count("\n") + 1}

    def append_notes(self, text: str) -> dict:
        notes = self.workdir / "NOTES.md"
        with open(notes, "a") as f:
            if notes.stat().st_size and not text.startswith("\n"):
                f.write("\n")
            f.write(text.rstrip() + "\n")
        self.stats["notes"] += 1
        return {"ok": True}

    def run_iterate(self) -> tuple[dict, Path | None]:
        """Returns (result_dict, sheet_path_or_None)."""
        cmd = [PY, "-m", "dexagent.score", "iterate", "--workdir", str(self.workdir)]
        if self.film:
            cmd += ["--film", self.film]
        env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return {"error": f"iterate timed out after {self.timeout_s}s"}, None
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode == 3 or "BUDGET EXHAUSTED" in out:
            return {"budget_exhausted": True, "stdout_tail": out[-2000:]}, None

        prog_path = self.workdir / "progress.jsonl"
        if not prog_path.exists():
            return {"error": "iterate produced no progress.jsonl", "returncode": proc.returncode,
                    "stdout_tail": out[-3000:]}, None
        rows = [json.loads(l) for l in prog_path.read_text().splitlines() if l.strip()]
        row = rows[-1]
        k = int(row["iter"])
        it = self.workdir / f"iter_{k}"
        cases = []
        sheet = None
        for cf in sorted(it.glob("case_*.json")):
            c = json.loads(cf.read_text())
            err = (c.get("error") or "").strip()
            cases.append({
                "case": c.get("case"),
                "success": bool(c.get("success")),
                "metrics": c.get("metrics"),
                "perturbation": c.get("perturbation"),
                "sim_s": c.get("sim_s"),
                "error_tail": "\n".join(err.splitlines()[-12:]) if err else None,
                "api_log_tail": (c.get("api_log") or [])[-20:],
            })
            if c.get("sheet") and Path(c["sheet"]).exists():
                sheet = Path(c["sheet"])
        # Prefer the filmed case's sheet; else any sheet_*.jpg in the iter dir.
        if sheet is None:
            sheets = sorted(it.glob("sheet_*.jpg"))
            sheet = sheets[0] if sheets else None
        self.last_sheet = sheet
        self.stats["iterates"] += 1
        result = {
            "iter": k,
            "successes": row["successes"],
            "n": row["n"],
            "by_severity": row.get("by_severity"),
            "wall_s": row.get("wall_s"),
            "cases": cases,
            "stdout_tail": "\n".join(out.strip().splitlines()[-40:]),
            "sheet": str(sheet) if sheet else None,
        }
        self.last_result_text = json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=1)  # compact cache
        return result, sheet

    def finish(self, final_iter=None, reason: str = "") -> dict:
        self.finished = True
        self.final_iter = final_iter
        self.finish_reason = reason or "model called finish"
        return {"ok": True, "final_iter": final_iter, "reason": self.finish_reason}

    def dispatch(self, name: str, args: dict) -> tuple[dict, Path | None]:
        """Run one tool. Returns (jsonable_result, optional_image_path to attach)."""
        try:
            if name == "read_file":
                return self.read_file(args["path"]), None
            if name == "write_policy":
                return self.write_policy(args["content"]), None
            if name == "append_notes":
                return self.append_notes(args["text"]), None
            if name == "run_iterate":
                return self.run_iterate()
            if name == "finish":
                return self.finish(args.get("final_iter"), args.get("reason", "")), None
            return {"error": f"unknown tool {name}"}, None
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}, None


def sheet_data_url(path: Path, max_side: int = 1024) -> str:
    """JPEG data URL for a contact sheet, optionally downscaled for the prompt."""
    import numpy as np
    import imageio.v3 as iio
    img = np.asarray(iio.imread(path))
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        # crude integer downsample
        step = max(1, int(round(1 / scale)))
        img = img[::step, ::step]
    import io
    buf = io.BytesIO()
    iio.imwrite(buf, img, extension=".jpg", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def used_api_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = sorted({n.attr for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "api"})
    return names
