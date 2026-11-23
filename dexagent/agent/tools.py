"""Per-level tool schemas and the dispatcher from a model tool call to a `levels` api method.

Each level offers the model exactly the commands that level has, nothing from another:

  joints       move (20 finger joint targets)
  fingertips   move (palm-frame pad targets, solved by IK), open
  synergies    move (k posture coefficients)
  contacts     approach, grasp, squeeze, rotate, translate, release, relocate, tuck, open, carry
  layered      move_joints, move_tips, move_synergy + the contacts tools
  force        layered + grip, ungrip

Every level also gets `hold`, `note` and `finish`. Palm orientation is given as a
named preset plus angles (`palm_orientation`), never a raw 3x3 matrix.
"""
import inspect
import json

import numpy as np
from scipy.spatial.transform import Rotation

from ..workcell import hand_down, hand_up

FINGERS = ["thumb", "index", "middle", "ring", "little"]
MAX_SECONDS = 8.0
MIN_SECONDS = 0.1

# ------------------------------------------------------------------ argument schemas
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_FINGER = {"type": "string", "enum": FINGERS}
_FINGER_LIST = {"type": "array", "items": _FINGER, "minItems": 1}


def _finger_map(item, desc):
    return {"type": "object", "description": desc, "properties": {f: item for f in FINGERS}, "additionalProperties": False}


_SECONDS = {"type": "number", "description": f"Duration of this motion [s], {MIN_SECONDS}-{MAX_SECONDS}."}
_PALM_XYZ = dict(_XYZ, description="Target palm origin in the WORLD frame [m] (z up, table top z = 0.75). Omit to keep the palm where it is.")
_PALM_ORI = {
    "type": "object",
    "description": ("Target palm orientation. preset 'down': palm faces down, fingers forward; 'up': palm faces up. "
                    "yaw [rad]: fingers point this far from world +x about vertical (default 0.7). tilt [rad], 'up' only: "
                    "raises the palm toward the fingers. pitch [rad]: extra rotation about the palm's thumb-side axis. "
                    "Omit to keep the current orientation."),
    "properties": {"preset": {"type": "string", "enum": ["down", "up"]}, "yaw": {"type": "number"},
                   "tilt": {"type": "number"}, "pitch": {"type": "number"}},
    "required": ["preset"], "additionalProperties": False,
}
_OBJ = {"type": "string", "description": "Name of the scene object (one of the scored objects listed in the task)."}
_FRAME = {"type": "string", "enum": ["palm", "world"], "description": "Frame the axis/delta is given in (default palm)."}


def _tool(name, description, properties=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}}}


def _palm(extra=None, required=()):
    p = {"palm_xyz": _PALM_XYZ, "palm_orientation": _PALM_ORI}
    p.update(extra or {})
    p["seconds"] = _SECONDS
    return p, required


T = {}

T["hold"] = _tool("hold", "Keep every target as it is for `seconds` (the physics keeps running).", {"seconds": _SECONDS}, ["seconds"])
T["note"] = _tool("note", "Write a short note to the run log. No time passes.", {"text": {"type": "string"}}, ["text"])
T["finish"] = _tool("finish", "Declare the task complete (or hopeless) and stop. The scorer then runs on the final scene.",
                    {"reason": {"type": "string"}})

p, r = _palm({"finger_joints": {
    "description": ("Finger joint targets [rad]: either 20 numbers (finger i owns entries 4i..4i+3: abduction, then three flexion "
                    "joints, base to tip; fingers in order thumb, index, middle, ring, little) or an object {finger: 4 angles}; "
                    "named fingers change and the rest keep their targets. Omit to keep the fingers."),
    "anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 20, "maxItems": 20},
              _finger_map({"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, "per-finger angles")]}})
T["joints.move"] = _tool("move", "Move the palm and set finger joint targets together, interpolated in joint space.", p, r)

_TIPS = _finger_map(dict(_XYZ), "{finger: [x, y, z]} pad-centre targets in the PALM frame [m] (+z along the fingers, +x out of the palm face, +y toward the thumb). Fingers not named keep their targets.")
p, r = _palm({"tips": _TIPS,
              "straight": {"type": "boolean", "description": "Move the pads along straight palm-frame lines (IK every step) instead of joint interpolation."},
              "directions": _finger_map(dict(_XYZ), "Optional {finger: unit vector} palm-frame direction to point the distal link along.")})
T["fingertips.move"] = _tool("move", "Move the palm and place finger pad centres at palm-frame targets (per-finger IK; residual over 4 mm is reported).", p, r)
T["open"] = _tool("open", "Return all fingers to an open posture.", {"seconds": _SECONDS})

p, r = _palm({"s": {"type": "array", "items": {"type": "number"}, "description": "Synergy coefficients, in standard deviations (length k, see the task prompt). Omit to keep the fingers."}})
T["synergies.move"] = _tool("move", "Move the palm and set the hand posture to mean + basis @ s (whole-hand synergies), interpolated in joint space.", p, r)

# contacts primitives
_POINTS = _finger_map(dict(_XYZ), "{finger: [x, y, z]} contact points in the OBJECT frame [m] (object origin = its pose position).")
_NORMALS = _finger_map(dict(_XYZ), "Optional {finger: outward surface normal in the OBJECT frame}; default is away from the object origin.")
T["approach"] = _tool("approach", "Place pads `standoff` m outside the given object-frame points (pre-contact), optionally moving the palm too.",
                      {"obj": _OBJ, "points": _POINTS, "standoff": {"type": "number"}, "seconds": _SECONDS, "normals": _NORMALS,
                       "palm_xyz": _PALM_XYZ, "palm_orientation": _PALM_ORI}, ["obj", "points"])
T["grasp"] = _tool("grasp", "Put pads on the object-frame points, pressed `squeeze` m into the surface: an opposition grasp of any finger set. The named fingers become the held contacts.",
                   {"obj": _OBJ, "points": _POINTS, "squeeze": {"type": "number"}, "seconds": _SECONDS, "normals": _NORMALS}, ["obj", "points"])
T["squeeze"] = _tool("squeeze", "Press held pads `delta` m further in (negative loosens).",
                     {"fingers": _FINGER_LIST, "obj": _OBJ, "delta": {"type": "number"}, "seconds": _SECONDS}, ["fingers", "obj", "delta"])
_AXIS = dict(_XYZ, description="Rotation axis (need not be unit).")
T["rotate"] = _tool("rotate", "Move the named fingers' pads rigidly as if fixed to the object turning by `angle` rad about `axis` through `centre`; other fingers hold.",
                    {"obj": _OBJ, "axis": _AXIS, "angle": {"type": "number", "description": "rad"}, "fingers": _FINGER_LIST,
                     "seconds": _SECONDS, "frame": _FRAME,
                     "centre": dict(_XYZ, description="Pivot in the OBJECT frame [m]; default the object origin.")},
                    ["obj", "axis", "angle", "fingers"])
T["translate"] = _tool("translate", "Move the named fingers' pads rigidly by `delta` m.",
                       {"obj": _OBJ, "delta": dict(_XYZ, description="m"), "fingers": _FINGER_LIST, "seconds": _SECONDS, "frame": _FRAME},
                       ["obj", "delta", "fingers"])
T["release"] = _tool("release", "Back pads off `clear` m away from the object origin, then optionally lift them by a palm-frame vector.",
                     {"fingers": _FINGER_LIST, "obj": _OBJ, "clear": {"type": "number"},
                      "lift": dict(_XYZ, description="Optional palm-frame lift vector [m] after backing off."), "seconds": _SECONDS},
                     ["fingers", "obj"])
T["relocate"] = _tool("relocate", "Release one finger, carry it over and place it on a new object-frame point while the others hold.",
                      {"finger": _FINGER, "obj": _OBJ, "point": dict(_XYZ, description="New contact point in the OBJECT frame [m]."),
                       "clear": {"type": "number"}, "lift": dict(_XYZ), "squeeze": {"type": "number"}, "seconds": _SECONDS,
                       "normal": dict(_XYZ, description="Optional outward normal at the new point (object frame).")},
                      ["finger", "obj", "point"])
T["tuck"] = _tool("tuck", "Curl the named fingers out of the way.", {"fingers": _FINGER_LIST, "seconds": _SECONDS}, ["fingers"])
p, r = _palm()
T["carry"] = _tool("carry", "Move the palm only; the fingers keep their targets.", p, r)

# move_* on the layered level (same arguments as the single-layer moves)
p, r = T["joints.move"]["function"]["parameters"]["properties"], []
T["move_joints"] = _tool("move_joints", "Joints layer: move the palm and set finger joint targets together, interpolated in joint space.", p)
T["move_tips"] = _tool("move_tips", "Fingertips layer: move the palm and place pad centres at palm-frame targets (per-finger IK).",
                       T["fingertips.move"]["function"]["parameters"]["properties"])
T["move_synergy"] = _tool("move_synergy", "Synergies layer: move the palm and set the whole-hand posture to mean + basis @ s.",
                          T["synergies.move"]["function"]["parameters"]["properties"])

# force servo
T["grip"] = _tool("grip", ("Engage a per-step force servo: each named pad moves along its closing direction to hold the target contact force [N] "
                           "with `obj`. It keeps running through later motions until ungrip. Waits up to `seconds` for the forces to settle."),
                  {"obj": _OBJ, "forces": _finger_map({"type": "number"}, "{finger: target newtons}"),
                   "directions": _finger_map(dict(_XYZ), "Optional {finger: unit vector in the OBJECT frame pointing into the object}."),
                   "seconds": _SECONDS, "max_travel": {"type": "number", "description": "m of pad travel allowed past the commanded target"}},
                  ["obj", "forces"])
T["ungrip"] = _tool("ungrip", "Stop the force servo (all fingers if `fingers` omitted); pads stay where the servo left them.",
                    {"fingers": _FINGER_LIST})

_CONTACTS = ["approach", "grasp", "squeeze", "rotate", "translate", "release", "relocate", "tuck", "open", "carry"]
# (tool key, api method name) per level
_LEVEL_TOOLS = {
    "joints": [("joints.move", "move")],
    "fingertips": [("fingertips.move", "move"), ("open", "open")],
    "synergies": [("synergies.move", "move")],
    "contacts": [(n, n) for n in _CONTACTS],
    "layered": [("move_joints", "move_joints"), ("move_tips", "move_tips"), ("move_synergy", "move_synergy")] + [(n, n) for n in _CONTACTS],
}
_LEVEL_TOOLS["force"] = _LEVEL_TOOLS["layered"] + [("grip", "grip"), ("ungrip", "ungrip")]
_COMMON = ["hold", "note", "finish"]
LEVELS = tuple(_LEVEL_TOOLS)


def tool_specs(level):
    """The OpenAI-style tool list for a level."""
    return [T[k] for k, _ in _LEVEL_TOOLS[level]] + [T[k] for k in _COMMON]


def method_for(level, tool):
    """api method name for a model-visible tool name at `level`, or None."""
    for key, method in _LEVEL_TOOLS[level]:
        if T[key]["function"]["name"] == tool:
            return method
    return {"hold": "hold", "note": "note", "finish": None}.get(tool) if tool in _COMMON else None


# ------------------------------------------------------------------ validation
def _check(value, schema, path):
    """Minimal JSON-schema subset: type, enum, items, min/maxItems, properties, required, anyOf."""
    if "anyOf" in schema:
        errs = [_check(value, s, path) for s in schema["anyOf"]]
        if all(errs):
            return f"{path}: matches none of the allowed forms ({'; '.join(e for e in errs if e)[:200]})"
        return None
    t = schema.get("type")
    if t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            return f"{path}: expected a finite number, got {value!r}"
    elif t == "boolean":
        if not isinstance(value, bool):
            return f"{path}: expected a boolean"
    elif t == "string":
        if not isinstance(value, str):
            return f"{path}: expected a string"
        if "enum" in schema and value not in schema["enum"]:
            return f"{path}: must be one of {schema['enum']}, got {value!r}"
    elif t == "array":
        if not isinstance(value, list):
            return f"{path}: expected an array"
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10 ** 9):
            return f"{path}: expected {schema.get('minItems', 0)}-{schema.get('maxItems', 'any')} items, got {len(value)}"
        for i, v in enumerate(value):
            e = _check(v, schema.get("items", {}), f"{path}[{i}]")
            if e:
                return e
    elif t == "object":
        if not isinstance(value, dict):
            return f"{path}: expected an object"
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                return f"{path}: missing required '{k}'"
        for k, v in value.items():
            if k not in props:
                return f"{path}: unknown key '{k}' (allowed: {sorted(props)})"
            e = _check(v, props[k], f"{path}.{k}")
            if e:
                return e
    return None


def coerce_args(args):
    """Parse values that models often double-encode as JSON strings (e.g. finger_joints: "[0,1,...]").

    Walks the tree once: any string that starts with `[` or `{` and `json.loads` successfully is replaced
    by the parsed value. Non-JSON strings are left alone. Returns the coerced object and how many
    strings were parsed (so the caller can log it).
    """
    n = 0

    def walk(x):
        nonlocal n
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            s = x.strip()
            if s[:1] in "[{":
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    return x
                n += 1
                return walk(parsed)  # in case of double-double encoding
            return x
        return x

    return walk(args), n


def validate(level, tool, args):
    """None if the call is well formed for this level, else an error string for the model."""
    if tool not in [T[k]["function"]["name"] for k, _ in _LEVEL_TOOLS[level]] + _COMMON:
        offered = [T[k]["function"]["name"] for k, _ in _LEVEL_TOOLS[level]] + _COMMON
        return f"unknown tool '{tool}' at level '{level}'; available: {offered}"
    spec = next((T[k] for k, _ in _LEVEL_TOOLS[level] if T[k]["function"]["name"] == tool), T.get(tool))
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    return _check(args, spec["function"]["parameters"], tool)


def count_dims(args):
    """Number of continuous scalars in a call's arguments (the cognitive cost of a command)."""
    if isinstance(args, bool) or args is None or isinstance(args, str):
        return 0
    if isinstance(args, (int, float)):
        return 1
    if isinstance(args, dict):
        return sum(count_dims(v) for k, v in args.items() if k != "seconds")
    if isinstance(args, (list, tuple)):
        return sum(count_dims(v) for v in args)
    return 0


# ------------------------------------------------------------------ dispatch
def orientation(spec):
    """Rotation matrix from a `palm_orientation` object."""
    yaw = float(spec.get("yaw", 0.7))
    if spec["preset"] == "up":
        R = hand_up(yaw, float(spec.get("tilt", 0.0)))
    else:
        R = hand_down(yaw)
    pitch = float(spec.get("pitch", 0.0))
    if pitch:
        R = R @ Rotation.from_rotvec([0, pitch, 0]).as_matrix()
    return R


_RENAME = {"palm_xyz": "xyz", "finger_joints": "q", "forces": "forces"}


def to_kwargs(args):
    kw = {}
    for k, v in args.items():
        if k == "palm_orientation":
            kw["R"] = orientation(v)
        elif k == "seconds":
            kw["seconds"] = float(np.clip(v, MIN_SECONDS, MAX_SECONDS))
        else:
            kw[_RENAME.get(k, k)] = v
    return kw


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x, 4).tolist()
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def execute(api, level, tool, args):
    """Generator: run one tool call on `api`; yields once per control step and returns a JSON-able result dict."""
    method = method_for(level, tool)
    kw = to_kwargs(args)
    if tool == "note":
        api.note(args["text"][:300])
        return {"ok": True}
    fn = getattr(api, method)
    if tool == "hold":
        yield from fn(kw["seconds"])
        return {"ok": True}
    res = yield from fn(**kw)
    return jsonable(res) if isinstance(res, dict) else {"ok": True}


def check_against_api(level, cls):
    """Every offered tool's arguments must exist in the api method's signature (guards schema drift)."""
    from .. import levels as L
    problems = []
    rename = {"palm_xyz": "xyz", "palm_orientation": "R", "finger_joints": "q"}
    # The layered wrappers take (*a, **k) and forward to these single-layer methods.
    underlying = {"move_joints": L.Joints.move, "move_tips": L.Fingertips.move, "move_synergy": L.Synergies.move,
                  "approach": L.Contacts.approach, "grasp": L.Contacts.grasp, "squeeze": L.Contacts.squeeze,
                  "rotate": L.Contacts.rotate, "translate": L.Contacts.translate, "release": L.Contacts.release,
                  "relocate": L.Contacts.relocate, "tuck": L.Contacts.tuck, "open": L.Contacts.open, "carry": L.Contacts.carry,
                  "grip": L.Force._grip_impl, "ungrip": L.Force._ungrip_impl}
    for key, method in _LEVEL_TOOLS[level]:
        fn = getattr(cls, method, None)
        if fn is None:
            problems.append(f"{level}.{method}: missing on {cls.__name__}")
            continue
        if level in ("layered", "force"):
            fn = underlying[method]
        params = set(inspect.signature(fn).parameters) - {"self"}
        for a in T[key]["function"]["parameters"]["properties"]:
            if rename.get(a, a) not in params:
                problems.append(f"{level}.{method}: argument '{a}' not in signature {sorted(params)}")
    return problems
