"""Infer which scene object a retarget run should drive."""
from __future__ import annotations

from pathlib import Path

# Folder name → primary free object in that twin (when --object is omitted).
_SCENE_OBJECTS = {
    "cube_pinch": "cube",
    "cube_palm": "cube",
    "scissors": "scissors",
    "pen_gravity_pivot": "pen",
    "pen_pickup_pivot": "pen",
    "pen_finger_roll": "pen",
    "pen_writing_grasp": "pen",
    "pen_table_slide": "pen",
    "cap_twist": "cap",
    "card_pickup": "card",
    "baoding": "ball_a",
    "cube_reorient": "cube",
}

_SKIP = frozenset(
    {
        "table",
        "reference",
        "overview",
        "closeup",
        "loops",
        "ur5e_dg5fs_right",
    }
)


def infer_object_name(scene_dir: Path | str, sim=None) -> str:
    """Best-effort primary object name for a twin scene folder."""
    name = Path(scene_dir).resolve().name
    if name in _SCENE_OBJECTS:
        return _SCENE_OBJECTS[name]
    if sim is not None:
        candidates = [
            n
            for n in sim.names()
            if n not in _SKIP and not n.endswith("_asset") and "/" not in n
        ]
        # Prefer names that look like free objects in handles.objects.
        objs = getattr(getattr(sim, "scene", None), "handles", None)
        if objs is not None and hasattr(objs, "objects"):
            owned = [n for n in candidates if n in objs.objects]
            if owned:
                return sorted(owned)[0]
        if candidates:
            return sorted(candidates)[0]
    raise ValueError(
        f"cannot infer --object for scene {name!r}; pass --object explicitly"
    )
