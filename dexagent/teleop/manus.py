"""Self-contained Manus glove reader (no rio-hw).

Depends only on the ``manus_glove`` SDK package and numpy. Feature extraction
and calibration JSON format match the rio-hw Manus interface so existing
``manus_glove_cal.json`` files keep working.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from manus_glove import HandMotion, ManusDataPublisher
except ImportError:  # pragma: no cover
    HandMotion = None  # type: ignore
    ManusDataPublisher = None  # type: ignore

_JOINT_ORDER = {"MCP": 0, "PIP": 1, "IP": 2, "DIP": 3, "TIP": 4}


def _chain(raw: list[dict[str, Any]], chain: str) -> list[dict[str, Any]]:
    ns = [n for n in raw if n.get("chainType") == chain]
    ns.sort(key=lambda n: _JOINT_ORDER.get(str(n.get("jointType", "")), 99))
    return ns


def _pick(raw: list[dict[str, Any]], chain: str, joints: tuple[str, ...]) -> np.ndarray | None:
    for j in joints:
        for n in raw:
            if n.get("chainType") == chain and str(n.get("jointType")) == j:
                return np.asarray(n["position"], float)
    return None


def _chord_open(ns: list[dict[str, Any]]) -> float:
    if len(ns) < 2:
        return 0.5
    p = [np.asarray(n["position"], float) for n in ns]
    arc = sum(float(np.linalg.norm(p[i + 1] - p[i])) for i in range(len(p) - 1))
    if arc < 1e-9:
        return 0.5
    return float(np.clip(np.linalg.norm(p[-1] - p[0]) / arc, 0.0, 1.0))


def _thumb_chain(ns: list[dict[str, Any]], chord_w: float) -> float:
    if len(ns) < 3:
        return _chord_open(ns)
    p = [np.asarray(n["position"], float) for n in ns]
    sc: list[float] = []
    for i in range(len(p) - 2):
        u, v = p[i + 1] - p[i], p[i + 2] - p[i + 1]
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if lu < 1e-9 or lv < 1e-9:
            continue
        c = float(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0))
        sc.append((c + 1.0) * 0.5)
    if not sc:
        return _chord_open(ns)
    bend = 0.5 * min(sc) + 0.5 * float(np.mean(sc))
    w = float(np.clip(chord_w, 0.0, 1.0))
    return (1.0 - w) * bend + w * _chord_open(ns)


def _thumb_flex_01(raw: list[dict[str, Any]], chord_w: float) -> float:
    tip = _pick(raw, "Thumb", ("TIP", "DIP", "IP"))
    mcp = _pick(raw, "Thumb", ("MCP",))
    if tip is None or mcp is None:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    mcps = [_pick(raw, ch, ("MCP",)) for ch in ("Index", "Middle", "Ring", "Pinky")]
    mcps = [x for x in mcps if x is not None]
    if len(mcps) < 2:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    palm = np.mean(np.stack(mcps, axis=0), axis=0)
    u, v = tip - mcp, palm - mcp
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu < 1e-9 or lv < 1e-9:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    c = float(np.clip(np.dot(u / lu, v / lv), -1.0, 1.0))
    palm_o = float(np.clip(1.0 - (c + 1.0) * 0.5, 0.0, 1.0))
    chain_o = _thumb_chain(_chain(raw, "Thumb"), chord_w)
    return float(np.clip(0.7 * palm_o + 0.3 * chain_o, 0.0, 1.0))


def _thumb_rot_01(raw: list[dict[str, Any]]) -> float:
    tip = _pick(raw, "Thumb", ("TIP", "DIP", "IP"))
    tm = _pick(raw, "Thumb", ("MCP",))
    im = _pick(raw, "Index", ("MCP",))
    mm = _pick(raw, "Middle", ("MCP",))
    if tip is None or tm is None or im is None or mm is None:
        return 0.5
    ex = im - mm
    n = float(np.linalg.norm(ex))
    if n < 1e-9:
        return 0.5
    ex /= n
    ez = np.cross(ex, tm - mm)
    nz = float(np.linalg.norm(ez))
    if nz < 1e-9:
        return 0.5
    ez /= nz
    ey = np.cross(ez, ex)
    v = tip - tm
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return 0.5
    v /= nv
    a = float(np.arctan2(float(np.dot(v, ey)), float(np.dot(v, ex))))
    return float(np.clip((a / np.pi + 1.0) * 0.5, 0.0, 1.0))


def features(raw: list[dict[str, Any]], thumb_chord_w: float = 0.25) -> np.ndarray:
    """``[pinky, ring, middle, index, thumb_flex, thumb_rot]`` openness in ``[0, 1]``."""
    return np.array(
        [
            _chord_open(_chain(raw, "Pinky")),
            _chord_open(_chain(raw, "Ring")),
            _chord_open(_chain(raw, "Middle")),
            _chord_open(_chain(raw, "Index")),
            _thumb_flex_01(raw, thumb_chord_w),
            _thumb_rot_01(raw),
        ],
        dtype=np.float64,
    )


@dataclass
class GloveCalibration:
    open_v: np.ndarray
    closed_v: np.ndarray
    thumb_chord_w: float
    thumb_flex_open: float | None = None
    thumb_flex_closed: float | None = None
    thumb_rot_open: float | None = None
    thumb_rot_closed: float | None = None

    @property
    def thumb_extra(self) -> bool:
        return all(
            v is not None
            for v in (self.thumb_flex_open, self.thumb_flex_closed, self.thumb_rot_open, self.thumb_rot_closed)
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": 2 if self.thumb_extra else 1,
            "thumb_chord_weight": self.thumb_chord_w,
            "open": self.open_v.astype(float).tolist(),
            "closed": self.closed_v.astype(float).tolist(),
        }
        if self.thumb_extra:
            out["thumb_flex"] = {"open": float(self.thumb_flex_open), "closed": float(self.thumb_flex_closed)}
            out["thumb_rot"] = {"open": float(self.thumb_rot_open), "closed": float(self.thumb_rot_closed)}
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "GloveCalibration":
        o = np.asarray(d["open"], dtype=np.float64).reshape(6)
        c = np.asarray(d["closed"], dtype=np.float64).reshape(6)
        w = float(d.get("thumb_chord_weight", 0.25))
        tfo = tfc = tro = trc = None
        tf, tr = d.get("thumb_flex"), d.get("thumb_rot")
        if isinstance(tf, dict) and isinstance(tr, dict):
            tfo, tfc = float(tf["open"]), float(tf["closed"])
            tro, trc = float(tr["open"]), float(tr["closed"])
        return cls(o, c, w, tfo, tfc, tro, trc)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GloveCalibration":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_calibration(
    feat: np.ndarray,
    cal: GloveCalibration,
    *,
    curl: float = 1.2,
    thumb_curl: float = 1.45,
    min_span: float = 0.08,
) -> np.ndarray:
    lo = cal.closed_v.astype(np.float64, copy=True)
    hi = cal.open_v.astype(np.float64, copy=True)
    if cal.thumb_extra:
        lo[4], hi[4] = cal.thumb_flex_closed, cal.thumb_flex_open
        lo[5], hi[5] = cal.thumb_rot_closed, cal.thumb_rot_open
    den = hi - lo
    span = np.sign(den) * np.maximum(np.abs(den), min_span)
    span = np.where(span == 0.0, min_span, span)
    x = np.clip((feat - lo) / span, 0.0, 1.0)
    e = float(np.clip(curl, 1.0, 3.0))
    te = float(np.clip(thumb_curl, 1.0, 3.0))
    if e > 1.0:
        x[:4] = np.clip(x[:4] ** e, 0.0, 1.0)
    if te > 1.0:
        x[4] = float(np.clip(x[4] ** te, 0.0, 1.0))
    return x.astype(np.float64)


class ManusReader:
    """Poll a Manus glove for 6-DoF finger openness. Call ``start()`` / ``close()``."""

    def __init__(
        self,
        *,
        calibration_file: str | Path | None = "manus_glove_cal.json",
        glove_id: int | None = None,
        curl: float = 1.2,
        thumb_curl: float = 1.45,
        thumb_chord_weight: float = 0.25,
    ):
        if ManusDataPublisher is None:
            raise RuntimeError(
                "manus_glove is not installed. "
                "pip install 'manus_glove @ git+https://github.com/etaoxing/manus_glove.git'"
            )
        self.calibration_file = Path(calibration_file) if calibration_file else None
        self.glove_id = glove_id
        self.curl = curl
        self.thumb_curl = thumb_curl
        self.thumb_chord_weight = thumb_chord_weight
        self._publisher = None
        self._gid = None
        self._cal: GloveCalibration | None = None
        self._last = np.ones(6)

    def start(self) -> None:
        pub = ManusDataPublisher(hand_motion=HandMotion["NoMotion"], debug=False)
        pub.Initialize()
        pub.Connect()
        print("[Manus] Waiting for landscape…", flush=True)
        while pub.GetLandscape() is None:
            time.sleep(0.05)
        pub.LoadCalibrationFiles()
        print("[Manus] Waiting for glove…", flush=True)
        gid = None
        while gid is None:
            ids = pub.GetGloveIds()
            if ids:
                gid = self.glove_id if (self.glove_id is not None and self.glove_id in ids) else min(ids)
            else:
                time.sleep(0.05)
        print(f"[Manus] Using glove id={gid}", flush=True)
        cal = None
        if self.calibration_file is not None and self.calibration_file.is_file():
            cal = GloveCalibration.load(self.calibration_file)
            print(f"[Manus] Loaded calibration {self.calibration_file.resolve()}", flush=True)
        else:
            print(
                "[Manus] No calibration file — raw openness. "
                "Run calibrate once and save manus_glove_cal.json for better mapping.",
                flush=True,
            )
        self._publisher = pub
        self._gid = gid
        self._cal = cal

    def close(self) -> None:
        pub = self._publisher
        self._publisher = None
        if pub is not None:
            try:
                pub.Disconnect()
            except Exception:
                pass

    def read(self) -> np.ndarray:
        """Return openness ``(6,)`` in ``[0, 1]`` (1=open). Last good sample if no data."""
        pub, gid = self._publisher, self._gid
        if pub is None or gid is None:
            return self._last.copy()
        ids = pub.GetGloveIds()
        if ids:
            if self.glove_id is not None and self.glove_id in ids:
                gid = self.glove_id
            else:
                gid = min(ids)
            self._gid = gid
        data = pub.GetGloveData(gid)
        if data is None:
            return self._last.copy()
        raw = data.get("raw_nodes") or []
        if len(raw) < 8:
            return self._last.copy()
        w = self._cal.thumb_chord_w if self._cal is not None else self.thumb_chord_weight
        feat = features(raw, w)
        if self._cal is not None:
            out = apply_calibration(feat, self._cal, curl=self.curl, thumb_curl=self.thumb_curl)
        else:
            out = feat
        self._last = out.astype(float)
        return self._last.copy()
