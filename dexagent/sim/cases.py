"""Initial-condition cases for benchmarking.

A case is a seed and a severity in [0, 1]. Severity 0 is the nominal scene,
exactly as authored; severity 1 is the full variation range the task's
TASK.md declares. A scene draws its perturbations at import time, in a fixed
order, from `current()`:

    from dexagent.sim.cases import current
    case = current()
    START = WORK + [case.u(0.02), case.u(0.02), 0]   # +-20 mm at severity 1
    MASS = 0.012 * case.scale(0.3)                   # +-30 %

Without DEXAGENT_CASE in the environment (the live page, plain dexagent.sim.run)
`current()` is the nominal case, so scenes behave as before.
"""
import os
import numpy as np


class Case:
    def __init__(self, seed=0, severity=0.0):
        self.seed, self.severity = int(seed), float(severity)
        self.rng = np.random.default_rng(self.seed)
        self.drawn = {}

    def u(self, half_range, name=None):
        """Uniform in +-half_range * severity. Always consumes one draw, so the
        sequence does not depend on severity."""
        v = float(self.rng.uniform(-1, 1)) * half_range * self.severity
        if name:
            self.drawn[name] = v
        return v

    def scale(self, frac, name=None):
        """Multiplier 1 +- frac * severity."""
        v = 1.0 + self.u(frac)
        if name:
            self.drawn[name] = v
        return v

    def __repr__(self):
        return f"Case(seed={self.seed}, severity={self.severity})"


def parse(text):
    seed, _, sev = str(text).partition(":")
    return Case(int(seed), float(sev or 0))


_CURRENT = None


def current():
    global _CURRENT
    text = os.environ.get("DEXAGENT_CASE")
    if _CURRENT is None or getattr(_CURRENT, "_text", None) != text:
        _CURRENT = parse(text) if text else Case(0, 0.0)
        _CURRENT._text = text
    return _CURRENT


class Robust:
    """Execution perturbations for the sim-side robustness proxy (RQ4), applied to the joint targets.

    spec keys (all optional), set through DEXAGENT_ROBUST (a JSON dict; dexagent.score.score(robust=...)):
      noise    std [rad] of white noise added to the 26 robot joint targets every control step
      latency  control steps (20 ms each) by which every target command is delayed
    Object-side variation (pose, size, mass) comes from the case severity instead; severity > 1
    extrapolates beyond the range the task declares.
    """

    def __init__(self, spec, seed=0):
        self.noise = float(spec.get("noise", 0.0))
        self.latency = int(spec.get("latency", 0))
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.buf = []

    def apply(self, target, idx):
        """Return the command actually written for this step: delayed, then noisy."""
        self.buf.append(np.array(target, float))
        cmd = self.buf[max(0, len(self.buf) - 1 - self.latency)].copy()
        del self.buf[:max(0, len(self.buf) - 1 - self.latency)]
        if self.noise:
            cmd[idx] += self.rng.normal(0.0, self.noise, len(idx))
        return cmd


_ROBUST = None


def robust():
    """The process-wide Robust from DEXAGENT_ROBUST, or None when no perturbation is requested."""
    global _ROBUST
    import json
    text = os.environ.get("DEXAGENT_ROBUST")
    if not text:
        return None
    if _ROBUST is None:
        _ROBUST = Robust(json.loads(text), current().seed)
    return _ROBUST
