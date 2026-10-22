"""Initial-condition cases for benchmarking.

A case is a seed and a severity in [0, 1]. Severity 0 is the nominal scene,
exactly as authored; severity 1 is the full variation range the task's
TASK.md declares. A scene draws its perturbations at import time, in a fixed
order, from `current()`:

    from dexagent.cases import current
    case = current()
    START = WORK + [case.u(0.02), case.u(0.02), 0]   # +-20 mm at severity 1
    MASS = 0.012 * case.scale(0.3)                   # +-30 %

Without DEXAGENT_CASE in the environment (the live page, plain dexagent.run)
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
