"""SHA-256 of recorded artifacts, as recorded at experiment time.

The October 20 restructure moved outputs to artifacts/ and assets to third-party/,
and rewrote the absolute asset paths inside recorded XML/JSON accordingly.
artifacts/RELOCATION.json maps each rewritten file to its original and current
digests, so provenance checks still compare against the original recording.
"""
import hashlib, json
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELOCATION = ROOT / 'artifacts/RELOCATION.json'


@cache
def _relocated():
    return json.loads(RELOCATION.read_text())['files'] if RELOCATION.exists() else {}


def recorded_sha256(path):
    """Digest of `path`, or its pre-relocation digest if only the path rewrite changed it."""
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = _relocated().get(str(path.relative_to(ROOT)))
    return entry['original_sha256'] if entry and entry['sha256'] == digest else digest
