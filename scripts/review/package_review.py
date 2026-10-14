"""Create an offline copy of the measured visual review, excluding local tooling."""

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "review_site/dist"


def main():
    archive = DIST / "assets/dexagent-review.zip"
    temporary = archive.with_suffix(".tmp")
    # These are first-look camera/debug outputs, not figures used by the review.
    for name in ["test_render.png", "camera_sheet.jpg"]:
        (DIST / "assets" / name).unlink(missing_ok=True)
    (DIST / "START_HERE.txt").write_text(
        "DEXAGENT SIMULATION REVIEW\n\n"
        "Extract this entire archive, then open index.html in a modern browser.\n"
        "Keep the assets/ and data/ directories beside index.html.\n"
        "This offline copy contains a dated snapshot, not a live server connection.\n"
        "Every video replays actual recorded Newton states. MuJoCo OpenGL performs\n"
        "the rendering; RTX and image/video generation models are not used.\n"
        "The 14 development clips are not the full industrial A-F benchmark.\n\n"
        "This download uses WebM videos to avoid duplicate encodings.\n"
        "The hosted review also offers MP4 copies of the same rendered frames.\n"
        "NPZ files contain the recorded qpos/time arrays; JSON files contain frame\n"
        "measurements, checkpoint hashes, scope, and rendering provenance.\n"
    )
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(DIST.rglob("*")):
            if not path.is_file() or path in {archive, temporary} or path.suffix == ".mp4":
                continue
            relative = path.relative_to(DIST)
            if path.name == "snapshot.js" or path.suffix == ".json":
                text = path.read_text()
                snapshot = path.name == "snapshot.js"
                value = json.loads(text.split(" = ", 1)[1].rstrip(";\n") if snapshot else text)
                # Offline media links must resolve to the included WebM encoding.
                def remove_mp4_fallback(node):
                    if isinstance(node, dict):
                        for view in node.get("views", {}).values():
                            view.pop("mp4", None)
                        for child in node.values():
                            remove_mp4_fallback(child)
                    elif isinstance(node, list):
                        for child in node:
                            remove_mp4_fallback(child)
                remove_mp4_fallback(value)
                encoded = json.dumps(value)
                z.writestr(str(relative), "window.REVIEW_DATA = " + encoded + ";\n" if snapshot else encoded)
            elif path.name == "index.html":
                z.writestr(str(relative), path.read_text().replace("Current video (.mp4)", "Current video (.webm)"))
            else:
                z.write(path, relative)
    temporary.replace(archive)
    print(json.dumps({"archive": str(archive), "size_mib": archive.stat().st_size/1024**2}))


if __name__ == "__main__":
    main()
