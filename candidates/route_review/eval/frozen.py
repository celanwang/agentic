"""Copy verified #11 runtime files into a caller-owned temporary workspace."""

import json
from pathlib import Path
from zipfile import ZipFile

from candidates.pair_eval import seal as pairs

from .seal import ROOT


RELEASE_SHA256 = "17d81820f1509eb0dbf3b3bea53f2c07db6e7779505c836e649a36865c6de686"
LAUNCHER_SHA256 = "e0816b0054cb6d7aa3c2646e5b7f16c5bba7b29c12fc1ed8c62d5db059cc7a16"


def copy_runtime(workspace):
    release_bytes = (ROOT / "evidence/pairs/release.json").read_bytes()
    assert pairs.digest(release_bytes) == RELEASE_SHA256
    release = json.loads(release_bytes)
    with ZipFile(ROOT / "evidence/pairs/round11-source.zip") as archive:
        for name, expected in {**release["runtime"], "agentic": LAUNCHER_SHA256}.items():
            path = Path(name)
            assert name == "agentic" or (path.parent == Path("packlab") and path.suffix == ".py"
                                          and path.as_posix() == name)
            raw = archive.read(name)
            assert pairs.digest(raw) == expected
            (workspace / path).parent.mkdir(parents=True, exist_ok=True)
            with (workspace / path).open("xb") as stream:
                stream.write(raw)
    return release["build"]
