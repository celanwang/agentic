"""Materialize the published Round 01 build for offline historical replay."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/pilot/round01-source.zip"
ARCHIVE_SHA256 = "8792b65bd022f487ac06e3f561fae0dd68c198018e0bef8dc2179f7d8fb30b48"
RELEASE = "evidence/pilot/release.json"
RUNTIME = frozenset(f"packlab/{name}.py" for name in (
    "__init__", "arena", "cli", "engine", "experiment", "memory", "planner", "recipes", "research", "witness",
))
EXECUTABLE = {"agentic", "demo-research"}
MEMBERS = RUNTIME | EXECUTABLE | {
    RELEASE, "evidence/mixed/legacy-events.jsonl", "scripts/restore_demo_states.py",
    "candidates/triples_strategy/engine.py", "candidates/pilot_search/replay_round01.py",
    *(f"candidates/pilot_eval/{name}" for name in (
        "suite.py", "blind.py", "learning_eval.py", "final_eval.py", "cold_eval.py", "stream_eval.py",
        "protocol.json", "blind_commitment.json", "learning_binding.json",
    )),
    *(f"examples/{name}.json" for name in (
        "first", "restart", "next", "triples-trigger", "triples-restart", "mixed-trigger", "mixed-restart", "pilot-transfer",
    )),
}


def digest(data):
    return sha256(data).hexdigest()


def permissions(name):
    return 0o755 if name in EXECUTABLE else 0o644


def validate_members(archive):
    seen = set()
    for entry in archive.infolist():
        name = entry.orig_filename
        if name not in MEMBERS or name != entry.filename or name in seen:
            raise ValueError(f"unexpected or duplicate frozen archive member: {name!r}")
        if entry.create_system != 3 or entry.external_attr >> 16 != (stat.S_IFREG | permissions(name)):
            raise ValueError(f"invalid file type or mode for frozen archive member: {name}")
        seen.add(name)
    if seen != MEMBERS:
        raise ValueError(f"missing frozen archive members: {', '.join(sorted(MEMBERS - seen))}")


def materialize(workspace):
    archive_data = ARCHIVE.read_bytes()
    if digest(archive_data) != ARCHIVE_SHA256:
        raise ValueError("frozen Round 01 source archive SHA-256 mismatch")
    if any(workspace.iterdir()):
        raise ValueError("frozen workspace must be empty")
    with ZipFile(BytesIO(archive_data)) as archive:
        validate_members(archive)
        contents = {name: archive.read(name) for name in sorted(MEMBERS)}

    if contents[RELEASE] != (ROOT / RELEASE).read_bytes():
        raise ValueError("published Round 01 release manifest differs from frozen source")
    release = json.loads(contents[RELEASE])
    if set(release["files"]) != RUNTIME:
        raise ValueError("unexpected files in the Round 01 release manifest")
    for name, expected in release["files"].items():
        if digest(contents[name]) != expected:
            raise ValueError(f"frozen runtime SHA-256 mismatch for {name}")

    for name, data in contents.items():
        destination = workspace / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        destination.chmod(permissions(name))
    return release


def run_python(workspace, *args):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONSAFEPATH"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", os.defpath)))
    child = subprocess.run([sys.executable, "-B", *args], cwd=workspace, env=env, capture_output=True)
    if child.returncode:
        message = (child.stderr or child.stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"frozen command {args[0]} exited with status {child.returncode}\n{message}")
    return child.stdout


def restore_states(workspace, state_dir):
    return run_python(workspace, "scripts/restore_demo_states.py", "--state-dir", str(state_dir.resolve()))
