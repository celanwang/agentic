"""Replay all four frozen evaluations in a disposable workspace."""

from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVAL = HERE.relative_to(ROOT)
PILOT_STATE = Path("state/pilot/round-01")
EVAL_INPUTS = (
    "suite.py", "blind.py", "learning_eval.py", "final_eval.py", "cold_eval.py", "stream_eval.py",
    "protocol.json", "blind_commitment.json", "learning_binding.json",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(workspace, relative, digest=None):
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / relative, destination)
    if digest is not None and sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError(f"frozen hash mismatch for {relative}")


def prepare(workspace, release, published_blind):
    for name, digest in release["files"].items():
        copy_file(workspace, Path(name), digest)
    copy_file(workspace, Path("agentic"), published_blind["cli_sha256"])
    v2_digest = read_json(HERE / "protocol.json")["reference"]["v2_engine_sha256"]
    copy_file(workspace, Path("candidates/triples_strategy/engine.py"), v2_digest)
    for name in EVAL_INPUTS:
        copy_file(workspace, EVAL / name)
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/restore_demo_states.py"),
                    "--state-dir", str(workspace / "state")], check=True)

    private = workspace / EVAL / ".sealed_pilot"
    private.mkdir(mode=0o700)
    with (private / "payload.json").open("x", encoding="utf-8") as stream:
        json.dump(published_blind["reveal"], stream, sort_keys=True)


def run_module(workspace, module, *args):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONSAFEPATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), os.defpath, env.get("PATH", "")))
    command = [sys.executable, "-B", "-m", f"candidates.pilot_eval.{module}", *args]
    child = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True)
    if child.returncode:
        raise RuntimeError(f"{module} exited with status {child.returncode}\n{child.stderr.strip() or child.stdout.strip()}")


def strip_learning_times(value):
    for seed in value["seeds"]:
        for round_data in seed["rounds"]:
            for arm in round_data["arms"].values():
                del arm["elapsed_ms"]


def difference(expected, actual, path="$"):
    if type(actual) is not type(expected):
        return f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return f"{path}: missing {sorted(expected.keys() - actual.keys())}, extra {sorted(actual.keys() - expected.keys())}"
        children = ((expected[key], actual[key], f"{path}.{key}") for key in expected)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path}: expected {len(expected)} entries, got {len(actual)}"
        children = ((old, new, f"{path}[{index}]") for index, (old, new) in enumerate(zip(expected, actual)))
    else:
        return None if actual == expected else f"{path}: expected {expected!r}, got {actual!r}"
    return next((mismatch for old, new, at in children if (mismatch := difference(old, new, at))), None)


def compare(workspace, filename):
    expected = read_json(HERE / filename)
    actual = read_json(workspace / EVAL / filename)
    if filename == "learning_result.json":
        strip_learning_times(expected)
        strip_learning_times(actual)
    mismatch = difference(expected, actual)
    if mismatch is not None:
        raise ValueError(f"{filename}: {mismatch}")
    print(f"Verified {filename}", flush=True)


def main():
    release = read_json(ROOT / "evidence/pilot/release.json")
    published_blind = read_json(HERE / "blind_result.json")
    build = release["build"]
    stages = (
        ("learning_eval", (build,), "learning_result.json"),
        ("final_eval", ("probe", str(PILOT_STATE / "field"), "7", build), "blind_result.json"),
        ("cold_eval", (), "cold_result.json"),
        ("stream_eval", ("--record",), "stream_result.json"),
    )
    with TemporaryDirectory(prefix=".replay-", dir=HERE) as temporary:
        workspace = Path(temporary)
        prepare(workspace, release, published_blind)
        for module, args, filename in stages:
            run_module(workspace, module, *args)
            compare(workspace, filename)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Replay failed: {error}")
