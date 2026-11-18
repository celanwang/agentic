"""Replay all four frozen evaluations in a disposable workspace."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from scripts.frozen_pilot import materialize, restore_states, run_python

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVAL = HERE.relative_to(ROOT)
PILOT_STATE = Path("state/pilot/round-01")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(workspace, published_blind):
    release = materialize(workspace)
    restore_states(workspace, workspace / "state")
    private = workspace / EVAL / ".sealed_pilot"
    private.mkdir(mode=0o700)
    with (private / "payload.json").open("x", encoding="utf-8") as stream:
        json.dump(published_blind["reveal"], stream, sort_keys=True)
    return release["build"]


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
    published_blind = read_json(HERE / "blind_result.json")
    with TemporaryDirectory(prefix="agentic-round01-replay-") as temporary:
        workspace = Path(temporary)
        build = prepare(workspace, published_blind)
        stages = (
            ("learning_eval", (build,), "learning_result.json"),
            ("final_eval", ("probe", str(PILOT_STATE / "field"), "7", build), "blind_result.json"),
            ("cold_eval", (), "cold_result.json"),
            ("stream_eval", ("--record",), "stream_result.json"),
        )
        for module, args, filename in stages:
            run_python(workspace, "-m", f"candidates.pilot_eval.{module}", *args)
            compare(workspace, filename)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Replay failed: {error}")
