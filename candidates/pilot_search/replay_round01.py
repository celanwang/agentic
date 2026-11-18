"""Replay the ninth Round 01 practice phase with the published planner."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.frozen_pilot import materialize, restore_states, run_python


def main():
    with TemporaryDirectory(prefix="agentic-round01-practice-") as folder:
        workspace = Path(folder)
        materialize(workspace)
        restore_states(workspace, workspace / "state")
        result = run_python(workspace, "candidates/pilot_search/replay_round01.py")
    if result != (HERE / "round01_replay.json").read_bytes():
        raise ValueError("Round 01 practice replay differs from the published result")
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Practice replay failed: {error}")
