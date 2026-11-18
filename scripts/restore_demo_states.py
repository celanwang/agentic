"""Restore published synthetic demo histories with their original runtime."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.frozen_pilot import materialize, restore_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state")
    location = parser.parse_args().state_dir.resolve()
    with TemporaryDirectory(prefix="agentic-round01-") as folder:
        workspace = Path(folder)
        materialize(workspace)
        sys.stdout.buffer.write(restore_states(workspace, location))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Restore failed: {error}")
