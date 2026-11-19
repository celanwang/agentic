"""Compare unmodified production answers and ledgers with and without a solve cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packlab import memory, research


def cli(state, *args, task=None):
    result = subprocess.check_output([str(ROOT / "agentic"), "--state", str(state), *args, "--json"],
                                     input=json.dumps(task) if task is not None else None, text=True)
    return json.loads(result)


def snapshot(src, dst):
    dst.mkdir()
    with closing(sqlite3.connect((src / "agentic.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        with closing(sqlite3.connect(dst / "agentic.sqlite3")) as out:
            db.backup(out)


def same_event(a, b):
    return {k: v for k, v in a.items() if k not in ("experiment_ms", "created")} == {
        k: v for k, v in b.items() if k not in ("experiment_ms", "created")}


def check(before, root):
    watched, control, raced = [root / name for name in ("watched", "control", "raced")]
    for state in (watched, control, raced):
        snapshot(before, state)
    start = cli(watched, "shadow", "start", "12", "--limit", "32")
    assert start["source"] == "solve" and start["arrivals"] == 0
    easy = dict(capacity=131, bins=2, items=[12, 17, 23])
    boundary = dict(capacity=107, bins=5, items=[33, 39, 46, 50, 74, 68, 61, 57, 107])
    assert not cli(watched, "probe", "-", task=boundary)["result"]["ok"]
    assert cli(watched, "probe", "-", "--nominee", "12", task=boundary)["result"]["ok"]
    assert cli(watched, "shadow", "status")["arrivals"] == 0
    manual = subprocess.run([str(ROOT / "agentic"), "--state", str(watched), "shadow", "observe", "-"],
                            input=json.dumps(easy), text=True, capture_output=True)
    assert manual.returncode != 0 and cli(watched, "shadow", "status")["arrivals"] == 0
    for task in (easy, boundary):
        left = cli(watched, "solve", "-", task=task)
        right = cli(control, "solve", "-", task=task)
        assert same_event(left, right), (left["id"], left["decision"], right["decision"])
    status = cli(watched, "shadow", "status")
    assert status["eligible"] == status["compared"] == 2 and status["gains"] == 1
    assert status["branch_changes"] == 1
    assert cli(watched, "status")["policies"] == cli(control, "status")["policies"]
    assert cli(raced, "shadow", "start", "12", "--limit", "2")["state"] == "active"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: cli(raced, "solve", "-", task=easy), range(2)))
    assert all(row["answer"] for row in responses)
    race = cli(raced, "shadow", "status")
    assert (race["arrivals"], race["eligible"], race["duplicates"]) == (2, 1, 1)
    cli(raced, "rollback")
    assert cli(raced, "shadow", "status")["state"] == "drift"

    manual_state = root / "manual"
    snapshot(before, manual_state)
    cli(manual_state, "shadow", "start", "12", "--source", "observe", "--limit", "2")
    cli(manual_state, "shadow", "observe", "-", task=easy)
    with memory.transaction(manual_state) as db:
        assert memory._exclusions(db)(research.family_id(easy), "holdout")
    stopped = cli(manual_state, "shadow", "stop")
    assert stopped["state"] == "stopped" and stopped["eligible"] == 1
    assert cli(manual_state, "shadow", "status", "--event", "12") == stopped
    retry = subprocess.run([str(ROOT / "agentic"), "--state", str(manual_state), "shadow", "start", "12"],
                           capture_output=True)
    assert retry.returncode != 0 and cli(manual_state, "shadow", "status") == stopped
    print("Normal answers and research ledgers match; source, concurrent arrivals, drift and future holdout checked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        before = args.state
        if before is None:
            from scripts.route_review import replay
            base = root / "restore"
            base.mkdir()
            replay(base)
            before = base / "live"
        check(before, root)
