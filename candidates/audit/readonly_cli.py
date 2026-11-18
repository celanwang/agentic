"""Exercise real Agentic read commands in isolated temporary working directories."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from contextlib import closing
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
TASK = {"capacity": 10, "bins": 2, "items": [4, 4, 6, 6]}
BASE = {"order": "arrival", "pick": "first", "repair": "none"}


def run(folder, *args, error=None):
    result = subprocess.run([str(ROOT / "agentic"), "--state", str(folder), *args, "--json"],
                            cwd=folder.parents[1], input=json.dumps(TASK), text=True,
                            capture_output=True, timeout=3)
    if error is not None:
        assert result.returncode == 2 and error in result.stderr, result
        assert not result.stdout, result
        return None
    assert result.returncode == 0, result
    return json.loads(result.stdout)


def read_views(folder, generation, events):
    state = run(folder, "status")
    assert state["generation"] == generation and state["events"] == events
    assert len(run(folder, "history")) == events
    probe = run(folder, "probe", "-")
    assert probe["policies"] == state["policies"]
    assert probe["result"]["ok"] == bool(generation)
    initial = run(folder, "probe", "-", "--initial")
    assert initial["policies"] == [BASE] and not initial["result"]["ok"]


def blank(root):
    for name in ("missing", "exists"):
        folder = root / name / "nested state"
        if name == "exists":
            folder.mkdir(parents=True)
        read_views(folder, 0, 0)
        for args in (("show", "1"), ("show", str(2**63)), ("explain", "1"),
                     ("probe", "-", "--nominee", "1")):
            run(folder, *args, error="没有编号为")
        run(folder, "rollback", error="没有更早的策略可恢复")
        run(folder, "research", "--budget", "1", error="研究预算须为")
        assert not (folder / "agentic.sqlite3").exists()
        if name == "exists":
            assert list(folder.iterdir()) == []
        else:
            assert not folder.parent.exists()


def snapshot(folder):
    return {file.name: (hashlib.sha256(file.read_bytes()).hexdigest(), file.stat().st_mtime_ns)
            for file in folder.iterdir()}


def committed_example(folder):
    learned = run(folder, "solve", "-")
    assert learned["generation"] == 1 and learned["decision"] == "kept"
    filename = folder / "agentic.sqlite3"
    # A legacy store needs only revisions/head/events. Supply a serialized
    # research record and an old record missing the optional nominee field.
    program = {"kind": "recipe", "steps": [], "solver": learned["nominee"], "guard": None}
    with closing(sqlite3.connect(str(filename))) as writer, writer:
        assert {r[0] for r in writer.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {
            "revisions", "head", "events"}
        for record in ({"kind": "research", "nominee": program,
                        "policies_before": [BASE], "practice_tasks": [TASK]}, {"kind": "research"}):
            writer.execute("INSERT INTO events (kind,start_revision,task,decision,reason,created,data) "
                           "VALUES ('research', 1, 'null', 'rejected', 'fixture', '2026-01-01', ?)",
                           (json.dumps(record),))
    return filename


def record_views(folder):
    assert run(folder, "show", "1")["decision"] == "kept"
    assert run(folder, "show", "2")["kind"] == "research"
    assert run(folder, "probe", "-", "--nominee", "2")["result"]["ok"]
    assert run(folder, "explain", "2")["status"] == "witness"
    run(folder, "explain", "3", error="没有冻结的研究配方")
    run(folder, "show", "0", error="没有编号为")


def committed_and_locked(root):
    folder = root / "existing" / "只读 #? % state"
    filename = committed_example(folder)
    before = snapshot(folder)
    filename.chmod(0o400)
    folder.chmod(0o500)
    try:
        read_views(folder, 1, 3)
        record_views(folder)
        assert snapshot(folder) == before
    finally:
        folder.chmod(0o700)
        filename.chmod(0o600)

    with closing(sqlite3.connect(str(filename))) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE head SET revision=0")
        writer.execute("DELETE FROM events WHERE id=2")
        read_views(folder, 1, 3)
        record_views(folder)
        writer.rollback()
        writer.execute("BEGIN EXCLUSIVE")
        run(folder, "status", error="database is locked")
        run(folder, "show", "1", error="database is locked")
        writer.rollback()
    assert snapshot(folder) == before

    rolled = run(folder, "rollback")
    assert rolled["generation"] == 0 and rolled["decision"] == "rollback"
    after = snapshot(folder)
    run(folder, "rollback", error="没有更早的策略可恢复")
    run(folder, "research", "--budget", "1", error="研究预算须为")
    read_views(folder, 0, 4)
    assert run(folder, "show", "4")["decision"] == "rollback"
    record_views(folder)
    assert snapshot(folder) == after


def invalid_state(root):
    folder = root / "invalid" / "state"
    folder.mkdir(parents=True)
    (folder / "agentic.sqlite3").touch()
    before = snapshot(folder)
    for args in (("status",), ("history",), ("show", "1")):
        run(folder, *args, error="no such table")
    assert snapshot(folder) == before


if __name__ == "__main__":
    with TemporaryDirectory(prefix="agentic-readonly-") as directory:
        root = Path(directory)
        blank(root)
        committed_and_locked(root)
        invalid_state(root)
    print("CLI state checks passed: read-only queries, snapshots, locks, legacy records, rollback, rejected writes")
