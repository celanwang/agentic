"""Check frozen Round 01 restoration from a minimal offline source export."""

from contextlib import closing
import json
from pathlib import Path
from shutil import copyfile
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path("evidence/pilot/round01-source.zip")
RELEASE = Path("evidence/pilot/release.json")
RESTORE = Path("scripts/restore_demo_states.py")
STREAM = Path("pilot/round-01/stream/agentic.sqlite3")


def restore(root, state):
    return subprocess.run([sys.executable, "-I", "-B", str(RESTORE), "--state-dir", str(state)],
                          cwd=root, capture_output=True, text=True)


def verify_round_nine(db_file, build):
    with closing(sqlite3.connect(db_file.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        assert db.execute("SELECT count(*) FROM events").fetchone() == (9,)
        kind, decision, start, raw = db.execute(
            "SELECT kind, decision, start_revision, data FROM events WHERE id=9").fetchone()
        revision, parent, policies = db.execute(
            "SELECT r.id, r.parent, r.policies FROM revisions r JOIN head h ON h.revision=r.id").fetchone()
    record = json.loads(raw)
    assert (kind, decision, start, revision, parent) == ("research", "kept", 0, 1, 0)
    assert record["build"] == build and record["generation"] == revision
    assert json.loads(policies) == [record["nominee"], *record["policies_before"]]


def verify():
    with TemporaryDirectory(prefix="agentic-frozen-audit-") as temp:
        exported = Path(temp)
        for name in (ARCHIVE, RELEASE, RESTORE, Path("scripts/frozen_pilot.py")):
            target = exported / name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT / name, target)
        live = exported / "packlab"
        live.mkdir()
        (live / "__init__.py").touch()
        (live / "memory.py").write_text('raise ImportError("live memory canary")\n', encoding="utf-8")
        assert not (exported / ".git").exists()

        state = exported / "state"
        first = restore(exported, state)
        assert first.returncode == 0, first.stderr
        stream = state / STREAM
        build = json.loads((exported / RELEASE).read_text(encoding="utf-8"))["build"]
        verify_round_nine(stream, build)

        # Force a second restore to publish one missing DB while #9 already exists.
        missing = state / "pilot/round-01/cold/agentic.sqlite3"
        missing.unlink()
        originals = {db: (db.stat().st_ino, db.read_bytes()) for db in state.rglob("agentic.sqlite3")}
        second = restore(exported, state)
        assert second.returncode == 0, second.stderr
        assert missing.is_file()
        assert all((db.stat().st_ino, db.read_bytes()) == original for db, original in originals.items())

        with (exported / ARCHIVE).open("ab") as archive:
            archive.write(b"changed\n")
        rejected_state = exported / "rejected-state"
        rejected = restore(exported, rejected_state)
        assert rejected.returncode != 0 and "archive SHA-256 mismatch" in rejected.stderr, rejected.stderr
        assert not rejected_state.exists()
    return {"offline_export": True, "restored_event": 9, "restored_generation": 1,
            "existing_databases_preserved": len(originals), "corrupt_archive_rejected": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
