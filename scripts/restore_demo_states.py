"""Restore published synthetic demo histories without overwriting local work."""
import argparse
import json
import os
from pathlib import Path
import runpy
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from packlab import memory
from packlab.experiment import ident, validate_task


LEGACY = {3: "demos/20261118-183901-uhbq90ps", 5: "triples/20261118-200049-3qo0skt6",
          6: "mixed/20261118-201146-mt88urf4"}
PILOT = Path("pilot/round-01")
DB = "agentic.sqlite3"


def append_legacy(folder, event_id, event):
    # These records came from three earlier builds. Keep their observed results and timestamps.
    with memory.transaction(folder) as conn:
        revision = event["generation_before"]
        if memory.head(conn)["id"] != revision:
            raise ValueError(f"Historical revision mismatch at event {event_id}")
        conn.execute("INSERT INTO events (id, kind, start_revision, task, decision, reason, created, data) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (event_id, event["kind"], revision, ident(event["task"]),
                     event["decision"], event["reason"], event["created"], ident(event)))
        if event["decision"] == "kept":
            policies = event["policies_before"] + [event["nominee"]]
            conn.execute("INSERT INTO revisions VALUES (?, ?, ?)", (event["generation"], revision, ident(policies)))
            conn.execute("UPDATE head SET revision=? WHERE slot=1", (event["generation"],))


def snapshot(source, destination):
    destination.mkdir(parents=True)
    copyfile(source / DB, destination / DB)


def rebuild(staging):
    history = staging / "legacy"
    source = ROOT / "evidence/mixed/legacy-events.jsonl"
    for event_id, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        append_legacy(history, event_id, json.loads(line))
        if event_id in LEGACY:
            snapshot(history, staging / LEGACY[event_id])

    field, cold, stream = (staging / PILOT / name for name in ("field", "cold", "stream"))
    snapshot(staging / LEGACY[6], field)
    memory.investigate(field)
    memory.investigate(cold)
    snapshot(cold, stream)
    for name in runpy.run_path(str(ROOT / "demo-research"))["ORDER"]:
        task = json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))
        memory.solve(stream, validate_task(task))


def restore(state_dir):
    destinations = [Path(name) for name in LEGACY.values()] + [PILOT / name for name in ("field", "cold", "stream")]
    missing = [name for name in destinations if not (state_dir / name / DB).is_file()]
    if not missing:
        return 0
    state_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    with TemporaryDirectory(dir=state_dir) as folder:
        staging = Path(folder)
        rebuild(staging)
        for name in missing:
            destination = state_dir / name
            destination.mkdir(parents=True, exist_ok=True)
            try:
                # Publish the completed DB atomically, even if another process created a state meanwhile.
                os.link(staging / name / DB, destination / DB)
                created += 1
            except FileExistsError:
                pass
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state")
    location = parser.parse_args().state_dir.resolve()
    print(f"在 {location} 重建了 {restore(location)} 份示例现场。")
