"""Post-hoc external check of the actual head after stream event 9."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from . import blind, final_eval, suite


STREAM = blind.ROOT / "state/pilot/round-01/stream"
COLD = blind.HERE / "cold_result.json"
RESULT = blind.HERE / "stream_result.json"
EVENT = 9


def run():
    from packlab import memory

    primary = json.loads(final_eval.RESULT.read_text())
    cold = json.loads(COLD.read_text())
    build = primary["build"]
    assert memory.build_id() == build == cold["build"]
    blind.verify(primary["reveal"], primary["commitment"])
    assert primary["commitment"]["blind_sha256"] == cold["blind_sha256"]

    formal = final_eval.formal_record(STREAM, EVENT)
    with final_eval.source(STREAM) as db:
        revision, policies = final_eval.head(db)
        initial = json.loads(db.execute("SELECT policies FROM revisions WHERE id=0").fetchone()[0])
        parent, = db.execute("SELECT parent FROM revisions WHERE id=?", (revision,)).fetchone()
        direct_tasks = [json.loads(raw) for raw, in db.execute("SELECT task FROM events") if raw and raw != "null"]
    assert formal["build"] == build and formal["decision"] == "kept"
    assert formal["rules"] == primary["formal"]["rules"]
    assert (parent, revision) == (formal["generation_before"], formal["generation"]) == (0, 1)
    assert initial == formal["policies_before"] == cold["formal"]["policies_before"]
    assert cold["active_revision"] == 0 and policies == [formal["program"], *initial]
    cores = primary["reveal"]["clusters"]
    assert {final_eval.forced_key(group["cases"][0]["task"]) for group in cores}.isdisjoint(
        final_eval.forced_key(task) for task in direct_tasks)

    executable = blind.ROOT / "agentic"
    assert blind.digest(executable.read_bytes()) == primary["cli_sha256"]
    rows = []
    with TemporaryDirectory(prefix=".stream-probe-", dir=blind.HERE) as temporary:
        snapshot = Path(temporary)
        with final_eval.source(STREAM) as source, closing(sqlite3.connect(snapshot / "agentic.sqlite3")) as copy:
            source.backup(copy)
            assert final_eval.head(copy) == (revision, policies)
        for cluster in cores:
            for case in cluster["cases"]:
                base = cold["rows"][len(rows)]
                assert (base["cluster"], base["profile"], base["variant"]) == (
                    cluster["id"], cluster["profile"], case["variant"])
                task = case["task"]
                old = base["modes"]["active"]
                assert "result" in old and (not old["result"]["ok"] or suite.check(task, old["result"]["packing"]))
                new = final_eval.probe(executable, snapshot, task, policies)
                rows.append({"cluster": cluster["id"], "profile": cluster["profile"], "variant": case["variant"],
                             "modes": {"v0": old, "active": new}})
    assert len(rows) == len(cold["rows"]) == 72
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    paired = [dict(row, modes=dict(row["modes"], v2=row["modes"]["v0"])) for row in rows]
    metrics = {mode: final_eval.summary(paired, mode) for mode in ("v0", "active")}
    gain = next((row for row in rows if not row["modes"]["v0"]["result"]["ok"]
                 and row["modes"]["active"].get("result", {}).get("ok")), None)
    first = {field: gain[field] for field in ("cluster", "profile", "variant")} if gain else None
    return {"assessment": "post-hoc external check on the previously revealed suite", "build": build,
            "formal": formal, "blind_sha256": primary["commitment"]["blind_sha256"],
            "v0_evidence_sha256": blind.digest(COLD.read_bytes()), "metrics": metrics,
            "first_gain_in_file_order": first, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="save a write-once evidence file")
    args = parser.parse_args()
    report = run()
    if args.record:
        blind.write_once(RESULT, report)
    elif RESULT.exists():
        assert report == json.loads(RESULT.read_text())
    print(json.dumps({k: report[k] for k in ("assessment", "metrics", "first_gain_in_file_order")}, indent=2))


if __name__ == "__main__":
    main()
