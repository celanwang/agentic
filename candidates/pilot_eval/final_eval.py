"""Reveal a preregistered suite after independent locked-release CLI probes."""

import argparse
from contextlib import closing
import json
from math import gcd
from pathlib import Path
import sqlite3
import subprocess

from . import blind, suite


RESULT = blind.HERE / "blind_result.json"
MODES = ("v2", "nominee", "active")


def source(state):
    path = (state.resolve() / "agentic.sqlite3")
    return closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True))


def head(db):
    generation, raw = db.execute("SELECT r.id,r.policies FROM revisions r JOIN head h ON h.revision=r.id").fetchone()
    return generation, json.loads(raw)


def snapshot(state, name):
    dest = blind.PRIVATE / ("snapshot-" + name)
    dest.mkdir(mode=0o700)
    with source(state) as db, closing(sqlite3.connect(dest / "agentic.sqlite3")) as copy:
        db.backup(copy)
        return dest, head(copy)


def formal_record(state, event_id):
    with source(state) as db:
        row = db.execute("""SELECT decision, json_extract(data,'$.build'),
            json_extract(data,'$.generation_before'), json_extract(data,'$.generation'),
            json_extract(data,'$.rules'), json_extract(data,'$.nominee'),
            json_extract(data,'$.nominee_sha256'), json_extract(data,'$.policies_before')
            FROM events WHERE id=? AND kind='research'""", (event_id,)).fetchone()
        if row is None:
            raise ValueError("formal research event is missing")
        decision, build, before, after, rules, program, program_sha, prior = row
        return {"event": event_id, "decision": decision, "build": build, "generation_before": before,
                "generation": after, "rules": json.loads(rules),
                "program": json.loads(program) if program else None,
                "nominee_sha256": program_sha, "policies_before": json.loads(prior)}


def forced_key(task):
    """Drop every file that necessarily occupies a disk on its own."""
    sizes, capacity = task["items"], task["capacity"]
    kept = [size for index, size in enumerate(sizes) if size > capacity or
            any(other_index != index and size + other <= capacity for other_index, other in enumerate(sizes))]
    unit = gcd(capacity, *kept)
    return task["bins"] - len(sizes) + len(kept), capacity // unit, tuple(sorted(size // unit for size in kept))


def reject_overlaps(payload, states):
    keys = {forced_key(cluster["cases"][0]["task"]) for cluster in payload["clusters"]}
    assert len(keys) == len(payload["clusters"])
    assert all(forced_key(case["task"]) == forced_key(cluster["cases"][0]["task"])
               for cluster in payload["clusters"] for case in cluster["cases"])
    for state in states:
        with source(state) as db:
            tasks = (json.loads(raw) for raw, in db.execute("SELECT task FROM events WHERE task IS NOT NULL"))
            if any(task and forced_key(task) in keys for task in tasks):
                raise ValueError("a sealed core overlaps a formal direct task")
    learning = suite.learning()
    all_keys = set(keys)
    for seed in learning:
        for row in seed["rounds"]:
            new = [forced_key(case["task"]) for case in row["train"]]
            new.extend(forced_key(group["cases"][0]["task"]) for group in row["checkpoint"])
            assert len(new) == len(set(new)) and all_keys.isdisjoint(new)
            all_keys.update(new)
    return len(all_keys)


def probe(executable, state, task, policies, nominee=None):
    args = [str(executable), "--state", str(state), "probe", "-", "--json"]
    if nominee is not None:
        args.extend(("--nominee", str(nominee)))
    try:
        child = subprocess.run(args, input=suite.canonical(task), capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"error": "probe_timeout"}
    if child.returncode:
        return {"error": "probe_exit", "exit": child.returncode,
                "stderr": child.stderr.decode("utf-8", errors="replace")}
    try:
        observed = json.loads(child.stdout)
        result = observed["result"]
        if observed["task"] != task or observed["policies"] != policies:
            return {"error": "wrong_task_or_policy_path", "observed": observed}
        if result["ok"] and not suite.check(task, result["packing"]):
            return {"error": "invalid_packing", "result": result}
        if type(result["ok"]) is not bool or type(result["cost"]) is not int or result["cost"] < 1:
            return {"error": "invalid_operations_or_status", "result": result}
        return {"result": result}
    except (TypeError, ValueError, KeyError):
        return {"error": "malformed_probe_json", "stdout": child.stdout.decode("utf-8", errors="replace")}


def compare_v2(task, modes, policies):
    from candidates.triples_strategy import engine

    if "error" in modes["v2"]:
        return None
    attempts = []
    for policy in policies:
        trial = engine.run(task, policy)
        attempts.append({"policy": policy, "ok": trial["ok"], "cost": trial["cost"]})
        trace = trial["trace"]
        if trial["ok"] or trace["search_status"] == "infeasible" or (
                (trace["blocked"] or {}).get("reason") in {"oversize", "total_excess"}):
            break
    action = modes["v2"]["result"]
    return action["attempts"] == attempts and action["packing"] == trial["packing"]


def summary(rows, mode):
    complete = [row for row in rows if "result" in row["modes"][mode]]
    common = [row for row in complete if "result" in row["modes"]["v2"]]
    counts = {}
    for row in complete:
        counts.setdefault(row["cluster"], []).append(row["modes"][mode]["result"]["ok"])
    old = [row["modes"]["v2"]["result"] for row in common]
    new = [row["modes"][mode]["result"] for row in common]
    return {"cases": len(rows), "valid_results": len(complete), "paired_results": len(common),
            "ok": sum(row["modes"][mode]["result"]["ok"] for row in complete),
            "old_paired_ok": sum(r["ok"] for r in old), "new_paired_ok": sum(r["ok"] for r in new),
            "gains": sum(b["ok"] and not a["ok"] for a, b in zip(old, new)),
            "losses": sum(a["ok"] and not b["ok"] for a, b in zip(old, new)),
            "all_variants": sum(len(ok) == len(suite.VARIANTS) and all(ok) for ok in counts.values()),
            "cost": sum(row["modes"][mode]["result"]["cost"] for row in complete),
            "paired_old_cost": sum(r["cost"] for r in old), "paired_new_cost": sum(r["cost"] for r in new)}


def analyze(rows):
    metrics = {}
    for mode in MODES:
        metrics[mode] = {"all": summary(rows, mode),
                         "by_profile": {name: summary([row for row in rows if row["profile"] == name], mode)
                                        for name in suite.PROFILES},
                         "by_variant": {name: summary([row for row in rows if row["variant"] == name], mode)
                                        for name in suite.VARIANTS}}
    return metrics


def run(state, event_id, build):
    from candidates.triples_strategy import engine as old_engine
    from packlab import memory, research
    from packlab.experiment import RULES, ident

    assert memory.build_id() == build
    formal = formal_record(state, event_id)
    protocol = json.loads(blind.PROTOCOL.read_text())
    assert formal["build"] == build and formal["rules"] == RULES
    assert blind.digest(Path(old_engine.__file__).read_bytes()) == protocol["reference"]["v2_engine_sha256"]
    with source(state) as db:
        active_revision, active = head(db)
    with source(blind.V2_STATE) as db:
        old_revision, old = head(db)
    assert old_revision == 2 and active_revision == formal["generation"]
    program = formal["program"]
    nominee = research.dispatch(program, formal["policies_before"]) if program is not None else None
    if formal["nominee_sha256"]:
        assert blind.digest(ident(program).encode()) == formal["nominee_sha256"]
    if formal["decision"] == "kept":
        assert active_revision != formal["generation_before"] and active == nominee
    else:
        assert active_revision == formal["generation_before"] and active == formal["policies_before"]
    committed = json.loads(blind.COMMITMENT.read_text())
    payload = json.loads(blind.PAYLOAD.read_text())
    blind.verify(payload, committed)
    unique_cores = reject_overlaps(payload, (blind.V2_STATE, state))
    old_snap, old_head = snapshot(blind.V2_STATE, "v2")
    active_snap, active_head = snapshot(state, "active")
    assert old_head == (old_revision, old) and active_head == (active_revision, active)

    executable = blind.ROOT / "agentic"
    cli_sha = blind.digest(executable.read_bytes())
    rows = []
    for cluster in payload["clusters"]:
        for case in cluster["cases"]:
            task = case["task"]
            modes = {"v2": probe(executable, old_snap, task, old),
                     "nominee": probe(executable, active_snap, task, nominee, event_id) if nominee is not None
                     else {"error": "no_nominee"},
                     "active": probe(executable, active_snap, task, active)}
            rows.append({"cluster": cluster["id"], "profile": cluster["profile"], "variant": case["variant"],
                         "modes": modes, "v2_matches_frozen": compare_v2(task, modes, old)})
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    assert blind.digest(executable.read_bytes()) == cli_sha
    report = {"build": build, "cli_sha256": cli_sha, "formal": formal,
              "active_revision": active_revision, "v2_revision": old_revision,
              "unique_disjoint_cores_in_all_independent_partitions": unique_cores,
              "commitment": committed, "reveal": payload, "metrics": analyze(rows), "rows": rows}
    blind.write_once(RESULT, report)
    print(json.dumps({"decision": formal["decision"], "generations": [old_revision, active_revision],
                      "v2_matches_frozen": sum(row["v2_matches_frozen"] is True for row in rows),
                      "metrics": {name: data["all"] for name, data in report["metrics"].items()}}, indent=2))


def verify():
    report = json.loads(RESULT.read_text())
    assert report["commitment"] == json.loads(blind.COMMITMENT.read_text())
    blind.verify(report["reveal"], report["commitment"])
    tasks = {(group["id"], case["variant"]): case["task"] for group in report["reveal"]["clusters"]
             for case in group["cases"]}
    assert len(tasks) == len(report["rows"]) == 72
    for row in report["rows"]:
        task = tasks.pop((row["cluster"], row["variant"]))
        for observed in row["modes"].values():
            result = observed.get("result")
            assert result is None or not result["ok"] or suite.check(task, result["packing"])
    assert analyze(report["rows"]) == report["metrics"]
    print(json.dumps({"commitment_verified": True, "cases": len(report["rows"]),
                      "valid_results_by_mode": {name: data["all"]["valid_results"] for name, data in report["metrics"].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    probe_command = commands.add_parser("probe")
    probe_command.add_argument("state", type=Path)
    probe_command.add_argument("event", type=int)
    probe_command.add_argument("build")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    else:
        run(args.state, args.event, args.build)


if __name__ == "__main__":
    main()
