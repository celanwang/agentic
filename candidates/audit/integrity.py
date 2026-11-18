"""Check repeat handling, evidence readback, and concurrent commits through the CLI."""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import reduce
import json
from math import gcd
from pathlib import Path
import sys
import tempfile
from threading import Barrier

sys.dont_write_bytecode = True
from audit import CLI, HERE, ROOT, AuditError, feasible, require, restart_cases, task, validate_packing


def workload(problem):
    divisor = reduce(gcd, problem["items"], problem["capacity"])
    return (problem["bins"], problem["capacity"] // divisor,
            tuple(sorted(size // divisor for size in problem["items"])))


def check_evidence(event, transfer):
    groups = (event["holdout"], event["regression"])
    seen = {workload(event["task"]), workload(transfer)}
    for problem in event["practice_tasks"] + [row["task"] for g in groups for row in g["rows"]]:
        key = workload(problem)
        require(key not in seen, "evidence groups or unseen transfer reuse a workload")
        seen.add(key)

    for group in groups:
        rows = group["rows"]
        totals = dict(total=len(rows),
                      gains=sum(r["new_ok"] and not r["old_ok"] for r in rows),
                      losses=sum(r["old_ok"] and not r["new_ok"] for r in rows))
        for field in ("old_ok", "new_ok", "old_cost", "new_cost"):
            totals[field] = sum(row[field] for row in rows)
        require(all(totals[k] == group[k] for k in totals), "summary differs from per-task evidence")
        require(totals["losses"] == 0, "kept evidence loses an existing success")

    require(groups[0]["total"] >= 30 and groups[0]["gains"] >= 3, "insufficient holdout gains")
    require(groups[1]["old_ok"] >= 8, "insufficient baseline successes in regression group")
    ratio = Fraction(sum(g["new_cost"] for g in groups), sum(g["old_cost"] for g in groups))
    require(ratio <= Fraction(9, 4), "kept evidence exceeds cost threshold")
    return dict(holdout=groups[0]["total"], gains=groups[0]["gains"],
                regression=groups[1]["total"], cost_ratio=round(float(ratio), 4))


def check_rejection(cli):
    state = "rejection"
    # The files total 21 across 22 units, outside the exact-fill triples domain.
    problem = task(11, 2, [3, 3, 3, 3, 4, 5])
    require(feasible(problem), "cost rejection fixture must have a valid packing")
    before = cli.response(state, "status")
    first = cli.solve(state, "cost-rejection", problem)
    require(first["decision"] == "rejected" and first["generation"] == 0, "expensive fallback was retained")
    require(not first["before"]["ok"] and first["before"] == first["after"], "rejection changed the active result")
    require(first["answer"]["source"] == "trial", "single-use packing was lost on rejection")
    validate_packing(problem, first["answer"]["packing"], complete=True)
    groups = (first["holdout"], first["regression"])
    ratio = Fraction(sum(g["new_cost"] for g in groups), sum(g["old_cost"] for g in groups))
    require(ratio > Fraction(9, 4), "rejection cost evidence is below the cutoff")
    again = cli.solve(state, "repeat-cost-rejection", problem)
    require(again["decision"] == "unchanged" and not again["trials"] and again["reused_event"] == first["id"],
            "rejected experiment was retried instead of reusing evidence")
    require(again["answer"] == {**first["answer"], "source": "record"}, "repeat lost the single-use packing")
    after = cli.response(state, "status")
    require(after == {**before, "events": 2}, "rejection or repeat changed the active state")
    return {"decision": first["decision"], "cost_ratio": round(float(ratio), 4), "repeat": again["decision"]}


def check_integrity():
    trigger, transfer = restart_cases()
    with tempfile.TemporaryDirectory(prefix="integrity-", dir=HERE) as directory:
        cli = CLI(ROOT / "agentic", Path(directory), 25)
        state = "race"
        initial = cli.response(state, "status")
        require(initial["generation"] == initial["events"] == 0, "state did not start blank")

        barrier = Barrier(4, timeout=10)

        def solve(index):
            barrier.wait()
            return cli.solve(state, f"parallel-{index}", trigger)

        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = sorted(pool.map(solve, range(4)), key=lambda row: row["id"])

        require(len({row["id"] for row in outcomes}) == 4, "concurrent events reused an ID")
        first, *duplicates = outcomes
        require(first["decision"] == "kept" and first["generation_before"] == 0, "no initial committed upgrade")
        require(first["generation"] == 1 and first["after"]["ok"], "first commit is incomplete")
        for row in duplicates:
            require(row["decision"] == "unchanged" and row["before"]["ok"], "parallel repeat ran a new trial")
            require(row["generation_before"] == row["generation"] == 1 and not row["trials"],
                    "parallel solve did not see the committed revision")

        active = cli.response(state, "status")
        require(active["generation"] == 1 and active["events"] == 4, "commit count and active head disagree")
        require(active["policies"] == initial["policies"] + [first["nominee"]], "wrong policy was retained")
        for row in outcomes:
            require(cli.response(state, "show", str(row["id"])) == row, "persisted evidence changed on readback")
        evidence = check_evidence(first, transfer)

        restored = cli.solve(state, "new-process", transfer)
        require(restored["before"]["ok"] and not restored["trials"] and restored["generation"] == 1,
                "fresh process could not use the retained policy on a distinct task")
        reverted = cli.response(state, "rollback")
        require(reverted["decision"] == "rollback" and reverted["generation"] == 0, "rollback did not restore v0")
        current = cli.response(state, "status")
        require(current["generation"] == 0 and current["policies"] == initial["policies"],
                "fresh process did not read the restored initial policy")
        require(current["events"] == 6, "rollback did not append exactly one event")
        require(cli.response(state, "show", str(first["id"])) == first, "rollback rewrote retained evidence")
        require(cli.response(state, "show", str(reverted["id"])) == reverted, "rollback event differs on readback")

        path = cli.work_dir / "new-process.json"
        require(not cli.response(state, "probe", str(path))["result"]["ok"],
                "rolled-back policy still solved the transfer fixture")
        repeated = cli.solve(state, "repeat-after-rollback", trigger)
        require(repeated["before"] == first["before"] and repeated["after"] == repeated["before"],
                "same-generation repeat changed the active strategy result")
        require(repeated["decision"] == "unchanged" and repeated["reused_event"] == first["id"],
                "same-generation repeat did not link its original experiment")
        require(not repeated["trials"] and repeated["answer"]["source"] == "record", "repeat did not reuse the recorded answer")
        validate_packing(trigger, repeated["answer"]["packing"], complete=True)
        current = cli.response(state, "status")
        require(current["generation"] == 0 and current["events"] == 7, "repeat changed revision or event count")

        bad = cli.work_dir / "invalid.json"
        bad.write_text('{"capacity":13,"bins":true,"items":[1]}', encoding="utf-8")
        invalid = cli.call(state, "solve", str(bad))
        require(invalid.returncode != 0 and "bins" in invalid.stderr, "invalid request lacks specific error evidence")
        require(cli.response(state, "status") == current, "invalid request changed the state or evidence count")
        rejection = check_rejection(cli)
        print(json.dumps({"parallel": [row["decision"] for row in outcomes], "evidence": evidence,
                          "rollback": reverted["generation"], "reused_event": repeated["reused_event"],
                          "events": current["events"], "invalid_exit": invalid.returncode,
                          "rejection": rejection}))


if __name__ == "__main__":
    try:
        check_integrity()
    except AuditError as error:
        sys.exit(f"integrity audit failed: {error}")
