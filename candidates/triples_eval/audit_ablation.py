"""Verify the frozen diagnostic against event 3 and repack certificates."""

from collections import Counter
from contextlib import closing
import json
import subprocess
import sys

from .blind_probe import PROJECT, V1, check, fingerprint, signature, source_database


STRATEGY = PROJECT / "candidates/triples_strategy"
REPORT_SHA256 = "aaec9ad433a85703163720179e742d28ac0a9cf920f40927bfd67c6de61ba70e"
SCRIPT_SHA256 = "3c22a677343c14fa7dbd74496a8b2fad61734e3764d10bbee3a9d795e9dae0db"
METHODS = {"search": 256, "deep": 2048, "triples": 256, "triples_deep": 2048}


def baseline_cost(task):
    bins = task["bins"]
    descending = sorted(task["items"], reverse=True)
    count = len(descending)
    cost = 1 + bins + count + count * (count - 1).bit_length()
    free = [task["capacity"]] * bins
    for step, size in enumerate(descending):
        for disk, left in enumerate(free):
            cost += 1
            if left >= size:
                free[disk] -= size
                cost += 1
                break
        else:
            return cost + count + bins + step
    raise AssertionError("historical case succeeded under descending first-fit")


def history_rows():
    with closing(source_database(V1)) as db:
        event = json.loads(db.execute("SELECT data FROM events WHERE id=3").fetchone()[0])
    assert event["seed"] == 177730
    rows = {index: row for index, row in enumerate(event["holdout"]["rows"])
            if row["profile"] == "triples" and not row["old_ok"] and not row["new_ok"]}
    assert len(rows) == len({signature(row["task"]) for row in rows.values()}) == 12
    return rows


def main():
    report_raw = (STRATEGY / "ablation.json").read_bytes()
    script = STRATEGY / "ablate.py"
    assert fingerprint(report_raw) == REPORT_SHA256 and fingerprint(script.read_bytes()) == SCRIPT_SHA256
    report = json.loads(report_raw)
    measured = subprocess.check_output([sys.executable, "-B", str(script)], cwd=PROJECT)
    assert json.loads(measured) == report
    rows = history_rows()
    matrix = {(row["row"], row["repair"]): row for row in report["rows"]}
    assert list(matrix) == [(index, name) for index in rows for name in METHODS]
    charges = Counter()
    for (index, method), row in matrix.items():
        historical = rows[index]
        task = historical["task"]
        bins, capacity, files = task["bins"], task["capacity"], task["items"]
        assert row["task"] == task
        assert len(files) == 3 * bins and sum(files) == bins * capacity
        assert all(4 * size > capacity and 2 * size < capacity for size in files)
        assert row["greedy_cost"] == baseline_cost(task)
        assert sum(row["extra_ops"].values()) + row["greedy_cost"] == row["cost"]
        assert all(isinstance(value, int) and value >= 0 for value in row["extra_ops"].values())
        assert 1 <= row["nodes"] <= METHODS[method]
        if row["ok"]:
            assert row["status"] == "solved" and check(task, row["packing"])
        else:
            assert row["status"] == "limit" and row["nodes"] == METHODS[method]
            assert method in {"search", "deep"}
            assert row["cost"] - row["greedy_cost"] == row["nodes"] * (bins + 3)
        if method == "search":
            assert row["cost"] == historical["new_cost"] - historical["old_cost"]
        if method == "triples":
            assert row["cost"] < matrix[index, "search"]["cost"]
            same = matrix[index, "triples_deep"]
            assert all(row[key] == same[key] for key in ("packing", "cost", "nodes", "extra_ops"))
            charges.update(row["extra_ops"])

    print(json.dumps({"history_event": 3, "history_seed": report["seed"], "diagnostic_cases": len(rows),
                      "summary": report["summary"],
                      "triples_charges_after_shared_greedy": {key: value for key, value in charges.items() if value}},
                     indent=2))


if __name__ == "__main__":
    main()
