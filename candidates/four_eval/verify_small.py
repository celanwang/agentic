"""Audit the two historical diagnostics without consulting stored witnesses."""

import json
from pathlib import Path
import subprocess
import sys

from candidates.triples_eval import blind_probe as common


HERE = Path(__file__).resolve().parent
STRATEGY = common.PROJECT / "candidates/four_strategy"


def fixed_prefix(task, count):
    files, bins, capacity = task["items"], task["bins"], task["capacity"]
    order = sorted(range(len(files)), key=files.__getitem__, reverse=True)
    free = [capacity] * bins
    pinned = [[] for _ in free]
    for index in order[:count]:
        disk = next(disk for disk, room in enumerate(free) if files[index] <= room)
        pinned[disk].append(index)
        free[disk] -= files[index]
    return pinned, free, [files[index] for index in order[count:]]


def count_suffix_states(free, remaining, limit):
    """Independent breadth-first enumeration of sorted disk capacity states."""
    states = {tuple(sorted(free))}
    visited = 1
    for size in remaining:
        successors = set()
        for state in states:
            for disk, room in enumerate(state):
                if room < size:
                    continue
                child = tuple(sorted((*state[:disk], room - size, *state[disk + 1:])))
                if child in successors:
                    continue
                if visited == limit:
                    return "limit", visited
                successors.add(child)
                visited += 1
        if not successors:
            return "exhausted", visited
        states = successors
    return "solved", visited


def replay(name):
    snapshot = json.loads((STRATEGY / (name + ".json")).read_text())
    current = subprocess.check_output([sys.executable, "-B", str(STRATEGY / (name + ".py"))], cwd=common.PROJECT)
    assert json.loads(current) == snapshot
    return snapshot


def main():
    diagnostic = {row["row"]: row for row in json.loads((HERE / "diagnostic.json").read_text())["rows"]}
    for row in replay("prefix"):
        task = row["task"]
        history = diagnostic[row["row"]]
        assert task == history["task"]
        for trial in row["trials"]:
            rollback, kept = trial["rollback"], trial["kept"]
            assert rollback in (0, 4, 8) and kept == history["attempts"][1]["blocked"]["step"] - rollback
            pinned, free, remaining = fixed_prefix(task, kept)
            status, states = count_suffix_states(free, remaining, 2048)
            assert status == trial["status"] and trial["ops"]["nodes"] <= 2048
            assert trial["cost"] == sum(trial["ops"].values())
            if status == "solved":
                assert common.check(task, trial["packing"])
                assert all(set(old) <= set(new) for old, new in zip(pinned, trial["packing"]))
            print(f"prefix row={row['row']} rollback={rollback}: {status}; independent states={states}")

    costs = {}
    for trial in replay("compare"):
        row = trial["row"]
        assert trial["repair"] in {"fill", "fill_deep"}
        entry = diagnostic[row]
        assert trial["status"] == "solved" and common.check(entry["task"], trial["packing"])
        assert trial["greedy"] == entry["attempts"][1]["cost"]
        ops = trial["search_ops"]
        assert all(type(value) is int and value >= 0 for value in ops.values())
        assert 0 < sum(ops.values()) == trial["cost"] - trial["greedy"] <= 4096
        assert ops["fill_nodes"] == entry["task"]["bins"] and ops["undo"] == 0
        if row in costs:
            assert costs[row] == (trial["cost"], trial["packing"], ops)
        else:
            costs[row] = trial["cost"], trial["packing"], ops
        print(f"fill row={row} repair={trial['repair']}: {trial['cost']} ops, {ops['fill_nodes']} states")


if __name__ == "__main__":
    main()
