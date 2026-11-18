"""Precommitted prefix ablation on the four historical v2 failures."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from candidates.triples_strategy import engine as v2

DB = ROOT / "state/triples/20261118-200049-3qo0skt6/agentic.sqlite3"
BASE_SHA = "f2be6568d4e580fb0fd2dbece6d3f4d8df9a0844cd5b97c28773a500c1f2b6cf"


def cases():
    assert hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest() == BASE_SHA
    with sqlite3.connect(DB.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        event = json.loads(db.execute("SELECT data FROM events WHERE id=4").fetchone()[0])
    assert event["seed"] == 282459
    for row in (20, 23, 25, 28):
        data = event["holdout"]["rows"][row]
        assert data["profile"] == "mixed" and data["feasible"] and not data["new_ok"]
        yield row, data["task"]


def valid(task, groups):
    return (len(groups) == task["bins"] and sorted(i for g in groups for i in g) ==
            list(range(len(task["items"]))) and
            all(sum(task["items"][i] for i in g) <= task["capacity"] for g in groups))


def probe(task, rollback):
    sizes = task["items"]
    greedy = v2.run(task, dict(v2.BASE, order="descending"))
    block = greedy["trace"]["blocked"]
    step = block["step"]
    kept = max(0, step - rollback)
    ops = dict(greedy=greedy["cost"], reconstruct_order=v2._rank_charge(len(sizes)),
               reconstruct_state=0, prefix_undo=0, nodes=0, bin_probes=0, placements=0, undo=0)
    order = sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True)
    bins = [group[:] for group in greedy["packing"]]
    free = block["free"][:]
    locations = {i: j for j, group in enumerate(bins) for i in group}
    ops["reconstruct_state"] = 2 * (len(free) + step) + len(order)
    for i in reversed(order[kept:step]):
        j = locations[i]
        assert bins[j].pop() == i
        free[j] += sizes[i]
        ops["prefix_undo"] += 1

    def visit(pos):
        if ops["nodes"] == 2048:
            return "limit"
        ops["nodes"] += 1
        if pos == len(order):
            return "solved"
        i = order[pos]
        seen = set()
        for b, gap in enumerate(free):
            ops["bin_probes"] += 1
            if gap < sizes[i] or gap in seen:
                continue
            seen.add(gap)
            free[b] -= sizes[i]
            bins[b].append(i)
            ops["placements"] += 1
            outcome = visit(pos + 1)
            if outcome == "solved":
                return outcome
            bins[b].pop()
            free[b] += sizes[i]
            ops["undo"] += 1
            if outcome == "limit":
                return outcome
        return "exhausted"

    status = visit(kept)
    if status == "solved":
        assert valid(task, bins)
    return dict(rollback=rollback, kept=kept, status=status,
                packing=bins if status == "solved" else None, cost=sum(ops.values()), ops=ops)


if __name__ == "__main__":
    print(json.dumps([dict(row=row, task=task, trials=[probe(task, k) for k in (0, 4, 8)])
                      for row, task in cases()], indent=2))
