"""Read-only, precommitted comparison of the published #9 pair routing boundary."""
import argparse
from contextlib import closing
from functools import reduce
import json
from math import gcd
from pathlib import Path
from random import Random
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packlab import engine
from packlab.experiment import Bench

SEED = 2026111809
PER_SIZE = 10
DEFAULT_STATE = ROOT / "state/pilot/round-01/stream"


def published_chain(folder):
    path = (folder / "agentic.sqlite3").resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        event = db.execute("""SELECT decision, json_extract(data, '$.generation'),
                                     json_extract(data, '$.nominee')
                              FROM events WHERE id=9 AND kind='research'""").fetchone()
        if event is None or event[0] != "kept":
            raise ValueError("找不到已发布且晋级的研究记录 #9")
        revision, nominee = event[1], json.loads(event[2])
        row = db.execute("SELECT policies FROM revisions WHERE id=?", (revision,)).fetchone()
    policies = json.loads(row[0])
    assert policies[0] == nominee and nominee["guard"] == {"feature": "bins", "op": "le", "value": 4}
    return revision, policies


def valid(task, packing):
    sizes = task["items"]
    return (len(packing) == task["bins"] and
            sorted(i for group in packing for i in group) == list(range(len(sizes))) and
            all(sum(sizes[i] for i in group) == task["capacity"] for group in packing))


def manifests():
    rng = Random(SEED)
    capacities, families = set(), set()
    for bins in range(2, 9):
        for _ in range(PER_SIZE):
            capacity = rng.randrange(100_000, 1_000_000)
            while capacity in capacities:
                capacity = rng.randrange(100_000, 1_000_000)
            capacities.add(capacity)
            small = rng.sample(range(capacity // 4 + 1, (capacity - 1) // 2 + 1), bins)
            permutation = list(range(bins))
            rng.shuffle(permutation)
            large = [capacity - small[i] for i in permutation]
            sizes = small + large
            unit = reduce(gcd, sizes, capacity)
            key = bins, capacity // unit, tuple(sorted(x // unit for x in sizes))
            assert key not in families
            families.add(key)
            task = dict(capacity=capacity, bins=bins, items=sizes)
            witness = [[i, bins + j] for j, i in enumerate(permutation)]
            assert min(small) * 4 > capacity and max(small) * 2 < capacity < min(large) * 2
            assert valid(task, witness)
            yield task, witness


def compare(folder):
    revision, active = published_chain(folder)
    unguarded = [dict(active[0], guard=None), *active[1:]]
    chains = dict(active=active, without_guard=unguarded,
                  descending=[dict(engine.BASE, order="descending")], cold=[engine.BASE])
    rows = []
    for task, witness in manifests():
        bench = Bench()
        arms = {}
        for name, chain in chains.items():
            result = bench.apply(task, chain)
            assert result["cost"] == sum(attempt["cost"] for attempt in result["attempts"])
            if result["ok"]:
                assert valid(task, result["packing"])
            arms[name] = {k: result[k] for k in ("ok", "cost", "attempts", "packing")}
        trace = bench.single(task, active[0])["trace"]
        rows.append(dict(task=task, witness=witness, guard_matched=trace["guard"]["matched"], arms=arms))
    summary = []
    for bins in range(2, 9):
        subset = [row for row in rows if row["task"]["bins"] == bins]
        aggregates = {}
        for name in chains:
            results = [row["arms"][name] for row in subset]
            costs = [arm["cost"] for arm in results]
            aggregates[name] = dict(ok=sum(r["ok"] for r in results), total_cost=sum(costs),
                                    min_cost=min(costs), max_cost=max(costs))
        summary.append(dict(bins=bins, cases=len(subset), guard_matches=sum(r["guard_matched"] for r in subset),
                            arms=aggregates))
    return dict(seed=SEED, cases_per_size=PER_SIZE, published_event=9, revision=revision,
                chains=chains, summary=summary, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--json", action="store_true", help="include every input, witness and attempt")
    args = parser.parse_args()
    report = compare(args.state)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"seed={SEED} version={report['revision']} per_size={PER_SIZE}")
        print("bins guard active no_guard descending cold")
        for row in report["summary"]:
            print(row["bins"], row["guard_matches"],
                  *(f"{r['ok']}/{PER_SIZE}({r['total_cost']})" for r in row["arms"].values()))
