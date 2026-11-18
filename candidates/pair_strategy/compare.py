"""Compare 40 independent exact-pair manifests with 40 matched feasible mixed ones."""
import argparse
import json
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packlab import arena, engine, recipes
from packlab.experiment import Bench

SEED = 2026111827
CASES = 40


def order_groups(rng, groups, small_first=False):
    indexed = [(s, j) for j, g in enumerate(groups) for s in g]
    rng.shuffle(indexed)
    if small_first:
        indexed.sort(key=lambda item: 2 * item[0] > sum(groups[0]))
    witness = [[] for _ in groups]
    for i, (_, j) in enumerate(indexed):
        witness[j].append(i)
    return [s for s, _ in indexed], witness


def manifests():
    rng = Random(SEED)
    capacities = rng.sample(range(50_000, 950_000), CASES)
    for i, capacity in enumerate(capacities):
        bins = 2 + i % 7
        small = rng.sample(range(capacity // 4 + 1, (capacity - 1) // 2 + 1), bins)
        groups = [[size, capacity - size] for size in small]
        sizes, witness = order_groups(rng, groups, small_first=True)
        yield i, "pairs", dict(capacity=capacity, bins=bins, items=sizes), witness
        # Swap two 2-file disks for one singleton and one 3-file disk.
        # This preserves the capacity, bin count, item count and exact total load.
        left, right = sorted(rng.sample(range(1, capacity), 2))
        mixed = [*groups[2:], [capacity], [left, right - left, capacity - right]]
        sizes, witness = order_groups(rng, mixed)
        yield i, "mixed", dict(capacity=capacity, bins=bins, items=sizes), witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    desc = dict(engine.BASE, order="descending")
    fallback = dict(kind="recipe", solver=dict(desc, repair="exchange"),
                    steps=["normalize_gcd", "peel_forced"],
                    guard=dict(feature="bins", op="le", value=4))
    route = dict(kind="recipe", steps=[], solver=desc, fallback=fallback,
                 guard=dict(feature="complementary_coverage", op="ge", value=1000))
    arms = dict(old=[fallback, engine.BASE], routed=[route, engine.BASE])
    rows = []
    match_stats = {}
    for i, profile, task, witness in manifests():
        assert arena.check(task, witness)
        stats = (task["capacity"], task["bins"], len(task["items"]), sum(task["items"]))
        assert len(task["items"]) == 2 * task["bins"] and stats[-1] == stats[0] * stats[1]
        if profile == "pairs":
            match_stats[i] = stats
        else:
            assert stats == match_stats[i]
        coverage = recipes.route_features(task)["complementary_coverage"]
        assert (coverage == 1000) == (profile == "pairs")
        bench = Bench()
        results = {key: bench.apply(task, policies) for key, policies in arms.items()}
        for result in results.values():
            assert not result["ok"] or arena.check(task, result["packing"])
        trace = bench.single(task, route)["trace"]
        rows.append(dict(pair_id=i, cohort=profile, task=task, packing=witness, coverage=coverage,
                         branch=trace["route"]["selected"], arms=results))
    summary = {}
    for label in ("pairs", "mixed"):
        group = [row for row in rows if row["cohort"] == label]
        summary[label] = dict(count=len(group), coverage=[min(r["coverage"] for r in group),
                                                        max(r["coverage"] for r in group)],
                              **{name: dict(ok=sum(r["arms"][name]["ok"] for r in group),
                                            cost=sum(r["arms"][name]["cost"] for r in group))
                                 for name in arms})
        assert len(group) == CASES
    print(json.dumps(dict(seed=SEED, summary=summary, **({"rows": rows} if args.json else {})), indent=2))


if __name__ == "__main__":
    main()
