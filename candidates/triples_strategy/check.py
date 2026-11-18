"""Independent small exact-partition oracle, legacy compatibility and bounds."""
import itertools
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import engine

sys.path.append(str(Path(__file__).resolve().parents[2]))
from candidates.strategy import engine as v1


def exact(capacity, bins, sizes):
    # Enumerate placements of individual sizes. No grouping or structural shortcuts.
    @lru_cache(None)
    def fits(i, gaps):
        if i == len(sizes):
            return True
        for j, g in enumerate(gaps):
            if g >= sizes[i]:
                updated = gaps[:j] + (g - sizes[i],) + gaps[j + 1:]
                if fits(i + 1, tuple(sorted(updated))):
                    return True
        return False

    return fits(0, (capacity,) * bins)


def validate(task, result):
    groups = result["packing"]
    assert len(groups) == task["bins"]
    indices = [i for group in groups for i in group]
    assert len(indices) == len(set(indices))
    assert all(0 <= i < len(task["items"]) for i in indices)
    assert all(sum(task["items"][i] for i in group) <= task["capacity"] for group in groups)
    assert result["ok"] == (len(indices) == len(task["items"]))
    assert sum(result["trace"]["ops"].values()) == result["cost"] > 0
    assert json.loads(json.dumps(result)) == result


def test_shapes():
    summary = dict(solved=0, infeasible=0, greedy_ok=0)
    for bins in (2, 3):
        for capacity in range(10, 33):
            domain = range(capacity // 4 + 1, capacity - 2 * (capacity // 4 + 1) + 1)
            for values in itertools.combinations_with_replacement(domain, 3 * bins):
                if sum(values) != bins * capacity:
                    continue
                task = dict(capacity=capacity, bins=bins, items=list(values))
                truth = exact(capacity, bins, values)
                policy = dict(engine.BASE, repair="triples")
                result = engine.run(task, policy)
                validate(task, result)
                assert result["ok"] == truth
                key = result["trace"]["search_status"] or "greedy_ok"
                summary[key] += 1
                first = engine.run(task)
                choices = engine.propose(task, engine.BASE, first)["candidates"]
                assert (policy in choices) == (not first["ok"])
                assert len(choices) <= 16
                if key == "infeasible":
                    assert not engine.propose(task, policy, result)["candidates"]
    assert all(summary.values())
    print("Shape-matched multisets checked against independent exact solver:", summary)


def test_guards_and_v1():
    rng = random.Random(8021)
    samples = [dict(capacity=10, bins=2, items=[4, 4, 6, 6]),
               dict(capacity=12, bins=2, items=[3, 3, 3, 5, 5, 5]),
               dict(capacity=12, bins=2, items=[6, 3, 4, 5, 3, 1])]
    for _ in range(60):
        task = dict(capacity=12, bins=rng.randrange(1, 5),
                    items=[rng.randrange(1, 15) for _ in range(rng.randrange(15))])
        samples.append(task)
    assert engine.BASE == v1.BASE
    for task in samples:
        assert engine.run(task) == v1.run(task)
        for order, pick, repair in itertools.product(v1._ORDERS, v1._PICKS, v1._REPAIRS):
            policy = dict(order=order, pick=pick, repair=repair)
            assert engine.run(task, policy) == v1.run(task, policy)
        for repair in engine._TRIPLE_LIMITS:
            policy = dict(engine.BASE, repair=repair)
            result = engine.run(task, policy)
            validate(task, result)
            choices = engine.propose(task, policy, result)["candidates"]
            fits, _ = engine._triple_shape(task["capacity"], task["bins"], task["items"])
            if not fits:
                assert not any(p["repair"] in engine._TRIPLE_LIMITS for p in choices)
                assert result["trace"]["ops"].get("group_nodes", 0) == 0
                assert result["trace"]["search_status"] != "infeasible"
    print(f"{len(samples)} manifests x 24 legacy policies: identical run results; shape gates passed")


def test_cutoff():
    # Three equal triples; deliberately call the search primitive with one node.
    sizes = [7, 7, 7, 6, 6, 6, 5, 5, 5]
    ops = dict(placements=0, undo=0)
    solution, status = engine._triple_search(list(range(len(sizes))), sizes, 18, 1, ops)
    assert status == "limit" and solution is None and ops["group_nodes"] == 1
    assert ops["placements"] == ops["undo"] == 3
    assert ops["group_candidates"] == 1 and ops["pair_probes"] >= 1
    print("Cutoff does not certify impossibility; attempted edge and undo charged")


if __name__ == "__main__":
    test_shapes()
    test_guards_and_v1()
    test_cutoff()
