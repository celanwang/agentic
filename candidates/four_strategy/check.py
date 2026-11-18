"""Test the fill primitive with a separate exact solver on synthetic tasks."""
import itertools
import random
from functools import lru_cache

import engine
from prefix import v2


def oracle(task):
    sizes = sorted(task["items"], reverse=True)

    @lru_cache(None)
    def place(i, gaps):
        if i == len(sizes):
            return True
        for j, gap in enumerate(gaps):
            if sizes[i] <= gap:
                next_gaps = gaps[:j] + (gap - sizes[i],) + gaps[j + 1:]
                if place(i + 1, tuple(sorted(next_gaps))):
                    return True
        return False

    return place(0, (task["capacity"],) * task["bins"])


def verify_packing(task, r):
    groups = r["packing"]
    assert len(groups) == task["bins"]
    ids = [i for g in groups for i in g]
    assert len(ids) == len(set(ids))
    assert all(0 <= i < len(task["items"]) for i in ids)
    assert all(sum(task["items"][i] for i in g) <= task["capacity"] for g in groups)
    assert (len(ids) == len(task["items"])) == r["ok"]
    assert sum(r["trace"]["ops"].values()) == r["cost"]


def compare(task, policy):
    result = engine.run(task, policy)
    verify_packing(task, result)
    old = v2.run(task, dict(policy, repair="none"))
    assert 0 <= result["cost"] - old["cost"] <= engine._FILL_LIMITS[policy["repair"]]
    truth = oracle(task)
    if result["ok"]:
        assert truth
    if result["trace"]["search_status"] == "infeasible":
        assert not truth
    return result


def try_fill():
    outcomes = {"solved": 0, "infeasible": 0}
    # Trigger the primitive directly on every small near-full multiset. This
    # also exercises combinations that ordinary first fit would solve first.
    for cap, count, n in ((9, 2, 6), (8, 3, 8), (11, 3, 9)):
        for sizes in itertools.combinations_with_replacement(range(1, cap + 1), n):
            if not 0 <= count * cap - sum(sizes) < cap:
                continue
            task = dict(capacity=cap, bins=count, items=sizes)
            ops = {}
            groups, status = engine._fill_search(list(range(n)), sizes, cap, count, False, 32768, ops)
            truth = oracle(task)
            assert (status == "solved") == truth and status != "limit"
            if groups is not None:
                verify_packing(task, {"packing": groups, "ok": True,
                                      "cost": sum(ops.values()), "trace": {"ops": ops}})
            outcomes[status] += 1
    assert min(outcomes.values()) > 0
    print("Independent exact comparisons:", outcomes)


def bounds_and_compatibility():
    task = dict(capacity=4095, bins=2, items=[100] * 40 + [2095] * 2)
    shallow = dict(engine.BASE, repair="fill")
    deep = dict(shallow, repair="fill_deep")
    lower = compare(task, shallow)
    upper = compare(task, deep)
    assert lower["trace"]["search_status"] == "limit" and upper["ok"]
    assert deep in engine.propose(task, shallow, lower)["candidates"]

    for c, sizes in [(10**9, [4*10**8, 4*10**8, 6*10**8, 6*10**8]),
                     (4096, [1600, 1600, 2496, 2496])]:
        other = dict(capacity=c, bins=2, items=sizes)
        trial = compare(other, shallow)
        assert trial["trace"]["search_status"] == "not_applicable"
        assert "fill_dp" not in trial["trace"]["ops"]
        assert not any(p["repair"] in engine._FILL_LIMITS for p in engine.propose(other, shallow, trial)["candidates"])

    rng = random.Random(43129)
    policy_grid = list(itertools.product(v2._ORDERS, v2._PICKS, v2._REPAIRS))
    for _ in range(100):
        task = dict(capacity=rng.randint(6, 24), bins=rng.randint(1, 5),
                    items=[rng.randint(1, 10) for _ in range(rng.randint(0, 18))])
        for order, pick, repair in policy_grid:
            policy = dict(order=order, pick=pick, repair=repair)
            assert engine.run(task, policy) == v2.run(task, policy)
        for repair in engine._FILL_LIMITS:
            compare(task, dict(engine.BASE, repair=repair))
    # Same task, two distinct observations. Only an actually attempted and
    # inapplicable triples repair motivates the new first proposal.
    task = dict(capacity=10, bins=2, items=[4, 4, 6, 6])
    for policy in (engine.BASE, dict(engine.BASE, repair="triples")):
        result = engine.run(task, policy)
        repairs = [p["repair"] for p in engine.propose(task, policy, result)["candidates"]]
        assert ("fill" in repairs) == (result["trace"]["search_status"] == "not_applicable")
        assert "fill_deep" not in repairs
    assert engine.BASE == v2.BASE
    print("Short limit/long solution, 10^9-size guard and", 100 * len(policy_grid), "v2 result dictionaries checked")


if __name__ == "__main__":
    try_fill()
    bounds_and_compatibility()
