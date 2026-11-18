"""Standalone checks on synthetic manifests; no evaluation data or generation labels."""
import itertools
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from candidates.pilot_programs import recipes
from packlab import arena, engine

STEPS = ([], ["normalize_gcd"], ["peel_forced"], list(recipes.STEPS), list(reversed(recipes.STEPS)))


def exact(task):
    sizes = sorted(task["items"], reverse=True)

    @lru_cache(None)
    def fits(i, gaps):
        if i == len(sizes):
            return True
        for j, gap in enumerate(gaps):
            if sizes[i] <= gap:
                nxt = gaps[:j] + (gap - sizes[i],) + gaps[j+1:]
                if fits(i+1, tuple(sorted(nxt))):
                    return True
        return False

    return fits(0, (task["capacity"],) * task["bins"])


def program(steps, solver, guard=None):
    return dict(kind="recipe", steps=steps, solver=solver, guard=guard)


def check_run(task, p, truth):
    original = json.dumps([task, p])
    r = recipes.run(task, p)
    assert json.loads(json.dumps(r)) == r and original == json.dumps([task, p])
    assert sum(r["trace"]["ops"].values()) == r["cost"] > 0
    assert len(r["packing"]) == task["bins"]
    ids = [i for group in r["packing"] for i in group]
    assert len(ids) == len(set(ids)) and all(0 <= i < len(task["items"]) for i in ids)
    assert all(sum(task["items"][i] for i in g) <= task["capacity"] for g in r["packing"])
    if r["ok"]:
        assert truth and arena.check(task, r["packing"])
    blocked = r["trace"]["blocked"]
    if blocked:
        assert task["items"][blocked["index"]] == blocked["size"]
        assert len(blocked["free"]) == task["bins"] and min(blocked["free"]) >= 0
    if (r["trace"]["search_status"] == "infeasible" or
            (blocked and blocked["reason"] in ("oversize", "total_excess"))):
        assert not truth
    return r


def examples():
    fill = dict(engine.BASE, order="descending", repair="fill")
    triples = dict(fill, repair="triples")
    unit_task = dict(capacity=10000, bins=2, items=[5000,4000,3000,2000,2000,2000,2000])
    peel_task = dict(capacity=13, bins=3, items=[4,4,12,4,4,5,5])
    both_task = dict(capacity=10000, bins=3, items=[2000,9999,5000,2000,4000,2000,3000,2000])
    for name, task, solver, expected in [
        ("unit", unit_task, fill, (False, True, False, True, True)),
        ("isolated", peel_task, triples, (False, False, True, True, True)),
        ("order", both_task, fill, (False, False, False, False, True)),
    ]:
        results = [check_run(task, program(steps, solver), True) for steps in STEPS]
        assert tuple(r["ok"] for r in results) == expected
        print(name, [(r["ok"], r["cost"]) for r in results], "restored", results[-1]["packing"])
    p = program(list(reversed(recipes.STEPS)), fill, dict(feature="gcd", op="ge", value=2))
    skip = check_run(both_task, p, True)
    assert skip["trace"]["search_status"] == "guard_skip" and not skip["trace"]["solver_ran"]
    assert skip["trace"]["guard"]["observed"] == 1 and not skip["trace"]["guard"]["matched"]
    p["guard"]["feature"] = "forced_singletons"
    p["guard"]["value"] = 1
    assert check_run(both_task, p, True)["ok"]
    # Pure local excess must not be presented as excess in the original units.
    residual = dict(capacity=10, bins=2, items=[8,5,4,3])
    result = check_run(residual, program(["peel_forced"], engine.BASE), False)
    assert result["trace"]["blocked"]["reason"] == "residual_total_excess"


def check_general():
    rng = random.Random(51092)
    tasks = [dict(capacity=c, bins=b, items=list(xs)) for c,b,xs in [
        (5, 2, ()), (7, 1, (7,)), (7, 4, (7,7)), (10, 2, (7,7,7)),
        (10, 2, (10,10,1)), (10, 2, (11,)), (10, 2, (3,3,7,7)),
        (10**9, 16, tuple([10**9]*16)), (10**9, 2, (500000001,500000001,500000001))]]
    for _ in range(200):
        unit = rng.choice((1,3,10000))
        tasks.append(dict(capacity=unit*rng.randrange(3,14), bins=rng.randrange(1,5),
                          items=[unit*rng.randrange(1, 14) for _ in range(rng.randrange(14))]))
    checked = 0
    for t in tasks:
        truth = exact(t)
        obs = recipes.features(t)
        assert tuple(obs) == recipes.FEATURES and all(type(x) is int for x in obs.values())
        xs, c = t["items"], t["capacity"]
        assert obs["forced_singletons"] == sum(
            size <= c and all(i == j or size + other > c for j,other in enumerate(xs))
            for i,size in enumerate(xs))
        for steps, repair in itertools.product(STEPS, ("none","exchange","search","triples","fill")):
            p = program(steps, dict(engine.BASE, repair=repair))
            check_run(t, p, truth)
            checked += 1
    for s, g in ((["unknown"], None), (["peel_forced"]*2, None),
                 ([], {"feature":"label","op":"le","value":3}),
                 ([], {"feature":"gcd","op":"eq","value":0}),
                 ([], {"feature":"gcd","op":"le","value":False})):
        try:
            recipes.validate(program(s, engine.BASE, g))
        except ValueError:
            continue
        raise AssertionError("invalid recipe accepted")
    print(checked, "recipe results checked against independent placements and original indices")


if __name__ == "__main__":
    examples()
    check_general()
