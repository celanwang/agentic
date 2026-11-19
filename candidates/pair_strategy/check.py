"""Check matching, one-level routing, proof exits, and full operation accounting."""
from functools import lru_cache
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packlab import arena, engine, recipes
from packlab.experiment import Bench, proven_impossible


def matching_oracle(capacity, sizes):
    @lru_cache(None)
    def maximum(left):
        if not left:
            return 0
        best = maximum(left[1:])
        for j, x in enumerate(left[1:], 1):
            if left[0] + x == capacity:
                best = max(best, 2 + maximum(left[1:j] + left[j+1:]))
        return best

    return 1000 * maximum(tuple(sizes)) // len(sizes) if sizes else 0


def check_coverage():
    cases = 0
    for capacity in range(1, 9):
        for n in range(7):
            for sizes in combinations_with_replacement(range(1, capacity + 2), n):
                task = dict(capacity=capacity, bins=4, items=list(sizes))
                ops = {}
                observed = recipes.route_features(task, ops=ops)
                coverage = observed.pop("complementary_coverage")
                assert coverage == matching_oracle(capacity, sizes)
                assert observed == recipes.features(task)
                assert ops["recipe_complements"] == n + len(set(sizes))
                again = dict(ops)
                recipes.route_features(task, ops=ops)
                assert ops == {k: 2 * v for k, v in again.items()}
                cases += 1
    print(f"{cases} coverage observations match exhaustive non-overlapping file matching")


def recipe(policy=None, guard=None, steps=()):
    return dict(kind="recipe", solver=policy if policy is not None else engine.BASE,
                steps=list(steps), guard=guard)


def routes(task, primary, fallback, selected, ok):
    a = recipes.run(task, primary)
    expected = recipes.run(task, fallback) if selected == "fallback" else a
    routed = dict(primary, fallback=fallback)
    result = recipes.run(task, routed)
    assert json.loads(json.dumps(result)) == result
    assert result["ok"] == expected["ok"] == ok and result["packing"] == expected["packing"]
    assert result["trace"]["guard"] == a["trace"]["guard"]
    assert result["trace"]["route"]["selected"] == selected
    primary_cost = a["cost"]
    fallback_cost = 0
    if selected == "fallback":
        fallback_cost = expected["cost"] - (3 + len(task["items"]))
        if a["trace"]["search_status"] == "guard_skip":
            primary_cost -= task["bins"]
    assert result["trace"]["route"]["primary"]["cost"] == primary_cost
    assert result["trace"]["blocked"] == expected["trace"]["blocked"]
    assert result["trace"]["search_status"] == expected["trace"]["search_status"]
    branch = result["trace"]["route"]["fallback"]
    assert branch == (dict(ok=expected["ok"], cost=fallback_cost,
                           guard=expected["trace"]["guard"],
                           search_status=expected["trace"]["search_status"])
                      if selected == "fallback" else None)
    assert result["cost"] == 1 + primary_cost + fallback_cost
    assert result["cost"] == sum(result["trace"]["ops"].values())
    if ok:
        assert arena.check(task, result["packing"])
    return result


def check_routing():
    coverage = dict(feature="complementary_coverage", op="ge", value=1000)
    bins = dict(feature="bins", op="ge", value=1)
    desc = dict(engine.BASE, order="descending")
    pairs = dict(capacity=10, bins=2, items=[4, 4, 6, 6])
    mixed = dict(capacity=10, bins=2, items=[4, 3, 10, 3])
    leaf = recipe(desc, steps=("peel_forced",))
    first = routes(pairs, recipe(desc, guard=coverage), leaf, "primary", True)
    assert first["trace"]["ops"]["recipe_complements"] == 6
    skipped = routes(mixed, recipe(desc, guard=coverage), leaf, "fallback", True)
    assert skipped["trace"]["guard"]["matched"] is False
    assert skipped["packing"][0] == [2]
    routes(pairs, recipe(guard=coverage), leaf, "fallback", True)
    proof = routes(dict(capacity=10, bins=3, items=[11]), recipe(guard=bins),
                   leaf, "primary", False)
    assert proven_impossible(proof)
    shortage = routes(dict(capacity=10, bins=2, items=[9, 8, 7]),
                      recipe(guard=bins, steps=("peel_forced",)), leaf, "primary", False)
    assert proven_impossible(shortage)
    big = dict(capacity=4095, bins=2, items=[100] * 40 + [2095] * 2)
    repaired = routes(big, recipe(dict(engine.BASE, repair="fill"), guard=bins),
                      recipe(dict(engine.BASE, repair="fill_deep")), "fallback", True)
    assert repaired["trace"]["route"]["primary"]["search_status"] == "limit"
    for bad in (dict(leaf, fallback=leaf), dict(recipe(guard=bins), fallback=dict(leaf, fallback=leaf)),
                dict(recipe(guard=bins), fallback=None)):
        try:
            recipes.validate(bad)
        except ValueError:
            continue
        raise AssertionError("accepted an invalid route")
    # Terminal unknown releases control to the external old ordinary chain.
    route = dict(recipe(guard=coverage), fallback=recipe())
    result = Bench().apply(pairs, [route, desc])
    assert result["ok"] and arena.check(pairs, result["packing"]) and len(result["attempts"]) == 2
    assert result["cost"] == sum(a["cost"] for a in result["attempts"])
    print("Route paths checked: primary, guard skip, unresolved, limit, proof, external ordinary fallback")


if __name__ == "__main__":
    check_coverage()
    check_routing()
