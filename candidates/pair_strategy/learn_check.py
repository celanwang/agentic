"""Self-built feasible counterfactuals; verify routing search and its accounting."""
import json
from pathlib import Path
from random import Random
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True

from packlab import arena, engine, recipes, routing
from packlab.experiment import Bench


def pairs():
    rng = Random(1199)
    tasks = []
    for _ in range(6):
        small = sorted(rng.sample(range(27, 49), 7))
        big = [101 - x for x in small]
        positive = dict(capacity=101, bins=8, items=small + big)
        assert arena.check(positive, [[i, i + 7] for i in range(7)] + [[]])
        altered = small[:]
        altered[2] += 1
        altered[4] -= 1
        negative = dict(positive, items=[s for pair in zip(altered, big) for s in pair])
        witness = [[2*i, 2*i+1] for i in range(7)]
        witness.append([witness[2].pop()])
        assert arena.check(negative, witness)
        assert recipes.features(positive) == recipes.features(negative)
        tasks.extend([positive, negative])
    return tasks


def run(tasks, policies, skills, budget=128):
    charged = []
    apply = Bench.apply

    def measured(bench, task, route):
        before = len(bench.cache)
        result = apply(bench, task, route)
        new = list(bench.cache.values())[before:]
        charged.append((int(bool(new)), len(new), sum(r["cost"] for r in new)))
        assert not result["ok"] or arena.check(task, result["packing"])
        return result

    with patch.object(Bench, "apply", measured):
        report = routing.search(tasks, policies, skills, budget)
    assert sum(c for c, _, _ in charged) == report["calls"] <= budget
    assert sum(n for _, n, _ in charged) == report["physical_calls"]
    assert sum(v for _, _, v in charged) + report["feature_spent"] + report["fit_spent"] == report["spent"]
    assert sum(report[k] for k in ("baseline_calls", "fallback_calls", "candidate_calls", "verification_calls")) == report["calls"]
    if report["finalist"]:
        ordinary = [p for p in policies if p.get("kind") != "recipe"]
        for i, verified in enumerate(report["verification"]):
            direct = Bench().apply(tasks[i], [report["finalist"], *ordinary])
            assert (direct["cost"], direct["ok"]) == (verified["cost"], verified["ok"])
    for saved in report["practice_solutions"]:
        raw = engine.run(tasks[saved["task_index"]], report["candidates"][saved["candidate"]]["program"])
        assert raw["cost"] == saved["cost"] and arena.check(tasks[saved["task_index"]], raw["packing"])
    return report


def check():
    tasks = pairs()
    base = engine.BASE
    descending = dict(base, order="descending")
    old = dict(kind="recipe", steps=[], solver=base, guard=dict(feature="bins", op="ge", value=12))
    controls = [dict(base, repair="search"), descending]
    primary = run(tasks, [base], controls)
    program = primary["program"]
    assert program["guard"] == dict(feature="complementary_coverage", op="ge", value=1000)
    assert program["solver"] == descending and "fallback" not in program
    assert primary["comparison"]["gains"] == 6 and primary["comparison"]["losses"] == 0
    assert [r["guard"]["matched"] for r in primary["verification"]] == [True, False] * 6

    # Blind the learner to the new observation. All earlier features coincide
    # within each pair, so no old predicate can isolate the six beneficiaries.
    observe = recipes.route_features
    def original_features(t, *, ops=None):
        values = observe(t, ops=ops)
        del values["complementary_coverage"]
        return values
    with patch.object(recipes, "route_features", original_features):
        without = run(tasks, [base], controls)
    assert without["program"] is None or without["comparison"]["gains"] < 6
    assert run(tasks[::2], [base], [descending])["program"] is None
    routed = run(tasks, [old, base], [descending])
    assert routed["program"]["fallback"] == old and routed["comparison"]["gains"] == 6
    upgraded = run(tasks, [routed["program"], base], controls)
    assert upgraded["fallback_calls"] == len(tasks)
    if upgraded["program"]:
        assert upgraded["program"]["fallback"] == old
    assert run(tasks, [base], controls, 23)["calls"] == 0
    for budget in (24, 35, 36, 47, 48, 63, 128):
        run(tasks, [base], controls, budget)
    assert primary["candidate_calls"] < len(primary["trials"])
    assert routed["baseline_calls"] == 2 * len(tasks)
    constrained = run(tasks, [old, base], controls * 2 + [dict(base, pick="roomy")], 49)
    assert constrained["calls"] <= 49

    # Different practice behavior can select another ordinary solver.
    others = []
    for a, b in ((3, 7), (4, 7), (5, 7), (5, 9), (6, 11), (8, 13)):
        for d in (0, 1):
            others.append(dict(capacity=31, bins=2, items=[a,b+d,31-a-d,31-b]))
    alternate = run(others, [base], [descending, dict(base, pick="roomy")])
    assert alternate["program"]["solver"]["pick"] == "roomy"
    return {key: {k: report[k] for k in ("calls", "physical_calls", "baseline_calls", "candidate_calls", "verification_calls",
                                         "feature_spent", "fit_spent", "spent", "program", "comparison")}
            for key, report in (("matched_feasible_controls", primary), ("legacy_features_only", without),
                                ("existing_fallback", routed), ("alternative_solver", alternate))}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
