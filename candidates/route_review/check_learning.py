"""Exercise bounded relearning on independent three-piece-per-bin constructions."""
import json
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidates.pair_strategy.learn_check import run
from packlab import arena, engine, research
from packlab.experiment import Bench

BASE = engine.BASE
DESC = dict(BASE, order="descending")


def leaf(solver, feature, op, value, steps=()):
    return dict(kind="recipe", steps=list(steps), solver=solver,
                guard=dict(feature=feature, op=op, value=value))


def triples(rng, capacity, prefer_desc):
    # Cut each full disk at two independently drawn integer positions.
    # Keep examples where arrival and descending have different outcomes.
    for _ in range(1000):
        bins = rng.randint(3, 6)
        items = []
        for _ in range(bins):
            a, b = sorted(rng.sample(range(1, capacity), 2))
            items.extend((a, b - a, capacity - b))
        if prefer_desc:
            rng.shuffle(items)
        task = dict(capacity=capacity, bins=bins, items=items)
        a, b = (engine.run(task, p) for p in (BASE, DESC))
        if a["ok"] != b["ok"] and b["ok"] == prefer_desc:
            assert arena.check(task, (b if prefer_desc else a)["packing"])
            return task
    raise AssertionError("no contrasting example found")


def check_costs(tasks, policies, report):
    # Verify every fitted intervention at its learned threshold, including
    # candidates which lost the internal ranking. Reuse the measured endpoints.
    for model in report["fitted"]:
        probe = report["candidates"][model["candidate"]]["program"]
        if model["position"] == "primary":
            program = dict(probe, guard=model["guard"])
        else:
            program = dict(probe, fallback=dict(probe["fallback"], guard=model["guard"]))
        charged = {row["task_index"] for row in report["trials"] if row["candidate"] == model["candidate"]}
        for i, task in enumerate(tasks):
            endpoint = probe if i in charged and i in model["matched"] else policies[0]
            bench = Bench()
            expected, actual = (bench.apply(task, [p, *policies[1:]]) for p in (endpoint, program))
            assert (expected["ok"], expected["cost"], expected["packing"]) == (actual["ok"], actual["cost"], actual["packing"])


def check_panel(position, seed):
    rng = Random(seed)
    positives, negatives = (57, 61, 67), (23, 29, 31)
    guard = leaf(DESC, "capacity", "ge", 91, steps=["normalize_gcd"])
    if position == "primary":
        old = dict(guard, fallback=dict(leaf(BASE, "bins", "le", 16), guard=None))
    else:
        positives, negatives = negatives, positives
        old = dict(leaf(BASE, "items", "ge", 40),
                   fallback=dict(guard, guard=dict(feature="capacity", op="le", value=12)))
    tasks = [triples(rng, c, True) for c in positives] + [triples(rng, c, False) for c in negatives]
    report = run(tasks, [old, BASE], [dict(BASE, repair="search"), DESC])
    check_costs(tasks, [old, BASE], report)
    program = report["program"]
    assert report["comparison"]["gains"] == len(positives) and not report["comparison"]["losses"]
    model = next(m for m in report["fitted"] if m["position"] == position)
    assert model["prediction"] == report["comparison"]
    learned = program if position == "primary" else program["fallback"]
    original = old if position == "primary" else old["fallback"]
    limit = min(positives) if position == "primary" else max(positives)
    assert learned["guard"] == dict(original["guard"], value=limit)
    assert program == (dict(old, guard=learned["guard"]) if position == "primary" else
                       dict(old, fallback=dict(original, guard=learned["guard"])))
    assert report["fallback_calls"] == 0
    assert run(tasks, [old, BASE], [], 3 * len(tasks))["program"] is None
    assert run(tasks + [tasks[0]], [old, BASE], [], 28)["calls"] <= 28

    # The practiced limits leave large unseen parts of both old and new domains.
    fresh = [triples(rng, c, want) for c in (9, 27, 38, 55, 75, 113) for want in (True, False)]
    fresh.append(dict(capacity=101, bins=5, items=[39, 41, 43, 45, 47, 62, 60, 58, 56, 54]))
    old_ok = gains = 0
    for task in fresh:
        actual = Bench()
        before, after = (actual.apply(task, [route, BASE]) for route in (old, program))
        assert not before["ok"] or after["ok"]
        old_ok += before["ok"]
        gains += after["ok"] and not before["ok"]
        if after["ok"]:
            assert arena.check(task, after["packing"])
    assert old_ok and gains
    return dict(guard=learned["guard"], practice=report["comparison"], calls=report["calls"],
                independent_cases=len(fresh), retained=old_ok, gains=gains)


def public_replay():
    # The completed, public round 11 probe. Only its twelve failures feed practice.
    fixture = Path(__file__).resolve().parents[1] / "pair_eval/probe_result.json"
    old_rows = json.loads(fixture.read_text())
    old = [r["policy"] for r in old_rows["rows"][1]["head"]["attempts"]]
    strict, *batch = [row["task"] for row in old_rows["reveal"]["cases"]]
    assert len(batch) == 12 and all(not Bench().apply(t, old)["ok"] for t in batch)
    assert Bench().apply(strict, old)["ok"]
    ordered = []
    for case in old_rows["reveal"]["cases"][1:]:
        task = case["task"]
        assert arena.check(task, case["witness"])
        control = dict(task, items=[task["items"][i] for group in case["witness"] for i in group])
        assert Bench().apply(control, old)["ok"]
        ordered.append(control)
    rows = []
    for seed in (44, 45, 46):
        practice, _ = research.batch_curriculum(seed, batch)
        skills, _, _ = research.experience([], old, practice)
        report = run(practice, old, skills)
        check_costs(practice, old, report)
        program = report["program"]
        assert program is not None
        policies = research.dispatch(program, old)
        replay = [Bench().apply(t, policies) for t in batch]
        strict_result = Bench().apply(strict, policies)
        assert strict_result["ok"] and arena.check(strict, strict_result["packing"])
        assert all(r["ok"] and arena.check(t, r["packing"]) for t, r in zip(batch, replay))
        assert report["comparison"] in [m["prediction"] for m in report["fitted"]]
        for task in ordered:
            result = Bench().apply(task, policies)
            assert result["ok"] and arena.check(task, result["packing"])
        rows.append(dict(seed=seed, calls=report["calls"], practice=report["comparison"],
                         public_gains=sum(r["ok"] for r in replay), strict_retained=strict_result["ok"],
                         same_items_reordered_retained=len(ordered)))
    return rows


if __name__ == "__main__":
    print(json.dumps(dict(primary=check_panel("primary", 3011), fallback=check_panel("fallback", 3012),
                          published=public_replay()), indent=2))
