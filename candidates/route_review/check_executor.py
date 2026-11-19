"""Compare route accounting to 954f605 on the previously published #11 data."""
import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from random import Random
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packlab import arena, engine, recipes
from packlab.experiment import Bench, proven_impossible
from scripts.pair_round import ARCHIVE, ARCHIVE_SHA256, replay

FROZEN = "954f605"
PUBLISHED = ROOT / "candidates/pair_eval/probe_result.json"


def frozen_recipes():
    data = ARCHIVE.read_bytes()
    if sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("第 11 轮源码归档摘要不符")
    with ZipFile(BytesIO(data)) as archive:
        source = archive.read("packlab/recipes.py")
    module = ModuleType("packlab._frozen_recipes")
    module.__package__ = "packlab"
    exec(compile(source, FROZEN + "/recipes.py", "exec"), module.__dict__)
    return module


def apply(task, policies, executor):
    cost = 0
    for program in policies:
        runner = executor if program.get("kind") == "recipe" else engine
        result = runner.run(task, program)
        cost += result["cost"]
        if result["ok"] or proven_impossible(result):
            break
    assert not result["ok"] or arena.check(task, result["packing"])
    return result["ok"], cost


def fingerprints(result):
    """Remove operation counters; every other observable result must agree."""
    trace = dict(result["trace"])
    trace.pop("ops")
    if "route" in trace:
        trace["route"] = {
            name: {key: x for key, x in value.items() if key != "cost"}
            if isinstance(value, dict) else value
            for name, value in trace["route"].items()
        }
    return dict(ok=result["ok"], packing=result["packing"], trace=trace)


def compare_route(task, program, frozen):
    before = frozen.run(task, program)
    after = recipes.run(task, program)
    assert fingerprints(before) == fingerprints(after)
    selected = after["trace"]["route"]["selected"]
    if selected == "fallback":
        primary_skipped = before["trace"]["route"]["primary"]["search_status"] == "guard_skip"
        old_setup = before["trace"]["ops"]["recipe_setup"]
        assert after["trace"]["ops"]["recipe_setup"] == old_setup - 3 - len(task["items"])
        delta = 3 + len(task["items"]) + (task["bins"] if primary_skipped else 0)
        assert before["cost"] - after["cost"] == delta
    else:
        assert after == before
    assert after["cost"] == sum(after["trace"]["ops"].values())
    return before, after


def check_executor(frozen):
    rng = Random(954605)
    legacy = routes = 0
    coverage = dict(feature="complementary_coverage", op="ge", value=1000)
    bins = dict(feature="bins", op="ge", value=1)
    steps = ([], ["normalize_gcd"], ["peel_forced"], list(recipes.STEPS))
    guarded = (None, coverage, dict(bins, op="le", value=2))
    for _ in range(160):
        task = dict(capacity=rng.randint(1, 50), bins=rng.randint(1, 8),
                    items=[rng.randint(1, 51) for _ in range(rng.randint(0, 18))])
        assert list(recipes.features(task).items()) == list(frozen.features(task).items())
        a, b = {}, {}
        assert recipes.route_features(task, ops=a) == frozen.route_features(task, ops=b) and a == b
        for prep, guard in product(steps, guarded):
            program = dict(kind="recipe", steps=prep, solver=engine.BASE, guard=guard)
            assert recipes.run(task, program) == frozen.run(task, program)
            legacy += 1
        fallback = dict(kind="recipe", steps=["peel_forced"], solver=dict(engine.BASE, order="descending"),
                        guard=dict(bins, op="le", value=3))
        for guard in (coverage, bins):
            route = dict(kind="recipe", steps=[], solver=engine.BASE, guard=guard, fallback=fallback)
            compare_route(task, route, frozen)
            routes += 1
    print(f"{legacy} four-field and {routes} routed cases retain identical behavior")


def mixed_rows(folder):
    statement = """SELECT j.value FROM events, json_each(events.data, ?) AS j
                   WHERE events.id=11 AND json_extract(j.value, '$.profile')='mixed'
                   ORDER BY j.key"""
    store = (folder / "agentic.sqlite3").resolve()
    with closing(sqlite3.connect(store.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        return [json.loads(row[0]) for split in ("holdout", "regression")
                for row in db.execute(statement, (f"$.{split}.rows",))]


def audit_cases(frozen, folder):
    public = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    program = public["event"]["nominee"]
    old = public["event"]["policies_before"]
    chain = [program] + [policy for policy in old if policy.get("kind") != "recipe"]
    new_public = {case["name"]: case["task"] for case in public["reveal"]["cases"]}
    failures = [dict(row, task=new_public[row["name"]], old_cost=row["old"]["cost"],
                     new_cost=row["head"]["cost"], new_ok=row["head"]["ok"])
                for row in public["rows"] if not row["head"]["ok"]]
    mixed = mixed_rows(folder)
    assert len(failures) == len(mixed) == 12
    desc = dict(engine.BASE, order="descending")
    for label, cases in (("published failures", failures), ("published #11 mixed", mixed)):
        sums = Counter()
        coverages = []
        for row in cases:
            task = row["task"]
            before, after = compare_route(task, program, frozen)
            trace = before["trace"]
            assert trace["guard"]["matched"] is False
            original_ok, original_cost = apply(task, old, frozen)
            saved_ok, saved_cost = apply(task, chain, frozen)
            actual = Bench().apply(task, chain)
            assert (original_cost, saved_cost, saved_ok) == (row["old_cost"], row["new_cost"], row["new_ok"])
            assert actual["ok"] == saved_ok and actual["cost"] == saved_cost - before["cost"] + after["cost"]
            primary, fallback = trace["route"]["primary"], trace["route"]["fallback"]
            leaf = frozen.run(task, {k: v for k, v in program.items() if k != "fallback"})
            coverage_cost = leaf["trace"]["ops"]["recipe_complements"]
            coverages.append(trace["guard"]["observed"])
            simple = engine.run(task, desc)
            assert not simple["ok"] or arena.check(task, simple["packing"])
            sums.update(old=original_cost, old_ok=int(original_ok), before=saved_cost, after=actual["cost"],
                        guard_observation=coverage_cost, guard_other=primary["cost"]-coverage_cost,
                        fallback=fallback["cost"], external=saved_cost-before["cost"],
                        dispatch=1, descending=simple["cost"], descending_ok=int(simple["ok"]))
        assert sums["guard_observation"]+sums["guard_other"]+sums["fallback"]+sums["external"]+sums["dispatch"] == sums["before"]
        print(label, (min(coverages), max(coverages)), json.dumps(dict(sums), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="包含已发布 #11 的状态目录，省略则在临时目录重建")
    args = parser.parse_args()
    baseline = frozen_recipes()
    check_executor(baseline)
    if args.state is not None:
        audit_cases(baseline, args.state)
    else:
        with TemporaryDirectory(prefix="agentic-check-route-") as temp:
            root = Path(temp)
            replay(root)
            audit_cases(baseline, root / "live")


if __name__ == "__main__":
    main()
