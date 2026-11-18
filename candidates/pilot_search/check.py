"""Reproduce a two-step guard and a memory intervention on local fixtures."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from packlab import arena, engine, recipes


def load_planner():
    spec = importlib.util.spec_from_file_location("packlab._candidate_planner", HERE / "planner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLANNER = load_planner()
INCUMBENT = [engine.BASE, dict(engine.BASE, repair="triples"), dict(engine.BASE, repair="fill")]
STEPS = ["peel_forced", "normalize_gcd"]


def task(capacity, bins, items):
    return dict(capacity=capacity, bins=bins, items=items)


def learning_panel(first_factor):
    # Scale known four-disk instances and add a mandatory disk holding cap-1.
    # In the original view the singleton hides the common unit of other files.
    sizes = [3, 8, 12, 41, 5, 33, 16, 17, 5, 13, 26, 3, 22, 8, 12]
    return [task(56 * factor, 5, [s * factor for s in sizes[:7]] + [56 * factor - 1]
                 + [s * factor for s in sizes[7:]]) for factor in range(first_factor, first_factor + 6)]


def first_gain(result):
    return next((i for i, row in enumerate(result["trials"], 1) if row["ok"] and not row["old_ok"]), None)


def check_ledger(result):
    trials, baseline = result["trials"], result["baseline"]
    assert result["calls"] == len(trials) + len(baseline) <= result["budget"]
    assert result["baseline_calls"] == len(baseline) and result["candidate_calls"] == len(trials)
    assert result["candidate_spent"] == sum(row["spent"] for row in trials)
    assert result["baseline_spent"] == sum(row["cost"] for row in baseline)
    assert result["spent"] == result["candidate_spent"] + result["baseline_spent"]
    assert len({(row["candidate"], row["task_index"]) for row in trials}) == len(trials)
    for row in trials:
        node = result["candidates"][row["candidate"]]
        assert node["parent"] == row["parent"] and node["direction"] == row["direction"]
        assert row["reason"] and node["program"]["guard"] == row["guard"]


def check_replay(result, panel, installed=None):
    # Independent accounting through engine.run and recipes.run, replayed only for the smoke test.
    for note in result["baseline"]:
        problem = panel[note["task_index"]]
        if note["role"] == "active-recipe":
            observed = recipes.run(problem, installed)
            assert note["ok"] == observed["ok"] and note["cost"] == observed["cost"]
            continue
        cost = 0
        for policy in INCUMBENT:
            old = engine.run(problem, policy)
            cost += old["cost"]
            reason = (old["trace"]["blocked"] or {}).get("reason")
            if old["ok"] or reason in ("oversize", "total_excess") or old["trace"]["search_status"] == "infeasible":
                break
        assert note["ok"] == old["ok"] and note["cost"] == cost

    for row in result["trials"]:
        program = result["candidates"][row["candidate"]]["program"]
        problem = panel[row["task_index"]]
        actual = recipes.run(problem, program)
        assert actual["ok"] == row["raw_ok"] and actual["cost"] == row["spent"]
        if actual["ok"]:
            assert arena.check(problem, actual["packing"])


def study():
    child = subprocess.run([sys.executable, "-B", str(HERE / "check.py"), "learn"],
                           cwd=ROOT, check=True, text=True, capture_output=True, timeout=20)
    learned = json.loads(child.stdout)
    assert learned["program"]["steps"] == STEPS and learned["calls"] == 128
    memory = learned["memory"]
    immutable = deepcopy(memory)
    later = learning_panel(250)
    warm = PLANNER.search(later, INCUMBENT, [], memory, 100, 30)
    reset = PLANNER.search(later, INCUMBENT, [], memory, 100, 30, "adaptive_reset")
    uniform = PLANNER.search(later, INCUMBENT, [], memory, 100, 30, "uniform")
    assert memory == immutable and warm == PLANNER.search(later, INCUMBENT, [], memory, 100, 30)
    assert warm["calls"] == reset["calls"] == uniform["calls"] == 30
    assert warm["program"]["steps"] == STEPS and first_gain(warm) is not None
    assert reset["program"] is None and first_gain(reset) is None
    assert warm["trials"][0]["candidate"] != reset["trials"][0]["candidate"]

    mixed = later[:3] + [task(17, 2, [3, 6, 7]), task(32, 3, [6, 8, 15]), task(42, 4, [8, 17, 21])]
    guarded = PLANNER.search(mixed, INCUMBENT, [], memory, 100, 128)
    program = guarded["program"]
    assert program["steps"] == STEPS and program["guard"] is not None and guarded["calls"] == 128
    for index, problem in enumerate(mixed):
        result = recipes.run(problem, program)
        if index < 3:
            assert result["ok"] and arena.check(problem, result["packing"])
        else:
            assert result["trace"]["search_status"] == "guard_skip"
    replaced = PLANNER.search(mixed, [program, *INCUMBENT], [], None, 100, 30)
    assert replaced["baseline_calls"] == 12 and replaced["candidate_calls"] == 18
    assert replaced["nominee"] is None
    repeated = PLANNER.search(later, INCUMBENT, [], warm["memory"], 100, 30)
    assert repeated["cached_pairs"] > 0
    for result in (learned, warm, reset, uniform, guarded, replaced, repeated):
        check_ledger(result)
    check_replay(guarded, mixed)
    check_replay(replaced, mixed, program)
    for mode in ("adaptive", "uniform"):
        for budget in (0, 1, 2, 9):
            check_ledger(PLANNER.search(mixed, INCUMBENT, [], None, 101, budget, mode))

    return {"training_calls": learned["calls"], "intervention_calls_each": warm["calls"],
            "first_gain_candidate_call": {"persistent": first_gain(warm), "reset": first_gain(reset),
                                          "uniform": first_gain(uniform)},
            "guarded_program": program, "guarded_practice": guarded["comparisons"][guarded["nominee"]],
            "repeated_reused_pairs": repeated["cached_pairs"],
            "active_recipe_baseline_calls": replaced["baseline_calls"]}


if __name__ == "__main__":
    if sys.argv[1:] == ["learn"]:
        print(json.dumps(PLANNER.search(learning_panel(160), INCUMBENT, [], None, 0, 128)))
    else:
        print(json.dumps(study(), ensure_ascii=False, indent=2))
