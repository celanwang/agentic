"""Compare locked planner arms at a common candidate–task budget."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter
from unittest.mock import patch

from . import blind, suite


ARMS = {"adaptive": "adaptive", "uniform": "uniform", "reset": "adaptive"}
BINDING = blind.HERE / "learning_binding.json"


def v2_inputs():
    path = (blind.V2_STATE / "agentic.sqlite3").resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        head, policy_json = db.execute("SELECT r.id,r.policies FROM revisions r JOIN head h ON h.revision=r.id").fetchone()
        assert head == 2
        experience = {}
        rows = db.execute("SELECT json_extract(data,'$.trials') FROM events WHERE kind='solve' AND id<=6 ORDER BY id DESC")
        for raw, in rows:
            for trial in json.loads(raw or "[]"):
                policy = trial["policy"]
                if policy.get("kind") != "recipe":
                    experience.setdefault(suite.canonical(policy), policy)
    return json.loads(policy_json), list(experience.values())


def observe_search(planner, tasks, policies, experience, memory, seed, mode):
    seen_recipes = []
    measured_cost = 0
    measured_ordinary = 0
    run = planner._checked

    def measured(task, policy):
        nonlocal measured_cost, measured_ordinary
        result = run(task, policy)
        assert not result["ok"] or suite.check(task, result["packing"])
        measured_cost += result["cost"]
        if policy.get("kind") == "recipe":
            seen_recipes.append((suite.canonical(task), suite.canonical(policy), result["cost"]))
        else:
            measured_ordinary += 1
        return result

    started = perf_counter()
    with patch.object(planner, "_checked", measured):
        result = planner.search(tasks, policies, experience, memory, seed, suite.CALL_BUDGET, mode=mode)
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    assert result["budget"] == suite.CALL_BUDGET and result["mode"] == mode
    assert result["calls"] == result["baseline_calls"] + result["candidate_calls"] <= suite.CALL_BUDGET
    assert result["baseline_calls"] == len(tasks) == len(result["baseline"])
    assert result["candidate_calls"] == len(result["trials"]) == len(seen_recipes)
    assert result["spent"] == result["baseline_spent"] + result["candidate_spent"] == measured_cost
    assert result["candidate_spent"] == sum(cost for _, _, cost in seen_recipes)
    assert len({(task, recipe) for task, recipe, _ in seen_recipes}) == len(seen_recipes)
    for (task_key, recipe_key, cost), counted in zip(seen_recipes, result["trials"]):
        assert task_key == suite.canonical(tasks[counted["task_index"]])
        assert recipe_key == suite.canonical(result["candidates"][counted["candidate"]]["program"])
        assert cost == counted["spent"]
    details = {k: result[k] for k in ("calls", "candidate_calls", "baseline_calls", "spent", "candidate_spent",
                                      "baseline_spent", "cached_pairs", "stop_reason", "program")}
    details.update(candidate_pool=len(result["candidates"]), physical_ordinary_calls=measured_ordinary,
                   observation_sha256=blind.digest(suite.canonical(result["trials"])), elapsed_ms=elapsed_ms)
    return details, json.loads(json.dumps(result["memory"]))


def checkpoint(groups, policies, program, bench, dispatch):
    route = dispatch(program, policies) if program is not None else policies
    rows = []
    for cluster in groups:
        for case in cluster["cases"]:
            task = case["task"]
            old, new = bench.apply(task, policies), bench.apply(task, route)
            assert not old["ok"] or suite.check(task, old["packing"])
            assert not new["ok"] or suite.check(task, new["packing"])
            rows.append({"cluster": cluster["id"], "profile": cluster["profile"], "variant": case["variant"],
                         "old_ok": old["ok"], "ok": new["ok"], "old_cost": old["cost"], "cost": new["cost"]})
    return rows


def summarize(rows):
    counts = Counter(row["cluster"] for row in rows if row["ok"])
    gains = sum(row["ok"] and not row["old_ok"] for row in rows)
    losses = sum(row["old_ok"] and not row["ok"] for row in rows)
    return dict(tasks=len(rows), old_ok=sum(row["old_ok"] for row in rows), ok=sum(row["ok"] for row in rows),
                gains=gains, losses=losses, net=gains-losses,
                old_cost=sum(row["old_cost"] for row in rows), cost=sum(row["cost"] for row in rows),
                all_variants=sum(solved == len(suite.VARIANTS) for solved in counts.values()))


def primary(seeds):
    matched = [r for seed in seeds for r in seed["rounds"] if r["matched"]]
    totals = {}
    for arm in ARMS:
        parts = [r["arms"][arm] for r in matched]
        totals[arm] = dict(summarize([row for part in parts for row in part["checkpoint"]]),
                           calls=sum(part["calls"] for part in parts),
                           candidate_calls=sum(part["candidate_calls"] for part in parts),
                           spent=sum(part["spent"] for part in parts),
                           nominated=sum(part["program"] is not None for part in parts))
    paired = {}
    for control in ("uniform", "reset"):
        deltas = []
        for episode in seeds:
            rounds = [r for r in episode["rounds"] if r["matched"]]
            if not rounds:
                continue
            diff = sum(r["arms"]["adaptive"]["summary"]["net"] - r["arms"][control]["summary"]["net"] for r in rounds)
            deltas.append({"seed_index": episode["index"], "rounds": len(rounds), "delta": diff})
        values = [entry["delta"] for entry in deltas]
        paired[control] = {"per_seed": deltas, "median": median(values) if values else None,
                           "wins": sum(d > 0 for d in values), "ties": sum(d == 0 for d in values),
                           "losses": sum(d < 0 for d in values)}
    return {"matched_rounds": len(matched), "all_rounds": len(seeds) * suite.ROUNDS, "totals": totals, "paired": paired}


def run(expected_build):
    from packlab import memory, planner, research
    from packlab.experiment import Bench

    assert memory.build_id() == expected_build
    blind.check_official_bound()
    binding = json.loads(BINDING.read_text())
    policies, experience = v2_inputs()
    assert (policies, experience) == (binding["policies"], binding["experience"])
    assert binding["protocol_sha256"] == blind.digest(blind.PROTOCOL.read_bytes())
    assert binding["runner_sha256"] == blind.digest(Path(__file__).read_bytes())
    seeds = suite.learning()
    result = []
    for lesson in seeds:
        memories = {arm: None for arm in ARMS}
        rounds = []
        for row in lesson["rounds"]:
            tasks = [entry["task"] for entry in row["train"]]
            bench = Bench()
            seed = int.from_bytes(suite.sha256(suite.canonical([lesson["seed"], row["round"]])).digest()[:8], "big")
            arms = {}
            for arm, mode in ARMS.items():
                details, updated = observe_search(planner, tasks, policies, experience, memories[arm], seed, mode)
                memories[arm] = updated if arm != "reset" else None
                values = checkpoint(row["checkpoint"], policies, details["program"], bench, research.dispatch)
                arms[arm] = dict(details, checkpoint=values, summary=summarize(values))
            counts = {(item["baseline_calls"], item["candidate_calls"]) for item in arms.values()}
            rounds.append({"round": row["round"], "focus": row["focus"], "planner_seed": seed,
                           "matched": len(counts) == 1, "arms": arms})
        result.append({"index": lesson["index"], "seed": lesson["seed"], "rounds": rounds})
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == expected_build
    return {"build": expected_build, "protocol": suite.VERSION, "experience": experience, "v2_policies": policies,
            "binding_sha256": blind.digest(BINDING.read_bytes()), "primary": primary(result), "seeds": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build")
    args = parser.parse_args()
    result = run(args.build)
    target = blind.HERE / "learning_result.json"
    blind.write_once(target, result)
    print(json.dumps(result["primary"], indent=2))


if __name__ == "__main__":
    main()
