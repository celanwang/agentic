"""Give the released head its missed eight-disk task once, then probe seventy new tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from candidates.audit import pair_boundary
from packlab import arena, engine, research
from packlab.experiment import Bench


def cli(folder, *args, task=None):
    data = json.dumps(task) if task is not None else None
    out = subprocess.check_output([str(ROOT / "agentic"), "--state", str(folder), *args, "--json"],
                                  input=data, text=True)
    return json.loads(out)


def run(state_dir):
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/restore_demo_states.py"), "--state-dir", str(state_dir)],
                   stdout=subprocess.DEVNULL, check=True)
    state = state_dir / "pilot/round-01/stream"
    source = json.loads((ROOT / "examples/pilot-pairs-sealed.json").read_text())
    task = source["task"]
    prior = cli(state, "probe", "-", task=task)["result"]
    assert not prior["ok"] and arena.check(task, source["packing"])
    event = cli(state, "solve", "-", task=task)
    assert event["kind"] == "research" and event["id"] == 10 and event["episode"] == 10
    assert not event["answer"] or arena.check(task, event["answer"]["packing"])
    records = [cli(state, "show", str(i)) for i in range(1, event["id"] + 1)]
    observed = set()
    for record in records:
        inputs = list(record.get("practice_tasks", []))
        if record.get("task"):
            inputs.append(record["task"])
        for phase in ("holdout", "regression"):
            inputs.extend(row["task"] for row in record.get(phase, {}).get("rows", []))
        observed.update(research.family(t) for t in inputs)
    rows = pair_boundary.compare(state)["rows"]
    for row in rows:
        t = row["task"]
        assert research.family(t) not in observed
        result = cli(state, "probe", "-", task=t)["result"]
        assert not result["ok"] or pair_boundary.valid(t, result["packing"])
        row["arms"]["deployed"] = {"ok": result["ok"], "cost": result["cost"]}
    assert cli(state, "status")["events"] == 10
    arms = {name: dict(ok=sum(r["arms"][name]["ok"] for r in rows), cost=sum(r["arms"][name]["cost"] for r in rows))
            for name in rows[0]["arms"]}
    # These specified interventions are measured after the decision, on practice only.
    policies = event["policies_before"]
    unguarded = dict(policies[0], guard=None)
    descending = dict(unguarded, steps=[], solver=dict(engine.BASE, order="descending"))
    practice = [(t, "practice", None) for t in event["practice_tasks"]]
    alternatives = {}
    for name, program in (("remove_guard", unguarded), ("use_descending_recipe", descending)):
        score = research.compare(Bench(), practice, policies, research.dispatch(program, policies))
        alternatives[name] = {k: score[k] for k in ("total", "gains", "losses", "old_cost", "new_cost")}
    return dict(seed=pair_boundary.SEED, cases=len(rows), input_bins=task["bins"], before_cost=prior["cost"],
                decision=event["decision"], reason=event["reason"], nominee=event["nominee"], record=event["id"],
                starting_generation=event["generation_before"], active_generation=event["generation"],
                budget=event["budget"], calls=event["search"]["calls"], history_from=event["memory_source_event"],
                gate={k: {f: event[k][f] for f in ("total","gains","losses","old_cost","new_cost")}
                      for k in ("holdout","regression","replay") if k in event},
                trigger_answer=event["answer"], practice_delivery=event.get("practice_delivery"),
                measured_finalists=event["search"]["comparisons"], posthoc_practice=alternatives,
                probe_cost_and_success=arms, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="include each input and its paired results")
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        report = run(Path(temp))
    if not args.details:
        del report["rows"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
