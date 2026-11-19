"""Rebuild v2, run one research episode on its 12 published misses, then restart."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidates.audit.pair_boundary import manifests
from candidates.audit.pair_followup import cli
from packlab import arena, research
from packlab import recipes
from packlab.cli import show_study
from packlab.experiment import Bench
from scripts.pair_round import replay as restore_v2

EVIDENCE = ROOT / "evidence/route-review"
PUBLIC = ROOT / "candidates/pair_eval/probe_result.json"


def known_cases():
    data = json.loads(PUBLIC.read_text())
    tasks = {row["name"]: row["task"] for row in data["reveal"]["cases"]}
    return [tasks[row["name"]] for row in data["rows"] if not row["head"]["ok"]]


def stages(root):
    before = root / "before"
    before.mkdir()
    restore_v2(before)
    prior = before / "live"
    live = root / "live"
    live.mkdir()
    with closing(sqlite3.connect((prior / "agentic.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        with closing(sqlite3.connect(live / "agentic.sqlite3")) as out:
            db.backup(out)
    return prior, live


def validate_gate(event, before):
    old_inputs, old_tests = set(), set()
    for i in range(1, event["id"]):
        e = cli(before, "show", str(i))
        inputs = e.get("practice_tasks", []) + e.get("batch_tasks", []) + ([e["task"]] if e.get("task") else [])
        old_inputs.update(research.family_id(t) for t in inputs)
        old_tests.update(research.family_id(r["task"]) for phase in ("holdout", "regression", "replay")
                         for r in e.get(phase, {}).get("rows", []))
    old_inputs.update(research.family_id(t) for t in event["practice_tasks"] + event["batch_tasks"])
    summary = {}
    for phase in ("holdout", "regression", "replay"):
        if phase not in event:
            continue
        score = event[phase]
        keys = {research.family_id(row["task"]) for row in score["rows"]}
        if phase != "replay":
            assert len(keys) == score["total"] and not keys & old_inputs
            old_inputs.update(keys)
        if phase == "holdout":
            assert not keys & old_tests
        summary[phase] = {k: v for k, v in score.items() if k != "rows"}
    if summary:
        outcome = research.judge(summary["holdout"], summary["regression"], summary["replay"])[0]
        if outcome == "kept" and not event["after"]["ok"]:
            outcome = "inconclusive"
        assert outcome == event["decision"]
    return summary


def comparison(tasks, old, candidate, active):
    rows = []
    for task in tasks:
        bench = Bench()
        row = {}
        for name, chain in (("v2", old), ("candidate", candidate), ("active", active)):
            if chain is None:
                continue
            answer = bench.apply(task, chain)
            assert not answer["ok"] or arena.check(task, answer["packing"])
            row[name] = dict(ok=answer["ok"], cost=answer["cost"])
        rows.append(row)
    return {name: dict(total=len(rows), ok=sum(r[name]["ok"] for r in rows), cost=sum(r[name]["cost"] for r in rows),
                       gains=sum(r[name]["ok"] and not r["v2"]["ok"] for r in rows),
                       losses=sum(not r[name]["ok"] and r["v2"]["ok"] for r in rows)) for name in rows[0]}


def coverage(event):
    """Audit actual branch decisions after the nominee and verdict are stored."""
    def path(task, policy):
        trace = recipes.run(task, policy)["trace"]["route"]
        return tuple(None if trace[step] is None or trace[step]["guard"] is None
                     else trace[step]["guard"]["matched"] for step in ("primary", "fallback"))

    groups = {"practice": event["practice_tasks"], "batch": event["batch_tasks"]}
    for phase in ("holdout", "regression", "replay"):
        if phase in event:
            groups[phase] = [row["task"] for row in event[phase]["rows"]]
    active, nominee = event["policies_before"][0], event["nominee"]
    if nominee is None:
        return {}
    return {name: dict(total=len(tasks), changed_paths=sum(path(t, active) != path(t, nominee) for t in tasks))
            for name, tasks in groups.items()}


def conduct(before, live):
    batch = known_cases()
    status = cli(live, "status")
    if (len(batch), status["generation"], status["events"]) != (12, 2, 11):
        raise ValueError("本轮需要已发布的 12 份失败清单和停在第 11 条记录的 v2")
    old = status["policies"]
    assert old == cli(before, "status")["policies"]
    assert all(not cli(before, "probe", "-", task=t)["result"]["ok"] for t in batch)
    event = cli(live, "research", "--batch", "-", task=batch)
    assert event["id"] == 12 and event["episode"] == 12 and event["search"]["calls"] <= 128
    assert event == cli(live, "show", "12") and event["policies_before"] == old
    assert event["memory_migration"] == "archive_only" and event["memory_source_event"] == 11
    gate = validate_gate(event, before)
    active = cli(live, "status")["policies"]
    candidate = research.dispatch(event["nominee"], old) if event["nominee"] is not None else None
    assert active == (candidate if event["decision"] == "kept" else old)
    before_event = cli(before, "show", "11")
    mixed = [r["task"] for phase in ("holdout", "regression") for r in before_event[phase]["rows"] if r["profile"] == "mixed"]
    panels = dict(published_misses=batch, previous_pairs=[t for t, _ in manifests() if t["bins"] >= 5], previous_mixed=mixed)
    return dict(build=event["build"], generation=event["generation"], record=event["id"], decision=event["decision"],
                reason=event["reason"], program=event["nominee"], nominee_sha256=event.get("nominee_sha256"), gate=gate,
                budget=event["budget"], calls=event["search"]["calls"], spent=event["search"]["spent"],
                practice=event["search"]["comparison"], trigger_delivered=bool(event["answer"]),
                diagnostics={name: comparison(tasks, old, candidate, active) for name, tasks in panels.items()})


def restart(before, live, report):
    reference = ROOT / "candidates/route_review/eval/probe_result.json"
    out = subprocess.check_output([sys.executable, "-B", "-m", "candidates.route_review.eval.probe",
                                   "--before", str(before), "--state", str(live), "--event", "12",
                                   "--build", report["build"], "--payload", str(reference)], cwd=ROOT, text=True)
    opened = json.loads(out)
    if opened != json.loads(reference.read_text()):
        raise RuntimeError("重启后的封存结果与发布记录不同")
    for key, label in (("historical", "发布时的 v2"), ("before", "现执行器的 v2"),
                       ("nominee", "冻结提名"), ("head", "实际活动策略")):
        row = opened["summary"][key]
        print(f"{label}  装完 {row['ok']}/{row['cases']}，操作 {row['cost']}")


def run(root):
    before, live = stages(root)
    report = conduct(before, live)
    if report != json.loads((EVIDENCE / "decision.json").read_text()):
        raise RuntimeError("本次重放的决策或逐组费用与发布时不符")
    stored = cli(live, "show", "12")
    show_study(stored)
    paths = coverage(stored)
    if paths != json.loads((EVIDENCE / "coverage.json").read_text()):
        raise RuntimeError("实际分支覆盖与发布记录不同")
    if "holdout" in paths:
        print(f"普通留出中实际走不同分支的清单  {paths['holdout']['changed_paths']}/{paths['holdout']['total']}")
    for label, group in report["diagnostics"].items():
        print(label, "同一执行器上的两端及提名", json.dumps(group, ensure_ascii=False))
    restart(before, live, report)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
