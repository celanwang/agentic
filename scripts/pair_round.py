"""Replay the 40 known misses, one batch decision, and its independent restart check."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
from contextlib import closing
from tempfile import TemporaryDirectory, mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from candidates.audit import pair_boundary
from candidates.audit.pair_followup import cli
from packlab import arena, research
from packlab.cli import show_packing, show_study


def prepare(root):
    release = json.loads((ROOT / "evidence/pairs/release.json").read_text())
    files = dict(release["runtime"])
    files["evidence/pairs/protocol.md"] = release["protocol"]
    files["candidates/pair_eval/blind_commitment.json"] = release["commitment_file"]
    for name, digest in files.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"本次发布的复跑文件摘要不符  {name}")
    before = root / "baseline/pilot/round-01/stream"
    live = root / "live"
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/restore_demo_states.py"),
                    "--state-dir", str(root / "baseline")], check=True, stdout=subprocess.DEVNULL)
    source = json.loads((ROOT / "examples/pilot-pairs-sealed.json").read_text())["task"]
    last = cli(before, "solve", "-", task=source)
    assert (last["id"], last["decision"], last["generation"], last["memory_source_event"]) == (10, "inconclusive", 1, 9)
    live.mkdir()
    with closing(sqlite3.connect((before / "agentic.sqlite3").as_uri() + "?mode=ro", uri=True)) as source_db:
        with closing(sqlite3.connect(live / "agentic.sqlite3")) as target:
            source_db.backup(target)
    return before, live


def totals(rows):
    return dict(total=len(rows), gains=sum(r["new_ok"] and not r["old_ok"] for r in rows),
                losses=sum(r["old_ok"] and not r["new_ok"] for r in rows),
                **{k: sum(r[k] for r in rows) for k in ("old_ok", "new_ok", "old_cost", "new_cost")})


def conduct(before, live):
    for path in (before, live):
        status = cli(path, "status")
        if (status["generation"], status["events"]) != (1, 10):
            raise ValueError("本轮需要停在第 10 条记录的 v1 副本")
    batch = [task for task, _ in pair_boundary.manifests() if task["bins"] >= 5]
    old = [cli(before, "probe", "-", task=t)["result"] for t in batch]
    assert len(batch) == 40 and all(not row["ok"] for row in old)
    event = cli(live, "research", "--batch", "-", task=batch)
    assert event["id"] == 11 and event["batch_tasks"] == batch and event["episode"] == 11
    assert event["memory_source_event"] == 10 and event["search"]["calls"] <= event["budget"]
    stored = cli(live, "show", "11")
    assert stored == event
    active = cli(live, "status")
    assert active["generation"] == event["generation"]
    known = []
    for task, a in zip(batch, old):
        b = cli(live, "probe", "-", task=task)["result"]
        assert not b["ok"] or arena.check(task, b["packing"])
        known.append(dict(task=task, old_ok=a["ok"], new_ok=b["ok"], old_cost=a["cost"], new_cost=b["cost"]))
    practiced = {research.family_id(t) for t in event["practice_tasks"]}
    excluded = practiced | {research.family_id(t) for t in batch}
    gates, independent = {}, []
    for phase in ("holdout", "regression", "replay"):
        if phase not in event:
            continue
        rows = event[phase]["rows"]
        gates[phase] = totals(rows)
        assert all(event[phase][k] == value for k, value in gates[phase].items())
        if phase != "replay":
            families = {research.family_id(row["task"]) for row in rows}
            assert len(families) == len(rows) and not families & excluded
            excluded |= families
            independent.extend(rows)
    if gates:
        decision, reason = research.judge(gates["holdout"], gates["regression"], gates["replay"])
        if decision == "kept":
            assert event["answer"] and arena.check(batch[0], event["answer"]["packing"])
        assert (decision, reason) == (event["decision"], event["reason"])
    profiles = sorted({row["profile"] for row in independent})
    return dict(record=event["id"], build=event["build"], decision=event["decision"], reason=event["reason"],
                generation=active["generation"], nominee=event["nominee"], digest=event.get("nominee_sha256"),
                practice=event["search"]["comparison"], calls=event["search"]["calls"], budget=event["budget"],
                practice_operations=event["search"]["spent"], preparation_operations=event["discovery_cost"],
                validation_operations=event["validation_cost"], known=totals(known),
                known_outside_practice=totals([r for r in known if research.family_id(r["task"]) not in practiced]),
                official=gates, by_profile={p: totals([r for r in independent if r["profile"] == p]) for p in profiles},
                non_pairs=totals([r for r in independent if r["profile"] != "pairs"]))


def restart(before, live):
    reference = ROOT / "candidates/pair_eval/probe_result.json"
    published = json.loads(reference.read_text())
    out = subprocess.check_output([sys.executable, "-B", "-m", "candidates.pair_eval.probe",
                                   "--state", str(live), "--event", "11", "--build", published["release_build"],
                                   "--before", str(before), "--payload", str(reference)], cwd=ROOT, text=True)
    opened = json.loads(out)
    if opened != published:
        raise RuntimeError("重启盲测与发布时的逐项结果不符")
    first = opened["rows"][0]
    task = opened["reveal"]["cases"][0]["task"]
    print(f"重启后打开首份封存清单  {task['bins']} 个盘，容量 {task['capacity']}。")
    for arm, label in (("old", "原策略"), ("head", "活动策略")):
        answer = first[arm]
        print(f"{label}  {'装完' if answer['ok'] else '未装完'}，操作 {answer['cost']}")
    if first["head"]["ok"]:
        show_packing(task, first["head"]["packing"])
    old, new = (opened["summary"][key] for key in ("old", "head"))
    print(f"全组封存清单  {old['ok']}/{old['cases']} → {new['ok']}/{new['cases']}，"
          f"丢失 {new['losses']}，操作 {old['cost']} → {new['cost']}。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="临时重建两端状态并逐项核对发布证据")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="agentic-pairs-") as temp:
        if args.check:
            root = Path(temp)
        else:
            demos = ROOT / "state/demos"
            demos.mkdir(exist_ok=True, parents=True)
            root = Path(mkdtemp(prefix="pair-round-", dir=str(demos)))
        before, live = prepare(root)
        print("已重建第 10 条记录的 v1。先在新进程逐份检查同一批已知清单，再提交一次批次研究。", flush=True)
        report = conduct(before, live)
        show_study(cli(live, "show", "11"))
        for label, row in (("已知失败", report["known"]), ("其中未抽入练习", report["known_outside_practice"]),
                           ("正式混合任务", report["by_profile"]["mixed"])):
            print(f"{label}  {row['old_ok']}/{row['total']} → {row['new_ok']}/{row['total']}，"
                  f"操作 {row['old_cost']} → {row['new_cost']}")
        expected = json.loads((ROOT / "evidence/pairs/decision.json").read_text())
        if report != expected:
            raise RuntimeError("本轮重放与发布时记录不一致")
        restart(before, live)
        if not args.check:
            print("继续使用本次记忆  " + shlex.join([str(ROOT / "agentic"), "--state", str(live)]))


if __name__ == "__main__":
    main()
