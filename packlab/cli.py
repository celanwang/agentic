import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from . import arena, engine, memory, recipes, research
from .experiment import Bench, RULES, task_key, validate_task

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT = PROJECT / "state" / "live"
DECISIONS = {"kept": "保留", "unchanged": "沿用", "inconclusive": "证据不足",
             "rejected": "放弃", "budget": "预算用尽", "rollback": "已回退"}


def js(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def policy_name(policy):
    if policy.get("kind") == "recipe":
        return recipes.describe(policy)
    return engine.describe(policy)


def read_task(path):
    if path == "-":
        raw = sys.stdin.read(65537)
    else:
        with open(path, encoding="utf-8") as stream:
            raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("清单超过 64 KiB")
    return validate_task(json.loads(raw))


def research_record(folder, record_id):
    row = memory.event(folder, record_id)
    if row["kind"] != "research" or row["nominee"] is None:
        raise ValueError(f"记录 #{record_id} 没有冻结的研究配方")
    return row


def demo_pilot(base):
    parent = PROJECT / "state" / "demos" if base == DEFAULT else base / "demos"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="pilot-", dir=str(parent))).resolve()
    print(f"空白研究现场  {folder}", flush=True)
    command = [str(PROJECT / "agentic"), "--state", str(folder)]
    example = str(PROJECT / "examples" / "restart.json")
    print("用开发者初始策略试一份公开的配对清单。", flush=True)
    subprocess.run(command + ["probe", example, "--initial"], check=True)
    subprocess.run(command + ["research"], check=True)
    print("\n研究进程已退出。用新进程读取活动策略，复查配对清单。", flush=True)
    subprocess.run(command + ["status"], check=True)
    subprocess.run(command + ["probe", example], check=True)
    print("再用含独占文件的公开难例测试能力边界。", flush=True)
    subprocess.run(command + ["probe", str(PROJECT / "examples" / "pilot-transfer.json")], check=True)
    print(f"查看试验依据  {shlex.join(command)} show 1 --json")
    return 0


def show_solution(task, result):
    label = "成功" if result["ok"] else "未装完"
    print(f"  {label}，操作成本 {result['cost']}")
    show_packing(task, result["packing"] or [])


def show_packing(task, bins):
    for i, contents in enumerate(bins):
        used = sum(task["items"][idx] for idx in contents)
        entries = ' '.join(f"#{idx + 1}({task['items'][idx]})" for idx in contents) or '空'
        print(f"    盘 {i + 1}   {used}/{task['capacity']}   {entries}")


def show_study(data):
    search = data["search"]
    print(f"研究 #{data['id']} · 第 {data['episode']} 轮 · v{data['generation_before']} → v{data['generation']}")
    print(f"练习 {len(data['practice_tasks'])} 个输入；可调用 {len(data['skill_sources'])} 种历史或现场发现的技能。")
    print(f"实验预算 {search['calls']}/{data['budget']} 次，实测操作 {search['spent']}。")
    trials = search.get("trials", [])
    print(f"实际试了 {len({t['candidate'] for t in trials})} 份配方；{len(search['comparisons'])} 份完成了整组练习。")
    if data["nominee"]:
        print("冻结提名  " + policy_name(data["nominee"]))
        choice = search["nominee"]
        selected = next((t for t in trials if t["candidate"] == choice), None)
        if selected:
            print("  " + selected["reason"])
        score = search["comparisons"][choice]
        print(f"  练习新增 {score['gains']}，丢失 {score['losses']}，两端操作 {score['old_cost']} → {score['new_cost']}。")
    for key, title in (("holdout", "普通留出"), ("regression", "回归"), ("replay", "历史复查")):
        if key in data:
            row = data[key]
            print(f"  {title}  {row['old_ok']}/{row['total']} → {row['new_ok']}/{row['total']}，"
                  f"丢失 {row['losses']}，操作 {row['old_cost']} → {row['new_cost']}")
    print(f"决定  {DECISIONS[data['decision']]}。{data['reason']}")
    print(f"准备操作 {data['discovery_cost']}，验收操作 {data['validation_cost']}，本轮用时 {data['experiment_ms']:.1f} ms。")
    if data.get("task"):
        if data["answer"]:
            print("触发清单的可交付装法")
            show_packing(data["task"], data["answer"]["packing"])
        else:
            print("本轮未找到触发清单的完整装法。")


def show_witness(data):
    print(data["finding"])
    witness = data["witness"]
    if witness:
        labels = {"without_peel_forced": "省去独占盘剥离", "without_normalize_gcd": "省去单位归约",
                  "reverse_steps": "颠倒步骤顺序", "without_guard": "移除使用条件", "previous_chain": "恢复原策略链"}
        reduced = witness["reduced"]
        task = reduced["task"]
        print(f"对照动作  {labels[witness['control']]}。练习输入序号 {witness['task_index'] + 1}。")
        print(f"局部删项后  {task['bins']} 个容量为 {task['capacity']} 的盘，文件 {task['items']}。")
        for key, name in (("full", "完整配方"), ("control", "对照")):
            endpoint = reduced[key]
            label = {"solved": "装完", "infeasible": "证实无解", "unresolved": "未解"}[endpoint["outcome"]]
            print(f"  {name}  {label}，{endpoint['cost']} 个操作。")
    print(f"已用 {data['runs_used']}/{data['budget']} 次执行、{data['executed_ops']} 个操作。"
          f"剩余 {data['unchecked_input_pairs']} 个输入与对照的组合未经检查。")


def show_round(data):
    if data.get("kind") == "research":
        return show_study(data)
    print(f"记录 #{data['id']}  ·  策略 v{data['generation_before']} → v{data['generation']}")
    if data.get("kind") == "rollback":
        print(data["reason"])
        return
    task = data["task"]
    print(f"清单  {task['bins']} 个容量为 {task['capacity']} 的盘，文件大小 {task['items']}")
    print("当前策略的首次作答")
    show_solution(task, data["before"])
    for attempt in data["before"]["attempts"]:
        print(f"  {'✓' if attempt['ok'] else '×'}  {policy_name(attempt['policy'])}  ({attempt['cost']})")
    if "observation" in data:
        print(f"失败诊断  {data['observation']}")
        fixed = sum(t["trigger_ok"] for t in data["trials"])
        print(f"尝试  {len(data['trials'])} 个候选，其中 {fixed} 个装下了这份清单。")
        for trial in data["trials"]:
            score = trial.get("practice")
            verdict = (f"练习多解 {score['gains']} 个，额外 {score['new_cost'] - score['old_cost']} 次操作"
                       if score else "未装下触发清单")
            print(f"  · {policy_name(trial['policy'])}   {verdict}")
    if "nominee" in data:
        print("按练习组的新增成功数 / 额外操作数提名  " + policy_name(data["nominee"]))
        for field, label in (("holdout", "独立留出"), ("regression", "回归复查")):
            score = data[field]
            feasible = f"（其中 {score['feasible']} 份有解）" if "feasible" in score else ""
            print(f"{label}  {score['old_ok']}/{score['total']} → {score['new_ok']}/{score['total']}{feasible}，"
                  f"丢失 {score['losses']}，总操作 {score['old_cost']} → {score['new_cost']}")
        h, r = data["holdout"], data["regression"]
        ratio = (h["new_cost"] + r["new_cost"]) / (h["old_cost"] + r["old_cost"])
        print(f"合计操作倍率  {ratio:.2f}×，本轮上限 {data['rules']['max_cost_ratio']}×。")
    print(f"决定  {DECISIONS.get(data['decision'], data['decision'])}。{data['reason']}")
    if data["decision"] == "kept":
        print("已保存策略的装法")
        show_solution(task, data["after"])
    elif data["answer"] and not data["after"]["ok"]:
        label = "本轮试验找到的装法" if data["answer"]["source"] == "trial" else "从这份清单的历史记录恢复的装法"
        print(label)
        show_packing(task, data["answer"]["packing"])
    if data["trials"]:
        print(f"整轮开销  实际执行求解器 {data['solver_calls']} 次，计 {data['executed_cost']} 个操作单位，"
              f"本机用时 {data['experiment_ms']:.1f} ms。")


def demo_followup(child, history, topic):
    name = {"triples": "三件配平", "mixed": "混合大小"}[topic]
    print(f"\n用历史记录中第一份未解的{name}清单触发新一轮", flush=True)
    learned = child("solve", PROJECT / "examples" / f"{topic}-trigger.json")
    show_round(learned)
    transfer_path = PROJECT / "examples" / f"{topic}-restart.json"
    transfer = read_task(transfer_path)
    seen = set()
    for event in [*history, learned]:
        seen.add(task_key(event["task"]))
        seen.update(task_key(t) for t in event.get("practice_tasks", []))
        for phase in ("holdout", "regression"):
            seen.update(task_key(row["task"]) for row in event.get(phase, {}).get("rows", []))
    if task_key(transfer) in seen:
        raise RuntimeError("重启清单与实验样本重合")
    old = Bench().apply(transfer, learned["policies_before"])
    print("\n学习进程已退出。在封存的清单上比较学习前和重启后的策略", flush=True)
    print(f"原 v{learned['generation_before']} 的作答")
    show_solution(transfer, old)
    # A probe cannot learn or read an answer from the task history.
    new = child("probe", transfer_path)["result"]
    print(f"重启后 v{learned['generation']} 的作答")
    show_solution(transfer, new)
    if new["ok"] and not arena.check(transfer, new["packing"]):
        raise RuntimeError("重启进程返回了无效装法")
    if not old["ok"] and new["ok"]:
        print("迁移通过。旧策略失败，新进程只读策略即装完全部文件。")
    else:
        print("本次重启对照未观察到新增成功。")
    return learned


def demo(base, triples=False, mixed=False):
    root = base / "demos" if base != DEFAULT else PROJECT / "state" / "demos"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=str(root))).resolve()
    program = str(PROJECT / "agentic")
    def child(*args):
        out = subprocess.run([program, "--state", str(folder), *map(str, args), "--json"],
                             capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(out.stderr.strip() or "子进程未成功退出")
        return json.loads(out.stdout)

    print("Agentic · 归档分盘实验", flush=True)
    print(f"本次空白状态  {folder}", flush=True)
    print(f"预设门槛  留出组至少多解 {RULES['min_holdout_gains']} 例，"
          f"回归零丢失，两组总操作 ≤ 原来的 {RULES['max_cost_ratio']} 倍。", flush=True)
    first_path = PROJECT / "examples" / "first.json"
    next_path = PROJECT / "examples" / "restart.json"
    print("\n一  第一次遇到碎片化", flush=True)
    first = child("solve", first_path)
    show_round(first)
    if first["before"]["ok"] or first["decision"] != "kept":
        raise RuntimeError("本次首轮未达到演示验收条件，已保留现场")
    other = read_task(next_path)
    samples = [first["task"]] + first["practice_tasks"]
    for phase in ("holdout", "regression"):
        samples.extend(row["task"] for row in first[phase]["rows"])
    if task_key(other) in {task_key(t) for t in samples}:
        raise RuntimeError("重启清单与第一轮实验同源，不能用作迁移证据")
    print("\n二  首个进程已退出。用开发者基线单独试新清单", flush=True)
    cold = child("probe", next_path, "--initial")
    show_solution(other, cold["result"])
    print("\n三  新进程从磁盘读回策略，处理同一份新清单", flush=True)
    second = child("solve", next_path)
    show_round(second)
    if (cold["result"]["ok"] or not second["before"]["ok"] or second["decision"] != "unchanged"
        or second["generation_before"] != first["generation"] or second["trials"]
        or not arena.check(other, second["before"]["packing"])):
        raise RuntimeError("重启迁移未通过，已保留现场")
    print("\n重启验收通过。新清单首次作答成功，本次新尝试 0 个。")
    print("\n四  再给它一份已经按大小排好的难例", flush=True)
    third = child("solve", PROJECT / "examples" / "next.json")
    show_round(third)
    if (third["before"]["ok"] or not third["trials"] or third["decision"] != "rejected"
        or third["generation"] != second["generation"] or not third["answer"]
        or not arena.check(third["task"], third["answer"]["packing"])):
        raise RuntimeError("第二轮的成本取舍未复现，已保留现场")
    if triples or mixed:
        history = [first, second, third]
        history.append(demo_followup(child, history, "triples"))
        if mixed:
            prior_transfer = child("solve", PROJECT / "examples" / "triples-restart.json")
            if prior_transfer["decision"] != "unchanged" or not prior_transfer["before"]["ok"]:
                raise RuntimeError("混合清单追试所需的 v2 起点未复现")
            history.append(prior_transfer)
            demo_followup(child, history, "mixed")
    cmd = f"{shlex.quote(program)} --state {shlex.quote(str(folder))}"
    print(f"看完整证据  {cmd} show 1 --json")
    print(f"用这份记忆启动交互窗口  {cmd}")
    return folder


def panel(folder):
    start_folder = folder
    print("Agentic · 本地归档分盘")
    print(f"记忆目录  {folder}")
    print("输入 容量 盘数 文件大小…，例如 10 2 4 4 6 6。research 启动配方研究，demo 演示，history 看记录，q 退出。")
    while True:
        try:
            line = input("agentic › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in ("q", "quit", "exit"):
            return 0
        if not line:
            continue
        try:
            if line == "demo":
                folder = demo(start_folder)
                print(f"现在可以直接输入下一份清单。记忆目录  {folder}")
            elif line == "history":
                show_history(memory.history(folder, 12))
            elif line == "research":
                show_round(memory.investigate(folder))
            else:
                vals = [int(x) for x in line.replace(",", " ").replace("，", " ").split()]
                if len(vals) < 3:
                    raise ValueError("需要容量、盘数和至少一个文件大小")
                task = validate_task(dict(capacity=vals[0], bins=vals[1], items=vals[2:]))
                show_round(memory.solve(folder, task))
        except (ValueError, OSError, sqlite3.Error, RuntimeError) as error:
            print(f"无法执行  {error}", file=sys.stderr)


def show_history(rows):
    for row in rows:
        print(f"#{row['id']}  {row['created']}  v{row['start_revision']}  "
              f"{DECISIONS[row['decision']]}  {row['reason']}")
    if not rows:
        print("目前没有记录。")


def main():
    parser = argparse.ArgumentParser(prog="agentic", description="能从失败中学习的本地归档分盘 Agent")
    parser.add_argument("--state", type=Path, default=DEFAULT, help="记忆目录")
    subs = parser.add_subparsers(dest="cmd")
    for action, label in (("solve", "装盘，失败时自主试验"), ("probe", "只执行策略一次"),
                          ("show", "读取一次完整记录"), ("status", "当前策略"),
                          ("history", "最近记录"), ("rollback", "恢复上一个策略版本")):
        sub = subs.add_parser(action, help=label)
        sub.add_argument("--json", action="store_true", help="输出机器可读 JSON")
        if action in ("solve", "probe"):
            sub.add_argument("task", help="JSON 清单路径，或 - 读取标准输入")
        if action == "probe":
            source = sub.add_mutually_exclusive_group()
            source.add_argument("--initial", action="store_true", help="使用开发者基线")
            source.add_argument("--nominee", type=int, help="只读试跑指定研究记录中的提名及当时回退链")
        if action == "show":
            sub.add_argument("id", type=int)
    demonstration = subs.add_parser("demo", help="在全新目录中跑完整重启实验")
    demonstration.add_argument("topic", nargs="?", choices=["triples", "mixed", "pilot"], help="选择演示")
    study = subs.add_parser("research", help="启动一轮配方研究，并为此状态开启后续失败的自主研究")
    study.add_argument("task", nargs="?", help="可选的触发清单")
    study.add_argument("--budget", type=int, default=research.DEFAULT_BUDGET, help="研究评估次数上限")
    study.add_argument("--json", action="store_true")
    why = subs.add_parser("explain", help="为冻结提名寻找可复算的对照例")
    why.add_argument("id", type=int, help="研究事件编号")
    why.add_argument("--task", help="在指定清单上解释，默认使用本轮练习")
    why.add_argument("--json", action="store_true")
    args = parser.parse_args()
    folder = args.state.expanduser().resolve()
    try:
        if args.cmd is None:
            return panel(folder)
        if args.cmd == "demo":
            if args.topic == "pilot":
                return demo_pilot(folder)
            demo(folder, triples=args.topic == "triples", mixed=args.topic == "mixed")
            return 0
        if args.cmd == "research":
            data = memory.investigate(folder, read_task(args.task) if args.task else None, args.budget)
        elif args.cmd == "explain":
            from .witness import explain
            row = research_record(folder, args.id)
            tasks = [read_task(args.task)] if args.task else row["practice_tasks"]
            data = explain(tasks, row["nominee"], row["policies_before"])
        elif args.cmd == "solve":
            data = memory.solve(folder, read_task(args.task))
        elif args.cmd == "probe":
            task = read_task(args.task)
            if args.nominee is not None:
                row = research_record(folder, args.nominee)
                policies = research.dispatch(row["nominee"], row["policies_before"])
            else:
                policies = [engine.BASE] if args.initial else memory.status(folder)["policies"]
            data = dict(task=task, result=Bench().apply(task, policies), policies=policies)
        elif args.cmd == "show":
            data = memory.event(folder, args.id)
        elif args.cmd == "rollback":
            data = memory.rollback(folder)
        elif args.cmd == "status":
            data = memory.status(folder)
        else:
            data = memory.history(folder, 30)
        if args.json:
            print(js(data))
        elif args.cmd in ("solve", "show", "rollback", "research"):
            show_round(data)
        elif args.cmd == "history":
            show_history(data)
        elif args.cmd == "status":
            print(f"当前 v{data['generation']}，共 {data['events']} 份记录\n{data['db']}")
            for i, policy in enumerate(data["policies"]):
                print(f"  {i + 1}. {policy_name(policy)}")
        elif args.cmd == "probe":
            show_solution(data["task"], data["result"])
        elif args.cmd == "explain":
            show_witness(data)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, sqlite3.Error, RuntimeError) as error:
        print(f"无法执行  {error}", file=sys.stderr)
        return 2
