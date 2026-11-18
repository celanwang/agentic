"""A bounded policy search. Only practice scores may influence the nominee."""
import hashlib
import json
import math
from functools import reduce
from fractions import Fraction

from . import arena, engine, recipes


RULES = {
    "max_candidates": 16, "max_practice": 18, "max_holdout": 80,
    "max_regression": 24, "max_policies": 4,
    "min_holdout": 30, "min_previous_successes": 8,
    "min_holdout_gains": 3, "allowed_losses": 0,
    "max_cost_ratio": 2.25,
}


def ident(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def task_key(task):
    # Reordering or rescaling the exact same workload is too weak a holdout.
    unit = reduce(math.gcd, task["items"], task["capacity"])
    return (task["bins"], task["capacity"] // unit,
            tuple(sorted(n // unit for n in task["items"])))


def validate_task(task):
    if not isinstance(task, dict) or set(task) != {"items", "capacity", "bins"}:
        raise ValueError("清单需要 capacity、bins、items 三个字段")
    if type(task["capacity"]) is not int or not 1 <= task["capacity"] <= 10**9:
        raise ValueError("capacity 必须为 1 到 10 亿之间的整数")
    if type(task["bins"]) is not int or not 1 <= task["bins"] <= 16:
        raise ValueError("bins 必须为 1 到 16 之间的整数")
    items = task["items"]
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("items 必须是最多含 64 个文件大小的数组")
    if any(type(size) is not int or not 1 <= size <= 10**9 for size in items):
        raise ValueError("文件大小必须为 1 到 10 亿之间的整数")
    return task


def sample(seed, phase, seen, labels=None):
    limit = RULES["max_" + phase]
    result = []
    for task, info in zip(arena.cases(seed, phase), arena.case_expectations(phase)):
        validate_task(task)
        key = task_key(task)
        if key in seen:
            continue
        seen.add(key)
        result.append(task)
        if labels is not None:
            labels[ident(task)] = info
        if len(result) == limit:
            break
    return result


class Bench:
    def __init__(self):
        self.cache = {}

    def single(self, task, policy):
        key = (ident(task), ident(policy))
        if key not in self.cache:
            executor = recipes if policy.get("kind") == "recipe" else engine
            result = executor.run(task, policy)
            if result["ok"] and not arena.check(task, result["packing"]):
                raise RuntimeError("策略返回了无效装法，本轮不提交")
            if type(result["cost"]) is not int or result["cost"] < 1:
                raise RuntimeError("策略成本无效，本轮不提交")
            self.cache[key] = result
        return self.cache[key]

    def apply(self, task, policies):
        attempts = []
        for policy in policies:
            last = self.single(task, policy)
            attempts.append({"policy": policy, "ok": last["ok"], "cost": last["cost"]})
            if last["ok"] or proven_impossible(last):
                break
        return {**last, "cost": sum(a["cost"] for a in attempts), "attempts": attempts}

    def compare(self, tasks, policies, candidate, labels=None):
        summary = dict(total=len(tasks), old_ok=0, new_ok=0, gains=0, losses=0,
                       old_cost=0, new_cost=0)
        rows = []
        for task in tasks:
            old = self.apply(task, policies)
            new = self.apply(task, policies + [candidate])
            summary["old_ok"] += old["ok"]
            summary["new_ok"] += new["ok"]
            summary["gains"] += new["ok"] and not old["ok"]
            summary["losses"] += old["ok"] and not new["ok"]
            summary["old_cost"] += old["cost"]
            summary["new_cost"] += new["cost"]
            if labels is not None:
                profile, possible = labels[ident(task)]
                rows.append(dict(task=task, profile=profile, feasible=possible, old_ok=old["ok"], new_ok=new["ok"],
                                 old_cost=old["cost"], new_cost=new["cost"]))
        if labels is not None:
            summary["rows"] = rows
            summary["feasible"] = sum(row["feasible"] for row in rows)
        return summary


def proven_impossible(result):
    trace = result["trace"]
    block = trace["blocked"] or {}
    return block.get("reason") in ("oversize", "total_excess") or trace["search_status"] == "infeasible"


def run_round(task, policies, generation, seed, previous_trial=None, bench=None):
    bench = bench if bench is not None else Bench()
    before = bench.apply(task, policies)
    report = dict(task=task, generation_before=generation, generation=generation,
                  policies_before=policies, seed=seed, rules=RULES, before=before,
                  after=before, trials=[], answer=None)

    def finish(decision, reason):
        if report["after"]["ok"]:
            report["answer"] = {"source": "active", "packing": report["after"]["packing"]}
        report.update(decision=decision, reason=reason, solver_calls=len(bench.cache),
                      executed_cost=sum(result["cost"] for result in bench.cache.values()))
        if "outcome" not in report:
            report["outcome"] = "packed" if report["answer"] is not None else "unresolved"
        return report

    if before["ok"]:
        return finish("unchanged", "已用当前策略装完全部文件。")
    if proven_impossible(before):
        report["outcome"] = "infeasible"
        return finish("unchanged", "容量约束或穷尽搜索已证明清单无解。")
    if previous_trial is not None:
        prior_answer = previous_trial["answer"]
        if prior_answer and arena.check(task, prior_answer["packing"]):
            report["answer"] = {"source": "record", "packing": prior_answer["packing"]}
        if previous_trial["outcome"] == "infeasible":
            report["outcome"] = "infeasible"
        report["reused_event"] = previous_trial["id"]
        return finish("unchanged", "该清单在相同策略下已做过实验，见记录 #%s。" % previous_trial["id"])
    if len(policies) >= RULES["max_policies"]:
        return finish("budget", "策略序列已达到长度上限。")

    proposal = engine.propose(task, policies[-1], bench.single(task, policies[-1]))
    report["observation"] = proposal["observation"]
    seen = {task_key(task)}
    practice = sample(seed, "practice", seen)
    report["practice_tasks"] = practice
    candidates = []
    for policy in proposal["candidates"][:RULES["max_candidates"]]:
        if policy not in policies and policy not in candidates:
            candidates.append(policy)
    eligible = []
    for index, policy in enumerate(candidates):
        attempt = bench.single(task, policy)
        trial = dict(policy=policy, trigger_ok=attempt["ok"], trigger_cost=attempt["cost"],
                     trigger_trace=attempt["trace"])
        report["trials"].append(trial)
        if not attempt["ok"] and proven_impossible(attempt):
            report["outcome"] = "infeasible"
            return finish("unchanged", "候选已穷尽适用的搜索分支，证明此清单无解。")
        if attempt["ok"]:
            score = bench.compare(practice, policies, policy)
            trial["practice"] = score
            extra_cost = max(1, score["new_cost"] - score["old_cost"])
            efficiency = Fraction(score["gains"], extra_cost)
            eligible.append(((-efficiency, -score["gains"], extra_cost, attempt["cost"], index), policy))
    if not eligible:
        return finish("inconclusive", "本轮候选都没能解决触发任务，保留当前策略。")

    candidate = min(eligible, key=lambda pair: pair[0])[1]
    report["nominee"] = candidate
    report["answer"] = {"source": "trial", "packing": bench.single(task, candidate)["packing"]}
    # The nominee is fixed. No feedback from either group reaches proposal or ranking.
    labels = {}
    holdout = bench.compare(sample(seed, "holdout", seen, labels), policies, candidate, labels)
    regression = bench.compare(sample(seed, "regression", seen, labels), policies, candidate, labels)
    report["holdout"], report["regression"] = holdout, regression
    old_cost = holdout["old_cost"] + regression["old_cost"]
    new_cost = holdout["new_cost"] + regression["new_cost"]
    if holdout["total"] < RULES["min_holdout"] or regression["old_ok"] < RULES["min_previous_successes"]:
        return finish("inconclusive", "独立清单或旧策略成功样本不足。")
    if holdout["losses"] + regression["losses"] > RULES["allowed_losses"]:
        return finish("rejected", "独立对照中丢失了原有成功案例。")
    if new_cost > RULES["max_cost_ratio"] * old_cost:
        return finish("rejected", "独立两组的总操作成本超过预设倍率。")
    if holdout["gains"] < RULES["min_holdout_gains"]:
        return finish("inconclusive", "留出组新增成功不足 %d 例。" % RULES["min_holdout_gains"])
    report["after"] = bench.apply(task, policies + [candidate])
    return finish("kept", "触发任务已解决，留出组增益、回归和成本均达到预设门槛。")


def runtime_id(files):
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
