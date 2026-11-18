"""Induce one measured conditional route from unlabeled practice tasks."""
import hashlib

from . import recipes
from .experiment import Bench, RULES, ident, proven_impossible, validate_task
from .research import family


def _id(value):
    return hashlib.sha256(ident(value).encode()).hexdigest()


def _summary(old, new):
    return dict(total=len(new), gains=sum(b["ok"] and not a["ok"] for a, b in zip(old, new)),
                losses=sum(a["ok"] and not b["ok"] for a, b in zip(old, new)),
                old_cost=sum(r["cost"] for r in old), new_cost=sum(r["cost"] for r in new))


def _combined(raw, fallback):
    if raw["ok"] or raw["proof"]:
        return raw
    return dict(ok=fallback["ok"], proof=fallback["proof"], cost=raw["cost"] + fallback["cost"])


def _fit(raw, fallback, old, features, families, evidence):
    """Use observed thresholds. Costs here exclude the gate; the finalist pays it in full."""
    branch = [_combined(r, f) for r, f in zip(raw, fallback)]
    best = None
    for feature in sorted(features[0]):
        values = [row[feature] for row in features]
        distinct = sorted(set(values))
        for lo, hi in zip(distinct, distinct[1:]):
            for op, threshold in (("le", lo), ("ge", hi)):
                hit = [v <= lo for v in values] if op == "le" else [v >= hi for v in values]
                evidence["fit_spent"] += len(values)
                positive, negative = set(), set()
                for i, matched in enumerate(hit):
                    a, b = fallback[i], branch[i]
                    if matched and b["ok"] and (not a["ok"] or b["cost"] < a["cost"]):
                        positive.add(families[i])
                    if not matched and not (b["ok"] and not a["ok"]) and b["cost"] >= a["cost"]:
                        negative.add(families[i])
                if min(len(positive), len(negative)) < 2:
                    continue
                prediction = [b if match else f for b, f, match in zip(branch, fallback, hit)]
                score = _summary(old, prediction)
                if score["losses"] or not (score["gains"] or score["new_cost"] < score["old_cost"]):
                    continue
                rank = (score["gains"], -score["new_cost"], min(len(positive), len(negative)))
                if best is None or rank > best[0]:
                    best = rank, dict(guard=dict(feature=feature, op=op, value=threshold),
                                      positive_families=len(positive), negative_families=len(negative),
                                      matched=[i for i, flag in enumerate(hit) if flag], prediction=score)
    return best


def search(tasks, policies, skills, budget=128):
    """Meter each executed task/path once, plus observed feature work and fitted predicate tests."""
    if type(budget) is not int or budget < 0 or len(tasks) > 18 or not policies:
        raise ValueError("路由研究需要非负整数预算、活动策略和最多 18 份练习")
    for task in tasks:
        validate_task(task)
    report = dict(program=None, finalist=None, trials=[], candidates={}, practice_solutions=[],
                  calls=0, spent=0, budget=budget, feature_spent=0, fit_spent=0, features=[],
                  baseline=[], fallback=[], fitted=[], comparison=_summary([], []), verification=[])
    solutions = {}
    bench = Bench()
    report["ledger"] = []

    def finish(reason):
        report["reason"] = reason
        report["spent"] += report["feature_spent"] + report["fit_spent"]
        for stage in ("baseline", "candidate", "fallback", "verification"):
            report[stage + "_calls"] = sum(row["calls"] for row in report["ledger"] if row["stage"] == stage)
        report["physical_calls"] = len(bench.cache)
        report["practice_solutions"] = [dict(task_index=i, candidate=key, cost=cost)
                                        for i, (cost, key) in sorted(solutions.items())]
        return report

    def bill(index, stage, path):
        # One ordinary path or one standalone recipe/solver per logical unit.
        # Bench charges only new physical executions, including short-circuiting.
        before = len(bench.cache)
        result = bench.apply(tasks[index], path)
        fresh = list(bench.cache.values())[before:]
        calls, spent = int(bool(fresh)), sum(r["cost"] for r in fresh)
        report["ledger"].append(dict(task_index=index, stage=stage, path=path, calls=calls,
                                     physical_calls=len(fresh), spent=spent))
        report["calls"] += calls
        report["spent"] += spent
        if report["calls"] > budget:
            raise RuntimeError("路由练习超出预算")
        return dict(task_index=index, ok=result["ok"], proof=proven_impossible(result),
                    cost=result["cost"], spent=spent)

    ordinary = [p for p in policies if p.get("kind") != "recipe"]
    active = next((p for p in policies if p.get("kind") == "recipe"), None)
    bottom = active.get("fallback", active) if active is not None else None
    options = {}
    for policy in [*skills, *ordinary]:
        if policy.get("kind") != "recipe":
            options.setdefault(_id(policy), policy)
    n = len(tasks)
    base_units = bool(ordinary) + int(active is not None) + int(bottom != active)
    if n < 4 or not options or budget < (base_units + 1) * n:
        return finish("练习样本、可试普通能力或配对验证预算不足。")

    downstream = ([bill(i, "baseline", ordinary) for i in range(n)] if ordinary else
                  [dict(ok=False, proof=False, cost=0) for _ in tasks])

    def routed(policy, stage):
        return [_combined(bill(i, stage, [policy]), downstream[i]) for i in range(n)]

    old = report["baseline"] = routed(active, "baseline") if active else downstream
    fallback = report["fallback"] = old if bottom == active else routed(bottom, "fallback")
    ops = {}
    features = report["features"] = [recipes.route_features(task, ops=ops) for task in tasks]
    report["feature_ops"] = ops
    report["feature_spent"] = sum(ops.values())
    families = [family(t) for t in tasks]

    # Reserve a complete finalist panel. Screen on both failures and old successes,
    # including costly successes where a shorter path could pay for its gate.
    room = budget - report["calls"] - n
    task_ids = [ident(t) for t in tasks]
    def price(key, indices):
        policy_id = ident(options[key])
        return len({task_ids[i] for i in indices if (task_ids[i], policy_id) not in bench.cache})

    blocked = sorted((i for i in range(n) if not old[i]["ok"] and not old[i]["proof"]),
                     key=lambda i: old[i]["cost"], reverse=True)
    solved = sorted((i for i in range(n) if old[i]["ok"]), key=lambda i: old[i]["cost"], reverse=True)
    order = list(dict.fromkeys(blocked[:1] + solved[:1] + solved[-1:] + blocked + list(range(n))))
    if sum(price(key, range(n)) for key in options) <= room:
        pilot = list(range(n))
    else:
        width = min(4, n, max(1, (room - n) // max(1, len(options) - 1)))
        pilot = order[:width]
    tested = {}
    screening = reserve = 0
    for key, policy in options.items():
        first = price(key, pilot)
        rest = price(key, range(n)) - first
        if screening + first + max(reserve, rest) <= room:
            report["candidates"][key] = {"program": policy}
            tested[key] = {}
            screening += first
            reserve = max(reserve, rest)

    def probe(key, index):
        row = bill(index, "candidate", [options[key]])
        tested[key][index] = row
        report["trials"].append(dict(row, candidate=key))
        if row["ok"]:
            choice = (row["cost"], key)
            if index not in solutions or choice < solutions[index]:
                solutions[index] = choice

    priorities = {}
    for key in tested:
        for i in pilot:
            probe(key, i)
        alternate = [_combined(tested[key][i], fallback[i]) for i in pilot]
        # Credit local improvement even if this solver fails on negative controls.
        rescues = sum(b["ok"] and not old[i]["ok"] for i, b in zip(pilot, alternate))
        savings = sum(max(0, old[i]["cost"] - b["cost"]) for i, b in zip(pilot, alternate))
        priorities[key] = (rescues, savings, -sum(r["cost"] for r in tested[key].values()))

    winner = None
    for key in sorted(tested, key=priorities.get, reverse=True):
        missing = [i for i in range(n) if i not in tested[key]]
        if report["calls"] + price(key, missing) + n > budget:
            continue
        for i in missing:
            probe(key, i)
        fit = _fit([tested[key][i] for i in range(n)], fallback, old, features, families, report)
        if fit is not None:
            rank, model = fit
            report["fitted"].append(dict(model, candidate=key))
            if winner is None or rank > winner[0]:
                winner = rank, key, model
    if winner is None:
        return finish("练习中未找到同时有正反例支持的数值条件。")

    _, key, model = winner
    program = dict(kind="recipe", steps=[], solver=options[key], guard=model["guard"])
    if bottom is not None:
        program["fallback"] = bottom
    recipes.validate(program)
    report["finalist"] = program
    fresh = report["verification"] = []
    for i in range(n):
        first = bill(i, "verification", [program])
        # The ordinary chain runs only after an inconclusive route failure.
        row = dict(_combined(first, downstream[i]), task_index=i)
        trace = bench.single(tasks[i], program)["trace"]
        row.update(guard=trace["guard"], route=trace.get("route"), recipe_cost=first["cost"],
                   fallback_cost=row["cost"] - first["cost"])
        fresh.append(row)
    score = report["comparison"] = _summary(old, fresh)
    if score["losses"] or score["new_cost"] > RULES["max_cost_ratio"] * score["old_cost"]:
        return finish("完整路径实测出现丢解或成本超过预算倍率。")
    if not score["gains"] and score["new_cost"] >= score["old_cost"]:
        return finish("计入条件判断及回退后，练习面板没有净改善。")
    report["program"] = program
    return finish("经全练习面板实测，这一条路由保住原有成功并获得新增解或节省操作。")
