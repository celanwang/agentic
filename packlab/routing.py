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


def _with_guard(route, position, guard):
    if position == "primary":
        return dict(route, guard=guard)
    return dict(route, fallback=dict(route["fallback"], guard=guard))


def _fit(branch, old, features, families, evidence, widening=None, eligible=None):
    """Fit observed splits, or strict supersets of one deployed predicate."""
    best = None
    for feature in ([widening["feature"]] if widening else sorted(features[0])):
        values = [row[feature] for row in features]
        distinct = sorted(set(values))
        for lo, hi in zip(distinct, distinct[1:]):
            for op, threshold in (("le", lo), ("ge", hi)):
                if widening:
                    boundary = widening["value"]
                    if op != widening["op"] or not (threshold > boundary if op == "le" else threshold < boundary):
                        continue
                hit = [v <= lo for v in values] if op == "le" else [v >= hi for v in values]
                evidence["fit_spent"] += len(values)
                positive, negative = set(), set()
                for i, matched in enumerate(hit):
                    if eligible is not None and i not in eligible:
                        continue
                    a, b = old[i], branch[i]
                    if matched and b["ok"] and (not a["ok"] or b["cost"] < a["cost"]):
                        positive.add(families[i])
                    if not matched and not (b["ok"] and not a["ok"]) and b["cost"] >= a["cost"]:
                        negative.add(families[i])
                if min(len(positive), len(negative)) < 2:
                    continue
                prediction = [b if match else f for b, f, match in zip(branch, old, hit)]
                score = _summary(old, prediction)
                if score["losses"] or not (score["gains"] or score["new_cost"] < score["old_cost"]):
                    continue
                if widening and score["new_cost"] > RULES["max_cost_ratio"] * score["old_cost"]:
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
    options = {}
    for policy in [*skills, *ordinary]:
        if policy.get("kind") != "recipe":
            options.setdefault(_id(policy), policy)
    n = len(tasks)
    base_units = bool(ordinary) + int(active is not None)
    occupied = active is not None and "fallback" in active
    if n < 4 or (not options and not occupied) or budget < (base_units + 1) * n:
        return finish("练习样本、可试普通能力或配对验证预算不足。")

    downstream = ([bill(i, "baseline", ordinary) for i in range(n)] if ordinary else
                  [dict(ok=False, proof=False, cost=0) for _ in tasks])

    def routed(policy, stage):
        return [_combined(bill(i, stage, [policy]), downstream[i]) for i in range(n)]

    old = report["baseline"] = routed(active, "baseline") if active else downstream
    report["fallback"] = old
    ops = {}
    features = report["features"] = [recipes.route_features(task, ops=ops) for task in tasks]
    report["feature_ops"] = ops
    report["feature_spent"] = sum(ops.values())
    families = [family(t) for t in tasks]

    # Reserve a complete finalist panel. Screen on both failures and old successes,
    # including costly successes where a shorter path could pay for its gate.
    def nominate_new():
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
            alternate = [_combined(tested[key][i], old[i]) for i in pilot]
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
            branch = [_combined(tested[key][i], old[i]) for i in range(n)]
            fit = _fit(branch, old, features, families, report)
            if fit is not None:
                rank, model = fit
                report["fitted"].append(dict(model, candidate=key))
                if winner is None or rank > winner[0]:
                    program = dict(kind="recipe", steps=[], solver=options[key], guard=model["guard"])
                    if active is not None:
                        program["fallback"] = active
                    winner = rank, program
        return winner

    if occupied:
        # Both DSL leaves are occupied. Widen one predicate and preserve both
        # actions, their order, and every previously admitted input. The cost
        # of a forced match is exactly its cost under any matching threshold.
        alternatives = []
        for position in ("primary", "fallback"):
            leaf = active if position == "primary" else active["fallback"]
            guard = leaf["guard"]
            if guard is None or guard["feature"] not in features[0]:
                continue
            values = [row[guard["feature"]] for row in features]
            upper = guard["op"] == "le"
            changed = [i for i, v in enumerate(values)
                       if (v > guard["value"] if upper else v < guard["value"])
                       and (position == "primary" or
                            bench.single(tasks[i], active)["trace"]["route"]["selected"] == "fallback")]
            if len({families[i] for i in changed}) < 2:
                continue
            force = dict(guard, value=(max if upper else min)(values))
            probe = _with_guard(active, position, force)
            alternatives.append((position, changed, probe, guard))

        # An incomplete counterfactual panel cannot nominate a winner. Try the
        # least expensive complete panel first when only one fits in the budget.
        winner = None
        for position, changed, probe, guard in sorted(alternatives, key=lambda entry: len(entry[1])):
            price = len({ident(tasks[i]) for i in changed if (ident(tasks[i]), ident(probe)) not in bench.cache})
            if report["calls"] + price + n > budget:
                continue
            key = _id(probe)
            report["candidates"][key] = {"program": probe}
            branch = old.copy()
            for i in changed:
                row = bill(i, "candidate", [probe])
                branch[i] = _combined(row, downstream[i])
                report["trials"].append(dict(row, candidate=key))
                if row["ok"] and (i not in solutions or (row["cost"], key) < solutions[i]):
                    solutions[i] = (row["cost"], key)
            fit = _fit(branch, old, features, families, report, widening=guard, eligible=set(changed))
            if fit:
                rank, model = fit
                report["fitted"].append(dict(model, candidate=key, position=position))
                if winner is None or rank > winner[0]:
                    program = _with_guard(active, position, model["guard"])
                    winner = rank, program
    else:
        winner = nominate_new()

    if winner is None:
        return finish("练习中未找到同时有正反例支持的数值条件。")

    _, program = winner
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
