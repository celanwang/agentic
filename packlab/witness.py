"""Read-only, bounded counterfactuals on caller-supplied user or practice inputs."""
from packlab import arena, recipes
from packlab.experiment import Bench, ident, proven_impossible, validate_task
from packlab.research import dispatch


class _OutOfRuns(Exception):
    pass


class _Meter(Bench):
    """Share serving-chain termination and cost accounting with Bench.apply."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.hits = 0
        self.charged_ops = 0

    def single(self, task, policy):
        key = (ident(task), ident(policy))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if len(self.cache) >= self.limit:
            raise _OutOfRuns
        result = super().single(task, policy)
        _check(task, result)
        self.charged_ops += result["cost"]
        return result


def _check(task, result):
    groups = result["packing"]
    if type(result["cost"]) is not int or result["cost"] <= 0 or len(groups) != task["bins"]:
        raise RuntimeError("见证求解器返回无效费用或盘数")
    items = task["items"]
    used = set()
    for group in groups:
        load = 0
        for i in group:
            if type(i) is not int or not 0 <= i < len(items) or i in used:
                raise RuntimeError("见证中的文件编号无效或重复")
            load += items[i]
            used.add(i)
        if load > task["capacity"]:
            raise RuntimeError("见证中的盘超容量")
    if result["ok"] and not arena.check(task, groups):
        raise RuntimeError("见证中的完整装法未通过核验")


def _controls(program, policies):
    steps = program["steps"]
    result = []
    for i, step in enumerate(steps):
        edit = dict(program, steps=steps[:i] + steps[i + 1:])
        result.append(("without_" + step, dispatch(edit, policies)))
    if len(steps) == 2:
        result.append(("reverse_steps", dispatch(dict(program, steps=steps[::-1]), policies)))
    if program["guard"] is not None:
        unguarded = dict(program, guard=None)
        backup = unguarded.pop("fallback", None)
        chain = dispatch(unguarded, policies)
        if backup is not None:
            chain.insert(1, backup)
        result.append(("without_guard", chain))
    result.append(("previous_chain", list(policies)))
    return result


def _endpoint(meter, task, chain):
    result = meter.apply(task, chain)
    attempts = []
    for attempt in result["attempts"]:
        trial = meter.cache[(ident(task), ident(attempt["policy"]))]
        trace = trial["trace"]
        guard = trace.get("guard")
        attempts.append(dict(attempt, guard_matched=guard["matched"] if guard else None,
                             solver_ran=trace.get("solver_ran", True),
                             search_status=trace["search_status"]))
    outcome = "solved" if result["ok"] else "infeasible" if proven_impossible(result) else "unresolved"
    return dict(outcome=outcome, ok=result["ok"], cost=result["cost"],
                packing=result["packing"], attempts=attempts,
                packing_check="complete" if result["ok"] else "partial")


def _difference(full, control):
    left, right = full["outcome"], control["outcome"]
    if {left, right} == {"solved", "infeasible"}:
        raise RuntimeError("同一清单既有核验成功的装法，又有相冲突的无解判定")
    if left == "solved" and right != left:
        return "success_gain"
    if right == "solved" and right != left:
        return "success_loss"
    if left == right and full["cost"] < control["cost"]:
        return "cost_saving"
    return None


def _signature(full, control):
    def path(endpoint):
        return (endpoint["outcome"], tuple(
            (a["guard_matched"], a["solver_ran"], a["search_status"])
            for a in endpoint["attempts"]))
    return _difference(full, control), path(full), path(control)


def _pair_view(task, kept, full, control):
    def mapped(endpoint):
        return dict(endpoint, packing_original=[[kept[i] for i in g] for g in endpoint["packing"]])
    return dict(task=task, kept_original=kept, full=mapped(full), control=mapped(control),
                difference=_difference(full, control), saved_ops=control["cost"] - full["cost"])


def explain(tasks, program, policies_before, budget=96):
    """budget counts actual uncached (task, solver/recipe) executions, including fallbacks."""
    recipes.validate(program)
    if type(budget) is not int or budget < 0:
        raise ValueError("见证预算须为非负整数")
    if not isinstance(tasks, (list, tuple)) or not isinstance(policies_before, (list, tuple)) or not policies_before:
        raise ValueError("需要用户或练习清单序列和非空原活动策略链")
    meter = _Meter(budget)
    chain = dispatch(program, policies_before)
    controls = _controls(program, policies_before)
    report = dict(scope="supplied_tasks", status="no_witness", witness=None, inspected=[],
                  program=program, full_chain=chain, controls=dict(controls),
                  budget_unit="uncached_task_policy_runs", budget=budget)
    selected = None
    best_rank = -1
    stopped = "all_tasks_checked"
    try:
        for i, task in enumerate(tasks):
            validate_task(task)
            full = _endpoint(meter, task, chain)
            item = dict(task_index=i, full_outcome=full["outcome"], full_cost=full["cost"], comparisons=[])
            report["inspected"].append(item)
            for label, alt in controls:
                other = _endpoint(meter, task, alt)
                diff = _difference(full, other)
                item["comparisons"].append(dict(control=label, outcome=other["outcome"],
                                                cost=other["cost"], difference=diff))
                if diff is None:
                    continue
                # A change in success outranks a change in operation cost.
                # On ties keep the earlier input and the smaller single edit.
                rank = 2 if diff != "cost_saving" else 1 if full["ok"] else 0
                if rank > best_rank:
                    selected = (i, task, label, alt, full, other)
                    best_rank = rank
            if best_rank == 2 or (selected is not None and len(meter.cache) * 2 >= budget):
                stopped = "witness_selected"
                break
    except _OutOfRuns:
        stopped = "budget"
    report["discovery_stop"] = stopped

    if selected is not None:
        i, source, label, alt, first_full, first_control = selected
        kept = list(range(len(source["items"])))
        initial = _pair_view(source, kept, first_full, first_control)
        signature = _signature(first_full, first_control)
        reduced = initial
        width = max(1, (len(kept) + 1) // 2)
        reductions = []
        checked = 0
        local_stop = "no_single_deletion_preserved_difference"
        try:
            while kept and width:
                accepted = False
                for start in range(0, len(kept), width):
                    candidate_ids = kept[:start] + kept[start + width:]
                    trial = dict(source, items=[source["items"][j] for j in candidate_ids])
                    left = _endpoint(meter, trial, chain)
                    right = _endpoint(meter, trial, alt)
                    checked += 1
                    if _signature(left, right) == signature:
                        reductions.append(kept[start:start + width])
                        kept = candidate_ids
                        reduced = _pair_view(trial, kept, left, right)
                        width = min(width, len(kept))
                        accepted = True
                        break
                if not accepted:
                    width //= 2
        except _OutOfRuns:
            local_stop = "budget"
        if not kept:
            local_stop = "empty_input"
        report.update(status="witness", witness=dict(task_index=i, control=label, initial=initial,
                      reduced=reduced, deleted_original=sorted(j for chunk in reductions for j in chunk),
                      accepted_deletions=reductions, completed_reduction_pairs=checked, local_stop=local_stop,
                      reduction_scope="只在容量、盘数及单位固定时局部删项，全局最小性未检验。"))
    checked_pairs = sum(len(item["comparisons"]) for item in report["inspected"])
    report.update(runs_used=len(meter.cache), cache_hits=meter.hits, executed_ops=meter.charged_ops,
                  tasks_total=len(tasks), unchecked_input_pairs=len(tasks) * len(controls) - checked_pairs,
                  scope_note="结论限于实际核验的清单及其局部删项。")
    if selected is None:
        report["finding"] = "已比较的两端中未找到成功差异或同终态的费用节省。"
    else:
        report["finding"] = {
            "success_gain": "完整配方链成功，对照链未解。",
            "success_loss": "完整配方链未解，对照链成功。",
            "cost_saving": "两条链终态一致，完整配方链使用更少的计费操作。",
        }[initial["difference"]]
    return report
