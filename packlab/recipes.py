"""Execute bounded, reversible recipe steps and one optional fallback."""
from collections import Counter
from math import gcd

from packlab import engine


STEPS = ("normalize_gcd", "peel_forced")
FEATURES = ("capacity", "gcd", "unit_capacity", "bins", "items", "forced_singletons",
            "slack", "load_permille", "items_per_bin_permille", "max_permille")
ROUTE_FEATURES = FEATURES + ("complementary_coverage",)
_FEATURE_NAMES = {
    "capacity": "单盘容量", "gcd": "共同单位", "unit_capacity": "约分后单盘容量",
    "bins": "盘数", "items": "文件数", "forced_singletons": "独占盘文件数",
    "slack": "总闲置量", "load_permille": "总装载率（‰）",
    "items_per_bin_permille": "每盘平均文件数（乘以 1000）", "max_permille": "最大文件占单盘容量（‰）",
    "complementary_coverage": "互补配对覆盖率（‰）",
}
_STEP_NAMES = {"normalize_gcd": "消除共同单位", "peel_forced": "剥离独占盘文件"}
_FIELDS = {"kind", "steps", "solver", "guard"}


def validate(program):
    """Validate the JSON grammar. No rewriting or mutation of the stored program."""
    if not isinstance(program, dict) or set(program) not in (_FIELDS, _FIELDS | {"fallback"}):
        raise ValueError("配方字段应为 kind、steps、solver、guard，可选 fallback")
    _validate_leaf(program)
    if "fallback" in program:
        fallback = program["fallback"]
        if program["guard"] is None or not isinstance(fallback, dict) or set(fallback) != _FIELDS:
            raise ValueError("路由配方须有前置条件和一层普通递补配方")
        _validate_leaf(fallback)


def _validate_leaf(program):
    if program["kind"] != "recipe" or not isinstance(program["solver"], dict):
        raise ValueError("需要 recipe 和普通求解策略")
    engine.describe(program["solver"])
    steps = program["steps"]
    if (not isinstance(steps, list) or len(steps) > len(STEPS) or
            any(step not in STEPS for step in steps) or len(set(steps)) != len(steps)):
        raise ValueError("步骤必须来自 STEPS，每种最多一次")
    guard = program["guard"]
    if guard is not None and (
            not isinstance(guard, dict) or set(guard) != {"feature", "op", "value"} or
            guard["feature"] not in ROUTE_FEATURES or guard["op"] not in ("le", "ge") or
            type(guard["value"]) is not int):
        raise ValueError("条件必须是整数特征与整数阈值的 le/ge 比较")


def describe(program):
    validate(program)
    primary = _describe_leaf(program)
    if "fallback" in program:
        return primary + "。跳过或仍待求解时递补 " + _describe_leaf(program["fallback"])
    return primary


def _describe_leaf(program):
    parts = []
    guard = program["guard"]
    if guard:
        symbol = "≤" if guard["op"] == "le" else "≥"
        parts.append(f"若{_FEATURE_NAMES[guard['feature']]} {symbol} {guard['value']}")
    parts.extend(_STEP_NAMES[s] for s in program["steps"])
    parts.append(engine.describe(program["solver"]))
    return " → ".join(parts)


def _inputs(task):
    capacity, bins, sizes = task["capacity"], task["bins"], task["items"]
    if (type(capacity) is not int or not 1 <= capacity <= 10**9 or
            type(bins) is not int or not 1 <= bins <= 16 or
            not isinstance(sizes, (list, tuple)) or len(sizes) > 64 or
            any(type(x) is not int or not 1 <= x <= 10**9 for x in sizes)):
        raise ValueError("无效的容量、盘数或文件大小")
    return capacity, bins, sizes


def _unit(capacity, sizes, ops):
    unit = capacity
    for x in sizes:
        if unit == 1:
            break
        ops["recipe_gcd"] += 1
        unit = gcd(unit, x)
    return unit


def _forced(capacity, sizes, ops):
    # Two order statistics include multiplicity. Each candidate must be tested
    # against a different file, including when its value is the minimum.
    first = second = capacity + 1
    for x in sizes:
        if x <= first:
            second, first = first, x
        elif x < second:
            second = x
    ops["recipe_isolation"] += 2 * len(sizes)
    return [x <= capacity and x + (second if x == first else first) > capacity for x in sizes]


def _observe(keys, capacity, bins, sizes, ops):
    n = len(sizes)
    values = {"capacity": capacity, "bins": bins, "items": n}
    if "gcd" in keys or "unit_capacity" in keys:
        values["gcd"] = _unit(capacity, sizes, ops)
        if "unit_capacity" in keys:
            values["unit_capacity"] = capacity // values["gcd"]
    if "forced_singletons" in keys:
        values["forced_singletons"] = sum(_forced(capacity, sizes, ops))
        ops["recipe_features"] += n
    if "slack" in keys or "load_permille" in keys:
        total = sum(sizes)
        ops["recipe_features"] += n
        values["slack"] = capacity * bins - total
        if "load_permille" in keys:
            values["load_permille"] = 1000 * total // (capacity * bins)
    if "items_per_bin_permille" in keys:
        values["items_per_bin_permille"] = 1000 * n // bins
    if "max_permille" in keys:
        values["max_permille"] = 1000 * max(sizes, default=0) // capacity
        ops["recipe_features"] += n
    if "complementary_coverage" in keys:
        counts = Counter(sizes)
        ops["recipe_complements"] = n + len(counts)
        pairs = 0
        for size, count in counts.items():
            complement = capacity - size
            if size < complement:
                pairs += min(count, counts[complement])
            elif size == complement:
                pairs += count // 2
        values["complementary_coverage"] = 2000 * pairs // n if n else 0
    ops["recipe_features"] += len(keys)
    return {key: values[key] for key in keys}


def features(task):
    """All public observations use original units; ratios are integer thousandths."""
    return _observe(FEATURES, *_inputs(task),
                    dict(recipe_gcd=0, recipe_features=0, recipe_isolation=0))


def route_features(task, *, ops=None):
    """Include structural observations and optionally accumulate their operation counts."""
    capacity, bins, sizes = _inputs(task)
    counts = dict(recipe_gcd=0, recipe_features=0, recipe_isolation=0,
                  recipe_feature_input=3 + len(sizes))
    observed = _observe(ROUTE_FEATURES, capacity, bins, sizes, counts)
    if ops is not None:
        for key, value in counts.items():
            ops[key] = ops.get(key, 0) + value
    return observed


def run(task, program):
    validate(program)
    primary = _run_leaf(task, program)
    if "fallback" not in program:
        return primary

    from .experiment import proven_impossible

    fallback = None
    if not primary["ok"] and not proven_impossible(primary):
        fallback = _run_leaf(task, program["fallback"])
    result = fallback if fallback is not None else primary
    ops = dict(primary["trace"]["ops"], recipe_route=1)
    if fallback is not None:
        for key, value in fallback["trace"]["ops"].items():
            ops[key] = ops.get(key, 0) + value

    def summarize(trial):
        trace = trial["trace"]
        return dict(ok=trial["ok"], cost=trial["cost"], guard=trace["guard"],
                    search_status=trace["search_status"])

    route = dict(selected="fallback" if fallback is not None else "primary",
                 primary=summarize(primary),
                 fallback=summarize(fallback) if fallback is not None else None)
    trace = dict(result["trace"], guard=primary["trace"]["guard"], ops=ops, route=route)
    if fallback is not None:
        trace["solver_ran"] = primary["trace"]["solver_ran"] or trace["solver_ran"]
        trace["solver_cost"] += primary["trace"]["solver_cost"]
    return dict(result, trace=trace, cost=sum(ops.values()))


def _run_leaf(task, program):
    capacity, count, sizes = _inputs(task)
    original_capacity, original_sizes = capacity, sizes
    ops = dict(recipe_setup=4 + len(sizes) + len(program["steps"]), recipe_features=0,
               recipe_guard=0, recipe_gcd=0, recipe_isolation=0,
               recipe_transform=0, recipe_inverse=0)
    trace = {"blocked": None, "repaired": [], "search_status": None, "ops": ops,
             "guard": None, "steps": [], "solver_ran": False, "solver_cost": 0}

    def finish(ok, packing):
        return {"ok": ok, "packing": packing, "cost": sum(ops.values()), "trace": trace}

    guard = program["guard"]
    if guard is not None:
        name = guard["feature"]
        observed = _observe((name,), capacity, count, sizes, ops)[name]
        matched = observed <= guard["value"] if guard["op"] == "le" else observed >= guard["value"]
        trace["guard"] = dict(guard, observed=observed, matched=matched)
        ops["recipe_guard"] += 1
        if not matched:
            trace["search_status"] = "guard_skip"
            ops["recipe_inverse"] += count
            return finish(False, [[] for _ in range(count)])

    origin = list(range(len(sizes)))
    ops["recipe_transform"] += len(origin)
    fixed = []
    unit = 1
    for step in program["steps"]:
        if step == "normalize_gcd":
            factor = _unit(capacity, sizes, ops)
            if factor > 1:
                capacity //= factor
                sizes = [x // factor for x in sizes]
                unit *= factor
                ops["recipe_transform"] += 2 + len(sizes)
            trace["steps"].append(dict(step=step, factor=factor))
        else:
            keep, new_sizes, peeled = [], [], []
            for i, forced in enumerate(_forced(capacity, sizes, ops)):
                if forced:
                    peeled.append(origin[i])
                else:
                    keep.append(origin[i])
                    new_sizes.append(sizes[i])
            ops["recipe_transform"] += 2 * len(origin)
            origin, sizes = keep, new_sizes
            fixed.extend(peeled)
            trace["steps"].append(dict(step=step, peeled=peeled))

    slots = count - len(fixed)
    trace["solver_space"] = dict(capacity=capacity, bins=max(0, slots), items=len(sizes), unit=unit)
    packing = [[i] for i in fixed[:count]]
    ops["recipe_inverse"] += len(packing)
    if slots < 0 or (slots == 0 and sizes):
        i = fixed[count] if slots < 0 else origin[0]
        gaps = [original_capacity - original_sizes[g[0]] for g in packing]
        trace["blocked"] = dict(reason="forced_disk_shortage", index=i, size=original_sizes[i],
                                step=len(packing), free=gaps,
                                smaller_before=sum(original_sizes[g[0]] < original_sizes[i] for g in packing))
        ops["recipe_inverse"] += 2 * len(packing) + 1
        trace["search_status"] = "infeasible"
        return finish(False, packing)
    if not sizes:
        packing.extend([] for _ in range(slots))
        ops["recipe_inverse"] += slots
        return finish(True, packing)

    reduced = dict(capacity=capacity, bins=slots, items=sizes)
    trial = engine.run(reduced, program["solver"])
    inner = trial["trace"]
    ops.update(inner["ops"])
    trace["solver_ran"], trace["solver_cost"] = True, trial["cost"]
    trace["search_status"] = inner["search_status"]
    for group in trial["packing"]:
        packing.append([origin[i] for i in group])
        ops["recipe_inverse"] += 1 + len(group)
    for repair in inner["repaired"]:
        trace["repaired"].append(dict(index=origin[repair["index"]], bin=len(fixed) + repair["bin"]))
        ops["recipe_inverse"] += 2
    if inner["blocked"]:
        b = inner["blocked"]
        idx = origin[b["index"]]
        gaps = [original_capacity - original_sizes[i] for i in fixed]
        gaps.extend(gap * unit for gap in b["free"])
        trace["blocked"] = dict(b, index=idx, size=original_sizes[idx], free=gaps,
                                step=len(fixed) + b["step"],
                                smaller_before=b["smaller_before"] + sum(
                                    original_sizes[i] < original_sizes[idx] for i in fixed))
        ops["recipe_inverse"] += 3 + len(gaps) + len(fixed)
        # Exclusively occupied disks can have unusable slack. Residual overload
        # is then a valid proof even if the original nominal total was enough.
        if fixed and b["reason"] == "total_excess":
            trace["blocked"]["reason"] = "residual_total_excess"
            trace["search_status"] = "infeasible"
    return finish(trial["ok"], packing)
