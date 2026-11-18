"""Learn a bounded dispatch program on practice data, then evaluate one frozen nominee."""
import hashlib
import math
import random
from functools import reduce

from . import arena, engine
from .experiment import Bench, RULES, ident, validate_task


DEFAULT_BUDGET = 128
MAX_BUDGET = 256


def family(task):
    """Remove mandatory singletons before normalizing units; equivalent views stay together."""
    cap, sizes = task["capacity"], task["items"]
    rest = [s for i, s in enumerate(sizes)
            if s > cap or any(i != j and s + other <= cap for j, other in enumerate(sizes))]
    unit = reduce(math.gcd, rest, cap)
    return (cap // unit, task["bins"] - len(sizes) + len(rest), tuple(sorted(s // unit for s in rest)))


def family_id(task):
    return hashlib.sha256(ident(family(task)).encode()).hexdigest()


def seed_for(episode):
    key = f"agentic-research-v1/{episode}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def dispatch(program, incumbent):
    return [program] + [policy for policy in incumbent if policy.get("kind") != "recipe"]


def curriculum(seed, history, focus):
    bases, used = [], set()
    def add(task, origin):
        if task is None or family(task) in used:
            return
        used.add(family(task))
        bases.append((task, origin))

    add(focus, "用户清单")
    for e in history:
        if len(bases) >= 2:
            break
        if e.get("task") and not e["before"]["ok"]:
            add(e["task"], f"历史记录 #{e['id']}")
    deck = list(zip(arena.cases(seed, "practice"), arena.case_expectations("practice")))
    for profile in ("mixed", "triples", "pairs", "ordered", "loose", "granularity"):
        if len(bases) >= 6:
            break
        add(next(task for task, (name, _) in deck if name == profile), "独立练习")
    tasks, metadata = [], []
    rng = random.Random(seed)
    for task, origin in bases:
        cap, count, sizes = task["capacity"], task["bins"], task["items"]
        factor = max(1, min(rng.choice((32, 64, 128)), 10**9 // max([cap, *sizes])))
        scaled = dict(capacity=cap * factor, bins=count, items=[n * factor for n in sizes])
        variants = [("原始", task), ("换单位", scaled)]
        if count < 16 and len(sizes) < 64:
            pad = scaled["items"][:]
            # A dedicated file one unit short of full can hide a large common factor.
            anchor = scaled["capacity"] - 1 if pad and min(pad) > 1 and scaled["capacity"] > 1 else scaled["capacity"]
            pad.insert(rng.randrange(len(pad) + 1), anchor)
            variants.append(("换单位并加入独占盘", dict(capacity=scaled["capacity"], bins=count + 1, items=pad)))
        emitted = set()
        for name, view in variants:
            if ident(view) not in emitted:
                tasks.append(validate_task(view))
                metadata.append({"source": origin, "view": name})
                emitted.add(ident(view))
    return tasks, metadata


def experience(history, policies, practice):
    skills, sources = [], []
    def remember(policy, source):
        if policy.get("kind") == "recipe":
            policy = policy["solver"]
        if policy not in skills:
            skills.append(policy)
            sources.append(dict(solver=policy, source=source))

    for p in policies:
        remember(p, "现役版本")
    for event in history:
        if event.get("kind") == "solve":
            for trial in event.get("trials", []):
                remember(trial["policy"], f"历史试验 #{event['id']}")
    discovery = Bench()
    # Ask the existing proposer about observed practice failures. Labels and future tests stay out.
    for task in practice:
        base = discovery.single(task, engine.BASE)
        if not base["ok"]:
            for p in engine.propose(task, engine.BASE, base)["candidates"]:
                remember(p, "练习失败轨迹")
    return skills, sources, sum(r["cost"] for r in discovery.cache.values())


def test_sets(seed, practice, prior_input):
    seen = {family_id(t) for t in practice}
    for phase in ("holdout", "regression"):
        tasks = []
        for task, (profile, feasible) in zip(arena.cases(seed, phase), arena.case_expectations(phase)):
            key = family_id(task)
            if key not in seen and not prior_input(key):
                tasks.append((task, profile, feasible))
                seen.add(key)
        yield tasks


def compare(bench, cases, old, new):
    rows = []
    for task, profile, feasible in cases:
        a, b = bench.apply(task, old), bench.apply(task, new)
        rows.append(dict(task=task, profile=profile, feasible=feasible, old_ok=a["ok"], new_ok=b["ok"],
                         old_cost=a["cost"], new_cost=b["cost"]))
    return dict(rows=rows, total=len(rows),
                gains=sum(r["new_ok"] and not r["old_ok"] for r in rows),
                losses=sum(r["old_ok"] and not r["new_ok"] for r in rows),
                **{k: sum(row[k] for row in rows) for k in ("old_ok", "new_ok", "old_cost", "new_cost")})


def judge(hold, regression, replay):
    if hold["total"] < RULES["min_holdout"] or regression["old_ok"] < RULES["min_previous_successes"]:
        return "inconclusive", "独立样本不足。"
    if hold["losses"] + regression["losses"] + replay["losses"] > RULES["allowed_losses"]:
        return "rejected", "配方丢失了已有成功作答。"
    if hold["new_cost"] + regression["new_cost"] > RULES["max_cost_ratio"] * (hold["old_cost"] + regression["old_cost"]):
        return "rejected", "配方与回退链的总操作数超过预设倍率。"
    if hold["gains"] < RULES["min_holdout_gains"]:
        return "inconclusive", "留出组新增成功不足 3 份。"
    return "kept", "一份冻结配方通过独立留出、回归、历史复查与成本门槛。"


def _deliver_practice_answer(data, bench):
    # Only recover a candidate already seen to solve this exact input. No new search or nomination.
    search = data["search"]
    for solved in search["practice_solutions"]:
        if data["practice_tasks"][solved["task_index"]] != data["task"]:
            continue
        program = search["candidates"][solved["candidate"]]["program"]
        calls = len(bench.cache)
        checked = bench.single(data["task"], program)
        if not checked["ok"] or checked["cost"] != solved["cost"]:
            raise RuntimeError("练习记录中的触发清单装法未能复现")
        executed = len(bench.cache) - calls
        data["practice_delivery"] = dict(solved, calls=executed, executed_cost=executed * checked["cost"])
        data["answer"] = dict(source="trial", candidate=solved["candidate"], packing=checked["packing"])
        break


def validate_budget(budget):
    if type(budget) is not int or not 2 <= budget <= MAX_BUDGET:
        raise ValueError(f"研究预算须为 2 到 {MAX_BUDGET} 之间的整数")


def study(policies, generation, history, memory, episode, budget, focus=None, reviewed=None, prior_input=None,
          bench=None):
    from . import planner, recipes
    validate_budget(budget)
    seed = seed_for(episode)
    practice, views = curriculum(seed, history, focus)
    skills, sources, setup_cost = experience(history, policies, practice)
    search = planner.search(practice, policies, skills, memory, seed, budget)
    program = search["program"]
    data = dict(kind="research", episode=episode, seed=seed, budget=budget, generation_before=generation,
                generation=generation, rules=RULES, policies_before=policies, practice_tasks=practice,
                practice_views=views, skill_sources=sources, discovery_cost=setup_cost,
                search=search, nominee=program)
    bench = bench if bench is not None else Bench()
    if focus is not None:
        before = bench.apply(focus, policies)
        data.update(task=focus, before=before, after=before, answer=None)
    if program is None:
        decision, reason = "inconclusive", "练习阶段没有提名可用配方。"
    else:
        recipes.validate(program)
        new = dispatch(program, policies)
        digest = hashlib.sha256(ident(program).encode()).hexdigest()
        data["nominee_sha256"] = digest
        if digest in (reviewed or {}):
            decision, reason = "unchanged", f"同一版本已检验过这份提名，见记录 #{reviewed[digest]}。"
        elif new == policies:
            decision, reason = "unchanged", "研究阶段重新选中了现役配方。"
        elif len(new) > RULES["max_policies"]:
            decision, reason = "budget", "插入配方将超过原策略链长度上限。"
        else:
            # Lock exactly one candidate before opening either acceptance split.
            hold_tasks, reg_tasks = test_sets(seed, practice, prior_input or (lambda _: False))
            hold = compare(bench, hold_tasks, policies, new)
            reg = compare(bench, reg_tasks, policies, new)
            replay_tasks = [e["task"] for e in history if e.get("task") and e.get("answer")][:32]
            replay = compare(bench, [(task, "历史作答", True) for task in replay_tasks], policies, new)
            data.update(holdout=hold, regression=reg, replay=replay)
            decision, reason = judge(hold, reg, replay)
        if focus is not None:
            one = bench.apply(focus, new)
            if not one["ok"] and decision == "kept":
                decision, reason = "inconclusive", "提名配方未解决本轮触发清单。"
            if one["ok"]:
                data["answer"] = dict(source="active" if decision == "kept" else "trial", packing=one["packing"])
            if decision == "kept":
                data["after"] = one
    data.update(decision=decision, reason=reason, validation_calls=len(bench.cache),
                validation_cost=sum(r["cost"] for r in bench.cache.values()))
    if focus is not None:
        if data["after"]["ok"]:
            data["answer"] = dict(source="active", packing=data["after"]["packing"])
        elif data["answer"] is None:
            _deliver_practice_answer(data, bench)
        data["outcome"] = "packed" if data["answer"] else "unresolved"
    return data
