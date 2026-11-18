"""Bounded online selection and synthesis of executable packing recipes."""

from dataclasses import dataclass, field
import hashlib
import json
import math
import random

from . import arena, engine, recipes


ARCHIVE_LIMIT = 8
POLICY_LIMIT = 16
STAT_LIMIT = 192
CACHE_LIMIT = 192
MEMORY_VERSION = 1
SOLVER_EDITS = {"order": ("arrival", "descending"), "pick": ("first", "tight", "roomy"),
                "repair": ("none", "exchange", "search")}


def _key(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:24]


def _recipe(solver, steps=(), guard=None):
    return {"kind": "recipe", "steps": list(steps), "solver": solver, "guard": guard}


@dataclass
class Candidate:
    program: dict
    parent: str
    direction: str
    origin: str
    predicted: int = 0
    id: str = field(init=False)

    def __post_init__(self):
        self.id = _key(self.program)

    def metadata(self):
        return dict(id=self.id, program=self.program, parent=self.parent,
                    direction=self.direction, origin=self.origin)


def _is_recipe(policy):
    return policy.get("kind") == "recipe"


def _ordinary(policy):
    solver = {**engine.BASE, **policy}
    engine.describe(solver)
    return solver


def grammar(policies, experience, memory=None):
    """Return the shared seed grammar before per-episode guard induction."""
    nodes = {}

    def add(program, parent, direction, origin):
        recipes.validate(program)
        node = Candidate(program, parent, direction, origin)
        return nodes.setdefault(node.id, node)

    def recall(raw):
        p = raw["program"]
        return add(p, raw["parent"], raw["direction"], "archive")

    roots = []
    for policy in policies:
        if _is_recipe(policy):
            roots.append(add(policy, "", "active-recipe", "active"))
    for raw in (memory or {}).get("archive", [])[:ARCHIVE_LIMIT]:
        roots.append(recall(raw))

    ordinary = {}
    for origin, pool in (("active", [p for p in policies if not _is_recipe(p)]),
                         ("experience", experience)):
        for policy in pool:
            solver = _ordinary(policy)
            ordinary.setdefault(_key(solver), (solver, origin))
    ordinary.setdefault(_key(engine.BASE), (engine.BASE.copy(), "baseline"))
    for key, (solver, origin) in list(ordinary.items())[:POLICY_LIMIT]:
        root = add(_recipe(solver), key, "recall-solver", origin)
        roots.append(root)
        for axis, values in SOLVER_EDITS.items():
            for value in values:
                if value != solver[axis]:
                    roots.append(add(_recipe({**solver, axis: value}), root.id,
                                     f"solver:{axis}={value}", origin))

    for root in roots:
        program = root.program
        steps, solver = program["steps"], program["solver"]
        if program["guard"] is not None:
            add(_recipe(solver, steps), root.id, "remove-guard", root.origin)
        if steps:
            for step in steps:
                edited = [old for old in steps if old != step]
                add(_recipe(solver, edited, program["guard"]), root.id,
                    f"remove:{step}", root.origin)
            if len(steps) > 1:
                add(_recipe(solver, reversed(steps), program["guard"]), root.id,
                    "reorder-steps", root.origin)
        for step in recipes.STEPS:
            if step not in steps:
                child = add(_recipe(solver, [*steps, step], program["guard"]), root.id,
                            f"append:{step}", root.origin)
                for second in recipes.STEPS:
                    if second != step and second not in steps:
                        add(_recipe(solver, [*steps, step, second], program["guard"]), child.id,
                            f"append:{second}", root.origin)

    active = {_key(p) for p in policies if _is_recipe(p)}
    return [node.metadata() for node in nodes.values() if node.id not in active]


def _proof(result):
    trace = result["trace"]
    blocked = trace.get("blocked") or {}
    return (blocked.get("reason") in ("oversize", "total_excess")
            or trace.get("search_status") == "infeasible")


def _checked(task, policy):
    run = recipes.run if _is_recipe(policy) else engine.run
    result = run(task, policy)
    if (type(result["ok"]) is not bool or type(result["cost"]) is not int
            or result["cost"] < 1 or (result["ok"] and not arena.check(task, result["packing"]))):
        raise RuntimeError("invalid recipe or incumbent execution evidence")
    return result


def _raw(result):
    guard = result["trace"].get("guard") or {}
    return dict(ok=result["ok"], cost=result["cost"], proof=_proof(result),
                matched=guard.get("matched"), observed=guard.get("observed"))


def _installed(task, policies):
    cost = 0
    for policy in policies:
        raw = _raw(_checked(task, policy))
        cost += raw["cost"]
        if raw["ok"] or raw["proof"]:
            break
    return {**raw, "cost": cost}


def _with_fallback(raw, ordinary):
    if raw["ok"] or raw["proof"]:
        return raw
    return {**ordinary, "cost": raw["cost"] + ordinary["cost"]}


def _utility(old_ok, old_cost, ok, cost):
    relative = 1000 * (old_cost - cost) // (old_cost + cost)
    return 2000 * (ok and not old_ok) - 3000 * (old_ok and not ok) + relative


def _context(values):
    return (values["gcd"] > 1, values["forced_singletons"] > 0,
            values["items_per_bin_permille"] // 1000, values["load_permille"] // 200)


def _packed_stat_keys(node, context):
    return (f"recipe:{node.id}", _key(["direction", node.direction]),
            _key(["context", node.direction, context]))


def _restore_entries(memory, name):
    # Start every episode in the same key order as a persisted SQLite snapshot.
    return dict(sorted(memory.get(name, {}).items()))


class Episode:
    def __init__(self, tasks, policies, indices, memory, budget, mode):
        self.tasks = tasks
        self.indices = indices
        self.budget = budget
        self.mode = mode
        self.views = {i: recipes.features(tasks[i]) for i in indices}
        self.keys = {i: _key(tasks[i]) for i in indices}
        history = {} if mode == "adaptive_reset" else memory
        self.stats = {key: list(pair) for key, pair in _restore_entries(history, "stats").items()}
        self.recent = _restore_entries(history, "recent")
        self.raw_keys = {}
        self.rows = {}
        self.trials = []
        self.cached_pairs = 0
        self.candidate_spent = 0
        self.baseline = []
        self.old = {}
        self.ordinary = {}
        ordinary = [p for p in policies if not _is_recipe(p)] or [engine.BASE]
        active = next((p for p in policies if _is_recipe(p)), None)
        for index in indices:
            task = tasks[index]
            fallback = _installed(task, ordinary)
            self.ordinary[index] = fallback
            self.baseline.append(dict(task_index=index, role="ordinary", **fallback))
            if active is None:
                self.old[index] = fallback
                continue
            raw = _raw(_checked(task, active))
            self.baseline.append(dict(task_index=index, role="active-recipe", **raw))
            self.old[index] = _with_fallback(raw, fallback)

    @property
    def remaining(self):
        return self.budget - len(self.baseline) - len(self.trials)

    def raw_key(self, node, index):
        return self.raw_keys.setdefault((node.id, index), f"{node.id}:{self.keys[index]}")

    def evaluate(self, node, index, stage, reason):
        rows = self.rows.setdefault(node.id, {})
        if index in rows:
            return
        key = self.raw_key(node, index)
        previous = self.recent.pop(key, None)
        if previous is None:
            if self.remaining <= 0:
                raise RuntimeError("candidate evaluation would exceed episode budget")
            raw = _raw(_checked(self.tasks[index], node.program))
        else:
            raw = previous
            self.cached_pairs += 1
        self.recent[key] = raw
        old = self.old[index]
        new = _with_fallback(raw, self.ordinary[index])
        reward = _utility(old["ok"], old["cost"], new["ok"], new["cost"])
        row = dict(candidate=node.id, parent=node.parent, direction=node.direction,
                   origin=node.origin, guard=node.program["guard"], task_index=index,
                   stage=stage, reason=reason, raw_ok=raw["ok"], matched=raw["matched"],
                   observed=raw["observed"], old_ok=old["ok"], ok=new["ok"], old_cost=old["cost"],
                   cost=new["cost"], spent=raw["cost"], reward=reward)
        rows[index] = row
        if previous is None:
            self.trials.append(row)
            self.candidate_spent += raw["cost"]
            for stat in _packed_stat_keys(node, _context(self.views[index])):
                count, total = self.stats.pop(stat, (0, 0))
                if count >= 16:
                    total = total * 15 // 16
                    count = 15
                self.stats[stat] = [count + 1, total + reward]

    def priority(self, node, probes):
        values = self.views[probes[0]]
        own, direction, context = _packed_stat_keys(node, _context(values))
        weighted = node.predicted
        weight = 1 if node.predicted else 0
        for key, cap in ((own, 4), (f"recipe:{node.parent}", 2), (direction, 2), (context, 2)):
            count, total = self.stats.get(key, (0, 0))
            if count:
                share = min(cap, count)
                weighted += share * total / count
                weight += share
        # A representation change needs some input to change. Still explore
        # every edit, including orderings whose prerequisites emerge mid-run.
        steps = node.program["steps"]
        applicability = 35 * (("peel_forced" in steps and values["forced_singletons"] > 0)
                              + ("normalize_gcd" in steps and values["gcd"] > 1))
        score = weighted / (1 + weight) + (350 + applicability) / math.sqrt(1 + weight)
        return score, weight

    def work(self, node):
        rows = self.rows.get(node.id, {})
        return [i for i in self.indices if i not in rows]

    def price(self, node, indices):
        return sum(self.raw_key(node, i) not in self.recent for i in indices)


def _induce(nodes, episode):
    """Fit at most one threshold per measured unguarded candidate."""
    for node in nodes:
        rows = episode.rows.get(node.id, {})
        if node.program["guard"] is not None or len(rows) < 2:
            continue
        rewards = {i: row["reward"] for i, row in rows.items()}
        if min(rewards.values()) >= 0 or max(rewards.values()) <= 0:
            continue
        reference = max(0, sum(rewards.values()))
        winner = None
        for feature in sorted(episode.views[next(iter(rows))]):
            values = {i: episode.views[i][feature] for i in rows}
            distinct = sorted(set(values.values()))
            for a, b in zip(distinct, distinct[1:]):
                cut = (a + b) // 2
                for op, predicate, value in (("le", lambda x: x <= cut, cut),
                                              ("ge", lambda x: x > cut, cut + 1)):
                    routed = [i for i in rows if predicate(values[i])]
                    score = sum(rewards[i] for i in routed)
                    if score <= reference:
                        continue
                    rank = (score, len(routed), -len(feature))
                    if winner is None or rank > winner[0]:
                        winner = rank, dict(feature=feature, op=op, value=value)
        if winner is not None:
            rank, guard = winner
            program = {**node.program, "guard": guard}
            yield Candidate(program, node.id, f"learn-guard:{guard['feature']}",
                            "measured", rank[0] // len(rows))


def _candidate_score(rows):
    gains = sum(row["ok"] and not row["old_ok"] for row in rows)
    losses = sum(row["old_ok"] and not row["ok"] for row in rows)
    old_cost = sum(row["old_cost"] for row in rows)
    cost = sum(row["cost"] for row in rows)
    utility = sum(row["reward"] for row in rows)
    return dict(gains=gains, losses=losses, old_cost=old_cost, new_cost=cost,
                utility=utility, total=len(rows))


def _worth_review(score):
    if score["losses"]:
        return False
    if score["gains"]:
        return 4 * score["new_cost"] <= 9 * score["old_cost"]
    return score["new_cost"] < score["old_cost"]


def _practice_solutions(rows):
    best = {}
    for candidate, results in rows.items():
        for index, row in results.items():
            if row["raw_ok"]:
                choice = (row["spent"], candidate)
                if index not in best or choice < best[index]:
                    best[index] = choice
    return [dict(task_index=i, candidate=best[i][1], cost=best[i][0]) for i in sorted(best)]


def search(practice, policies, experience, memory, seed, budget, mode="adaptive"):
    """Return the sole practice nominee, resumable memory, and exact call accounting.

    A call is an ordinary incumbent-path, active-recipe, or candidate evaluation
    on one practice task. A fallback uses the shared ordinary result. ``spent``
    adds real solver operation costs, including the shared baseline work.
    """
    if mode not in ("adaptive", "adaptive_reset", "uniform"):
        raise ValueError("unknown planner mode")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    memory = memory if memory and memory.get("version") == MEMORY_VERSION else {}
    seeds = grammar(policies, experience, memory)
    nodes = {row["id"]: Candidate(row["program"], row["parent"], row["direction"], row["origin"])
             for row in seeds}
    rng = random.Random(seed)
    unique_indices = {}
    for index, task in enumerate(practice):
        unique_indices.setdefault(_key(task), index)
    baseline_per_task = 1 + any(_is_recipe(p) for p in policies)
    count = min(len(unique_indices), budget // (baseline_per_task + 1)) if nodes else 0
    indices = sorted(rng.sample(list(unique_indices.values()), count))
    episode = Episode(practice, policies, indices, memory, budget, mode)
    ties = {key: rng.random() for key in nodes}
    finalist = {}
    active = {_key(p) for p in policies if _is_recipe(p)}

    def choose(pool, probes):
        if mode == "uniform":
            key = min(pool, key=ties.__getitem__)
            return key, "固定种子下为未完成候选分配独立随机次序。"
        key = max(pool, key=lambda option: (episode.priority(nodes[option], probes)[0], -ties[option]))
        score, weight = episode.priority(nodes[key], probes)
        node = nodes[key]
        return key, (f"由父版本 {node.parent or '初始根'} 沿 {node.direction} 试验，"
                     f"配对收益优先分 {score:.1f}，证据权重 {weight}。")

    def learn_guards(parents):
        added = []
        for node in _induce(parents, episode):
            if node.id not in nodes and node.id not in active:
                recipes.validate(node.program)
                nodes[node.id] = node
                ties[node.id] = rng.random()
                added.append(node)
        return added

    if indices:
        # Start with cheap measurements from both easy and blocked incumbent cases.
        cheap = sorted(indices, key=lambda i: episode.old[i]["cost"])
        blocked = [i for i in cheap if not episode.old[i]["ok"] and not episode.old[i]["proof"]]
        solved = [i for i in cheap if episode.old[i]["ok"]]
        probes = list(dict.fromkeys(blocked[:1] + solved[:1] + blocked + cheap))[:min(3, len(indices))]
        screen_target = min(episode.remaining // 2, max(0, episode.remaining - len(indices)))
        pool = list(nodes)
        screen_start = episode.remaining
        while pool and screen_start - episode.remaining < screen_target:
            key, reason = choose(pool, probes)
            pool.remove(key)
            for index in probes:
                if screen_start - episode.remaining == screen_target:
                    break
                episode.evaluate(nodes[key], index, "probe", reason)

        learn_guards(list(nodes.values()))

        pending = {key: episode.work(node) for key, node in nodes.items()}
        while pending:
            complete = {key: missing for key, missing in pending.items()
                        if episode.price(nodes[key], missing) <= episode.remaining}
            if complete:
                key, reason = choose(complete, probes)
                missing = pending.pop(key)
                for index in missing:
                    episode.evaluate(nodes[key], index, "compare", reason)
                finalist[key] = _candidate_score(list(episode.rows[key].values()))
                for guard in learn_guards([nodes[key]]):
                    pending[guard.id] = episode.work(guard)
                continue
            if not episode.remaining:
                break
            key, reason = choose(pending, probes)
            missing = pending.pop(key)
            for index in missing:
                if not episode.remaining:
                    break
                episode.evaluate(nodes[key], index, "probe-remainder", reason)

    eligible = {key: score for key, score in finalist.items() if _worth_review(score)}

    def rank(key):
        score = eligible[key]
        return (score["gains"], -score["new_cost"], score["utility"], -len(nodes[key].program["steps"]), -ties[key])

    winners = sorted(eligible, key=rank, reverse=True)
    nominee = winners[0] if winners else None
    archive = {key: nodes[key].metadata() for key in winners}
    # Keep useful local exceptions even when they did not win a whole panel.
    useful = sorted((key for key, rows in episode.rows.items() if any(r["reward"] > 0 for r in rows.values())),
                    key=lambda key: max(r["reward"] for r in episode.rows[key].values()), reverse=True)
    for key in useful:
        archive.setdefault(key, nodes[key].metadata())
    for row in memory.get("archive", ()):
        archive.setdefault(row["id"], row)
    baseline_spent = sum(row["cost"] for row in episode.baseline)
    stop = "budget" if episode.remaining == 0 or (not indices and practice and nodes) else "exhausted"
    # Within this episode new evidence and cache hits move to the end. Retain
    # those entries first; the next episode canonicalizes the surviving maps.
    return dict(program=nodes[nominee].program if nominee else None, nominee=nominee,
                memory=dict(version=MEMORY_VERSION, archive=list(archive.values())[:ARCHIVE_LIMIT],
                            stats=dict(list(episode.stats.items())[-STAT_LIMIT:]),
                            recent=dict(list(episode.recent.items())[-CACHE_LIMIT:])),
                trials=episode.trials, candidates={key: node.metadata() for key, node in nodes.items()},
                practice_solutions=_practice_solutions(episode.rows),
                calls=len(episode.baseline) + len(episode.trials), spent=baseline_spent + episode.candidate_spent,
                baseline_calls=len(episode.baseline), baseline_spent=baseline_spent, baseline=episode.baseline,
                candidate_calls=len(episode.trials), candidate_spent=episode.candidate_spent,
                cached_pairs=episode.cached_pairs, budget=budget, mode=mode, stop_reason=stop,
                practice_indices=indices, comparisons=finalist)
