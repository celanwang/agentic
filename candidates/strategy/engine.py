"""Fixed-bin archive packer. Every policy is a composition of finite primitives."""

BASE = {"order": "arrival", "pick": "first", "repair": "none"}
_ORDERS = {"arrival": "到达顺序", "descending": "大文件优先"}
_PICKS = {"first": "首个可放盘", "tight": "最紧可放盘", "roomy": "最空可放盘"}
_REPAIRS = {
    "none": "直接放置", "exchange": "局部搬移或交换",
    "search": "回退搜索（256 节点）", "deep": "回退搜索（2048 节点）",
}
_EXCHANGE_LIMIT = 512


def _policy(policy):
    p = BASE.copy()
    if policy is None:
        return p
    if set(policy) - set(p):
        raise ValueError("unknown policy key")
    p.update(policy)
    if p["order"] not in _ORDERS or p["pick"] not in _PICKS or p["repair"] not in _REPAIRS:
        raise ValueError("unknown policy primitive")
    return p


def describe(policy):
    p = _policy(policy)
    return "、".join((_ORDERS[p["order"]], _PICKS[p["pick"]], _REPAIRS[p["repair"]]))


def _rank_charge(n):
    return n * (n - 1).bit_length() if n > 1 else 0


def _pick(free, size, mode, ops):
    choice = None
    for j, gap in enumerate(free):
        ops["bin_probes"] += 1
        if gap < size:
            continue
        if mode == "first":
            return j
        if choice is None or (gap < free[choice] if mode == "tight" else gap > free[choice]):
            choice = j
    return choice


def _exchange(idx, sizes, packing, free, ops):
    """Move one resident or swap two so the blocked file fits on the donor."""
    for donor, residents in enumerate(packing):
        for pos, old in enumerate(residents):
            released = free[donor] + sizes[old]
            for other, members in enumerate(packing):
                if other == donor:
                    continue
                if ops["exchange_probes"] >= _EXCHANGE_LIMIT:
                    return None
                ops["exchange_probes"] += 1
                if released < sizes[idx]:
                    continue
                if sizes[old] <= free[other]:
                    residents.pop(pos)
                    members.append(old)
                    free[donor] = released
                    free[other] -= sizes[old]
                    ops["placements"] += 1
                    return donor
                for k, peer in enumerate(members):
                    if ops["exchange_probes"] >= _EXCHANGE_LIMIT:
                        return None
                    ops["exchange_probes"] += 1
                    net = sizes[old] - sizes[peer]
                    if free[donor] + net >= sizes[idx] and free[other] >= net:
                        residents[pos], members[k] = peer, old
                        free[donor] += net
                        free[other] -= net
                        ops["placements"] += 2
                        return donor
    return None


def _search(order, sizes, capacity, count, mode, limit, ops):
    """Try distinct residual capacities at each depth. Budget spans the whole tree."""
    bins = [[] for _ in range(count)]
    free = [capacity] * count
    used = 0

    def visit(depth):
        nonlocal used
        if used == limit:
            return "limit"
        used += 1
        ops["search_nodes"] += 1
        if depth == len(order):
            return "solved"
        idx = order[depth]
        size = sizes[idx]
        options = []
        seen = set()
        for j, gap in enumerate(free):
            ops["bin_probes"] += 1
            if gap >= size and gap not in seen:
                seen.add(gap)
                options.append(j)
        if mode != "first":
            ops["ranking"] += _rank_charge(len(options))
            options.sort(key=free.__getitem__, reverse=mode == "roomy")
        for j in options:
            bins[j].append(idx)
            free[j] -= size
            ops["placements"] += 1
            state = visit(depth + 1)
            if state == "solved":
                return state
            bins[j].pop()
            free[j] += size
            ops["undo"] += 1
            if state == "limit":
                return state
        return "infeasible"

    status = visit(0)
    return bins if status == "solved" else None, status


def run(task, policy=None):
    """Return a valid partial packing on failure, retaining original file indices."""
    capacity, count, sizes = task["capacity"], task["bins"], task["items"]
    if type(capacity) is not int or capacity < 1 or type(count) is not int or not 1 <= count <= 16:
        raise ValueError("expected positive integer capacity and 1..16 bins")
    if len(sizes) > 64 or any(type(s) is not int or s < 1 for s in sizes):
        raise ValueError("expected at most 64 positive integer sizes")
    p = _policy(policy)
    order = list(range(len(sizes)))
    bins = [[] for _ in range(count)]
    free = [capacity] * count
    ops = dict(setup=1 + count + len(order), ranking=0, bin_probes=0,
               placements=0, exchange_probes=0, search_nodes=0, undo=0, diagnosis=0)
    if p["order"] == "descending":
        ops["ranking"] += _rank_charge(len(order))
        order.sort(key=sizes.__getitem__, reverse=True)

    trace = {"blocked": None, "repaired": [], "search_status": None, "ops": ops}
    ok = True
    for step, idx in enumerate(order):
        size = sizes[idx]
        j = _pick(free, size, p["pick"], ops)
        if j is None:
            ops["diagnosis"] += len(sizes) + len(free) + step
            if max(sizes) > capacity:
                reason = "oversize"
            elif sum(sizes) > capacity * count:
                reason = "total_excess"
            else:
                reason = "fragmentation"
            trace["blocked"] = {
                "reason": reason, "index": idx, "size": size, "step": step,
                "free": free.copy(), "smaller_before": sum(sizes[k] < size for k in order[:step]),
            }
            if reason == "fragmentation" and p["repair"] == "exchange":
                j = _exchange(idx, sizes, bins, free, ops)
                if j is not None:
                    trace["repaired"].append({"index": idx, "bin": j})
            if j is None:
                ok = False
                if reason == "fragmentation" and p["repair"] in ("search", "deep"):
                    limit = 256 if p["repair"] == "search" else 2048
                    solution, status = _search(order, sizes, capacity, count, p["pick"], limit, ops)
                    trace["search_status"] = status
                    if solution is not None:
                        bins, ok = solution, True
                break
        bins[j].append(idx)
        free[j] -= size
        ops["placements"] += 1
    return {"ok": ok, "packing": bins, "cost": sum(ops.values()), "trace": trace}


def propose(task, policy, result):
    """Read one trial's measured obstruction and construct neighboring policies."""
    p = _policy(policy)
    if result["ok"]:
        return {"observation": "所有文件已装入指定盘数。", "candidates": []}
    trace = result["trace"]
    b = trace["blocked"]
    if b["reason"] == "oversize":
        observation = f"最大文件大小 {max(task['items'])} 超过单盘容量 {task['capacity']}。"
        return {"observation": observation, "candidates": []}
    if b["reason"] == "total_excess":
        observation = f"文件总量 {sum(task['items'])} 超过总容量 {task['bins'] * task['capacity']}。"
        return {"observation": observation, "candidates": []}
    if trace["search_status"] == "infeasible":
        return {"observation": "搜索已穷尽不同余量的分支，指定盘数没有可行装法。", "candidates": []}

    free = b["free"]
    observation = (f"下标 {b['index']} 的文件需要 {b['size']}，各盘余量 {free}，"
                   f"合计 {sum(free)}，最大单盘余量 {max(free)}。")
    candidates = []

    def add(**changes):
        candidate = dict(p, **changes)
        if candidate != p and candidate not in candidates:
            candidates.append(candidate)

    if b["smaller_before"] and p["order"] == "arrival":
        observation += f"此前已放入 {b['smaller_before']} 个较小文件，可先放大文件。"
        add(order="descending")
        add(order="descending", pick="tight")
    if b["step"] >= 2 and len(free) >= 2:
        observation += "改变选盘顺序可能保留较大的连续余量，还可搬移或交换已放文件。"
        add(pick="tight")
        add(pick="roomy")
        add(repair="exchange")
    if trace["search_status"] == "limit":
        observation += "回退搜索已达本轮节点上限。"
        add(repair="deep")
    else:
        observation += "可在节点预算内重选较早的放置位置。"
        add(repair="search")
    return {"observation": observation, "candidates": candidates}
