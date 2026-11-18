"""Fixed-bin archive packer. Every policy is a composition of finite primitives."""

BASE = {"order": "arrival", "pick": "first", "repair": "none"}
_ORDERS = {"arrival": "到达顺序", "descending": "大文件优先"}
_PICKS = {"first": "首个可放盘", "tight": "最紧可放盘", "roomy": "最空可放盘"}
_REPAIRS = {
    "none": "直接放置", "exchange": "局部搬移或交换",
    "search": "回退搜索（256 节点）", "deep": "回退搜索（2048 节点）",
}
_TRIPLE_LIMITS = {"triples": 256, "triples_deep": 2048}
for _name, _limit in _TRIPLE_LIMITS.items():
    _REPAIRS[_name] = f"精确三件组搜索（{_limit} 个整盘节点）"
_FILL_LIMITS = {"fill": 4096, "fill_deep": 32768}
for _name, _limit in _FILL_LIMITS.items():
    _REPAIRS[_name] = f"按每盘可用余量组合文件（{_limit} 操作）"
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


def _triple_shape(capacity, count, sizes):
    """One count check, then at most two full scans. Return proxy cost too."""
    if len(sizes) != 3 * count:
        return False, 1
    if sum(sizes) != capacity * count:
        return False, 1 + len(sizes)
    return 4 * min(sizes) > capacity, 1 + 2 * len(sizes)


def _triple_search(order, sizes, capacity, limit, ops):
    """Descending indices. Fix the largest remaining file to break disk symmetry."""
    packing = []
    ops.update(group_nodes=0, pair_probes=0, group_candidates=0,
               duplicate_checks=0, residual_scans=0)

    def visit(remaining):
        if ops["group_nodes"] == limit:
            return "limit"
        ops["group_nodes"] += 1
        if not remaining:
            return "solved"
        anchor = remaining[0]
        target = capacity - sizes[anchor]
        left, right = 1, len(remaining) - 1
        while left < right:
            a, b = remaining[left], remaining[right]
            pair_sum = sizes[a] + sizes[b]
            ops["pair_probes"] += 1
            if pair_sum > target:
                left += 1
            elif pair_sum < target:
                right -= 1
            else:
                ops["group_candidates"] += 1
                group = [anchor, a, b]
                packing.append(group)
                ops["placements"] += 3
                ops["residual_scans"] += len(remaining)
                rest = [i for pos, i in enumerate(remaining) if pos not in (0, left, right)]
                status = visit(rest)
                if status == "solved":
                    return status
                packing.pop()
                ops["undo"] += 3
                if status == "limit":
                    return status
                # Other pairs of these same sizes leave an equivalent multiset.
                left += 1
                right -= 1
                while left < right:
                    ops["duplicate_checks"] += 1
                    if sizes[remaining[left]] != sizes[a]:
                        break
                    left += 1
        return "infeasible"

    status = visit(order)
    return packing if status == "solved" else None, status


def _fill_shape(capacity, bins, sizes):
    # Up to 64 words per subset-sum bitset. Every disk must receive a file.
    return capacity < 4096 and 0 <= bins * capacity - sum(sizes) < capacity


class _FillLimit(Exception):
    pass


def _fill_search(order, sizes, capacity, bins, descending, limit, ops):
    """Each edge selects an entire disk, allowing any size or number of files."""
    remaining_budget = limit
    packing = []

    def charge(key, amount=1):
        nonlocal remaining_budget
        if amount > remaining_budget:
            raise _FillLimit
        ops[key] = ops.get(key, 0) + amount
        remaining_budget -= amount

    def visit(remaining, slots, slack):
        charge("fill_nodes")
        if slots == 1:
            # Invariant: remaining load == capacity - slack, and slack >= 0.
            charge("placements", len(remaining))
            packing.append(remaining)
            return True
        anchor = remaining[0]
        room = capacity - sizes[anchor]
        words = (room + 64) // 64
        n = len(remaining)
        charge("fill_dp", n + 1 + words)
        mask = (1 << (room + 1)) - 1
        suffix = [1] * (n + 1)
        for j in range(n - 1, 0, -1):
            size = sizes[remaining[j]]
            reachable = suffix[j + 1]
            charge("fill_dp")
            if size <= room:
                charge("fill_dp", 3 * words)
                reachable |= (reachable << size) & mask
            suffix[j] = reachable

        def choose(start, need, selected, unused):
            if need == 0:
                charge("fill_candidates")
                charge("residual_scans", n)
                group, rest = [], []
                for k, idx in enumerate(remaining):
                    (group if selected & (1 << k) else rest).append(idx)
                if len(rest) < slots - 1:
                    return False
                charge("placements", len(group))
                packing.append(group)
                if visit(rest, slots - 1, slack - unused):
                    return True
                charge("undo", len(group))
                packing.pop()
                return False

            last_value = None
            for k in range(start, n):
                size = sizes[remaining[k]]
                charge("fill_probes")
                if size > need or size == last_value:
                    continue
                last_value = size
                charge("fill_bit_tests", words)
                if (suffix[k + 1] >> (need - size)) & 1:
                    if choose(k + 1, need - size, selected | (1 << k), unused):
                        return True
            return False

        for unused in range(min(room, slack) + 1):
            need = room - unused
            charge("fill_bit_tests", words)
            if (suffix[1] >> need) & 1 and choose(1, need, 1, unused):
                return True
        return False

    try:
        charge("fill_guard", len(sizes) + 2)
        if not _fill_shape(capacity, bins, sizes):
            return None, "not_applicable"
        if not descending:
            charge("ranking", _rank_charge(len(order)))
            order = sorted(order, key=sizes.__getitem__, reverse=True)
        charge("fill_guard", len(sizes))
        slack = bins * capacity - sum(sizes)
        if visit(order, bins, slack):
            return packing, "solved"
        return None, "infeasible"
    except _FillLimit:
        return None, "limit"


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
                elif reason == "fragmentation" and p["repair"] in _TRIPLE_LIMITS:
                    matches, ops["shape_checks"] = _triple_shape(capacity, count, sizes)
                    if matches:
                        if p["order"] != "descending":
                            ops["ranking"] += _rank_charge(len(order))
                            order.sort(key=sizes.__getitem__, reverse=True)
                        solution, status = _triple_search(order, sizes, capacity,
                                                          _TRIPLE_LIMITS[p["repair"]], ops)
                        trace["search_status"] = status
                        if solution is not None:
                            bins, ok = solution, True
                    else:
                        trace["search_status"] = "not_applicable"
                elif reason == "fragmentation" and p["repair"] in _FILL_LIMITS:
                    solution, status = _fill_search(order, sizes, capacity, count,
                                                    p["order"] == "descending",
                                                    _FILL_LIMITS[p["repair"]], ops)
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
        detail = ("合法的三件组合" if p["repair"] in _TRIPLE_LIMITS else "不同余量的分支")
        if p["repair"] in _FILL_LIMITS:
            detail = "允许闲置量内的整盘组合"
        return {"observation": f"搜索已穷尽{detail}，指定盘数没有可行装法。", "candidates": []}

    free = b["free"]
    observation = (f"下标 {b['index']} 的文件需要 {b['size']}，各盘余量 {free}，"
                   f"合计 {sum(free)}，最大单盘余量 {max(free)}。")
    candidates = []
    triples, _ = _triple_shape(task["capacity"], task["bins"], task["items"])

    # Generalize grouping after a run actually found triples inapplicable.
    # A learned fill policy can also request more work after a recorded cutoff.
    group_unavailable = p["repair"] in _TRIPLE_LIMITS and trace["search_status"] == "not_applicable"
    fill_cutoff = p["repair"] in _FILL_LIMITS and trace["search_status"] == "limit"
    fill = (group_unavailable or fill_cutoff) and not triples and _fill_shape(
        task["capacity"], task["bins"], task["items"])

    def add(**changes):
        candidate = dict(p, **changes)
        if candidate["repair"] in _FILL_LIMITS and not fill:
            return
        if candidate["repair"] in _TRIPLE_LIMITS and not triples:
            return
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
        if p["repair"] in _FILL_LIMITS:
            add(repair="fill_deep")
        else:
            add(repair="triples_deep" if p["repair"] in _TRIPLE_LIMITS else "deep")
    else:
        observation += "可在节点预算内重选较早的放置位置。"
        add(repair="search")
    if triples:
        observation += "总量恰满，文件数是盘数的三倍，每件都超过四分之一盘；可按恰好装满的三件组重排。"
        add(repair="triples")
    if fill:
        if group_unavailable:
            observation += "这次已试的三件组修复不适用于这张清单。"
        slack = task["bins"] * task["capacity"] - sum(task["items"])
        observation += (f"整份清单最多闲置 {slack}，每盘至少要装 "
                        f"{task['capacity'] - slack}；可按这个区间重选每盘的文件组合。")
        add(repair="fill")
    return {"observation": observation, "candidates": candidates}
