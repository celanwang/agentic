"""Evaluate pair evidence on the published 40 tasks and same-size controls."""

from collections import defaultdict
import json
from random import Random

from candidates.audit.pair_boundary import manifests
from packlab.recipes import route_features
from .seal import check, core, exact_pairs, matched, stats


SEED = 2026111811


def coverage(task):
    """Count nonoverlapping complements via a sorted two-pointer scan."""
    items = sorted(task["items"])
    left, right, pairs = 0, len(items)-1, 0
    while left < right:
        combined = items[left] + items[right]
        if combined == task["capacity"]:
            pairs += 1
            left += 1
            right -= 1
        elif combined < task["capacity"]:
            left += 1
        else:
            right -= 1
    return 2000 * pairs // len(items) if items else 0


def first_fit(task, descending):
    items = task["items"]
    order = range(len(items))
    if descending:
        order = sorted(order, key=items.__getitem__, reverse=True)
    free = [task["capacity"]] * task["bins"]
    groups = [[] for _ in free]
    for index in order:
        for disk, room in enumerate(free):
            if items[index] <= room:
                free[disk] -= items[index]
                groups[disk].append(index)
                break
        else:
            return False
    assert check(task, groups)
    return True


def observations(task, witness, known):
    assert check(task, witness)
    cap, bins, items = (task[k] for k in ("capacity", "bins", "items"))
    effective, unit_cap, sizes = core(task)
    exact = exact_pairs(unit_cap, effective, sizes)
    half = len(sizes) // 2
    by_sort = len(sizes) == 2*effective and all(sizes[i] + sizes[-i-1] == unit_cap for i in range(half))
    assert by_sort == exact == known
    score = coverage(task)
    assert route_features(task)["complementary_coverage"] == score
    return {"known_exact_pairs_after_peel": known, "legacy_bins_le_4": bins <= 4,
            "raw_fill_and_two_items_per_bin": sum(items) == bins*cap and len(items) == 2*bins,
            "raw_near_full": 1000*sum(items) >= 999*bins*cap,
            "raw_exact_pairs": exact_pairs(cap, bins, items), "peeled_exact_pairs": exact,
            "arrival_first_fit": first_fit(task, False), "descending_first_fit": first_fit(task, True)}, score


def run():
    rng = Random(SEED)
    by_size = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals = defaultdict(lambda: defaultdict(int))
    count = 0
    for original, witness in manifests():
        bins, cap = original["bins"], original["capacity"]
        if bins < 5:
            continue
        count += 1
        near = dict(original, items=original["items"][:])
        near["items"][-1] -= 1
        pair, mix = matched(rng, cap, bins)
        assert stats(pair[0]) == stats(mix[0])
        data = (("historical_strict", original, witness, True), ("near", near, witness, False),
                ("matched_pair", *pair, True), ("matched_mixed", *mix, False))
        for family, task, answer, exact in data:
            flags, score = observations(task, answer, exact)
            for row in (by_size[bins][family], totals[family]):
                row["cases"] += 1
                for name, passed in flags.items():
                    row[name] += passed
                row.setdefault("raw_coverage_histogram", defaultdict(int))[score] += 1
    assert count == 40
    return {"control_seed": SEED, "historical_cases": count, "totals": totals, "by_bins": by_size}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
