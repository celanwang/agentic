"""Reproducible bin-packing tasks and an independent solution checker.

Use practice (16 tasks) to select a single candidate. Commit the evaluation
seed and decision rule before opening holdout (64) or regression (20), then
use each of those splits once. Keep the candidate fixed across both splits.
Each generated task gets its own seed derived from the run seed, split,
profile, and occurrence. Fixed regression fixtures are seed-independent.
All tasks have at most 64 files and at most 16 disks.

Feasibility certificates by profile:
* pairs and triples partition into exact-capacity disks before shuffling.
* mixed, loose, and tiny also start from legal disks, with varying slack,
  file counts, or size distributions. Shuffling preserves that witness.
* ordered and ordered_regression fill one disk at a time in arrival order,
  so first-fit succeeds. Extra disks are empty. The nine ordered_regression
  cases contain a full-disk file and have distinct total loads in disk units.
  replay, empty, and one_disk have explicit witnesses.
* overload exceeds total capacity. oversized has a file larger than a disk.
* large_count has more than k * bins files larger than capacity / (k + 1),
  so at most k of those files fit on any disk, despite adequate total space.
* granularity uses multiples of a common unit and requires more space than
  bins * (capacity // unit) * unit, despite adequate nominal total space.
* subset_gap fills the nominal capacity of exactly two disks. An independent
  subset-sum bitset proves that a full disk cannot be formed.

The quotas represent a stratified stress test. Generalization is limited by
planted feasible cases, tractable impossibility certificates, and the file
count cap. Estimating production success rates requires a workload sample.
"""

from hashlib import sha256
from math import gcd
from random import Random


_QUOTAS = {
    "practice": (
        ("pairs", 2), ("triples", 3), ("mixed", 3), ("loose", 1),
        ("tiny", 1), ("ordered", 2), ("overload", 1), ("large_count", 1),
        ("granularity", 1), ("subset_gap", 1),
    ),
    "holdout": (
        ("pairs", 8), ("triples", 12), ("mixed", 11), ("loose", 7),
        ("tiny", 5), ("ordered", 7), ("overload", 3), ("large_count", 3),
        ("granularity", 3), ("subset_gap", 4), ("oversized", 1),
    ),
    "regression": (
        ("replay", 1), ("empty", 1), ("one_disk", 1), ("ordered_regression", 9),
        ("pairs", 1), ("triples", 2), ("mixed", 1), ("overload", 1),
        ("large_count", 1), ("granularity", 1), ("subset_gap", 1),
    ),
}

_IMPOSSIBLE = {"overload", "large_count", "granularity", "subset_gap", "oversized"}
_MAX_FILES = 64


def _rng(seed, phase, profile, index):
    key = f"agentic-arena-v1/{phase}/{profile}/{index}/{seed}".encode("ascii")
    return Random(int.from_bytes(sha256(key).digest(), "big"))


def _task(capacity, bins, items):
    return {"capacity": capacity, "bins": bins, "items": list(items)}


def _parts(rng, total, count):
    """Split total into count positive parts using distinct internal cuts."""
    cuts = [0, *sorted(rng.sample(range(1, total), count - 1)), total]
    return [right - left for left, right in zip(cuts, cuts[1:])]


def _shuffled(rng, capacity, disks):
    items = [size for disk in disks for size in disk]
    rng.shuffle(items)
    return _task(capacity, len(disks), items)


def _pairs(rng):
    capacity = rng.randint(64, 220)
    bins = rng.randint(2, 13)
    small = [rng.randint(capacity // 4 + 1, (capacity - 1) // 2) for _ in range(bins)]
    return _shuffled(rng, capacity, [(size, capacity - size) for size in small])


def _triples(rng):
    capacity = rng.randint(80, 185)
    bins = rng.randint(3, 10)
    low = capacity // 4 + 1
    high = (capacity - 1) // 2
    disks = []
    for _ in range(bins):
        first = rng.randint(low, min(high, capacity - 2 * low))
        second = rng.randint(max(low, capacity - first - high),
                             min(high, capacity - first - low))
        disks.append([first, second, capacity - first - second])
    return _shuffled(rng, capacity, disks)


def _mixed(rng):
    capacity = rng.randint(110, 300)
    bins = rng.randint(2, 12)
    max_slack = rng.choice((0, 2, capacity // 18))
    max_parts = min(7, _MAX_FILES // bins - 1)
    disks = []
    for _ in range(bins):
        load = capacity - rng.randint(0, max_slack)
        anchor = rng.randint(capacity // 8, 4 * capacity // 5)
        disks.append([anchor, *_parts(rng, load - anchor, rng.randint(2, max_parts))])
    return _shuffled(rng, capacity, disks)


def _loose(rng):
    capacity = rng.randint(75, 240)
    bins = rng.randint(2, 10)
    max_parts = min(8, _MAX_FILES // bins)
    disks = []
    for _ in range(bins):
        load = rng.randint(3 * capacity // 5, 4 * capacity // 5)
        disks.append(_parts(rng, load, rng.randint(2, max_parts)))
    return _shuffled(rng, capacity, disks)


def _tiny(rng):
    capacity = rng.randint(75, 160)
    bins = rng.randint(3, 10)
    max_tiny_files = _MAX_FILES // bins - 1
    max_tiny_size = 7
    disks = []
    for _ in range(bins):
        load = capacity - rng.randint(0, 3)
        anchor = rng.randint(max(capacity // 2, load - max_tiny_files * max_tiny_size),
                             5 * capacity // 6)
        remaining = load - anchor
        min_tiny_files = (remaining + max_tiny_size - 1) // max_tiny_size
        tiny_files = rng.randint(min_tiny_files, min(remaining, max_tiny_files))
        disk = [anchor]
        for following_files in range(tiny_files - 1, -1, -1):
            size = rng.randint(max(1, remaining - following_files * max_tiny_size),
                               min(max_tiny_size, remaining - following_files))
            disk.append(size)
            remaining -= size
        disks.append(disk)
    return _shuffled(rng, capacity, disks)


def _ordered(rng, used=None):
    capacity = rng.randint(65, 200)
    if used is None:
        used = rng.randint(2, 10)
        disks = []
    else:
        disks = [[capacity]]
    split_disks = used - len(disks)
    max_parts = min(8, (_MAX_FILES - len(disks)) // split_disks)
    disks.extend(_parts(rng, capacity, rng.randint(2, max_parts)) for _ in range(split_disks))
    rng.shuffle(disks)
    items = [size for disk in disks for size in disk]
    return _task(capacity, used + rng.randint(0, 3), items)


def _overload(rng):
    capacity = rng.randint(70, 190)
    bins = rng.randint(2, 10)
    disks = [_parts(rng, capacity, rng.randint(2, 6)) for _ in range(bins)]
    disks[-1].append(rng.randint(1, capacity // 10))
    return _shuffled(rng, capacity, disks)


def _large_count(rng):
    capacity = rng.randint(80, 220)
    bins = rng.randint(2, 10)
    per_disk = rng.randint(1, 3)
    count = per_disk * bins + 1
    low = capacity // (per_disk + 1) + 1
    high = capacity * bins // count
    items = [rng.randint(low, high) for _ in range(count)]
    return _task(capacity, bins, items)


def _granularity(rng):
    bins = rng.randint(3, 10)
    unit = rng.randint(3, 9)
    usable_units = rng.randint(15, 40)
    remainder = rng.randint((unit + bins - 1) // bins, unit - 1)
    capacity = usable_units * unit + remainder
    disks = []
    for _ in range(bins):
        disks.append([part * unit for part in _parts(rng, usable_units, rng.randint(2, 6))])
    disks[-1].append(unit)
    return _shuffled(rng, capacity, disks)


def _subset_gap(rng):
    capacity = rng.randint(70, 160)
    for _ in range(64):
        items = _parts(rng, 2 * capacity, rng.randint(5, 8))
        if max(items) > capacity or gcd(*items) != 1:
            continue
        if any(sum(size > capacity // (k + 1) for size in items) > 2 * k
               for k in range(1, len(items) // 2 + 1)):
            continue
        reachable = 1
        for size in items:
            reachable |= reachable << size
        if not reachable & (1 << capacity):
            rng.shuffle(items)
            return _task(capacity, 2, items)

    # With an 8, the rest would need to sum to 7. Without the 8s, all
    # remaining items sum to 14. Scaling preserves the gap.
    scale = rng.randint(5, 12)
    items = [8 * scale, 8 * scale, 6 * scale, 5 * scale, 3 * scale]
    rng.shuffle(items)
    return _task(15 * scale, 2, items)


def _oversized(rng):
    capacity = rng.randint(60, 220)
    items = [rng.randint(1, capacity // 5) for _ in range(rng.randint(1, 8))]
    items.insert(rng.randrange(len(items) + 1), capacity + rng.randint(1, 10))
    return _task(capacity, rng.randint(3, 6), items)


_GENERATORS = {
    "pairs": _pairs, "triples": _triples, "mixed": _mixed, "loose": _loose,
    "tiny": _tiny, "ordered": _ordered, "overload": _overload,
    "large_count": _large_count, "granularity": _granularity,
    "subset_gap": _subset_gap, "oversized": _oversized,
}

_FIXTURES = {
    "replay": (10, 2, (4, 4, 6, 6)),  # [0, 2], [1, 3]
    "empty": (10, 3, ()),  # [], [], []
    "one_disk": (10, 1, (2, 1, 7)),  # [0, 1, 2]
}


def case_expectations(phase):
    """Return (profile, feasible) tuples in the same order as cases()."""
    if phase not in _QUOTAS:
        raise ValueError(f"unknown phase: {phase!r}")
    return [(profile, profile not in _IMPOSSIBLE)
            for profile, count in _QUOTAS[phase] for _ in range(count)]


def cases(seed, phase):
    """Return fresh tasks for an integer seed and a named split."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if phase not in _QUOTAS:
        raise ValueError(f"unknown phase: {phase!r}")
    tasks = []
    for profile, count in _QUOTAS[phase]:
        if profile in _FIXTURES:
            tasks.append(_task(*_FIXTURES[profile]))
            continue
        for index in range(count):
            rng = _rng(seed, phase, profile, index)
            if profile == "ordered_regression":
                tasks.append(_ordered(rng, used=index + 2))
            else:
                tasks.append(_GENERATORS[profile](rng))
    return tasks


def check(task, packing):
    """Check exactly bins groups of distinct, zero-based file indices."""
    if not isinstance(task, dict):
        return False
    capacity = task.get("capacity")
    bins = task.get("bins")
    items = task.get("items")
    if type(capacity) is not int or capacity <= 0 or type(bins) is not int or bins <= 0:
        return False
    if not isinstance(items, (list, tuple)) or any(type(size) is not int or size <= 0 for size in items):
        return False
    if not isinstance(packing, (list, tuple)) or len(packing) != bins:
        return False

    seen = set()
    for disk in packing:
        if not isinstance(disk, (list, tuple)):
            return False
        load = 0
        for index in disk:
            if type(index) is not int or not 0 <= index < len(items) or index in seen:
                return False
            seen.add(index)
            load += items[index]
            if load > capacity:
                return False
    return len(seen) == len(items)
