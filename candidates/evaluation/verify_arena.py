"""Reproduce generator, checker, and practice diagnostics before selection."""

from collections import Counter, defaultdict, deque
from math import gcd
from unittest.mock import patch

from baseline.v0 import pack
from candidates.evaluation import arena


def _baseline(task):
    return pack(task["capacity"], task["bins"], task["items"])


def _descending(task):
    sizes = task["items"]
    indices = sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True)
    packing = pack(task["capacity"], task["bins"], [sizes[index] for index in indices])
    if packing is None:
        return None
    return [[indices[index] for index in disk] for disk in packing]


def _signature(task):
    capacity, bins, sizes = task["capacity"], task["bins"], task["items"]
    unit = gcd(capacity, *sizes)
    return capacity // unit, bins, tuple(sorted(size // unit for size in sizes))


def _check_shape(task):
    assert type(task["capacity"]) is int and task["capacity"] > 0
    assert type(task["bins"]) is int and 1 <= task["bins"] <= 16
    assert 0 <= len(task["items"]) <= 64
    assert all(type(size) is int and size > 0 for size in task["items"])


def _check_certificates():
    shuffle = arena._shuffled

    def verify_shuffled_witness(rng, capacity, disks):
        task = shuffle(rng, capacity, disks)
        positions = defaultdict(deque)
        for index, size in enumerate(task["items"]):
            positions[size].append(index)
        witness = [[positions[size].popleft() for size in disk] for disk in disks]
        assert arena.check(task, witness)
        return task

    for profile, generator in arena._GENERATORS.items():
        for seed in range(1024):
            rng = arena._rng(seed, "generator-self-test", profile, 0)
            if profile in {"pairs", "triples", "mixed", "loose", "tiny"}:
                with patch.object(arena, "_shuffled", verify_shuffled_witness):
                    task = generator(rng)
            else:
                task = generator(rng)
            _check_shape(task)
            capacity, bins, sizes = task["capacity"], task["bins"], task["items"]
            assert capacity not in sizes
            if profile == "ordered":
                assert arena.check(task, _baseline(task))
            elif profile == "overload":
                assert max(sizes) <= capacity and sum(sizes) > capacity * bins
            elif profile == "oversized":
                assert max(sizes) > capacity and sum(sizes) <= capacity * bins
            elif profile == "large_count":
                assert sum(sizes) <= capacity * bins and max(sizes) <= capacity
                assert any(sum(size * (k + 1) > capacity for size in sizes) > bins * k
                           for k in range(1, 4))
            elif profile == "granularity":
                unit = gcd(*sizes)
                assert bins * (capacity // unit) * unit < sum(sizes) <= bins * capacity
            elif profile == "subset_gap":
                sums = {0}
                for size in sizes:
                    sums |= {total + size for total in sums if total + size <= capacity}
                assert bins == 2 and sum(sizes) == 2 * capacity and capacity not in sums

    for seed in range(1024):
        ordered = []
        for index in range(9):
            rng = arena._rng(seed, "generator-self-test", "ordered_regression", index)
            task = arena._ordered(rng, used=index + 2)
            _check_shape(task)
            assert arena.check(task, _baseline(task))
            assert task["capacity"] in task["items"]
            ordered.append(_signature(task))
        assert len(set(ordered)) == len(ordered)

    print("checked 11264 generic cases and 9216 isolated ordered regression shapes")


def _check_checker():
    task = {"capacity": 10, "bins": 2, "items": [4, 4, 6, 6]}
    assert _baseline(task) is None
    assert arena.check(task, [[0, 2], [1, 3]])
    assert arena.check(task, ((3, 1), (2, 0)))
    invalid = (None, [], [[0, 2], [1, 3], []], [[0, 1], [2, 3]],
               [[0, 2], [0, 3]], [[0, 2], [1]], [[False, 2], [1, 3]],
               [[-1, 2], [1, 3]], [[0.0, 2], [1, 3]], [[4, 2], [1, 3]],
               [[0, 2], 3], ["02", "13"])
    for packing in invalid:
        assert not arena.check(task, packing), packing
    assert not arena.check({"capacity": 20, "bins": 3, "items": [False]}, [[0], [], []])

    for profile, witness in {"replay": [[0, 2], [1, 3]], "empty": [[], [], []],
                             "one_disk": [[0, 1, 2]]}.items():
        task = arena._task(*arena._FIXTURES[profile])
        assert arena.check(task, witness)
        if profile != "replay":
            assert arena.check(task, _baseline(task))


def _practice_diagnostics():
    phases = ("practice", "holdout", "regression")
    assert [len(arena.case_expectations(phase)) for phase in phases] == [16, 64, 20]
    assert arena.case_expectations("regression").count(("ordered_regression", True)) == 9
    assert arena.cases(42, "practice") != arena.cases(43, "practice")
    mutable = arena.cases(42, "practice")
    mutable[0]["items"].clear()
    assert arena.cases(42, "practice")[0]["items"]

    stats = defaultdict(Counter)
    seeds_with_gap = 0
    for seed in range(128):
        tasks = arena.cases(seed, "practice")
        assert len(tasks) == 16 and tasks == arena.cases(seed, "practice")
        has_gap = False
        for task, (profile, feasible) in zip(tasks, arena.case_expectations("practice")):
            baseline = _baseline(task)
            descending = _descending(task)
            assert baseline is None or arena.check(task, baseline)
            assert descending is None or arena.check(task, descending)
            if not feasible:
                assert baseline is None and descending is None
                continue
            gap = baseline is None and descending is None
            has_gap |= gap
            stats[profile].update(feasible=1, baseline=baseline is not None,
                                  descending=descending is not None, both_failed=gap)
        seeds_with_gap += has_gap

    print("profile feasible baseline descending both_failed")
    for profile, counts in stats.items():
        print(profile, *(counts[key] for key in ("feasible", "baseline", "descending", "both_failed")))
    print(f"practice seeds with at least one feasible failure of both: {seeds_with_gap}/128")


if __name__ == "__main__":
    _check_checker()
    _check_certificates()
    _practice_diagnostics()
