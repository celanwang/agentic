"""Replay the strategy candidate's independent checks with Python 3.9+."""
import itertools
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from engine import BASE, propose, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.v0 import pack


def check_result(task, result):
    packing = result["packing"]
    assert len(packing) == task["bins"]
    indices = [i for group in packing for i in group]
    assert len(indices) == len(set(indices))
    assert all(type(i) is int and 0 <= i < len(task["items"]) for i in indices)
    assert result["ok"] == (len(indices) == len(task["items"]))
    assert all(sum(task["items"][i] for i in group) <= task["capacity"] for group in packing)
    assert result["cost"] == sum(result["trace"]["ops"].values()) > 0
    assert json.loads(json.dumps(result)) == result


def compare_baseline(task):
    raw = pack(task["capacity"], task["bins"], task["items"])
    result = run(task)
    check_result(task, result)
    assert result == run(task, BASE) == run(task, {})
    assert result["ok"] == (raw is not None)
    if raw is not None:
        assert result["packing"] == raw
    else:
        k = result["trace"]["blocked"]["step"]
        assert result["packing"] == pack(task["capacity"], task["bins"], task["items"][:k])
    return result


def two_disk_oracle(task):
    # A reachable subset load fitting disk 1 leaves the complement on disk 2.
    bits = 1
    capacity = task["capacity"]
    mask = (1 << (capacity + 1)) - 1
    for size in task["items"]:
        bits |= (bits << size) & mask
    return bool(bits >> max(0, sum(task["items"]) - capacity))


def check_small():
    trials = 0
    for n in range(6):
        for sizes in itertools.product(range(1, 6), repeat=n):
            task = dict(capacity=6, bins=2, items=list(sizes))
            result = compare_baseline(task)
            expected = two_disk_oracle(task)
            search = run(task, dict(BASE, repair="search"))
            check_result(task, search)
            assert search["ok"] == expected
            proposals = propose(task, BASE, result)
            assert len(proposals["candidates"]) <= 16
            assert json.loads(json.dumps(proposals)) == proposals
            for policy in proposals["candidates"]:
                trial = run(task, policy)
                check_result(task, trial)
                assert expected or not trial["ok"]
            trials += 1
    print(f"{trials} ordered tasks: baseline prefix, subset oracle, and proposals passed")


def check_obstructions():
    for sizes, repair in [([4, 4, 6, 6], "none"), ([5, 4, 3, 2, 2, 2, 2], "exchange")]:
        task = dict(capacity=10, bins=2, items=sizes)
        result = compare_baseline(task)
        proposal = propose(task, BASE, result)
        assert not result["ok"]
        if repair == "exchange":
            assert all(not run(task, dict(BASE, order=o, pick=p))["ok"]
                       for o in ("arrival", "descending") for p in ("first", "tight", "roomy"))
        for candidate in proposal["candidates"]:
            trial = run(task, candidate)
            check_result(task, trial)
            if candidate["repair"] == repair and trial["ok"]:
                print(sizes, proposal["observation"], candidate, trial["packing"],
                      f"cost {result['cost']} + {trial['cost']}")
                break
        else:
            raise AssertionError("no successful proposal for target primitive")
    for sizes, reason in [([1, 1, 11], "oversize"), ([6, 6, 6, 6], "total_excess")]:
        task = dict(capacity=10, bins=2, items=sizes)
        result = compare_baseline(task)
        assert result["trace"]["blocked"]["reason"] == reason
        assert not propose(task, BASE, result)["candidates"]


def check_second_round():
    task = dict(capacity=20, bins=4, items=[7, 2, 1, 6, 2, 10, 6, 6, 6, 9, 3, 6, 2, 2, 5, 7])
    base = compare_baseline(task)
    short = dict(BASE, repair="search")
    assert short in propose(task, BASE, base)["candidates"]
    first = run(task, short)
    check_result(task, first)
    assert first["trace"]["search_status"] == "limit"
    deep = dict(short, repair="deep")
    assert deep in propose(task, short, first)["candidates"]
    second = run(task, deep)
    check_result(task, second)
    assert second["ok"]
    print("two rounds: cutoff at 256 nodes, solved in", second["trace"]["ops"]["search_nodes"])


def check_limits():
    rng = random.Random(7291)
    bounded = 0
    max_nodes = max_cost = 0
    for case in range(80):
        # Fill sixteen disks exactly, then hide their known assignments.
        items = []
        for _ in range(16):
            endpoints = [0] + sorted(rng.sample(range(1, 100), 3)) + [100]
            items.extend(b - a for a, b in zip(endpoints, endpoints[1:]))
        rng.shuffle(items)
        task = dict(capacity=100, bins=16, items=items)
        compare_baseline(task)
        for repair in ("exchange", "search", "deep"):
            policy = dict(BASE, order=("arrival", "descending")[case % 2],
                          pick=("first", "tight", "roomy")[case % 3], repair=repair)
            trial = run(task, policy)
            check_result(task, trial)
            ops = trial["trace"]["ops"]
            assert ops["exchange_probes"] <= 512
            assert ops["search_nodes"] <= (256 if repair == "search" else 2048)
            assert trial["trace"]["search_status"] != "infeasible"
            if trial["trace"]["search_status"] == "limit":
                bounded += 1
                if repair == "search":
                    assert dict(policy, repair="deep") in propose(task, policy, trial)["candidates"]
            max_nodes = max(max_nodes, ops["search_nodes"])
            max_cost = max(max_cost, trial["cost"])
    assert bounded and max_nodes == 2048
    print(f"80 known-feasible 64-file/16-disk tasks: {bounded} budget cutoffs,"
          f" peak {max_nodes} nodes, peak cost {max_cost}")


if __name__ == "__main__":
    check_small()
    check_obstructions()
    check_second_round()
    check_limits()
