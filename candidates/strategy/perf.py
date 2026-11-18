"""Repeatable cost/timing comparison on three locally constructed inputs."""
import argparse
import hashlib
import platform
import random
import statistics
import time
from pathlib import Path

import engine

ENGINE_SHA = "7e1acb65e9e1b08a14db42c3b17c855d784031d4b6f125633f353dd969ad0710"
CASES = {
    # Exact fill: each disk can hold eight units plus seven singletons.
    "small-first": dict(capacity=15, bins=4, items=[1] * 28 + [8] * 4),
    # Known feasible witness loads: [10,7,3], [9,7,2,2], [6,6,6,2], [6,6,5,2,1].
    "four-disks": dict(capacity=20, bins=4, items=[7, 2, 1, 6, 2, 10, 6, 6, 6, 9, 3, 6, 2, 2, 5, 7]),
    # Each disk holds at most five twos: twenty places for twenty-two files.
    "even-files": dict(capacity=11, bins=4, items=[2] * 22),
}
POLICIES = {
    "base": engine.BASE,
    "descending": dict(engine.BASE, order="descending"),
    "roomy": dict(engine.BASE, pick="roomy"),
    "exchange": dict(engine.BASE, repair="exchange"),
    "search": dict(engine.BASE, repair="search"),
    "deep": dict(engine.BASE, repair="deep"),
}


def attempt(task, chain):
    cost = 0
    for policy in chain:
        result = engine.run(task, policy)
        cost += result["cost"]
        if result["ok"]:
            break
    return cost, result


def measure(rounds, batch):
    digest = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()
    if digest != ENGINE_SHA:
        raise RuntimeError("benchmark requires the frozen engine")
    print(f"engine={digest} python={platform.python_version()} platform={platform.platform()}")
    print(f"rounds={rounds} batch={batch} clock=perf_counter_ns seed=4921")
    jobs = {}
    for case, task in CASES.items():
        for label, policy in POLICIES.items():
            jobs[case, label] = task, [policy]
            if label != "base":
                jobs[case, "base+" + label] = task, [engine.BASE, policy]

    refs = {key: attempt(*args) for key, args in jobs.items()}
    samples = {key: [] for key in jobs}
    for args in jobs.values():
        for _ in range(3):
            attempt(*args)
    rng = random.Random(4921)
    keys = list(jobs)
    for _ in range(rounds):
        rng.shuffle(keys)
        for key in keys:
            task, chain = jobs[key]
            start = time.perf_counter_ns()
            for _ in range(batch):
                last = attempt(task, chain)
            elapsed = time.perf_counter_ns() - start
            assert last == refs[key]
            samples[key].append(elapsed / (batch * 1000))
    print("case chain ok search_status nodes cost median_us p25_us p75_us")
    for key, timings in samples.items():
        cost, result = refs[key]
        trace = result["trace"]
        q1, _, q3 = statistics.quantiles(timings, method="inclusive")
        print(*key, int(result["ok"]), trace["search_status"] or "-", trace["ops"]["search_nodes"],
              cost, f"{statistics.median(timings):.2f}", f"{q1:.2f}", f"{q3:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--batch", type=int, default=40)
    args = parser.parse_args()
    if args.rounds < 2 or args.batch < 1:
        parser.error("rounds must be >= 2, batch must be >= 1")
    measure(args.rounds, args.batch)
