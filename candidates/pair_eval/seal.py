"""Construct and commit the pre-release pair and boundary tasks locally."""

from collections import Counter
from hashlib import sha256
import json
from math import gcd
from pathlib import Path
from random import Random
import secrets


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRIVATE = HERE / ".sealed_pairs"
COMMITMENT = HERE / "blind_commitment.json"
VERSION = "agentic-pair-blind-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest(value):
    return sha256(value).hexdigest()


def write_once(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")


def rng_for(nonce, label):
    return Random(int(digest(canonical([VERSION, nonce, label])), 16))


def core(task):
    cap, sizes = task["capacity"], task["items"]
    kept = [size for i, size in enumerate(sizes) if size > cap or any(
        j != i and size + other <= cap for j, other in enumerate(sizes))]
    unit = gcd(cap, *kept)
    return task["bins"] - len(sizes) + len(kept), cap // unit, tuple(sorted(size // unit for size in kept))


def check(task, witness):
    if not isinstance(witness, list) or len(witness) != task["bins"]:
        return False
    files = task["items"]
    used = []
    for disk in witness:
        if not isinstance(disk, list) or any(type(i) is not int or not 0 <= i < len(files) for i in disk):
            return False
        if sum(files[i] for i in disk) > task["capacity"]:
            return False
        used.extend(disk)
    return sorted(used) == list(range(len(files)))


def stats(task):
    sizes, cap = task["items"], task["capacity"]
    return (task["bins"], cap, len(sizes), sum(sizes), min(sizes), max(sizes),
            sum(2 * size > cap for size in sizes), gcd(cap, *sizes))


def exact_pairs(cap, bins, sizes):
    if len(sizes) != 2 * bins:
        return False
    counts = Counter(sizes)
    return (all(0 < size < cap and quantity == counts[cap - size] for size, quantity in counts.items())
            and (cap % 2 != 0 or counts[cap // 2] % 2 == 0))


def strict_pair(rng, capacity, bins, shortfall=0):
    high = (capacity - 1) // 2
    first = rng.randint(capacity // 4 + 1, high - 1)
    small = [first, first + 1, *rng.sample(range(capacity // 4 + 1, high + 1), bins - 2)]
    rng.shuffle(small)
    owners = list(range(bins))
    rng.shuffle(owners)
    large = [capacity - small[i] for i in owners]
    large[-1] -= shortfall
    return {"capacity": capacity, "bins": bins, "items": small + large}, [[i, bins+j] for j, i in enumerate(owners)]


def from_disks(disks, order, capacity):
    flat = [size for disk in disks for size in disk]
    inverse = {original: i for i, original in enumerate(order)}
    witness = []
    offset = 0
    for disk in disks:
        witness.append([inverse[i] for i in range(offset, offset + len(disk))])
        offset += len(disk)
    return {"capacity": capacity, "bins": len(disks), "items": [flat[i] for i in order]}, witness


def matched(rng, capacity, bins):
    a = rng.randint(capacity // 8, capacity // 6)
    b = a + 1
    smalls = rng.sample(range(capacity // 4 + 1, (capacity - 1) // 2 + 1), bins - 3)
    same = [[capacity], *[[s, capacity-s] for s in smalls]]
    left = [*same, [a, capacity-a], [b, capacity-b]]
    right = [*same, [a, b, capacity-a-b], [capacity]]
    order = list(range(2*bins-1))
    rng.shuffle(order)
    pair, mix = from_disks(left, order, capacity), from_disks(right, order, capacity)
    assert stats(pair[0]) == stats(mix[0])
    return pair, mix


def construct(nonce):
    assert len(bytes.fromhex(nonce)) == 32
    rng = rng_for(nonce, "strict")
    strict = strict_pair(rng, rng.randrange(1_100_000, 1_200_000), rng.randint(5, 8))
    tasks = [dict(name="strict", task=strict[0], witness=strict[1], matching=True)]
    for bins in range(5, 9):
        rng = rng_for(nonce, f"matched-{bins}")
        capacity = rng.randrange(1_200_000 + (bins-5)*100_000, 1_300_000 + (bins-5)*100_000)
        for label, case in zip(("pair", "mixed"), matched(rng, capacity, bins)):
            tasks.append(dict(name=f"{label}-{bins}", task=case[0], witness=case[1], matching=label == "pair"))
        rng = rng_for(nonce, f"near-{bins}")
        cap = rng.randrange(1_600_000 + (bins-5)*100_000, 1_700_000 + (bins-5)*100_000)
        near = strict_pair(rng, cap, bins, shortfall=1)
        tasks.append(dict(name=f"near-{bins}", task=near[0], witness=near[1], matching=False))
    keys = set()
    for row in tasks:
        task = row["task"]
        effective_bins, cap, sizes = core(task)
        assert cap == task["capacity"] and gcd(cap, *task["items"]) == 1 and check(task, row["witness"])
        assert exact_pairs(cap, effective_bins, sizes) == row["matching"]
        keys.add(core(task))
    assert len(keys) == len(tasks) == 13
    return dict(version=VERSION, nonce=nonce, cases=tasks)


def main():
    PRIVATE.mkdir(mode=0o700)
    payload = construct(secrets.token_hex(32))
    write_once(PRIVATE / "payload.json", payload)
    record = dict(version=VERSION, commitment=digest(canonical(payload)), sampler_sha256=digest(Path(__file__).read_bytes()),
                  cases=13, profiles={"strict": 1, "anchored_pairs": 4, "anchored_mixed": 4, "near": 4},
                  capacity_domain=[1_100_000, 1_999_999], normalized=True,
                  selection="one 32-byte system nonce, first draw, no solver invocation")
    write_once(COMMITMENT, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
