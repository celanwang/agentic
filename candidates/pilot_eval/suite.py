"""Independent seeded bin-packing workloads with verified reversible variants."""

from collections import Counter
from hashlib import sha256
import json
from math import gcd
from random import Random


VERSION = "agentic-pilot-eval-v1"
ARENA_SHA256 = "57d55bf92c56258f7bd306f16a0f2c4603706c4f065a1e19327d95a928a20f7d"
SEEDS = tuple(int.from_bytes(sha256(f"{VERSION}/learning/{i}".encode()).digest()[:8], "big")
              for i in range(6))
ROUNDS = 8
CALL_BUDGET = 128
PROFILES = ("ordered", "pairs", "triples", "mixed_exact", "mixed_slack", "two_disk")
VARIANTS = ("raw", "scaled", "anchored", "both")
_CAPACITY = {
    "train": (389, 401, 419, 433, 449, 461, 479, 491),
    "checkpoint": (397, 409, 421, 439, 457, 467, 487, 499),
    "blind": (383, 431, 443, 463, 503, 509, 521, 523),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _rng(*parts):
    return Random(int.from_bytes(sha256(canonical([VERSION, *parts])).digest(), "big"))


def key(task):
    capacity = task["capacity"]
    items = [size for size in task["items"] if size != capacity]
    bins = task["bins"] - (len(task["items"]) - len(items))
    unit = gcd(capacity, *items)
    return bins, capacity // unit, tuple(sorted(size // unit for size in items))


def check(task, packing):
    files = task["items"]
    if not isinstance(packing, (tuple, list)) or len(packing) != task["bins"]:
        return False
    seen = Counter()
    for disk in packing:
        if not isinstance(disk, (tuple, list)):
            return False
        if any(type(index) is not int or not 0 <= index < len(files) for index in disk):
            return False
        if sum(files[index] for index in disk) > task["capacity"]:
            return False
        seen.update(disk)
    return seen == Counter(range(len(files)))


def _parts(rng, load, count):
    cuts = [0, *sorted(rng.sample(range(1, load), count - 1)), load]
    return [right - left for left, right in zip(cuts, cuts[1:])]


def _disks(rng, profile, capacity):
    if profile == "two_disk":
        load = capacity - rng.randint(0, 2)
        return [_parts(rng, original, rng.randint(3, 8)) for original in (capacity, load)]
    bins = rng.randint(3, 10)
    if profile == "ordered":
        return [_parts(rng, capacity, rng.randint(2, 6)) for _ in range(bins)]
    if profile == "pairs":
        smalls = [rng.randint(capacity // 4 + 1, capacity // 2) for _ in range(bins)]
        return [[small, capacity - small] for small in smalls]
    if profile == "triples":
        low, high = capacity // 4 + 1, capacity // 2
        disks = []
        for _ in range(bins):
            first = rng.randint(low, min(high, capacity - 2 * low))
            second = rng.randint(max(low, capacity - first - high), min(high, capacity - first - low))
            disks.append([first, second, capacity - first - second])
        return disks
    max_parts = min(7, 63 // bins - 1)
    disks = []
    for _ in range(bins):
        slack = rng.randint(0, capacity // 25) if profile == "mixed_slack" else 0
        anchor = rng.randint(capacity // 8, 4 * capacity // 5)
        disks.append([anchor, *_parts(rng, capacity - slack - anchor, rng.randint(2, max_parts))])
    return disks


def _cluster(split, seed, profile, ordinal, seen):
    for retry in range(128):
        rng = _rng(split, seed, profile, ordinal, retry)
        capacity = rng.choice(_CAPACITY[split])
        disks = _disks(rng, profile, capacity)
        flat = [size for disk in disks for size in disk]
        order = list(range(len(flat)))
        if profile != "ordered":
            rng.shuffle(order)
        files = [flat[index] for index in order]
        task = {"capacity": capacity, "bins": len(disks), "items": files}
        signature = key(task)
        if signature in seen:
            continue
        seen.add(signature)
        break
    else:
        raise ValueError("independent generator exhausted the collision retry limit")

    inverse = {original: index for index, original in enumerate(order)}
    witness = []
    start = 0
    for disk in disks:
        end = start + len(disk)
        witness.append([inverse[index] for index in range(start, end)])
        start = end

    insertion = rng.randrange(len(files) + 1)
    scale = rng.choice((2, 3, 5, 11))
    cases = []
    for variant in VARIANTS:
        factor = scale if variant in {"scaled", "both"} else 1
        variant_task = {"capacity": capacity * factor, "bins": len(disks),
                        "items": [size * factor for size in files]}
        variant_proof = [disk[:] for disk in witness]
        if variant in {"anchored", "both"}:
            variant_task["items"].insert(insertion, variant_task["capacity"])
            variant_task["bins"] += 1
            variant_proof = [[index + (index >= insertion) for index in disk] for disk in witness]
            variant_proof.append([insertion])
        assert key(variant_task) == signature and check(variant_task, variant_proof)
        assert variant_task["bins"] <= 16 and len(variant_task["items"]) <= 64
        cases.append({"variant": variant, "task": variant_task, "witness": variant_proof})
    return {"profile": profile, "id": f"{split}-{ordinal}", "cases": cases}


def learning():
    seen = set()
    ordinal = 0
    seeds = []
    for index, seed in enumerate(SEEDS):
        schedule = list(PROFILES[2:])
        _rng("schedule", seed).shuffle(schedule)
        rounds = []
        for i in range(ROUNDS):
            ordinal += 1
            focus = schedule[i % 4]
            neighbor = schedule[(i + 1) % 4]
            train_profiles = [focus] * 3 + [neighbor, "ordered", "pairs"]
            _rng("train-order", seed, i).shuffle(train_profiles)
            train = [_cluster("train", seed, profile, 10 * ordinal + slot, seen)
                     for slot, profile in enumerate(train_profiles)]
            train = [dict(profile=group["profile"], **group["cases"][(i + slot) % 4])
                     for slot, group in enumerate(train)]
            checkpoint = [_cluster("checkpoint", seed, profile, 10 * ordinal + slot, seen)
                          for slot, profile in enumerate((focus, neighbor, PROFILES[i % 2]))]
            rounds.append({"round": i + 1, "focus": focus, "train": train, "checkpoint": checkpoint})
        seeds.append({"index": index, "seed": seed, "rounds": rounds})
    return seeds


def blind(nonce):
    if len(bytes.fromhex(nonce)) != 32:
        raise ValueError("the blind nonce must contain 32 bytes")
    seen = set()
    groups = [_cluster("blind", nonce, profile, i * 3 + j, seen)
              for i, profile in enumerate(PROFILES) for j in range(3)]
    _rng("blind-order", nonce).shuffle(groups)
    return {"version": VERSION, "nonce": nonce, "clusters": groups}
