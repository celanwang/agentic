"""Commit independent routing inputs without evaluating either strategy."""

import json
from pathlib import Path
from random import Random
import secrets

from candidates.pair_eval import seal as pairs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIVATE = HERE / ".sealed_route"
COMMITMENT = HERE / "commitment.json"
VERSION = "agentic-route-review-blind-v1"
PAIR_HELPER = "32a74ce4c444c277388b38b2a30237a23dc1082b637cdc1b4ea2296d702d46d9"


def draw(nonce, bins, label, low):
    seed = int(pairs.digest(pairs.canonical([VERSION, nonce, bins, label])), 16)
    rng = Random(seed)
    return rng, rng.randrange(low, low + 100_000)


def ordered(case):
    task, witness = case
    items = task["items"]
    groups = [list(items[i] for i in disk) for disk in witness]
    groups.sort(key=sum, reverse=True)
    assert all(sum(group) == task["capacity"] for group in groups[:-1])
    return pairs.from_disks(groups, range(len(items)), task["capacity"])


def construct(nonce):
    if len(bytes.fromhex(nonce)) != 32:
        raise ValueError("expected a 32-byte nonce")
    assert pairs.digest(Path(pairs.__file__).read_bytes()) == PAIR_HELPER
    cases = []
    for bins in range(5, 9):
        base = 2_100_000 + (bins - 5) * 400_000
        rng, cap = draw(nonce, bins, "strict", base)
        strict = pairs.strict_pair(rng, cap, bins)
        rng, cap = draw(nonce, bins, "matched", base + 100_000)
        anchored, mixed = pairs.matched(rng, cap, bins)
        rng, cap = draw(nonce, bins, "near", base + 200_000)
        near = pairs.strict_pair(rng, cap, bins, shortfall=1)
        rng, cap = draw(nonce, bins, "old-control", base + 300_000)
        control = pairs.matched(rng, cap, bins)[1] if bins % 2 else pairs.strict_pair(rng, cap, bins, shortfall=1)
        entries = (("strict", strict), ("anchored", anchored), ("mixed", mixed),
                   ("near", near), ("old_control", ordered(control)))
        for profile, (task, witness) in entries:
            effective, normalized, sizes = pairs.core(task)
            assert normalized == task["capacity"] and pairs.check(task, witness)
            assert pairs.exact_pairs(normalized, effective, sizes) == (profile in {"strict", "anchored"})
            cases.append(dict(name=f"{profile}-{bins}", profile=profile, task=task, witness=witness))
    assert len({pairs.core(row["task"]) for row in cases}) == len(cases) == 20
    return dict(version=VERSION, nonce=nonce, cases=cases)


def main():
    PRIVATE.mkdir(mode=0o700)
    payload = construct(secrets.token_hex(32))
    pairs.write_once(PRIVATE / "payload.json", payload)
    commitment = dict(version=VERSION, commitment=pairs.digest(pairs.canonical(payload)), cases=20,
                      sampler_sha256=pairs.digest(Path(__file__).read_bytes()), pair_helper_sha256=PAIR_HELPER,
                      bins=[5, 6, 7, 8], capacity_domain=[2_100_000, 3_699_999],
                      profiles=["strict", "anchored", "mixed", "near", "old_control"],
                      selection="one system-random nonce; construction only, no solver screening")
    pairs.write_once(COMMITMENT, commitment)
    print(json.dumps(commitment, indent=2))


if __name__ == "__main__":
    main()
