"""Seal one legacy failure and reveal it only after independent CLI probes."""

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import sqlite3
import subprocess


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
V1 = PROJECT / "state/demos/20261118-183901-uhbq90ps"
ARENA_SHA256 = "57d55bf92c56258f7bd306f16a0f2c4603706c4f065a1e19327d95a928a20f7d"
PRIVATE = HERE / ".sealed_blind"
COMMITMENT = HERE / "blind_commitment.json"
REVEAL = HERE / "blind_result.json"
CAPACITY = 373
BINS = 6


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def write_once(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2) + "\n")


def check(task, packing):
    if type(packing) is not list or len(packing) != task["bins"]:
        return False
    files = task["items"]
    seen = []
    for disk in packing:
        if type(disk) is not list or any(type(i) is not int or not 0 <= i < len(files) for i in disk):
            return False
        if sum(files[i] for i in disk) > task["capacity"]:
            return False
        seen.extend(disk)
    return sorted(seen) == list(range(len(files)))


def first_fit(task, order):
    free = [task["capacity"]] * task["bins"]
    for index in order:
        for disk, left in enumerate(free):
            if task["items"][index] <= left:
                free[disk] -= task["items"][index]
                break
        else:
            return False
    return True


def sample(nonce):
    seed = hashlib.sha256(b"agentic-triples-blind-v1\0" + bytes.fromhex(nonce)).digest()
    rng = random.Random(int.from_bytes(seed, "big"))
    low, high = CAPACITY // 4 + 1, (CAPACITY - 1) // 2
    for draw in range(1, 1025):
        items = []
        for _ in range(BINS):
            first = rng.randint(low, min(high, CAPACITY - 2 * low))
            second = rng.randint(max(low, CAPACITY - first - high), min(high, CAPACITY - first - low))
            items.extend((first, second, CAPACITY - first - second))
        order = list(range(len(items)))
        rng.shuffle(order)
        task = {"capacity": CAPACITY, "bins": BINS, "items": [items[index] for index in order]}
        arrival = range(len(items))
        descending = sorted(arrival, key=task["items"].__getitem__, reverse=True)
        if first_fit(task, arrival) or first_fit(task, descending):
            continue
        inverse = {original: index for index, original in enumerate(order)}
        witness = [[inverse[index] for index in range(start, start + 3)]
                   for start in range(0, len(items), 3)]
        assert check(task, witness) and math.gcd(CAPACITY, *items) == 1
        return {"nonce": nonce, "draw": draw, "task": task, "witness": witness}
    raise RuntimeError("the sealed sampler exhausted 1024 draws")


def source_database(folder):
    return sqlite3.connect((folder.resolve() / "agentic.sqlite3").as_uri() + "?mode=ro", uri=True)


def signature(task):
    capacity, bins, items = task["capacity"], task["bins"], task["items"]
    unit = math.gcd(capacity, *items)
    return bins, capacity // unit, tuple(sorted(size // unit for size in items))


def exclude_direct_tasks(folder, blind):
    key = signature(blind)
    with closing(source_database(folder)) as db:
        tasks = (json.loads(raw) for raw, in db.execute("SELECT task FROM events WHERE kind='solve'"))
        if any(task and signature(task) == key for task in tasks):
            raise ValueError("sealed task duplicates a direct task in the selected state")


def check_arena():
    if fingerprint((PROJECT / "packlab/arena.py").read_bytes()) != ARENA_SHA256:
        raise ValueError("arena differs from the sealed capacity proof")


def seal():
    check_arena()
    PRIVATE.mkdir(mode=0o700)
    payload = sample(secrets.token_hex(32))
    exclude_direct_tasks(V1, payload["task"])
    write_once(PRIVATE / "payload.json", payload)
    public = {"commitment": fingerprint(canonical(payload)),
              "sampler_sha256": fingerprint(Path(__file__).read_bytes()),
              "arena_sha256": ARENA_SHA256,
              "protocol": "agentic-triples-blind-v1",
              "shape": {"bins": BINS, "files_per_disk": 3, "normalized_capacity": CAPACITY},
              "selection": "first independently generated tight triple fixture missed by both v1 policies"}
    write_once(COMMITMENT, public)
    print(json.dumps(public, indent=2))


def snapshot(source, label):
    target = PRIVATE / label
    target.mkdir(mode=0o700)
    with closing(source_database(source)) as db, closing(sqlite3.connect(target / "agentic.sqlite3")) as copy:
        db.backup(copy)
        generation, raw = copy.execute(
            "SELECT r.id, r.policies FROM revisions r JOIN head h ON r.id=h.revision WHERE h.slot=1"
        ).fetchone()
    return target, generation, json.loads(raw)


def cli_probe(target, task):
    cmd = [str(PROJECT / "agentic"), "--state", str(target), "probe", "-", "--json"]
    try:
        child = subprocess.run(cmd, input=canonical(task), capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"error": "CLI probe exceeded 30 seconds"}
    if child.returncode:
        return {"exit": child.returncode, "error": child.stderr.decode("utf-8", errors="replace")}
    try:
        return {"exit": child.returncode, "output": json.loads(child.stdout)}
    except ValueError:
        return {"exit": child.returncode, "error": "CLI probe returned invalid JSON"}


def honored(task, attempt, policies):
    output = attempt.get("output")
    return output and output["task"] == task and output["policies"] == policies


def verify_commitment(public, payload):
    if (fingerprint(canonical(payload)) != public["commitment"] or
            fingerprint(Path(__file__).read_bytes()) != public["sampler_sha256"] or
            sample(payload["nonce"]) != payload):
        raise ValueError("sealed payload or sampler differs from the commitment")


def test(new_state):
    check_arena()
    public = json.loads(COMMITMENT.read_text())
    payload = json.loads((PRIVATE / "payload.json").read_text())
    verify_commitment(public, payload)
    task = payload["task"]
    exclude_direct_tasks(new_state, task)
    before, old_rev, old_policies = snapshot(V1, "v1")
    after, new_rev, new_policies = snapshot(new_state, "active")
    if old_rev != 1 or new_rev <= 1 or new_policies[:len(old_policies)] != old_policies:
        raise ValueError("selected state has no newer revision extending the published v1")
    (PRIVATE / "started").touch(exist_ok=False)
    old, new = cli_probe(before, task), cli_probe(after, task)
    transferred = bool(honored(task, old, old_policies) and honored(task, new, new_policies)
                       and not old["output"]["result"]["ok"]
                       and new["output"]["result"]["ok"]
                       and check(task, new["output"]["result"]["packing"]))
    public_result = {"sealed": payload, "commitment": public["commitment"], "transfer": transferred,
                     "v1": {"revision": old_rev, "persisted_policies": old_policies, **old},
                     "active": {"revision": new_rev, "persisted_policies": new_policies, **new}}
    write_once(REVEAL, public_result)
    print(json.dumps({"transfer": transferred, "evidence": str(REVEAL)}, indent=2))


def verify():
    public = json.loads(COMMITMENT.read_text())
    evidence = json.loads(REVEAL.read_text())
    payload = evidence["sealed"]
    verify_commitment(public, payload)
    task = payload["task"]
    old, new = evidence["v1"], evidence["active"]
    transferred = bool(honored(task, old, old["persisted_policies"])
                       and honored(task, new, new["persisted_policies"])
                       and not old["output"]["result"]["ok"]
                       and new["output"]["result"]["ok"]
                       and check(task, new["output"]["result"]["packing"]))
    assert evidence["commitment"] == public["commitment"] and transferred == evidence["transfer"]
    print(json.dumps({"commitment_verified": True, "transfer": transferred}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("verify")
    probe = commands.add_parser("probe")
    probe.add_argument("committed_state", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        seal()
    elif args.command == "verify":
        verify()
    else:
        test(args.committed_state)


if __name__ == "__main__":
    main()
