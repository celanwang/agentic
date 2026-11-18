"""Commit one new mixed v2 failure, then probe only the persisted decision."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
from random import Random
import secrets

from candidates.triples_eval import blind_probe as common
from candidates.triples_strategy import engine as v2


HERE = Path(__file__).resolve().parent
V2_STATE = common.PROJECT / "state/triples/20261118-200049-3qo0skt6"
V2_SHA = "f2be6568d4e580fb0fd2dbece6d3f4d8df9a0844cd5b97c28773a500c1f2b6cf"
COMMON_SHA = "4a4958a88eda5a36f7aed17b3371f20ca039b91a313340fee25c427fd453e947"
PRIVATE = HERE / ".sealed_mixed"
COMMITMENT = HERE / "blind_commitment.json"
RESULT = HERE / "blind_result.json"
CAPACITY = 379
POLICIES = [dict(v2.BASE), dict(v2.BASE, order="descending"),
            dict(v2.BASE, order="descending", repair="triples")]


def sample(nonce):
    key = hashlib.sha256(b"agentic-mixed-blind-v1\0" + bytes.fromhex(nonce)).digest()
    rng = Random(int.from_bytes(key, "big"))
    for draw in range(1, 1025):
        bins = rng.randint(2, 12)
        max_slack = rng.choice((0, 2, CAPACITY // 18))
        max_parts = min(7, 64 // bins - 1)
        witness, files = [], []
        for _ in range(bins):
            load = CAPACITY - rng.randint(0, max_slack)
            anchor = rng.randint(CAPACITY // 8, 4 * CAPACITY // 5)
            count = rng.randint(2, max_parts)
            cuts = [0, *sorted(rng.sample(range(1, load - anchor), count - 1)), load - anchor]
            disk = [anchor, *(right - left for left, right in zip(cuts, cuts[1:]))]
            witness.append(list(range(len(files), len(files) + len(disk))))
            files.extend(disk)
        order = list(range(len(files)))
        rng.shuffle(order)
        task = {"bins": bins, "capacity": CAPACITY, "items": [files[index] for index in order]}
        if any(v2.run(task, policy)["ok"] for policy in POLICIES):
            continue
        inverse = {original: index for index, original in enumerate(order)}
        packing = [[inverse[index] for index in disk] for disk in witness]
        assert common.check(task, packing) and common.signature(task)[1] == CAPACITY
        return {"nonce": nonce, "draw": draw, "task": task, "witness": packing}
    raise ValueError("the sealed mixed sampler exhausted 1024 draws")


def assert_frozen_sources():
    common.check_arena()
    assert common.fingerprint(Path(v2.__file__).read_bytes()) == V2_SHA
    assert common.fingerprint(Path(common.__file__).read_bytes()) == COMMON_SHA


def seal():
    assert_frozen_sources()
    PRIVATE.mkdir(mode=0o700)
    payload = sample(secrets.token_hex(32))
    for state in (common.V1, V2_STATE):
        common.exclude_direct_tasks(state, payload["task"])
    common.write_once(PRIVATE / "payload.json", payload)
    public = {"commitment": common.fingerprint(common.canonical(payload)),
              "sampler_sha256": common.fingerprint(Path(__file__).read_bytes()),
              "frozen_v2_sha256": V2_SHA, "helper_sha256": COMMON_SHA,
              "arena_sha256": common.ARENA_SHA256, "normalized_capacity": CAPACITY,
              "protocol": "agentic-mixed-blind-v1",
              "selection": "first legacy v2 failure drawn from the mixed profile at prime capacity 379"}
    common.write_once(COMMITMENT, public)
    print(json.dumps(public, indent=2))


def verify_commitment(payload):
    assert_frozen_sources()
    public = json.loads(COMMITMENT.read_text())
    assert common.fingerprint(common.canonical(payload)) == public["commitment"]
    assert common.fingerprint(Path(__file__).read_bytes()) == public["sampler_sha256"]
    assert sample(payload["nonce"]) == payload
    return public["commitment"]


def decision(state, event_id):
    with closing(common.source_database(state)) as db:
        row = db.execute("""SELECT decision, json_extract(data,'$.generation_before'),
                             json_extract(data,'$.generation'), json_extract(data,'$.nominee'),
                             json_extract(data,'$.rules') FROM events WHERE id=? AND kind='solve'""",
                         (event_id,)).fetchone()
    if row is None:
        raise ValueError("formal decision is absent from the state")
    status, before, after, nominee, rules = row
    return dict(event=event_id, decision=status, before=before, after=after,
                nominee=json.loads(nominee) if nominee else None, rules=json.loads(rules))


def snapshot(state, label):
    import sqlite3

    target = PRIVATE / label
    target.mkdir(mode=0o700)
    with closing(common.source_database(state)) as db, closing(sqlite3.connect(target / "agentic.sqlite3")) as copy:
        db.backup(copy)
        revision, raw = copy.execute(
            "SELECT r.id, r.policies FROM revisions r JOIN head h ON r.id=h.revision WHERE h.slot=1"
        ).fetchone()
    return target, revision, json.loads(raw)


def transfer(task, old, new):
    if not common.honored(task, old, old["policies"]) or not common.honored(task, new, new["policies"]):
        return None
    control = old["output"]["result"]
    active = new["output"]["result"]
    if active["ok"] and not common.check(task, active["packing"]):
        raise ValueError("invalid formal CLI packing")
    return not control["ok"] and active["ok"]


def probe(state, event_id):
    payload = json.loads((PRIVATE / "payload.json").read_text())
    commitment = verify_commitment(payload)
    task = payload["task"]
    common.exclude_direct_tasks(state, task)
    formal = decision(state, event_id)
    reference = decision(V2_STATE, 4)
    assert formal["rules"] == reference["rules"] and formal["before"] == 2
    before, old_rev, old = snapshot(V2_STATE, "v2")
    after, new_rev, active = snapshot(state, "active")
    assert old_rev == 2 and old == POLICIES and new_rev == formal["after"]
    if formal["decision"] == "kept":
        assert new_rev > old_rev and active == old + [formal["nominee"]]
    else:
        assert formal["decision"] in {"rejected", "inconclusive"} and new_rev == old_rev and active == old
    (PRIVATE / "started").touch(exist_ok=False)
    control = {"revision": old_rev, "policies": old, **common.cli_probe(before, task)}
    assessed = {"revision": new_rev, "policies": active, **common.cli_probe(after, task)}
    moved = transfer(task, control, assessed)
    common.write_once(RESULT, {"sealed": payload, "commitment": commitment, "formal": formal,
                               "v2": control, "active": assessed, "transfer": moved})
    print(json.dumps({"formal_decision": formal["decision"], "transfer": moved, "result": str(RESULT)}, indent=2))


def verify():
    result = json.loads(RESULT.read_text())
    payload = result["sealed"]
    assert verify_commitment(payload) == result["commitment"]
    assert transfer(payload["task"], result["v2"], result["active"]) == result["transfer"]
    print(json.dumps({"commitment_verified": True, "transfer": result["transfer"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal")
    commands.add_parser("verify")
    active = commands.add_parser("probe")
    active.add_argument("state", type=Path)
    active.add_argument("event", type=int)
    args = parser.parse_args()
    if args.command == "seal":
        seal()
    elif args.command == "probe":
        probe(args.state, args.event)
    else:
        verify()


if __name__ == "__main__":
    main()
