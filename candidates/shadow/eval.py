"""Commit and independently verify a synthetic CLI shadow arrival stream."""

import argparse
import inspect
import json
import os
from pathlib import Path
from random import Random
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory, mkstemp

from candidates.pair_eval import seal as pairs
from candidates.route_review.eval import seal as route


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMITMENT = HERE / "commitment.json"
VERSION = "agentic-shadow-arrivals-v1"
LIMIT = 12
PLAN = ("strict", "anchored", "near", "mixed", "old_control", "easy") * 2
DEPENDENCIES = {
    "candidates/pair_eval/seal.py": "32a74ce4c444c277388b38b2a30237a23dc1082b637cdc1b4ea2296d702d46d9",
    "candidates/route_review/eval/seal.py": "0f90aaf8f1e22ebf0feab38597b6ee7355d57f8e333b9ac66889e00d98c852d9",
    "candidates/pair_eval/probe_result.json": "6f67091b0cdb4742071d0458a02cede77965c759ab6cea211ea86c72509ff1f5",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fresh(nonce, index, profile):
    key = pairs.digest(pairs.canonical([VERSION, nonce, index, profile]))
    rng = Random(int(key, 16))
    cap = rng.randrange(3_900_000 + index * 100_000, 4_000_000 + index * 100_000)
    bins = rng.randint(5, 8)
    if profile == "strict":
        case = pairs.strict_pair(rng, cap, bins)
    elif profile == "near":
        case = pairs.strict_pair(rng, cap, bins, shortfall=1)
    elif profile in ("anchored", "mixed"):
        case = pairs.matched(rng, cap, bins)[profile == "mixed"]
    elif profile == "old_control":
        case = route.ordered(pairs.strict_pair(rng, cap, bins, shortfall=1))
    elif profile == "easy":
        disks = []
        for _ in range(bins):
            a = rng.randrange(cap // 6, cap // 5)
            disks.append([a, a + 1, cap - 2*a - 1])
        case = pairs.from_disks(disks, range(3 * bins), cap)
    else:
        raise ValueError(profile)
    task, witness = case
    assert pairs.check(task, witness) and pairs.core(task)[1] == cap
    return {"label": f"fresh-{index+1}", "profile": profile, "expected": "eligible", "task": task, "witness": witness}


def transform(row, label, expected, factor=1, reverse=False):
    task = row["task"]
    sizes = task["items"]
    order = list(reversed(range(len(sizes)))) if reverse else list(range(len(sizes)))
    inverse = {source: target for target, source in enumerate(order)}
    transformed = dict(task, capacity=task["capacity"] * factor, items=[sizes[i] * factor for i in order])
    witness = [[inverse[i] for i in disk] for disk in row["witness"]]
    assert pairs.core(transformed) == pairs.core(task) and pairs.check(transformed, witness)
    return dict(row, label=label, expected=expected, task=transformed, witness=witness)


def construct(nonce):
    if len(bytes.fromhex(nonce)) != 32:
        raise ValueError("the seal requires 32 random bytes")
    for name, digest in DEPENDENCIES.items():
        assert pairs.digest((ROOT / name).read_bytes()) == digest
    layout = list(PLAN)
    seed = pairs.digest(pairs.canonical([VERSION, nonce, "arrival order"]))
    Random(int(seed, 16)).shuffle(layout)
    rows = [fresh(nonce, index, label) for index, label in enumerate(layout)]
    # pair-5 is a publicly revealed #11 miss included in the #12 research batch.
    published = read(ROOT / "candidates/pair_eval/probe_result.json")
    known = next(row for row in published["reveal"]["cases"] if row["name"] == "pair-5")
    known = {"label": "known-batch", "profile": "prior_control", "expected": "prior",
             "task": known["task"], "witness": known["witness"]}
    old = {"label": "known-trigger", "profile": "prior_control", "expected": "prior",
           "task": read(ROOT / "examples/first.json"), "witness": [[0, 2], [1, 3]]}
    assert pairs.check(old["task"], old["witness"]) and pairs.check(known["task"], known["witness"])
    arrivals = [known, rows[0], transform(rows[0], "repeat-exact", "duplicate"), *rows[1:3],
                transform(rows[0], "repeat-scaled", "duplicate", factor=3, reverse=True), rows[3], old,
                *rows[4:7], transform(rows[1], "repeat-reordered", "duplicate", reverse=True), *rows[7:9],
                transform(known, "known-scaled", "prior", factor=5, reverse=True), *rows[9:],
                dict(fresh(nonce, LIMIT, "near"), label="over-limit-fresh", expected="closed")]
    assert len({pairs.core(row["task"]) for row in arrivals if row["expected"] in {"eligible", "closed"}}) == LIMIT+1
    return {"version": VERSION, "nonce": nonce, "limit": LIMIT, "arrivals": arrivals}


def sampler_digest():
    sources = [inspect.getsource(function) for function in (fresh, transform, construct)]
    return pairs.digest(pairs.canonical([VERSION, LIMIT, PLAN, DEPENDENCIES, sources]))


def seal():
    if COMMITMENT.exists():
        raise ValueError("the first-draw commitment already exists")
    payload = construct(secrets.token_hex(32))
    descriptor, path = mkstemp(prefix="agentic-shadow-sealed-", suffix=".json")
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(pairs.canonical(payload))
    record = {"version": VERSION, "commitment": pairs.digest(pairs.canonical(payload)),
              "sampler_sha256": sampler_digest(), "synthetic": True, "source": "observe", "event": 12,
              "limit": LIMIT, "arrivals": len(payload["arrivals"]), "profiles": list(PLAN[:6]),
              "planned": {"eligible": LIMIT, "prior": 3, "duplicate": 3, "closed": 1},
              "selection": "one system-random nonce; construction only, no solver screening"}
    pairs.write_once(COMMITMENT, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Local sealed payload is stored at {path}", file=sys.stderr)


def open_commitment(path):
    promise = read(COMMITMENT)
    data = read(path)
    payload = data.get("reveal", data)
    assert promise["sampler_sha256"] == sampler_digest()
    assert promise["commitment"] == pairs.digest(pairs.canonical(payload))
    assert payload == construct(payload["nonce"])
    assert payload["limit"] == promise["limit"] == LIMIT
    return promise, payload


def cli(state, *args, task=None):
    command = [sys.executable, "-B", str(ROOT / "agentic"), "--state", str(state), *args, "--json"]
    process = subprocess.run(command, input=pairs.canonical(task) if task is not None else None,
                             capture_output=True, cwd=ROOT, timeout=30)
    if process.returncode:
        raise RuntimeError(f"{' '.join(args)}: {process.stderr.decode('utf-8', errors='replace').strip()}")
    return json.loads(process.stdout)


def db_identity(folder, source):
    with source(folder) as conn:
        return {table: conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
                for table, order in (("head", "slot"), ("revisions", "id"), ("events", "id"))}


def family_key(task):
    bins, capacity, multiset = pairs.core(task)
    return pairs.digest(pairs.canonical([capacity, bins, multiset]))


def prior_keys(folder, source):
    with source(folder) as conn:
        keys = {family_key(json.loads(task)) for task, in conn.execute("SELECT task FROM events WHERE task<>'null'")}
        for name in ("practice_tasks", "batch_tasks", "holdout.rows", "regression.rows", "replay.rows"):
            for raw, in conn.execute("SELECT j.value FROM events e, json_each(e.data,?) j", ("$." + name,)):
                item = json.loads(raw)
                keys.add(family_key(item["task"] if name.endswith(".rows") else item))
    return keys


def expected_order(payload, prior, family_id):
    seen = set()
    expected = []
    for item in payload["arrivals"]:
        key = family_key(item["task"])
        assert key == family_id(item["task"])
        if len(seen) == LIMIT:
            outcome = "closed"
            assert key not in prior | seen, "post-limit control should have been new"
        elif key in prior:
            outcome = "prior"
        elif key in seen:
            outcome = "duplicate"
        else:
            outcome = "eligible"
            seen.add(key)
        assert outcome == item["expected"], item["label"]
        expected.append((key, outcome))
    assert len(seen) == LIMIT
    return expected


def baseline_probe(folder, task, chain, nominee=False):
    flags = ("--nominee", "12") if nominee else ()
    response = cli(folder, "probe", "-", *flags, task=task)
    assert response["task"] == task and response["policies"] == chain
    outcome = response["result"]
    assert outcome["cost"] == sum(attempt["cost"] for attempt in outcome["attempts"])
    if outcome["ok"]:
        assert pairs.check(task, outcome["packing"])
    else:
        assert outcome["trace"]["search_status"] != "infeasible"
        assert (outcome["trace"]["blocked"] or {}).get("reason") not in {"oversize", "total_excess"}
    return outcome


def path_for(task, actual, coverage):
    """Reconstruct branch identity for #12's fixed two-leaf recipe from probe attempts."""
    path = []
    for attempt in actual["attempts"]:
        policy = attempt["policy"]
        if policy.get("kind") != "recipe":
            path.append(dict(slot="ordinary", matched=True, steps=[], solver=policy))
            continue
        guard, fallback = policy["guard"], policy["fallback"]
        assert guard["feature"] == "complementary_coverage" and guard["op"] == "ge"
        assert fallback["guard"] == {"feature": "bins", "op": "le", "value": 4} and task["bins"] > 4
        matched = coverage >= guard["value"]
        path.append(dict(slot="primary", matched=matched, steps=policy["steps"] if matched else [],
                         solver=policy["solver"] if matched else None))
        if not (matched and attempt["ok"]):
            path.append(dict(slot="fallback", matched=False, steps=[], solver=None))
    return path


def verify_endpoint(task, shadow, actual, coverage):
    assert {key: shadow[key] for key in ("ok", "packing", "cost")} == {key: actual[key] for key in ("ok", "packing", "cost")}
    assert shadow["outcome"] == ("solved" if actual["ok"] else "unresolved")
    if shadow["ok"]:
        assert pairs.check(task, shadow["packing"])
    projected = [{key: attempt[key] for key in ("ok", "policy", "cost")} for attempt in shadow["attempts"]]
    assert projected == actual["attempts"] and shadow["path"] == path_for(task, actual, coverage)
    for attempt in shadow["attempts"]:
        if attempt["policy"].get("kind") == "recipe":
            guard = attempt["route"]["primary"]["guard"]
            assert guard["observed"] == coverage and guard["matched"] == (coverage >= guard["value"])


def verify_charges(row, old, new):
    unique = {}
    for attempt in old["attempts"] + new["attempts"]:
        key = pairs.canonical(attempt["policy"])
        cost = attempt["cost"]
        assert unique.setdefault(key, cost) == cost
    assert row["shadow_calls"] == len(unique) and row["shadow_spent"] == sum(unique.values())


def aggregate(rows):
    old, new = [row["old"] for row in rows], [row["new"] for row in rows]
    return dict(compared=len(rows), errors=0, old_ok=sum(r["ok"] for r in old), new_ok=sum(r["ok"] for r in new),
                gains=sum(b["ok"] and not a["ok"] for a, b in zip(old, new)),
                losses=sum(a["ok"] and not b["ok"] for a, b in zip(old, new)),
                old_cost=sum(r["cost"] for r in old), new_cost=sum(r["cost"] for r in new),
                old_success_cost=sum(r["cost"] for r in old if r["ok"]),
                new_on_old_success_cost=sum(b["cost"] for a, b in zip(old, new) if a["ok"]),
                branch_changes=sum(r["branch_changed"] for r in rows),
                shadow_calls=sum(r["shadow_calls"] for r in rows), shadow_spent=sum(r["shadow_spent"] for r in rows))


def run(state, build, payload_path):
    from candidates.pair_eval import probe as state_tools
    from candidates.pair_eval.development import coverage
    from packlab import memory, research

    assert memory.build_id() == build
    original_fingerprint = state_tools.fingerprint(state)
    original_db = db_identity(state, state_tools.source)
    with state_tools.source(state) as conn:
        decision = state_tools.project_event(conn, 12)
        revision, active = state_tools.head(conn)
    assert decision["decision"] == "inconclusive" and decision["before"] == decision["after"] == revision == 2
    assert active == decision["policies_before"] and decision["nominee"] is not None
    assert pairs.digest(pairs.canonical(decision["nominee"])) == decision["nominee_sha256"]
    candidate = research.dispatch(decision["nominee"], active)
    promise, payload = open_commitment(payload_path)
    prior = prior_keys(state, state_tools.source)
    expected = expected_order(payload, prior, research.family_id)

    with TemporaryDirectory(prefix="agentic-shadow-eval-") as temporary:
        folder = Path(temporary) / "state"
        assert state_tools.snapshot(state, folder) == (revision, active)
        assert cli(folder, "shadow", "status")["state"] == "inactive"
        fingerprint = state_tools.fingerprint(folder)
        probes = {}
        for item, (key, outcome) in zip(payload["arrivals"], expected):
            if outcome == "eligible":
                task = item["task"]
                probes[key] = (baseline_probe(folder, task, active), baseline_probe(folder, task, candidate, nominee=True))
        assert state_tools.fingerprint(folder) == fingerprint

        start = cli(folder, "shadow", "start", "12", "--limit", str(LIMIT), "--source", "observe")
        assert (start["state"], start["source"], start["event_id"], start["limit"], start["arrivals"]) == (
            "active", "observe", 12, LIMIT, 0)
        assert start["build"] == build and start["old_chain"] == active and start["new_chain"] == candidate
        assert start["nominee_sha256"] == decision["nominee_sha256"]
        rows, observed, outcomes = [], [], {}
        profiles = {name: [] for name in PLAN[:6]}
        for index, (item, (key, outcome)) in enumerate(zip(payload["arrivals"], expected), 1):
            task = item["task"]
            if outcome == "closed":
                as_closed = state_tools.fingerprint(folder)
            response = cli(folder, "shadow", "observe", "-", task=task)
            assert response["outcome"] == outcome and response["event_id"] == 12, item["label"]
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome == "eligible":
                row = response["row"]
                assert row["arrival"] == index and row["family"] == key and row["task"] == task
                assert "error" not in row and pairs.check(task, item["witness"])
                old, new = probes[key]
                verify_endpoint(task, row["old"], old, coverage(task))
                verify_endpoint(task, row["new"], new, coverage(task))
                verify_charges(row, old, new)
                assert row["branch_changed"] == (row["old"]["path"] != row["new"]["path"])
                row.pop("observed")
                rows.append(row)
                profiles[item["profile"]].append(row)
            else:
                assert "row" not in response
                if outcome == "closed":
                    assert state_tools.fingerprint(folder) == as_closed
                else:
                    assert response["family"] == key
            observed.append(dict(index=index, label=item["label"], outcome=outcome, family=key))
        assert outcomes == promise["planned"]

        closed = state_tools.fingerprint(folder)
        status = cli(folder, "shadow", "status")
        assert status["state"] == "closed" and status["eligible"] == LIMIT
        assert (status["arrivals"], status["prior"], status["duplicates"]) == (
            len(observed)-1, outcomes["prior"], outcomes["duplicate"])
        persisted = [{key: value for key, value in row.items() if key != "observed"} for row in status["rows"]]
        assert persisted == rows
        for key, count in aggregate(rows).items():
            assert status[key] == count
        assert state_tools.fingerprint(folder) == closed
        assert db_identity(folder, state_tools.source) == original_db
    assert state_tools.fingerprint(state) == original_fingerprint
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    status.pop("started")
    status.pop("rows")
    return {"scope": "precommitted synthetic observe stream; no estimate of real workload frequency",
            "commitment": promise, "release_build": build, "research_event": decision,
            "state_source_unchanged": True, "original_events_and_head_unchanged": True,
            "status": status, "observed": observed, "rows": rows,
            "profiles": {name: aggregate(selected) for name, selected in profiles.items()}, "reveal": payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("seal")
    evaluation = actions.add_parser("run")
    evaluation.add_argument("--state", type=Path, required=True)
    evaluation.add_argument("--build", required=True)
    evaluation.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        seal()
    else:
        print(json.dumps(run(args.state.resolve(), args.build, args.payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
