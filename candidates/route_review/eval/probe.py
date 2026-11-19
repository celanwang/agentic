"""Open the committed routing suite after a completed research decision."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from candidates.pair_eval import probe as previous, seal as pairs

from .frozen import copy_runtime
from .seal import COMMITMENT, HERE, PRIVATE, ROOT, construct


ARMS = ("historical", "before", "nominee", "head")


def committed_payload(path):
    record = previous.read(COMMITMENT)
    payload = previous.read(path)
    if "reveal" in payload:
        payload = payload["reveal"]
    assert record["sampler_sha256"] == pairs.digest((HERE / "seal.py").read_bytes())
    assert record["commitment"] == pairs.digest(pairs.canonical(payload))
    assert payload == construct(payload["nonce"]) and len(payload["cases"]) == record["cases"]
    return record["commitment"], payload


def nonoverlap(state, event, payload, research):
    def key(task):
        bins, cap, sizes = pairs.core(task)
        independent = pairs.digest(pairs.canonical([cap, bins, sizes]))
        assert independent == research.family_id(task)
        return independent

    blind = {key(row["task"]) for row in payload["cases"]}
    assert len(blind) == len(payload["cases"])
    public = {key(task) for task in previous.public_tasks()}
    published_pairs = previous.read(ROOT / "candidates/pair_eval/probe_result.json")
    public.update(key(row["task"]) for row in published_pairs["reveal"]["cases"])
    past, current = set(), set()
    with previous.source(state) as db:
        for event_id, raw in db.execute("SELECT id,task FROM events WHERE task IS NOT NULL AND task <> 'null'"):
            (current if event_id == event else past).add(key(json.loads(raw)))
        for name in ("practice_tasks", "batch_tasks", "holdout.rows", "regression.rows", "replay.rows"):
            for event_id, raw in db.execute("SELECT e.id,j.value FROM events e, json_each(e.data,?) j", ("$." + name,)):
                value = json.loads(raw)
                (current if event_id == event else past).add(key(value["task"] if name.endswith(".rows") else value))
    overlap = blind & (past | current | public)
    assert not overlap
    return dict(blind_keys=len(blind), prior_keys=len(past | public), event_keys=len(current), overlap=len(overlap),
                rule="exact independent normalized keys checked against research.family_id")


def cli(executable, state, *args, task=None):
    result = subprocess.run([sys.executable, "-B", str(executable), "--state", str(state), *args, "--json"],
                            input=pairs.canonical(task) if task is not None else None, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return json.loads(result.stdout)


def checked_probe(executable, state, task, policies, event=None):
    flags = ("--nominee", str(event)) if event is not None else ()
    response = cli(executable, state, "probe", "-", *flags, task=task)
    assert response["task"] == task and response["policies"] == policies
    result = response["result"]
    assert type(result["ok"]) is bool and type(result["cost"]) is int
    assert result["cost"] > 0 and result["cost"] == sum(attempt["cost"] for attempt in result["attempts"])
    assert not result["ok"] or pairs.check(task, result["packing"])
    blocked = result["trace"]["blocked"] or {}
    assert blocked.get("reason") not in {"oversize", "total_excess"}
    assert result["trace"]["search_status"] != "infeasible"
    return result


def summarize(rows, reference):
    result = {}
    for arm in ARMS:
        compared = [(row[reference], row[arm]) for row in rows if row[arm] is not None]
        result[arm] = dict(cases=len(compared), ok=sum(b["ok"] for a, b in compared), cost=sum(b["cost"] for a, b in compared),
                           gains=sum(b["ok"] and not a["ok"] for a, b in compared),
                           losses=sum(a["ok"] and not b["ok"] for a, b in compared))
    return result


def run(before, state, event, build, payload_path):
    from packlab import memory, research
    from packlab.experiment import RULES, ident

    assert memory.build_id() == build
    source_fingerprints = {folder: previous.fingerprint(folder) for folder in (before, state)}
    with previous.source(before) as db:
        assert db.execute("SELECT max(id) FROM events").fetchone() == (event - 1,)
        old_revision, old = previous.head(db)
    with previous.source(state) as db:
        record = previous.project_event(db, event)
        active_revision, active = previous.head(db)
    assert record["build"] == build and record["rules"] == RULES
    assert record["before"] == old_revision and record["after"] == active_revision and record["policies_before"] == old
    nominee = record["nominee"]
    candidate = research.dispatch(nominee, old) if nominee is not None else None
    if nominee is not None:
        assert pairs.digest(ident(nominee).encode("utf-8")) == record["nominee_sha256"]
    if record["decision"] == "kept":
        assert active == candidate and old_revision != active_revision
    else:
        assert active == old and old_revision == active_revision

    commitment, payload = committed_payload(payload_path)
    dedupe = nonoverlap(state, event, payload, research)
    rows = []
    with TemporaryDirectory(prefix=".route-probe-", dir=HERE) as temporary:
        root = Path(temporary)
        old_runtime, old_state, new_state = root / "runtime", root / "before", root / "after"
        historic_build = copy_runtime(old_runtime)
        assert previous.snapshot(before, old_state) == (old_revision, old)
        assert previous.snapshot(state, new_state) == (active_revision, active)
        fingerprints = {folder: previous.fingerprint(folder) for folder in (old_state, new_state)}
        historic_cli, current_cli = old_runtime / "agentic", ROOT / "agentic"
        assert cli(historic_cli, old_state, "status")["build"] == historic_build
        assert cli(current_cli, new_state, "status")["build"] == build
        for row in payload["cases"]:
            task = row["task"]
            rows.append(dict(name=row["name"], profile=row["profile"],
                             historical=checked_probe(historic_cli, old_state, task, old),
                             before=checked_probe(current_cli, old_state, task, old),
                             nominee=checked_probe(current_cli, new_state, task, candidate, event) if candidate else None,
                             head=checked_probe(current_cli, new_state, task, active)))
        assert fingerprints == {folder: previous.fingerprint(folder) for folder in fingerprints}
    assert source_fingerprints == {folder: previous.fingerprint(folder) for folder in source_fingerprints}
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    if active == old:
        assert all(row["before"] == row["head"] for row in rows)
    if candidate is not None and active == candidate:
        assert all(row["nominee"] == row["head"] for row in rows)
    return {"commitment": commitment, "historical_build": historic_build, "release_build": build, "event": record,
            "database_bytes_unchanged": True, "dedupe": dedupe, "summary": summarize(rows, "before"),
            "relative_to_historical": summarize(rows, "historical"),
            "profiles": {profile: summarize([row for row in rows if row["profile"] == profile], "before")
                         for profile in sorted({row["profile"] for row in rows})}, "rows": rows, "reveal": payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--event", type=int, default=12)
    parser.add_argument("--build", required=True)
    parser.add_argument("--payload", type=Path, default=PRIVATE / "payload.json")
    args = parser.parse_args()
    print(json.dumps(run(args.before.resolve(), args.state.resolve(), args.event, args.build, args.payload),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
