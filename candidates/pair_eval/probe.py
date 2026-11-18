"""Verify the sealed pair tasks against a completed research event by read-only CLI probes."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory

from .development import coverage
from . import seal


DB = "agentic.sqlite3"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(folder):
    db = (folder.resolve() / DB)
    return closing(sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True))


def head(db):
    revision, raw = db.execute("SELECT r.id,r.policies FROM revisions r JOIN head h ON h.revision=r.id WHERE h.slot=1").fetchone()
    return revision, json.loads(raw)


def project_event(db, event):
    row = db.execute("""SELECT decision, json_extract(data,'$.build'), json_extract(data,'$.generation_before'),
                               json_extract(data,'$.generation'), json_extract(data,'$.nominee'),
                               json_extract(data,'$.nominee_sha256'), json_extract(data,'$.policies_before'),
                               json_extract(data,'$.rules') FROM events WHERE id=? AND kind='research'""",
                     (event,)).fetchone()
    if row is None or db.execute("SELECT max(id) FROM events").fetchone() != (event,):
        raise ValueError("the requested research event must be present at the end of the supplied state")
    decision, build, before, after, nominee, sha, policies, rules = row
    return dict(id=event, decision=decision, build=build, before=before, after=after, nominee_sha256=sha,
                nominee=json.loads(nominee) if nominee else None, policies_before=json.loads(policies), rules=json.loads(rules))


def fingerprint(folder):
    return {path.name: seal.digest(path.read_bytes()) for path in folder.glob(DB + "*") if path.is_file()}


def snapshot(folder, target, revision=None):
    target.mkdir()
    with source(folder) as original, closing(sqlite3.connect(target / DB)) as copy:
        original.backup(copy)
        if revision is not None:
            with copy:
                copy.execute("UPDATE head SET revision=? WHERE slot=1", (revision,))
        return head(copy)


def committed_payload(path):
    published = read(seal.COMMITMENT)
    payload = read(path)
    if "reveal" in payload:
        payload = payload["reveal"]
    assert published["sampler_sha256"] == seal.digest(Path(seal.__file__).read_bytes())
    assert published["commitment"] == seal.digest(seal.canonical(payload))
    assert payload == seal.construct(payload["nonce"])
    return published["commitment"], payload


def public_tasks():
    from candidates.audit.pair_boundary import manifests

    for task, _ in manifests():
        yield task
    yield read(seal.ROOT / "examples/pilot-pairs-sealed.json")["task"]
    for name in ("triples_eval", "four_eval"):
        yield read(seal.ROOT / "candidates" / name / "blind_result.json")["sealed"]["task"]
    pilot = read(seal.ROOT / "candidates/pilot_eval/blind_result.json")
    for group in pilot["reveal"]["clusters"]:
        for case in group["cases"]:
            yield case["task"]


def compare_keys(state, event, payload, research):
    def key(task):
        bins, capacity, items = seal.core(task)
        independent = seal.digest(seal.canonical([capacity, bins, items]))
        assert independent == research.family_id(task)
        return independent

    keys = {key(case["task"]) for case in payload["cases"]}
    assert len(keys) == len(payload["cases"])
    public = {key(task) for task in public_tasks()}
    prior, formal = set(public), set()
    with source(state) as db:
        for event_id, direct in db.execute("SELECT id,task FROM events WHERE task IS NOT NULL"):
            task = json.loads(direct)
            if task is not None:
                (formal if event_id == event else prior).add(key(task))
        for field in ("practice_tasks", "batch_tasks", "holdout.rows", "regression.rows", "replay.rows"):
            path = "$." + field
            for event_id, raw in db.execute("SELECT e.id,j.value FROM events e, json_each(e.data,?) j", (path,)):
                row = json.loads(raw)
                (formal if event_id == event else prior).add(key(row["task"] if field.endswith(".rows") else row))
    if keys & (prior | formal):
        raise ValueError("a sealed core overlaps a public, training, or acceptance input")
    return dict(identity="forced singletons removed, GCD normalized, multiset sorted; agrees with research.family_id",
                sealed_cores=len(keys), prior_cores=len(prior), event_cores=len(formal), overlap=0)


def probe(folder, task, policies, nominee=None):
    args = [str(seal.ROOT / "agentic"), "--state", str(folder), "probe", "-", "--json"]
    if nominee is not None:
        args.extend(("--nominee", str(nominee)))
    child = subprocess.run(args, input=seal.canonical(task), capture_output=True, timeout=30)
    if child.returncode:
        raise RuntimeError(child.stderr.decode("utf-8", errors="replace").strip())
    outcome = json.loads(child.stdout)
    assert outcome["task"] == task and outcome["policies"] == policies
    result = outcome["result"]
    assert type(result["ok"]) is bool and type(result["cost"]) is int
    assert result["cost"] > 0 and result["cost"] == sum(a["cost"] for a in result["attempts"])
    if result["ok"]:
        assert seal.check(task, result["packing"])
    return result


def summarize(rows):
    result = {}
    for arm in ("old", "nominee", "head"):
        paired = [(row["old"], row[arm]) for row in rows if row[arm] is not None]
        result[arm] = dict(cases=len(paired), ok=sum(b["ok"] for a, b in paired),
                           cost=sum(b["cost"] for a, b in paired),
                           gains=sum(b["ok"] and not a["ok"] for a, b in paired),
                           losses=sum(a["ok"] and not b["ok"] for a, b in paired))
    return result


def run(state, event_id, build, path, before_state):
    from packlab import memory, research
    from packlab.experiment import RULES, ident
    from packlab.recipes import route_features

    assert memory.build_id() == build
    with source(state) as db:
        record = project_event(db, event_id)
        revision, active = head(db)
        prior, = db.execute("SELECT policies FROM revisions WHERE id=?", (record["before"],)).fetchone()
    old = json.loads(prior)
    assert record["build"] == build and record["rules"] == RULES and revision == record["after"]
    assert record["policies_before"] == old
    program = record["nominee"]
    candidate = research.dispatch(program, old) if program is not None else None
    if program is not None:
        assert seal.digest(ident(program).encode("utf-8")) == record["nominee_sha256"]
    if record["decision"] == "kept":
        assert active == candidate and revision != record["before"]
    else:
        assert revision == record["before"] and active == old
    if before_state is not None:
        with source(before_state) as db:
            assert head(db) == (record["before"], old)
            assert db.execute("SELECT max(id) FROM events").fetchone() == (event_id - 1,)

    commitment, payload = committed_payload(path)
    dedupe = compare_keys(state, event_id, payload, research)
    initial = {folder: fingerprint(folder) for folder in (state, before_state) if folder is not None}
    rows = []
    with TemporaryDirectory(prefix=".pair-probe-", dir=seal.HERE) as temporary:
        workspace = Path(temporary)
        old_folder, active_folder = workspace / "old", workspace / "active"
        assert snapshot(before_state or state, old_folder, None if before_state else record["before"]) == (record["before"], old)
        assert snapshot(state, active_folder) == (revision, active)
        snapshots = {folder: fingerprint(folder) for folder in (old_folder, active_folder)}
        for case in payload["cases"]:
            task = case["task"]
            score = coverage(task)
            assert route_features(task)["complementary_coverage"] == score
            rows.append(dict(name=case["name"], coverage=score, matching=case["matching"],
                             old=probe(old_folder, task, old), head=probe(active_folder, task, active),
                             nominee=probe(active_folder, task, candidate, event_id) if candidate is not None else None))
        assert snapshots == {folder: fingerprint(folder) for folder in snapshots}
    assert initial == {folder: fingerprint(folder) for folder in initial}
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    return {"commitment": commitment, "release_build": build, "event": record, "source_unchanged": True,
            "dedupe": dedupe, "summary": summarize(rows), "rows": rows, "reveal": payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--event", type=int, default=11)
    parser.add_argument("--build", required=True)
    parser.add_argument("--payload", type=Path, default=seal.PRIVATE / "payload.json")
    parser.add_argument("--before", type=Path, help="optional independently saved pre-event state")
    args = parser.parse_args()
    print(json.dumps(run(args.state.resolve(), args.event, args.build, args.payload,
                         args.before.resolve() if args.before else None), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
