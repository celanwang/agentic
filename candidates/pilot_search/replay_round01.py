"""Replay only the recorded practice phase of round-01's ninth event."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "state" / "pilot" / "round-01" / "stream" / "agentic.sqlite3"
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from packlab import planner


def practice_snapshot():
    with closing(sqlite3.connect(f"{DB.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
        memory, = db.execute("SELECT json_extract(data, '$.search.memory') FROM events WHERE id=8").fetchone()
        practice, skills, policies, seed, budget, expected = db.execute("""
            SELECT json_extract(data, '$.practice_tasks'), json_extract(data, '$.skill_sources'),
                   json_extract(data, '$.policies_before'), json_extract(data, '$.seed'),
                   json_extract(data, '$.budget'), json_extract(data, '$.search.nominee')
            FROM events WHERE id=9
        """).fetchone()
    if budget != 128:
        raise RuntimeError(f"unexpected recorded practice budget {budget!r}")
    return (json.loads(memory), json.loads(practice), [s["solver"] for s in json.loads(skills)],
            json.loads(policies), seed, budget, expected)


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def positions(result, candidate):
    if candidate is None:
        return None
    calls = [index for index, row in enumerate(result["trials"], 1) if row["candidate"] == candidate]
    return dict(first_candidate_call=calls[0] if calls else None,
                last_candidate_call=calls[-1] if calls else None,
                first_total_call=result["baseline_calls"] + calls[0] if calls else None,
                actual_candidate_calls=len(calls),
                full_panel=candidate in result["comparisons"])


def summarize(result, expected):
    nominee = result["nominee"]
    score = result["comparisons"].get(nominee)
    if result["calls"] != result["baseline_calls"] + result["candidate_calls"]:
        raise RuntimeError("candidate and baseline counts do not match total calls")
    if result["candidate_calls"] != len(result["trials"]) or result["calls"] > result["budget"]:
        raise RuntimeError("candidate trace and actual budget disagree")
    return dict(calls=result["calls"], baseline_calls=result["baseline_calls"],
                candidate_calls=result["candidate_calls"], cached_pairs=result["cached_pairs"],
                spent=result["spent"], nominee_id=nominee,
                nominee_sha256=digest(result["program"]) if nominee is not None else None,
                program=result["program"], reproduce_event_9=(nominee == expected),
                practice={k: score[k] for k in ("total", "gains", "losses", "old_cost", "new_cost")} if score else None,
                exact_nominee_positions=positions(result, nominee),
                event_9_nominee_positions=positions(result, expected))


def main():
    before = DB.stat()
    memory, practice, experience, policies, seed, budget, expected = practice_snapshot()
    definitions = (("history_adaptive", memory, "adaptive"),
                   ("same_archive_reset", memory, "adaptive_reset"),
                   ("cold_adaptive", None, "adaptive"),
                   ("history_uniform", memory, "uniform"))
    summary = {}
    for label, arm_memory, mode in definitions:
        result = planner.search(practice, policies, experience, arm_memory, seed, budget, mode=mode)
        summary[label] = summarize(result, expected)
    after = DB.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source database changed during the replay")
    print(json.dumps(dict(source=str(DB.relative_to(ROOT)), memory_event=8, practice_event=9,
                          planner_sha256=hashlib.sha256(Path(planner.__file__).read_bytes()).hexdigest(),
                          seed=seed, budget=budget, practice_tasks=len(practice), skill_count=len(experience),
                          event_9_nominee_id=expected, arms=summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
