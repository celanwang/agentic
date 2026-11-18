"""Compare live and canonically persisted planner histories on the frozen learning benchmark."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from candidates.pilot_eval import learning_eval as frozen, suite
from packlab import planner, research
from packlab.experiment import Bench, ident


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe(tasks, policies, experience, memory, seed, mode):
    # Retain the actual in-process object before the frozen observer's JSON copy.
    result = None
    search = planner.search

    def capture(*args, **kwargs):
        nonlocal result
        result = search(*args, **kwargs)
        return result

    with patch.object(planner, "search", capture):
        measured, _ = frozen.observe_search(planner, tasks, policies, experience, memory, seed, mode)
    assert measured["calls"] == suite.CALL_BUDGET
    assert measured["baseline_calls"] == 6 and measured["candidate_calls"] == 122
    return result, measured


def exported(memory, variant):
    if variant != "lru":
        return memory
    # Experimental sidecar: keep within-episode eviction order across episodes.
    return {**memory, "audit_order": {k: list(memory[k]) for k in ("stats", "recent")}}


def lru_entries(memory, name):
    values = memory.get(name, {})
    return {key: values[key] for key in memory.get("audit_order", {}).get(name, values)}


def compare(folder, details, variant):
    here = Path(frozen.__file__).parent
    released = json.loads((here / "learning_result.json").read_text())
    binding = json.loads(frozen.BINDING.read_text())
    policies, experience = binding["policies"], binding["experience"]
    assert file_digest(frozen.BINDING) == released["binding_sha256"]
    assert file_digest(Path(frozen.__file__)) == binding["runner_sha256"]
    assert file_digest(here / "protocol.json") == binding["protocol_sha256"]
    assert file_digest(Path(suite.__file__)) == json.loads((here / "blind_commitment.json").read_text())["suite_sha256"]
    assert (policies, experience) == (released["v2_policies"], released["experience"])
    results, changes = [], []
    for lesson in suite.learning():
        index = lesson["index"]
        states = dict.fromkeys(frozen.ARMS)
        rounds = []
        for offset, row in enumerate(lesson["rounds"]):
            number = row["round"]
            tasks = [entry["task"] for entry in row["train"]]
            reference = released["seeds"][index]["rounds"][offset]
            seed = int.from_bytes(suite.sha256(suite.canonical([lesson["seed"], number])).digest()[:8], "big")
            assert reference["planner_seed"] == seed and reference["round"] == number
            arms = {}
            for arm, mode in frozen.ARMS.items():
                path = folder / f"{index}-{arm}.json"
                restored = json.loads(path.read_text()) if path.exists() else None
                live, live_stats = observe(tasks, policies, experience, states[arm], seed, mode)
                disk, disk_stats = observe(tasks, policies, experience, restored, seed, mode)
                assert live == disk, (index, number, arm, "planner report or memory mismatch")
                if arm != "reset":
                    states[arm] = exported(live["memory"], variant)
                    disk_state = exported(disk["memory"], variant)
                    assert states[arm] == disk_state
                    path.write_text(ident(disk_state), encoding="utf-8")
                for stats in (live_stats, disk_stats):
                    stats.pop("elapsed_ms")
                    rows = frozen.checkpoint(row["checkpoint"], policies, stats["program"], Bench(), research.dispatch)
                    stats.update(checkpoint=rows, summary=frozen.summarize(rows))
                assert live_stats == disk_stats, (index, number, arm, "checkpoint mismatch")
                previous = reference["arms"][arm]
                different = [key for key in live_stats if live_stats[key] != previous[key]]
                if different:
                    changes.append(dict(seed_index=index, round=number, arm=arm, fields=different))
                arms[arm] = live_stats
            rounds.append(dict(round=number, matched=True, arms=arms))
        results.append(dict(index=index, rounds=rounds))
    current = frozen.primary(results)
    assert current["matched_rounds"] == current["all_rounds"] == 48
    old = released["primary"]["totals"]
    delta = {arm: {key: val - old[arm][key] for key, val in total.items()}
             for arm, total in current["totals"].items()}
    output = dict(planner_sha256=file_digest(Path(planner.__file__)), storage_comparisons=48 * len(frozen.ARMS),
                  equal_storage_reports=True, current_both_storage_modes=current, release=old,
                  delta=delta, changed_release_arm_rounds=changes)
    if details:
        output["seeds"] = results
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="include every arm, round and checkpoint row")
    parser.add_argument("--boundary", choices=("production", "lru", "both"), default="both")
    args = parser.parse_args()
    reports = {}
    for variant in (("production", "lru") if args.boundary == "both" else (args.boundary,)):
        with TemporaryDirectory(prefix="planner-roundtrip-") as tmp:
            use_order = patch.object(planner, "_restore_entries", lru_entries) if variant == "lru" else nullcontext()
            with use_order:
                reports[variant] = compare(Path(tmp), args.details, variant)
    print(json.dumps(reports, indent=2))
