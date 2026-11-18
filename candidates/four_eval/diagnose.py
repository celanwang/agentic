"""Reproduce four published v2 failures and verify their packing witnesses."""

from collections import defaultdict, deque
from contextlib import closing
import json
import subprocess
from unittest.mock import patch

from candidates.triples_eval import blind_probe as common
from packlab import arena
from .blind_mixed import POLICIES, V2_STATE, V2_SHA, assert_frozen_sources, v2


def historical_failures():
    with closing(common.source_database(V2_STATE)) as db:
        seed, raw, policies = db.execute("""SELECT json_extract(data,'$.seed'),
                         json_extract(data,'$.holdout.rows'), (SELECT policies FROM revisions WHERE id=2)
                         FROM events WHERE id=4 AND decision='kept'""").fetchone()
    assert seed == 282459 and json.loads(policies) == POLICIES
    result = []
    occurrence = 0
    for number, row in enumerate(json.loads(raw)):
        if row["feasible"] and not row["new_ok"]:
            assert row["profile"] == "mixed"
            result.append((number, occurrence, row))
        occurrence += row["profile"] == "mixed"
    assert [number for number, _, _ in result] == [20, 23, 25, 28]
    return seed, result


def construction_witness(task, seed, occurrence):
    captured = []
    shuffle = arena._shuffled

    def record(rng, capacity, disks):
        created = shuffle(rng, capacity, disks)
        assert created == task
        positions = defaultdict(deque)
        for index, size in enumerate(task["items"]):
            positions[size].append(index)
        captured.extend([[positions[size].popleft() for size in disk] for disk in disks])
        return created

    with patch.object(arena, "_shuffled", record):
        arena._mixed(arena._rng(seed, "holdout", "mixed", occurrence))
    assert common.check(task, captured)
    return captured


def prefix_proof(task):
    """Relax each disk independently. Failure certifies this frozen prefix."""
    files, capacity, bins = task["items"], task["capacity"], task["bins"]
    order = sorted(range(len(files)), key=files.__getitem__, reverse=True)
    free = [capacity] * bins
    assignments = []
    slack = bins * capacity - sum(files)
    mask = (1 << (capacity + 1)) - 1
    updates = 0
    # The suffix bitsets are disjoint from the packing certificate.
    reachable = [0] * (len(files) + 1)
    reachable[-1] = 1
    for step in range(len(files) - 1, -1, -1):
        suffix = reachable[step + 1]
        reachable[step] = (suffix | suffix << files[order[step]]) & mask
        updates += 1
    for step, index in enumerate(order):
        best = [(reachable[step] & ((1 << (gap + 1)) - 1)).bit_length() - 1 for gap in free]
        if sum(gap - fill for gap, fill in zip(free, best)) > slack:
            return dict(result="proven_no_extension", placed=step, last_placement=assignments[-1],
                        next_original_index=index, free=free, max_fill=best, total_slack=slack,
                        unused_lower_bound=sum(free)-sum(best),
                        smallest_remaining=files[order[-1]], bitset_updates=updates,
                        suffix_capacity_bits=capacity+1)
        for disk, gap in enumerate(free):
            if files[index] <= gap:
                free[disk] -= files[index]
                assignments.append(dict(original_index=index, size=files[index], disk=disk))
                break
        else:
            return dict(result="unknown_before_observed_block", placed=step,
                        next_original_index=index, free=free, max_fill=best, total_slack=slack,
                        unused_lower_bound=sum(free)-sum(best), bitset_updates=updates,
                        suffix_capacity_bits=capacity+1)
    raise AssertionError("historical failure unexpectedly packed")


def inspect(task, historical):
    command = [str(common.PROJECT / "agentic"), "--state", str(V2_STATE), "probe", "-", "--json"]
    observed = json.loads(subprocess.check_output(command, input=common.canonical(task), timeout=30))
    assert observed["task"] == task and observed["policies"] == POLICIES
    assert not observed["result"]["ok"] and observed["result"]["cost"] == historical["new_cost"]
    observations = []
    for policy, attempt in zip(POLICIES, observed["result"]["attempts"]):
        frozen = v2.run(task, policy)
        trace = frozen["trace"]
        assert policy == attempt["policy"] and attempt["ok"] is False and frozen["ok"] is False
        assert attempt["cost"] == frozen["cost"] and trace["blocked"]["reason"] == "fragmentation"
        assert sum(trace["blocked"]["free"]) >= trace["blocked"]["size"] > max(trace["blocked"]["free"])
        observations.append(dict(policy=policy, cost=frozen["cost"], **trace))
    assert len(observations) == len(POLICIES)
    assert observations[-1]["search_status"] == "not_applicable"
    assert observations[-1]["blocked"] == observed["result"]["trace"]["blocked"]
    assert observations[-1]["ops"]["shape_checks"] == 1 and len(task["items"]) != task["bins"] * 3
    return observations


def main():
    assert_frozen_sources()
    seed, rows = historical_failures()
    evidence = []
    for number, occurrence, historical in rows:
        task = historical["task"]
        witness = construction_witness(task, seed, occurrence)
        evidence.append(dict(row=number, task=task, witness=witness, attempts=inspect(task, historical),
                             first_descending_prefix_proof=prefix_proof(task)))
    print(json.dumps({"event": 4, "seed": seed, "frozen_engine_sha256": V2_SHA, "rows": evidence}, indent=2))


if __name__ == "__main__":
    main()
