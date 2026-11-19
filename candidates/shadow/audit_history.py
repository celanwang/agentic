"""Recompute public round-12 history aggregates without reading local state."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISION = "evidence/route-review/decision.json"
COVERAGE = "evidence/route-review/coverage.json"
PROBE = "candidates/route_review/eval/probe_result.json"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def audit():
    decision = read(DECISION)
    coverage = read(COVERAGE)
    released = read(PROBE)
    event = released["event"]
    old, new = event["policies_before"][0], event["nominee"]
    first, second = old["guard"], new["guard"]
    assert decision["record"] == event["id"] == 12
    assert decision["nominee_sha256"] == event["nominee_sha256"]
    assert decision["build"] == released["release_build"] and released["database_bytes_unchanged"]
    assert new == decision["program"] and dict(old, guard=second) == new
    assert first["feature"] == second["feature"] == "complementary_coverage"
    assert first["op"] == second["op"] == "ge" and second["value"] < first["value"]

    ordinary = {}
    for name, result in decision["gate"].items():
        paths = coverage[name]
        assert result["total"] == paths["total"]
        assert paths["changed_paths"] == 0
        # Only the numeric cutoff changed. Equal paths therefore execute the
        # same observations, comparisons, solvers and fallback work per row.
        assert result["old_cost"] == result["new_cost"] and result["losses"] == result["gains"] == 0
        ordinary[name] = dict(inputs=result["total"], additional_first_branch_hits=0,
                              old_successes=result["old_ok"], old_success_extra_ops=0,
                              before_ops=result["old_cost"], candidate_ops=result["new_cost"])

    by_profile = {}
    by_change = {}
    targeted = Counter()
    for row in released["rows"]:
        before, active, candidate = row["before"], row["head"], row["nominee"]
        assert before == active
        assert before["attempts"][0]["policy"] == old and candidate["attempts"][0]["policy"] == new
        guard = candidate["trace"]["guard"]
        assert {k: guard[k] for k in second} == second
        observed = guard["observed"]
        old_hit = observed >= first["value"]
        new_hit = observed >= second["value"]
        assert guard["matched"] == new_hit
        old_success = before["ok"]
        gained = candidate["ok"] and not old_success
        lost = old_success and not candidate["ok"]
        delta = candidate["cost"] - before["cost"]
        metrics = dict(inputs=1, old_first_branch_hits=int(old_hit), candidate_first_branch_hits=int(new_hit),
                       additional_first_branch_hits=int(new_hit and not old_hit),
                       old_successes=int(old_success), candidate_successes=int(candidate["ok"]),
                       gains=int(gained), losses=int(lost), before_ops=before["cost"],
                       candidate_ops=candidate["cost"], extra_ops=delta,
                       old_success_extra_ops=delta if old_success else 0)
        targeted.update(metrics)
        by_profile.setdefault(row["profile"], Counter()).update(metrics)
        if new_hit and not old_hit:
            name = "old_success" if old_success else "old_failure"
        else:
            name = "unchanged_first_branch"
        by_change.setdefault(name, Counter()).update(metrics)

    return {
        "round": 12,
        "candidate_decision": decision["decision"],
        "new_hit_interval_permille": [second["value"], first["value"] - 1],
        "controlled_gate": ordinary,
        "targeted_published_probes": {"total": targeted, "by_profile": by_profile,
                                     "by_first_branch_change": by_change},
        "provenance": {
            "fixed_continuous_demo": {"source": "evidence/pilot/stream-protocol.md",
                                     "inputs": 8, "origin": "preselected_public_examples"},
            "ordinary_holdout": {"source": "evidence/route-review/protocol.md",
                                 "origin": "arena_generator_with_prior_family_exclusions"},
            "targeted_probe": {"source": "candidates/route_review/eval/seal.py",
                               "origin": "five_predefined_profiles_across_5_to_8_bins"},
            "verified_public_organic_user_streams": 0,
            "organic_prevalence_identifiable": False,
            "unverified_origin_is_excluded": True,
        },
        "public_inputs": [DECISION, COVERAGE, PROBE],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
