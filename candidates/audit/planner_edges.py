"""Check resumable eviction at real and reduced memory limits, using practice tasks only."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from packlab import engine, planner


def stored(value):
    return json.loads(json.dumps(value, sort_keys=True))


def panel(start):
    return [dict(capacity=10 * k, bins=2, items=[size * k for size in sizes])
            for sizes in ([4, 4, 6, 6], [6, 4, 6, 4]) for k in range(start, start + 3)]


def episode(start, memory, seed, budget, mode="adaptive"):
    return planner.search(panel(start), [engine.BASE], [], memory, seed, budget, mode=mode)


class PlannerEdges(unittest.TestCase):
    def test_resume_across_real_cache_eviction(self):
        memory = episode(1, None, 5, 128)["memory"]
        self.assertEqual(memory, stored(memory))
        live = episode(4, memory, 6, 128)
        restarted = episode(4, stored(memory), 6, 128)
        self.assertEqual(len(live["memory"]["recent"]), planner.CACHE_LIMIT)
        self.assertTrue(memory["recent"].keys() - live["memory"]["recent"].keys())
        self.assertEqual(live["memory"], restarted["memory"])
        a = episode(1, live["memory"], 0, 24)
        b = episode(1, stored(restarted["memory"]), 0, 24)
        self.assertEqual(a, b)
        self.assertEqual(a["calls"], 24)
        # Prior released SQLite behavior for this reproduction.
        self.assertEqual((a["cached_pairs"], a["spent"], a["nominee"]),
                         (59, 992, "4e5826e2157364dfcb5c77f2"))

    def test_credit_and_raw_results_at_eviction_boundary(self):
        # The limit is enforced only when saving. Read the untrimmed ordering
        # as an independent reference for which eight entries must survive.
        full = episode(1, None, 5, 128)["memory"]
        with patch.multiple(planner, STAT_LIMIT=8, CACHE_LIMIT=8):
            short = episode(1, None, 5, 128)["memory"]
            for field in ("stats", "recent"):
                expected = list(full[field])[-8:]
                self.assertEqual(list(short[field]), expected)
                self.assertEqual(list(planner._restore_entries(short, field)), sorted(expected))
            live = episode(4, short, 6, 24)
            restarted = episode(4, stored(short), 6, 24)
            self.assertEqual(live, restarted)

    def test_legacy_snapshot_and_reset(self):
        memory = episode(1, None, 5, 128)["memory"]
        legacy = dict(memory)
        legacy["stats"] = {key: tuple(pair) for key, pair in legacy["stats"].items()}
        live = episode(1, legacy, 6, 24)
        self.assertEqual(live, episode(1, stored(legacy), 6, 24))
        self.assertGreater(live["cached_pairs"], 0)
        untouched = episode(1, stored(legacy), 7, 0)
        self.assertEqual(untouched["calls"], 0)
        for field in ("stats", "recent"):
            self.assertEqual(untouched["memory"][field], stored(legacy[field]))
        reset = episode(1, stored(memory), 7, 0, "adaptive_reset")["memory"]
        self.assertEqual(reset["archive"], memory["archive"])
        self.assertFalse(reset["stats"] or reset["recent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
