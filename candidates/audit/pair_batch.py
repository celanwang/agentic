"""Exercise batch input, abstention, and isolation for families outside practice."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidates.audit.pair_boundary import manifests
from packlab import memory, research


class BatchInput(unittest.TestCase):
    def test_reject_invalid_batch_before_state_creation(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "research"
            command = [str(ROOT / "agentic"), "--state", str(path), "research", "--batch", "-"]
            for value in (None, {}, [], [None], [dict(capacity=1, bins=1, items=[])] * 65):
                result = subprocess.run(command, input=json.dumps(value), text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(path.exists())

    def test_low_budget_preserves_head_and_excludes_even_unsampled_tasks_next_round(self):
        batch = [t for t, _ in manifests() if t["bins"] >= 5]
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            event = memory.investigate(folder, budget=2, batch=batch)
            self.assertEqual(event["decision"], "inconclusive")
            self.assertLessEqual(event["search"]["calls"], 2)
            self.assertNotIn("holdout", event)
            self.assertEqual(memory.status(folder)["generation"], 0)
            keys = {research.family_id(t) for t in batch}
            practice = {research.family_id(t) for t in event["practice_tasks"]}
            self.assertEqual(len(keys - practice), 34)
            memory.investigate(folder, budget=2)
            with sqlite3.connect(folder / "agentic.sqlite3") as db:
                stored = set(r[0] for r in db.execute("SELECT family FROM research_inputs"))
            self.assertTrue(keys <= stored)


if __name__ == "__main__":
    unittest.main()
