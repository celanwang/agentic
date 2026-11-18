"""Check durable answers and separation of observed inputs from the official gate."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from packlab import arena, memory, research
from scripts import frozen_pilot


def command(folder, cmd, task, *options, root=ROOT):
    output = subprocess.check_output([str(root / "agentic"), "--state", str(folder), cmd, "-", *options, "--json"],
                                     input=json.dumps(task), text=True)
    return json.loads(output)


class HistoryEdges(unittest.TestCase):
    def test_unpromoted_packing_survives_earlier_failure_and_explicit_research(self):
        task = json.loads((ROOT / "examples/first.json").read_text())
        with TemporaryDirectory() as folder, TemporaryDirectory() as release:
            published = Path(release)
            frozen_pilot.materialize(published)
            attempts = [command(folder, "research", task, root=published) for _ in range(3)]
            self.assertIsNone(attempts[0]["answer"])
            self.assertEqual(attempts[2]["answer"]["source"], "trial")
            out = command(folder, "solve", task)
            self.assertEqual((out["answer"]["source"], out["reused_event"]), ("record", 3))
            self.assertTrue(arena.check(task, out["answer"]["packing"]))
            self.assertFalse(out["after"]["ok"])
            explicit = command(folder, "research", task, "--budget", "2")
            self.assertTrue(arena.check(task, explicit["answer"]["packing"]))
            self.assertEqual(explicit["generation"], 0)
            self.assertEqual((explicit["answer"]["source"], explicit["reused_event"]), ("record", 3))

    def test_practice_success_is_delivered_even_when_nomination_fails(self):
        task = json.loads((ROOT / "examples/first.json").read_text())
        with TemporaryDirectory() as folder:
            event = command(folder, "research", task)
            self.assertEqual(event["decision"], "inconclusive")
            self.assertFalse(event["after"]["ok"])
            self.assertTrue(arena.check(task, event["answer"]["packing"]))
            successes = [r for r in event["search"]["trials"] if r["task_index"] == 0 and r["raw_ok"]]
            delivery = event["practice_delivery"]
            self.assertEqual((delivery["cost"], delivery["candidate"]), min((r["spent"], r["candidate"]) for r in successes))
            self.assertIn(delivery["calls"], (0, 1))
            self.assertEqual(delivery["executed_cost"], delivery["calls"] * delivery["cost"])
            self.assertEqual(event["search"]["calls"], 128)
            replay = command(folder, "solve", task)
            self.assertEqual((replay["reused_event"], replay["answer"]["source"]), (1, "record"))

    def test_old_input_and_its_scaled_view_do_not_enter_holdout(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            control = memory.investigate(root / "control")
            target = next(row["task"] for row in control["holdout"]["rows"]
                          if row["old_ok"] and row["task"]["bins"] >= 2 and len(row["task"]["items"]) >= 2)
            folder = root / "historical"
            scaled = dict(target, capacity=3 * target["capacity"], items=[3 * x for x in target["items"]])
            self.assertEqual(memory.solve(folder, scaled)["decision"], "unchanged")
            for i in range(36):
                memory.solve(folder, dict(bins=2, capacity=100, items=[i + 1, 40]))
            # Shape of the partial index in a pre-upgrade database, with no coverage marker.
            with sqlite3.connect(folder / "agentic.sqlite3") as db:
                db.execute("CREATE TABLE research_inputs (family TEXT PRIMARY KEY)")
            row = memory.investigate(folder)
            self.assertIn("holdout", row)
            tested = {research.family_id(r["task"]) for group in ("holdout", "regression") for r in row[group]["rows"]}
            self.assertNotIn(research.family_id(target), tested)
            with sqlite3.connect(folder / "agentic.sqlite3") as db:
                self.assertEqual(db.execute("SELECT event_id FROM research_input_cursor").fetchone(), (37,))
            memory.investigate(folder, budget=2)
            with sqlite3.connect(folder / "agentic.sqlite3") as db:
                self.assertEqual(db.execute("SELECT event_id FROM research_input_cursor").fetchone(), (38,))

    def test_published_memory_migrates_without_installing_a_new_policy(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/restore_demo_states.py"),
                            "--state-dir", str(root)], check=True, stdout=subprocess.DEVNULL)
            mixed = root / "mixed/20261118-201146-mt88urf4"
            older_answer = memory.event(mixed, 3)
            recovered = memory.solve(mixed, older_answer["task"])
            self.assertNotEqual(recovered["generation_before"], older_answer["generation_before"])
            self.assertEqual((recovered["answer"]["source"], recovered["reused_event"]), ("record", 3))
            folder = root / "pilot/round-01/stream"
            prior = memory.event(folder, 9)
            current = memory.investigate(folder, budget=2)
            self.assertEqual(current["memory_source_event"], 9)
            self.assertEqual(current["generation"], prior["generation"])
            self.assertTrue(current["search"]["memory"]["archive"])
            followup = memory.investigate(folder, budget=2)
            self.assertEqual(followup["memory_source_event"], current["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
