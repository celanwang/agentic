"""Check a pre-upgrade index, fixed regression controls, and bounded user-batch replay."""
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packlab import arena, experiment, memory, research


def verify(folder):
    first = memory.investigate(folder)
    published = first["holdout"]["rows"][0]["task"]
    prior = dict(published, capacity=2 * published["capacity"], items=[2 * x for x in reversed(published["items"])])
    assert prior != published and research.family_id(prior) == research.family_id(published)
    with sqlite3.connect(folder / "agentic.sqlite3") as db:
        # An upgraded database may already have consumed this event with the old cursor.
        db.execute("DROP TABLE research_assessments")
        db.execute("DELETE FROM research_input_cursor WHERE slot=2")
        db.execute("UPDATE research_input_cursor SET event_id=1 WHERE slot=1")
    with memory.transaction(folder) as db:
        exclude = memory._exclusions(db)
        assert exclude(research.family_id(prior), "holdout") and not exclude(research.family_id(prior), "regression")
        assert exclude(research.family_id(first["practice_tasks"][0]), "regression")
        with patch.object(arena, "cases", return_value=[prior]), patch.object(
                arena, "case_expectations", return_value=[("fixture", True)]):
            hold, reg = research.test_sets(1, [], lambda key: False, exclude)
            assert not hold and reg == [(prior, "fixture", True)]
            assert not experiment.sample(1, "holdout", set(), exclude=exclude)
            assert experiment.sample(1, "regression", set(), exclude=exclude) == [prior]
        count, = db.execute("SELECT count(*) FROM research_assessments").fetchone()
        memory._exclusions(db)
        assert db.execute("SELECT count(*) FROM research_assessments").fetchone()[0] == count

    public = json.loads((ROOT / "candidates/pair_eval/probe_result.json").read_text())
    batch = [r["task"] for r in public["reveal"]["cases"]]
    records = [dict(task=prior, answer=True), dict(task=published, answer=True, batch_tasks=batch * 3)]
    cases = research.historical_cases(records, 10)
    assert len(cases) == 10 and cases[0][:2] == (prior, "历史作答") and cases[1][:2] == (published, "历史作答")
    assert all(label == "历史批次" and possible is None for t, label, possible in cases[2:])
    assert {t["bins"] for t, _, _ in cases[2:]} == {5, 6, 7, 8}
    assert len({experiment.ident(t) for t, _, _ in cases}) == 10


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        verify(Path(folder))
    print("Upgraded historical index, both gates, fixed controls and batch replay verified.")
