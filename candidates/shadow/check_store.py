"""Check prospective enrollment, paired route costs and SQLite transaction boundaries."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packlab import arena, engine, memory, research, shadow
from packlab.experiment import Bench, ident


def task(bins, capacity, items):
    return dict(bins=bins, capacity=capacity, items=items)


def recipe(solver, feature, op, value):
    return dict(kind="recipe", solver=solver, steps=[], guard=dict(feature=feature, op=op, value=value))


BASE = engine.BASE
OLD = dict(recipe(dict(BASE, order="descending"), "bins", "le", 2),
           fallback=recipe(BASE, "capacity", "ge", 27))
NEW = dict(OLD, guard=dict(OLD["guard"], value=4))
GAIN = task(3, 10, [4, 4, 4, 6, 6, 6])
MIX = task(3, 38, [12, 18, 8, 7, 8, 23, 9, 17, 12])
STEADY = task(2, 10, [4, 4, 6, 6])
HISTORY = [task(4, 103, [i, 20, 39, 50]) for i in range(1, 8)]


def fixture(db, nominee=NEW, **changes):
    rev = memory.head(db)
    evidence = dict(generation_before=rev["id"], generation=rev["id"], decision="inconclusive",
                    reason="synthetic fixture", research_build="older research build",
                    policies_before=json.loads(rev["policies"]), task=HISTORY[0], batch_tasks=[HISTORY[1]],
                    practice_tasks=[HISTORY[2]], holdout=dict(rows=[dict(task=HISTORY[3])]),
                    regression=dict(rows=[dict(task=HISTORY[4])]), replay=dict(rows=[dict(task=HISTORY[5])]),
                    nominee=nominee, nominee_sha256=hashlib.sha256(ident(nominee).encode()).hexdigest())
    evidence.update(changes)
    return memory.add_event(db, "research", evidence)["id"]


def rejected(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected rejection")


def scale(t):
    # Unit change, reversal, and an independently forced full disk.
    return task(t["bins"] + 1, 2*t["capacity"], [2*t["capacity"], *(2*i for i in reversed(t["items"]))])


def install(db):
    rev = db.execute("INSERT INTO revisions(parent,policies) VALUES (0,?)", (ident([OLD, BASE]),)).lastrowid
    db.execute("UPDATE head SET revision=? WHERE slot=1", (rev,))
    memory.add_event(db, "solve", dict(generation_before=rev, decision="unchanged", reason="history fixture",
                                      task=HISTORY[6], answer=dict(packing=[[0],[1],[2],[3]])))
    return fixture(db)


def run_rows(folder):
    assert shadow.stop(None) == shadow.status(None)
    assert shadow.status(None)["state"] == "inactive" and not shadow.contains(None, "missing")
    rejected(shadow.status, None, 1)
    with memory.transaction(folder) as db:
        assert shadow.status(db)["state"] == "inactive" and not shadow.contains(db, "x")
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name LIKE 'shadow_%'").fetchone()
        event_id = install(db)
        bad = fixture(db, nominee_sha256="altered")
        rejected(shadow.start, db, bad)
        rejected(shadow.start, db, 1)  # ordinary solve
        for invalid in (0, 129, True):
            rejected(shadow.start, db, event_id, invalid)
        rejected(shadow.start, db, event_id, source="other")
        state = shadow.start(db, event_id, 3, "observe")
        assert state["old_chain"] == [OLD, BASE] and state["new_chain"] == [NEW, BASE]
        assert state["research_build"] != state["build"] and state["build"] == memory.build_id()
        assert state["limit"] == 3 and state["state"] == "active"
        assert shadow.observe(db, GAIN, source="solve")["outcome"] == "skipped"
        assert shadow.status(db)["arrivals"] == 0
        second = fixture(db, nominee=dict(OLD, fallback=recipe(BASE, "capacity", "ge", 9)))
        rejected(shadow.start, db, second)
        for t in HISTORY:
            assert shadow.observe(db, scale(t))["outcome"] == "prior"
        for i, t in enumerate((GAIN, MIX, STEADY)):
            serving = Bench()
            serving.apply(t, [OLD, BASE])
            saved = dict(serving.cache)
            row = shadow.observe(db, t, bench=serving)["row"]
            assert serving.cache == saved
            assert row["task"] == t and row["family"] == research.family_id(t)
            assert row["branch_changed"] == (i < 2)
            ref = Bench()
            planned_cost = 0
            for arm, policy in (("old", OLD), ("new", NEW)):
                expected = ref.apply(t, [policy, BASE])
                a = row[arm]
                assert a["cost"] == expected["cost"] == sum(k["cost"] for k in a["attempts"])
                assert a["ok"] == expected["ok"] and a["packing"] == expected["packing"]
                if a["ok"]:
                    assert arena.check(t, a["packing"])
                planned_cost += a["cost"]
            fresh = [r for k, r in ref.cache.items() if k not in saved]
            assert row["shadow_spent"] == sum(r["cost"] for r in fresh) < planned_cost
            assert row["shadow_calls"] == len(fresh)
            if t == GAIN:
                assert shadow.observe(db, scale(t))["outcome"] == "duplicate"
            if t == MIX:
                a, b = (row[k]["attempts"][0]["route"] for k in ("old","new"))
                assert a["selected"] == b["selected"] == "fallback"
                assert not a["primary"]["guard"]["matched"] and b["primary"]["guard"]["matched"]
        s = shadow.status(db)
        assert (s["state"], s["arrivals"], s["eligible"], s["prior"], s["duplicates"]) == ("closed",11,3,7,1)
        assert (s["compared"],s["branch_changes"],s["gains"],s["losses"],s["old_ok"],s["new_ok"]) == (3,2,1,0,2,3)
        assert [r["arrival"] for r in s["rows"]] == [8,10,11]
        assert s["old_success_cost"] == sum(r["old"]["cost"] for r in s["rows"] if r["old"]["ok"])
        assert s["new_on_old_success_cost"] > s["old_success_cost"]
        assert shadow.observe(db, HISTORY[2])["outcome"] == "closed" and shadow.status(db) == s
        rejected(shadow.start, db, event_id)
    with memory._read_state(folder) as db:
        assert shadow.status(db) == s and shadow.contains(db, research.family_id(scale(MIX)))
    with memory.transaction(folder) as db:
        assert shadow.start(db, second)["limit"] == 32
        rejected(shadow.observe, db, GAIN)
        assert shadow.observe(db, scale(GAIN), "solve")["outcome"] == "duplicate"
        assert shadow.status(db)["rows"] == []
        # Even a return to the same head cannot reopen a window once invalidated.
        db.execute("UPDATE head SET revision=0")
        assert shadow.status(db)["state"] == "drift"
        shadow.invalidate(db)
        db.execute("UPDATE head SET revision=1")
        assert shadow.observe(db, GAIN, "solve")["outcome"] == "drift"
        third = fixture(db)
        shadow.start(db, third, 32, "observe")
        with patch.object(memory, "BUILD", "another-runtime"):
            assert shadow.observe(db, STEADY)["outcome"] == "drift"
        assert shadow.status(db)["state"] == "drift"
        rejected(shadow.start, db, third)
        assert shadow.stop(db)["state"] == "drift" and shadow.status(db, event_id) == s
        fourth = fixture(db)
        shadow.start(db, fourth, 3, "observe")
        extra = task(3, 43, [5, 7, 18, 29])
        assert shadow.observe(db, extra)["outcome"] == "eligible"
        stopped = shadow.stop(db)
        assert stopped["state"] == "stopped" and stopped["eligible"] == 1
        assert shadow.stop(db) == stopped and shadow.observe(db, GAIN)["outcome"] == "stopped"
        rejected(shadow.start, db, fourth)
    with memory._read_state(folder) as db:
        assert shadow.status(db, fourth) == stopped and shadow.status(db, event_id) == s


def failure_and_races(folder):
    with memory.transaction(folder) as db:
        event = install(db)
        shadow.start(db, event, 2, "observe")
    try:
        with memory.transaction(folder) as db:
            assert shadow.observe(db, GAIN)["outcome"] == "eligible"
            raise RuntimeError("caller rolled back")
    except RuntimeError as exc:
        assert str(exc) == "caller rolled back"
    def submit(t):
        with memory.transaction(folder) as db:
            return shadow.observe(db, t)["outcome"]
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(submit, [GAIN, scale(GAIN)])) == ["duplicate","eligible"]
    with memory.transaction(folder) as db:
        count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        with patch.object(Bench, "apply", side_effect=ValueError("synthetic evaluation error")):
            row = shadow.observe(db, MIX)
        assert row["outcome"] == "error" and "synthetic" in row["row"]["error"]
        s = shadow.status(db)
        assert (s["state"],s["arrivals"],s["eligible"],s["compared"],s["errors"]) == ("error",3,2,1,1)
        assert shadow.contains(db, research.family_id(MIX)) and shadow.observe(db, STEADY)["outcome"] == "error"
        assert memory.head(db)["id"] == 1 and db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == count


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        run_rows(folder)
    with TemporaryDirectory() as folder:
        failure_and_races(folder)
    print("Shadow enrollment, route evidence, cache accounting, drift, failure and concurrent transactions verified.")
