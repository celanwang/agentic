"""One transaction owns a decision, its evidence, and the active policy pointer."""
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import arena, engine, planner, research, shadow
from .experiment import Bench, ident, proven_impossible, run_round, runtime_id, validate_task


ROOT = Path(__file__).resolve().parent
BUILD = None
# The original release stored only its full research build. Its raw solver results
# and rewards are compatible with v1; sorted JSON no longer contains its original recency order.
LEGACY_MEMORY_FORMATS = {
    "7b5066f82fc768c31657767eeb18a9be68d7dea811e948fe150bade791048cd4":
        "1:10d85ca8dc0c58b8803727bd89558c9231f097a0c00a87d80371b2eb4b271b09"
}
# The route executor preserves the values, costs and traces of four-field recipes.
# This one-way migration applies only to the verified pair of solver builds.
COMPATIBLE_PREDECESSORS = {
    "1:4edca22ea390932922c4f812d3a251d0dfb8feabf73d752cf687021760e8b278":
        "1:10d85ca8dc0c58b8803727bd89558c9231f097a0c00a87d80371b2eb4b271b09",
    # The pre-route release could only run four-field recipes, whose costs still agree.
    "1:b453159202ae53df02f754ff626f1925626d200673749821867c0ea4e29ee914":
        "1:10d85ca8dc0c58b8803727bd89558c9231f097a0c00a87d80371b2eb4b271b09"
}
ROUTE_COST_MIGRATION = ("1:4edca22ea390932922c4f812d3a251d0dfb8feabf73d752cf687021760e8b278",
                        "1:b453159202ae53df02f754ff626f1925626d200673749821867c0ea4e29ee914")


def build_id():
    global BUILD
    if BUILD is None:
        BUILD = runtime_id(sorted(ROOT.glob("*.py")))
    return BUILD


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def transaction(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(folder / "agentic.sqlite3"), timeout=30)) as db:
        db.row_factory = sqlite3.Row
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS revisions (id INTEGER PRIMARY KEY, parent INTEGER, policies TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS head (slot INTEGER PRIMARY KEY, revision INTEGER NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS events
                (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, start_revision INTEGER NOT NULL,
                 task TEXT, decision TEXT NOT NULL, reason TEXT NOT NULL, created TEXT NOT NULL, data TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS events_task_id ON events(task, id)")
            db.execute("INSERT OR IGNORE INTO revisions VALUES (0, NULL, ?)", (ident([engine.BASE]),))
            db.execute("INSERT OR IGNORE INTO head VALUES (1, 0)")
            yield db


@contextmanager
def _read_state(folder):
    path = Path(folder).resolve() / "agentic.sqlite3"
    try:
        path.stat()
    except FileNotFoundError:
        yield None
        return
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
        db.row_factory = sqlite3.Row
        # A deferred transaction keeps the revision and event counts in one snapshot.
        db.execute("BEGIN")
        yield db


def head(db):
    return db.execute("SELECT r.* FROM revisions r JOIN head h ON r.id=h.revision WHERE h.slot=1").fetchone()


def add_event(db, kind, data):
    data.update(kind=kind, created=now(), build=build_id())
    event_id = db.execute(
        "INSERT INTO events (kind, start_revision, task, decision, reason, created, data) VALUES (?,?,?,?,?,?,?)",
        (kind, data["generation_before"], ident(data.get("task")), data["decision"], data["reason"], data["created"], ident(data))
    ).lastrowid
    data["id"] = event_id
    return data


def _index_prior_inputs(db):
    """Track every observed task family, including events from before research was enabled."""
    db.execute("CREATE TABLE IF NOT EXISTS research_inputs (family TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE IF NOT EXISTS research_input_cursor (slot INTEGER PRIMARY KEY, event_id INTEGER NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS research_assessments (family TEXT PRIMARY KEY)")
    # Slot 2 signals that formal rows were indexed as well. Old databases have only slot 1.
    marker = db.execute("SELECT event_id FROM research_input_cursor WHERE slot=2").fetchone()
    last_id = marker[0] if marker else 0
    for row in db.execute("SELECT id, data FROM events WHERE id>? AND kind IN ('solve','research') ORDER BY id",
                          (last_id,)):
        event = json.loads(row["data"])
        task = event.get("task")
        tasks = event.get("practice_tasks", []) + event.get("batch_tasks", [])
        if task is not None:
            tasks.append(task)
        keys = {research.family_id(t) for t in tasks}
        db.executemany("INSERT OR IGNORE INTO research_inputs VALUES (?)", ((k,) for k in keys))
        assessed = {research.family_id(r["task"]) for phase in ("holdout", "regression", "replay")
                    for r in event.get(phase, {}).get("rows", [])}
        db.executemany("INSERT OR IGNORE INTO research_assessments VALUES (?)", ((k,) for k in assessed))
        last_id = row["id"]
    db.execute("INSERT OR REPLACE INTO research_input_cursor VALUES (1,?)", (last_id,))
    db.execute("INSERT OR REPLACE INTO research_input_cursor VALUES (2,?)", (last_id,))


def _exclusions(db):
    _index_prior_inputs(db)
    def exclude(key, phase):
        if db.execute("SELECT 1 FROM research_inputs WHERE family=?", (key,)).fetchone():
            return True
        return phase == "holdout" and (bool(db.execute(
            "SELECT 1 FROM research_assessments WHERE family=?", (key,)).fetchone()) or shadow.contains(db, key))
    return exclude


def _previous_attempt(db, task, revision=None, kind=None):
    """A verified packing survives policy changes. An unsuccessful trial is specific to its context."""
    previous = None
    for row in db.execute("SELECT id, data FROM events WHERE task=? ORDER BY id", (ident(task),)):
        event = json.loads(row["data"])
        answer = event.get("answer")
        if answer and arena.check(task, answer["packing"]):
            return {"id": row["id"], "answer": {"source": "record", "packing": answer["packing"]}, "outcome": "packed"}
        if (event["kind"] == kind and event["generation_before"] == revision and event["build"] == build_id()
                and (previous is None or event.get("outcome") == "infeasible")):
            previous = {"id": row["id"], "answer": None, "outcome": event["outcome"]}
    return previous


def _research(db, revision, policies, focus, budget, bench=None, started=None, batch=None):
    started = time.perf_counter() if started is None else started
    # Search only receives past tasks, observed production outcomes and trial actions.
    # Never pass formal holdout rows or blind answers to the planner.
    exclude = _exclusions(db)
    history = []
    for row in db.execute("SELECT id, data FROM events WHERE task <> 'null' ORDER BY id DESC LIMIT 32"):
        record = json.loads(row["data"])
        history.append(dict(id=row["id"], **{k: record[k] for k in
                            ("kind", "task", "before", "answer", "trials", "batch_tasks") if k in record}))
    row = db.execute("SELECT id, data FROM events WHERE kind='research' ORDER BY id DESC LIMIT 1").fetchone()
    last = json.loads(row["data"]) if row else {}
    signature = build_id()
    solver_build = runtime_id([ROOT / name for name in ("engine.py", "recipes.py", "experiment.py")])
    memory_format = f"{planner.MEMORY_VERSION}:{solver_build}"
    prior_format = last.get("research_memory_format", LEGACY_MEMORY_FORMATS.get(last.get("research_build")))
    compatible = prior_format is not None and prior_format in (memory_format, COMPATIBLE_PREDECESSORS.get(memory_format))
    memory = last["search"]["memory"] if compatible else {}
    archive_only = (prior_format, memory_format) == ROUTE_COST_MIGRATION
    if archive_only:
        prior = last["search"]["memory"]
        memory = dict(version=planner.MEMORY_VERSION, archive=prior.get("archive", []))
    db.execute("""CREATE TABLE IF NOT EXISTS research_reviews
                  (revision INTEGER, build TEXT, digest TEXT, event_id INTEGER NOT NULL,
                   PRIMARY KEY (revision, build, digest))""")
    reviewed = dict(db.execute("SELECT digest,event_id FROM research_reviews WHERE revision=? AND build=?",
                               (revision["id"], signature)))
    report = research.study(policies, revision["id"], history, memory, last.get("episode", 0) + 1,
                            budget, focus, reviewed, bench=bench, batch=batch, exclude=exclude)
    if focus is not None and not report["answer"]:
        saved = _previous_attempt(db, focus)
        if saved and saved["answer"]:
            report.update(answer=saved["answer"], outcome="packed", reused_event=saved["id"])
    report["research_build"] = signature
    report["research_memory_format"] = memory_format
    report["memory_source_event"] = row["id"] if memory else None
    if archive_only:
        report["memory_migration"] = "archive_only"
    if report["decision"] == "kept":
        updated = research.dispatch(report["nominee"], policies)
        version = db.execute("INSERT INTO revisions(parent, policies) VALUES (?, ?)",
                             (revision["id"], ident(updated))).lastrowid
        db.execute("UPDATE head SET revision=? WHERE slot=1", (version,))
        shadow.invalidate(db)
        report["generation"] = version
    report["experiment_ms"] = round((time.perf_counter() - started) * 1000, 3)
    event = add_event(db, "research", report)
    if "holdout" in report:
        db.execute("INSERT INTO research_reviews VALUES (?,?,?,?)",
                   (revision["id"], signature, report["nominee_sha256"], event["id"]))
    return event


def investigate(folder, focus=None, budget=research.DEFAULT_BUDGET, batch=None):
    research.validate_budget(budget)
    if batch is not None:
        if focus is not None or not isinstance(batch, list) or not 1 <= len(batch) <= 64:
            raise ValueError("批次需要 1 到 64 份清单，并以第一份作为触发输入")
        batch = [validate_task(task) for task in batch]
        focus = batch[0]
    with transaction(folder) as db:
        rev = head(db)
        return _research(db, rev, json.loads(rev["policies"]), focus, budget, batch=batch)


def solve(folder, task):
    with transaction(folder) as db:
        revision = head(db)
        policies = json.loads(revision["policies"])
        # New assessment episodes get a fresh split, independent of read-only visits.
        number = db.execute("SELECT count(*) FROM events WHERE kind='solve' AND decision<>'unchanged'").fetchone()[0]
        started = time.perf_counter()
        bench = Bench()
        current = bench.apply(task, policies)
        shadow.observe(db, task, source="solve", bench=bench)
        previous = None
        if not current["ok"] and not proven_impossible(current):
            research_enabled = db.execute("SELECT 1 FROM events WHERE kind='research' LIMIT 1").fetchone() is not None
            previous = _previous_attempt(db, task, revision["id"], "research" if research_enabled else "solve")
            if previous is None and research_enabled:
                return _research(db, revision, policies, task, research.DEFAULT_BUDGET, bench, started)
        exclude = None
        if (previous is None and not current["ok"] and not proven_impossible(current)
                and db.execute("SELECT 1 FROM events LIMIT 1").fetchone()):
            exclude = _exclusions(db)
        report = run_round(task, policies, revision["id"], 73001 + 104729 * number, previous, bench, exclude)
        if report.get("answer") and report["answer"]["source"] == "record":
            report["reason"] = f"已按本次清单重新核验记录 #{report['reused_event']} 的装法。"
        report["experiment_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if report["decision"] == "kept":
            updated = policies + [report["nominee"]]
            version = db.execute("INSERT INTO revisions(parent, policies) VALUES (?, ?)",
                                 (revision["id"], ident(updated))).lastrowid
            db.execute("UPDATE head SET revision=? WHERE slot=1", (version,))
            shadow.invalidate(db)
            report["generation"] = version
        return add_event(db, "solve", report)


def status(folder):
    path = str(Path(folder).resolve() / "agentic.sqlite3")
    with _read_state(folder) as db:
        if db is None:
            return dict(generation=0, parent=None, policies=[engine.BASE.copy()], db=path,
                        events=0, build=build_id(), research_rounds=0)
        rev = head(db)
        if rev is None:
            raise ValueError("状态库缺少有效的活动版本")
        return dict(generation=rev["id"], parent=rev["parent"], policies=json.loads(rev["policies"]),
                    db=path,
                    events=db.execute("SELECT count(*) FROM events").fetchone()[0], build=build_id(),
                    research_rounds=db.execute("SELECT count(*) FROM events WHERE kind='research'").fetchone()[0])


def history(folder, limit):
    with _read_state(folder) as db:
        if db is None:
            return []
        return [dict(row) for row in db.execute(
            "SELECT id, created, start_revision, decision, reason FROM events ORDER BY id DESC LIMIT ?", (limit,))]


def shadow_status(folder, event_id=None):
    with _read_state(folder) as db:
        return shadow.status(db, event_id)


def start_shadow(folder, event_id, limit=32, source="solve"):
    if not (Path(folder) / "agentic.sqlite3").is_file():
        raise ValueError("需要包含冻结研究提名的已有状态")
    with transaction(folder) as db:
        return shadow.start(db, event_id, limit=limit, source=source)


def observe_shadow(folder, task):
    validate_task(task)
    if not (Path(folder) / "agentic.sqlite3").is_file():
        return {"outcome": "inactive"}
    with transaction(folder) as db:
        return shadow.observe(db, task, source="observe")


def stop_shadow(folder):
    if not (Path(folder) / "agentic.sqlite3").is_file():
        return shadow.status(None)
    with transaction(folder) as db:
        return shadow.stop(db)


def event(folder, event_id):
    if 0 < event_id <= 2**63 - 1:
        with _read_state(folder) as db:
            if db is not None:
                row = db.execute("SELECT id, data FROM events WHERE id=?", (event_id,)).fetchone()
                if row is not None:
                    return dict(json.loads(row["data"]), id=row["id"])
    raise ValueError("没有编号为 %s 的记录" % event_id)


def rollback(folder):
    initial_state = "当前位于初始版本，没有更早的策略可恢复"
    try:
        (Path(folder) / "agentic.sqlite3").stat()
    except FileNotFoundError:
        raise ValueError(initial_state) from None
    with transaction(folder) as db:
        rev = head(db)
        parent = rev["parent"]
        if parent is None:
            raise ValueError(initial_state)
        db.execute("UPDATE head SET revision=? WHERE slot=1", (parent,))
        shadow.invalidate(db)
        return add_event(db, "rollback", dict(generation_before=rev["id"], generation=parent,
                         decision="rollback", reason="手动恢复到上一份已验证策略，实验记录仍可查阅。"))
