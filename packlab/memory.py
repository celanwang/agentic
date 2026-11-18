"""One transaction owns a decision, its evidence, and the active policy pointer."""
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import engine, research
from .experiment import Bench, ident, proven_impossible, run_round, runtime_id


ROOT = Path(__file__).resolve().parent
BUILD = None


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


def _research(db, revision, policies, focus, budget, bench=None, started=None):
    started = time.perf_counter() if started is None else started
    # Search only receives past tasks, observed production outcomes and trial actions.
    # Never pass formal holdout rows or blind answers to the planner.
    db.execute("CREATE TABLE IF NOT EXISTS research_inputs (family TEXT PRIMARY KEY)")
    history = []
    for row in db.execute("SELECT id, data FROM events WHERE task <> 'null' ORDER BY id DESC LIMIT 32"):
        record = json.loads(row["data"])
        inputs = [record["task"], *record.get("practice_tasks", [])]
        db.executemany("INSERT OR IGNORE INTO research_inputs VALUES (?)", ((research.family_id(t),) for t in inputs))
        history.append(dict(id=row["id"], **{k: record[k] for k in
                            ("kind", "task", "before", "answer", "trials") if k in record}))
    last = db.execute("SELECT data FROM events WHERE kind='research' ORDER BY id DESC LIMIT 1").fetchone()
    last = json.loads(last[0]) if last else {}
    signature = runtime_id([ROOT / name for name in ("engine.py", "recipes.py", "planner.py", "research.py", "experiment.py")])
    memory = last.get("search", {}).get("memory", {}) if last.get("research_build") == signature else {}
    db.execute("""CREATE TABLE IF NOT EXISTS research_reviews
                  (revision INTEGER, build TEXT, digest TEXT, event_id INTEGER NOT NULL,
                   PRIMARY KEY (revision, build, digest))""")
    reviewed = dict(db.execute("SELECT digest,event_id FROM research_reviews WHERE revision=? AND build=?",
                               (revision["id"], signature)))
    def prior_input(key):
        return db.execute("SELECT 1 FROM research_inputs WHERE family=?", (key,)).fetchone() is not None
    report = research.study(policies, revision["id"], history, memory, last.get("episode", 0) + 1,
                            budget, focus, reviewed, prior_input, bench)
    db.executemany("INSERT OR IGNORE INTO research_inputs VALUES (?)",
                   ((research.family_id(t),) for t in report["practice_tasks"]))
    report["research_build"] = signature
    if report["decision"] == "kept":
        updated = research.dispatch(report["nominee"], policies)
        version = db.execute("INSERT INTO revisions(parent, policies) VALUES (?, ?)",
                             (revision["id"], ident(updated))).lastrowid
        db.execute("UPDATE head SET revision=? WHERE slot=1", (version,))
        report["generation"] = version
    report["experiment_ms"] = round((time.perf_counter() - started) * 1000, 3)
    event = add_event(db, "research", report)
    if "holdout" in report:
        db.execute("INSERT INTO research_reviews VALUES (?,?,?,?)",
                   (revision["id"], signature, report["nominee_sha256"], event["id"]))
    return event


def investigate(folder, focus=None, budget=research.DEFAULT_BUDGET):
    research.validate_budget(budget)
    with transaction(folder) as db:
        rev = head(db)
        return _research(db, rev, json.loads(rev["policies"]), focus, budget)


def solve(folder, task):
    with transaction(folder) as db:
        revision = head(db)
        policies = json.loads(revision["policies"])
        repeats = db.execute(
            """SELECT id, data FROM events WHERE kind IN ('solve','research') AND start_revision=? AND task=?
               ORDER BY id ASC LIMIT 1""",
            (revision["id"], ident(task))).fetchone()
        # New assessment episodes get a fresh split, independent of read-only visits.
        number = db.execute("SELECT count(*) FROM events WHERE kind='solve' AND decision<>'unchanged'").fetchone()[0]
        started = time.perf_counter()
        bench = Bench()
        previous = dict(json.loads(repeats["data"]), id=repeats["id"]) if repeats else None
        if previous is None and db.execute("SELECT 1 FROM events WHERE kind='research' LIMIT 1").fetchone():
            current = bench.apply(task, policies)
            if not current["ok"] and not proven_impossible(current):
                return _research(db, revision, policies, task, research.DEFAULT_BUDGET, bench, started)
        report = run_round(task, policies, revision["id"], 73001 + 104729 * number, previous, bench)
        report["experiment_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if report["decision"] == "kept":
            updated = policies + [report["nominee"]]
            version = db.execute("INSERT INTO revisions(parent, policies) VALUES (?, ?)",
                                 (revision["id"], ident(updated))).lastrowid
            db.execute("UPDATE head SET revision=? WHERE slot=1", (version,))
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
        return add_event(db, "rollback", dict(generation_before=rev["id"], generation=parent,
                         decision="rollback", reason="手动恢复到上一份已验证策略，实验记录仍可查阅。"))
