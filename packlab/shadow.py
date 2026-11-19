"""Prospective comparison of a frozen research nominee on first-seen arrivals."""
import hashlib
import json

from . import memory, recipes, research
from .experiment import Bench, RULES, ident, proven_impossible, validate_task


def _exists(db, table):
    return db is not None and db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def contains(db, key):
    """Read-only exclusion for any previously enrolled shadow family."""
    return _exists(db, "shadow_rows") and db.execute(
        "SELECT 1 FROM shadow_rows WHERE family=?", (key,)).fetchone() is not None


def _latest(db):
    if _exists(db, "shadow_cohorts"):
        return db.execute("SELECT * FROM shadow_cohorts ORDER BY id DESC LIMIT 1").fetchone()


def _state(db, cohort, frozen):
    if cohort["state"] != "active":
        return cohort["state"]
    head = memory.head(db)
    if (frozen["build"] != memory.build_id() or frozen["revision"] != head["id"]
            or frozen["old_chain"] != json.loads(head["policies"])):
        return "drift"
    # Also detect a move away and back made by a client without the invalidation hook.
    moved = db.execute("""SELECT 1 FROM events WHERE id>? AND decision IN ('kept','rollback') LIMIT 1""",
                       (frozen["after_event"],)).fetchone()
    return "drift" if moved else "active"


def invalidate(db):
    """Called in the transaction that changes the serving revision."""
    cohort = _latest(db)
    if cohort and cohort["state"] == "active" and _state(db, cohort, json.loads(cohort["frozen"])) == "drift":
        db.execute("UPDATE shadow_cohorts SET state='drift' WHERE id=?", (cohort["id"],))


def start(db, event_id, limit=32, source="solve"):
    """Freeze the event, chain, source and sampling limit in the caller's write transaction."""
    if type(limit) is not int or not 1 <= limit <= 128 or source not in ("solve", "observe"):
        raise ValueError("影子窗口需要 1 到 128 个家族，来源须为 solve 或 observe")
    if type(event_id) is not int or not 1 <= event_id <= 2**63 - 1:
        raise ValueError("需要有效的研究记录编号")
    entry = db.execute("SELECT data,start_revision FROM events WHERE id=? AND kind='research'", (event_id,)).fetchone()
    if not entry:
        raise ValueError("没有这份研究记录")
    event = json.loads(entry["data"])
    nominee = event.get("nominee")
    if not nominee or not all(k in event for k in ("holdout", "regression", "nominee_sha256")):
        raise ValueError("这份记录缺少经过正式研究的冻结提名")
    recipes.validate(nominee)
    digest = hashlib.sha256(ident(nominee).encode()).hexdigest()
    if digest != event["nominee_sha256"]:
        raise ValueError("研究记录的提名摘要不一致")
    head = memory.head(db)
    old = json.loads(head["policies"])
    if entry["start_revision"] != head["id"] or old != event.get("policies_before"):
        raise ValueError("研究记录的原版本或策略链与现役不符")
    new = research.dispatch(nominee, old)
    if new == old or len(new) + sum("fallback" in p for p in new) > RULES["max_policies"]:
        raise ValueError("冻结提名缺少可比较的有效策略链")

    db.execute("""CREATE TABLE IF NOT EXISTS shadow_cohorts (
        id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL UNIQUE, frozen TEXT NOT NULL,
        state TEXT NOT NULL, arrivals INTEGER NOT NULL DEFAULT 0,
        prior INTEGER NOT NULL DEFAULT 0, duplicates INTEGER NOT NULL DEFAULT 0)""")
    db.execute("""CREATE TABLE IF NOT EXISTS shadow_rows (
        family TEXT PRIMARY KEY, cohort_id INTEGER NOT NULL, arrival INTEGER NOT NULL,
        data TEXT NOT NULL, UNIQUE(cohort_id,arrival))""")
    if db.execute("SELECT 1 FROM shadow_cohorts WHERE event_id=?", (event_id,)).fetchone():
        raise ValueError("同一份研究记录只能开启一次影子窗口")
    invalidate(db)
    current = _latest(db)
    if current and current["state"] == "active":
        raise ValueError("当前影子窗口尚未结束")
    frozen = dict(event_id=event_id, revision=head["id"], build=memory.build_id(),
                  research_build=event.get("research_build", event.get("build")), nominee_sha256=digest,
                  old_chain=old, new_chain=new, source=source, limit=limit, started=memory.now(),
                  after_event=db.execute("SELECT MAX(id) FROM events").fetchone()[0])
    db.execute("INSERT INTO shadow_cohorts(event_id,frozen,state) VALUES (?,?,'active')", (event_id, ident(frozen)))
    return status(db)


def _endpoint(bench, task, chain):
    result = bench.apply(task, chain)
    attempts, path = [], []
    for attempt in result["attempts"]:
        policy = attempt["policy"]
        trial = bench.single(task, policy)
        trace = trial["trace"]
        route = trace.get("route")
        attempts.append(dict(attempt, guard=trace.get("guard"), route=route,
                             solver_ran=trace.get("solver_ran", True), search_status=trace["search_status"]))
        if route:
            leaves = [("primary", policy, route["primary"])]
            if route["fallback"] is not None:
                leaves.append(("fallback", policy["fallback"], route["fallback"]))
        else:
            leaves = [("primary" if policy.get("kind") == "recipe" else "ordinary", policy, trace)]
        for slot, leaf, executed in leaves:
            guard = executed.get("guard")
            hit = guard is None or guard["matched"]
            # Changing a threshold without changing which arm executes is not a switch.
            path.append(dict(slot=slot, matched=hit, steps=leaf.get("steps", []) if hit else [],
                             solver=leaf.get("solver", leaf) if hit else None))
    outcome = "solved" if result["ok"] else "infeasible" if proven_impossible(result) else "unresolved"
    return dict(ok=result["ok"], outcome=outcome, cost=result["cost"], packing=result["packing"],
                attempts=attempts, path=path)


def observe(db, task, source="observe", bench=None):
    """Enroll one arrival before solve records its task. All writes use the caller's transaction."""
    if source not in ("solve", "observe"):
        raise ValueError("影子输入来源须为 solve 或 observe")
    cohort = _latest(db)
    if cohort is None:
        return dict(outcome="inactive")
    frozen = json.loads(cohort["frozen"])
    state = _state(db, cohort, frozen)
    if state != "active":
        if state != cohort["state"]:
            db.execute("UPDATE shadow_cohorts SET state=? WHERE id=?", (state, cohort["id"]))
        return dict(outcome=state, event_id=cohort["event_id"])
    if source != frozen["source"]:
        if source == "observe":
            raise ValueError("这个窗口只观察正常 solve 到达的清单")
        return dict(outcome="skipped", event_id=cohort["event_id"])
    task = validate_task(task)
    key = research.family_id(task)
    if contains(db, key):
        reason = "duplicate"
    elif memory._exclusions(db)(key, "holdout"):
        reason = "prior"
    else:
        reason = "eligible"
    db.execute("""UPDATE shadow_cohorts SET arrivals=arrivals+1,
                  duplicates=duplicates+?,prior=prior+? WHERE id=?""",
               (reason == "duplicate", reason == "prior", cohort["id"]))
    if reason != "eligible":
        return dict(outcome=reason, family=key, event_id=cohort["event_id"])

    private = Bench()
    if bench is not None:
        private.cache.update(bench.cache)
    before = len(private.cache)
    row = dict(family=key, arrival=cohort["arrivals"] + 1, task=task, observed=memory.now())
    try:
        row["old"] = _endpoint(private, task, frozen["old_chain"])
        row["new"] = _endpoint(private, task, frozen["new_chain"])
        row["branch_changed"] = row["old"]["path"] != row["new"]["path"]
    except Exception as exc:
        # Preserve this arrival and close the cohort. The serving caller can still answer.
        row["error"] = f"{type(exc).__name__}: {exc}"
    fresh = list(private.cache.values())[before:]
    row.update(shadow_calls=len(fresh), shadow_spent=sum(r["cost"] for r in fresh))
    db.execute("INSERT INTO shadow_rows VALUES (?,?,?,?)", (key, cohort["id"], row["arrival"], ident(row)))
    eligible = row["arrival"] - cohort["duplicates"] - cohort["prior"]
    if "error" in row or eligible == frozen["limit"]:
        state = "error" if "error" in row else "closed"
        db.execute("UPDATE shadow_cohorts SET state=? WHERE id=?", (state, cohort["id"]))
    return dict(outcome="error" if "error" in row else "eligible", event_id=cohort["event_id"], row=row)


def stop(db):
    """End the current sample without discarding arrivals or permitting a restart."""
    invalidate(db)
    cohort = _latest(db)
    if cohort and cohort["state"] == "active":
        db.execute("UPDATE shadow_cohorts SET state='stopped' WHERE id=?", (cohort["id"],))
    return status(db)


def status(db, event_id=None):
    """Read a specific frozen event or the most recent cohort in this snapshot."""
    report = dict(state="inactive", event_id=None, limit=0, source=None, arrivals=0, eligible=0,
                  prior=0, duplicates=0, compared=0, errors=0, branch_changes=0, old_ok=0, new_ok=0,
                  gains=0, losses=0, old_cost=0, new_cost=0, old_success_cost=0, new_on_old_success_cost=0,
                  shadow_calls=0, shadow_spent=0, rows=[])
    if event_id is None:
        cohort = _latest(db)
    else:
        if type(event_id) is not int or not 1 <= event_id <= 2**63 - 1:
            raise ValueError("需要有效的研究记录编号")
        cohort = None
        if _exists(db, "shadow_cohorts"):
            cohort = db.execute("SELECT * FROM shadow_cohorts WHERE event_id=?", (event_id,)).fetchone()
        if cohort is None:
            raise ValueError("这份研究记录没有影子窗口")
    if cohort is None:
        return report
    frozen = json.loads(cohort["frozen"])
    report.update(frozen, state=_state(db, cohort, frozen), arrivals=cohort["arrivals"],
                  prior=cohort["prior"], duplicates=cohort["duplicates"])
    for entry in db.execute("SELECT data FROM shadow_rows WHERE cohort_id=? ORDER BY arrival", (cohort["id"],)):
        row = json.loads(entry[0])
        report["rows"].append(row)
        report["eligible"] += 1
        report["shadow_calls"] += row["shadow_calls"]
        report["shadow_spent"] += row["shadow_spent"]
        if "error" in row:
            report["errors"] += 1
            continue
        a, b = row["old"], row["new"]
        report["compared"] += 1
        report["branch_changes"] += row["branch_changed"]
        report["old_ok"] += a["ok"]
        report["new_ok"] += b["ok"]
        report["gains"] += b["ok"] and not a["ok"]
        report["losses"] += a["ok"] and not b["ok"]
        report["old_cost"] += a["cost"]
        report["new_cost"] += b["cost"]
        if a["ok"]:
            report["old_success_cost"] += a["cost"]
            report["new_on_old_success_cost"] += b["cost"]
    return report
