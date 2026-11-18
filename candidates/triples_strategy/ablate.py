"""Read only the historical #3 diagnostic cases; dump the precommitted 2x2."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import engine as candidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from candidates.strategy import engine as v1

DB = ROOT / "state/demos/20261118-183901-uhbq90ps/agentic.sqlite3"
V1_SHA = "7e1acb65e9e1b08a14db42c3b17c855d784031d4b6f125633f353dd969ad0710"


def read_cases():
    with sqlite3.connect(DB.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        data = json.loads(db.execute("SELECT data FROM events WHERE id=3").fetchone()[0])
    cases = [(i, row["task"]) for i, row in enumerate(data["holdout"]["rows"])
             if row["profile"] == "triples" and not row["old_ok"] and not row["new_ok"]]
    assert data["seed"] == 177730 and len(cases) == 12
    return cases


def validate(task, packing):
    assert len(packing) == task["bins"]
    assert sorted(i for group in packing for i in group) == list(range(len(task["items"])))
    assert all(sum(task["items"][i] for i in group) == task["capacity"] for group in packing)


def experiment():
    assert hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest() == V1_SHA
    cases = read_cases()
    methods = ((v1, "search"), (v1, "deep"), (candidate, "triples"), (candidate, "triples_deep"))
    rows = []
    for index, task in cases:
        greedy = v1.run(task, dict(v1.BASE, order="descending"))
        assert not greedy["ok"]
        for module, repair in methods:
            trial = module.run(task, dict(v1.BASE, order="descending", repair=repair))
            if trial["ok"]:
                validate(task, trial["packing"])
            ops = trial["trace"]["ops"]
            extra = {k: v - greedy["trace"]["ops"].get(k, 0) for k, v in ops.items()}
            assert min(extra.values()) >= 0 and greedy["cost"] + sum(extra.values()) == trial["cost"]
            rows.append(dict(row=index, task=task, repair=repair, ok=trial["ok"],
                             status=trial["trace"]["search_status"], packing=trial["packing"],
                             nodes=ops.get("group_nodes", ops["search_nodes"]),
                             greedy_cost=greedy["cost"], extra_ops=extra, cost=trial["cost"]))
    summary = {}
    for _, repair in methods:
        part = [r for r in rows if r["repair"] == repair]
        summary[repair] = {k: sum(r[k] for r in part) for k in ("ok", "nodes", "cost", "greedy_cost")}
        summary[repair]["limits"] = sum(r["status"] == "limit" for r in part)
    return dict(source=str(DB.relative_to(ROOT)), event=3, seed=177730,
                v1_sha256=V1_SHA, summary=summary, rows=rows)


if __name__ == "__main__":
    print(json.dumps(experiment(), ensure_ascii=False, indent=2))
