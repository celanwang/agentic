"""Install only the runtime and examples, replay across processes and check recovery."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil
import sqlite3
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def run(root, state, *args, raw=None, check=True):
    result = subprocess.run([str(root / "agentic"), "--state", str(state), *args, "--json"], cwd=root.parent,
                            input=raw, capture_output=True, text=True, check=check)
    return json.loads(result.stdout) if check else result.returncode


def verify():
    with tempfile.TemporaryDirectory(prefix="agentic-standalone-") as temp:
        root = Path(temp) / "install"
        root.mkdir()
        for name in ("agentic", "demo-research"):
            shutil.copy2(ROOT / name, root / name)
        for name in ("examples", "packlab"):
            shutil.copytree(ROOT / name, root / name)
        subprocess.run([str(root / "demo-research")], cwd=temp, capture_output=True, text=True, check=True)
        state, = (root / "state" / "demos").iterdir()
        fixed = ROOT / "state" / "pilot" / "round-01" / "stream"
        for index in range(1, 10):
            trial = run(root, state, "show", str(index))
            released = run(root, fixed, "show", str(index))
            for ignored in ("experiment_ms", "created"):
                trial.pop(ignored)
                released.pop(ignored)
            assert trial == released
        status = run(root, state, "status")
        assert (status["events"], status["generation"]) == (9, 1)
        assert run(root, state, "research", "--budget", "1", check=False) == 2
        assert run(root, state, "status") == status
        example = str(root / "examples" / "pilot-independent.json")
        answer = run(root, state, "probe", example)["result"]
        assert answer["ok"] and not run(root, state, "probe", example, "--initial")["result"]["ok"]
        explanation = run(root, state, "explain", "9", "--task", example)
        assert explanation["runs_used"] <= explanation["budget"] and explanation["status"] == "witness"
        assert run(root, state, "status") == status
        run(root, state, "rollback")
        assert not run(root, state, "probe", example)["result"]["ok"]
        # Two independent writers encounter the same previously unseen failure.
        task = '{"bins":3,"capacity":503,"items":[133,140,159,197,166,135,197,211,171]}'
        with ThreadPoolExecutor(max_workers=2) as pool:
            pair = list(pool.map(lambda _: run(root, state, "solve", "-", raw=task), range(2)))
        by_id = sorted(pair, key=lambda record: record["id"])
        assert by_id[0]["kind"] == "research" and by_id[1]["kind"] == "solve"
        if by_id[0]["decision"] == "kept":
            assert by_id[1]["before"]["ok"]
        else:
            assert by_id[1]["reused_event"] == by_id[0]["id"]
        with sqlite3.connect(state / "agentic.sqlite3") as db:
            assert db.execute("pragma integrity_check").fetchone() == ("ok",)
        return dict(reproduced_events=9, same_evidence_except_clock=True, standalone=True,
                    transfer_ops=answer["cost"], rollback_generation=0, invalid_budget_atomic=True,
                    concurrent_decisions=[event["decision"] for event in by_id], sqlite_integrity="ok")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
