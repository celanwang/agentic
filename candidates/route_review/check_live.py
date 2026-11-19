"""Exercise the installed CLI on an upgrade from the published v2 database."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidates.audit.pair_followup import cli
from candidates.route_review.replay import conduct, known_cases, stages
from packlab import arena


def check(root):
    before, live = stages(root)
    report = conduct(before, live)
    event_id = report["record"]
    event = cli(live, "show", str(event_id))
    status = cli(live, "status")
    assert status["build"] == event["build"]
    if event["nominee"]:
        witness = cli(live, "explain", str(event_id))
        assert witness["runs_used"] <= witness["budget"]
    assert cli(live, "status") == status
    first = known_cases()[0]
    delivered = cli(live, "solve", "-", task=first)
    if event["answer"]:
        assert delivered["answer"] and arena.check(first, delivered["answer"]["packing"])
        if not delivered["before"]["ok"]:
            assert delivered["reused_event"] == event_id
    rolled = cli(live, "rollback")
    assert rolled["generation"] == status["parent"]
    assert cli(before, "status")["events"] == 11


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        check(Path(temp))
    print("Current CLI upgraded the frozen store, explained the decision, recovered an answer and rolled back.")
