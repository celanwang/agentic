"""Reproduce the public #11 misses and isolate the next-route tradeoff."""

import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory

from candidates.pair_eval import seal as pairs
from .frozen import copy_runtime
from .seal import HERE, ROOT, ordered


PUBLIC = ("candidates/pair_eval/probe_result.json", "candidates/pilot_eval/blind_result.json",
          "candidates/pilot_eval/stream_result.json")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def summarize(rows, arm):
    old, new = [row["active"] for row in rows], [row[arm] for row in rows]
    return dict(cases=len(rows), solved=sum(r["ok"] for r in new), operations=sum(r["cost"] for r in new),
                gains=sum(b["ok"] and not a["ok"] for a, b in zip(old, new)),
                losses=sum(a["ok"] and not b["ok"] for a, b in zip(old, new)),
                changed_costs=sum(a["cost"] != b["cost"] for a, b in zip(old, new)))


def evaluate():
    from packlab import memory, research, recipes
    from packlab.experiment import Bench

    record = read("candidates/pair_eval/probe_result.json")
    assert memory.build_id() == record["release_build"]
    original = record["event"]["nominee"]
    chain = research.dispatch(original, record["event"]["policies_before"])
    assert len(chain) == 2
    old_guard = original["guard"]
    assert old_guard == {"feature": "complementary_coverage", "op": "ge", "value": 1000}
    # An 800‰ cut would admit 4 of 5 exact pairs. This is a diagnostic edit only.
    arms = {"active": chain, "open": [dict(original, guard=dict(old_guard, value=0)), chain[1]],
            "raw_ge_800": [dict(original, guard=dict(old_guard, value=800)), chain[1]],
            "arrival_first": [chain[1], chain[0]]}

    def observe(task, label, cohort):
        bench = Bench()
        bins, cap, sizes = pairs.core(task)
        result = dict(label=label, cohort=cohort, coverage=recipes.route_features(task)["complementary_coverage"],
                      exact_after_peel=pairs.exact_pairs(cap, bins, sizes))
        for arm, policies in arms.items():
            outcome = bench.apply(task, policies)
            if outcome["ok"]:
                assert pairs.check(task, outcome["packing"])
            trial = bench.single(task, policies[0])
            trace = trial["trace"]
            result[arm] = dict(ok=outcome["ok"], cost=outcome["cost"],
                               guard=trace.get("guard"), route=trace.get("route"),
                               ops=sum(item["cost"] for item in outcome["attempts"]))
            assert result[arm]["ops"] == result[arm]["cost"]
        return result

    public, reordered = [], []
    for case, old in zip(record["reveal"]["cases"], record["rows"]):
        assert case["name"] == old["name"]
        if old["head"]["ok"]:
            continue
        task = case["task"]
        row = observe(task, case["name"], "public_miss")
        assert (row["active"]["ok"], row["active"]["cost"]) == (old["head"]["ok"], old["head"]["cost"])
        assert row["active"]["route"]["primary"]["search_status"] == "guard_skip"
        assert row["active"]["route"]["fallback"]["search_status"] == "guard_skip"
        public.append(row)
        control, _ = ordered((task, case["witness"]))
        assert pairs.core(control) == pairs.core(task)
        control_row = observe(control, case["name"], "same_core_arrival_order_control")
        assert control_row["active"]["ok"] and control_row["arrival_first"]["ok"]
        reordered.append(control_row)
    assert len(public) == 12

    archive = read("candidates/pilot_eval/blind_result.json")
    known = read("candidates/pilot_eval/stream_result.json")
    lookup = {(group["id"], item["variant"]): item["task"] for group in archive["reveal"]["clusters"]
              for item in group["cases"]}
    old_successes = [observe(lookup[row["cluster"], row["variant"]], row["cluster"] + "/" + row["variant"],
                             "published_stream_success") for row in known["rows"] if row["modes"]["active"]["result"]["ok"]]
    assert len(old_successes) == 36
    all_rows = public + reordered + old_successes
    print(json.dumps({"build": record["release_build"], "arms": arms,
                      "summary": {name: {arm: summarize([row for row in all_rows if row["cohort"] == name], arm)
                                           for arm in arms}
                                  for name in ("public_miss", "same_core_arrival_order_control", "published_stream_success")},
                      "rows": all_rows}, indent=2, sort_keys=True))


def main():
    with TemporaryDirectory(prefix=".route-development-", dir=HERE) as temporary:
        workspace = Path(temporary)
        copy_runtime(workspace)
        for name in (*PUBLIC, str(Path(__file__).resolve().relative_to(ROOT)),
                     str((HERE / "seal.py").relative_to(ROOT)), str((HERE / "frozen.py").relative_to(ROOT)),
                     str(Path(pairs.__file__).resolve().relative_to(ROOT))):
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT / name, target)
        subprocess.run([sys.executable, "-B", "-m", "candidates.route_review.eval.development", "--frozen-child"],
                       cwd=workspace, check=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--frozen-child"]:
        evaluate()
    else:
        main()
