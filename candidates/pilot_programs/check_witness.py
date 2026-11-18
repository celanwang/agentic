"""Short reproducible witness from a user-style task. Uses no saved state or splits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from candidates.pilot_programs.witness import explain
from packlab import arena, engine


def main():
    task = dict(capacity=10000, bins=4, items=[3000,6000,9999,1000,2000,2000,5000,9000,2000])
    p = dict(kind="recipe", steps=["peel_forced", "normalize_gcd"], guard=None,
             solver=dict(engine.BASE, order="descending", repair="fill"))
    data = explain([task], p, [engine.BASE])
    witness = data["witness"]
    reduced = witness["reduced"]
    assert witness["control"] == "without_peel_forced" and witness["deleted_original"] == [3]
    assert reduced["difference"] == "success_gain"
    assert arena.check(reduced["task"], reduced["full"]["packing"])
    assert sorted(i for g in reduced["full"]["packing_original"] for i in g) == reduced["kept_original"]
    assert data["runs_used"] <= data["budget"]
    print(json.dumps(dict(control=witness["control"], deleted_original=witness["deleted_original"],
                          packing_original=reduced["full"]["packing_original"],
                          full_cost=reduced["full"]["cost"], control_cost=reduced["control"]["cost"],
                          control_outcome=reduced["control"]["outcome"], local_stop=witness["local_stop"],
                          budget=data["budget"], runs_used=data["runs_used"], executed_ops=data["executed_ops"]),
                     ensure_ascii=False, indent=2))

    # Removing a guard can spend more operations for the very same outcome.
    t = dict(capacity=10, bins=2, items=[4,4,6,6])
    gated = dict(kind="recipe", steps=[], guard=dict(feature="capacity", op="le", value=1),
                 solver=dict(engine.BASE, repair="deep"))
    compare = explain([t], gated, [dict(engine.BASE, order="descending")])
    saved = compare["witness"]
    assert saved["control"] == "without_guard" and saved["reduced"]["difference"] == "cost_saving"
    assert saved["reduced"]["full"]["cost"] < saved["reduced"]["control"]["cost"]
    noop = dict(kind="recipe", steps=[], solver=engine.BASE, guard=None)
    prior_winner = dict(noop, solver=dict(engine.BASE, order="descending"))
    loss = explain([t], noop, [prior_winner])
    assert loss["witness"]["reduced"]["difference"] == "success_loss"
    same = explain([t], noop, [noop, engine.BASE])
    assert same["witness"] is None and same["unchecked_input_pairs"] == 0
    small = explain([task], p, [engine.BASE], budget=1)
    assert small["runs_used"] == 1 and small["unchecked_input_pairs"] > 0 and small["witness"] is None


if __name__ == "__main__":
    main()
