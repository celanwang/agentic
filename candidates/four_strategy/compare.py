"""Count the same variable-cardinality bin search at two fixed work limits."""
import json

import engine
from prefix import cases, valid, v2


def trials():
    rows = []
    for row, task in cases():
        common = v2.run(task, dict(v2.BASE, order="descending"))
        for repair in ("fill", "fill_deep"):
            result = engine.run(task, dict(v2.BASE, order="descending", repair=repair))
            assert result["ok"] == valid(task, result["packing"])
            delta = {k: v - common["trace"]["ops"].get(k, 0) for k, v in result["trace"]["ops"].items()}
            assert 0 < sum(delta.values()) == result["cost"] - common["cost"] <= engine._FILL_LIMITS[repair]
            rows.append(dict(row=row, repair=repair, status=result["trace"]["search_status"],
                             cost=result["cost"], greedy=common["cost"], search_ops=delta,
                             packing=result["packing"]))
    return rows


if __name__ == "__main__":
    print(json.dumps(trials(), indent=2))
