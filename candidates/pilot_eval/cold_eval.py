"""Describe the fixed cold-start event separately from the primary blind result."""

import json

from . import blind, final_eval, suite


def main():
    from packlab import memory, research

    primary = json.loads(final_eval.RESULT.read_text())
    build = primary["build"]
    assert memory.build_id() == build
    blind.verify(primary["reveal"], primary["commitment"])
    state = blind.ROOT / "state/pilot/round-01/cold"
    formal = final_eval.formal_record(state, 1)
    assert formal["build"] == build and formal["rules"] == primary["formal"]["rules"]
    with final_eval.source(state) as db:
        direct = [json.loads(raw) for raw, in db.execute("SELECT task FROM events")]
    sealed_keys = {final_eval.forced_key(group["cases"][0]["task"]) for group in primary["reveal"]["clusters"]}
    assert sealed_keys.isdisjoint(final_eval.forced_key(task) for task in direct if task)
    folder, (generation, active) = final_eval.snapshot(state, "cold")
    assert generation == formal["generation"] == formal["generation_before"] == 0
    assert active == formal["policies_before"]
    nominee = research.dispatch(formal["program"], active)
    executable = blind.ROOT / "agentic"
    rows = []
    for group in primary["reveal"]["clusters"]:
        for case in group["cases"]:
            task = case["task"]
            a = final_eval.probe(executable, folder, task, active)
            b = final_eval.probe(executable, folder, task, nominee, 1)
            rows.append({"cluster": group["id"], "profile": group["profile"], "variant": case["variant"],
                         "modes": {"active": a, "nominee": b}})
    paired_rows = [dict(row, modes=dict(row["modes"], v2=row["modes"]["active"])) for row in rows]
    metrics = {mode: final_eval.summary(paired_rows, mode) for mode in ("active", "nominee")}
    assert memory.runtime_id(sorted(memory.ROOT.glob("*.py"))) == build
    path = blind.HERE / "cold_result.json"
    blind.write_once(path, {"build": build, "blind_sha256": primary["commitment"]["blind_sha256"],
                            "formal": formal, "active_revision": generation, "metrics": metrics, "rows": rows})
    print(json.dumps({"decision": formal["decision"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
