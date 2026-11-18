"""Independent checks for the frozen packer and the released Agentic CLI."""

import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECISIONS = {"kept", "rejected", "unchanged", "budget", "inconclusive"}


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def task(capacity, bins, items):
    return {"capacity": capacity, "bins": bins, "items": list(items)}


def feasible(problem):
    """Decide feasibility by covering the file set with whole-disk subsets."""
    capacity = problem["capacity"]
    bins = problem["bins"]
    items = problem["items"]
    if len(items) > 14:
        raise AuditError("subset oracle expects at most 14 items")
    if any(size > capacity for size in items) or sum(items) > bins * capacity:
        return False

    weights = [0] * (1 << len(items))
    for mask in range(1, len(weights)):
        bit = mask & -mask
        weights[mask] = items[bit.bit_length() - 1] + weights[mask ^ bit]

    @lru_cache(maxsize=None)
    def cover(remaining, slots):
        if remaining == 0:
            return True
        total = weights[remaining]
        if total <= capacity:
            return slots > 0
        if total > slots * capacity:
            return False

        # Disks are interchangeable. The next disk may always take this file.
        anchor = remaining & -remaining
        others = remaining ^ anchor
        min_weight = total - (slots - 1) * capacity
        subset = others
        while True:
            disk = subset | anchor
            if min_weight <= weights[disk] <= capacity:
                if cover(remaining ^ disk, slots - 1):
                    return True
            if subset == 0:
                return False
            subset = (subset - 1) & others

    return cover(len(weights) - 1, bins)


def verify_oracle():
    checked = 0
    for capacity, bins, length in itertools.product(range(1, 5), range(1, 4), range(5)):
        for items in itertools.combinations_with_replacement(range(1, 6), length):
            # Tiny cross-check using every labeled assignment, with no subset pruning.
            expected = any(
                all(
                    sum(size for size, slot in zip(items, assignment) if slot == disk) <= capacity
                    for disk in range(bins)
                )
                for assignment in itertools.product(range(bins), repeat=length)
            )
            require(feasible(task(capacity, bins, items)) == expected, f"oracle error for {items}")
            checked += 1
    return checked


def validate_packing(problem, packing, complete):
    require(isinstance(packing, list), "packing must be a list of disks")
    require(len(packing) == problem["bins"], "incorrect disk count")
    items = problem["items"]
    seen = set()
    for disk in packing:
        require(isinstance(disk, list), "disk must contain a list of file indices")
        for index in disk:
            require(type(index) is int and 0 <= index < len(items), f"bad item index {index!r}")
            require(index not in seen, f"duplicate item index {index}")
            seen.add(index)
        require(sum(items[index] for index in disk) <= problem["capacity"], "overloaded disk")
    if complete:
        require(len(seen) == len(items), "successful packing omits files")


def find_reorder_regression(pack, descending):
    for length in range(1, 7):
        for bins, capacity in itertools.product(range(2, 5), range(2, 16)):
            for sizes in itertools.combinations_with_replacement(range(1, capacity + 1), length):
                ordered = tuple(reversed(sizes)) if descending else sizes
                if sum(sizes) > capacity * bins or pack(capacity, bins, ordered) is not None:
                    continue
                if not feasible(task(capacity, bins, sizes)):
                    continue
                for arrival in sorted(set(itertools.permutations(sizes))):
                    original = pack(capacity, bins, arrival)
                    if original is not None:
                        problem = task(capacity, bins, arrival)
                        validate_packing(problem, original, complete=True)
                        return {"task": problem, "arrival_packing": original, "reordered": ordered}
    raise AuditError("no reordering regression found in the enumeration bounds")


def audit_baseline():
    baseline = ROOT / "baseline" / "v0.py"
    output = subprocess.run(
        [sys.executable, str(baseline)], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    require(output == "None", f"baseline example returned {output!r}")
    pack = runpy.run_path(str(baseline))["pack"]
    report = {
        "baseline_output": output,
        "oracle_cross_checks": verify_oracle(),
        "bounds": {"bins": [2, 4], "capacity": [2, 15], "file_count": [1, 6]},
        "ascending": find_reorder_regression(pack, descending=False),
        "descending": find_reorder_regression(pack, descending=True),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def reject_constant(token):
    raise ValueError(f"invalid JSON constant {token}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


class CLI:
    def __init__(self, executable, work_dir, timeout):
        self.executable = executable
        self.work_dir = work_dir
        self.timeout = timeout

    def call(self, state, *args):
        command = [str(self.executable), "--state", str(self.work_dir / state), *args, "--json"]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=self.timeout)

    def response(self, state, *args):
        result = self.call(state, *args)
        require(result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip()!r}")
        try:
            output = parse_json(result.stdout)
        except ValueError as error:
            raise AuditError(f"invalid response JSON: {error}; stdout {result.stdout[:300]!r}") from error
        require(isinstance(output, dict), "expected a JSON object")
        return output

    def generation(self, state):
        generation = self.response(state, "status").get("generation")
        require(type(generation) is int and generation >= 0, "invalid status generation")
        return generation

    def solve(self, state, label, problem):
        filename = self.work_dir / f"{label}.json"
        filename.write_text(json.dumps(problem), encoding="utf-8")
        output = self.response(state, "solve", str(filename))
        observed_task = output.get("task")
        require(
            json.dumps(observed_task, sort_keys=True) == json.dumps(problem, sort_keys=True),
            "response task differs from input",
        )
        require(
            type(output.get("generation")) is int and output["generation"] >= 0,
            "invalid solve generation",
        )
        decision = output.get("decision")
        require(isinstance(decision, str) and decision in DECISIONS, f"unknown decision {decision!r}")
        possible = feasible(problem)
        for phase in ("before", "after"):
            attempt = output.get(phase)
            require(isinstance(attempt, dict), f"missing {phase} attempt")
            ok = attempt.get("ok")
            cost = attempt.get("cost")
            require(type(ok) is bool, f"{phase}.ok is not boolean")
            require(type(cost) is int and cost > 0, f"{phase}.cost is not a positive integer")
            require("trace" in attempt, f"{phase} has no trace")
            require("packing" in attempt, f"{phase} has no packing")
            require(possible or not ok, f"{phase} claims success on an infeasible instance")
            if ok or attempt["packing"] is not None:
                validate_packing(problem, attempt["packing"], complete=ok)
        return output


def restart_cases():
    trigger = task(19, 3, [3, 6, 8, 11, 13, 16])
    transfer = task(23, 4, [3, 6, 9, 11, 12, 14, 17, 20])
    return trigger, transfer


def regression_cases():
    yield "descending-regression", task(11, 2, [3, 3, 5, 3, 3, 5])
    yield "empty-spare-disk", task(17, 3, [12, 5, 9, 8])
    rng = random.Random(704533)
    for index in range(10):
        capacity = rng.randrange(11, 28)
        items = []
        for _ in range(3):
            first = rng.randrange(1, capacity - 1)
            second = rng.randrange(1, capacity - first)
            items.extend([first, second, capacity - first - second])
        # Each contiguous triple fills one disk in arrival order.
        yield f"arrival-regression-{index}", task(capacity, 3, items)


def print_result(label, output):
    print(json.dumps({
        "check": label,
        "before": output["before"]["ok"],
        "after": output["after"]["ok"],
        "before_cost": output["before"]["cost"],
        "after_cost": output["after"]["cost"],
        "decision": output["decision"],
        "generation": output["generation"],
    }))


def audit_cli(executable, timeout):
    trigger, transfer = restart_cases()
    require(feasible(trigger) and feasible(transfer), "restart fixtures must be feasible")
    with tempfile.TemporaryDirectory(prefix="run-", dir=HERE) as directory:
        cli = CLI(executable, Path(directory), timeout)
        errors = []

        def check(label, action):
            try:
                action()
            except (AuditError, subprocess.TimeoutExpired) as error:
                errors.append(label)
                print(json.dumps({"check": label, "error": str(error)}), flush=True)

        initial = cli.generation("learned")
        require(initial == 0, "a fresh state must start at generation 0")
        learned = cli.solve("learned", "trigger", trigger)
        print_result("trigger", learned)
        require(not learned["before"]["ok"], "fresh policy unexpectedly solved the trigger")
        require(learned["after"]["ok"] and learned["decision"] == "kept", "trigger did not retain an improvement")
        generation = learned["generation"]
        require(generation == initial + 1, "keep must advance generation by one")
        require(cli.generation("learned") == generation, "fresh status process lost the kept generation")

        def restart_transfer():
            restored = cli.solve("learned", "transfer", transfer)
            print_result("restart-transfer", restored)
            require(restored["before"]["ok"], "restarted policy could not solve the unseen task")
            require(restored["after"]["ok"], "accepted result regressed on the unseen task")
            require(restored["decision"] == "unchanged", "successful transfer should not require a new change")
            require(restored["generation"] == generation, "transfer changed the generation")
            require(cli.generation("learned") == generation, "transfer persisted a different generation")

        def fresh_control():
            require(cli.generation("control") == initial, "separate state inherited a generation")
            output = cli.solve("control", "transfer-control", transfer)
            print_result("fresh-control", output)
            require(not output["before"]["ok"], "unseen transfer would also succeed with a fresh policy")

        check("restart-transfer", restart_transfer)
        check("fresh-control", fresh_control)
        for label, problem in regression_cases():
            def regression():
                before_generation = cli.generation("learned")
                output = cli.solve("learned", label, problem)
                print_result(label, output)
                require(output["before"]["ok"] and output["after"]["ok"], "retained policy regressed on a baseline success")
                require(output["generation"] == before_generation, "baseline success changed the generation")
            check(label, regression)

        for label, problem in (
            ("volume-balanced-impossible", task(13, 2, [9, 9, 3, 3, 2])),
            ("single-oversize-file", task(13, 3, [14, 5, 6])),
        ):
            def infeasible_case():
                initial_generation = cli.generation("learned")
                require(not feasible(problem), "infeasible fixture has a packing")
                output = cli.solve("learned", label, problem)
                print_result(label, output)
                require(output["decision"] != "kept", "kept a change after a proven infeasible task")
                require(output["generation"] == initial_generation, "infeasible task changed generation")
                require(cli.generation("learned") == initial_generation, "infeasible task changed persisted generation")
            check(label, infeasible_case)

        for label, raw in (
            ("invalid-json", '{"capacity":'),
            ("invalid-bool", '{"capacity":13,"bins":true,"items":[1]}'),
            ("invalid-zero", '{"capacity":13,"bins":2,"items":[0]}'),
            ("invalid-float", '{"capacity":13,"bins":2,"items":[2.0]}'),
        ):
            def invalid_case():
                initial_generation = cli.generation("learned")
                path = cli.work_dir / f"{label}.json"
                path.write_text(raw, encoding="utf-8")
                output = cli.call("learned", "solve", str(path))
                require(output.returncode != 0, "invalid task exited successfully")
                require(bool(output.stdout.strip() or output.stderr.strip()), "invalid task has no error evidence")
                require(cli.generation("learned") == initial_generation, "invalid task changed persisted generation")
                print(json.dumps({"check": label, "exit": output.returncode}))
            check(label, invalid_case)

        require(not errors, f"failed checks: {', '.join(errors)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("baseline", help="reproduce v0 and search for reordering regressions")
    cli_parser = commands.add_parser("cli", help="run isolated acceptance checks for the released CLI")
    cli_parser.add_argument("--agentic", type=Path, default=ROOT / "agentic")
    cli_parser.add_argument("--timeout", type=float, default=25, help="timeout per subprocess in seconds")
    options = parser.parse_args()
    try:
        if options.command == "baseline":
            audit_baseline()
        else:
            audit_cli(options.agentic.resolve(), options.timeout)
    except (AuditError, OSError, subprocess.SubprocessError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
