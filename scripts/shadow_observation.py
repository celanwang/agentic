"""Restore the published v2, replay the sealed stream, then open an empty solve cohort."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.route_review import replay as restore


def command(state, *args):
    raw = subprocess.check_output([str(ROOT / "agentic"), "--state", str(state), *args, "--json"], text=True)
    return json.loads(raw)


def run(root, check):
    print("正在重建第 12 轮现场。", flush=True)
    restored = root / "history"
    restored.mkdir()
    restore(restored)
    state = restored / "live"
    build = command(state, "status")["build"]
    result_path = ROOT / "candidates/shadow/result.json"
    raw = subprocess.check_output([sys.executable, "-B", "-m", "candidates.shadow.eval", "run",
                                  "--state", str(state), "--build", build, "--payload", str(result_path)],
                                 cwd=ROOT, text=True)
    tested = json.loads(raw)
    published = json.loads(result_path.read_text())
    release = json.loads((ROOT / "evidence/shadow/release.json").read_text())
    if published["commitment"]["commitment"] != release["commitment"]:
        raise RuntimeError("发布结果与事前封存的输入摘要不符")
    if not replay_matches(tested, published, build, release["build"]):
        raise RuntimeError("封存观察流与发布时的计数或逐项结果不同")
    tally = tested["status"]
    print(f"封存合成流共对照 {tally['compared']} 个新家族，{tally['branch_changes']} 个换分支。"
          f"活动策略装完 {tally['old_ok']} 份，候选装完 {tally['new_ok']} 份。"
          f"两端各用 {tally['old_cost']} 和 {tally['new_cost']} 次操作。", flush=True)
    if check:
        from candidates.shadow.check_solve import check as check_serving
        check_serving(state, root)
    opened = command(state, "shadow", "start", "12")
    if opened["source"] != "solve" or opened["arrivals"] != 0 or opened["state"] != "active":
        raise RuntimeError("日常窗口未从零开始")
    print("日常窗口已开启。接下来按到达顺序观察前 32 个新家族。")
    if not check:
        base = [str(ROOT / "agentic"), "--state", str(state)]
        print("提交自己的清单  " + shlex.join(base))
        print("查看进度  " + shlex.join(base + ["shadow", "status"]))


def replay_matches(tested, published, running_build, released_build):
    def evidence(result, expected_build):
        # Only the current observer may change between releases. The underlying
        # research build, sealed stream, individual paths and costs must match.
        status = dict(result["status"])
        observer = status.pop("build")
        fields = dict(result, status=status)
        if fields.pop("release_build") != expected_build or observer != expected_build:
            raise ValueError("观察程序的构建摘要不符")
        return fields

    return evidence(tested, running_build) == evidence(published, released_build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="用临时状态验收真实入口和封存流")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="agentic-shadow-") as temp:
        root = Path(temp)
        if not args.check:
            demos = ROOT / "state/demos"
            demos.mkdir(parents=True, exist_ok=True)
            root = Path(mkdtemp(prefix="shadow-", dir=str(demos)))
        run(root, args.check)


if __name__ == "__main__":
    main()
