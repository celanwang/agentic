"""Run the current regression suite and optionally replay the published evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(label, *args, expected=None):
    env = dict(os.environ)
    for key in ("PYTHONOPTIMIZE", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).resolve().parent), env.get("PATH", os.defpath)))
    print(f"正在核验  {label}", flush=True)
    process = subprocess.run([sys.executable, "-B", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if process.returncode:
        sys.exit((process.stderr or process.stdout).strip() or f"{label} 失败")
    if expected and json.loads(process.stdout) != json.loads((ROOT / expected).read_text(encoding="utf-8")):
        sys.exit(f"{label} 的输出与 {expected} 不符")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="重放冻结评测、搜索预算对照和 8 盘追试")
    args = parser.parse_args()
    check("分盘与 CLI", "candidates/audit/audit.py", "cli")
    check("并发、回退和预算", "candidates/audit/integrity.py")
    check("只读状态与失败写入", "candidates/audit/readonly_cli.py")
    check("历史装法与留出隔离", "candidates/audit/history_edges.py")
    check("规划器跨轮记忆", "candidates/audit/planner_edges.py")
    check("现役版从空白重启", "candidates/audit/research_cli.py")
    check("冻结历史可独立恢复", "candidates/audit/frozen_replay.py")
    if args.full:
        check("历史发布的四份逐项评测", "-m", "candidates.pilot_eval.replay")
        check("历史第 9 轮因果对照", "candidates/pilot_search/replay_round01.py",
              expected="candidates/pilot_search/round01_replay.json")
        check("两种记忆顺序的等预算复验", "candidates/audit/planner_release_compare.py",
              expected="evidence/review/planner-boundaries.json")
        check("8 盘触发与 70 份只读复查", "candidates/audit/pair_followup.py", expected="evidence/review/pair-followup.json")
    print("全部通过。")


if __name__ == "__main__":
    main()
