"""Replay the published eleventh decision with its own offline runtime."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory, mkdtemp
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frozen_pilot import run_python

ARCHIVE = ROOT / "evidence/pairs/round11-source.zip"
ARCHIVE_SHA256 = "ecd7b6d36e44c052823f4043360e0fe3525c6a0204bf10d52a39ff86f4110f00"
# This runs inside the frozen checkout. The output state belongs to the caller.
REPLAY = """
from pathlib import Path
import json, sys
from scripts import pair_round as published
root = Path(sys.argv[1])
before, live = published.prepare(root)
report = published.conduct(before, live)
expected = json.loads((published.ROOT / "evidence/pairs/decision.json").read_text())
if report != expected:
    raise RuntimeError("第 11 轮逐项结果与发布记录不同")
published.show_study(published.cli(live, "show", "11"))
print("已知失败", report["known"]["old_ok"], "→", report["known"]["new_ok"], "/ 40")
published.restart(before, live)
"""


def replay(root):
    if any(root.iterdir()):
        raise ValueError("重建第 11 轮需要空目录")
    data = ARCHIVE.read_bytes()
    if sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("第 11 轮源码归档摘要不符")
    with TemporaryDirectory(prefix="agentic-round11-") as temp:
        workspace = Path(temp)
        with ZipFile(BytesIO(data)) as archive:
            for name in ("evidence/pairs/release.json", "evidence/pairs/protocol.md", "evidence/pairs/decision.json",
                         "candidates/pair_eval/blind_commitment.json", "candidates/pair_eval/probe_result.json"):
                if archive.read(name) != (ROOT / name).read_bytes():
                    raise ValueError(f"第 11 轮发布证据与归档不符  {name}")
            archive.extractall(workspace)
        (workspace / "agentic").chmod(0o755)
        return run_python(workspace, "-c", REPLAY, str(root.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="在临时目录核对发布证据")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="agentic-pairs-") as temp:
        if args.check:
            root = Path(temp)
        else:
            parent = ROOT / "state/demos"
            parent.mkdir(exist_ok=True, parents=True)
            root = Path(mkdtemp(prefix="pair-round-", dir=str(parent)))
        sys.stdout.buffer.write(replay(root))
        if not args.check:
            print("用现役程序继续这份记忆  " + shlex.join([str(ROOT / "agentic"), "--state", str(root / "live")]))


if __name__ == "__main__":
    main()
