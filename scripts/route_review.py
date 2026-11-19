"""Replay round 12 with its archived code and independently checked public inputs."""
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

ARCHIVE = ROOT / "evidence/route-review/round12-source.zip"
ARCHIVE_SHA256 = "a1114cd69ea53c936e0a4e380fd6e88ad3dcbcd1369c9ea653eaaed2660d420c"


def replay(root):
    raw = ARCHIVE.read_bytes()
    if sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("第 12 轮源码归档摘要不符")
    with TemporaryDirectory(prefix="agentic-round12-code-") as temp:
        workspace = Path(temp)
        with ZipFile(BytesIO(raw)) as archive:
            archive.extractall(workspace)
            assets = json.loads(archive.read("assets.json"))
        (workspace / "agentic").chmod(0o755)
        for name, digest in assets.items():
            data = (ROOT / name).read_bytes()
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"第 12 轮公开输入摘要不符  {name}")
            target = workspace / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(data)
        return run_python(workspace, "-m", "candidates.route_review.replay", str(root.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="在临时目录核对两轮决策和重启盲测")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="agentic-route-review-") as temp:
        root = Path(temp)
        if not args.check:
            demos = ROOT / "state/demos"
            demos.mkdir(exist_ok=True, parents=True)
            root = Path(mkdtemp(prefix="route-review-", dir=str(demos)))
        print("正在从发布源码重建 v2，再执行第 12 轮研究和重启验收。", flush=True)
        sys.stdout.buffer.write(replay(root))
        if not args.check:
            print("用现役程序继续这份记忆  " + shlex.join([str(ROOT / "agentic"), "--state", str(root / "live")]))


if __name__ == "__main__":
    main()
