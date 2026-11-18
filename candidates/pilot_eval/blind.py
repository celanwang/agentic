"""Publish a salted suite commitment without disclosing tasks or witnesses."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sqlite3

from . import suite


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V2_STATE = ROOT / "state/mixed/20261118-201146-mt88urf4"
PRIVATE = HERE / ".sealed_pilot"
PROTOCOL = HERE / "protocol.json"
COMMITMENT = HERE / "blind_commitment.json"
PAYLOAD = PRIVATE / "payload.json"


def digest(data):
    return sha256(data).hexdigest()


def write_once(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")


def check_official_bound():
    assert digest((ROOT / "packlab/arena.py").read_bytes()) == suite.ARENA_SHA256


def direct_signatures(folder):
    db = folder.resolve() / "agentic.sqlite3"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        return {suite.key(task) for raw, in connection.execute("SELECT task FROM events WHERE kind='solve'")
                if (task := json.loads(raw)) is not None}


def blind_keys(payload):
    signatures = set()
    for group in payload["clusters"]:
        assert tuple(case["variant"] for case in group["cases"]) == suite.VARIANTS
        core = suite.key(group["cases"][0]["task"])
        assert core not in signatures and core[1] > 379
        for case in group["cases"]:
            assert suite.key(case["task"]) == core and suite.check(case["task"], case["witness"])
        signatures.add(core)
    return signatures


def seal():
    check_official_bound()
    PRIVATE.mkdir(mode=0o700)
    payload = suite.blind(secrets.token_hex(32))
    signatures = blind_keys(payload)
    assert len(signatures) == 18 and signatures.isdisjoint(direct_signatures(V2_STATE))
    write_once(PAYLOAD, payload)
    public = {"protocol": suite.VERSION, "protocol_sha256": digest(PROTOCOL.read_bytes()),
              "suite_sha256": digest(Path(suite.__file__).read_bytes()),
              "seal_sha256": digest(Path(__file__).read_bytes()),
              "arena_sha256": suite.ARENA_SHA256,
              "blind_sha256": digest(suite.canonical(payload)),
              "clusters": len(signatures), "variants": list(suite.VARIANTS)}
    write_once(COMMITMENT, public)
    print(json.dumps(public, indent=2))


def verify(payload, committed):
    check_official_bound()
    assert committed["suite_sha256"] == digest(Path(suite.__file__).read_bytes())
    assert committed["protocol_sha256"] == digest(PROTOCOL.read_bytes())
    assert committed["seal_sha256"] == digest(Path(__file__).read_bytes())
    assert committed["blind_sha256"] == digest(suite.canonical(payload))
    assert payload == suite.blind(payload["nonce"])
    assert len(blind_keys(payload)) == committed["clusters"]


if __name__ == "__main__":
    seal()
