#!/usr/bin/env python
"""Download the ungated base-model candidates of area "CAD" into plain directories (host step, network on, no GPU).

Models go to `models/<name>/` as plain files (no symbolic links, MM-20). huggingface_hub's own download metadata
(`<local_dir>/.cache/huggingface/`) is moved out to `hf-meta/<name>/`, so each directory holds exactly the
repository files at the pinned commit. No token is used or read: `token=False` on every call,
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh `hf-home/` of this directory (never an
existing cache); a token in the environment is removed from this process's environment before huggingface_hub is
imported (its value is never read). Xet transfer is disabled (plain HTTPS). There is no dataset download: the
training and item data are generated (`make_data.py`, README "Data").

`models/MODELS.json` records per repository: repo, revision (40-hex commit), license, the files and their SHA-256
and bytes; `--verify` re-hashes and compares. Run with any venv that has huggingface_hub (the CPU pilot used the
area-4 venv read-only, huggingface-hub 2.3.0).

Usage (host):
  python fetch_models.py            # download what is missing, write MODELS.json
  python fetch_models.py --verify   # re-hash every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

# Revisions resolved 2026-11-13 from the Hub API (`sha` of `main`); both repos reported `gated: false`,
# license apache-2.0. Only safetensors weights (no .bin). LICENSE kept for provenance.
QWEN_FILES = ["LICENSE", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "vocab.json"]
MODEL_SPECS = {
    "qwen2.5-coder-0.5b-instruct": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
                                    QWEN_FILES, "apache-2.0"),
    # Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca (Tier B's base) was downloaded for the pilot, not
    # chosen, and moved to ../area6-ref/models/qwen3-0.6b with its record (../area6-ref/models/MODELS.json).
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _hub_env() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)


def _move_meta(dest: Path, name: str) -> None:
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))


def download() -> int:
    _hub_env()
    from huggingface_hub import hf_hub_download

    record = {"schema": "ptc_mm.cad_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {}}
    for name, (repo, revision, files, lic) in MODEL_SPECS.items():
        dest = MODELS / name
        for f in files:
            hf_hub_download(repo, f, revision=revision, local_dir=dest, token=False)
        _move_meta(dest, name)
        got = plain_files(dest)
        if sorted(got) != sorted(files):
            raise SystemExit(f"{dest} holds {sorted(got)}, not {sorted(files)}")
        record["models"][name] = {"repo": repo, "revision": revision, "license": lic, "files": got,
                                  "bytes": sum(f["bytes"] for f in got.values())}
        print(f"{name}: {len(got)} files, {record['models'][name]['bytes'] / 1e6:.1f} MB", flush=True)
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def verify() -> int:
    bad = 0
    record = json.loads((MODELS / "MODELS.json").read_text(encoding="utf-8"))
    for name, m in record["models"].items():
        got = plain_files(MODELS / name)
        ok = got == m["files"]
        bad += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)
    return verify() if args.verify else download()


if __name__ == "__main__":
    sys.exit(main())
