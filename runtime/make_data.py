#!/usr/bin/env python
"""Build the generated text-to-CadQuery data of area "CAD" (host step, CPU, executor venv).

Family split (decided 2026-11-13, area 6): 5 of the 15 families of
`cad_families.py` are held out of SFT (HELD_OUT below; README "Task and data" gives the reasons). Four disjoint sets:
- `data/sft/`   : SFT rows from the 10 SFT families, `output` = reference program in one fenced block
                  (cad_protocol.completion). NeMo RL `ResponseDataset` keys `input` / `output`.
- `data/grpo/`  : GRPO rows from all 15 families, `output` = canonical JSON of the reference's OCCT features
                  `{row_id, volume, bbox}` (NeMo RL passes it to the environment as `ground_truth`).
- `items-sft/`  : SFT gate items, 30 per SFT family (300), from the 10 SFT families only.
- `items-grpo/` : GRPO gate items, 60 per held-out family (300), from the 5 held-out families only.
Item rows: `{item, family, text, description, reference_code, ref_mesh, ref_mesh_sha256, ref_volume, ref_bbox,
params}` plus `<set>/ref/<item>.stl` (the reference mesh for the scorer).

Each training set is written as shards of 500 rows (`shards/shard-00000.jsonl`, ...) and their concatenation in
file-name order `train.jsonl` (EXT-65: `train_file_sha256` = SHA-256 of that concatenation), as the Tier B export.

Disjointness: a geometry (family plus every dimension, phrasing ignored) appears in at most one of the four sets; the
exact prompt text likewise (re-checked over the written files). Every reference program is executed in the sandbox
(cad_sandbox.Zygote) and must yield a valid solid; a draw whose reference fails is redrawn and counted. Draws:
`random.Random` seeded with SHA-256("<seed>:<family>") per family, in the order: that family's gate items, GRPO rows,
SFT rows; so the item sets do not depend on the training-set sizes. The rows of each training set are then permuted
once (seed "<seed>:order:<split>"), because the trainer reads the file in order.

Recorded script hash: `generator.script_sha256` in the three metadata files (META_FILES) records the revision of this
script that generated the committed data. Later revisions changed only documentation and this bookkeeping, not the
output. So when a run reproduces the committed metadata byte for byte except for that one hash, it keeps the recorded
hash and the committed files stay unchanged (their SHA-256 enter the registered item-set hashes); any other difference
writes fresh metadata with this file's own hash. README "Setup" describes the check.

Usage (host): executor/.venv/bin/python -I make_data.py [--seed 20261113]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))   # cad_protocol, cad_sandbox (the executor mount)
import cad_families  # noqa: E402
import cad_protocol  # noqa: E402
from cad_sandbox import Zygote  # noqa: E402

ROWS_PER_SHARD = 500
# Held out of SFT; trained by GRPO; the GRPO gate's items (split B, README "Task and data"). Each differs from an SFT
# family by one step (blind_hole: a depth on the plate_hole / cbore_plate hole; plate_4holes: construction-rect
# vertices; tube: a second circle; flange: polarArray on a cylinder; chamfer_box: chamfer instead of fillet_box's
# fillet). Split A (stepped_shaft, fillet_box, chamfer_box, cup, flange: operations no SFT family uses) left the CPU
# SFT proxy at mean IoU 0.10 with 74 % zeros on its held-out items (pilot/v2-split-a/), degenerate under MM-38.
HELD_OUT = ("blind_hole", "plate_4holes", "tube", "flange", "chamfer_box")
ITEM_SETS = {"items-sft": 30, "items-grpo": 60}     # per family
META_FILES = ("data/data_meta.json", "items-sft/item_set_meta.json", "items-grpo/item_set_meta.json")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def family_rng(seed: int, family: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{family}".encode()).digest()[:8], "big"))


def split_counts(total: int, fams: list[str]) -> dict[str, int]:
    n = len(fams)
    return {f: total // n + (1 if i < total % n else 0) for i, f in enumerate(fams)}


def write_set(root: Path, rows: list[dict]) -> dict:
    if root.exists():
        shutil.rmtree(root)
    (root / "shards").mkdir(parents=True)
    lines = [canon(r) + "\n" for r in rows]
    shard_hashes = {}
    for k in range(0, len(lines), ROWS_PER_SHARD):
        name = f"shard-{k // ROWS_PER_SHARD:05d}.jsonl"
        data = "".join(lines[k:k + ROWS_PER_SHARD]).encode("utf-8")
        (root / "shards" / name).write_bytes(data)
        shard_hashes[name] = sha256_bytes(data)
    concat = "".join(lines).encode("utf-8")
    (root / "train.jsonl").write_bytes(concat)
    return {"rows": len(rows), "rows_per_shard": ROWS_PER_SHARD, "shards": shard_hashes,
            "train_jsonl_sha256": sha256_bytes(concat)}


def render(meta: dict) -> bytes:
    return (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")


def recorded_script_sha256(fresh: dict[str, dict], committed: dict[str, bytes]) -> str | None:
    """The `generator.script_sha256` that all `committed` metadata files record, if `fresh` (path -> metadata with this
    file's own hash) renders to exactly the committed bytes once that hash is replaced by the recorded one; else None."""
    if set(committed) != set(fresh):
        return None
    try:
        recorded = {json.loads(b)["generator"]["script_sha256"] for b in committed.values()}
    except (ValueError, KeyError, TypeError):
        return None
    if len(recorded) != 1:
        return None
    (h,) = recorded
    if not isinstance(h, str):
        return None
    for rel, meta in fresh.items():
        if render({**meta, "generator": {**meta["generator"], "script_sha256": h}}) != committed[rel]:
            return None
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--n-grpo", type=int, default=2000)
    ap.add_argument("--n-sft", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    committed = {rel: (HERE / rel).read_bytes() for rel in META_FILES if (HERE / rel).is_file()}

    fams = list(cad_families.FAMILIES)
    assert all(f in fams for f in HELD_OUT)
    sft_fams = [f for f in fams if f not in HELD_OUT]
    want = {"grpo": split_counts(a.n_grpo, fams), "sft": split_counts(a.n_sft, sft_fams)}
    zyg = Zygote(workers=a.workers)
    seen_geom, seen_text = set(), set()
    out = {"items-sft": [], "items-grpo": [], "grpo": [], "sft": []}
    stats = {"redrawn_duplicate": 0, "redrawn_reference_failed": 0, "reference_failures": {}}
    tmp = Path(tempfile.mkdtemp(prefix="cad-make-data-"))
    try:
        for fam in fams:
            rng = family_rng(a.seed, fam)
            gate = "items-grpo" if fam in HELD_OUT else "items-sft"
            plan = [(gate, ITEM_SETS[gate]), ("grpo", want["grpo"][fam]), ("sft", want["sft"].get(fam, 0))]
            for split, need in plan:
                got = []
                is_item = split.startswith("items")
                while len(got) < need:
                    batch = []
                    while len(batch) < (need - len(got)):
                        key, desc, code = cad_families.draw(fam, rng)
                        g = cad_families.geometry_key(key)
                        prompt = cad_protocol.user_prompt(desc)
                        if g in seen_geom or prompt in seen_text:
                            stats["redrawn_duplicate"] += 1
                            continue
                        seen_geom.add(g)
                        seen_text.add(prompt)
                        batch.append((key, desc, code, prompt))
                    progs = [{"id": str(i), "code": c, "mesh": is_item} for i, (_, _, c, _) in enumerate(batch)]
                    keep = tmp / f"{fam}-{split}-{len(got)}"
                    res = zyg.run_batch(progs, keep_dir=str(keep) if is_item else None)
                    for i, ((key, desc, code, prompt), r) in enumerate(zip(batch, res)):
                        if r["status"] != "ok":
                            stats["redrawn_reference_failed"] += 1
                            stats["reference_failures"][r["status"]] = stats["reference_failures"].get(
                                r["status"], 0) + 1
                            continue
                        got.append({"family": fam, "key": list(key), "description": desc, "code": code,
                                    "prompt": prompt, "volume": r["volume"], "bbox": r["bbox"],
                                    "mesh": str(keep / str(i) / "mesh.stl") if is_item else None})
                out[split].extend(got)
                print(f"{fam:14s} {split:10s} {len(got)}", flush=True)

        metas = {}
        for iset in ITEM_SETS:
            d = HERE / iset
            if d.exists():
                shutil.rmtree(d)
            (d / "ref").mkdir(parents=True)
            rows = []
            tag = iset.split("-")[1]
            for k, rec in enumerate(out[iset]):
                item = f"cad-{tag}-{rec['family']}-{k:04d}"
                dst = d / "ref" / f"{item}.stl"
                shutil.copyfile(rec["mesh"], dst)
                rows.append({"item": item, "family": rec["family"], "text": rec["prompt"],
                             "description": rec["description"], "reference_code": rec["code"],
                             "ref_mesh": f"ref/{item}.stl", "ref_mesh_sha256": sha256_file(dst),
                             "ref_volume": rec["volume"], "ref_bbox": rec["bbox"], "params": rec["key"]})
            data = "".join(canon(r) + "\n" for r in rows).encode("utf-8")
            (d / "items.jsonl").write_bytes(data)
            metas[iset] = {"n_items": len(rows), "items_sha256": sha256_bytes(data),
                           "families": sorted({r["family"] for r in rows}), "per_family": ITEM_SETS[iset]}

        for split in ("sft", "grpo"):
            order = list(out[split])
            family_rng(a.seed, f"order:{split}").shuffle(order)
            rows = []
            for k, rec in enumerate(order):
                row_id = f"{split}-{k:05d}"
                output = cad_protocol.completion(rec["code"]) if split == "sft" else canon(
                    {"row_id": row_id, "volume": rec["volume"], "bbox": rec["bbox"]})
                rows.append({"row_id": row_id, "family": rec["family"], "input": rec["prompt"], "output": output,
                             "text": rec["prompt"], "params": rec["key"]})
            metas[split] = write_set(HERE / "data" / split, rows)
            metas[split]["families"] = sorted({r["family"] for r in rows})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- disjointness re-check over what was written ----
    def load(p):
        return [json.loads(x) for x in open(p, encoding="utf-8")]
    sets = {"items-sft": load(HERE / "items-sft" / "items.jsonl"), "items-grpo": load(HERE / "items-grpo" / "items.jsonl"),
            "sft": load(HERE / "data" / "sft" / "train.jsonl"), "grpo": load(HERE / "data" / "grpo" / "train.jsonl")}
    geo = {k: {(r["family"], *r["params"][1:-1]) for r in v} for k, v in sets.items()}
    txt = {k: {r["text"] for r in v} for k, v in sets.items()}
    names = list(sets)
    overlap = {f"{x}&{y}": {"geometry": len(geo[x] & geo[y]), "text": len(txt[x] & txt[y])}
               for i, x in enumerate(names) for y in names[i + 1:]}
    if any(v["geometry"] or v["text"] for v in overlap.values()):
        raise SystemExit(f"overlap: {overlap}")
    if any(r["family"] in HELD_OUT for r in sets["sft"]) or any(r["family"] not in HELD_OUT for r in sets["items-grpo"]):
        raise SystemExit("family split violated")

    gen = {"script": "make_data.py", "script_sha256": sha256_file(HERE / "make_data.py"),
           "families_sha256": sha256_file(HERE / "cad_families.py"),
           "protocol_sha256": sha256_file(HERE / "executor" / "cad_protocol.py"),
           "prompt_block_sha256": cad_protocol.prompt_block_sha256(), "seed": a.seed,
           "rng": "random.Random(int(sha256('<seed>:<family>')[:8]))",
           "order": "per family: its gate items, then grpo, then sft; training files permuted once",
           "families": fams, "held_out_of_sft": list(HELD_OUT), "sft_families": sft_fams,
           "executor_versions": zyg.versions}
    meta = {"schema": "ptc_mm.cad_data", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
            "generator": gen,
            "license": "generated by this script (no third-party data); the reference programs are original",
            "counts": {k: len(v) for k, v in sets.items()}, "per_family": want, "stats": stats,
            "overlap": overlap, "sets": metas}
    fresh = {"data/data_meta.json": meta}
    for iset in ITEM_SETS:
        fresh[f"{iset}/item_set_meta.json"] = {
            "schema": "ptc_mm.cad_item_set", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
            "set": iset, **metas[iset], "generator": gen, "overlap": overlap,
            "ref_mesh": "OCCT STL from cad_sandbox (relative deflection 0.002 of the bbox diagonal, angular "
                        "0.1 rad, serial)"}
    recorded = recorded_script_sha256(fresh, committed)
    if recorded is not None:
        gen["script_sha256"] = recorded
        print(f"metadata reproduced byte for byte; kept the recorded script_sha256 {recorded}", flush=True)
    elif committed:
        print("metadata differs from the committed files; writing fresh metadata with this script's hash", flush=True)
    for rel, m in fresh.items():
        (HERE / rel).write_bytes(render(m))
    print(json.dumps({"counts": meta["counts"], "stats": stats, "overlap": overlap,
                      "hashes": {k: v.get("train_jsonl_sha256") or v.get("items_sha256") for k, v in metas.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
