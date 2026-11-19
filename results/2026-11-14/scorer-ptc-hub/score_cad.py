#!/usr/bin/env python
"""Scoring step of the area-"CAD" evaluator `cad.cadquery_items@0.1` (CPU; runs no generated code).

The evaluator has three steps: (1) gen_cad.py generates responses (GPU); (2a) `cad_sandbox.py --run-batch` executes
every response in the executor container, which mounts no reference data, and writes each candidate's status, OCCT
measurements and mesh; (2b) this script, in a separate container that never runs generated code, reads the
responses, the execution records and the item set (references), and scores (decided after review 11:
no process that runs generated code can see reference data). Scoring against the item's reference mesh:

- deciding metric `iou` (higher is better, [0, 1]): volumetric IoU of the candidate and reference solids by
  vertical-line integration (geom_metrics.volumetric_iou; numpy, 16,384 seeded columns, seed = first 8 bytes of
  SHA-256("iou:<item>")). A program that yields no valid solid (any status other than `ok`) scores 0 (MM-4: the
  policy's fault).
- sub-metrics: `status`, `executes` (status ok), `chamfer` (symmetric mean surface distance / reference bbox
  diagonal; null unless ok), `volume_ratio` and `bbox_iou` (the training reward's OCCT terms, reported only),
  `reward` (cad_protocol.reward, reported only), `n_new_tokens`, `finish_reason`, `unbalanced_columns`.
- supporting arm `valid` (binary, [0, 1]): 1 when the program executes and yields a valid solid; the same OCCT
  validity check as the training reward, so tier T0 (supporting only).

Errors (evaluator faults, MM-4): `executor_error` (the sandbox itself failed: fork, zygote, unreadable result),
`scorer_error` (a reference mesh missing or with a wrong SHA-256, an unreadable candidate mesh, a reference column
integration that does not close), `missing_output` (no gen file for a registered (item, seed)).

Outputs (`--out`): `outputs/<item>__r<k>.json` (response text, extracted program, status, metrics), `scores.jsonl`,
`summary.json` (per-arm statistics, MM-38 checks, timing), `env-runtime.json`, a byte copy of the generation step's
`gen_meta.json` (model file hashes, sampler, backend versions), `arm-iou/` (evaluator `cad.cadquery_items@0.1`) and
`arm-valid/` (evaluator `cad.cadquery_valid_items@0.1`), each with MM-1 `items.jsonl` (sub_metrics: finite numbers
only; status as `status_code`, its index in cad_sandbox.STATUSES; strings stay in `scores.jsonl`), MM-2
`eval_meta.json` (`framework` = `{name, version}` of the generation backend) and hard-linked `outputs/`.

Usage (scoring container or host CPU):
  score_cad.py --gen DIR --exec DIR --items DIR --out DIR [--eval-seeds 11] [--model-hash H] [--item-set-hash H]
"""

from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import hashlib
import json
import math
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))   # cad_protocol, cad_sandbox (the executor mount)
import cad_protocol  # noqa: E402
from cad_sandbox import STATUSES  # noqa: E402  (constants only; no CadQuery import, no execution)

EVALUATOR_ID = "cad.cadquery_items@0.1"               # arm-iou (deciding)
VALID_EVALUATOR_ID = "cad.cadquery_valid_items@0.1"   # arm-valid (supporting, T0)
ADAPTER = "nemo_rl_ext"
SCORER_VERSION = "cad_iou_v1"


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def seed_of(tag: str) -> int:
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def stats(vals) -> dict:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    counts: dict[float, int] = {}
    for v in vals:
        counts[round(v, 9)] = counts.get(round(v, 9), 0) + 1
    mode, mode_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals), "distinct": len(counts),
            "mode": mode, "mode_share": mode_n / n, "share_zero": sum(v == 0 for v in vals) / n,
            "share_one": sum(v >= 1 for v in vals) / n}


# MM-1: an arm line's sub_metrics hold finite numbers only. Strings (status name, failure detail, finish reason)
# stay in scores.jsonl and the output file; a value that does not exist for a response (chamfer of a program that
# made no solid) is left out, never written as null. The status is also given as its index in
# cad_sandbox.STATUSES (`status_code`, 0 = ok) and the finish reason as `truncated` (1 when the token bound cut it).
ARM_NUMERIC = ("executes", "chamfer", "volume_ratio", "bbox_iou", "reward", "n_new_tokens", "unbalanced_columns",
               "mc_volume_rel_err")


def arm_sub_metrics(sub: dict | None) -> dict | None:
    if sub is None:
        return None
    out = {}
    for k in ARM_NUMERIC:
        v = sub.get(k)
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = v
    out["status_code"] = STATUSES.index(sub["status"])
    out["truncated"] = int(sub.get("finish_reason") == "length")
    return out


def framework_of(gen_meta: dict) -> dict:
    """MM-2 `framework`: {name, version} of the generation backend that produced the responses."""
    v = gen_meta.get("versions") or {}
    if gen_meta.get("backend") == "vllm":
        return {"name": "vllm", "version": v.get("vllm")}
    if gen_meta.get("backend") == "hf":
        return {"name": "transformers", "version": v.get("transformers")}
    return {"name": str(gen_meta.get("backend")), "version": None}


def mm38(st: dict, lo: float, hi: float, binary: bool) -> dict:
    if not st.get("n"):
        return {"pass": False, "reason": "no scores"}
    mean, sd = st["mean"], st.get("sd") or 0.0
    if binary:
        ok = 0 < mean < 1
        return {"pass": ok, "binary": True, "mean_strictly_inside": ok}
    margin = 0.05 * (hi - lo)
    checks = {"mode_share_lt_0_5": st["mode_share"] < 0.5, "sd_gt_0": sd > 0,
              "mean_inside_by_5pct": lo + margin <= mean <= hi - margin}
    return {"pass": all(checks.values()), **checks}


def score_one(task: tuple) -> dict:
    """(candidate mesh path, reference mesh path, item id) -> metrics or {"error": message}. Runs in a pool worker."""
    import geom_metrics as geom
    cand_mesh, ref_mesh, item = task
    try:
        cand = geom.read_stl(str(cand_mesh))
        ref_tri = geom.read_stl(str(ref_mesh))
        r = geom.volumetric_iou(cand, ref_tri, seed=seed_of(f"iou:{item}"))
        ch = geom.chamfer(cand, ref_tri, seed=seed_of(f"chamfer:{item}"))
        return {**r, "chamfer": ch}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gen", type=Path, required=True, help="directory with gen/ and gen_meta.json (gen_cad.py)")
    p.add_argument("--exec", dest="exec_dir", type=Path, required=True,
                   help="output of `cad_sandbox.py --run-batch` on the same --gen")
    p.add_argument("--items", type=Path, required=True, help="items-sft (SFT gate) or items-grpo (GRPO gate)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    a = p.parse_args(argv)
    t0 = time.time()
    import geom_metrics as geom

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    gen_meta = json.loads((a.gen / "gen_meta.json").read_text(encoding="utf-8"))
    items = [json.loads(x) for x in open(a.items / "items.jsonl", encoding="utf-8")]
    gen_files = {f.name for f in (a.gen / "gen").iterdir()}
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    if gen_meta["eval_seeds"] != seeds:
        raise SystemExit(f"gen_meta eval_seeds {gen_meta['eval_seeds']} != --eval-seeds {seeds}")
    if a.limit:
        items = [it for it in items if f"{it['item']}__s{seeds[0]}.json" in gen_files][:a.limit]

    # reference meshes: hash check first (a wrong reference is a scorer error for that item)
    ref_ok = {}
    for it in items:
        mp = a.items / it["ref_mesh"]
        ref_ok[it["item"]] = mp.exists() and sha256_file(mp) == it["ref_mesh_sha256"]

    jobs = []
    for it in items:
        for k, s in enumerate(seeds):
            gp = a.gen / "gen" / f"{it['item']}__s{s}.json"
            doc = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
            if doc is not None and doc["prompt_sha256"] != cad_protocol.sha256_text(it["text"]):
                raise SystemExit(f"{gp}: prompt hash differs from the item's text")
            jobs.append((it, k, s, doc))
    programs = [{"id": f"{it['item']}__r{k}", "code": cad_protocol.extract_program(doc["text"]) if doc else None,
                 "mesh": True} for it, k, s, doc in jobs]
    # execution records of step 2a: every file must be listed in exec_meta.json with its hash, nothing else
    ex = a.exec_dir
    results, exec_error, exec_meta = [None] * len(programs), None, None
    try:
        exec_meta = json.loads((ex / "exec_meta.json").read_text(encoding="utf-8"))
        have = {str(f.relative_to(ex)) for f in ex.rglob("*") if f.is_file()} - {"exec_meta.json", "exec_timing.jsonl"}
        if have != set(exec_meta["files"]):
            raise ValueError(f"files differ from exec_meta.json: {sorted(have ^ set(exec_meta['files']))[:5]}")
        bad = [f for f, h in exec_meta["files"].items() if sha256_file(ex / f) != h]
        if bad:
            raise ValueError(f"hash mismatch {bad[:5]}")
        if exec_meta["gen_meta_sha256"] != sha256_file(a.gen / "gen_meta.json"):
            raise ValueError("exec_meta.json names another gen_meta.json")
        for idx, ((it, k, s, doc), prog) in enumerate(zip(jobs, programs)):
            rp = ex / "programs" / f"{it['item']}__s{s}" / "result.json"
            if doc is None or not rp.exists():
                continue
            r = json.loads(rp.read_text(encoding="utf-8"))
            want = cad_protocol.sha256_text(prog["code"]) if prog["code"] is not None else None
            if r.get("program_sha256") != want:
                raise ValueError(f"{rp}: program hash differs from the response's program")
            results[idx] = r
    except (OSError, ValueError, KeyError) as e:
        results, exec_error = [None] * len(programs), f"{type(e).__name__}: {e}"
    timing = ([json.loads(x) for x in open(ex / "exec_timing.jsonl")] if (ex / "exec_timing.jsonl").exists() else [])
    exec_seconds = sum(t.get("wall_s") or 0 for t in timing)
    if exec_error is None:
        exec_error = "no execution record for this response"
    exec_versions = (exec_meta or {}).get("versions", {})
    exec_limits = (exec_meta or {}).get("limits", {})

    # geometry scoring in a fork pool (after all programs ran: the zygote no longer needs to be single-threaded)
    t_score = time.time()
    tasks = {}
    for idx, ((it, k, s, doc), res) in enumerate(zip(jobs, results)):
        if doc is not None and res is not None and res["status"] == "ok" and ref_ok[it["item"]]:
            tasks[idx] = (ex / "programs" / f"{it['item']}__s{s}" / "mesh.stl", a.items / it["ref_mesh"], it["item"])
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(a.workers) as pool:
        metrics = dict(zip(tasks, pool.map(score_one, list(tasks.values()), chunksize=1)))
    rows = []
    for idx, ((it, k, s, doc), prog, res) in enumerate(zip(jobs, programs, results)):
        row = {"item": it["item"], "family": it["family"], "repeat": k, "seed": s, "error": None}
        if doc is None:
            row["error"] = {"class": "missing_output", "message": "no generation for this (item, seed)"}
        elif res is None:
            row["error"] = {"class": "executor_error", "message": exec_error[:300]}
        else:
            ref = {"volume": it["ref_volume"], "bbox": it["ref_bbox"]}
            st = res["status"]
            sub = {"status": st, "executes": int(st == "ok"), "chamfer": None, "volume_ratio": None,
                   "bbox_iou": None, "reward": cad_protocol.reward(res, ref), "n_new_tokens": doc["n_new_tokens"],
                   "finish_reason": doc["finish_reason"], "unbalanced_columns": None}
            score = 0.0
            if st == "ok":
                v = res["volume"]
                sub["volume_ratio"] = min(v, it["ref_volume"]) / max(v, it["ref_volume"])
                sub["bbox_iou"] = cad_protocol.bbox_iou(res["bbox"], it["ref_bbox"])
                if not ref_ok[it["item"]]:
                    row["error"] = {"class": "scorer_error", "message": "reference mesh missing or hash mismatch"}
                else:
                    m = metrics[idx]
                    if "error" in m:
                        row["error"] = {"class": "scorer_error", "message": m["error"]}
                    else:
                        # unbalanced columns (a surface not closed along some lines) are reported, still scored
                        score = m["iou"]
                        sub.update({"chamfer": m["chamfer"], "unbalanced_columns": m["unbalanced_columns"],
                                    "mc_volume_rel_err": (m["cand_volume_mc"] - v) / v if v else None})
            elif res.get("detail"):
                sub["detail"] = res["detail"][:200]
            row["score"] = None if row["error"] else score
            row["sub_metrics"] = sub
        rel = f"outputs/{it['item']}__r{k}.json"
        write_json(out / rel, {"item": it["item"], "repeat": k, "seed": s,
                               "response": doc["text"] if doc else None, "program": prog["code"],
                               "status": (res or {}).get("status"), "score": row.get("score"),
                               "sub_metrics": row.get("sub_metrics")})
        row["output_ref"] = {"path": rel, "sha256": sha256_file(out / rel)}
        rows.append(row)
    score_seconds = time.time() - t_score
    write_jsonl(out / "scores.jsonl", rows)

    code_sha = {"score_cad.py": sha256_file(Path(__file__).resolve()),
                "geom_metrics.py": sha256_file(HERE / "geom_metrics.py"),
                "cad_sandbox.py": sha256_file(HERE / "executor" / "cad_sandbox.py"),
                "cad_protocol.py": sha256_file(HERE / "executor" / "cad_protocol.py")}
    if exec_meta is not None and exec_meta.get("code_sha256") != {k: code_sha[k] for k in ("cad_sandbox.py",
                                                                                            "cad_protocol.py")}:
        raise SystemExit("the execution step ran other cad_sandbox.py / cad_protocol.py than this scorer pins")
    scorer_code = sha256_bytes(canonical(code_sha))
    sampler = {**gen_meta["sampler"], "backend": gen_meta["backend"],
               "prompt_block_sha256": gen_meta["prompt_block_sha256"]}
    sampler_hash = sha256_bytes(canonical(sampler))
    framework = framework_of(gen_meta)
    arms = {"iou": {"evaluator_id": EVALUATOR_ID, "direction": "higher_is_better", "range": [0, 1], "binary": False,
                    "scorer": {"name": "volumetric_iou_vertical_lines",
                               "version": f"{SCORER_VERSION}; numpy {__import__('numpy').__version__}; "
                                          f"{geom.IOU_COLUMNS} columns; executor cadquery "
                                          f"{exec_versions.get('cadquery')} ocp {exec_versions.get('ocp')}; limits "
                                          f"{json.dumps(exec_limits, sort_keys=True)}",
                               "weights_sha256": None, "code_sha256": scorer_code}},
            "valid": {"evaluator_id": VALID_EVALUATOR_ID, "direction": "higher_is_better", "range": [0, 1], "binary": True,
                      "scorer": {"name": "occt_valid_solid",
                                 "version": f"{SCORER_VERSION}; cadquery {exec_versions.get('cadquery')} isValid",
                                 "weights_sha256": None, "code_sha256": scorer_code}}}
    for metric, spec in arms.items():
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for r in rows:
            link_or_copy(out / r["output_ref"]["path"], arm / r["output_ref"]["path"])
            if r["error"]:
                sc = None
            elif metric == "iou":
                sc = r["score"]
            else:
                sc = float(r["sub_metrics"]["executes"])
            lines.append({"item": r["item"], "repeat": r["repeat"], "seed": r["seed"], "score": sc,
                          "error": r["error"], "sub_metrics": arm_sub_metrics(r.get("sub_metrics")),
                          "output_ref": r["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        write_json(arm / "eval_meta.json", {
            "schema": "ptc_hub.item_eval", "schema_version": "1.0", "evaluator_id": spec["evaluator_id"],
            "adapter": ADAPTER, "metric": metric, "direction": spec["direction"], "score_range": spec["range"],
            "checkpoint": {"model_hash": a.model_hash, "base_model_hash": a.base_model_hash},
            "normalizer_sha256": None, "env_version_sha256": None, "item_set_hash": a.item_set_hash,
            "eval_seeds": seeds, "sampler": sampler, "sampler_hash": sampler_hash, "scorer": spec["scorer"],
            "framework": framework,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    ok = [r for r in rows if not r["error"]]
    iou_st = stats([r["score"] for r in ok])
    valid_st = stats([float(r["sub_metrics"]["executes"]) for r in ok])
    status_counts: dict[str, int] = {}
    for r in ok:
        status_counts[r["sub_metrics"]["status"]] = status_counts.get(r["sub_metrics"]["status"], 0) + 1
    fam: dict[str, list] = {}
    for r in ok:
        fam.setdefault(r["family"], []).append(r["score"])
    exec_ok = [r for r in ok if r["sub_metrics"]["executes"]]
    summary = {"items": len(items), "repeats": len(seeds), "responses": len(rows),
               "errors": len(rows) - len(ok),
               "error_classes": sorted({r["error"]["class"] for r in rows if r["error"]}),
               "iou": iou_st, "iou_given_executes": stats([r["score"] for r in exec_ok]),
               "valid": valid_st, "statuses": status_counts,
               "reward_mean": (sum(r["sub_metrics"]["reward"] for r in ok) / len(ok)) if ok else None,
               "chamfer_given_executes": stats([r["sub_metrics"]["chamfer"] for r in exec_ok]),
               "per_family_iou_mean": {f: sum(v) / len(v) for f, v in sorted(fam.items())},
               "degeneracy_mm38": {"iou": mm38(iou_st, 0.0, 1.0, False), "valid": mm38(valid_st, 0.0, 1.0, True)},
               "exec_seconds": round(exec_seconds, 2), "score_seconds": round(score_seconds, 2),
               "wall_seconds": round(time.time() - t0, 1),
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "sampler_hash": sampler_hash, "gen_backend": gen_meta["backend"], "model_dir": gen_meta["model_dir"],
               "missing_hashes": [k for k, v in (("model_hash", a.model_hash), ("item_set_hash", a.item_set_hash))
                                  if v is None]}
    lock = HERE / "executor" / "uv.lock"
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable, "uv_lock_sha256": sha256_file(lock),
        "code_sha256": code_sha, "scorer_code_sha256": scorer_code, "executor": exec_versions,
        "limits": exec_limits, "workers": a.workers,
        "exec_meta_sha256": sha256_file(ex / "exec_meta.json") if (ex / "exec_meta.json").exists() else None,
        "items_sha256": sha256_file(a.items / "items.jsonl"), "gen_meta_sha256": sha256_file(a.gen / "gen_meta.json"),
        "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    # The generation and execution steps' records, byte-for-byte (model_files, sampler, backend versions; executor
    # limits, ids and output hashes), for ext-collect.
    shutil.copyfile(a.gen / "gen_meta.json", out / "gen_meta.json")
    shutil.copyfile(ex / "exec_meta.json", out / "exec_meta.json")
    print(json.dumps({k: summary[k] for k in ("items", "errors", "iou", "valid", "statuses", "degeneracy_mm38",
                                              "exec_seconds", "score_seconds")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
