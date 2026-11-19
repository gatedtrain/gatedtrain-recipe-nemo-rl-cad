#!/usr/bin/env python
"""CPU pilot of area "CAD": a few supervised steps on the SFT shard with plain transformers + torch (CPU only).

NOT the trainer. NeMo RL needs its GPU image and cannot run on CPU (README "Trainer"). This proxy
exercises everything around the trainer that can run on CPU, with the trainer-side conventions of run_cad_nemo.py
and ptc_cad_hooks.py:
- the same rows in the same order (data/sft/train.jsonl, file order, as NeMo RL with data.shuffle=false), the same
  chat template call as NeMo RL's sft_processor (user + assistant message; loss on the assistant tokens only);
- an observed sample log written before each optimizer step (`samples.jsonl`, `train_steps.jsonl`, fsynced);
- atomic cuts `checkpoint-<S>/` built in `.checkpoint-<S>.partial/`, files fsynced, `cut.json` written last
  (`global_step`, `examples_consumed`, per-file SHA-256, `deployable`, seed), then renamed;
- the seed recorded (`torch.manual_seed`, no dropout in Qwen2.5 / Qwen3 configs; the order is the file order).
It also gives the power pilot its first paired differences (README "Items and power").

Usage: <area4 venv>/bin/python sft_cpu_pilot.py --model models/<name> --out DIR [--steps 30 --batch 8 --lr 1e-5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def append(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_cut(model, tok, out: Path, step: int, consumed: int, seed: int, base: Path) -> Path:
    final, partial = out / f"checkpoint-{step}", out / f".checkpoint-{step}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    model.save_pretrained(partial, safe_serialization=True)
    tok.save_pretrained(partial)
    shutil.copyfile(base / "generation_config.json", partial / "generation_config.json")
    files = {}
    for f in sorted(partial.iterdir()):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
        files[f.name] = sha256_file(f)
    cut = {"schema": "ptc_mm.cad_cut", "schema_version": "1.0", "global_step": step, "examples_consumed": consumed,
           "files": files, "deployable": sorted(files), "seed": seed, "profile": "cpu-pilot", "stage": "sft",
           "script_sha256": sha256_file(Path(__file__).resolve())}
    with open(partial / "cut.json", "w") as fh:
        fh.write(json.dumps(cut, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(partial, final)
    return final


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=6000)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--threads", type=int, default=24)
    a = p.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), dtype=torch.float32, local_files_only=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    rows = [json.loads(x) for x in open(HERE / "data" / "sft" / "train.jsonl", encoding="utf-8")]
    (out / "run_args.json").write_text(json.dumps({
        "stage": "sft", "profile": "cpu-pilot", "seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr,
        "model_name_or_path": str(a.model), "train_file": str(HERE / "data" / "sft" / "train.jsonl"),
        "train_file_sha256": sha256_file(HERE / "data" / "sft" / "train.jsonl"),
        "script_sha256": sha256_file(Path(__file__).resolve()), "torch": torch.__version__}, indent=2) + "\n")
    t0 = time.time()
    log = []
    for step in range(1, a.steps + 1):
        batch = rows[(step - 1) * a.batch: step * a.batch]
        ids, labels = [], []
        for r in batch:
            prompt = tok.apply_chat_template([{"role": "user", "content": r["input"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            full = tok.apply_chat_template([{"role": "user", "content": r["input"]},
                                            {"role": "assistant", "content": r["output"]}], tokenize=False,
                                           enable_thinking=False)
            if not full.startswith(prompt):
                raise SystemExit("chat template: the prompt is not a prefix of the full conversation")
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            ids.append(f_ids)
            labels.append([-100] * len(p_ids) + f_ids[len(p_ids):])
        L = max(map(len, ids))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        x = torch.tensor([i + [pad] * (L - len(i)) for i in ids])
        y = torch.tensor([l + [-100] * (L - len(l)) for l in labels])
        att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids])
        loss = model(input_ids=x, attention_mask=att, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        append(out / "samples.jsonl", [{"step": step, "epoch": 0, "position": k, "row_id": r["row_id"],
                                        "row": "sha256:" + hashlib.sha256(r["row_id"].encode()).hexdigest(),
                                        "attempt": 1, "utc": utc} for k, r in enumerate(batch)])
        append(out / "train_steps.jsonl", [{"step": step, "epoch": 0, "n_examples": len(batch),
                                            "example_ids": [r["row_id"] for r in batch]}])
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
        append(out / "train_log.jsonl", [log[-1]])
        print(json.dumps(log[-1]), flush=True)
        if (a.save_every and step % a.save_every == 0) or step == a.steps:
            model.eval()
            write_cut(model, tok, out, step, step * a.batch, a.seed, a.model)
            model.train()
    print(json.dumps({"steps": a.steps, "seconds": round(time.time() - t0, 1),
                      "s_per_step": round((time.time() - t0) / a.steps, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
