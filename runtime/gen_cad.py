#!/usr/bin/env python
"""Generation half of the area-"CAD" evaluator `cad.cadquery_items@0.1`: one response per (item, evaluation seed).

The evaluator is split in two runs because the generated programs must execute in the B18 executor container,
which has no GPU and no network: this script generates on the GPU (or on CPU for the pilot) and writes text only;
`score_cad.py` then executes and scores in the executor container. No generated code is executed here.

Backends:
- `vllm` (GPU run; inside the NeMo RL image with its vLLM worker venv, vLLM 0.29.0, as Tier B serving) - the
  deployment sampler below, bf16, one `SamplingParams(seed=eval_seed)` per request. Not run in preparation (no GPU
  allowed in preparation); the smoke run exercises it.
- `hf` (CPU pilot; transformers `generate`, fp32, greedy) - used for the base-model pilot in README "Model choice".

Deployment sampler (05 section 3.3, the "SAMPLER" block, also written into every cut by the trainer wrapper):
greedy (temperature 0), one sample per (item, seed), at most 512 new tokens, the user message of
cad_protocol.user_prompt through the checkpoint's own chat template with `add_generation_prompt=True` and
`enable_thinking=False` (Qwen3 templates read it; Qwen2.5 templates ignore it), no system message (the template's
default applies, as in NeMo RL's processors).

Outputs (`--out`): `gen/<item>__s<seed>.json` `{item, seed, repeat, prompt_sha256, text, n_new_tokens,
finish_reason}`, `gen_meta.json` (model dir, file hashes of the model's deployable files, sampler, backend,
versions, timing).

Usage:
  gen_cad.py --backend hf   --model models/qwen2.5-coder-0.5b-instruct --items items-sft --out DIR --limit 60
  gen_cad.py --backend vllm --model /inputs/cut --items .../items-grpo --out /out --eval-seeds 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))   # cad_protocol, cad_sandbox (the executor mount)
import cad_protocol  # noqa: E402

# ---- SAMPLER (begin) -------------------------------------------------------------------------------------------
SAMPLER = {
    "strategy": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "samples_per_seed": 1,
    "chat_template": "checkpoint tokenizer_config chat template, add_generation_prompt=True, enable_thinking=False",
    "messages": "[{role: user, content: cad_protocol.user_prompt(description)}]",
}
# ---- SAMPLER (end) ---------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def read_items(items_dir: Path, limit: int | None, families_round_robin: bool) -> list[dict]:
    items = [json.loads(x) for x in open(items_dir / "items.jsonl", encoding="utf-8")]
    if limit and families_round_robin:
        by_fam: dict[str, list] = {}
        for it in items:
            by_fam.setdefault(it["family"], []).append(it)
        picked, k = [], 0
        while len(picked) < limit:
            for fam in sorted(by_fam):
                if k < len(by_fam[fam]) and len(picked) < limit:
                    picked.append(by_fam[fam][k])
            k += 1
        return picked
    return items[:limit] if limit else items


def chat_prompts(tokenizer, items: list[dict]) -> list[str]:
    out = []
    for it in items:
        msgs = [{"role": "user", "content": it["text"]}]
        out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False))
    return out


def gen_hf(model_dir: Path, prompts: list[str], seed: int, batch_size: int, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32,
                                                 local_files_only=True).eval()
    res = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b:b + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=SAMPLER["max_new_tokens"],
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            eos = {tok.eos_token_id, tok.pad_token_id} | set(
                (model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                 else [model.generation_config.eos_token_id]))
            n = next((i for i, t in enumerate(ids) if t in eos), None)
            stop = "stop" if n is not None else "length"
            ids = ids[:n] if n is not None else ids
            res.append((tok.decode(ids, skip_special_tokens=True), len(ids), stop))
        print(f"  generated {min(b + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    import transformers
    return res, {"transformers": transformers.__version__, "torch": torch.__version__, "dtype": "float32"}


def gen_vllm(model_dir: Path, prompts: list[str], seed: int):
    from vllm import LLM, SamplingParams
    import vllm
    llm = LLM(model=str(model_dir), dtype="bfloat16", seed=0, max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=False)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=SAMPLER["max_new_tokens"], seed=seed)
    outs = llm.generate(prompts, sp)
    res = [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]
    return res, {"vllm": vllm.__version__, "dtype": "bfloat16"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=("hf", "vllm"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True, help="items-sft (SFT gate) or items-grpo (GRPO gate)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    p.add_argument("--round-robin", action="store_true", help="with --limit: take items family by family")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args(argv)
    from transformers import AutoTokenizer
    t0 = time.time()
    out = a.out.resolve()
    if (out / "gen").exists() and any((out / "gen").iterdir()):
        raise SystemExit(f"{out}/gen exists and is not empty")
    (out / "gen").mkdir(parents=True, exist_ok=True)
    items = read_items(a.items, a.limit, a.round_robin)
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    cut = None
    if (a.model / "cut.json").exists():   # a cut: every listed file must match before it is used (MM-20)
        cut = json.loads((a.model / "cut.json").read_text(encoding="utf-8"))
        bad = [f for f, h in cut["files"].items() if sha256_file(a.model / f) != h]
        extra = sorted({f.name for f in a.model.iterdir()} - set(cut["files"]) - {"cut.json"})
        if bad or extra:
            raise SystemExit(f"cut {a.model}: hash mismatch {bad}, unlisted files {extra}")
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    prompts = chat_prompts(tok, items)
    versions = {}
    for k, seed in enumerate(seeds):
        if a.backend == "hf":
            res, versions = gen_hf(a.model, prompts, seed, a.batch_size, a.threads)
        else:
            res, versions = gen_vllm(a.model, prompts, seed)
        for it, (text, n, stop) in zip(items, res):
            doc = {"item": it["item"], "seed": seed, "repeat": k, "prompt_sha256": cad_protocol.sha256_text(it["text"]),
                   "text": text, "n_new_tokens": n, "finish_reason": stop}
            (out / "gen" / f"{it['item']}__s{seed}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                                                                   encoding="utf-8")
    deploy = [f for f in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
              if (a.model / f).exists()]
    meta = {"backend": a.backend, "model_dir": str(a.model.resolve()),
            "cut": {"global_step": cut["global_step"], "cut_json_sha256": sha256_file(a.model / "cut.json")}
            if cut else None,
            "model_files": {f: sha256_file(a.model / f) for f in deploy},
            "sampler": SAMPLER, "eval_seeds": seeds, "items": len(items),
            "items_sha256": sha256_file(a.items / "items.jsonl"), "limit": a.limit, "round_robin": a.round_robin,
            "prompt_block_sha256": cad_protocol.prompt_block_sha256(),
            "script_sha256": sha256_file(Path(__file__).resolve()), "versions": versions,
            "wall_seconds": round(time.time() - t0, 1),
            "seconds_per_response": round((time.time() - t0) / max(1, len(items) * len(seeds)), 3)}
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("items", "wall_seconds", "seconds_per_response", "versions")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
