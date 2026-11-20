"""Launcher-side hooks for NeMo RL runs of area "CAD" (MM-27: wrappers around framework objects at run time; no
NeMo RL source is edited). Imported by run_sft_cad.py and run_grpo_cad.py in the NeMo RL driver process (the
image's system Python, NeMo RL at 4f39fd86289a2ffd5aaf5d202702f7fd7185c924).

The hook points need the GPU image: they were read from the pinned clone (file:line in README "Trainer") and the smoke
run verifies them. The wrappers themselves run on CPU in tests/test_runtime.py against stand-in NeMo RL modules. Every
hook fails loudly (raises) rather than writing a partial record.

1. Observed SFT consumption (`samples.jsonl`, `train_steps.jsonl`): `nemo_rl.algorithms.sft.prepare_sft_batch` is
   wrapped. sft_train calls it once per training step, before `policy.train` (sft.py:736-757); validation also
   calls it (sft.py:579), so the launcher refuses any validation setting. The batch carries the dataset `idx`
   (collate_fn.py:228), which is the 0-based line of train.jsonl (ResponseDataset keeps file order, shuffle=false);
   each line is checked against the batch's own text (the row's `input` must occur in the logged user message).
2. Atomic cuts with `cut.json` last: `CheckpointManager` storage `publish` (tmp_step_N -> step_N rename,
   checkpoint_storage.py:80-97) is wrapped; after the rename the cut `<out>/cuts/checkpoint-<N>/` is built in
   `<out>/cuts/.checkpoint-<N>.partial/` from hard links to the consolidated HF weights and tokenizer, plus
   `generation_config.json` (copied from the base), `sampler.json` (the evaluator's deployment sampler),
   `base_ref.json`, `training_info.json`; every file fsynced; `cut.json` written last; the directory renamed.
3. GRPO consumption comes from the environment's own log (cad_env.py, `env_calls.jsonl`) joined with NeMo RL's
   per-step `train_data_step<N>.jsonl` (grpo_sync.py:1459); see README "Consumption".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _append_fsync(path: Path, lines: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------------------- SFT sample log ---
def install_sft_sample_log(out: Path, train_file: Path, batch_size: int, attempt: int) -> None:
    import nemo_rl.algorithms.sft as sft
    from cad_env import text_of

    rows = [json.loads(x) for x in open(train_file, encoding="utf-8")]
    if len(rows) < batch_size:
        raise RuntimeError(f"{train_file}: {len(rows)} rows, fewer than the batch size {batch_size} (drop_last=True "
                           "would leave no training step)")
    steps_per_epoch = len(rows) // batch_size          # drop_last=True (sft.py:353)
    orig = sft.prepare_sft_batch
    state = {"step": 0}

    def wrapped(batch, **kw):
        state["step"] += 1
        step = state["step"]
        idx = [int(i) for i in batch["idx"]]
        lines = []
        for pos, i in enumerate(idx):
            row = rows[i]
            content = "".join(text_of(m.get("content")) for m in batch["message_log"][pos] if m.get("role") == "user")
            if row["input"] not in content:
                raise RuntimeError(f"SFT step {step}: idx {i} ({row['row_id']}) does not match the batch text")
            lines.append({"step": step, "epoch": (step - 1) // steps_per_epoch, "position": pos, "idx": i,
                          "row_id": row["row_id"], "row": "sha256:" + hashlib.sha256(row["row_id"].encode()).hexdigest(),
                          "attempt": attempt, "utc": _utc()})
        _append_fsync(out / "samples.jsonl", lines)
        _append_fsync(out / "train_steps.jsonl", [{"step": step, "epoch": lines[0]["epoch"], "n_examples": len(idx),
                                                   "example_ids": [x["row_id"] for x in lines]}])
        return orig(batch, **kw)

    sft.prepare_sft_batch = wrapped


# ------------------------------------------------------------------------------------------------ cut writer ---
CUT_DEPLOYABLE = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "sampler.json")


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_cut(step_dir: Path, step: int, out: Path, base_dir: Path, run_meta: dict) -> Path:
    cuts = out / "cuts"
    cuts.mkdir(exist_ok=True)
    final = cuts / f"checkpoint-{step}"
    partial = cuts / f".checkpoint-{step}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    cons = step_dir / "policy" / "weights" / "model" / "consolidated"
    shards = sorted(cons.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"{cons}: no consolidated safetensors file")
    if len(shards) > 1:
        raise RuntimeError(f"{cons}: several consolidated weight files {[s.name for s in shards]}: this recipe's adapter "
                           "and generation read one model.safetensors; adapt rows.py components, the adapter's cut "
                           "checks and gen_cad.py deploy list before training a sharded model")
    os.link(shards[0], partial / "model.safetensors")
    os.link(cons / "config.json", partial / "config.json")
    tok = step_dir / "policy" / "tokenizer"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if (tok / name).exists():
            # Copied, not linked: NeMo RL keeps the base download's mtimes on these, and ext-collect refuses
            # outputs older than the launch's START (EXT-STAGE; first SFT full run).
            shutil.copyfile(tok / name, partial / name)
    shutil.copyfile(base_dir / "generation_config.json", partial / "generation_config.json")
    sys.path.insert(0, str(HERE))
    import gen_cad  # noqa: E402  (SAMPLER only; importing it loads no model)
    (partial / "sampler.json").write_text(json.dumps(gen_cad.SAMPLER, indent=2, sort_keys=True) + "\n")
    base_models = json.loads((base_dir.parent / "MODELS.json").read_text())["models"]
    base = base_models[base_dir.name]
    (partial / "base_ref.json").write_text(json.dumps(
        {"repo": base["repo"], "revision": base["revision"], "files": base["files"],
         "init_checkpoint": run_meta.get("init_checkpoint")}, indent=2, sort_keys=True) + "\n")
    if (step_dir / "training_info.json").exists():
        shutil.copyfile(step_dir / "training_info.json", partial / "training_info.json")
    files = {}
    for f in sorted(partial.iterdir()):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
        files[f.name] = sha256_file(f)
    missing = [n for n in CUT_DEPLOYABLE if n not in files]
    if missing:
        raise RuntimeError(f"cut {step}: missing {missing}")
    deployable = [n for n in files if n in CUT_DEPLOYABLE or n in ("chat_template.jinja", "vocab.json", "merges.txt")]
    cut = {"schema": "ptc_mm.cad_cut", "schema_version": "1.0", "global_step": step, "files": files,
           "deployable": sorted(deployable), "seed": run_meta["seed"], "profile": run_meta["profile"],
           "stage": run_meta["stage"], "script_sha256": run_meta["script_sha256"],
           "examples_consumed": run_meta.get("examples_consumed", lambda s: None)(step)
           if callable(run_meta.get("examples_consumed")) else None,
           "nemo_rl_step_dir": step_dir.name}
    with open(partial / "cut.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(cut, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(partial)
    os.rename(partial, final)
    _fsync_dir(cuts)
    _append_fsync(out / "cuts.jsonl", [{"step": step, "cut": final.name, "utc": _utc()}])
    return final


def install_cut_writer(out: Path, base_dir: Path, run_meta: dict) -> None:
    from nemo_rl.utils import checkpoint as ckpt

    orig_init = ckpt.CheckpointManager.__init__

    def init(self, config):
        orig_init(self, config)
        storage = self._storage
        orig_publish = storage.publish

        def publish(path):
            orig_publish(path)
            tmp = Path(path)
            step = int(tmp.name.split("_")[2])
            write_cut(tmp.parent / f"step_{step}", step, out, base_dir, run_meta)

        storage.publish = publish

    ckpt.CheckpointManager.__init__ = init


# ------------------------------------------------------------------------------------------------ run record ---
def write_run_args(out: Path, doc: dict) -> None:
    import importlib.metadata as md
    pkgs = {}
    for name in ("nemo-rl", "nemo_rl", "torch", "transformers", "vllm", "ray"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    commit = None
    for cand in (Path("/opt/nemo-rl/.git"),):
        head = cand / "HEAD"
        if head.exists():
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                rp = cand / ref.split()[1]
                commit = rp.read_text().strip() if rp.exists() else None
            else:
                commit = ref
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    (out / "env-freeze.txt").write_text("\n".join(freeze) + f"\n# nemo-rl commit {commit}\n")
    doc = {**doc, "framework": {"name": "nemo_rl", "commit": commit, "versions": pkgs},
           "python": sys.version.split()[0], "created_at": _utc()}
    (out / "run_args.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")


def record_attempt(out: Path, event: str, **kw) -> int:
    path = out / "attempts.jsonl"
    prev = [json.loads(x) for x in open(path)] if path.exists() else []
    attempt = 1 + sum(1 for p in prev if p["event"] == "start") if event == "start" else \
        sum(1 for p in prev if p["event"] == "start")
    _append_fsync(path, [{"event": event, "attempt": attempt, "utc": _utc(), "pid": os.getpid(), **kw}])
    return attempt


def wait_until(pred, timeout: float, what: str) -> None:
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise RuntimeError(f"timed out waiting for {what}")
        time.sleep(1)
