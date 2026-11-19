#!/usr/bin/env python
"""NeMo RL launcher of area "CAD": stage `sft` (`sft.nemo_rl@0.1`) or stage `grpo` (`rl.nemo_rl_env@0.1`).

Runs inside the Tier B NeMo RL image (`nemo-rl:4f39fd8`, image ID sha256:8cd5b714...a2113a, NeMo RL commit
4f39fd86289a2ffd5aaf5d202702f7fd7185c924), cwd /opt/nemo-rl, the image's own interpreter (`uv run python`, as Tier
B). It records the run, installs the hooks of ptc_cad_hooks.py, builds the NeMo RL overrides of the profile and calls
the unmodified example entry point (`examples/run_sft.py` or `examples/run_grpo.py`, `main()`), so NeMo RL itself
parses the overrides exactly as on its command line.

UNTESTED IN PREPARATION (GPU image). The overrides follow Tier B's long run (gatedtrain's docs/tier-b-runbook.md step 7) plus the
area's settings; README "Trainer" lists each with its source line.

Usage (container):
  run_cad_nemo.py --stage sft  --profile smoke|full --out /out --seed 6000 --base <dir>
  run_cad_nemo.py --stage grpo --profile smoke|full --out /out --seed 6100 --base <dir> --init /inputs/sft-cut
                  --socket /spool/cadexec.sock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))   # cad_protocol, cad_sandbox (the executor mount)
import ptc_cad_hooks as hooks  # noqa: E402

DATA = {"sft": HERE / "data" / "sft" / "train.jsonl", "grpo": HERE / "data" / "grpo" / "train.jsonl"}

# Steps, batch and cut cadence per stage and profile (README "Profiles and cost").
PROFILES = {
    ("sft", "smoke"): {"steps": 10, "batch": 32, "save_period": 5},
    ("sft", "full"): {"steps": 60, "batch": 32, "save_period": 20},
    ("grpo", "smoke"): {"steps": 5, "prompts": 16, "generations": 8, "save_period": 5},
    ("grpo", "full"): {"steps": 60, "prompts": 16, "generations": 8, "save_period": 20},
}
MAX_SEQ = 1280          # prompt v2 (570-657 tokens with the chat template, measured) + at most 512 new tokens


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def overrides(stage: str, prof: dict, a, out: Path) -> list[str]:
    model = str(a.init if a.init else a.base)
    common = [
        f"policy.model_name={model}",
        f"policy.tokenizer.name={model}",
        "++policy.tokenizer.chat_template=default",             # sft.yaml sets a Llama template; keep the model's
                                                                # (grpo_math_1B.yaml has no such key, hence ++)
        "policy.tokenizer.chat_template_kwargs={enable_thinking:false}",
        f"policy.max_total_sequence_length={MAX_SEQ}",
        "data.shuffle=false",                                   # file order: MM-22 / observed logs
        "data.train.dataset_name=ResponseDataset",
        f"++data.train.data_path={DATA[stage]}",
        "++data.train.input_key=input",
        "++data.train.output_key=output",
        "++data.train.split_validation_size=0",                 # sft.yaml has no such key, hence ++
        "data.validation=null",
        "data.default.prompt_file=null",
        "data.default.system_prompt_file=null",
        "checkpointing.enabled=true",
        "checkpointing.keep_top_k=null",
        f"checkpointing.checkpoint_dir={out}/nemo/checkpoints",
        f"checkpointing.save_period={prof['save_period']}",
        f"logger.log_dir={out}/nemo/logs",
        "logger.wandb_enabled=false",
        "logger.tensorboard_enabled=false",
        "policy.automodel_cfg.checkpoint.save_consolidated=every",
        "policy.automodel_cfg.activation_checkpointing=true",    # as in Tier B's long run
    ]
    if stage == "sft":
        return common + [
            f"sft.max_num_steps={prof['steps']}", "sft.max_num_epochs=1", f"sft.seed={a.seed}",
            "sft.val_period=0", "sft.val_at_start=false", "sft.val_at_end=false",
            f"policy.train_global_batch_size={prof['batch']}", "policy.train_micro_batch_size=4",   # 8 hit the 24 GB ceiling (SFT smoke 2: allocator retries)
            "policy.optimizer.kwargs.lr=1.0e-5",
            "data.default.processor=sft_processor",
        ]
    n = prof["prompts"] * prof["generations"]
    return common + [
        f"grpo.max_num_steps={prof['steps']}", f"grpo.seed={a.seed}",
        f"grpo.num_prompts_per_step={prof['prompts']}", f"grpo.num_generations_per_prompt={prof['generations']}",
        "grpo.val_period=-1", "grpo.val_at_start=false", "grpo.val_at_end=false",
        f"policy.train_global_batch_size={n}", "policy.train_micro_batch_size=4",   # 8 hit the 24 GB ceiling (SFT smoke 2: allocator retries)
        "policy.generation.max_new_tokens=512",
        "policy.generation.vllm_cfg.gpu_memory_utilization=0.25",   # 0.4 peaked at 23.6 GB (GRPO smoke 2)
        # Packed microbatches of 5,120 tokens peaked at 23.6 GB in the training phase with either vLLM share
        # (GRPO smokes 2 and 3; logits of 5,120 x 151,936 alone are 3.1 GB in fp32): half the tokens.
        "policy.sequence_packing.train_mb_tokens=2560",
        "policy.sequence_packing.logprob_mb_tokens=2560",
        "data_plane.enabled=true",                               # as Tier B (grpo_sync path)
        "data.default.processor=math_hf_data_processor",         # ground_truth = the row's `output` JSON
        "data.default.env_name=cad",
        "++env.cad.num_workers=1",
        f"++env.cad.socket={a.socket}",
        f"++env.cad.log_path={out}/env_calls.jsonl",
        "++env.cad.timeout_s=900",
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("sft", "grpo"), required=True)
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--base", type=Path, required=True, help="base model directory (models/<name>)")
    p.add_argument("--init", type=Path, help="grpo: the SFT cut to start from (read-only mount)")
    p.add_argument("--socket", help="grpo: executor Unix socket")
    p.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    if a.stage == "grpo" and not (a.init and a.socket):
        p.error("grpo needs --init (the SFT cut) and --socket")
    prof = dict(PROFILES[(a.stage, a.profile)])
    if a.max_steps:
        prof["steps"] = a.max_steps
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt = hooks.record_attempt(out, "start", stage=a.stage, profile=a.profile)
    ov = overrides(a.stage, prof, a, out)
    init_ref = None
    if a.init:
        cut = json.loads((a.init / "cut.json").read_text())
        bad = [f for f, h in cut["files"].items() if sha256_file(a.init / f) != h]
        if bad:
            raise SystemExit(f"init cut {a.init}: hash mismatch {bad}")
        init_ref = {"path": str(a.init), "global_step": cut["global_step"], "cut_json_sha256": sha256_file(a.init / "cut.json")}
    per_step = prof["batch"] if a.stage == "sft" else prof["prompts"]
    run_meta = {"seed": a.seed, "profile": a.profile, "stage": a.stage, "init_checkpoint": init_ref,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "examples_consumed": lambda s: s * per_step}
    hooks.write_run_args(out, {
        "stage": a.stage, "profile": a.profile, "seed": a.seed, "dry_run": False, "max_steps": prof["steps"],
        "profile_settings": prof, "model_name_or_path": str(a.init or a.base), "base_model": str(a.base),
        "init_checkpoint": init_ref, "train_file": str(DATA[a.stage]), "train_file_sha256": sha256_file(DATA[a.stage]),
        "nemo_rl_overrides": ov, "attempt": attempt, "script_sha256": run_meta["script_sha256"],
        "hooks_sha256": sha256_file(HERE / "ptc_cad_hooks.py"), "env_sha256": sha256_file(HERE / "cad_env.py"),
        "protocol_sha256": sha256_file(HERE / "executor" / "cad_protocol.py"),
        "seed_derivation": "NeMo RL sft.seed / grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; "
                           "data order = file order (data.shuffle=false)"})
    hooks.install_cut_writer(out, a.base, run_meta)
    if a.stage == "sft":
        hooks.install_sft_sample_log(out, DATA["sft"], prof["batch"], attempt)
        sys.path.insert(0, "/opt/nemo-rl/examples")
        import run_sft as entry  # noqa: E402
        cfg = "/opt/nemo-rl/examples/configs/sft.yaml"
    else:
        from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
        from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
        from nemo_rl.environments.utils import register_env
        fqn = "cad_env.CadQueryEnvironment"
        os.environ["PYTHONPATH"] = f"{HERE}:{HERE / 'executor'}:{os.environ.get('PYTHONPATH', '')}"   # Ray workers import cad_env
        ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM
        register_env("cad", fqn)
        import cad_protocol
        cad_protocol.call_executor(a.socket, [{"id": "ping", "code": "result = cq.Workplane('XY').box(1, 1, 1)"}], 60)
        sys.path.insert(0, "/opt/nemo-rl/examples")
        import run_grpo as entry  # noqa: E402
        cfg = "/opt/nemo-rl/examples/configs/grpo_math_1B.yaml"
    sys.argv = [entry.__file__, "--config", cfg, *ov]
    try:
        entry.main()
    finally:
        hooks.record_attempt(out, "end", stage=a.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
