"""Registry rows of adapter `nemo_rl_ext`: NeMo RL SFT and GRPO for CAD (area 6).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (multimodal spec section 6)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gatedtrain.families.nemo_rl import NEMO_RL_COMMIT, NEMO_RL_RUN_OPTIONS
from gatedtrain.ext.executor import EXECUTOR_ENV, EXECUTOR_LIMITS, EXECUTOR_LIMIT_OPTIONS, EXECUTOR_TMPFS_OPTIONS
from gatedtrain.ext.rowkit import CONTAINER_PATHS, python_tree, runtime_dir

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "CAD" (multimodal spec section 6 row 6 and its deviations; the area 6 decisions in runtime/README.md;
# runtime/README.md "What gatedtrain needs" items 1-9, "Launch request facts", "Executor sandbox", read 2026-11-13; gatedtrain's
# tests/README.md section 11.23, EXT-180 to EXT-199). One adapter covers both stages of the
# MM-15 chain: NeMo RL SFT (`sft.nemo_rl@0.1`, base Qwen2.5-Coder-0.5B-Instruct) and NeMo RL GRPO with the CadQuery
# environment (`rl.nemo_rl_env@0.1`, base the SFT run's gated cut). Every execution of model-written code runs in the
# B18 executor container: the GRPO launch pairs it as a sidecar (`sidecar` below), the scoring launch runs in it.
NAME = "nemo_rl_ext"
# The runtime directory: this repository's runtime/ (scripts, data, models, items, executor/ with its venv; mounted
# read-only at its own host path, so its container path is its host path), or the override; earlier runs lie under the
# host's output root as `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_CAD_RUNTIME"
OUT_NAME = "nemo-rl-cad"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
AREA6 = RUNTIME
# The tree must be a standalone CPython tree (python-build-standalone, as `uv python install` provides) named
# `cpython-<X.Y[.Z]>-<platform>`: ext-launch mounts it and the directory holding it, which a system interpreter's
# prefix (`/usr`) cannot be. README "Setup" gives the command that builds the venv from such a tree.
_TREE_NAME = re.compile(r"cpython-(\d+)\.(\d+)(?:\.\d+)?-[^/]+")


def standalone_tree_version(tree: str, venv: str) -> str:
    """The `X.Y` of a standalone CPython tree `<dir>/cpython-<X.Y[.Z]>-<platform>`; ValueError for any other tree."""
    m = _TREE_NAME.fullmatch(Path(tree).name)
    if m is None:
        raise ValueError(
            f"{venv} was made from the interpreter tree {tree!r}, which is not a standalone CPython tree "
            "(<dir>/cpython-<X.Y.Z>-<platform>, python-build-standalone). The executor's interpreter tree and the "
            "directory holding it are mounted read-only into the containers, so a system interpreter (such as one "
            "under /usr) cannot be used. Rebuild the venv from a uv-managed CPython (README \"Setup\": "
            "uv python install 3.12.16; UV_PYTHON_PREFERENCE=only-managed uv sync --frozen --python 3.12.16).")
    return f"{m.group(1)}.{m.group(2)}"


# The interpreter tree the executor venv was made from (executor/.venv/pyvenv.cfg), mounted read-only at its own host
# path, and the directory holding it (an input root and a sidecar mount); None without the executor venv (ext-launch
# then refuses the interpreter and the sidecar).
AREA6_CPYTHON = python_tree(f"{RUNTIME}/executor")
AREA6_PYTHON = str(Path(AREA6_CPYTHON).parent) if AREA6_CPYTHON else None
_PY_VERSION = (standalone_tree_version(AREA6_CPYTHON, f"{RUNTIME}/executor/.venv")   # 3.12
               if AREA6_CPYTHON else None)
EXECUTOR_IMAGE = "sha256:9b0cfb6e125b8986ad562ef18cff5268460b833d046374b292ca164c4558b89a"   # executor/BUILD.md
EXECUTOR_LOCK_SHA256 = "sha256:8f4264b967785942044058ecd8fcd66468bbb8f4421dbb2e6d941998c3366c55"
AREA6_EXECUTOR = f"{AREA6}/executor"               # cad_sandbox.py, cad_protocol.py, the venv (review 11 redesign)
# The executor code that runs generated programs (executor/cad_sandbox.py) and its wire protocol, pinned: every
# execution record (exec_meta.json) and scoring record names exactly these (EXT-199).
EXECUTOR_CODE_SHA256 = {"cad_sandbox.py": "sha256:b5f19863d54ed35177f1479a805179f0338ca6a4186631ba221e8946be77862e",
                        "cad_protocol.py": "sha256:0f9d801de4119147762a267b0fd7ab71b2bdd3d254f3a84e661f93462decf9b1"}
TRAINER_ROWS[NAME] = {
    "uses": ("sft.nemo_rl@0.1", "rl.nemo_rl_env@0.1"),          # SFT first, then GRPO from its gated cut (MM-15)
    # The framework is pinned by its commit (env-freeze.txt `# nemo-rl commit <40 hex>`, ptc_cad_hooks.write_run_args);
    # no `name==version` line is required, since which packages the image's main venv lists is not known before the
    # GPU smoke (README open question 2).
    "pins": (),
    "nemo_rl_commit": NEMO_RL_COMMIT,
    "outputs": ("cuts/checkpoint-<step>/ full-weight cuts with cut.json written last", "run_args.json",
                "env-freeze.txt", "samples.jsonl", "train_steps.jsonl", "env_calls.jsonl",
                "nemo/logs/exp_<n>/train_data_step<N>.jsonl"),
    # EXT-182: ptc_cad_hooks.py CUT_DEPLOYABLE plus the tokenizer files NeMo RL may save (README item 2); base_ref.json,
    # training_info.json and cut.json are not components.
    "components": ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "sampler.json", "chat_template.jinja", "vocab.json", "merges.txt"),
    # EXT-185, EXT-186: observed per-row logs (SFT sample log; GRPO env log joined with NeMo RL's step log); a log that
    # does not match the run refuses. The README proposed the MM-22 `iteration` fallback when the GRPO join fails at the
    # smoke; that is a decision after the smoke, not an automatic downgrade (gatedtrain's docs/findings.md).
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 1.0,                           # estimate (README "Profiles and cost": area total)
    "evaluators": ("cad.cadquery_items@0.1", "cad.cadquery_valid_items@0.1"),   # deciding first (MM-38)
    "framework": "nemo_rl",
    # EXT-193: the interpreter tree, the runtime directory (scripts, data, models, items, the executor venv) and the
    # training runs' outputs (the GRPO run mounts the SFT cut read-only at /inputs/sft-cut, an evaluation a cut at
    # /inputs/cut and the generation output at /inputs/gen).
    "input_roots": tuple(([AREA6_PYTHON] if AREA6_PYTHON else []) + [AREA6, "{output_root}/" + OUT_NAME]),
    "spec_binding": False,                           # the smoke launches precede the registrations
    # README "Launch request facts": every name, none a credential (HF_TOKEN is never allowed).
    "env_names": ("FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                  "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST", "XDG_CACHE_HOME"),
    "scan_skip": tuple([f"{AREA6}/executor/.venv"]
                       + ([f"{AREA6_CPYTHON}/lib/python{_PY_VERSION}/site-packages"] if AREA6_CPYTHON else [])),
    "interpreter_trees": (AREA6_CPYTHON,) if AREA6_CPYTHON else (),
    # The host-mounted interpreter (score_cad.py in the executor container); the two NeMo RL scripts run the image's
    # own interpreters (`image_interpreters`, pinned by the image ID; nemo-rl docker/Dockerfile:227, :621 for the main
    # venv, gatedtrain's docs/tier-b-runbook.md step 8 for the vLLM worker venv).
    "interpreter": f"{AREA6}/executor/.venv/bin/python",
    "image_interpreters": {
        "run_cad_nemo.py": "/opt/nemo_rl_venv/bin/python",
        "gen_cad.py": "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"},
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME", "XDG_CACHE_HOME"),
    # Fixed docker options per script (code, never request values): NeMo RL's shared memory, ulimits and cwd.
    "run_options": {"run_cad_nemo.py": NEMO_RL_RUN_OPTIONS, "gen_cad.py": NEMO_RL_RUN_OPTIONS},
    # EXT-199 (review 11 redesign): the executor image's two evaluation steps run `python -I <script>` (isolated mode)
    # and are printed exactly as executor/run-executor.sh exec|score prints them (`b18`: read-only root, the noexec
    # tmpfs, the limits, the three environment names; no HF_HUB_OFFLINE: they load no model).
    "interpreter_flags": {"cad_sandbox.py": "-I", "score_cad.py": "-I"},
    "b18": {"scripts": ("cad_sandbox.py", "score_cad.py"), "tmpfs_options": EXECUTOR_TMPFS_OPTIONS,
            "limit_options": EXECUTOR_LIMIT_OPTIONS, "env_names": EXECUTOR_ENV},
    # EXT-190 (decided 2026-11-13): the GRPO launch pairs the executor container as a sidecar, started before
    # and stopped after the trainer; the two share only the spool directory (the Unix socket). Exactly what
    # `executor/run-executor.sh serve` prints; cad_sandbox.py binds the socket with umask 077 (cad_sandbox.py:445), so
    # the trainer runs as the sidecar's user.
    "sidecar": {
        "image": EXECUTOR_IMAGE,
        "venv_lock_sha256": EXECUTOR_LOCK_SHA256,      # executor/uv.lock (README "Pins")
        "argv": (f"{AREA6_EXECUTOR}/.venv/bin/python", "-I", f"{AREA6_EXECUTOR}/cad_sandbox.py", "--serve",
                 "/spool/cadexec.sock", "--workers", "8"),
        # identity mounts (the venv's links name absolute host paths); never the area root, which holds the references
        "ro_mounts": tuple(([AREA6_PYTHON] if AREA6_PYTHON else []) + [AREA6_EXECUTOR]),
        "name_prefix": "gatedtrain-cadexec-",               # run-executor.sh serve: gatedtrain-cadexec-<label>
        "spool": "/spool",
        "socket": "/spool/cadexec.sock",
        "env_names": EXECUTOR_ENV,
        "env_paths": ("HOME",),
        "tmpfs": ("/tmp",),
        "tmpfs_options": EXECUTOR_TMPFS_OPTIONS,
        "limits": EXECUTOR_LIMITS,
    },
}

# Area "CAD" evaluators (runtime/README.md "Decisions": deciding `iou`, T3 with the construct-alignment disclosure,
# blocking; `valid` supporting, T0). Scorer: score_cad.py in the executor container; code_sha256 is its composite
# `scorer_code_sha256` (SHA-256 of the compact sorted JSON of the SHA-256 of score_cad.py, geom_metrics.py,
# executor/cad_sandbox.py and executor/cad_protocol.py; score_cad.py sha256 f7215cf1..., the review-11 version: it scores
# the records of the execution step and runs no generated code; numeric sub_metrics with `status_code` and `truncated`,
# `framework` {name, version}, arm-valid under its own id, gen_meta.json copied), the version the string it writes (a CPU
# gen -> exec -> score of 8 pilot items, tests/fixtures/cad/real-arm/). An item is one registered part description; its full prompt `text`
# is the exact-overlap key against the training rows' `text` (= `input`, the prompt NeMo RL reads).
SCORE_CAD_CODE_SHA256 = "sha256:6f44fcf8187644e6f9e5db5c31c0bd64516eaa28ba2fa7eb39c5817c394d3e42"
_CAD = {
    "direction": "higher_is_better",
    "score_range": (0, 1),
    # MM-4: evaluator faults (the sandbox itself failed; a reference mesh or the geometry failed; no generation).
    "error_classes": ("executor_error", "scorer_error", "missing_output"),
    # MM-4: the policy's faults, scored 0 (cad_sandbox.py STATUSES other than `ok`).
    "score_outcomes": ("no_code", "syntax_error", "runtime_error", "no_result", "not_solid", "invalid_solid", "timeout",
                       "resource_limit", "crash"),
    "payload_text_field": "text",
    # Strings only (MM-6): `ref_volume` (a number) and `ref_bbox` (a list) stay in the item rows, unhashed by name.
    "item_payload": ("text", "reference_code", "ref_mesh", "ref_mesh_sha256"),
    "normalizer": False,
    "env_version": False,
    # The evaluator's sampler (score_cad.py: gen_cad.py SAMPLER plus the generation backend and the SHARED PROMPT hash).
    "sampler_keys": ("backend", "chat_template", "max_new_tokens", "messages", "prompt_block_sha256",
                     "samples_per_seed", "strategy", "temperature", "top_p"),
    "framework": "vllm",
    "samples": "300, 1",
}
EVALUATOR_ROWS["cad.cadquery_items@0.1"] = {
    **_CAD, "metric": "iou", "binary": False,
    # 05 section 6.1, R15-1 (decided 2026-11-13): another engine computes the score (numpy line integration over
    # an OCCT triangulation) than the reward (OCCT mass properties and bounding box), so T3; the two share the CadQuery/OCCT
    # kernel that interprets the program, OCCT's tessellation and the valid-solid requirement, so construct-aligned in
    # part, stated in the registration and the report.
    "tier": "T3",
    "construct_aligned": {
        "reward": "cad_reward_v1",
        "statement": ("iou measures agreement of the generated solid with the reference solid, the construct the "
                      "training reward cad_reward_v1 (OCCT validity, volume ratio and bounding-box IoU) also rewards, "
                      "with another engine (numpy volume integration over an OCCT triangulation, not OCCT mass "
                      "properties); both share the CadQuery/OCCT kernel that interprets the program and the "
                      "valid-solid requirement. A pass shows that the generated solids agree better with the "
                      "references under an independent engine, partly through the same validity requirement the "
                      "reward pays for, not a general gain in CAD skill (05 section 6.1, R15-1)")},
    "scorer": {"name": "volumetric_iou_vertical_lines",
               "version": ('cad_iou_v1; numpy 2.5.4; 16384 columns; executor cadquery 2.8.0 ocp 7.9.3.1; limits '
                           '{"as_extra_mb": 2048, "cpu_s": 10, "fsize_mb": 64, "nofile": 64, "wall_s": 15.0}'),
               "weights_sha256": None, "code_sha256": SCORE_CAD_CODE_SHA256}}
EVALUATOR_ROWS["cad.cadquery_valid_items@0.1"] = {
    **_CAD, "metric": "valid", "binary": True, "tier": "T0",   # the reward's validity check: supporting, never deciding
    "scorer": {"name": "occt_valid_solid", "version": "cad_iou_v1; cadquery 2.8.0 isValid", "weights_sha256": None,
               "code_sha256": SCORE_CAD_CODE_SHA256}}

PLUGIN = {"adapters": {NAME: "gatedtrain_recipe_nemo_rl_cad.adapter:NemoRlExtAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
