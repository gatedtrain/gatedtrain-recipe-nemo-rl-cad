"""Area "CAD" test helpers (verifier-owned; tests/README.md section 11.23, EXT-180 to EXT-199).

Written from the area runtime (`runtime/`: README.md "Trainer details", "Outputs",
"What gatedtrain needs", "Launch request facts", "Executor sandbox"; `run_cad_nemo.py` `overrides`/`main`,
`ptc_cad_hooks.py` `install_sft_sample_log`, `write_cut`, `write_run_args`, `record_attempt`; `cad_env.py`
`CadEnvCore.score` and the Ray actor's `step`; `gen_cad.py`, `score_cad.py`, `cad_protocol.py`, `cad_sandbox.py`
`serve`; `executor/run-executor.sh`, `executor/BUILD.md`; read 2026-11-13), from NeMo RL's per-step log as the pinned
clone writes it (`nemo_rl/algorithms/grpo_sync.py:1427-1460`, `nemo_rl/utils/logger.py:1409-1443`: one line per
sample, `{"rewards": [r], ..., "idx": i}`, Tier B's `out/smoke/rl/logs/exp_001/train_data_step4.jsonl`) and from the
area's CPU pilot outputs (`pilot/cut30-sftgate-score/`), not from gatedtrain. Only `executor/uv.lock` is copied (into
`fixtures/cad/executor-uv.lock`): its SHA-256 is the pinned executor lock. Weights are byte stand-ins (minimal
safetensors files); nothing in gatedtrain loads them (MM-12).

The NeMo RL code that writes the training logs had not run when this was written (GPU image): the fixture follows the
hooks' code as written, which the GPU smoke must confirm (gatedtrain's docs/findings.md, area 6 "Unverified").

Host paths are mirrored: the runtime directory and the Python tree are mounted read-only at their own host paths
(README "Launch request facts"), so the rows name them as container paths. The tests pin them to a neutral layout
under one temporary root `PATHS_ROOT` (below, in the git-ignored `build/`): `GATEDTRAIN_RECIPE_NEMO_RL_CAD_RUNTIME` names `<PATHS_ROOT>/recipe/runtime`, a
stub holding only `executor/.venv/pyvenv.cfg` (whose `home` names `<PATHS_ROOT>/python/cpython-3.12.16-.../bin`), set
before gatedtrain's registry loads the rows. `GATEDTRAIN_EXT_MIRROR_ROOT` = `<root>/host` stands for the container's `/`
(EXT-112). Layout of `CadScenario(root)` under `<root>/host/<PATHS_ROOT>/`:
- `python/cpython-3.12.16-linux-x86_64-gnu/` a stand-in interpreter tree;
- `recipe/runtime/` the runtime stand-in: the scripts (stand-ins, never executed), `executor/.venv/bin/python` and
  `executor/uv.lock` (the real lock), `models/qwen2.5-coder-0.5b-instruct/` (the 8 file names of the real base) and
  `models/MODELS.json`, `data/{sft,grpo}/shards/shard-0000{0..3}.jsonl` and their concatenation `train.jsonl`,
  `items-sft/` (30 items of the 10 SFT families) and `items-grpo/` (30 items of the 5 held-out families);
- `out/nemo-rl-cad/sft-1/` an SFT run (seed 6000, batch 32, 160 rows = 5 steps per epoch, 6 steps: it crosses an
  epoch; cuts 3 and 6), `grpo-1/train/` a GRPO run from the SFT cut 6 (seed 6100, 16 prompts x 8, 48 rows = 3 steps per epoch,
  4 steps; cuts 2 and 4; one env call of 128 samples per step), `grpo-1/spool/` its executor spool (the decided sibling layout), and four scoring
  runs (`eval/{sft-base, sft-cand, grpo-base, grpo-cand}`) as score_cad.py writes them;
- `<root>/sft/` and `<root>/grpo/`: spec, registration, trainer.yaml, launch record, smoke record per stage.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import pwd
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml

# The neutral layout (module docstring): set before gatedtrain's registry imports the recipe's rows.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_CAD_RUNTIME"
REPO = Path(__file__).resolve().parents[1]
REAL_RUNTIME = Path(os.environ.get(RUNTIME_ENV) or REPO / "runtime")   # the real runtime (real-file tests)
EVIDENCE = REPO / "results" / "2026-11-14"                          # gate files and the CPU pilot's evidence
assert "gatedtrain.ext.registry" not in sys.modules, "import _nemo_rl_ext before gatedtrain.ext.registry (the rows read the runtime)"
# Not under /tmp: the containers' tmpfs is /tmp, and a mount never nests in it (EXT-88). build/ is git-ignored.
(REPO / "build").mkdir(exist_ok=True)
PATHS_ROOT = tempfile.mkdtemp(prefix="test-paths-", dir=REPO / "build")
atexit.register(shutil.rmtree, PATHS_ROOT, True)
AREA6 = f"{PATHS_ROOT}/recipe/runtime"                       # container path = host path (identity mount)
PYTHON_TREE = f"{PATHS_ROOT}/python"
CPYTHON = f"{PYTHON_TREE}/cpython-3.12.16-linux-x86_64-gnu"
OUT_ROOT = f"{PATHS_ROOT}/out"
DATA_ROOT = PATHS_ROOT
OUT_NAME = "nemo-rl-cad"
(Path(AREA6) / "executor" / ".venv").mkdir(parents=True)
(Path(AREA6) / "executor" / ".venv" / "pyvenv.cfg").write_text(f"home = {CPYTHON}/bin\nversion_info = 3.12.16\n",
                                                                 encoding="utf-8")
os.environ[RUNTIME_ENV] = AREA6

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, gpu_step_script, launch_label, safetensors_bytes, sha_of, write_canonical
from gatedtrain.testing.support import file_sha256, manifest_entries, manifest_hash, run_cli

CAD = EXT / "cad"
ADAPTER = "nemo_rl_ext"
SFT_ID, GRPO_ID = "sft.nemo_rl@0.1", "rl.nemo_rl_env@0.1"
EVALUATOR_ID, SUPPORT_ID = "cad.cadquery_items@0.1", "cad.cadquery_valid_items@0.1"
NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"
NEMO_RL_IMAGE = "sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a"
EXECUTOR_IMAGE = "sha256:9b0cfb6e125b8986ad562ef18cff5268460b833d046374b292ca164c4558b89a"
EXECUTOR_LOCK = CAD / "executor-uv.lock"
EXECUTOR_LOCK_SHA = "sha256:8f4264b967785942044058ecd8fcd66468bbb8f4421dbb2e6d941998c3366c55"
TRAIN_PY = "/opt/nemo_rl_venv/bin/python"                   # nemo-rl docker/Dockerfile:227, :621
GEN_PY = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
EXEC_PY = f"{AREA6}/executor/.venv/bin/python"
BASE_REL = "models/qwen2.5-coder-0.5b-instruct"
BASE_REPO, BASE_REVISION = "Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
BASE_FILES = ("LICENSE", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
              "tokenizer_config.json", "vocab.json")
# The SHARED PROMPT block of cad_protocol.py (README "Task and data", data_meta.json `prompt_block_sha256`).
PROMPT_BLOCK_SHA256 = "sha256:b9aea19b65e447006db92eb9e868357f519a37455d59169a778075c18ffd7e69"
# gen_cad.py SAMPLER (every cut's sampler.json, ptc_cad_hooks.write_cut) and the evaluator's sampler (score_cad.py:
# SAMPLER plus the backend and the prompt hash), the registered sampler of a GPU (vLLM) evaluation.
CUT_SAMPLER = {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "samples_per_seed": 1,
               "chat_template": ("checkpoint tokenizer_config chat template, add_generation_prompt=True, "
                                 "enable_thinking=False"),
               "messages": "[{role: user, content: cad_protocol.user_prompt(description)}]"}
SAMPLER = {**CUT_SAMPLER, "backend": "vllm", "prompt_block_sha256": PROMPT_BLOCK_SHA256}
SCORER_CODE = "sha256:958ab45f7e79582b352052753b0794832011289cfc746b951bd5e2762b8a16e6"
# score_cad.py env-runtime.json `code_sha256` of the review-11 runtime with the RFC 8785 sampler hash
# (scorer_code_sha256 = its canonical JSON hash, the same bytes as compact sorted JSON for these strings).
CODE_SHA256 = {"cad_protocol.py": "sha256:0f9d801de4119147762a267b0fd7ab71b2bdd3d254f3a84e661f93462decf9b1",
               "cad_sandbox.py": "sha256:b5f19863d54ed35177f1479a805179f0338ca6a4186631ba221e8946be77862e",
               "geom_metrics.py": "sha256:c8f4f0510dbceaa8407d0685d87c6aa69b0e3e61e6a12ce6384c63efd3ce3de4",
               "score_cad.py": "sha256:9301737632efefb69fe743c620f63f6ecc8287553c5e07cc3e961bb94afb1783"}
EXECUTOR = f"{AREA6}/executor"
EXEC_LIMITS = {"as_extra_mb": 2048, "cpu_s": 10, "fsize_mb": 64, "nofile": 64, "wall_s": 15.0}
EXEC_VERSIONS = {"cadquery": "2.8.0", "ocp": "7.9.3.1", "python": "3.12.16"}
EVAL_CUT = "/inputs/cut"
SCORER = {"name": "volumetric_iou_vertical_lines",
          "version": ('cad_iou_v1; numpy 2.5.4; 16384 columns; executor cadquery 2.8.0 ocp 7.9.3.1; limits '
                      '{"as_extra_mb": 2048, "cpu_s": 10, "fsize_mb": 64, "nofile": 64, "wall_s": 15.0}'),
          "weights_sha256": None, "code_sha256": SCORER_CODE}
SUPPORT_SCORER = {"name": "occt_valid_solid", "version": "cad_iou_v1; cadquery 2.8.0 isValid", "weights_sha256": None,
                  "code_sha256": SCORER_CODE}
SFT_FAMILIES = ("box", "cylinder", "plate_hole", "hex_prism", "stepped_shaft", "fillet_box", "cup", "slot_plate",
                "l_bracket", "cbore_plate")
HELD_OUT = ("blind_hole", "plate_4holes", "tube", "flange", "chamfer_box")
STATUSES_FAIL = ("no_code", "syntax_error", "runtime_error", "no_result", "not_solid", "invalid_solid", "timeout",
                 "resource_limit", "crash")
SEEDS = [11]
SFT_SEED, GRPO_SEED = 6000, 6100
BATCH, PROMPTS, GENERATIONS = 32, 16, 8
ENV_NAMES = ["FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
             "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"]
TRAIN_ENV = {"FLASHINFER_CUDA_ARCH_LIST": "8.9", "HF_HUB_OFFLINE": "1", "HOME": "/tmp/home",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "RAY_memory_monitor_refresh_ms": "0",
             "TORCH_CUDA_ARCH_LIST": "8.9"}
SIDECAR_ENV = {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
SIDECAR_ARGV = [EXEC_PY, "-I", f"{AREA6}/executor/cad_sandbox.py", "--serve", "/spool/cadexec.sock", "--workers", "8"]
SIDECAR_LIMITS = {"cpus": 8, "memory": "8g", "memory_swap": "8g", "pids": 256, "nofile": "1024:1024", "core": 0}
INSTRUCTION = "Write CadQuery (Python) code that builds the part described below. (fixture stand-in of prompt v2)"
SCRIPTS = ("run_cad_nemo.py", "ptc_cad_hooks.py", "cad_env.py", "executor/cad_protocol.py", "gen_cad.py",
           "score_cad.py", "executor/cad_sandbox.py", "geom_metrics.py")   # review-11 layout: executor code apart
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*)\Z", re.DOTALL | re.IGNORECASE)


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def other_user() -> str:
    """A `<uid>:<gid>` other than run_user()'s, for a sidecar whose user differs from the request's."""
    uid = os.getuid() + 1
    return f"{uid}:{uid}"


def sha_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, doc: Any) -> Path:
    """The runtime's `json.dumps(..., indent=2, sort_keys=True) + "\\n"`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def extract_program(text: str | None) -> str | None:
    """cad_protocol.extract_program (written from cad_protocol.py :52-68)."""
    if text is None:
        return None
    m = _FENCE.search(text) or _OPEN_FENCE.search(text)
    if not m:
        return None
    code = m.group(1).strip("\n")
    return code if code.strip() else None


def float32(x: float) -> float:
    """The value NeMo RL logs: cad_env.py returns `torch.tensor(rewards, dtype=torch.float32)`."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def compact_sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def host_config_sha256(host_work: Path, output_root: Path, input_roots: list, *, mirror_root: Path,
                       interpreter_trees: list) -> str:
    """EXT-85, EXT-117, EXT-192: the effective layout's hash; an adapter with image interpreters adds them."""
    return sha_of({"host_work": str(host_work), "output_root": str(output_root),
                   "input_roots": [str(r) for r in input_roots], "mirror_root": str(mirror_root),
                   "interpreter": EXEC_PY, "interpreter_trees": [str(t) for t in interpreter_trees],
                   "image_interpreters": {"gen_cad.py": GEN_PY, "run_cad_nemo.py": TRAIN_PY}})


# ---------------------------------------------------------------------------
# The runtime directory (README "Task and data", "Pins")
# ---------------------------------------------------------------------------

def prompt(description: str) -> str:
    return f"{INSTRUCTION}\n\nPart: {description}"


def data_rows(prefix: str, families: tuple[str, ...], n: int) -> list[dict]:
    """make_data.py rows: `input` = `text` = the prompt, `output` the SFT target or the GRPO reference features."""
    rows = []
    for k in range(n):
        fam = families[k % len(families)]
        desc = f"A {fam} number {k} ({prefix})"
        rid = f"{prefix}-{k:05d}"
        out = (f"```python\nimport cadquery as cq\n\nresult = cq.Workplane(\"XY\").box({k + 1}, 2, 3)\n```"
               if prefix == "sft" else json.dumps({"bbox": [-1, -1, -1, 1, 1, 1], "row_id": rid, "volume": 8.0},
                                                  sort_keys=True, separators=(",", ":")))
        rows.append({"family": fam, "input": prompt(desc), "output": out, "params": [fam, k], "row_id": rid,
                     "text": prompt(desc)})
    return rows


def write_data(d: Path, rows: list[dict], n_shards: int = 4) -> Path:
    """`<d>/shards/shard-0000<i>.jsonl` (compact JSON lines, as make_data.py) and `<d>/train.jsonl`, their
    concatenation; returns the shard directory (the spec's training data)."""
    per = len(rows) // n_shards
    text = ""
    for i in range(n_shards):
        chunk = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows[i * per:(i + 1) * per])
        (d / "shards").mkdir(parents=True, exist_ok=True)
        (d / "shards" / f"shard-{i:05d}.jsonl").write_text(chunk, encoding="utf-8")
        text += chunk
    (d / "train.jsonl").write_text(text, encoding="utf-8")
    return d / "shards"


def write_items(d: Path, prefix: str, families: tuple[str, ...], n: int) -> list[dict]:
    items = []
    for k in range(n):
        fam = families[k % len(families)]
        item = f"cad-{prefix}-{fam}-{k:04d}"
        mesh = d / "ref" / f"{item}.stl"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(f"solid stand-in {item}\nendsolid\n".encode())
        desc = f"An item {fam} {k} ({prefix} gate)"
        items.append({"description": desc, "family": fam, "item": item, "params": [fam, k],
                      "ref_bbox": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], "ref_mesh": f"ref/{item}.stl",
                      "ref_mesh_sha256": file_sha256(mesh), "ref_volume": 8.0,
                      "reference_code": "import cadquery as cq\n\nresult = cq.Workplane(\"XY\").box(2, 2, 2)",
                      "text": prompt(desc)})
    write_jsonl(d / "items.jsonl", items)
    write_json(d / "item_set_meta.json", {"families": sorted(set(families)), "n_items": n})
    return items


def build_base(d: Path) -> Path:
    for name in BASE_FILES:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == "model.safetensors":
            p.write_bytes(safetensors_bytes("qwen2.5-coder-0.5b-instruct"))
        elif name.endswith(".json"):
            write_json(p, {"_fixture": name, "architectures": ["Qwen2ForCausalLM"]})
        else:
            p.write_text(f"fixture stand-in {name}\n", encoding="utf-8")
    return d


def models_json(base: Path) -> dict:
    """fetch_models.py MODELS.json (the base entry): files {name: {bytes, sha256}}."""
    return {"models": {base.name: {"bytes": sum((base / n).stat().st_size for n in BASE_FILES),
                                   "files": {n: {"bytes": (base / n).stat().st_size, "sha256": file_sha256(base / n)}
                                             for n in BASE_FILES},
                                   "license": "apache-2.0", "repo": BASE_REPO, "revision": BASE_REVISION}}}


# ---------------------------------------------------------------------------
# Cuts (ptc_cad_hooks.write_cut) and identity (README item 2)
# ---------------------------------------------------------------------------

def build_cut(run: Path, step: int, *, stage: str, profile: str, seed: int, script_sha256: str, base: Path,
              per_step: int, init: dict | None = None, tag: str = "", tokenizer_files=("chat_template.jinja",)) -> Path:
    """`<run>/nemo/checkpoints/step_<N>/` (NeMo RL's resume state, never a cut) and `<run>/cuts/checkpoint-<N>/` as
    write_cut builds it, `cut.json` last; `cuts.jsonl` appended."""
    step_dir = run / "nemo" / "checkpoints" / f"step_{step}"
    cons = step_dir / "policy" / "weights" / "model" / "consolidated"
    cons.mkdir(parents=True)
    (cons / "model-00001-of-00001.safetensors").write_bytes(safetensors_bytes(f"{stage} step {step} {tag}"))
    write_json(cons / "config.json", {"_fixture": "config", "architectures": ["Qwen2ForCausalLM"]})
    tok = step_dir / "policy" / "tokenizer"
    tok.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", *tokenizer_files):
        (tok / name).write_text(f"tokenizer stand-in {name}\n", encoding="utf-8")
    (step_dir / "training_info.json").write_text(json.dumps({"step": step, "total_steps": step}), encoding="utf-8")
    (step_dir / "optimizer" / "state").mkdir(parents=True)
    d = run / "cuts" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    os.link(cons / "model-00001-of-00001.safetensors", d / "model.safetensors")
    os.link(cons / "config.json", d / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", *tokenizer_files):
        os.link(tok / name, d / name)
    shutil.copyfile(base / "generation_config.json", d / "generation_config.json")
    (d / "sampler.json").write_text(json.dumps(CUT_SAMPLER, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = models_json(base)["models"][base.name]["files"]
    (d / "base_ref.json").write_text(json.dumps({"repo": BASE_REPO, "revision": BASE_REVISION, "files": files,
                                                 "init_checkpoint": init}, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    shutil.copyfile(step_dir / "training_info.json", d / "training_info.json")
    files = {f.name: file_sha256(f) for f in sorted(d.iterdir())}
    deploy = sorted(n for n in files if n in ("model.safetensors", "config.json", "generation_config.json",
                                              "tokenizer.json", "tokenizer_config.json", "sampler.json",
                                              "chat_template.jinja", "vocab.json", "merges.txt"))
    write_json(d / "cut.json", {"schema": "ptc_mm.cad_cut", "schema_version": "1.0", "global_step": step,
                                "files": files, "deployable": deploy, "seed": seed, "profile": profile, "stage": stage,
                                "script_sha256": script_sha256, "examples_consumed": step * per_step,
                                "nemo_rl_step_dir": f"step_{step}"})
    with open(run / "cuts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"cut": d.name, "step": step, "utc": "2026-11-14T10:00:00.000000Z"}, sort_keys=True) + "\n")
    return d


def deploy_hash(cut: Path) -> str:
    """MM-17: the manifest hash of the cut's `cut.json` `deployable` files (full weights, no composition)."""
    deploy = json.loads((cut / "cut.json").read_text(encoding="utf-8"))["deployable"]
    entries = [e for e in manifest_entries(cut) if e["path"] in deploy]
    from gatedtrain.testing.support import canonical_bytes
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


# ---------------------------------------------------------------------------
# Training runs (run_cad_nemo.py + ptc_cad_hooks.py + cad_env.py)
# ---------------------------------------------------------------------------

def run_args(*, stage: str, profile: str, seed: int, steps: int, save_period: int, runtime: Path, script_sha: str,
             init: dict | None) -> dict:
    """run_cad_nemo.py `main` -> ptc_cad_hooks.write_run_args."""
    data = f"{AREA6}/data/{stage}/train.jsonl"
    prof = ({"steps": steps, "batch": BATCH, "save_period": save_period} if stage == "sft" else
            {"steps": steps, "prompts": PROMPTS, "generations": GENERATIONS, "save_period": save_period})
    ov = ["data.shuffle=false", "data.train.dataset_name=ResponseDataset", f"++data.train.data_path={data}",
          f"checkpointing.save_period={save_period}", "logger.log_dir=/out/nemo/logs"]
    return {"stage": stage, "profile": profile, "seed": seed, "dry_run": False, "max_steps": steps,
            "profile_settings": prof, "model_name_or_path": init["path"] if init else f"{AREA6}/{BASE_REL}",
            "base_model": f"{AREA6}/{BASE_REL}", "init_checkpoint": init, "train_file": data,
            "train_file_sha256": file_sha256(runtime / "data" / stage / "train.jsonl"), "nemo_rl_overrides": ov,
            "attempt": 1, "script_sha256": script_sha, "hooks_sha256": file_sha256(runtime / "ptc_cad_hooks.py"),
            "env_sha256": file_sha256(runtime / "cad_env.py"), "protocol_sha256": file_sha256(runtime / "executor" / "cad_protocol.py"),
            "seed_derivation": "NeMo RL sft.seed / grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; "
                               "data order = file order (data.shuffle=false)",
            "framework": {"name": "nemo_rl", "commit": NEMO_RL_COMMIT, "versions": {"torch": "2.10.0", "ray": "2.49.2"}},
            "python": "3.13.9", "created_at": "2026-11-14T09:00:00.000000Z"}


def common_files(out: Path, stage: str) -> None:
    (out / "env-freeze.txt").write_text("nemo-rl==0.5.0rc0\ntorch==2.10.0\ntransformers==4.57.1\n"
                                        f"# nemo-rl commit {NEMO_RL_COMMIT}\n", encoding="utf-8")
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "event": "start", "pid": 1, "profile": "full", "stage": stage,
                                          "utc": "2026-11-14T09:00:00.000000Z"},
                                         {"attempt": 1, "event": "end", "pid": 1, "stage": stage,
                                          "utc": "2026-11-14T10:00:00.000000Z"}])
    (out / "nemo" / "logs" / "exp_001").mkdir(parents=True, exist_ok=True)


def sft_logs(rows: list[dict], steps: int) -> tuple[list[dict], list[dict]]:
    """ptc_cad_hooks.install_sft_sample_log: batch `idx` = 0-based line of train.jsonl in file order (shuffle off,
    drop_last), epoch = (step - 1) // (len(rows) // batch)."""
    spe = len(rows) // BATCH
    samples, steps_log = [], []
    for k in range(1, steps + 1):
        epoch = (k - 1) // spe
        start = ((k - 1) % spe) * BATCH
        lines = []
        for pos in range(BATCH):
            r = rows[start + pos]
            lines.append({"attempt": 1, "epoch": epoch, "idx": start + pos, "position": pos,
                          "row": sha_text(r["row_id"]), "row_id": r["row_id"], "step": k,
                          "utc": "2026-11-14T09:10:00.000000Z"})
        samples += lines
        steps_log.append({"epoch": epoch, "example_ids": [x["row_id"] for x in lines], "n_examples": BATCH, "step": k})
    return samples, steps_log


def build_sft_run(out: Path, runtime: Path, *, profile: str = "full", steps: int = 6, save_period: int = 3,
                  seed: int = SFT_SEED) -> Path:
    out.mkdir(parents=True)
    script_sha = file_sha256(runtime / "run_cad_nemo.py")
    common_files(out, "sft")
    write_json(out / "run_args.json", run_args(stage="sft", profile=profile, seed=seed, steps=steps,
                                               save_period=save_period, runtime=runtime, script_sha=script_sha,
                                               init=None))
    rows = read_jsonl(runtime / "data" / "sft" / "train.jsonl")
    samples, steps_log = sft_logs(rows, steps)
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "train_steps.jsonl", steps_log)
    base = runtime / BASE_REL
    for k in range(save_period, steps + 1, save_period):
        build_cut(out, k, stage="sft", profile=profile, seed=seed, script_sha256=script_sha, base=base, per_step=BATCH)
    return out


def reward_of(step: int, j: int, m: int) -> tuple[str, float]:
    """(status, cad_protocol.reward) of generation m of prompt j at a step: two failures, then OCCT terms that are not
    float32-exact."""
    if m == 0:
        return "no_code", 0.0
    if m == 1:
        return "runtime_error", 0.0
    return "ok", 0.2 + 0.4 * (m / 7) + 0.4 * (((j + step) % 5) / 9)


def grpo_logs(rows: list[dict], steps: int) -> tuple[list[dict], dict[int, list[dict]]]:
    """cad_env.CadEnvCore.score lines (one call per step, the step's 16 prompts x 8 generations, rows in file order,
    16 per step, drop_last) and NeMo RL's train_data_step<N>.jsonl lines (logger.log_batched_dict_as_jsonl, rewards as
    float32 values in the same order)."""
    spe = len(rows) // PROMPTS
    env, data = [], {}
    for n in range(1, steps + 1):
        start = ((n - 1) % spe) * PROMPTS
        lines = []
        for j in range(PROMPTS):
            r = rows[start + j]
            for m in range(GENERATIONS):
                status, reward = reward_of(n, j, m)
                resp = "I cannot write that." if status == "no_code" else \
                    f"```python\nimport cadquery as cq\n\nresult = cq.Workplane(\"XY\").box({m + 1}, {j + 1}, {n})\n```"
                prog = extract_program(resp)
                lines.append({"call": n, "position": len(lines), "row_id": r["row_id"], "status": status,
                              "reward": reward, "program_sha256": sha_text(prog) if prog else None,
                              "response_sha256": sha_text(resp), "exec_batch_s": 0.68,
                              "utc": "2026-11-14T11:00:00.000000Z"})
        env += lines
        data[n] = [{"rewards": [float32(x["reward"])], "input_lengths": [640], "token_loss_mask": [[1, 1]],
                    "sample_loss_mask": [1.0], "advantages": [[0.0, 0.0]], "generation_logprobs": [[0.0, 0.0]],
                    "prev_logprobs": [[0.0, 0.0]], "token_ids": [[1, 2]], "content": ["stand-in"], "idx": i}
                   for i, x in enumerate(lines)]
    return env, data


def build_grpo_run(out: Path, runtime: Path, sft_cut: Path, *, profile: str = "full", steps: int = 4,
                   save_period: int = 2, seed: int = GRPO_SEED) -> Path:
    out.mkdir(parents=True)
    script_sha = file_sha256(runtime / "run_cad_nemo.py")
    common_files(out, "grpo")
    init = {"path": "/inputs/sft-cut", "global_step": json.loads((sft_cut / "cut.json").read_text())["global_step"],
            "cut_json_sha256": file_sha256(sft_cut / "cut.json")}
    write_json(out / "run_args.json", run_args(stage="grpo", profile=profile, seed=seed, steps=steps,
                                               save_period=save_period, runtime=runtime, script_sha=script_sha,
                                               init=init))
    rows = read_jsonl(runtime / "data" / "grpo" / "train.jsonl")
    env, data = grpo_logs(rows, steps)
    write_jsonl(out / "env_calls.jsonl", env)
    for n, lines in data.items():
        (out / "nemo" / "logs" / "exp_001" / f"train_data_step{n}.jsonl").write_text(
            "".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    base = runtime / BASE_REL
    for k in range(save_period, steps + 1, save_period):
        build_cut(out, k, stage="grpo", profile=profile, seed=seed, script_sha256=script_sha, base=base,
                  per_step=PROMPTS, init=init)
    return out


# ---------------------------------------------------------------------------
# Scoring runs (score_cad.py, the arm format MM-1/MM-2 requires: numeric sub_metrics, framework {name, version})
# ---------------------------------------------------------------------------

def baseline_score(i: int) -> tuple[str, float]:
    if i % 5 == 0:
        return "runtime_error", 0.0
    return "ok", ((i * 7) % 9) / 8


def candidate_score(i: int) -> tuple[str, float]:
    st, s = baseline_score(i)
    return ("ok", min(1.0, s + 0.25 + 0.125 * (i % 3))) if i % 10 else (st, s)


STATUSES = ("ok",) + STATUSES_FAIL                          # cad_sandbox.py STATUSES (status_code = the index)
GEN_MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")   # gen_cad.py `deploy`


def gen_meta(model_dir: Path, items_dir: Path, n_items: int) -> dict:
    """gen_cad.py gen_meta.json of a vLLM generation of `model_dir` (a base directory or a cut)."""
    cut = None
    if (model_dir / "cut.json").is_file():
        cut = {"global_step": json.loads((model_dir / "cut.json").read_text())["global_step"],
               "cut_json_sha256": file_sha256(model_dir / "cut.json")}
    return {"backend": "vllm", "model_dir": EVAL_CUT if cut else f"{AREA6}/{BASE_REL}", "cut": cut,
            "model_files": {f: file_sha256(model_dir / f) for f in GEN_MODEL_FILES if (model_dir / f).is_file()},
            "sampler": CUT_SAMPLER, "eval_seeds": list(SEEDS), "items": n_items,
            "items_sha256": file_sha256(items_dir / "items.jsonl"), "limit": None, "round_robin": False,
            "prompt_block_sha256": PROMPT_BLOCK_SHA256, "script_sha256": "sha256:" + "3" * 64,
            "versions": {"vllm": "0.29.0", "dtype": "bfloat16"}, "wall_seconds": 60.0, "seconds_per_response": 0.2}


def build_score(out: Path, items: list[dict], items_dir: Path, score_fn: Callable[[int], tuple[str, float]], *,
                model: str | None, item_set: str | None, model_dir: Path) -> Path:
    """`<out>/outputs/<item>__r0.json`, `scores.jsonl`, `env-runtime.json`, `summary.json`, `gen_meta.json` (copied from
    the generation run) and the arms `arm-iou/`, `arm-valid/` (hard-linked outputs) as score_cad.py (sha256 cffdd9ea...)
    writes them: arm sub_metrics finite numbers only, with `status_code` and `truncated`."""
    (out / "outputs").mkdir(parents=True)
    write_json(out / "gen_meta.json", gen_meta(model_dir, items_dir, len(items)))
    rows = []
    for i, it in enumerate(items):
        status, score = score_fn(i)
        resp = f"```python\nimport cadquery as cq\n\nresult = cq.Workplane(\"XY\").box({i + 1}, 2, 2)\n```"
        prog = extract_program(resp)
        sub = ({"status": status, "executes": 1, "chamfer": 0.01, "volume_ratio": score, "bbox_iou": score,
                "reward": 0.2 + 0.8 * score, "n_new_tokens": 31, "finish_reason": "stop", "unbalanced_columns": 0}
               if status == "ok" else
               {"status": status, "executes": 0, "chamfer": None, "volume_ratio": None, "bbox_iou": None,
                "reward": 0.0, "n_new_tokens": 40, "finish_reason": "stop", "unbalanced_columns": None})
        rel = f"outputs/{it['item']}__r0.json"
        write_json(out / rel, {"item": it["item"], "repeat": 0, "seed": 11, "response": resp, "program": prog,
                               "status": status, "score": score, "sub_metrics": sub})
        rows.append({"item": it["item"], "family": it["family"], "repeat": 0, "seed": 11, "error": None,
                     "score": score, "sub_metrics": sub, "output_ref": {"path": rel, "sha256": file_sha256(out / rel)}})
    write_jsonl(out / "scores.jsonl", rows)
    # The execution step's record (cad_sandbox.py --run-batch exec_meta.json), copied beside the arms.
    ids = [f"{it['item']}__s11" for it in items]
    files = {}
    for i, eid in enumerate(ids):
        files[f"programs/{eid}/result.json"] = "sha256:" + hashlib.sha256(f"result {eid}".encode()).hexdigest()
        if rows[i]["sub_metrics"]["status"] == "ok":
            files[f"programs/{eid}/mesh.stl"] = "sha256:" + hashlib.sha256(f"mesh {eid}".encode()).hexdigest()
    (out / "exec_meta.json").write_text(json.dumps({
        "schema": "ptc_mm.cad_exec", "schema_version": "1.0", "ids": ids,
        "gen_meta_sha256": file_sha256(out / "gen_meta.json"), "versions": EXEC_VERSIONS, "limits": EXEC_LIMITS,
        "code_sha256": {k: CODE_SHA256[k] for k in ("cad_sandbox.py", "cad_protocol.py")}, "files": files},
        sort_keys=True, indent=1) + "\n", encoding="utf-8")
    write_json(out / "env-runtime.json", {
        "code_sha256": dict(CODE_SHA256), "executable": EXEC_PY, "executor": dict(EXEC_VERSIONS),
        "exec_meta_sha256": file_sha256(out / "exec_meta.json"),
        "gen_meta_sha256": file_sha256(out / "gen_meta.json"), "gid": 1000,
        "items_sha256": file_sha256(items_dir / "items.jsonl"), "limits": dict(EXEC_LIMITS), "python": "3.12.16",
        "scorer_code_sha256": SCORER_CODE, "uid": 1000, "uv_lock_sha256": EXECUTOR_LOCK_SHA, "workers": 8})
    write_json(out / "summary.json", {"items": len(items)})
    for metric, eid, scorer in (("iou", EVALUATOR_ID, SCORER), ("valid", SUPPORT_ID, SUPPORT_SCORER)):
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for r in rows:
            dst = arm / r["output_ref"]["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(out / r["output_ref"]["path"], dst)
            s = r["sub_metrics"]
            sub = {k: s[k] for k in ("executes", "chamfer", "volume_ratio", "bbox_iou", "reward", "n_new_tokens",
                                     "unbalanced_columns") if s.get(k) is not None}
            sub["status_code"] = STATUSES.index(s["status"])
            sub["truncated"] = int(s["finish_reason"] == "length")
            sc = r["score"] if metric == "iou" else float(s["executes"])
            lines.append({"item": r["item"], "repeat": 0, "seed": 11, "score": sc, "error": None,
                          "sub_metrics": sub, "output_ref": r["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        write_json(arm / "eval_meta.json", {
            "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": eid, "adapter": ADAPTER,
            "metric": metric, "direction": "higher_is_better", "score_range": [0, 1],
            "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": None,
            "env_version_sha256": None, "item_set_hash": item_set, "eval_seeds": list(SEEDS), "sampler": SAMPLER,
            "sampler_hash": sha_of(SAMPLER) if model else compact_sha(SAMPLER), "scorer": scorer,
            "framework": {"name": "vllm", "version": "0.29.0"}, "created_at": "2026-11-14T12:00:00Z"})
    return out


# ---------------------------------------------------------------------------
# Spec, registration, records
# ---------------------------------------------------------------------------

REGISTRATION = """registration_version: 1
registration_id: ext-cad-{stage}-fixture-superiority-v1
signed_by: prototype-unsigned
workflow: ext-cad-{stage}-fixture
gate: quality_gate
clause:
  type: superiority
  metric: iou
  direction: higher_is_better
  margin: 0.0
  alpha: 0.025
  power: 0.8
  planning_effect: 0.2
  sigma_plan: fixed
  unit: item
  samples_per_item: 1
  eval_seeds: [11]
  interval: cluster_bootstrap
  blocking: true
margin_justification: Fixture value (superiority, delta_sup = 0).
sigma_pilot:
  sigma: 0.3
  source: Fixture value (n_required = ceil(((1.95996 + 0.84162) x 0.3 / 0.2)^2) = 18 <= 30 items).
baseline_pilot_mean: 0.34
construct_alignment: >-
  iou measures agreement with the reference solid, the construct the OCCT reward also rewards, with another engine
  (numpy volume integration); both share the CadQuery/OCCT kernel and the valid-solid requirement. A pass shows better
  agreement under an independent engine, not a general gain in CAD skill.
eval_distribution:
  item_set_hash: "{item_set_hash}"
sampler: {sampler}
evaluators:
  - id: cad.cadquery_items@0.1
    manifest_hash: "{evaluator_hash}"
multiplicity:
  plan: single_gate
  family: null
analysis_code_version: gatedtrain 0.3.0
bootstrap:
  replicates: 10000
  seed: 20261114
pairing:
  max_unpaired: 0
min_paired: 10
contamination:
  allow_exact_overlap: false
"""

SPEC = """workflow: ext-cad-{stage}-fixture
version: 1
budget: {{gpu_hours: 24}}
artifacts:
  base:         {{type: model_checkpoint, uri: "file:{base}"}}
  train:        {{type: training_dataset, uri: "file:{data}"}}
  items:        {{type: item_set, uri: "file:{items}", disjoint_from: [train]}}
  trainer_cfg:  {{type: trainer_config, uri: "file:trainer.yaml"}}
  registration: {{type: registration, uri: "file:registration.yaml@sha256:{registration_sha256}"}}
stages:
  - id: {stage}
    uses: {uses}
    algorithm: {stage}
    model_family: llm.qwen2_5_coder@0.1
    inputs: {{model: base, data: train, config: trainer_cfg}}
    outputs: {{model: ckpt}}
  - id: quality_gate
    uses: gate.superiority@1.0
    evaluator: cad.cadquery_items@0.1
    sampler: deployment
    inputs: {{model: ckpt, baseline: base, items: items, registration: registration}}
    test: {{metric: iou, alpha: 0.025, power: 0.80, mde: 0.2, cluster_by: item, seeds: 1}}
    on_fail: stop
    owner: skeleton-demo
    escalation: [skeleton-demo-lead]
"""


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding metric `iou`: 20 baseline scores, no value shared by half."""
    rec = {"schema": "gatedtrain.smoke_record", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 6, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 14.0, "host_ram_peak_gb": 20.0, "swap_peak_gb": 2.0, "seconds_per_step": 20.0,
           "projected_gpu_h": 0.45, "gpu_log_total_h": 6.0,
           "baseline_scores": [((i * 7) % 9) / 8 if i % 5 else 0.0 for i in range(1, 21)], "error_count": 0,
           "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


class CadScenario:
    """Both stages of area 6 on disk (module docstring)."""

    def __init__(self, root: Path, *, n_sft_rows: int = 160, n_grpo_rows: int = 48, n_items: int = 30):
        self.root = root
        self.mirror = root / "host"
        self.data_root = self.mirror / DATA_ROOT.lstrip("/")
        self.runtime = self.mirror / AREA6.lstrip("/")
        self.output_root = self.mirror / OUT_ROOT.lstrip("/")
        self.python_tree = self.mirror / PYTHON_TREE.lstrip("/")
        self.host_work = root / "host-work"
        rt = self.runtime
        rt.mkdir(parents=True)
        (rt / "executor").mkdir()
        for name in SCRIPTS:
            (rt / name).write_text(f"# fixture stand-in for {AREA6}/{name}; never executed\n", encoding="utf-8")
        (rt / "executor" / ".venv" / "bin").mkdir(parents=True)
        (rt / "executor" / ".venv" / "bin" / "python").write_text("# executor venv interpreter stand-in\n")
        shutil.copyfile(EXECUTOR_LOCK, rt / "executor" / "uv.lock")
        (self.python_tree / "cpython-3.12.16-linux-x86_64-gnu" / "bin").mkdir(parents=True)
        (self.python_tree / "cpython-3.12.16-linux-x86_64-gnu" / "bin" / "python3.12").write_text("# stand-in\n")
        self.base = build_base(rt / BASE_REL)
        write_json(rt / "models" / "MODELS.json", models_json(self.base))
        self.sft_rows = data_rows("sft", SFT_FAMILIES, n_sft_rows)
        self.grpo_rows = data_rows("grpo", SFT_FAMILIES + HELD_OUT, n_grpo_rows)
        self.sft_data = write_data(rt / "data" / "sft", self.sft_rows)
        self.grpo_data = write_data(rt / "data" / "grpo", self.grpo_rows)
        self.items = {"sft": write_items(rt / "items-sft", "sft", SFT_FAMILIES, n_items),
                      "grpo": write_items(rt / "items-grpo", "grpo", HELD_OUT, n_items)}
        self.items_dir = {"sft": rt / "items-sft", "grpo": rt / "items-grpo"}
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        runs = self.output_root / OUT_NAME
        self.sft_out = build_sft_run(runs / "sft-1", rt)
        self.sft_cut = self.sft_out / "cuts" / "checkpoint-6"
        self.grpo_out = build_grpo_run(runs / "grpo-1" / "train", rt, self.sft_cut)
        self.spool = runs / "grpo-1" / "spool"               # the decided layout: <run>/train and <run>/spool
        self.spool.mkdir()
        self.grpo_cut = self.grpo_out / "cuts" / "checkpoint-4"
        self.base_model = manifest_hash(self.base)
        self.sft_model = deploy_hash(self.sft_cut)
        self.grpo_model = deploy_hash(self.grpo_cut)
        self.item_set_hash = {s: manifest_hash(d) for s, d in self.items_dir.items()}
        ev = runs / "eval"
        self.arms = {
            "sft": (build_score(ev / "sft-base", self.items["sft"], self.items_dir["sft"], baseline_score,
                                model=self.base_model, item_set=self.item_set_hash["sft"], model_dir=self.base),
                    build_score(ev / "sft-cand", self.items["sft"], self.items_dir["sft"], candidate_score,
                                model=self.sft_model, item_set=self.item_set_hash["sft"], model_dir=self.sft_cut)),
            "grpo": (build_score(ev / "grpo-base", self.items["grpo"], self.items_dir["grpo"], baseline_score,
                                 model=self.sft_model, item_set=self.item_set_hash["grpo"],
                                 model_dir=self.sft_cut),
                     build_score(ev / "grpo-cand", self.items["grpo"], self.items_dir["grpo"], candidate_score,
                                 model=self.grpo_model, item_set=self.item_set_hash["grpo"],
                                 model_dir=self.grpo_cut))}
        self.labels = {}
        for stage in ("sft", "grpo"):
            d = root / stage
            d.mkdir()
            (d / "trainer.yaml").write_text("backend: nemo_rl_ext\nprofile: full\n", encoding="utf-8")
            self.render(stage)
            rec = self.launch_record(stage)
            write_canonical(d / "launch.json", rec)
            self.labels[stage] = rec["label"]
            base_model = self.base_model if stage == "sft" else self.sft_model
            write_canonical(d / "smoke.json", smoke_record(base_model=base_model, item_set=self.item_set_hash[stage]))
        self.guard_log()

    # --- paths and environment ---

    def stage_out(self, stage: str) -> Path:
        return self.sft_out if stage == "sft" else self.grpo_out

    def spec_base(self, stage: str) -> Path:
        return self.base if stage == "sft" else self.sft_cut

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.data_root), ENV_MIRROR_ROOT: str(self.mirror),
                ENV_INTERPRETER_TREES: str(self.runtime / "executor" / ".venv")}

    def host_config(self):
        from gatedtrain.ext.launch import host_config
        return host_config(self.env())

    def config_sha(self) -> str:
        return host_config_sha256(self.host_work, self.output_root, [self.data_root], mirror_root=self.mirror,
                                  interpreter_trees=[self.runtime / "executor" / ".venv"])

    def guard_log(self) -> None:
        lines = []
        for stage in ("sft", "grpo"):
            lab = self.labels[stage]
            lines += [f"2026-11-11T09:00:00Z START {lab} attempt=1", f"2026-11-11T10:00:00Z END {lab} attempt=1 rc=0"]
        (self.host_work / "gpu-guard.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    def render(self, stage: str, reg_mutate: Callable[[dict], None] | None = None) -> None:
        d = self.root / stage
        text = REGISTRATION.format(stage=stage, item_set_hash=self.item_set_hash[stage], sampler=json.dumps(SAMPLER),
                                   evaluator_hash=sha_of({"evaluator": EVALUATOR_ID, "metric": "iou",
                                                          "direction": "higher_is_better", "scorer": SCORER}))
        if reg_mutate is not None:
            reg = yaml.safe_load(text)
            reg_mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (d / "registration.yaml").write_text(text, encoding="utf-8")
        spec = SPEC.format(stage=stage, uses=SFT_ID if stage == "sft" else GRPO_ID, base=self.spec_base(stage),
                           data=self.sft_data if stage == "sft" else self.grpo_data, items=self.items_dir[stage],
                           registration_sha256=file_sha256(d / "registration.yaml")[len("sha256:"):])
        (d / "spec.yaml").write_text(spec, encoding="utf-8")

    # --- launch ---

    def mounts(self, stage: str, out: Path, *, spool: Path | None = None) -> list[dict]:
        m = [{"host": str(self.runtime), "container": AREA6, "mode": "ro"}]
        if stage == "grpo":
            m.append({"host": str(self.sft_cut), "container": "/inputs/sft-cut", "mode": "ro"})
        m.append({"host": str(out), "container": "/out", "mode": "rw"})
        if stage == "grpo":
            m.append({"host": str(spool or self.spool), "container": "/spool", "mode": "rw"})
        return m

    def argv(self, stage: str, profile: str = "full") -> list[str]:
        a = [TRAIN_PY, f"{AREA6}/run_cad_nemo.py", "--stage", stage, "--profile", profile, "--out", "/out",
             "--seed", str(SFT_SEED if stage == "sft" else GRPO_SEED), "--base", f"{AREA6}/{BASE_REL}"]
        if stage == "grpo":
            a += ["--init", "/inputs/sft-cut", "--socket", "/spool/cadexec.sock"]
        return a

    def sidecar_mounts(self, spool: Path | None = None) -> list[dict]:
        return [{"host": str(self.python_tree), "container": PYTHON_TREE, "mode": "ro"},
                {"host": str(self.runtime / "executor"), "container": EXECUTOR, "mode": "ro"},
                {"host": str(spool or self.spool), "container": "/spool", "mode": "rw"}]

    def sidecar_request(self, spool: Path | None = None, **overrides) -> dict:
        side = {"runtime": {"kind": "venv_in_container", "image": EXECUTOR_IMAGE,
                            "venv_lock": str(self.runtime / "executor" / "uv.lock")},
                "argv": list(SIDECAR_ARGV), "mounts": self.sidecar_mounts(spool), "env": dict(SIDECAR_ENV),
                "network": "none", "user": run_user(), "tmpfs": ["/tmp"], "read_only": True, "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges"], "gpus": False, "limits": dict(SIDECAR_LIMITS)}
        side.update(overrides)
        return side

    def sidecar_record(self, spool: Path | None = None) -> dict:
        side = self.sidecar_request(spool)
        env = side.pop("env")
        side["env_names"] = sorted(env)
        side["runtime"] = {"kind": "venv_in_container", "image": EXECUTOR_IMAGE, "venv_lock_sha256": EXECUTOR_LOCK_SHA}
        side["interpreter_sha256"] = file_sha256(self.runtime / "executor" / ".venv" / "bin" / "python")
        side["script_sha256"] = file_sha256(self.runtime / "executor" / "cad_sandbox.py")
        return side

    def launch_record(self, stage: str, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full training run (README "Launch request facts")."""
        out = self.stage_out(stage)
        rec = {"schema": "gatedtrain.launch_record", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 6, "profile": "full", "label": f"mm-a6-cad-{stage}-full",
               "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": NEMO_RL_IMAGE, "venv_lock_sha256": None},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SFT_SEED if stage == "sft" else GRPO_SEED, "gpus": True,
               "argv": self.argv(stage), "mounts": self.mounts(stage, out), "tmpfs": ["/tmp"], "network": "none",
               "network_reason": None, "user": run_user(), "env_names": list(ENV_NAMES),
               "gpu_est_h": 0.1 if stage == "sft" else 0.45, "gpu_used_h": 6.0, "output_dir": str(out),
               "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": TRAIN_PY, "interpreter_sha256": None,
               "script_path": f"{AREA6}/run_cad_nemo.py", "script_sha256": file_sha256(self.runtime / "run_cad_nemo.py"),
               "host_config_sha256": self.config_sha()}
        if stage == "grpo":
            rec["sidecar"] = self.sidecar_record()
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, stage: str, **overrides) -> Path:
        """A changed launch record of `stage`, its label re-hashed and logged in gpu-guard.log."""
        rec = self.launch_record(stage, **overrides)
        p = write_canonical(self.root / stage / "launch.json", rec)
        self.labels[stage] = rec["label"]
        self.guard_log()
        return p

    def request(self, stage: str, out: Path, *, spool: Path | None = None, **overrides) -> dict[str, Any]:
        req = {"area": 6, "label": f"mm-a6-cad-{stage}-smoke", "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": NEMO_RL_IMAGE},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SFT_SEED if stage == "sft" else GRPO_SEED, "gpus": True, "network": "none",
               "network_reason": None, "user": run_user(), "env": dict(TRAIN_ENV),
               "mounts": self.mounts(stage, out, spool=spool), "tmpfs": ["/tmp"], "output_dir": str(out),
               "gpu_est_h": 0.11, "argv": self.argv(stage, "smoke")}
        if stage == "grpo":
            req["sidecar"] = self.sidecar_request(spool)
        req.update(overrides)
        return req

    def eval_request(self, script: str, out: Path, *, gen: Path | None = None, exec_dir: Path | None = None,
                     items: str = "items-sft", **overrides) -> dict[str, Any]:
        """A generation (gen_cad.py, NeMo RL image, GPU), execution (executor/cad_sandbox.py --run-batch) or scoring
        (score_cad.py) request; the last two in the executor image on CPU, exactly as executor/run-executor.sh exec|score
        prints them (README "Launch request facts", review-11 layout)."""
        executor_rt = {"kind": "venv_in_container", "image": EXECUTOR_IMAGE,
                       "venv_lock": str(self.runtime / "executor" / "uv.lock")}
        py = {"host": str(self.python_tree), "container": PYTHON_TREE, "mode": "ro"}
        if script == "gen_cad.py":
            req = {"runtime": {"kind": "container", "image": NEMO_RL_IMAGE}, "gpus": True,
                   "argv": [GEN_PY, f"{AREA6}/gen_cad.py", "--backend", "vllm", "--model", f"{AREA6}/{BASE_REL}",
                            "--items", f"{AREA6}/{items}", "--out", "/out", "--eval-seeds", "11"],
                   "mounts": [{"host": str(self.runtime), "container": AREA6, "mode": "ro"},
                              {"host": str(out), "container": "/out", "mode": "rw"}], "gpu_est_h": 0.04,
                   "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "PYTHONUNBUFFERED": "1"}}
        elif script == "cad_sandbox.py":
            req = {"runtime": executor_rt, "gpus": False,
                   "argv": [EXEC_PY, "-I", f"{EXECUTOR}/cad_sandbox.py", "--run-batch", "--gen", "/inputs/gen", "--out",
                            "/out", "--workers", "8"],
                   "mounts": [py, {"host": str(self.runtime / "executor"), "container": EXECUTOR, "mode": "ro"},
                              {"host": str(gen), "container": "/inputs/gen", "mode": "ro"},
                              {"host": str(out), "container": "/out", "mode": "rw"}], "gpu_est_h": 0,
                   "env": dict(SIDECAR_ENV)}
        else:
            req = {"runtime": executor_rt, "gpus": False,
                   "argv": [EXEC_PY, "-I", f"{AREA6}/score_cad.py", "--gen", "/inputs/gen", "--exec", "/inputs/exec",
                            "--items", f"{AREA6}/{items}", "--out", "/out", "--eval-seeds", "11", "--workers", "8"],
                   "mounts": [py, {"host": str(self.runtime), "container": AREA6, "mode": "ro"},
                              {"host": str(gen), "container": "/inputs/gen", "mode": "ro"},
                              {"host": str(exec_dir), "container": "/inputs/exec", "mode": "ro"},
                              {"host": str(out), "container": "/out", "mode": "rw"}], "gpu_est_h": 0,
                   "env": dict(SIDECAR_ENV)}
        req.update({"area": 6, "label": f"mm-a6-cad-{script[:-3].replace('_', '-')}", "host_work": str(self.host_work),
                    "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
                    "training_seed": 0, "network": "none", "network_reason": None, "user": run_user(),
                    "tmpfs": ["/tmp"], "output_dir": str(out)})
        req.update(overrides)
        return req

    def new_out(self, name: str) -> Path:
        p = self.output_root / OUT_NAME / name
        p.mkdir(parents=True)
        return p

    def launch(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
               extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out

    def recheck(self, record: Path):
        return run_cli(["ext-launch", "--adapter", ADAPTER, "--recheck", str(record)], cwd=self.root, env=self.env())

    # --- collect ---

    def collect(self, stage: str, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, timeout: int = 300):
        d = self.root / stage
        base_arm, cand_arm = self.arms[stage]
        args = ["ext-collect", str(d / "spec.yaml"), "--adapter", ADAPTER, "--stage-out", str(self.stage_out(stage)),
                "--launch-record", str(launch or d / "launch.json"), "--smoke-record", str(d / "smoke.json"),
                "--eval-baseline", str(baseline or base_arm / "arm-iou"),
                "--eval-candidate", str(candidate or cand_arm / "arm-iou"), "--runs-root", str(runs_root)]
        return run_cli(args, cwd=d, timeout=timeout, env=self.env())

    def edit_json(self, path: Path, fn: Callable[[Any], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def edit_jsonl(self, path: Path, fn: Callable[[list[dict]], None]) -> None:
        rows = read_jsonl(path)
        fn(rows)
        write_jsonl(path, rows)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in d["files"]}))
