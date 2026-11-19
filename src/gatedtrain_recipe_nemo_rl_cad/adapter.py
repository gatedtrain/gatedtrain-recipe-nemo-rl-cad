"""NeMo RL adapter for area "CAD" (text-to-CadQuery on Qwen2.5-Coder-0.5B-Instruct: NeMo RL SFT, then NeMo RL GRPO with
a custom CadQuery environment whose programs run in the B18 executor sidecar; multimodal spec section 6 row 6 and
the area 6 decisions in runtime/README.md; gatedtrain's tests/README.md section 11.23, EXT-180 to EXT-199).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and the safetensors header (standard library `struct` and `json`), and hashes everything else.

Run layout, read from the area runtime's code `runtime/` (`run_cad_nemo.py`,
`ptc_cad_hooks.py`, `cad_env.py`; README.md "Trainer details", "Outputs", "What gatedtrain needs"; read 2026-11-13) and
NeMo RL 4f39fd8. That code had not run when this was written (it needs the GPU image); where the README and the code
differ the code is followed, and what only a GPU run can show is listed in gatedtrain's docs/findings.md (area 6, "Unverified"):

- one adapter, two stages (MM-15): `sft.nemo_rl@0.1` (run_args.json `stage` "sft", base the Qwen2.5-Coder-0.5B-Instruct
  directory) and `rl.nemo_rl_env@0.1` (stage "grpo", base the SFT run's gated cut, `init_checkpoint`);
- `write_cut` (ptc_cad_hooks.py) builds `cuts/checkpoint-<N>/` through `cuts/.checkpoint-<N>.partial/` after NeMo RL
  publishes `nemo/checkpoints/step_<N>` (checkpoint_storage.py:80-97): hard links to the consolidated weights and the
  tokenizer files NeMo RL saved, `generation_config.json` (the base's), `sampler.json` (gen_cad.py SAMPLER),
  `base_ref.json` (MODELS.json repo, revision and files; `init_checkpoint`), `training_info.json` (NeMo RL's,
  checkpoint.py:406), then `cut.json` last (`global_step`, `files`, `deployable`, `seed`, `profile`, `stage`,
  `script_sha256`, `examples_consumed` = per-step count x step, `nemo_rl_step_dir`). NeMo RL's `step_<N>` directories
  are resume state, never cuts;
- SFT (`install_sft_sample_log`): `samples.jsonl` `{step, epoch, position, idx, row_id, row = sha256(row_id), attempt,
  utc}` and `train_steps.jsonl` `{step, epoch, n_examples, example_ids}`, written before each `policy.train`; `idx` is
  the 0-based line of train.jsonl, which NeMo RL reads in file order (data.shuffle=false, drop_last);
- GRPO (`cad_env.CadEnvCore.score`): `env_calls.jsonl` `{call, position, row_id, status, reward, program_sha256,
  response_sha256, exec_batch_s, utc}` at each reward call; NeMo RL's own `nemo/logs/exp_<n>/train_data_step<N>.jsonl`
  (grpo_sync.py:1427-1460, logger.py:1409-1443) holds one line per sample, `{"rewards": [r], ..., "idx": i}`, r the
  float32 value of the environment's reward (cad_env.py returns a float32 tensor). Step N owns the env-log lines
  128(N-1)+1 .. 128N; the join is the rewards in order;
- `run_args.json` (`write_run_args`): `stage`, `profile`, `seed`, `dry_run`, `max_steps`, `profile_settings`,
  `model_name_or_path`, `base_model`, `init_checkpoint`, `train_file`, `train_file_sha256`, `nemo_rl_overrides`,
  `attempt`, `script_sha256`, `hooks_sha256`, `env_sha256`, `protocol_sha256`, `framework` {name, commit, versions};
  `env-freeze.txt` package lines and `# nemo-rl commit <40 hex>`.

Identity (MM-17): full weights, no composition with the base (MM-18 does not apply). The plain base directory's model
hash is the manifest hash of all its files; a cut's is the manifest hash of its `cut.json` `deployable` files.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.families.nemo_rl import (CHECKPOINT_DIR_RE, CUTS_DIR, CUT_MARKER, EXP_DIR_RE, FREEZE_COMMIT_RE, FREEZE_FILE,
                                 NEMO_LOGS, SHA_RE, TRAIN_DATA_RE)
from gatedtrain.families.nemo_rl import NEMO_RL_IMAGES
from gatedtrain.ext.executor import EXECUTOR_ENV
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain_recipe_nemo_rl_cad.rows import (AREA6_EXECUTOR, AREA6_PYTHON, EXECUTOR_CODE_SHA256, EXECUTOR_LOCK_SHA256,
                                        RUNTIME)   # RUNTIME: the runtime's container path (identity mount)
from gatedtrain.ext.sdk import (compact_sha256 as _compact_sha256, float32, is_int as _int, is_num as _num, norm_path as _norm_path,
                   parse_options, plain_file as _plain_file, read_cut_marker, read_json as _read_json, read_jsonl,
                   read_run_args, safetensors_problem, sha_or_none as _sha_or_none, shards_concat_sha256)

NAME = "nemo_rl_ext"
ROW = TRAINER_ROWS[NAME]
SFT_ID, GRPO_ID = ROW["uses"]
STAGE_OF = {SFT_ID: "sft", GRPO_ID: "grpo"}
LIBRARY = ("NeMo RL 4f39fd8 (Tier B image) with the area runtime's cut writer (ptc_cad_hooks.write_cut after NeMo RL "
           "publishes step_<N>; cut.json written last)")
NEMO_RL_COMMIT = ROW["nemo_rl_commit"]
EXECUTOR_IMAGE = ROW["sidecar"]["image"]
BASE_REL = "models/qwen2.5-coder-0.5b-instruct"
BASE_REPO, BASE_REVISION = "Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
BASE_REQUIRED = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
CUT_SCHEMA = "ptc_mm.cad_cut"
WEIGHTS = "model.safetensors"
REQUIRED_COMPONENTS = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                       "tokenizer_config.json", "sampler.json")   # ptc_cad_hooks.py CUT_DEPLOYABLE
OPTIONAL_COMPONENTS = ("chat_template.jinja", "vocab.json", "merges.txt")   # copied when NeMo RL saved them
COMPONENTS = tuple(ROW["components"])
assert set(COMPONENTS) == set(REQUIRED_COMPONENTS) | set(OPTIONAL_COMPONENTS)
CUT_EXTRA = ("base_ref.json", "training_info.json")
BASE_REF = "base_ref.json"
SAMPLER_FILE = "sampler.json"
PROFILES = ("smoke", "full")
SFT_BATCH, GRPO_PROMPTS, GRPO_GENERATIONS = 32, 16, 8       # README "Trainer details"; run_cad_nemo.py PROFILES
HELD_OUT = ("blind_hole", "plate_4holes", "tube", "flange", "chamfer_box")   # make_data.py HELD_OUT (data_meta.json)
OK = "ok"
FAILURES = tuple(EVALUATOR_ROWS["cad.cadquery_items@0.1"]["score_outcomes"])  # cad_sandbox.py STATUSES besides ok
# cad_sandbox.py STATUSES in order (cad_sandbox.py:94-95): score_cad.py writes `status_code` = the index (0 = ok).
STATUSES = (OK,) + FAILURES
# gen_cad.py `deploy`: the model files whose SHA-256 gen_meta.json `model_files` records.
GEN_MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
REWARD_OK_MIN, REWARD_MAX = 0.2, 1.0                         # cad_protocol.reward: 0.2 + 0.4 x [0, 1] + 0.4 x [0, 1]
REVISION_RE = re.compile(r"[0-9a-f]{40}")
SAMPLES, STEPS, ENV_CALLS = "samples.jsonl", "train_steps.jsonl", "env_calls.jsonl"
# run_args.json SHA-256 of the modules the launched script imports from its own directory (run_cad_nemo.py main).
HELPERS = {"hooks_sha256": "ptc_cad_hooks.py", "env_sha256": "cad_env.py", "protocol_sha256": "executor/cad_protocol.py"}
SHUFFLE_OFF = "data.shuffle=false"
# The SHARED PROMPT block of cad_protocol.py (data_meta.json `prompt_block_sha256`): training rows, items and the
# evaluator's prompts are built from it.
PROMPT_BLOCK_SHA256 = "sha256:b9aea19b65e447006db92eb9e868357f519a37455d59169a778075c18ffd7e69"
# The evaluator's sampler adds these to the cuts' sampler.json (score_cad.py `sampler`).
EVAL_ONLY_SAMPLER_KEYS = ("backend", "prompt_block_sha256")
SCRIPT_TRAIN, SCRIPT_GEN, SCRIPT_SCORE, SCRIPT_EXEC = "run_cad_nemo.py", "gen_cad.py", "score_cad.py", "cad_sandbox.py"
FLAGGED = (SCRIPT_EXEC, SCRIPT_SCORE)                     # `python -I <script>` (executor/run-executor.sh)
SCRIPT_PATHS = {SCRIPT_TRAIN: f"{RUNTIME}/{SCRIPT_TRAIN}", SCRIPT_GEN: f"{RUNTIME}/{SCRIPT_GEN}",
                SCRIPT_SCORE: f"{RUNTIME}/{SCRIPT_SCORE}", SCRIPT_EXEC: f"{AREA6_EXECUTOR}/{SCRIPT_EXEC}"}
# argparse options of the three scripts (run_cad_nemo.py, gen_cad.py, score_cad.py main): option -> takes a value.
TRAIN_OPTIONS = {"--stage": True, "--profile": True, "--out": True, "--seed": True, "--base": True, "--init": True,
                 "--socket": True, "--max-steps": True}
GEN_OPTIONS = {"--backend": True, "--model": True, "--items": True, "--out": True, "--eval-seeds": True, "--limit": True,
               "--round-robin": False, "--batch-size": True, "--threads": True}
SCORE_OPTIONS = {"--gen": True, "--exec": True, "--items": True, "--out": True, "--eval-seeds": True, "--limit": True,
                 "--workers": True, "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True}
EXEC_OPTIONS = {"--serve": True, "--self-test": False, "--run-batch": False, "--gen": True, "--out": True,
                "--probe-xor": True, "--workers": True, "--cpu-s": True, "--wall-s": True}   # cad_sandbox.py main
OPTIONS = {SCRIPT_TRAIN: TRAIN_OPTIONS, SCRIPT_GEN: GEN_OPTIONS, SCRIPT_SCORE: SCORE_OPTIONS, SCRIPT_EXEC: EXEC_OPTIONS}
ITEM_SETS = ("items-sft", "items-grpo")
SFT_CUT_CONTAINER = "/inputs/sft-cut"                         # README "Launch request facts", GRPO trainer
EVAL_CUT_CONTAINER = "/inputs/cut"
GEN_CONTAINER = "/inputs/gen"
EXEC_CONTAINER = "/inputs/exec"
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*)\Z", re.DOTALL | re.IGNORECASE)

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a cuts/checkpoint-<N> directory is a cut iff its cut.json, written last after an atomic rename from "
             "cuts/.checkpoint-<N>.partial, parses with schema ptc_mm.cad_cut, global_step = N, nemo_rl_step_dir step_<N>, "
             "a positive integer examples_consumed, stage sft or grpo, profile smoke or full, and files listing exactly the "
             "other files of the directory (the six required components, the tokenizer files NeMo RL saved, base_ref.json "
             "and training_info.json), each with its SHA-256; deployable lists exactly the components present; "
             "model.safetensors is whole; NeMo RL's nemo/checkpoints/step_<N> (and tmp_step_<N>) are resume state, never "
             "cuts; the cut's last step included is N (SFT optimizer step, GRPO step)"),
    "complete_dir_pattern": f"{CUTS_DIR}/{CHECKPOINT_DIR_RE.pattern}",
    "sources": ["runtime/ptc_cad_hooks.py write_cut, install_cut_writer (cut.json "
                "last, os.rename)",
                "nemo-rl 4f39fd8 nemo_rl/utils/checkpoint_storage.py:80-97 (publish: tmp_step_N -> step_N)",
                "nemo-rl 4f39fd8 nemo_rl/utils/checkpoint.py:406 (training_info.json)",
                "runtime/README.md \"What gatedtrain needs\" item 1"],
    "verified": False,                   # the hooks are untested until the GPU smoke (README "CPU sanity check")
}
SEED_DERIVATION = ("run_cad_nemo.py passes the launch seed as NeMo RL sft.seed / grpo.seed (set_seed in NeMo RL's setup); "
                   "the data order is the file order (data.shuffle=false), observed per row in samples.jsonl (SFT) and "
                   "env_calls.jsonl (GRPO); vLLM's sampling seed is whatever NeMo RL derives and is not claimed. The "
                   "training seed is run_args.json `seed`, which must equal the launch record's training_seed and every "
                   "cut's cut.json `seed`. Evaluation is greedy (K = 1; eval seed 11 is recorded, nothing is drawn); "
                   "05 section 14 H(run_seed, step, data_ref, sample_index) is not claimed.")


def extract_program(text: Any) -> str | None:
    """cad_protocol.extract_program (cad_protocol.py :52-68): the first complete fenced block, else an unterminated
    final fence, else None; stripped of newlines, None when blank."""
    if not isinstance(text, str):
        return None
    m = _FENCE.search(text) or _OPEN_FENCE.search(text)
    if not m:
        return None
    code = m.group(1).strip("\n")
    return code if code.strip() else None



def _shards(data: Path) -> list[tuple[str, list[dict]]]:
    """[(shard name, rows)] of the spec's training shards in file-name order (the concatenation order of train.jsonl).
    Raises OSError; ValueError when a shard does not parse."""
    from gatedtrain.ledger import shard_table

    out = []
    for name, _ in shard_table(data):
        rows, why = read_jsonl(Path(data) / name)
        if rows is None:
            raise ValueError(f"training shard {name}: {why}")
        out.append((name, rows))
    return out


def _stage_of_data(data: Path) -> str | None:
    """`sft` or `grpo` for the runtime's `data/<stage>/shards` layout, else None."""
    data = Path(data)
    return data.parent.name if data.name == "shards" and data.parent.name in ("sft", "grpo") else None


class NemoRlExtAdapter:
    """The MM-12 interface for NeMo RL SFT and GRPO runs of the area runtime (EXT-180 to EXT-199)."""

    name = NAME
    trainer_id = SFT_ID
    trainer_ids = (SFT_ID, GRPO_ID)                     # collect accepts either `uses` (MM-15)
    evaluator_ids = tuple(ROW["evaluators"])
    framework_name = ROW["framework"]
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(ROW["spec_binding"])
    wants_uses = True                                   # checks() receives the spec's `uses` (stage binding)
    arm_needs_items_dir = True                          # arm_problems() receives the item set and evaluated model
    consumption_fix = ("collect the run's own samples.jsonl and train_steps.jsonl (SFT) or env_calls.jsonl and NeMo "
                       "RL's train_data_step<N>.jsonl (GRPO) as the run wrote them (MM-23, MM-27)")

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- the base directory and cuts (MM-17, MM-20; EXT-181, EXT-182) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is an HF causal-LM directory: its configs and tokenizer files and whole weights."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not a model directory)"
        return safetensors_problem(Path(d) / WEIGHTS)

    def _cut_files_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        if (doc.get("schema"), doc.get("schema_version")) != (CUT_SCHEMA, "1.0"):
            return f"{CUT_MARKER} schema is {doc.get('schema')!r} {doc.get('schema_version')!r}, not {CUT_SCHEMA} 1.0"
        files = doc["files"]
        allowed = set(REQUIRED_COMPONENTS) | set(OPTIONAL_COMPONENTS) | set(CUT_EXTRA)
        unknown = sorted(set(files) - allowed)
        if unknown:
            return f"{CUT_MARKER} lists {unknown}, which write_cut never writes (a cut holds {sorted(allowed)})"
        missing = sorted((set(REQUIRED_COMPONENTS) | set(CUT_EXTRA)) - set(files))
        if missing:
            return f"{CUT_MARKER} does not list {missing}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*files, CUT_MARKER]):
            extra = sorted(set(present) - set(files) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in sorted(files):
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        deploy = sorted(n for n in files if n in COMPONENTS)
        if not (isinstance(doc.get("deployable"), list) and all(isinstance(x, str) for x in doc["deployable"])
                and sorted(doc["deployable"]) == deploy):                       # review 11 L1
            return f"{CUT_MARKER} deployable is {doc.get('deployable')!r}, not the components present {deploy}"
        gs = doc.get("global_step")
        if not (_int(gs) and gs >= 1):
            return f"{CUT_MARKER} global_step is {gs!r}, not a positive integer"
        if doc.get("nemo_rl_step_dir") != f"step_{gs}":
            return (f"{CUT_MARKER} nemo_rl_step_dir is {doc.get('nemo_rl_step_dir')!r}, not step_{gs} (the NeMo RL "
                    "checkpoint the cut was built from)")
        if not (_int(doc.get("examples_consumed")) and doc["examples_consumed"] >= 1):
            return f"{CUT_MARKER} examples_consumed is {doc.get('examples_consumed')!r}, not a positive integer"
        if doc.get("stage") not in ("sft", "grpo"):
            return f"{CUT_MARKER} stage is {doc.get('stage')!r}, not sft or grpo"
        if doc.get("profile") not in PROFILES:
            return f"{CUT_MARKER} profile is {doc.get('profile')!r}, not smoke or full"
        if not _int(doc.get("seed")):
            return f"{CUT_MARKER} seed is {doc.get('seed')!r}, not an integer"
        return safetensors_problem(Path(d) / WEIGHTS)

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (it holds cut.json) or a plain base model directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step} (the directory's step)"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `cuts/checkpoint-<N>` directory, complete or not, in step order; `.checkpoint-<N>.partial` and NeMo RL's
        `nemo/checkpoints/step_<N>` are never one."""
        cuts = Path(stage_out) / CUTS_DIR
        if cuts.is_symlink() or not cuts.is_dir():
            return []
        out = []
        for p in cuts.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in step order, incomplete [path]) (EXT-181)."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _base_entries(self, base: Path) -> list[dict]:
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-182 (MM-17). The base alone: `model_hash` = `cut_hash` = its manifest hash (every file). A cut: `model_hash`
        over its deployable files, `cut_hash` over the whole cut; full weights, so `base` is not used, `base_ref` null."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        comps = [e for e in entries if e["path"] in COMPONENTS]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    def needs_base(self, d: Path) -> bool:
        """ext-arm-fill: full weights, never composed with a base (MM-18 does not apply)."""
        return False

    # --- consumption (MM-23, MM-27; EXT-185, EXT-186) ---

    def _common(self, stage_out: Path, data: Path, cuts: list, stage: str) -> tuple[dict | None, list, list, list[str]]:
        """(run_args, [(shard, row index in shard, row)] in file order, [(global_step, examples_consumed, name)],
        problems) shared by both stages."""
        args, why = read_run_args(stage_out)
        if args is None:
            return None, [], [], [why]
        if not cuts:
            return None, [], [], ["no complete cut"]
        if args.get("stage") != stage:
            return None, [], [], [f"run_args.json stage is {args.get('stage')!r}, not {stage!r}"]
        try:
            shards = _shards(data)
        except (OSError, ValueError) as exc:
            return None, [], [], [f"the spec's training shards cannot be read ({exc})"]
        rows = [(name, k, r) for name, rs in shards for k, r in enumerate(rs)]
        if not rows:
            return None, [], [], ["the spec's training shards hold no rows"]
        ids = [r.get("row_id") for _, _, r in rows]
        if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
            return None, [], [], ["the training rows have no unique non-empty string row_id each"]
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return None, [], [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("global_step"), doc.get("examples_consumed"), Path(d).name))
        return args, rows, marks, []

    def _sft_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[tuple[dict, int, str]], list[str]]:
        """([(samples.jsonl line, global row index, shard)] of the steps up to the last cut, in file order) and the
        problems that refuse the logs. Every field the sample-log hook derives is re-derived (README item 6)."""
        args, rows, marks, out = self._common(stage_out, data, cuts, "sft")
        if args is None:
            return [], out
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        if prof.get("batch") != SFT_BATCH or not _int(prof.get("batch")):
            return [], [f"run_args.json profile_settings.batch is {prof.get('batch')!r}, not {SFT_BATCH} (README "
                        "\"Trainer details\": global batch 32)"]
        attempt, max_steps = args.get("attempt"), args.get("max_steps")
        if not (_int(attempt) and attempt >= 1) or not (_int(max_steps) and max_steps >= 1):
            return [], [f"run_args.json attempt {attempt!r} or max_steps {max_steps!r} is not a positive integer"]
        held = sorted({r.get("family") for _, _, r in rows if r.get("family") in HELD_OUT})
        if held:
            return [], [f"the SFT training rows hold the held-out families {held} (runtime/README.md \"Decisions\": "
                        "five families are held out of SFT; the split is part of the spec)"]
        spe = len(rows) // SFT_BATCH                      # drop_last (sft.py:353)
        if spe < 1:
            return [], [f"{len(rows)} training rows do not fill one batch of {SFT_BATCH}"]
        samples, why = read_jsonl(Path(stage_out) / SAMPLES)
        if samples is None:
            return [], [why]
        steps, why = read_jsonl(Path(stage_out) / STEPS)
        if steps is None:
            return [], [why]
        out = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        nums = [st.get("step") for st in steps]
        if not all(_int(k) for k in nums) or nums != list(range(1, len(steps) + 1)):
            return [], [f"{STEPS} steps {nums[:10]}{' ...' if len(nums) > 10 else ''} are not consecutive from 1"]
        want_ids: dict[int, list[int]] = {}
        for st in steps:
            k = st["step"]
            start = ((k - 1) % spe) * SFT_BATCH
            idx = list(range(start, start + SFT_BATCH))
            want_ids[k] = idx
            xs = st.get("example_ids")
            if st.get("epoch") != (k - 1) // spe or not _int(st.get("epoch")):
                if bad(f"{STEPS} step {k} is in epoch {st.get('epoch')!r}, not {(k - 1) // spe} ({spe} steps per epoch: "
                       f"{len(rows)} rows at batch {SFT_BATCH}, drop_last)"):
                    return [], out
            if not (isinstance(xs, list) and all(isinstance(x, str) for x in xs)):
                if bad(f"{STEPS} step {k} has no list of string example_ids"):
                    return [], out
                continue
            if len(xs) != SFT_BATCH or st.get("n_examples") != len(xs) or not _int(st.get("n_examples")):
                if bad(f"{STEPS} step {k} applies {len(xs)} examples (n_examples {st.get('n_examples')!r}), not 32"):
                    return [], out
                continue
            if xs != [rows[i][2]["row_id"] for i in idx]:
                if bad(f"{STEPS} step {k} applies {xs[:2]}..., not rows {start} .. {start + SFT_BATCH - 1} of the "
                       "training file in file order (data.shuffle=false)"):
                    return [], out
        if out:
            return [], out
        if len(steps) != max_steps:                       # review 9 L2: NeMo RL stops at max_num_steps
            return [], [f"{STEPS} logs {len(steps)} steps, not run_args.json max_steps {max_steps}"]
        by_step: dict[int, list[tuple[int, dict]]] = {}
        for n, x in enumerate(samples, start=1):
            k = x.get("step")
            if not (_int(k) and k in want_ids):
                if bad(f"{SAMPLES} line {n}: step {k!r} is not a step {STEPS} logs"):
                    return [], out
                continue
            by_step.setdefault(k, []).append((n, x))
        plan: list[tuple[dict, int, str]] = []
        last = max(m[0] for m in marks)
        for k, idx in want_ids.items():
            lines = by_step.get(k, [])
            ids = [x.get("row_id") for _, x in lines]
            if ids != [rows[i][2]["row_id"] for i in idx]:
                if bad(f"{SAMPLES} has {len(lines)} lines for step {k} whose row_ids are not the step's {STEPS} "
                       "example_ids in order"):
                    return [], out
                continue
            for pos, ((n, x), i) in enumerate(zip(lines, idx)):
                where = f"{SAMPLES} line {n} (step {k})"
                if not (_int(x.get("idx")) and x["idx"] == i):
                    bad(f"{where}: idx {x.get('idx')!r} is not {i}, the row's 0-based line in train.jsonl")
                if not (_int(x.get("position")) and x["position"] == pos):
                    bad(f"{where}: position {x.get('position')!r} is not {pos}")
                if not (_int(x.get("epoch")) and x["epoch"] == (k - 1) // spe):
                    bad(f"{where}: epoch {x.get('epoch')!r}, not {(k - 1) // spe}")
                if x.get("row") != sha256_bytes(str(x.get("row_id")).encode("utf-8")):
                    bad(f"{where}: row is not sha256(row_id)")
                if x.get("attempt") != attempt or not _int(x.get("attempt")):
                    bad(f"{where}: attempt {x.get('attempt')!r} is not run_args.json attempt {attempt!r}")
                if len(out) >= 8:
                    return [], out
                if k <= last:
                    plan.append((x, i, rows[i][0]))
        for gs, consumed, name in marks:                  # MM-27
            if not (_int(gs) and gs <= len(steps)):
                bad(f"{name}: global_step {gs!r} is beyond the {len(steps)} steps {STEPS} logs")
            elif consumed != SFT_BATCH * gs:
                bad(f"{name} cut.json examples_consumed is {consumed!r}, not {SFT_BATCH} x {gs}")
        return ([], out) if out else (plan, [])

    def _grpo_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[tuple[dict, int, int, str, int]], list[str]]:
        """([(env_calls.jsonl line, step, member, shard, global row index)] of the steps up to the last cut) and the
        problems that refuse the logs (README item 6: the step's env-log lines joined with NeMo RL's step log by rewards
        in order; 16 rows of the epoch in file order, 8 samples each)."""
        args, rows, marks, out = self._common(stage_out, data, cuts, "grpo")
        if args is None:
            return [], out
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        for key, want in (("prompts", GRPO_PROMPTS), ("generations", GRPO_GENERATIONS)):
            if not (_int(prof.get(key)) and prof[key] == want):
                return [], [f"run_args.json profile_settings.{key} is {prof.get(key)!r}, not {want} (README \"Trainer "
                            "details\": 16 prompts x 8 generations per step)"]
        max_steps = args.get("max_steps")
        if not (_int(max_steps) and max_steps >= 1):
            return [], [f"run_args.json max_steps {max_steps!r} is not a positive integer"]
        per_step = GRPO_PROMPTS * GRPO_GENERATIONS
        spe = len(rows) // GRPO_PROMPTS
        if spe < 1:
            return [], [f"{len(rows)} training rows do not fill one step of {GRPO_PROMPTS} prompts"]
        index_of = {r["row_id"]: i for i, (_, _, r) in enumerate(rows)}
        env, why = read_jsonl(Path(stage_out) / ENV_CALLS)
        if env is None:
            return [], [why]
        out = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        if len(env) != per_step * max_steps:
            return [], [f"{ENV_CALLS} has {len(env)} lines, not {per_step} x run_args.json max_steps {max_steps} (one "
                        "line per sample; synchronous GRPO without dynamic sampling scores 128 samples per step)"]
        call, pos = 0, 0
        for n, x in enumerate(env, start=1):
            c, p, rid, st, rw = (x.get(k) for k in ("call", "position", "row_id", "status", "reward"))
            where = f"{ENV_CALLS} line {n}"
            if not (_int(c) and _int(p)):
                if bad(f"{where}: call {c!r} or position {p!r} is not an integer"):
                    return [], out
                continue
            if c == call + 1 and p == 0:
                call, pos = c, 0
            elif c == call and p == pos + 1:
                pos = p
            else:
                if bad(f"{where}: call {c} position {p} does not follow call {call} position {pos} (calls are "
                       "numbered from 1, positions from 0 within a call)"):
                    return [], out
                call, pos = c, p
            if not (isinstance(rid, str) and rid in index_of):
                bad(f"{where}: row_id {rid!r} is not a row of the training shards")
            if st != OK and st not in FAILURES:
                bad(f"{where}: status {st!r} is not a cad_sandbox.py status")
            if not _num(rw):
                bad(f"{where}: reward {rw!r} is not a finite number")
            elif st != OK and rw != 0:
                bad(f"{where}: status {st} with reward {rw!r}; cad_protocol.reward is 0 unless the program yields a valid "
                    "solid")
            elif st == OK and not (REWARD_OK_MIN <= rw <= REWARD_MAX):
                bad(f"{where}: status ok with reward {rw!r}, outside [0.2, 1] (cad_protocol.reward)")
            prog = x.get("program_sha256")
            if (prog is None) != (st == "no_code") or (prog is not None and not (isinstance(prog, str)
                                                                                   and SHA_RE.fullmatch(prog))):
                bad(f"{where}: program_sha256 {prog!r} with status {st!r}: a response without a program is exactly "
                    "status no_code")
            resp = x.get("response_sha256")
            if not (isinstance(resp, str) and SHA_RE.fullmatch(resp)):
                bad(f"{where}: response_sha256 {resp!r} is not a SHA-256")
            if len(out) >= 8:
                return [], out
        if out:
            return [], out
        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        try:
            exps = sorted(p for p in logs.iterdir() if p.is_dir() and EXP_DIR_RE.fullmatch(p.name)) if logs.is_dir() else []
        except OSError as exc:
            return [], [f"{'/'.join(NEMO_LOGS)} cannot be listed ({exc.__class__.__name__})"]
        if len(exps) != 1:
            return [], [f"{'/'.join(NEMO_LOGS)} holds {[p.name for p in exps]}, not one exp_<n> directory (a rerun "
                        "starts from an empty output directory)"]
        exp = exps[0]
        try:
            names = {int(m.group(1)): p for p in exp.iterdir() if (m := TRAIN_DATA_RE.fullmatch(p.name))}
        except OSError as exc:
            return [], [f"{exp.name} cannot be listed ({exc.__class__.__name__})"]
        beyond = sorted(k for k in names if not 1 <= k <= max_steps)
        if beyond:
            return [], [f"{exp.name} holds train_data_step{beyond[0]}.jsonl, beyond run_args.json max_steps {max_steps}"]
        plan: list[tuple[dict, int, int, str, int]] = []
        last = max(m[0] for m in marks)
        for step in range(1, max_steps + 1):
            fname = f"train_data_step{step}.jsonl"
            if step not in names:
                if bad(f"{exp.name}/{fname} is missing (NeMo RL logs every step, grpo_sync.py:1459)"):
                    return [], out
                continue
            logged, why = read_jsonl(names[step])
            if logged is None:
                if bad(f"{exp.name}/{why}"):
                    return [], out
                continue
            lines = env[per_step * (step - 1):per_step * step]
            if len(logged) != per_step:
                if bad(f"{exp.name}/{fname} logs {len(logged)} samples, not {per_step}"):
                    return [], out
                continue
            for i, (got, x) in enumerate(zip(logged, lines)):
                r = got.get("rewards")
                if not (isinstance(r, list) and len(r) == 1 and _num(r[0])):
                    bad(f"{exp.name}/{fname} line {i + 1}: rewards {r!r} is not one finite number")
                elif got.get("idx") != i or not _int(got.get("idx")):
                    bad(f"{exp.name}/{fname} line {i + 1}: idx {got.get('idx')!r} is not {i}")
                elif float32(x["reward"]) != r[0]:
                    bad(f"step {step} sample {i}: {ENV_CALLS} reward {x['reward']!r} (float32 {float32(x['reward'])!r}) "
                        f"is not {exp.name}/{fname} rewards {r[0]!r}: the environment's log and NeMo RL's step log do "
                        "not join in order")
                if len(out) >= 8:
                    return [], out
            start = ((step - 1) % spe) * GRPO_PROMPTS
            want = [rows[i][2]["row_id"] for i in range(start, start + GRPO_PROMPTS)]
            counts: dict[str, int] = {}
            for x in lines:
                counts[x["row_id"]] = counts.get(x["row_id"], 0) + 1
            if sorted(counts) != sorted(want):
                if bad(f"step {step} scores rows {sorted(counts)[:3]}..., not rows {start} .. {start + GRPO_PROMPTS - 1} "
                       "of the training file in file order (data.shuffle=false, drop_last)"):
                    return [], out
                continue
            off = sorted(rid for rid, n in counts.items() if n != GRPO_GENERATIONS)
            if off:
                if bad(f"step {step}: row {off[0]} is scored {counts[off[0]]} times, not 8 times (num_generations)"):
                    return [], out
                continue
            if step <= last:
                seen: dict[str, int] = {}
                for x in lines:
                    member = seen.get(x["row_id"], 0)
                    seen[x["row_id"]] = member + 1
                    gi = index_of[x["row_id"]]
                    plan.append((x, step, member, rows[gi][0], gi))
        for gs, consumed, name in marks:                  # MM-27
            if not (_int(gs) and gs <= max_steps):
                bad(f"{name}: global_step {gs!r} is beyond run_args.json max_steps {max_steps}")
            elif consumed != GRPO_PROMPTS * gs:
                bad(f"{name} cut.json examples_consumed is {consumed!r}, not {GRPO_PROMPTS} prompts x {gs}")
        return ([], out) if out else (plan, [])

    def _stage(self, stage_out: Path) -> str | None:
        args, _ = read_run_args(stage_out)
        return args.get("stage") if isinstance(args, dict) and args.get("stage") in ("sft", "grpo") else None

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-185, EXT-186: why the observed logs cannot give the per-row consumption up to the last cut."""
        stage = self._stage(stage_out)
        if stage is None:
            args, why = read_run_args(stage_out)
            return [why or f"run_args.json stage is {args.get('stage')!r}, not sft or grpo"]
        return (self._sft_plan if stage == "sft" else self._grpo_plan)(stage_out, data, cuts)[1]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed`. SFT: one record per samples.jsonl line, `sample_id` = H(run, attempt,
        epoch, step, row_id). GRPO: one record per env_calls.jsonl line, `rewards` = [the environment's reward],
        `sample_id` = H(run, attempt, step, row_id, member) (member: the sample's place among its row's 8 in the step).
        `data_ref.row` = sha256(row_id), shard = the shard holding the row (NUL-joined text keys)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        stage = self._stage(stage_out)
        plan_fn = self._sft_plan if stage == "sft" else self._grpo_plan
        plan, bad = plan_fn(stage_out, data, cuts)
        if bad or stage is None:                          # collect refuses these before writing
            raise ValueError("; ".join(bad) or "run_args.json has no stage")
        check_precision("row", "observed")
        snapshot = manifest_hash(data)
        rows = []
        for entry in plan:
            if stage == "sft":
                x, _, shard = entry
                step, rid, rewards = x["step"], x["row_id"], None
                key = (run_id, attempt_id, x["epoch"], step, rid)
            else:
                x, step, member, shard, _ = entry
                rid, rewards = x["row_id"], [float(x["reward"])]
                key = (run_id, attempt_id, step, rid, member)
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step,
                "sample_id": "sha256:" + hashlib.sha256("\x00".join(str(v) for v in key).encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": sha256_bytes(rid.encode("utf-8")),
                             "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": rewards, "sampler_config_hash": None, "seed_inputs": None,
                "chunk": None, "precision": "row", "derivation": "observed", "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        how = (f"{SAMPLES} logs every example of each optimizer step before policy.train and {STEPS} the step's example "
               "ids (the wrapped prepare_sft_batch, MM-23, MM-27)" if stage == "sft" else
               f"{ENV_CALLS} logs every sample at reward time and NeMo RL's train_data_step<N>.jsonl the step's rewards in "
               "the same order (the join holds for every step; MM-23, MM-27)")
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{how}: {len(rows)} records of steps 1 .. {last}, data_ref.row = sha256(row_id) in the shard "
                           "holding the row, derivation observed; no sample is filtered")}

    def extra_read_outputs(self, stage_out: Path) -> list[Path]:
        """EXT-64: NeMo RL's step logs, which collect reads besides READ_OUTPUTS; each must postdate the launch."""
        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        out = []
        try:
            for exp in sorted(logs.iterdir()) if logs.is_dir() else []:
                if exp.is_dir() and EXP_DIR_RE.fullmatch(exp.name):
                    out += sorted(p for p in exp.iterdir() if TRAIN_DATA_RE.fullmatch(p.name))
        except OSError:
            pass                                          # the consumption check reports an unlistable directory
        return out

    # --- item sets and exact overlap (MM-11; README item 7; EXT-189) ---

    @staticmethod
    def item_overlap_problems(item_rows: list[dict], data: Path) -> list[str]:
        """README item 7 and the decided split: the spec's data is `<runtime>/data/<stage>/shards`; every training row of
        both stages has `text` = `input` (NeMo RL reads `input`, the overlap is judged on `text`); no item's `text` is a
        training row's of either stage (the chain trained on both); the SFT gate's items come from the SFT families, the
        GRPO gate's from families that never occur in the SFT data. Problems (each refuses); OSError propagates."""
        stage = _stage_of_data(data)
        if stage is None:
            return [f"the spec's training data {data} is not <runtime>/data/sft/shards or <runtime>/data/grpo/shards"]
        out: list[str] = []
        texts: dict[str, str] = {}
        families: dict[str, set] = {}
        for other in ("sft", "grpo"):
            d = Path(data).parent.parent / other / "shards"
            if not d.is_dir():
                out.append(f"{d} is missing: the item set is checked against both training sets of the chain")
                continue
            try:
                shards = _shards(d)
            except ValueError as exc:
                out.append(str(exc))
                continue
            families[other] = set()
            for name, rows in shards:
                for n, r in enumerate(rows, start=1):
                    if r.get("text") != r.get("input") or not isinstance(r.get("text"), str):
                        out.append(f"data/{other}/shards/{name} line {n} ({r.get('row_id')!r}): `text` is not its "
                                   "`input` (NeMo RL reads `input`; the exact overlap is judged on `text`)")
                        continue
                    texts.setdefault(r["text"], f"data/{other}/shards/{name} line {n}")
                    families[other].add(r.get("family"))
        out = out[:8] + ([f"{len(out) - 8} more row problems"] if len(out) > 8 else [])
        hits = [(r.get("item"), texts[r["text"]]) for r in item_rows if isinstance(r.get("text"), str)
                and r["text"] in texts]
        if hits:
            out.append(f"{len(hits)} item(s) have a `text` equal to a training row's: {hits[0][0]!r} = {hits[0][1]}"
                       f"{' ...' if len(hits) > 1 else ''} (README item 7: both training sets of the chain)")
        sft_fams = families.get("sft")
        if sft_fams is not None:
            if stage == "sft":
                off = sorted({str(r.get("family")) for r in item_rows if r.get("family") not in sft_fams})
                if off:
                    out.append(f"SFT gate items of families {off} that the SFT data does not train (the SFT gate's items "
                               "come from the 10 SFT families)")
            else:
                off = sorted({str(r.get("family")) for r in item_rows if r.get("family") in sft_fams})
                if off:
                    out.append(f"GRPO gate items of families {off}, which occur in the SFT data (the GRPO gate's families "
                               "are held out of SFT; runtime/README.md \"Decisions\")")
        return out

    # --- arms (README item 8; EXT-194) ---

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict], items_dir: Path | None = None,
                     model_dir: Path | None = None) -> list[str]:
        """EXT-194 (README "Outputs", item 8; score_cad.py sha256 cffdd9ea...). Per record without an error: the
        sub_metrics carry `status_code` (an index into cad_sandbox.STATUSES, 0 = ok), `executes` and `truncated` (0 or 1);
        `executes` is 1 exactly for status_code 0; a failed program scores 0 in the iou arm, the valid arm scores
        `executes`. Every referenced output (`outputs/<item>__r<k>.json`) names its record's item, repeat and seed, its
        status is STATUSES[status_code], and its `program` is cad_protocol.extract_program of its `response`. Beside the
        arm (the scoring output root): `env-runtime.json` names the registered items.jsonl, the pinned executor lock, the
        arm's scorer code and the SHA-256 of `gen_meta.json`, which is there too; `gen_meta.json` names the same items, the
        SHARED PROMPT, the arm's sampler (with its backend) and framework, and, with `model_dir` (the model collect expects
        the arm to have evaluated), exactly that directory's deployable files with their SHA-256 (and its cut.json)."""
        arm = Path(arm)
        out: list[str] = []
        meta, _ = _read_json(arm / "eval_meta.json")
        meta = meta if isinstance(meta, dict) else {}
        metric = meta.get("metric")
        known = {r.get("item") for r in item_rows}
        statuses: dict[str, Any] = {}                    # exec id `<item>__s<seed>` -> the output's status
        for rec in records:
            ref, item = rec.get("output_ref"), rec.get("item")
            if item not in known:
                continue                                  # an unknown item: the gate reports it
            if not (isinstance(ref, dict) and isinstance(ref.get("path"), str)):
                out.append(f"item {item!r} has no output_ref (score_cad.py writes one per response)")
                continue
            where = f"output {ref['path']}"
            sub = rec.get("sub_metrics") if isinstance(rec.get("sub_metrics"), dict) else None
            code = sub.get("status_code") if sub is not None else None
            if rec.get("error") is None:
                score = rec.get("score")
                if sub is None or not (_int(code) and 0 <= code < len(STATUSES)):
                    out.append(f"{where}: sub_metrics status_code {code!r} is not an index into cad_sandbox.py "
                               f"STATUSES (0 .. {len(STATUSES) - 1})")
                elif sub.get("executes") != (1 if code == 0 else 0) or sub.get("truncated") not in (0, 1):
                    out.append(f"{where}: executes {sub.get('executes')!r} and truncated {sub.get('truncated')!r} with "
                               f"status_code {code}: executes is 1 exactly for status ok (0), truncated is 0 or 1")
                elif metric == "valid" and score != sub["executes"]:
                    out.append(f"{where}: the valid arm scores {score!r}, not executes {sub['executes']}")
                elif metric == "iou" and code != 0 and score != 0:
                    out.append(f"{where}: status {STATUSES[code]} but iou {score!r}; any outcome other than a valid solid "
                               "scores 0 (MM-4)")
            doc, why = _read_json(arm / ref["path"])
            if isinstance(doc, dict):
                statuses[f"{item}__s{rec.get('seed')}"] = doc.get("status")
            if not isinstance(doc, dict):
                out.append(f"{where}: {why or 'not a JSON object'}")
            elif doc.get("item") != item or doc.get("repeat") != rec.get("repeat") or doc.get("seed") != rec.get("seed"):
                out.append(f"{where} names item {doc.get('item')!r} repeat {doc.get('repeat')!r} seed {doc.get('seed')!r}, "
                           f"not the record's {item!r} {rec.get('repeat')!r} {rec.get('seed')!r}")
            elif doc.get("program") != extract_program(doc.get("response")):
                out.append(f"{where}: program is not cad_protocol.extract_program of its response (README item 8)")
            elif _int(code) and 0 <= code < len(STATUSES) and doc.get("status") != STATUSES[code]:
                out.append(f"{where}: status {doc.get('status')!r} is not STATUSES[{code}] = {STATUSES[code]}")
            if len(out) >= 8:
                return out
        root = arm.parent
        rt, why = _read_json(root / "env-runtime.json")
        if not isinstance(rt, dict):
            out.append(f"the scoring run's env-runtime.json beside the arm: {why or 'not an object'} (score_cad.py writes "
                       "it next to arm-iou/ and arm-valid/)")
            return out
        want_items = _sha_or_none(Path(items_dir) / "items.jsonl") if items_dir is not None else None
        if items_dir is not None and rt.get("items_sha256") != want_items:
            out.append(f"env-runtime.json items_sha256 {rt.get('items_sha256')!r} is not the registered item set's "
                       f"items.jsonl {want_items}: the scoring run read another item set (README item 8)")
        if rt.get("uv_lock_sha256") != EXECUTOR_LOCK_SHA256:
            out.append(f"env-runtime.json uv_lock_sha256 {rt.get('uv_lock_sha256')!r} is not the executor lock "
                       f"{EXECUTOR_LOCK_SHA256} (README \"Pins\")")
        scorer = meta.get("scorer") if isinstance(meta.get("scorer"), dict) else {}
        if rt.get("scorer_code_sha256") != scorer.get("code_sha256"):
            out.append(f"env-runtime.json scorer_code_sha256 {rt.get('scorer_code_sha256')!r} is not the arm's scorer "
                       f"code_sha256 {scorer.get('code_sha256')!r}")
        code = rt.get("code_sha256") if isinstance(rt.get("code_sha256"), dict) else {}
        if {k: code.get(k) for k in EXECUTOR_CODE_SHA256} != EXECUTOR_CODE_SHA256:
            out.append(f"env-runtime.json code_sha256 names executor code {[code.get(k) for k in EXECUTOR_CODE_SHA256]}, "
                       f"not the pinned {list(EXECUTOR_CODE_SHA256.values())} (executor/cad_sandbox.py, cad_protocol.py)")
        elif rt.get("scorer_code_sha256") != _compact_sha256(code):
            out.append("env-runtime.json scorer_code_sha256 is not the hash of its code_sha256")
        out += self._gen_meta_problems(root, rt, meta, want_items if items_dir is not None else None, model_dir)
        out += self._exec_meta_problems(root, rt, statuses)
        return out

    def _exec_meta_problems(self, root: Path, rt: dict, statuses: dict[str, Any]) -> list[str]:
        """EXT-199: the execution step's record (exec_meta.json of `cad_sandbox.py --run-batch`), beside the arm: its
        SHA-256 is env-runtime.json `exec_meta_sha256`; schema ptc_mm.cad_exec; it ran the pinned executor code, on the
        generation the arm names (`gen_meta_sha256`), with the versions and limits the scoring record states; every
        scored response `<item>__s<seed>` is one of its ids, with a `result.json` and, exactly when the status is ok, a
        `mesh.stl` in its `files`."""
        em, why = _read_json(root / "exec_meta.json")
        if not isinstance(em, dict):
            return [f"exec_meta.json beside the arm: {why or 'not an object'} (the execution step's record, copied by "
                    "score_cad.py into the scoring output)"]
        out = []
        if rt.get("exec_meta_sha256") != _sha_or_none(root / "exec_meta.json"):
            out.append(f"exec_meta.json is not the execution record the scoring run read (env-runtime.json "
                       f"exec_meta_sha256 {rt.get('exec_meta_sha256')!r})")
        if (em.get("schema"), em.get("schema_version")) != ("ptc_mm.cad_exec", "1.0"):
            out.append(f"exec_meta.json schema {em.get('schema')!r} {em.get('schema_version')!r} is not ptc_mm.cad_exec 1.0")
        if em.get("code_sha256") != EXECUTOR_CODE_SHA256:
            out.append(f"exec_meta.json code_sha256 {em.get('code_sha256')!r} is not the pinned executor code "
                       f"{EXECUTOR_CODE_SHA256} (the programs ran under another sandbox)")
        if em.get("gen_meta_sha256") != _sha_or_none(root / "gen_meta.json"):
            out.append("exec_meta.json gen_meta_sha256 is not the gen_meta.json beside the arm: the execution ran "
                       "another generation")
        for key, rkey in (("versions", "executor"), ("limits", "limits")):
            if em.get(key) != rt.get(rkey):
                out.append(f"exec_meta.json {key} {em.get(key)!r} is not env-runtime.json {rkey} {rt.get(rkey)!r}")
        ids = em.get("ids") if isinstance(em.get("ids"), list) else []
        files = em.get("files") if isinstance(em.get("files"), dict) else {}
        for eid, status in sorted(statuses.items()):
            if eid not in ids:
                out.append(f"exec_meta.json has no execution of {eid}")
            elif f"programs/{eid}/result.json" not in files:
                out.append(f"exec_meta.json files have no programs/{eid}/result.json")
            elif (f"programs/{eid}/mesh.stl" in files) != (status == OK):
                out.append(f"exec_meta.json files {'list' if status != OK else 'lack'} programs/{eid}/mesh.stl for "
                           f"status {status!r} (a mesh exactly for a valid solid)")
            if len(out) >= 8:
                break
        return out

    def _gen_meta_problems(self, root: Path, rt: dict, meta: dict, items_sha: str | None,
                           model_dir: Path | None) -> list[str]:
        """EXT-197: gen_meta.json, copied by score_cad.py into the scoring output root, against the scoring record, the
        arm and the evaluated model."""
        out: list[str] = []
        gm, why = _read_json(root / "gen_meta.json")
        if not isinstance(gm, dict):
            return [f"gen_meta.json beside the arm: {why or 'not an object'} (score_cad.py copies the generation run's)"]
        if rt.get("gen_meta_sha256") != _sha_or_none(root / "gen_meta.json"):
            out.append(f"gen_meta.json is not the file the scoring run read (env-runtime.json gen_meta_sha256 "
                       f"{rt.get('gen_meta_sha256')!r})")
        if items_sha is not None and gm.get("items_sha256") != items_sha:
            out.append(f"gen_meta.json items_sha256 {gm.get('items_sha256')!r} is not the registered item set's "
                       f"{items_sha}: the generation read another item set")
        if gm.get("prompt_block_sha256") != PROMPT_BLOCK_SHA256:
            out.append(f"gen_meta.json prompt_block_sha256 {gm.get('prompt_block_sha256')!r} is not the SHARED PROMPT "
                       f"{PROMPT_BLOCK_SHA256}")
        sampler = gm.get("sampler")
        want = ({**sampler, "backend": gm.get("backend"), "prompt_block_sha256": gm.get("prompt_block_sha256")}
                if isinstance(sampler, dict) else None)
        from gatedtrain.ext import arm as _arm
        if not _arm.same_json(meta.get("sampler"), want):
            out.append("eval_meta.json sampler is not gen_meta.json's sampler with its backend and prompt block")
        versions = gm.get("versions") if isinstance(gm.get("versions"), dict) else {}
        fw = {"vllm": {"name": "vllm", "version": versions.get("vllm")},
              "hf": {"name": "transformers", "version": versions.get("transformers")}}.get(gm.get("backend"))
        if fw is None or meta.get("framework") != fw:
            out.append(f"eval_meta.json framework {meta.get('framework')!r} is not the generation backend "
                       f"{gm.get('backend')!r} of gen_meta.json ({fw!r})")
        if model_dir is not None:
            files = gm.get("model_files")
            try:
                want_files = {f: file_sha256(Path(model_dir) / f) for f in GEN_MODEL_FILES
                              if _plain_file(Path(model_dir) / f)}
                cut_sha = _sha_or_none(Path(model_dir) / CUT_MARKER) if self.is_cut(model_dir) else None
            except OSError as exc:
                return out + [f"the evaluated model {model_dir} cannot be hashed ({exc.__class__.__name__})"]
            if files != want_files:
                out.append(f"gen_meta.json model_files are not the deployable files of the evaluated model {model_dir} "
                           "(the arm generated with another model; README item 8)")
            cut = gm.get("cut")
            if (cut.get("cut_json_sha256") if isinstance(cut, dict) else None) != cut_sha:
                out.append(f"gen_meta.json cut {cut!r} does not name the evaluated model's cut.json ({cut_sha!r})")
        return out

    # --- pins and checks (MM-21, MM-29, MM-33; EXT-187, EXT-188) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """env-freeze.txt names NeMo RL's commit (`# nemo-rl commit <40 hex>`, ptc_cad_hooks.write_run_args), which is the
        pinned commit and the launch record's framework {name nemo_rl, version = commit = the pin} (MM-33). Raises
        ValueError with the reason (OSError for an unreadable freeze)."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        commits = [m.group(1) for line in text_lines(freeze.read_text(encoding="utf-8"))
                   if (m := FREEZE_COMMIT_RE.fullmatch(line.strip()))]
        if commits != [NEMO_RL_COMMIT]:
            raise ValueError(f"{freeze} names NeMo RL commits {commits}, not exactly the pinned {NEMO_RL_COMMIT} (the "
                             "Tier B image's /opt/nemo-rl)")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if (fw.get("name"), fw.get("version"), fw.get("commit")) != (self.framework_name, NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            raise ValueError(f"{freeze} names NeMo RL {NEMO_RL_COMMIT}, but the launch record claims framework "
                             f"{fw.get('name')!r} version {fw.get('version')!r} commit {fw.get('commit')!r} (NeMo RL is "
                             "pinned by commit: version and commit are the commit)")
        return {"name": self.framework_name, "version": NEMO_RL_COMMIT, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: run_args.json `script_sha256` (run_cad_nemo.py hashes itself) is the launch record's."""
        args, why = read_run_args(stage_out)
        got = args.get("script_sha256") if isinstance(args, dict) else None
        if got != launch.get("script_sha256"):
            return [Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 {got!r} is not "
                            f"the launch record's {launch.get('script_sha256')!r}{f' ({why})' if why else ''} (EXT-83)",
                            "collect the outputs of the run launched with this record's training script")]
        return []

    BASE_FIX = ("train from the runtime's base directory (run_cad_nemo.py --base <area6-cad>/models/"
                "qwen2.5-coder-0.5b-instruct, SFT) or the SFT run's gated cut mounted read-only at /inputs/sft-cut "
                "(--init, GRPO), and gate against that directory")
    DATA_FIX = ("train on the runtime's data/<stage>/train.jsonl (the concatenation of the spec's shards) inside the "
                "read-only runtime mount")

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter; EXT-211, slice 7): the spec's base (the base directory, or
        the SFT cut for the GRPO stage) and its training shards lie inside read-only mounts. Before slice 7 this
        adapter had no such method and `--spec` raised AttributeError."""
        from gatedtrain.ext.launch import _ro_table, _within

        table = _ro_table(mounts)
        out = []
        if not any(_within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base {base}", self.BASE_FIX))
        if not any(_within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's training data {data}", self.DATA_FIX))
        return out

    def _base_files_problem(self, ref: dict, base_dir: Path, where: str) -> str | None:
        """None when base_ref.json names the pinned base and its `files` are the base directory's files (MODELS.json
        shape {name: {bytes, sha256}})."""
        if (ref.get("repo"), ref.get("revision")) != (BASE_REPO, BASE_REVISION):
            return (f"{where} names base {ref.get('repo')!r} revision {ref.get('revision')!r}, not {BASE_REPO} @ "
                    f"{BASE_REVISION} (a branch name is refused, MM-18)")
        files = ref.get("files")
        try:
            entries = self._base_entries(base_dir)
        except (OSError, ValueError):
            return f"the base directory {base_dir} cannot be hashed"
        want = {e["path"]: {"bytes": e["size"], "sha256": "sha256:" + e["sha256"]} for e in entries}
        if files != want:
            return f"{where} files are not the SHA-256 and sizes of the base directory {base_dir} (MM-21)"
        return None

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", uses: str | None = None, **_: Any) -> list[Problem]:
        """EXT-187 base and data binding, EXT-188 seed, profile, last cut, script, helper modules and NeMo RL settings."""
        from gatedtrain.ext.launch import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent.parent
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base (EXT-187)",
                            self.BASE_FIX)]
        out: list[Problem] = []
        mounts = launch.get("mounts")
        stage = args.get("stage")
        want_stage = STAGE_OF.get(uses) if uses is not None else stage
        if stage not in ("sft", "grpo") or stage != want_stage:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json stage is {stage!r}, but the spec's stage uses "
                               f"{uses!r} (sft: {SFT_ID}, grpo: {GRPO_ID}; MM-15)",
                               "collect each stage with its own spec"))
            return out
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        cut_docs = []
        for d in cuts:
            doc, _ = read_cut_marker(d)
            cut_docs.append((Path(d).name, doc or {}))

        # EXT-187 (MM-21): the base the run loaded, through a read-only mount, and every cut's base_ref.json.
        base_model = args.get("base_model")
        hit = container_file(f"{base_model.rstrip('/')}/config.json", mounts) if isinstance(base_model, str) else None
        base_dir = Path(hit[1]).parent if hit else None
        if base_dir is None or (stage == "sft" and base_dir.resolve() != Path(base).resolve()):
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_model {base_model!r} is not, through a "
                               f"read-only mount, {'the spec base ' + str(base) if stage == 'sft' else 'a base directory'}"
                               " (MM-21)", self.BASE_FIX))
        init = args.get("init_checkpoint")
        if stage == "sft":
            if args.get("model_name_or_path") != base_model:
                out.append(Problem("EXT-BASE", base_field, f"run_args.json model_name_or_path "
                                   f"{args.get('model_name_or_path')!r} is not its base_model {base_model!r}",
                                   self.BASE_FIX))
            if init is not None:
                out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint is {init!r}: SFT starts from "
                                   "the base (--init is the GRPO stage's)", self.BASE_FIX))
        else:
            out += self._grpo_init_problems(args, init, base, mounts, base_field)
        for name, doc in cut_docs:
            ref, why = _read_json(Path(stage_out) / CUTS_DIR / name / BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{name}/{BASE_REF}: {why or 'not an object'}", self.BASE_FIX))
                continue
            if base_dir is not None:
                bad = self._base_files_problem(ref, base_dir, f"{name}/{BASE_REF}")
                if bad:
                    out.append(Problem("EXT-BASE", base_field, bad, self.BASE_FIX))
            if ref.get("init_checkpoint") != (init if stage == "grpo" else None):
                out.append(Problem("EXT-BASE", base_field, f"{name}/{BASE_REF} init_checkpoint "
                                   f"{ref.get('init_checkpoint')!r} is not run_args.json init_checkpoint {init!r}",
                                   self.BASE_FIX))

        # EXT-187 (EXT-65 analogue): the file the run read is the concatenation of the spec's shards, in the mount.
        if data is not None:
            out += self._data_problems(args, Path(data), stage, mounts, data_field)

        out += self._launch_binding_problems(stage_out, args, launch, stage, init)              # review 11 H1

        # EXT-188: seed, profile, dry run, steps, script, helpers, NeMo RL settings, cut fields.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json `seed` is {seed!r} (MM-29)", "launch with the seed the record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json dry_run is {args.get('dry_run')!r}, not false",
                               "collect a real training run"))
        if not (_int(args.get("attempt")) and args["attempt"] >= 1):
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json attempt is {args.get('attempt')!r}, not a "
                               "positive integer", "collect the run as run_cad_nemo.py wrote it"))
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        if not (_int(args.get("max_steps")) and args["max_steps"] == last and prof.get("steps") == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but run_args.json max_steps is "
                               f"{args.get('max_steps')!r} (profile_settings.steps {prof.get('steps')!r}): the run did "
                               "not save its last step", "collect a run that finished and saved its last step"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        script = container_file(launch.get("script_path"), mounts) if isinstance(launch.get("script_path"), str) else None
        for key, fname in HELPERS.items():
            host = Path(script[1]).parent / fname if script else None
            now = _sha_or_none(host) if host is not None else None
            if now is None or args.get(key) != now:
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"run_args.json {key} "
                                   f"{args.get(key)!r} is not the SHA-256 of {fname} beside the launched script "
                                   f"({now!r}): the run imported other code than the record binds (EXT-83)",
                                   "collect the run launched from this runtime directory, unchanged since"))
        fw = args.get("framework") if isinstance(args.get("framework"), dict) else {}
        if fw.get("commit") != NEMO_RL_COMMIT:
            out.append(Problem("EXT-FREEZE", "--stage-out", f"run_args.json framework.commit is {fw.get('commit')!r}, not "
                               f"the pinned NeMo RL {NEMO_RL_COMMIT}", "train in the pinned NeMo RL image"))
        ov = args.get("nemo_rl_overrides")
        if not (isinstance(ov, list) and SHUFFLE_OFF in ov):
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json nemo_rl_overrides do not hold {SHUFFLE_OFF!r}: "
                               "the observed logs assume file order", "train with run_cad_nemo.py's overrides"))
        for name, doc in cut_docs:
            for key, want, rule, fld in (("seed", claimed, "EXT-LAUNCH", field),
                                         ("profile", launch.get("profile"), "EXT-PROFILE", "--launch-record.profile"),
                                         ("stage", stage, "EXT-STAGE", "--stage-out"),
                                         ("script_sha256", launch.get("script_sha256"), "EXT-LAUNCH",
                                          "--launch-record.script_sha256")):
                if doc.get(key) != want or (key == "seed" and not _int(doc.get(key))):
                    out.append(Problem(rule, fld, f"{name}/cut.json {key} is {doc.get(key)!r}, not {want!r}",
                                       "collect the cuts of the run launched with this record"))
        return out

    def _launch_binding_problems(self, stage_out: Path, args: dict, launch: dict, stage: str,
                                 init: Any) -> list[Problem]:
        """EXT-198 (review 11 H1): the record's launch is the run the outputs show. The record's argv runs
        run_cad_nemo.py with `--stage` = run_args.json stage and `--seed` = its seed; a GRPO record has `--init` = the
        init_checkpoint path, `--socket` = the sidecar's socket and a `sidecar`; an SFT record has none of the three,
        and an SFT output holds no environment log (no model-written code ran)."""
        fix = "collect the run with the launch record ext-launch wrote for it"
        fld = "--launch-record.argv"
        out = []
        argv = launch.get("argv") if isinstance(launch.get("argv"), list) else []
        script = argv[1].rsplit("/", 1)[-1] if len(argv) > 1 and isinstance(argv[1], str) else None
        if script != SCRIPT_TRAIN:
            return [Problem("EXT-LAUNCH", fld, f"the record's argv[1] {argv[1] if len(argv) > 1 else None!r} is not "
                            f"{SCRIPT_TRAIN}: the outputs are a training run", fix)]
        given = {o: v for o, v, _ in parse_options([a for a in argv[2:] if isinstance(a, str)], TRAIN_OPTIONS)}
        if given.get("--stage") != stage:
            out.append(Problem("EXT-LAUNCH", fld, f"the record's --stage {given.get('--stage')!r} is not run_args.json "
                               f"stage {stage!r}: the record launched another stage than the outputs show", fix))
        if given.get("--seed") != str(args.get("seed")):
            out.append(Problem("EXT-LAUNCH", fld, f"the record's --seed {given.get('--seed')!r} is not run_args.json seed "
                               f"{args.get('seed')!r}", fix))
        socket = ROW["sidecar"]["socket"]
        if stage == "grpo":
            path = init.get("path") if isinstance(init, dict) else None
            if given.get("--init") != path:
                out.append(Problem("EXT-LAUNCH", fld, f"the record's --init {given.get('--init')!r} is not run_args.json "
                                   f"init_checkpoint.path {path!r}", fix))
            if given.get("--socket") != socket:
                out.append(Problem("EXT-LAUNCH", fld, f"the record's --socket {given.get('--socket')!r} is not the "
                                   f"sidecar's {socket}", fix))
            if not isinstance(launch.get("sidecar"), dict):
                out.append(Problem("EXT-LAUNCH", "--launch-record.sidecar", "the outputs are a GRPO run (run_args.json "
                                   "stage grpo), whose programs run only in the paired executor sidecar, but the record "
                                   "has no sidecar (runtime/README.md \"Decisions\")", fix))
        else:
            for opt in ("--init", "--socket"):
                if opt in given:
                    out.append(Problem("EXT-LAUNCH", fld, f"the record's {opt} with an SFT run", fix))
            if "sidecar" in launch:
                out.append(Problem("EXT-LAUNCH", "--launch-record.sidecar", "an SFT run pairs no sidecar", fix))
            logs = [n for n in (ENV_CALLS,) if (Path(stage_out) / n).exists() or (Path(stage_out) / n).is_symlink()]
            logs += [p.name for p in self.extra_read_outputs(stage_out)]
            if logs:
                out.append(Problem("EXT-STAGE", "--stage-out", f"the SFT run's output holds {logs[:3]} (environment or "
                                   "GRPO step logs): model-written code ran in it", "collect each stage's own output"))
        return out

    def _grpo_init_problems(self, args: dict, init: Any, base: Path, mounts: Any, base_field: str) -> list[Problem]:
        """EXT-187 (README item 3, EXT-101 analogue): the GRPO run started from the spec's base, an SFT cut, mounted
        read-only at /inputs/sft-cut and named by run_args.json init_checkpoint {path, global_step, cut_json_sha256}."""
        from gatedtrain.ext.launch import container_file

        out = []
        if not (isinstance(init, dict) and set(init) == {"path", "global_step", "cut_json_sha256"}):
            return [Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint is {init!r}, not {{path, global_step, "
                            "cut_json_sha256}: a GRPO run starts from the SFT cut (--init)", self.BASE_FIX)]
        doc, why = read_cut_marker(base)
        if doc is None or doc.get("stage") != "sft":
            out.append(Problem("EXT-BASE", base_field, f"the spec's base {base} is not an SFT cut (cut.json stage "
                               f"{(doc or {}).get('stage')!r}{'; ' + why if why else ''}): the GRPO stage gates against "
                               "the SFT run's gated cut (MM-15)", self.BASE_FIX))
            return out
        path = init["path"]
        hit = container_file(f"{path.rstrip('/')}/{CUT_MARKER}", mounts) if isinstance(path, str) else None
        if path != SFT_CUT_CONTAINER or hit is None or Path(hit[1]).parent.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint.path {path!r} is not, through the "
                               f"read-only mount {SFT_CUT_CONTAINER}, the spec's base cut {base}", self.BASE_FIX))
        if args.get("model_name_or_path") != path:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json model_name_or_path "
                               f"{args.get('model_name_or_path')!r} is not init_checkpoint.path {path!r}", self.BASE_FIX))
        if init["cut_json_sha256"] != _sha_or_none(Path(base) / CUT_MARKER):
            out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint.cut_json_sha256 "
                               f"{init['cut_json_sha256']!r} is not the SHA-256 of the spec's base cut's cut.json",
                               self.BASE_FIX))
        if not (_int(init["global_step"]) and init["global_step"] == doc.get("global_step")):
            out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint.global_step "
                               f"{init['global_step']!r} is not the spec's base cut's global_step "
                               f"{doc.get('global_step')!r}", self.BASE_FIX))
        return out

    def _data_problems(self, args: dict, data: Path, stage: str, mounts: Any, field: str) -> list[Problem]:
        from gatedtrain.ext.launch import container_file

        out = []
        if _stage_of_data(data) != stage:
            return [Problem("EXT-DATA", field, f"the spec's training data {data} is not <runtime>/data/{stage}/shards",
                            self.DATA_FIX)]
        concat = data.parent / "train.jsonl"
        path = args.get("train_file")
        hit = container_file(path, mounts) if isinstance(path, str) else None
        if hit is None or Path(hit[1]).resolve() != concat.resolve():
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file {path!r} is not, through a read-only mount, "
                               f"{concat} (EXT-187)", self.DATA_FIX))
            return out
        now = _sha_or_none(concat)
        try:
            shards = shards_concat_sha256(data)
        except (OSError, ValueError) as exc:
            return out + [Problem("EXT-DATA", field, f"cannot hash the training shards ({exc.__class__.__name__})",
                                  "keep the training dataset readable")]
        if args.get("train_file_sha256") != now:
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file_sha256 {args.get('train_file_sha256')!r} is "
                               f"not the SHA-256 of {concat} ({now})", self.DATA_FIX))
        if now != shards:
            out.append(Problem("EXT-DATA", field, f"{concat} is not the concatenation of the spec's shards {data} in "
                               "file-name order: the run read other rows than the spec names", self.DATA_FIX))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-188: every cut's deployment sampler (sampler.json, a component) is the registered sampler without the
        evaluator's keys (backend, prompt_block_sha256), and the registered prompt block is the SHARED PROMPT the data
        was built with."""
        from gatedtrain.ext import arm

        out = []
        if not isinstance(sampler, dict):
            return [Problem("EXT-SAMPLER", "registration.sampler", "the registered sampler is not a mapping",
                            "register score_cad.py's sampler")]
        if sampler.get("prompt_block_sha256") != PROMPT_BLOCK_SHA256:
            out.append(Problem("EXT-SAMPLER", "registration.sampler.prompt_block_sha256", f"the registered "
                               f"prompt_block_sha256 {sampler.get('prompt_block_sha256')!r} is not the SHARED PROMPT "
                               f"{PROMPT_BLOCK_SHA256} the training data and items were built with",
                               "register the evaluation of the area's prompt v2"))
        want = {k: v for k, v in sampler.items() if k not in EVAL_ONLY_SAMPLER_KEYS}
        for d in cuts:
            doc, _ = _read_json(Path(d) / SAMPLER_FILE)
            if not arm.same_json(doc, want):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/{SAMPLER_FILE} is not the "
                                   "registered sampler without backend and prompt_block_sha256: the cut deploys another "
                                   "sampler than the gate evaluates", "register gen_cad.py's SAMPLER the cuts carry"))
        return out

    # --- launch side (EXT-191 to EXT-193, EXT-199) ---

    @staticmethod
    def _script(argv: Any) -> tuple[str | None, int]:
        """(the script's file name, its argv index): argv[2] after the executor scripts' `-I`, else argv[1]."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return None, 1
        i = 2 if (len(argv) > 2 and argv[1] == "-I" and argv[2].rsplit("/", 1)[-1] in FLAGGED) else 1
        return argv[i].rsplit("/", 1)[-1], i

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """ext-launch, --recheck and ext-collect. The script is one of the runtime's at its container path: run_cad_nemo.py
        (`--stage` sft or grpo, `--profile` the launch's, `--out /out`, an integer `--seed`, `--base` the runtime's
        base; GRPO: `--init /inputs/sft-cut` and `--socket` the sidecar's socket; SFT: neither), gen_cad.py (`--backend
        vllm`, `--model` the base or /inputs/cut, `--items` one of the two registered item sets, `--out /out`),
        `-I executor/cad_sandbox.py --run-batch --gen /inputs/gen --out /out` (execution: no other mode, no limit option)
        or `-I score_cad.py --gen /inputs/gen --exec /inputs/exec --items <one of the two> --out /out`; the two executor
        scripts need `-I`, the NeMo RL scripts have none; `--limit` is refused under `--profile full`. Options are
        resolved as argparse resolves them (a unique prefix counts)."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return []                                     # bind_argv reports the shape
        script, si = self._script(argv)
        if script not in OPTIONS or argv[si] != SCRIPT_PATHS[script]:
            return [f"argv[{si}] {argv[si]!r} is not one of the runtime's scripts {sorted(SCRIPT_PATHS.values())} (a "
                    "copy elsewhere imports other helpers and data)"]
        if (script in FLAGGED) != (si == 2):
            return [f"{script} runs as `python -I {SCRIPT_PATHS[script]}`" if script in FLAGGED else
                    f"{script} follows the interpreter directly"]
        parsed = parse_options(argv[si + 1:], OPTIONS[script])
        given: dict[str, str | None] = {}
        for opt, value, _ in parsed:
            given[opt] = value                            # argparse keeps the last value
        out = []

        def need(opt: str, allowed: tuple, what: str) -> None:
            if given.get(opt) not in allowed:
                out.append(f"{script} {opt} {given.get(opt)!r} is not {what}")

        items = tuple(f"{RUNTIME}/{s}" for s in ITEM_SETS)
        if script == SCRIPT_TRAIN:
            stage = given.get("--stage")
            need("--stage", ("sft", "grpo"), "sft or grpo")
            need("--profile", (profile,) if profile in PROFILES else PROFILES, f"the launch profile {profile!r}")
            need("--out", ("/out",), "/out")
            if not re.fullmatch(r"[0-9]+", given.get("--seed") or ""):
                out.append(f"{script} --seed {given.get('--seed')!r} is not an integer")
            need("--base", (f"{RUNTIME}/{BASE_REL}",), f"the runtime's base {RUNTIME}/{BASE_REL}")
            socket = TRAINER_ROWS[NAME]["sidecar"]["socket"]
            if stage == "grpo":
                need("--init", (SFT_CUT_CONTAINER,), f"{SFT_CUT_CONTAINER} (the SFT cut, read-only)")
                need("--socket", (socket,), f"the sidecar's socket {socket}")
            elif stage == "sft":
                for opt in ("--init", "--socket"):
                    if opt in given:
                        out.append(f"{script} {opt} with --stage sft: SFT starts from the base and executes no program")
        elif script == SCRIPT_GEN:
            need("--backend", ("vllm",), "vllm (the hf backend is the CPU pilot's)")
            need("--model", (f"{RUNTIME}/{BASE_REL}", EVAL_CUT_CONTAINER), f"the runtime's base or {EVAL_CUT_CONTAINER}")
            need("--items", items, "one of the runtime's item sets")
            need("--out", ("/out",), "/out")
        elif script == SCRIPT_EXEC:
            if "--run-batch" not in given:
                out.append(f"{script} without --run-batch: the execution step's mode (the sidecar's --serve is "
                           "launched only as the GRPO sidecar)")
            for opt in ("--serve", "--self-test", "--probe-xor", "--cpu-s", "--wall-s"):
                if opt in given:
                    out.append(f"{script} {opt}: the execution step runs --run-batch with the sandbox's own limits")
            need("--gen", (GEN_CONTAINER,), GEN_CONTAINER)
            need("--out", ("/out",), "/out")
        else:
            need("--gen", (GEN_CONTAINER,), GEN_CONTAINER)
            need("--exec", (EXEC_CONTAINER,), EXEC_CONTAINER)
            need("--items", items, "one of the runtime's item sets")
            need("--out", ("/out",), "/out")
        if script in (SCRIPT_GEN, SCRIPT_SCORE) and "--limit" in given and profile == "full":
            out.append(f"{script} --limit {given['--limit']!r} under --profile full: a gate arm evaluates the whole "
                       "registered item set; --limit is for the smoke arms")
        return out

    def launch_problems(self, doc: dict, *, kind: str) -> list[tuple[str, str]]:
        """[(field, message)] of this adapter's launch rules (request or record): the NeMo RL scripts run in the pinned
        NeMo RL image (`container`, GPU), the execution and scoring steps in the pinned executor image with its lock
        (`venv_in_container`, CPU, network none, tmpfs /tmp, exactly HOME, PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED);
        each script has exactly its mount set (README "Launch request facts"; EXT-199: a container that runs generated
        code never mounts the area root, which holds the references); the GRPO trainer pairs the executor sidecar,
        nothing else does; the framework is NeMo RL pinned by commit."""
        out: list[tuple[str, str]] = []
        argv = doc.get("argv")
        script, si = self._script(argv)
        rt = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
        fw = doc.get("framework") if isinstance(doc.get("framework"), dict) else {}
        if (fw.get("version"), fw.get("commit")) != (NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            out.append(("framework", f"framework version {fw.get('version')!r} and commit {fw.get('commit')!r} are not "
                        f"the pinned NeMo RL commit {NEMO_RL_COMMIT} (pinned by commit)"))
        stage = None
        if script == SCRIPT_TRAIN:
            given = {opt: value for opt, value, _ in parse_options(argv[2:], TRAIN_OPTIONS)}
            stage = given.get("--stage")
            seed = doc.get("training_seed")
            if _int(seed) and given.get("--seed") != str(seed):              # review 11 L3
                out.append(("argv", f"run_cad_nemo.py --seed {given.get('--seed')!r} is not the training_seed {seed} "
                            "(MM-29)"))
        if script in (SCRIPT_TRAIN, SCRIPT_GEN):
            if rt.get("kind") != "container" or rt.get("image") not in NEMO_RL_IMAGES:
                out.append(("runtime", f"{script} runs in a pinned NeMo RL image: runtime {{kind: container, image: "
                            f"one of {list(NEMO_RL_IMAGES)}}}, not {rt.get('kind')!r} {rt.get('image')!r}"))
            if doc.get("gpus") is not True:
                out.append(("gpus", f"{script} needs the GPU (gpus true)"))
        elif script in FLAGGED:
            if (rt.get("kind"), rt.get("image")) != ("venv_in_container", EXECUTOR_IMAGE):
                out.append(("runtime", f"{script} runs in the executor image: runtime {{kind: venv_in_container, image: "
                            f"{EXECUTOR_IMAGE}}}, not {rt.get('kind')!r} {rt.get('image')!r}"))
            lock = (_sha_or_none(Path(rt["venv_lock"])) if isinstance(rt.get("venv_lock"), str) else None) \
                if kind == "request" else rt.get("venv_lock_sha256")
            if lock != EXECUTOR_LOCK_SHA256:
                out.append(("runtime.venv_lock", f"the executor venv lock is {lock!r}, not the pinned "
                            f"{EXECUTOR_LOCK_SHA256} (executor/uv.lock)"))
            if doc.get("gpus") is not False:
                out.append(("gpus", f"{script} runs on CPU (gpus false)"))
            if doc.get("network") != "none":
                out.append(("network", f"{script} runs in the executor container: network none, not "
                            f"{doc.get('network')!r} (B18)"))
            if doc.get("tmpfs") != ["/tmp"]:
                out.append(("tmpfs", f"tmpfs {doc.get('tmpfs')!r} is not ['/tmp'] (the read-only root's scratch)"))
            names = sorted(doc.get("env") or {}) if kind == "request" and isinstance(doc.get("env"), dict) \
                else doc.get("env_names")
            if names != sorted(EXECUTOR_ENV):
                out.append(("env" if kind == "request" else "env_names", f"environment names {names!r} are not exactly "
                            f"{sorted(EXECUTOR_ENV)} (executor/run-executor.sh)"))
        if script is not None and script in OPTIONS:
            want = MOUNT_SETS[script](stage)
            if script == SCRIPT_GEN:
                model = {o: v for o, v, _ in parse_options(argv[si + 1:], GEN_OPTIONS)}.get("--model")
                want = [(RUNTIME, "ro"), ("/out", "rw")] + ([(EVAL_CUT_CONTAINER, "ro")]
                                                            if model == EVAL_CUT_CONTAINER else [])
            mounts = doc.get("mounts") if isinstance(doc.get("mounts"), list) else []
            got = sorted((_norm_path(m.get("container")), m.get("mode")) for m in mounts if isinstance(m, dict))
            if got != sorted(want) or len(got) != len(mounts):
                out.append(("mounts", f"{script} mounts {got}, not exactly {sorted(want)} (README \"Launch request "
                            "facts\"" + ("; a container that runs generated code never mounts the area root"
                                         if script == SCRIPT_EXEC else "") + ")"))
        has = "sidecar" in doc
        if stage == "grpo" and not has:
            out.append(("sidecar", "a GRPO launch pairs the executor sidecar (EXT-190): the request "
                        "names it under `sidecar`, exactly as executor/run-executor.sh serve prints it"))
        elif stage != "grpo" and has:
            out.append(("sidecar", f"only the GRPO trainer pairs the executor sidecar, not {script} "
                        f"{'--stage ' + stage if stage else ''}".rstrip()))
        return out



# EXT-199: the exact (container path, mode) mount set per script (README "Launch request facts"; review 11 M1). The
# trainer's and generation's sets name the area root (no generated code runs there); the executing step names only the
# Python tree and the executor directory besides its input and output; scoring reads the references and runs no
# generated code. The GRPO sidecar's set is the registry row's (`sidecar_problems`).
MOUNT_SETS = {
    SCRIPT_TRAIN: lambda stage: ([(RUNTIME, "ro"), (SFT_CUT_CONTAINER, "ro"), ("/out", "rw"), ("/spool", "rw")]
                                 if stage == "grpo" else [(RUNTIME, "ro"), ("/out", "rw")]),
    SCRIPT_GEN: lambda stage: None,               # replaced below: the base or a cut at /inputs/cut
    SCRIPT_EXEC: lambda stage: [(AREA6_PYTHON, "ro"), (AREA6_EXECUTOR, "ro"), (GEN_CONTAINER, "ro"), ("/out", "rw")],
    SCRIPT_SCORE: lambda stage: [(AREA6_PYTHON, "ro"), (RUNTIME, "ro"), (GEN_CONTAINER, "ro"), (EXEC_CONTAINER, "ro"),
                                 ("/out", "rw")],
}
