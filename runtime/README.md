# Area "CAD": text-to-CadQuery with NeMo RL, SFT then GRPO against a sandboxed CadQuery executor (prototype, unpromotable)

Multimodal prototype, area 6 ("CAD | NeMo RL + CadQuery environment | small LLM, SFT then GRPO"; sixth in the
run order), multimodal spec row 6, MM-15 (two stages), B18 deviation (design 06 B18). Two runtimes: the Tier B
NeMo RL image for training and generation (runtime kind `image`), and a separate executor container for every
execution of model-written code (runtime kind `venv_in_container`: a small Ubuntu image plus this directory's
executor venv, mounted read-only). Nothing here starts a container or uses the GPU; you run the
commands below through `ext-launch` and `gpu-step.sh`. Numbers marked **estimate** are unmeasured on
the GPU; the smoke runs measure them.

Status (2026-11-13): runtime prepared; executor, data, metric and CPU pilots measured. The NeMo RL
launch code (`run_cad_nemo.py`, `ptc_cad_hooks.py`, `cad_env.py`'s Ray actor, `gen_cad.py --backend vllm`) is
**written from the pinned clone but untested** (it needs the GPU image); the smoke runs test it. Proposals are
marked "proposal" and collected in "Open questions".

**Decisions (2026-11-13):** trainer NeMo RL (TRL as
fallback); deciding metric `iou`, T3 with the construct-alignment disclosure, `valid` supporting; `ext-launch` gets a
paired sidecar record for the executor (gatedtrain work, not here); GRPO headroom by holding 5 families out of SFT
(GRPO trains on all 15; the GRPO gate's items come only from the 5 held-out families). This README reflects them.

## Pins

| Item | Value |
|---|---|
| Trainer image | Tier B's `nemo-rl:4f39fd8`, image ID `sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a` (59.7 GB; built 2026-11-10 with one recorded local edit, `pyproject.toml:617` flash-attn build parallelism). NeMo RL commit `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`, vLLM 0.29.0. Nothing is installed into it: area code is mounted read-only. |
| Executor image | `executor/Containerfile`: `ubuntu:noble-20260930.1@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac` (index digest, updated 2026-10-17) + `libgl1 libx11-6 libxcb1 libexpat1 libbsd0` from the Ubuntu snapshot 2026-10-29 (`apt --snapshot`). **Not built** (no docker in preparation); open question 1. |
| Python (executor) | CPython 3.12.16, `<python-root>/cpython-3.12.16-linux-x86_64-gnu` (shared with areas 1-4, unchanged) |
| Executor lock | `executor/uv.lock` (46 packages, `exclude-newer = 2026-10-29`, Linux x86_64), sha256 `8f4264b9…6c55` |
| Executor venv | `executor/.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 1.8 GB, 13,854 files): **cadquery 2.8.0** (2026-06-21, Apache-2.0), cadquery-ocp 7.9.3.1.1 (OCCT 7.9.3), numpy 2.5.4, vtk 9.6.2, casadi 3.8.2, scipy 1.18.1 (cadquery's own dependencies) |
| Base model | `models/qwen2.5-coder-0.5b-instruct` (`models/MODELS.json`, sha256 `3ce6797a…0e4d`); the unused candidate `qwen3-0.6b` was moved (not deleted) to `<mm-root>/area6-ref/models/qwen3-0.6b` with its record `area6-ref/models/MODELS.json` (hashes re-checked after the move) |
| Host-only pilot interpreter | the area-4 venv, read-only (transformers 5.21.0, torch 2.15.0, huggingface-hub 2.3.0) for `fetch_models.py`, `gen_cad.py --backend hf` and `sft_cpu_pilot.py`; no venv of its own (it would duplicate 5.7 GB) |

Rebuild the executor venv (host, no GPU): `cd executor; UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy
UV_PYTHON_DOWNLOADS=never uv sync --frozen --compile-bytecode --python
<python-root>/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12` (the build used a temporary
`UV_CACHE_DIR` in this directory, deleted afterwards).

## Trainer plan

### Is NeMo RL practical here? Yes, with the Tier B image (verified facts)

- **Already built and run.** Tier B ran NeMo RL GRPO on a single 24 GB RTX 4090 with Qwen3-0.6B: 150 steps x 256
  samples (32 prompts x 8, 1,024 tokens) in 1.22 GPU-h with `activation_checkpointing=true`, peak policy memory
  21 GB, with host swap in use. No new install is needed: the 59.7 GB image exists; installing NeMo RL's full stack
  again is not proposed (it took about 3 h 40 min and a flash-attn build-parallelism edit).
- **Custom environments are a supported extension point.** `register_env(name, actor_class_fqn)`
  (nemo_rl/environments/utils.py:132-136) plus a runtime entry `ACTOR_ENVIRONMENT_REGISTRY[fqn] =
  PY_EXECUTABLES.SYSTEM` for actors defined outside the repository (the error text of
  nemo_rl/distributed/ray_actor_environment_registry.py:64-80 names exactly this); `create_env` starts the actor
  with the driver's environment variables (utils.py:111-129), so `PYTHONPATH` reaches the Ray worker. The data path
  binds a task to `data.default.env_name` (nemo_rl/data/utils.py:198-233). `EnvironmentInterface.step` returns
  `EnvironmentReturn(observations, metadata, next_stop_strings, rewards, terminateds, answers)`
  (environments/interfaces.py:26-50, :54-90). `math_hf_data_processor` passes the row's assistant text as
  `extra_env_info["ground_truth"]` (data/processors.py:553-563), which carries the reference features here.
- **Single GPU, 24 GB.** A 0.5-0.6 B full fine-tune fits (Tier B: Qwen3-0.6B at 256 x 1,024 tokens with activation
  checkpointing). 1.5 B does not without CPU offload (fp32 master weights + AdamW for 1.5 B = 24 GB before
  activations), so Qwen2.5-Coder-1.5B is out of scope.
- **SFT** uses the same image: `examples/run_sft.py` with `ResponseDataset` (response_dataset.py:20-80, keys
  `input`/`output`), `sft_processor` (processors.py:145-190), `data.shuffle=false` and `drop_last=True`
  (sft.py:351-353).

Host risks (all seen in Tier B, none blocking): limited host RAM, about 31 GB (Ray's memory monitor killed workers once;
Tier B runs use `RAY_memory_monitor_refresh_ms=0` and swap, decided 2026-11-11);
the Docker daemon Tier B used had no bridge network, so Tier B containers ran with `--network host` (open question 3:
the trainer container needs no network, the executor must have none); NeMo RL startup (Ray, vLLM, policy workers) costs minutes per launch.

### Options (decided 2026-11-13: (A) NeMo RL, (B) TRL as fallback)

| | (A) **NeMo RL** (decision row 6; recommended) | (B) TRL GRPO + TRL SFT (area-1 runtime) |
|---|---|---|
| Runtime | Tier B image, unchanged; area code mounted read-only (`run_cad_nemo.py` calls `examples/run_sft.py` / `run_grpo.py` `main()`) | area-1 venv (trl 1.16.0, vLLM 0.31.0, transformers 5.17.0) |
| Reward | Ray actor `cad_env.CadQueryEnvironment` -> executor socket | a TRL reward function -> the same executor socket (same `cad_protocol.reward`) |
| GPU-h (estimate, area total) | about 0.95 (0.8-1.3; table "Profiles and cost") | about 0.8-1.1: TRL starts faster (area 1: about 140 s) |
| Host risk | Ray + colocated offload: heavy swap use (Tier B) | lower (area 1: 9 GB RAM, little swap) |
| gatedtrain work | new adapter `nemo_rl_ext` for the cut and log layout below (Tier B's `tierb.py` reads `step_N`, `iteration` only) | `trl_grpo` exists for GRPO; SFT needs a TRL-SFT path; same cut and log conventions to add |
| Matches decision | yes | no (a deviation would need its own decision) |

Recommendation: (A). It is feasible on a single 24 GB GPU, it is what the decision names, the environment interface is a
documented extension point, and Tier B already exercised the image, the GPU and the memory fallbacks. Keep (B) as the
fallback if the NeMo RL smoke fails for a reason that is not quickly fixable (decided 2026-11-13: NeMo RL, TRL as fallback).

## Model choice (measured on CPU: 30 or 60 items, 2 or 4 per family, greedy, HF transformers fp32)

Candidates (both Apache-2.0, ungated, `fetch_models.py`, no token): `Qwen/Qwen2.5-Coder-0.5B-Instruct` @
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25` (494 M parameters, bf16 safetensors 1.0 GB) and `Qwen/Qwen3-0.6B` @
`c1899de289a04d12100db370d81485cdf75e47ca` (Tier B's base, non-thinking chat template). Scores are the deciding
`iou` (failures 0); "mode share" is the MM-38 rule (FAIL at >= 0.5).

| Prompt | Base | Items | Executes | Mean IoU | Mode share (value) | IoU given executes | MM-38 |
|---|---|---|---|---|---|---|---|
| v0: instruction only | Coder-0.5B | 60 | 0 / 60 | 0 | 1.00 (0) | - | FAIL |
| v1: + API reminder list | Coder-0.5B | 60 | 3 / 60 | 0.042 | 0.95 (0) | 0.835 | FAIL |
| **v2: + API reminder + two worked examples** | **Coder-0.5B** | 30 | 17 / 30 | 0.307 | 0.43 (0) | 0.543 | PASS (borderline) |
| v2 | Qwen3-0.6B | 30 | 15 / 30 | 0.255 | 0.50 (0) | 0.509 | FAIL (mode rule) |
| **v2 (chosen, full pilot)** | **Coder-0.5B** | 60 | 32 / 60 | **0.273** (sd 0.389) | 0.47 (0) | 0.511 | **PASS** (close to the bound) |

v0 failures: invented API (`cq.Cylinder`, `cq.Box`, `cq.Circle`). v1: the model copied the whole reminder list into
its program (every response called box, cylinder, circle, rect, polygon in sequence). v2's examples (a square plate
with a square hole; a cylinder with a block on top, both outside the 15 families) fixed the style. **Chosen:
Qwen2.5-Coder-0.5B-Instruct with prompt v2** (smaller, executes more often, and passes the mode rule where Qwen3
does not). Risk: the base's mode share sits close to 0.5, and the GPU arm (vLLM bf16 greedy) is not the CPU pilot
(HF fp32 greedy), so the SFT smoke's 20-item baseline decides (MM-38); if it fails, open question 7. On the 10 SFT families
alone (SFT gate items, 50-item pilot) the base scores mean 0.344 with mode share 0.48: again just inside the rule.

## Task and data (`cad_families.py`, `make_data.py`)

**Dataset survey (2026-11-13, Hub API, no token).** No ungated, license-clear text-to-CadQuery set fits:
`filapro/cad-recode` and `cad-recode-v1.5` are CC-BY-NC-4.0 and point-cloud-to-code, not text; `SadilKhan/Text2CAD`
is gated (`auto`) and CC-BY-NC-SA-4.0; `Admiralpp/Text-to-CADquery` is gated and CC-BY-NC-SA-4.0;
`CADCODER/GenCAD-Code` (image-to-code) declares no license; `rumike7/cadquery-creating-basic-2d-and-3d-forms` and
`ennucore/cadquery` declare MIT but their cards are empty (no provenance; plausibly derived from NC data). So the
data is **generated**: no third-party content, no license question.

**Families.** 15 parametric families with two phrasings each, every description fixing the placement (the
deciding metric compares solids in absolute coordinates): box, cylinder, tube, plate with a hole, plate with four
corner holes, hexagonal prism, stepped shaft, filleted box, chamfered box, open box (shell), cylinder with a blind
hole, slotted plate, L-bracket (union), counterbored plate, flange with a bolt circle (polar array). Dimensions are
whole millimetres. Every reference program is executed in the sandbox at build time and must give a valid solid
(0 failures of 4,600; 29 duplicate draws redrawn).

**Prompt** (the "SHARED PROMPT" block of `cad_protocol.py`, SHA-256 `b9aea19b…7e69`, prompt v2): an instruction
(millimetres, `import cadquery as cq`, assign `result`, reply with one short fenced block), a one-paragraph CadQuery
API reminder (Workplane, box, cylinder, extrude, faces/workplane, hole, cboreHole, slot2D, polarArray, fillet,
chamfer, shell, union, translate) and two worked examples outside the 15 families (a square plate with a square hole;
a cylinder with a block on top), then `Part: <description>`. The prompt is part of the task for training and
evaluation alike ("Model choice" shows why each part is there). Prompts are 1,679-1,893 characters (570-657 tokens with the chat template); SFT targets 26-94 tokens; reference
programs 79-251 characters.

**Family split (decided 2026-11-13).** Five families are held out of SFT: **cylinder with a blind hole,
plate with four corner holes, tube, flange (polar array), chamfered box** (`make_data.py` `HELD_OUT`, split B).
SFT keeps 10 families: box, cylinder, plate with a hole, hexagonal prism, stepped shaft, filleted box, open box
(shell), slotted plate, L-bracket, counterbored plate. GRPO trains on all 15 (prompts and rewards only).

How the five were chosen (all measured on CPU with the SFT proxy, `sft_cpu_pilot.py`, 30 steps x 8 on the
10-family SFT set, then greedy generation on the GRPO gate items):
- **Split A (rejected), `pilot/v2-split-a/`:** stepped shaft, filleted box, chamfered box, open box, flange, chosen
  because each needs an operation no SFT family uses and none was at zero for the base. Result: the SFT cut got
  *worse* on them than the base (150 items: mean IoU **0.104**, **74 % at 0**, MM-38 FAIL; the base on 50 of the
  same items: 0.247, 30 % at 0). The cut copied SFT templates (`faces(">Z").workplane().chamfer(...)` 30 of 30
  chamfer items fail; a box with four holes for a filleted box). Without examples of an operation, SFT on 10
  families destroys the base's partial ability, so a GRPO gate on such families starts degenerate.
- **Split B (chosen):** families one step away from an SFT family: blind hole (a depth on the plate-with-hole /
  counterbore hole), four-hole plate (construction-rectangle vertices), tube (a second circle), flange (polar array
  on a cylinder), chamfered box (chamfer instead of the filleted box's fillet). Result below: the SFT cut is mid-range
  on them and not degenerate.

| Set | Rows | Families | Use | SHA-256 |
|---|---|---|---|---|
| SFT, `data/sft/train.jsonl` (= `shards/shard-0000{0..3}.jsonl` concatenated) | 2,000 (200 per family) | the 10 SFT families | `input` = prompt, `output` = reference program in one fenced block | `517e3c05fba7f267d5278cd8f11e20e0bbabf8e2f81a4a76e0356f34b1ead66c` |
| GRPO, `data/grpo/train.jsonl` (4 shards) | 2,000 (133-134 per family) | all 15 | `input` = prompt, `output` = canonical JSON `{row_id, volume, bbox}` of the reference (OCCT) | `a4c74e1743fc28e09eea4e288fda1b9abbfe02607cd39b113864deb7251724b7` |
| SFT gate items, `items-sft/items.jsonl` + `items-sft/ref/<item>.stl` | 300 (30 per family) | the 10 SFT families | SFT run's gate (base vs SFT cut) | `9176194301e2a42af745747c6d17b37f799f9b05f5132e8aa41b69b876320963` |
| GRPO gate items, `items-grpo/items.jsonl` + `items-grpo/ref/<item>.stl` | 300 (60 per family) | the 5 held-out families | GRPO run's gate (SFT cut vs GRPO cut) | `58de2520a13bd570e3f3c9470fe9b24c9fea7b4f8123d800e29c3ebfbaebcc5c` |

`data/data_meta.json` (schema 2.0, sha256 `d8225f24…666a`) records the generator, the split, the counts and the
overlap check. Split A's hashes (superseded, for the record): SFT `ac44ae53…dd7c`, GRPO `9f3ea9b9…c28f`, SFT items
`65427d91…83d6`, GRPO items `9a7c748c…a5ff`.

Item row: `{item, family, text (= prompt), description, reference_code, ref_mesh, ref_mesh_sha256, ref_volume,
ref_bbox, params}`; item ids `cad-sft-<family>-<k>` and `cad-grpo-<family>-<k>`. Metadata: `data/data_meta.json`
(schema 2.0), `items-*/item_set_meta.json`.

**Why the SFT gate items come from the 10 SFT families only (not all 15).** The SFT gate then measures what SFT
trained on, and the two gates use disjoint families as well as disjoint geometries: the GRPO gate's families are
never seen with a reference program by any stage, and the SFT gate's mean is not mixed with five families the SFT
run never saw (which would dilute it and make its result depend on transfer). The SFT gate is far over-powered either
way ("Items and power").

**Disjointness.** A geometry (family plus every dimension, phrasing ignored) occurs in at most one of the four sets,
and so does the exact prompt text; re-checked over the written files for all six pairs: 0 shared geometries, 0
shared texts (`data/data_meta.json` `overlap`). The build also refuses an SFT row from a held-out family or a GRPO
gate item from an SFT family. Neighbouring dimensions do occur across sets (a 40 x 25 x 10 box item and a 40 x 25 x
11 training box): the SFT gate tests interpolation within its families; the GRPO gate tests families SFT never
showed (GRPO trains on their prompts, with rewards only, never reference programs). **Order.** Per family: its gate
items first, then GRPO rows, then SFT rows (`random.Random` seeded with SHA-256("20261113:<family>")); each training
file is then permuted once (seed "20261113:order:<split>") because NeMo RL reads it in file order. **Reproducible:**
two builds gave identical bytes for all 615 files.

## Executor sandbox (B18) (`executor/cad_sandbox.py`, `executor/cad_protocol.py`)

One subprocess per program. A single-threaded zygote imports CadQuery once (1.1 s) and forks one child per program;
the child sets its own limits and executes; the zygote never runs model code. Entry points: `--serve SOCKET`
(sidecar for GRPO; length-prefixed JSON over a Unix socket, one batch per connection), `--run-batch --gen DIR --out
DIR` (evaluation step 2a) and the `Zygote` class (`make_data.py`, trusted reference programs only). The trainer
side never imports CadQuery (`cad_protocol.py` is standard library).

**No reference data where generated code runs (decided after independent review 11).** Review 11
found that the executor container mounted the whole area read-only, including `items-*/` and `data/` with reference
programs and meshes, and that the audit hook blocked writes but not reads, so a generated program could read its own
item's answer. Design now:
1. **Files.** `cad_sandbox.py` and `cad_protocol.py` moved into `executor/`, next to the executor venv. Containers
   that run generated code (sidecar `serve`, evaluation `exec`) mount only the Python tree and `<area>/executor`,
   plus the spool socket directory or the gen input and an empty exec output; never `items-*/`, `data/`, `pilot/`,
   `models/` (`executor/run-executor.sh`).
2. **Memory.** The zygote never loads reference data: `--serve` receives `{id, code}` only; `--run-batch` reads only
   the gen responses. Children are forked from it, so nothing of a reference is in their memory.
3. **Scoring elsewhere.** The zygote returns the candidate's own status, OCCT volume, bounding box and (run-batch)
   STL mesh. The GRPO reward is computed by the environment actor in the trainer container (it holds the row's
   ground truth); the evaluation's IoU, chamfer and reward by `score_cad.py` in a separate scoring container that
   runs no generated code (step 2b). The scorer verifies the exec output: every file listed in `exec_meta.json`
   with its SHA-256 and nothing else (a stray file is `executor_error`), the `gen_meta.json` hash, each program's
   SHA-256 against the response, and the executor code hashes against its own pins.
4. **Defence in depth (host).** The audit hook now also refuses opening or listing anything outside the read roots
   (Python tree, `<area>/executor` with the venv, the work directory, `/usr`, `/lib`, `/lib64`, `/dev`), so the host
   self-test can show reference reads failing even where the files exist; `/proc/self/mem` is outside the roots.
   After each child exits, the zygote kills its process group (leftover processes). `--run-batch` checks that its
   output directory is still empty after every child has exited and before writing.

Throughput is unchanged (the serve protocol did not change): 150 programs in 0.47 s.

| Property | Where it is realized | Tested on the host |
|---|---|---|
| One process per program, killed on wall-clock deadline (15 s) | `cad_sandbox.py` | yes (`--self-test`: `timeout`) |
| CPU time 10 s (`RLIMIT_CPU`), address space = size at fork + 2 GiB (`RLIMIT_AS`), file size 64 MiB, 64 open files, no core | `cad_sandbox.py` | yes (`resource_limit` on an endless loop and on an 8 GiB allocation) |
| Fresh empty work directory, stdin `/dev/null`, own process group, cleared limits on threads (OCCT on one thread) | `cad_sandbox.py` | yes |
| Python audit hook: no sockets, subprocesses, exec/fork/spawn, signals, ctypes, deletes/renames, writes outside the work directory, reads or listings outside the read roots | `cad_sandbox.py` (defence in depth only; native code can bypass it) | yes (5 escape attempts and 8 reference-read attempts refused) |
| Jobs cannot touch each other's results (M3): private work directory per job (`mkdtemp`, 0700, random name), result over the job's own pipe (never a file), inherited descriptors closed, zygote and children not dumpable (no `/proc/<pid>/{cwd,fd,mem}` of a sibling), kept STL checked against the SHA-256 its job reported. One uid for all jobs: a sibling with native code can still write into another job's directory, it just cannot change what the zygote takes from it, except by racing the mesh between export and hash | `cad_sandbox.py` | yes (`tests/test_executor_isolation.py`, with the audit hook disabled to model native code) |
| **No reference data visible**: not mounted (`serve`, `exec` mount only the Python tree and `<area>/executor`) and not in the zygote's memory | executor containers (mounts); `cad_sandbox.py` (zygote never loads references) | memory: yes (probe below); mounts: no (needs docker) |
| **No network** (`--network none`) | executor container only | **no** (not testable without docker) |
| **Non-root** (`--user 1000:1000`), **no capabilities** (`--cap-drop ALL`), `no-new-privileges` | executor container only | no |
| **Read-only root filesystem** (`--read-only`), writable only tmpfs `/tmp` (noexec) and one spool or output mount | executor container only | no |
| **CPU / memory / pids** (`--cpus 8 --memory 8g --memory-swap 8g --pids-limit 256`) | executor container only | no |
| File-system confinement (the program cannot read host files) | executor container (mount set); on the host the audit hook's read roots | host: yes for Python-level reads; container: no (needs docker) |

`executor/run-executor.sh serve <run_dir> <label>`, `exec <gen_dir> <exec_dir>` and `score <gen_dir> <exec_dir>
<out_dir> <items-sft|items-grpo>` print the docker commands (they never run docker). **Self-test**
(`executor/cad_sandbox.py --self-test`, host, 2026-11-13): 23 programs, 23 PASS: ok, no_code, syntax_error,
runtime_error, no_result, not_solid, timeout (5 s test limit), resource_limit (CPU), resource_limit (memory),
socket / subprocess / write-outside / os.system / ctypes refused, os.abort (crash), and **8 reference-access
attempts refused** (open `items-grpo/items.jsonl`, `items-sft/items.jsonl`, `data/grpo/train.jsonl`,
`data/sft/train.jsonl`, a reference STL by glob and by path, `os.listdir(<area>)`, `/proc/self/mem`; inside the
container the same programs fail with FileNotFoundError, still `runtime_error`). Plus **zygote memory probe**: a
fresh `--run-batch` process executes two responses (one tries to open `items-grpo/items.jsonl`: `runtime_error`),
then a trusted forked child scans all readable memory it inherited from the zygote (1,010 MiB) for markers passed
XOR-ed so the probe itself holds no plain copy: the positive control (a response text the zygote read) is found
(4 hits); the first GRPO item's `reference_code`, its `ref_volume` repr, its `ref_mesh_sha256` and 150 bytes of its
reference mesh are found 0 times. PASS. Inside the container the probe is skipped (no reference files to make
markers from); the mount set covers it there.

**Found and fixed: a fork-safety crash.** With OCCT's default thread pool, forked children building the flange
references (boolean cuts of a polar array) died with SIGSEGV in 1 of 3,000 runs, and in 14 of 6,000 after switching
meshing to serial; OCCT creates its pool lazily in the child after fork. The child now forces a one-thread default
pool (`OSD_ThreadPool.DefaultPool_s(1)`, `OSD_Parallel.SetUseOcctThreads_s(True)`) before running the program, and
the zygote checks it has exactly one thread before each fork (numerical libraries pinned to one thread by
environment). After the fix: **0 crashes in 30,000 executions** of the 300 references (100 rounds, 16 workers,
1 min 56 s). A remaining crash is still scored `crash` (the policy's program), which is the right class for real
OCCT segfaults on degenerate geometry, so the fix matters for fairness.

**Throughput.** 128 programs (a GRPO step's batch) in 0.68 s through the socket with 8 workers (host, mixed
statuses, `pilot/` not kept; `exec_batch_s` in the env log). Reference programs take 0.01-0.06 s each. A step's
executor time is therefore about 1 s, at most the 15 s wall limit when a program hangs; it blocks the GPU (synchronous
GRPO) and is counted in the step time below.

## Metric and tier (decided 2026-11-13)

**Deciding metric `iou`** (evaluator `cad.cadquery_items@0.1`, `score_cad.py`, higher is better, range [0, 1]):
volumetric IoU of the generated solid and the item's reference solid. The program is executed once in the
executor; the solid is tessellated by OCCT (STL, linear deflection 0.002 of the bounding-box diagonal, serial);
`geom_metrics.py` (numpy) integrates both solids along 16,384 seeded vertical lines over the union of their XY
boxes: each line's inside intervals are exact (signed crossings with the triangles), only the XY placement is
sampled. Seed = SHA-256("iou:<item>"). Any outcome other than a valid solid scores 0 (MM-4). Checks: identical
meshes give exactly 1; two offset cubes 0.337 (exact 1/3); cube vs inscribed cylinder 0.780 (exact 0.785); the
mesh-integrated volume matches OCCT's (median 0.23 %, 95th percentile 0.9 %, max 4 % on thin parts). Sub-metrics: `status`, `executes`, `chamfer` (symmetric mean
surface distance / reference diagonal), the reward's own terms `volume_ratio`, `bbox_iou` and `reward` (reported
only), `n_new_tokens`, `finish_reason`, `unbalanced_columns`.

**Training reward** (`cad_protocol.reward`, `cad_reward_v1`): 0 unless the program yields a valid solid; else
0.2 + 0.4 x volume ratio + 0.4 x bounding-box IoU, all three quantities from OCCT (`Shape.isValid`, `Shape.Volume`
via BRepGProp, `Shape.BoundingBox`).

**Tier: T3, construct-aligned in part (decided 2026-11-13).** Under R15-1 the tool is the engine that computes the score: the
reward's engine is OCCT's mass properties and bounding box; the metric's is numpy line integration over a
triangulation. The two share the CadQuery/OCCT kernel that *interprets* the program (there is no other way to turn
CadQuery code into geometry) and OCCT's tessellation feeds the metric; the registration should disclose both. The
constructs are related, not identical: both reward agreement with the reference solid, and a valid solid is
necessary for both. IoU is strictly stronger: **reward-hacking probe** (host, all 300 items of the first single item set, 15 families): the box that exactly
fills each reference's bounding box earns reward **0.902** on average but IoU **0.755** (open box 0.71 vs 0.28,
L-bracket 0.73 vs 0.33, tube 0.82 vs 0.56, box 1.00 vs 1.00). So a pass shows that the generated solids agree better
with the references under an independent engine, partly through the same validity requirement the reward pays for
(registration disclosure `construct_alignment`; the report should show `executes` and `iou` given `executes`
beside the gated mean). Supporting arm **`valid`** (binary, [0, 1]): executes and yields a valid solid, the same
OCCT check as the reward, **T0**, supporting only. The spec's row-6 text (`valid` deciding, T0, MM-GAP-12) predates
B18 and this decision; the proposed replacement is in "Proposed multimodal spec row 6 text" below.

Scorer validity (G-21): exact and deterministic; two scorings of the same outputs give byte-identical `scores.jsonl` and `arm-iou/items.jsonl`; reference
re-execution is byte-stable (identical STL hashes across builds). Scoring speed on the host: 300 items in 35-90 s with 4-8 workers
(execution 1-3 s, geometry the rest).

**MM-38 on the deciding metric (CPU, HF fp32 greedy; the GPU smokes decide).**

| Gate | Baseline | Items (pilot) | Mean IoU | sd | Mode (share) | At 1.0 | MM-38 |
|---|---|---|---|---|---|---|---|
| SFT gate | base Coder-0.5B | 50 of `items-sft` (5 per family) | 0.344 | 0.435 | 0 (0.48) | 4 % | PASS, close to the mode bound |
| GRPO gate | SFT proxy cut 30 (10-family SFT) | 150 of `items-grpo` (30 per family) | **0.479** | 0.420 | 0 (**0.25**) | 20 % | **PASS** |
| (for reference) GRPO gate items | base Coder-0.5B | 50 of `items-grpo` | 0.116 | 0.233 | 0 (0.46) | 0 % | PASS |
| (rejected split A) GRPO gate items | SFT proxy cut 30 | 150 | 0.104 | 0.223 | 0 (0.74) | 0 % | FAIL |

Per family, SFT proxy cut on the GRPO gate items (30 each; mean / at 0 / at 1.0): blind hole 0.13 / 16 / 0, four-hole
plate 0.87 / 0 / 0, tube 0.17 / 21 / 0, flange 0.23 / 1 / 0, **chamfered box 1.00 / 0 / 30** (the cut transfers the
filleted box's template exactly). So 4 of 5 families leave GRPO room and one (20 % of the items) is at the ceiling
already (open question 6). The SFT cut on its own families (SFT gate items, 50): mean 0.915, 78 % at 1.0 - expected
for the SFT gate's candidate, which MM-38 does not judge. Caveat: the proxy saw 240 examples; the GPU SFT full run
sees 1,920, which may move held-out transfer either way, so the GRPO smoke's 20-item baseline (the SFT full cut on
`items-grpo`) is the binding check.

## Items and power (proposal)

Two item sets of **N = 300**, **K = 1**, `eval_seeds [11]` (greedy: a seed draws nothing; it is recorded):
`items-sft` (30 per SFT family) for the SFT gate, `items-grpo` (60 per held-out family) for the GRPO gate.
Single deciding clause per gate, superiority, δ 0, one-sided α 0.025, power 0.8, n = ((1.960 + 0.842) σ_d / Δ)^2.
- **SFT gate:** CPU pilot base vs SFT proxy cut on 50 `items-sft`: mean difference **+0.571**, **σ_d = 0.422**
  (the all-family pilot gave +0.659, σ_d 0.425). n = 140 / 63 / 35 for Δ = 0.10 / 0.15 / 0.20; at N = 300 the MDE
  is 0.068: far over-powered.
- **GRPO gate:** σ_d of SFT cut vs GRPO cut is unknown (no GRPO on CPU). For scale, base vs SFT proxy cut on 50
  `items-grpo` gave σ_d 0.405 (difference +0.355); with σ_d about 0.4 the MDE at N = 300 is 0.065, at N = 600 0.046.
  Proposal: `sigma_plan: internal_pilot` from the GRPO smoke (as areas 1-3), planning effect set then; if the smoke
  shows a smaller effect than the MDE, raise N (the generator draws more items under the same disjointness rule;
  a new item set means a new `item_set_hash` before registration). `min_paired` 10, `max_unpaired` 0.

## Trainer details (`run_cad_nemo.py`, `ptc_cad_hooks.py`, `cad_env.py`)

`run_cad_nemo.py --stage sft|grpo --profile smoke|full --out /out --seed S --base <models/...> [--init <SFT cut>
--socket /spool/cadexec.sock]` records the run (`run_args.json`, `env-freeze.txt` with the NeMo RL commit read from
`/opt/nemo-rl/.git`, `attempts.jsonl`), installs the hooks, sets NeMo RL overrides, then calls the unmodified
`examples/run_sft.py` or `examples/run_grpo.py` `main()` with `sys.argv` = `--config <recipe> <overrides>`, so NeMo RL
parses them as on its command line. Overrides, common: model and tokenizer = base (SFT) or the SFT cut (GRPO),
`++policy.tokenizer.chat_template=default` (sft.yaml sets a Llama template; grpo_math_1B.yaml has no such key), `chat_template_kwargs={enable_thinking:
false}` (Tier B), `max_total_sequence_length=1280` (prompt v2 570-657 tokens with the chat template, measured, + 512 new), `data.shuffle=false`, `ResponseDataset` on this area's
`train.jsonl` with `input`/`output`, no validation split (`++data.train.split_validation_size=0`: sft.yaml has no such key; the first SFT smoke failed on it at config parsing) or set, no prompt or system-prompt file, checkpointing on with
`keep_top_k=null`, `save_consolidated=every`, `activation_checkpointing=true` (Tier B), wandb and tensorboard off.
SFT: global batch 32, micro 4 (micro 8 reached the 24 GB device ceiling in SFT smoke 2; the 151,936-token logits of 8 x 1280 tokens), AdamW lr 1e-5, `sft.seed=S`, validation off (the sample log depends on it). GRPO: 16
prompts x 8 generations per step, global batch 128, micro 4, `max_new_tokens=512`, vLLM `gpu_memory_utilization=0.25` (0.4 peaked at 23.6 GB on the device in GRPO smoke 2, over the 22.5 GB smoke bound), sequence packing 2,560 tokens per microbatch for training and log-probabilities (5,120 peaked at 23.6 GB in the training phase with vLLM at 0.25 too, GRPO smoke 3)
(Tier B fallback), `data_plane.enabled=true` (Tier B's `grpo_sync` path), `math_hf_data_processor`, `env_name=cad`,
`env.cad.{socket, log_path, timeout_s}`, KL and loss settings of `grpo_math_1B.yaml` unchanged; before training the
launcher pings the executor with one program and stops if it fails.

- **Observed SFT consumption, `row`.** `nemo_rl.algorithms.sft.prepare_sft_batch` is wrapped; `sft_train` calls it
  once per step, before `policy.train` (sft.py:736-757). Each batch's `idx` (collate_fn.py:228) is the 0-based line
  of `train.jsonl`; each line is checked against the batch text (the row's `input` must occur in the logged user
  message) and written with `{step, epoch, position, idx, row_id, row = sha256(row_id), attempt, utc}` to
  `samples.jsonl` and `{step, epoch, n_examples, example_ids}` to `train_steps.jsonl`, fsynced. Validation also calls
  `prepare_sft_batch` (sft.py:579), so validation stays off.
- **Observed GRPO consumption, `row`.** The environment writes one line per sample at reward time
  (`env_calls.jsonl`: `{call, position, row_id, status, reward, program_sha256, response_sha256}`); NeMo RL writes its
  own per-step log `nemo/logs/exp_*/train_data_step<N>.jsonl` with the step's rewards in order (grpo_sync.py:1459,
  Tier B saw it). Synchronous GRPO without dynamic sampling scores exactly 128 samples per step, so step N owns
  env-log lines 128(N-1)+1 .. 128N; the adapter must check that those lines' rewards equal the step's logged rewards
  (refuse otherwise) and that each row_id appears 8 times per step. Fallback `iteration` (MM-22: file order, 16 rows
  per step). Both are verified at the smoke (open question 4).
- **Atomic cuts, `cut.json` last.** NeMo RL writes `nemo/checkpoints/tmp_step_N` and renames it to `step_N`
  (checkpoint_storage.py:80-97, checkpoint.py:423-506). The hook wraps that `publish` and then builds
  `cuts/checkpoint-N/` in `cuts/.checkpoint-N.partial/`: hard links to the consolidated `model.safetensors` and
  `config.json` (`policy/weights/model/consolidated/`, Tier B layout) and the tokenizer files (`policy/tokenizer/`),
  `generation_config.json` copied from the base, `sampler.json` (the evaluator's deployment sampler), `base_ref.json`
  (base repo, revision, file hashes, and for GRPO the SFT cut it started from), `training_info.json`; every file
  fsynced; `cut.json` written last (`global_step`, `examples_consumed`, per-file SHA-256, `deployable`, `seed`,
  `profile`, `stage`, `script_sha256`; no wall-clock field), directory fsynced, renamed; `cuts.jsonl` appended. A
  failure raises in NeMo RL's finalize thread, which fails the run (finalize_pending re-raises, checkpoint.py:494-506).
- **Seeds (MM-29).** `sft.seed` / `grpo.seed` = the launch seed (6000 SFT, 6100 GRPO, proposal); the data order is
  the file order; vLLM's sampling seed is whatever NeMo RL derives (not claimed). Recorded in `run_args.json`.
- **Preemption.** As Tier B: a rerun starts from scratch (`rm -rf` of the run's outputs first); `attempts.jsonl`
  counts attempts and the logs carry the attempt.

## Outputs

| File (`/out`) | Written | Purpose |
|---|---|---|
| `run_args.json`, `env-freeze.txt`, `attempts.jsonl` | first | settings, overrides, seed, base/init/data bindings and hashes; NeMo RL commit and package versions |
| `samples.jsonl`, `train_steps.jsonl` (SFT) | before each `policy.train`, fsync | observed consumption |
| `env_calls.jsonl` (GRPO) | at each reward call, fsync | observed consumption, executor status per sample |
| `nemo/logs/exp_001/...`, `nemo/checkpoints/step_N/` | NeMo RL | its own logs (`train_data_step<N>.jsonl` for GRPO) and checkpoints (resume state) |
| `cuts/checkpoint-N/`, `cuts.jsonl` | after each published checkpoint | the cuts |

Evaluator, three steps: (1) `gen_cad.py` (GPU, NeMo RL image, vLLM backend) writes `gen/<item>__s<seed>.json` and
`gen_meta.json`; (2a) `executor/cad_sandbox.py --run-batch` (executor container, no references) writes
`programs/<item>__s<seed>/{result.json, mesh.stl}`, `exec_timing.jsonl` and, last, `exec_meta.json` (versions,
limits, ids, `gen_meta.json` SHA-256, code hashes, SHA-256 of every file but the timing file; two runs give identical
`exec_meta.json`); (2b) `score_cad.py --gen --exec --items` (scoring container, runs no generated code) writes `outputs/<item>__r<k>.json`, `scores.jsonl`, `summary.json`,
`env-runtime.json`, byte copies of the generation step's `gen_meta.json` (with `model_files`) and the execution step's `exec_meta.json`, `arm-iou/` (evaluator
`cad.cadquery_items@0.1`) and `arm-valid/` (evaluator `cad.cadquery_valid_items@0.1`), each with MM-1 `items.jsonl`,
MM-2 `eval_meta.json` and hard-linked outputs. Arm `sub_metrics` are finite numbers only (`executes`, `chamfer`,
`volume_ratio`, `bbox_iou`, `reward`, `n_new_tokens`, `unbalanced_columns`, `mc_volume_rel_err`, `status_code` =
index in `cad_sandbox.STATUSES` (0 ok, 1 no_code, 2 syntax_error, 3 runtime_error, 4 no_result, 5 not_solid,
6 invalid_solid, 7 timeout, 8 resource_limit, 9 crash), `truncated`); a value that does not exist is left out, not
null; status names and failure details stay in `scores.jsonl` and the output files. A failed program is a score
(0, `error` null) with its `status_code`. `framework` = `{name, version}` of the generation backend (`vllm` on the
GPU, `transformers` for the CPU pilot). Scorer `code_sha256` (both arms) = SHA-256 of the canonical JSON of the
four files' hashes (`score_cad.py`, `geom_metrics.py`, `executor/cad_sandbox.py`, `executor/cad_protocol.py`;
keys are the base names), currently `sha256:958ab45f…16e6` (`SCORE_CAD_CODE_SHA256` in the adapter's `rows.py`).
`sampler_hash` = SHA-256 of the RFC 8785 canonical JSON of `sampler`, the encoding gatedtrain's ext-collect recomputes
(`score_cad.canonical` copies gatedtrain's encoder; before the fix the scorer wrote compact sorted JSON, which differs
for the sampler's floats 0.0 and 1.0, and only `ext-arm-fill` made such an arm collectable).
`gen_cad.py` checks a cut before loading it (every `cut.json` hash, no unlisted file).

## Profiles and cost

All GPU numbers are **estimates** (no GPU was used in preparation). Basis: Tier B NeMo RL GRPO on an RTX 4090 with
Qwen3-0.6B, 256 samples x up to 1,024 tokens per step, activation checkpointing: 1.22 GPU-h for 150 steps, about
29 s per step including saves; Tier B smoke 10 steps of 8 samples: 12 min 21 s wall including startup
and host-RAM thrashing. Area-6 steps are smaller (128 samples; prompt 570-657
tokens, programs 30-120 tokens after SFT, at most 512 new tokens) on a smaller model (494 M).

| Run | Steps x batch | Examples | Cuts | GPU time (estimate) | Basis |
|---|---|---|---|---|---|
| SFT smoke | 10 x 32 | 320 | 5, 10 | 0.07 h | about 3 min startup (Ray, policy workers), about 1 s per step, two saves |
| SFT full | 60 x 32 | 1,920 (0.96 epoch) | 20, 40, 60 | 0.10 h | as smoke, three saves of about 30 s (consolidated 1 GB + DCP state with optimizer) |
| GRPO smoke | 5 x (16 x 8) | 80 prompts, 640 samples | 5 | 0.11 h | about 4 min startup (Ray, vLLM, policy), 15-30 s per step |
| GRPO full | 60 x (16 x 8) | 960 prompts (0.48 epoch), 7,680 samples | 20, 40, 60 | 0.45 h (0.35-0.6) | 20 s per step (15-30), of which executor about 1 s (CPU, blocks the GPU; at most 15 s when a program hangs) |
| Evaluation, generation | 300 items x 1 (greedy) | | | 0.04 h per arm; 0.03 per 20-item smoke arm | vLLM startup about 2 min; 300 x <= 512 tokens on a 0.5 B model well under a minute |
| Evaluation, scoring | | | | 0 (CPU, executor container) | host: 300 items about 35 s (8 workers) |

Area total with the two item sets: SFT smoke 0.07 + 2 smoke arms on `items-sft` (base, smoke cut) 0.06 + SFT full
0.10 + 2 arms on `items-sft` (base, SFT cut) 0.08 + GRPO smoke 0.11 + 2 smoke arms on `items-grpo` (SFT cut, GRPO
smoke cut) 0.06 + GRPO full 0.45 + 2 arms on `items-grpo` (SFT cut, GRPO cut) 0.08 = **about 1.0 GPU-h (range
0.85-1.4)**; the SFT cut's `items-grpo` arm is needed because the gates use different item sets. The executor's
CPU time is inside the GRPO step estimate; scoring needs no GPU. Host: NeMo RL's offload used 18 GB of RAM in Tier B
(Qwen3-0.6B); the 0.5 B model needs less, and the executor container is capped at 8 GB.

**CPU alternative (not recommended):** the CPU SFT proxy ran at 29.8-39.7 s per step of 8 examples
(`pilot/cpu-sft10b/`, `pilot/v1-allfam/cpu-sft/`), so the full SFT stage would take about 2-2.6 h of CPU (240 steps of 8) and GRPO is not practical on CPU
(generation alone: 7.8-21.6 s per response with HF fp32).

## CPU sanity check (2026-11-13, no GPU, no container)

| Check | Result |
|---|---|
| `uv lock` / `uv sync --frozen` (executor) | 46 packages, 1.8 GB; `import cadquery` 1.1 s, 500 MB RSS, 3 GB address space |
| Executor self-test (`executor/cad_sandbox.py --self-test`) | 23 / 23 PASS (statuses, CPU / memory / wall limits, 5 escape attempts and 8 reference-read attempts refused) + zygote memory probe PASS (positive control found, 0 reference markers in 1,010 MiB) |
| Executor stress | 30,000 executions of the 300 references, 16 workers: 30,000 ok, 0 crashes (after the OCCT one-thread fix; before: 1-14 SIGSEGV per 3,000-6,000) |
| Reference isolation, no change in results (after review 11) | `--run-batch` + `score_cad.py --exec` on the 150 pilot GRPO-gate responses: IoU, reward and status identical to the old in-process scorer for 150 / 150; on 15 items `arm-iou/items.jsonl`, `arm-valid/items.jsonl` and `scores.jsonl` byte-identical to before the change; two `--run-batch` runs give identical `exec_meta.json`; a stray file in the exec output gives `executor_error`; the serve path (`cad_env.CadEnvCore` with the items' ground truth) gives the same reward for 150 / 150 (batch of 150 in 0.47 s) |
| Executor socket (sidecar protocol, `cad_env.CadEnvCore` against `--serve`) | 128 programs per batch in 0.68 s; statuses ok / runtime_error / no_code as constructed; rewards: bbox-exact box 0.92, wrong box 0.24, failures 0; `env_calls.jsonl` written and fsynced per line |
| Data build (split B) | 2,000 SFT (10 families) + 2,000 GRPO (15) + 300 SFT gate items (10) + 300 GRPO gate items (5), 15 s; 0 reference failures; 0 overlap in all six pairs; two builds byte-identical (615 files) |
| Metric | oracle (reference programs as responses): 300 / 300 IoU exactly 1.0; perturbed oracle (1/3 shifted 3 mm, 2/15 without `result`): mean 0.796, `no_result` 40; two scorings byte-identical (`arm-iou/items.jsonl`, `scores.jsonl`); mesh volume vs OCCT volume median 0.23 %, 95th percentile 0.9 %, max 4 % |
| Reward-hacking probe | bbox-exact box for every item: reward 0.902, IoU 0.755 |
| Base-model generations (HF fp32, CPU, 28 threads) | Coder-0.5B, prompt v2: 60 items of the earlier single item set 21.6 s per response, 32 / 60 execute, mean IoU 0.273 (`pilot/v1-allfam/base-coder05*`); 50 `items-sft`: mean 0.344, mode share 0.48 (`pilot/base-sftgate*`); 50 `items-grpo`: mean 0.116, mode share 0.46 (`pilot/base-grpogate*`); earlier prompts and Qwen3 in "Model choice" |
| Training steps on CPU | NeMo RL cannot run on CPU (GPU image, Ray + vLLM + CUDA policy workers). Proxy `sft_cpu_pilot.py` (plain transformers, same rows, order and chat template, loss on assistant tokens), three runs of 30 steps x 8 = 240 examples, lr 1e-5, seed 6000: all-family set (`pilot/v1-allfam/cpu-sft`, 39.7 s per step, loss 0.67 -> 0.009), split A (`pilot/v2-split-a/cpu-sft10`), split B (`pilot/cpu-sft10b`, 29.8 s per step, 14 min 53 s); peak RSS 29 GB; each wrote `samples.jsonl` 240 lines and `train_steps.jsonl` 30 steps x 8 ids (rows sft-00000 .. sft-00239 in file order) |
| Cut write and reload | cuts written through `.checkpoint-N.partial/` with `cut.json` last; all file hashes re-checked, no unlisted file; `gen_cad.py` re-verified each cut's hashes before loading it. Split B cut 30: 150 `items-grpo` (4.1 s per response) mean IoU 0.479, mode share 0.25; 50 `items-sft` mean 0.915 (`pilot/cut30-*`). The 2 GB fp32 weights were deleted afterwards (the `cut.json` files are kept in `pilot/*/cut-json/`) to keep the read-only mount small |
| Not tested (needs GPU image or docker) | `run_cad_nemo.py`, `ptc_cad_hooks.py` hook points, `cad_env.CadQueryEnvironment` as a Ray actor, `gen_cad.py --backend vllm`, the executor container and its isolation flags |

## What gatedtrain needs (adapter `nemo_rl_ext`; trainer ids `sft.nemo_rl@0.1`, `rl.nemo_rl_env@0.1`)

Two `ext-collect` runs (MM-15): SFT (base -> SFT cut) and GRPO (SFT cut -> GRPO cut). The GRPO spec's baseline
`model` is the SFT run's gated cut; its manifest hash must equal the SFT run's `model_hash` of that cut.

1. **Cuts and completeness (MM-20).** `<run>/cuts/checkpoint-(\d+)/` with `cut.json` whose `files` SHA-256 all match,
   whose `global_step` equals the directory number, whose `stage` is the run's stage, and that contains no unlisted
   file; `cuts/.checkpoint-*.partial/` or a directory without `cut.json` is incomplete. NeMo RL's own
   `nemo/checkpoints/step_N/` (and `tmp_step_N`) are resume state, never cuts. The last cut's step equals
   `run_args.json` `max_steps`. `last_step_included` = `global_step` (SFT: optimizer step; GRPO: GRPO step).
2. **Identity (MM-17).** Full weights, no LoRA, no composition. `model_hash` = manifest hash over `cut.json`
   `deployable` (`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `sampler.json`, plus `chat_template.jinja` / `vocab.json` / `merges.txt` when NeMo RL
   saved them, to be confirmed at the smoke); `cut_hash` adds `base_ref.json`, `training_info.json`, `cut.json`.
   Readable without torch (JSON, safetensors header).
3. **Base binding (MM-21).** `base_ref.json` `{repo, revision, files}` = the chosen base in `models/MODELS.json`
   (`Qwen/Qwen2.5-Coder-0.5B-Instruct` @ `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`); SFT `run_args.json` `model_name_or_path` = `base_model` = the base directory in
   the read-only mount. GRPO: `run_args.json` `init_checkpoint` `{path, global_step, cut_json_sha256}` names the SFT
   cut (read-only mount `/inputs/sft-cut`), checked against the SFT run (EXT-101 analogue of area 2's continuation).
   Baseline arm of the SFT gate: `checkpoint.model_hash` = the base directory's manifest hash (8 files); note the
   base's tokenizer files differ from a cut's (`vocab.json` + `merges.txt` vs NeMo RL's saved set).
4. **Data binding (EXT-65).** SFT spec data = `data/sft/` (shards `shards/shard-0000{0..3}.jsonl`; `train.jsonl` is
   their concatenation), `run_args.json` `train_file` = `<runtime>/data/sft/train.jsonl`,
   `train_file_sha256` = `sha256:517e3c05fba7f267d5278cd8f11e20e0bbabf8e2f81a4a76e0356f34b1ead66c`. GRPO: `data/grpo/`,
   `sha256:a4c74e1743fc28e09eea4e288fda1b9abbfe02607cd39b113864deb7251724b7`.
5. **Seed and pins.** `run_args.json` `seed` = launch `training_seed`; `env-freeze.txt` carries the NeMo RL commit
   line (`# nemo-rl commit 4f39fd8...`) and the image's packages; the launch record's framework is
   `{name: nemo_rl, commit: 4f39fd86289a2ffd5aaf5d202702f7fd7185c924, image_id: sha256:8cd5b714...}`; the executor
   pins (cadquery 2.8.0, OCP 7.9.3.1, executor `uv.lock`) are in each arm's `scorer.version` and
   `env-runtime.json`; `run_args.json` `script_sha256`, `hooks_sha256`, `env_sha256`, `protocol_sha256`.
6. **Consumption.** SFT: precision `row`, derivation `observed`, one ledger record per `samples.jsonl` line:
   `data_ref.snapshot` = manifest hash of `data/sft/`, shard = the `shards/` file holding the row (row k in
   `shard-{k // 500}`), `row` = the line's `row`, `step`, `sample_id` = H(run, attempt, epoch, step, row_id)
   (NUL-joined as areas 3-4), `rewards` null, `seed_inputs` null. Refuse when: steps are not consecutive from 1; a
   step's ids differ from `train_steps.jsonl`; a step does not carry exactly 32 ids (drop_last: 62 full steps per
   epoch of 2,000); a row of a held-out family appears in the SFT log (refuse: the split is part of the spec); a row_id or idx is not that line of `train.jsonl`; epoch/position disagree with file order; a
   cut's `examples_consumed` != 32 x step. GRPO: precision `row`, `observed`, one record per (step, sample) from
   `env_calls.jsonl` lines 128(N-1)+1..128N, `rewards` = [reward], `sample_id` = H(run, attempt, step, row_id,
   member), joined with `train_data_step<N>.jsonl` (rewards must match in order); refuse on any mismatch, on a step
   with other than 16 distinct rows x 8, or on rows out of file order (step N must use rows 16(N-1)..16N-1 of the
   epoch). If the join fails at the smoke, fall back to MM-22 `iteration` exactly as Tier B.
7. **Item sets and exact overlap (MM-11).** Two item sets: the SFT run's spec names `items-sft/` (`item_set_hash`
   over `items-sft/`, 300 items, 10 families), the GRPO run's spec names `items-grpo/` (300 items, the 5 held-out
   families). The GRPO run's baseline arm is the SFT cut evaluated on `items-grpo` (a separate arm from the SFT gate's
   candidate arm on `items-sft`; same `model_hash`). Overlap: `payload_text_field: "text"` on both sides (items'
   `text` = training rows' `text` = the full prompt); each gate's items against both training sets of the chain
   (SFT and GRPO shards): 0 by construction, and 0 geometries shared (stronger than text). Optional family check for
   the GRPO gate: no `items-grpo` family occurs in `data/sft/` (`family` field), as `make_data.py` enforces.
8. **Arms.** `arm-iou/` (deciding, `cad.cadquery_items@0.1`) and `arm-valid/` (supporting, `cad.cadquery_valid_items@0.1`) from `score_cad.py`, plus the copied `gen_meta.json`; each evaluation is three launches (generation GPU; execution and scoring CPU, `gpus: false`), and ext-launch should refuse an execution launch that mounts the area root or any `items-*`/`data/` path; `ext-arm-fill` supplies
   `model_hash`, `item_set_hash` and the RFC 8785 `sampler_hash` (the published gates' scorer wrote compact sorted
   JSON, which differs from RFC 8785 for the floats 0.0 and 1.0, and ext-arm-fill converted it; the scorer now writes
   the RFC 8785 hash itself, which ext-arm-fill verifies). `framework` in
   `eval_meta.json` is the generation backend's versions (vLLM) from `gen_meta.json`. Collect should check that every
   output's `program` is what `cad_protocol.extract_program` gives for its `response`, and that `gen_meta.json`
   `model_files` equal the evaluated cut's files.
9. **Registry rows (proposal).** Trainers `sft.nemo_rl@0.1` and `rl.nemo_rl_env@0.1`: `framework "nemo_rl"`, pins
   (NeMo RL commit, image ID; cadquery for the env), consumption `{target: row, fallback: iteration}`,
   `gpu_h_estimate` SFT 0.25, GRPO 0.6 (including arms), `input_roots`: the Python directory,
   `<runtime>`, `<output-root>`; `scan_skip`:
   `<runtime>/executor/.venv` (13,854 files) and the interpreter's site-packages; `interpreter`: the image's
   (trainer, open question 2) and `.../executor/.venv/bin/python` (executor). Evaluator `cad.cadquery_items@0.1`:
   metric `iou`, `higher_is_better`, `score_range (0, 1)`, `samples "300, 1"` (per item set), `error_classes ("executor_error",
   "scorer_error", "missing_output")`, `score_outcomes ("no_code", "syntax_error", "runtime_error", "no_result",
   "not_solid", "invalid_solid", "timeout", "resource_limit", "crash")`, `payload_text_field "text"`,
   `item_payload ("text", "reference_code", "ref_mesh", "ref_mesh_sha256", "ref_volume", "ref_bbox")`, `binary
   false`, tier T3 with `construct_alignment` disclosure (decided); supporting `valid`: `binary true`, T0. The
   paired executor sidecar of the GRPO launch (decided 2026-11-13) needs its own launch-record fields: image ID, argv,
   container name `gatedtrain-cadexec-<label>`, mounts (Python tree and `<area>/executor` ro only, `<run>/spool` rw; never
   the area root), `--network none`, user 1000:1000, `--cap-drop ALL`,
   `no-new-privileges`, `--read-only`, tmpfs, `--cpus 8 --memory 8g --memory-swap 8g --pids-limit 256`, and start
   before / stop after the trainer (`executor/run-executor.sh serve` prints the exact command).

## Launch request facts

Four container kinds. Mounts at their host paths unless stated. `<runtime>` is this directory, `<output-root>` the
area's output directory, `<python-root>` the directory holding the CPython tree. The area directory is mounted (read-only) only
where no generated code runs (trainer, generation, scoring); the two containers that execute generated code see only
the Python tree and `<area>/executor` (review 11).

**Trainer (NeMo RL image, GPU).**

| Host | Container | Mode |
|---|---|---|
| `<runtime>` (scripts, data, models, items) | same | ro |
| GRPO only: `<output-root>/<sft-run>/cuts/checkpoint-<S>` | `/inputs/sft-cut` | ro |
| GRPO only: `<output-root>/<grpo-run>/spool` (holds the executor socket) | `/spool` | rw |
| `<output-root>/<run>` (new, empty) | `/out` | rw |

`--runtime=nvidia --gpus all` (CDI, gatedtrain's docs/tier-b-runbook.md §8), `--shm-size=16g`, `--ulimit memlock=-1 --ulimit
nofile=65535:65535`, `-e RAY_memory_monitor_refresh_ms=0` (as Tier B), `-e
TORCH_CUDA_ARCH_LIST=8.9 -e FLASHINFER_CUDA_ARCH_LIST=8.9`, `-e HF_HUB_OFFLINE=1`, `-w /opt/nemo-rl`; user: the
image default (root, as Tier B; open question 3), network: `none` proposed (Tier B used `host` because its Docker
daemon had no bridge; NeMo RL needs no network once models are local; Ray on loopback is unverified, open question 3).

```
sft smoke:  uv run python <runtime>/run_cad_nemo.py --stage sft --profile smoke
            --out /out --seed 6000 --base <runtime>/models/qwen2.5-coder-0.5b-instruct
            label mm-a6-cad-sft-smoke, gpu_est_h 0.07
sft full:   ... --stage sft --profile full ...                       label mm-a6-cad-sft-full, gpu_est_h 0.10
grpo smoke: ... --stage grpo --profile smoke --out /out --seed 6100 --base <same> --init /inputs/sft-cut
            --socket /spool/cadexec.sock                             label mm-a6-cad-grpo-smoke, gpu_est_h 0.11
grpo full:  ... --stage grpo --profile full ...                      label mm-a6-cad-grpo-full, gpu_est_h 0.45
```

**Executor sidecar (CPU, during GRPO only; runs generated code).** Started before the GRPO container, stopped
after it; `executor/run-executor.sh serve <grpo-run dir> <label>` prints both commands. Container name
`gatedtrain-cadexec-<label>` (the run label, e.g. `gatedtrain-cadexec-mm-a6-cad-grpo-full-1`), so the printed `docker stop` names
only this run's sidecar. No GPU. `ext-launch` gets a paired sidecar record for it (decided 2026-11-13).

| Host | Container | Mode |
|---|---|---|
| `<python-root>` | same | ro |
| `<runtime>/executor` (cad_sandbox.py, cad_protocol.py, `.venv`, lock, Containerfile, run-executor.sh) | same | ro |
| `<output-root>/<grpo-run>/spool` | `/spool` | rw |
| tmpfs | `/tmp` | `rw,nosuid,nodev,noexec,size=4g` |

argv `<runtime>/executor/.venv/bin/python -I <runtime>/executor/cad_sandbox.py --serve /spool/cadexec.sock
--workers 8`.

**Evaluation arm, three steps.** (1) Generation, NeMo RL image, GPU: `/opt/ray_venvs/nemo_rl.models.generation.vllm.
vllm_worker.VllmGenerationWorker/bin/python <runtime>/gen_cad.py --backend vllm
--model <base dir | /inputs/cut> --items <runtime>/items-sft (SFT gate) or items-grpo (GRPO gate) --out /out
--eval-seeds 11` (the worker venv Tier B served with, vLLM 0.29.0; add `--limit 20 --round-robin` for smoke arms),
gpu_est_h 0.04 (0.03 smoke); mounts as the trainer (area ro, cut ro at `/inputs/cut`, `/out` rw).

(2a) Execution, executor container, CPU, **runs generated code**: `executor/run-executor.sh exec <gen out> <exec out>`.

| Host | Container | Mode |
|---|---|---|
| `<python-root>` | same | ro |
| `<runtime>/executor` | same | ro |
| `<gen out>` (step 1's `/out`: `gen/`, `gen_meta.json`) | `/inputs/gen` | ro |
| `<exec out>` (new, empty) | `/out` | rw |
| tmpfs | `/tmp` | `rw,nosuid,nodev,noexec,size=4g` |

argv `.../executor/.venv/bin/python -I .../executor/cad_sandbox.py --run-batch --gen /inputs/gen --out /out
--workers 8`; 0 GPU-h, seconds.

(2b) Scoring, same image, CPU, **runs no generated code**: `executor/run-executor.sh score <gen out> <exec out>
<score out> <items-sft|items-grpo>`.

| Host | Container | Mode |
|---|---|---|
| `<python-root>` | same | ro |
| `<runtime>` (score_cad.py, geom_metrics.py, items, executor venv) | same | ro |
| `<gen out>` | `/inputs/gen` | ro |
| `<exec out>` | `/inputs/exec` | ro |
| `<score out>` (new, empty) | `/out` | rw |
| tmpfs | `/tmp` | `rw,nosuid,nodev,noexec,size=4g` |

argv `<runtime>/executor/.venv/bin/python -I <runtime>/score_cad.py --gen /inputs/gen --exec /inputs/exec --items
<runtime>/<items-sft|items-grpo> --out /out --eval-seeds 11 --workers 8 --model-hash <H> --item-set-hash <H>`;
0 GPU-h, about 1 min. All three executor-image containers: `--network none --user 1000:1000 --cap-drop ALL
--security-opt no-new-privileges --read-only --cpus 8 --memory 8g --memory-swap 8g --pids-limit 256 --ulimit
nofile=1024:1024 --ulimit core=0`, environment `HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1`.

Environment names (no credential): `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`,
`HOME=/tmp` (executor), `RAY_memory_monitor_refresh_ms`, `TORCH_CUDA_ARCH_LIST`, `FLASHINFER_CUDA_ARCH_LIST`.
`framework`: `{name: "nemo_rl", version: null, commit: "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"}`; `runtime`:
`{kind: "image", image_id: "sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a"}` for trainer
and generation, `{kind: "venv_in_container", image: <gatedtrain-cadexec digest after build>, venv_lock:
"<runtime>/executor/uv.lock"}` for the executor.

## Proposed multimodal spec row 6 text

Training side, row 6:

| # | Adapter, `uses` | Pin to record | Outputs read | `model_hash` components | Consumption (target / fallback) | GPU-h |
|---|---|---|---|---|---|---|
| 6 | `nemo_rl_ext`, `sft.nemo_rl@0.1` then `rl.nemo_rl_env@0.1` | NeMo RL commit `4f39fd8` and Tier B image ID; executor: cadquery 2.8.0, OCP 7.9.3.1, executor `uv.lock` and image ID; base `Qwen/Qwen2.5-Coder-0.5B-Instruct` @ `ea3f2471` | `cuts/checkpoint-<step>/` full-weight cuts with `cut.json` last (built from NeMo RL's published `step_N`), `run_args.json`, `env-freeze.txt`; SFT `samples.jsonl` + `train_steps.jsonl`; GRPO `env_calls.jsonl` + NeMo RL `train_data_step<N>.jsonl` | weights, config, generation config, tokenizer files, deployment sampler `sampler.json` | `row`, observed (SFT sample log; GRPO env log joined with NeMo RL's step log) / `iteration` (MM-22) | 1.0 |

Evaluation side, row 6:

| # | Evaluator id | Approach | Metric, direction, range | Items n, K | Score vs error | Tier, risk |
|---|---|---|---|---|---|---|
| 6 | `cad.cadquery_items@0.1` | two steps: greedy generation (vLLM in the NeMo RL image, at most 512 new tokens, prompt v2), then execution and scoring in the B18 executor container (one forked process per program, rlimits, 15 s wall; container `--network none`, uid 1000, no capabilities, read-only root, CPU / memory / pids limits); volumetric IoU against the item's reference mesh by vertical-line integration (numpy) | `iou`, higher, [0, 1]; sub-metrics status, executes, chamfer, volume ratio, bbox IoU, reward | SFT gate: `items-sft`, 300 from the 10 SFT families, 1; GRPO gate: `items-grpo`, 300 from the 5 families held out of SFT, 1 | any program that yields no valid solid scores 0 (`no_code`, `syntax_error`, `runtime_error`, `no_result`, `not_solid`, `invalid_solid`, `timeout`, `resource_limit`, `crash`); `executor_error`, `scorer_error`, `missing_output` are errors | T3, construct-aligned with the OCCT reward in part (R15-1: shared CadQuery/OCCT kernel, tessellation and validity requirement; disclosure `construct_alignment`); B18 prototype deviation (decision 2026-11-13) replaces MM-GAP-12's block |
| 6s | `cad.cadquery_valid_items@0.1` | as row 6, `arm-valid` of the same scoring run | `valid`, higher, [0, 1], binary | as row 6 | as row 6 | T0 (the reward's validity check; supporting, never deciding) |

Row 6 deviations (proposal): data generated by `make_data.py` (15 parametric families, no third-party data);
5 families held out of SFT (blind hole, four-hole plate, tube, flange, chamfered box), GRPO on all 15; two item
sets, one per gate, disjoint from each other and from both training sets in geometry and prompt text; the GRPO spec's
baseline is the SFT run's gated cut evaluated on `items-grpo` (MM-15); the executor runs as a paired sidecar of the
GRPO launch (ext-launch extension, decided 2026-11-13); runtime directory `area6-cad`, request `area` 6.

## Open questions (at preparation time, 2026-11-13)

Marked **[choice]** where a design or option choice is involved, **[routine]** otherwise.
Decided on 2026-11-13: trainer NeMo RL with TRL as fallback; `iou` T3 with construct-alignment disclosure,
`valid` supporting; ext-launch paired sidecar; GRPO headroom by holding out 5 families. The questions below remain.

1. **[routine] Build the executor image** (`executor/Containerfile`; needs `docker build` with network for apt) and
   check inside it: `import cadquery`, `executor/cad_sandbox.py --self-test` (the reference-read cases must fail with FileNotFoundError; the memory probe reports SKIP), no network interface but `lo`, uid 1000, root
   filesystem read-only, `/tmp` writable. Record its image ID in the launch records.
2. **[routine] Interpreter of the trainer container.** Tier B ran `uv run python ...` from `/opt/nemo-rl`. `ext-launch`
   wants argv[0] to be a pinned interpreter: find the venv path behind `uv run` in the image (one `docker run
   --entrypoint` command) and use it directly.
3. **[routine] Trainer container network.** Proposal `--network none` (models and data are local,
   `HF_HUB_OFFLINE=1`); Ray on loopback is unverified. If the smoke fails at Ray start, use `--network host` as Tier B
   did; the executor keeps `--network none` in every case. The trainer runs as root inside its container, as Tier B.
4. **[routine] Hooks and consumption verified at the smokes:** `samples.jsonl` steps and texts, cut layout
   (`policy/weights/model/consolidated/`, `policy/tokenizer/` file names under `save_consolidated=every`), the GRPO
   join of `env_calls.jsonl` with `train_data_step<N>.jsonl` (128 per step, rewards equal in order). If the join does
   not hold, GRPO consumption falls back to MM-22 `iteration`, exactly Tier B.
5. **[choice] If the GRPO smoke baseline is degenerate after all.** The CPU proxy saw 240 SFT examples; the GPU SFT
   full run sees 1,920, which may move transfer to the held-out families either way (split A showed SFT can make
   held-out families worse than the base). The binding check is the GRPO smoke's 20-item baseline (SFT full cut on
   `items-grpo`). If it fails MM-38: (a) gate GRPO from an earlier SFT cut (cut 20 of the full run, also a published
   cut of the SFT run, so the MM-15 chain holds; recommended); (b) shorten SFT full to 20 steps; (c) change the
   held-out set again (new data hashes, another CPU pilot of about 1 h).
6. **[choice] Chamfered box sits at the ceiling for the GRPO baseline** (30 of 30 items at IoU 1.0 for the
   SFT proxy; it reuses the filleted box's template). Options: (a) keep it (recommended: 4 of 5 families leave room,
   overall mean 0.48 and mode share 0.25 pass MM-38; the ceiling family only dilutes the effect); (b) swap it for
   another family and re-run the CPU pilot (about 1 h; candidates are limited: split A showed families with new
   operations collapse to 0 after SFT).
7. **[choice] Base model and prompt.** Qwen2.5-Coder-0.5B-Instruct with prompt v2 passes MM-38 on the SFT gate items
   only narrowly (CPU: mode share 0.48 on 50 `items-sft`). If the SFT smoke's 20-item baseline fails the mode rule:
   (a) add a third worked example to the prompt (raises executability; the prompt is shared by training and
   evaluation, so all data hashes change; recommended); (b) Qwen2.5-Coder-1.5B-Instruct (needs LoRA or CPU offload on
   24 GB; unmeasured).
8. **[routine] Power.** N = 300 per gate, K = 1 greedy; `sigma_plan: internal_pilot` from each stage's smoke. The SFT
   gate is far over-powered (MDE 0.068); the GRPO gate may need more items if the smoke shows a small effect (more
   items are a generator run, before registration).
9. **[routine] Steps.** SFT full 60 x 32 (about one epoch of the 10-family set) and GRPO full 60 x 16 x 8 are
   placeholders until the smokes have run; the GPU estimate assumes them.
10. **[routine] Mount contents.** Done: `models/qwen3-0.6b/` moved to `../area6-ref/models/` (not deleted). In the
    mount remain `pilot/` (11 MB, CPU evidence, no weights), `hf-meta/` (download metadata), empty `hf-home/`.
11. **[routine] Spec text.** Adopt the proposed row-6 text above into the multimodal spec.
