Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-14): both gates PASS

Qwen/Qwen2.5-Coder-0.5B-Instruct @ `ea3f2471`, text to CadQuery on 15 procedurally generated part families (no
third-party data). SFT trains on 10 families; GRPO trains on all 15 from the gated SFT cut and is gated on the 5
families held out of SFT. Deciding metric `iou` (volumetric IoU by numpy line integration over an OCCT triangulation;
a failed program scores 0), T3 with a construct-alignment disclosure (it shares the CadQuery/OCCT kernel and the
valid-solid requirement with the GRPO reward, which uses OCCT volume and bounding box). `valid` is a supporting arm.

## Stage 1: NeMo RL SFT (base model -> SFT cut 60)

NeMo RL 4f39fd8 (Tier B image `nemo-rl:4f39fd8`), trainer as uid 1000 with `--network none`, micro batch 4: 60 steps x
32 rows (1,920 rows in file order; cuts 20, 40, 60); peak GPU memory 18.35 GB, host 16.7 GiB.

| | |
|---|---|
| Deciding metric | `iou` on 300 items of the 10 SFT families |
| Baseline (base model) | 0.397 |
| Candidate (cut 60) | 0.985 (95 % of items at 1.0) |
| Delta, 95% cluster-bootstrap interval | +0.588 [0.536, 0.637] |
| p-value | <= 1.6e-04 |
| Items, samples per item | 300, 1 (greedy, eval seed 11) |
| Achieved power | 0.999 (sd of paired differences 0.439; sigma_plan 0.79 from the smoke) |
| Exact overlap with training | 0 |
| Consumption precision | `row` (observed sample log, 1,920 records) |

The impact query on `data/sft/shards/shard-00000.jsonl` returns cuts 20, 40, 60 at `row`.

## Stage 2: NeMo RL GRPO with the CadQuery environment (SFT cut 60 -> GRPO cut 60)

Derived image `ptc-a6-nemo-rl:1` (ptc, ptc-skel and ptc-hub are earlier names of gatedtrain; see "Rename to gatedtrain
0.3.0"; uid 1000, `--network none`), the B18 executor as a sidecar: 60 steps x 16 prompts x 8
generations (7,680 rewarded samples, rows 0-959 of the 2,000-row GRPO shard set in file order), cuts 20, 40, 60;
vLLM `gpu_memory_utilization` 0.25, packed microbatches of 2,560 tokens; 0.44 GPU-hours, peak GPU memory 24.2 GB.

| | |
|---|---|
| Deciding metric | `iou` on 300 items of the 5 held-out families |
| Baseline (SFT cut 60) | 0.646 |
| Candidate (GRPO cut 60) | 0.894 |
| Delta, 95% cluster-bootstrap interval | +0.248 [0.210, 0.289] |
| p-value | <= 6.2e-05 |
| Items, samples per item | 300, 1 (greedy, eval seed 11) |
| Achieved power | 0.986 (sd of paired differences 0.354; sigma_plan fixed 0.50) |
| Exact overlap with training | 0 |
| Consumption precision | `row` (the environment's call log joined with NeMo RL's per-step rewards for all 7,680 samples) |

Per family: blind hole 0.74 to 0.89, chamfered box 0.65 to 1.00, flange 0.54 to 0.96, four-hole plate 0.79 to 0.93,
tube 0.51 to 0.69. The impact query returns cuts 20, 40, 60 for GRPO shard 0 and no checkpoint for shard 3 (rows
1,500-1,999, never reached). Area total 1.49 GPU-hours (including three superseded GRPO smokes).

## Re-collect with this repository's adapter (2026-11-16)

Both stages were re-collected with this repository's adapter (installed as a plugin of ptc, the runtime override
pointing at the original area directory) on the original runs' outputs and compared with a re-collect by ptc before
the migration: every field of both gate reports is equal except the run-id-derived `lockfile_hash`; the ledgers
(1,920 and 7,680 rows) and the lineage are equal, and the impact queries above give the same cuts. The GRPO re-collects
also equal the canonical 2026-11-14 report.

The SFT run cannot be re-collected as it stands: `run_cad_nemo.py` was edited for the GRPO stage after the SFT launch
and its canonical collect, and the launch-time version (sha256 `dc550392…`) was not kept, so `ext-collect` refuses the
SFT launch record (EXT-LAUNCH, `script_sha256`), before and after the migration alike. The SFT comparison above ran
both sides with only that one file hash answered from the launch record; with that, both also equal the canonical
report except `lockfile_hash`.

## Publication cleanup (2026-11-16)

Before publication, host paths and internal process notes were removed from the repository. The result above was
re-collected unchanged before this cleanup; the cleanup changed no behaviour.

The files `results/2026-11-14/gate-sft/spec.yaml` and `results/2026-11-14/gate-grpo/spec.yaml` are redacted copies
(host paths replaced by `<data-root>`); the gate-time originals are not published.

The other gate files (`registration.yaml`, `trainer.yaml`) are unchanged, so the hashes the specs pin for them still
hold. The registrations' header comments name `decisions/MVP-MULTIMODAL.md`: that is the maintainers' decision
record, not published; the comment stays because the registrations are pinned by hash. The redacted specs say "the
maintainers' decision record (not published)" instead.

The published code differs from the code that produced the result only by this cleanup (docstrings and comments).
Two pinned runtime scripts changed, so their pins changed with them (the original scripts are not published):

| Pin | Original (produced the result) | Published |
|---|---|---|
| `runtime/executor/cad_sandbox.py` | not published | `sha256:c1ef3014aab81f71af898e766199bca5158cc2f2fa657e845fc7da11f5838577` |
| `runtime/score_cad.py` | not published | `sha256:f1115f08d3aea50837a49e3356dba411e20ea9c43485ec50424464ff7fb2e9fa` |
| scorer `code_sha256` (both evaluator rows) | `sha256:c5b19935c37cb0ae262d88dcaf01d4ccdfc0157e5c17962b4d2e31080b507181` | `sha256:b2632011e5332e6a46e25ec3679798bc39872b7481a023fb863280a6ddba72fc` |

The 2026-11-14 arms record the original scorer hash, so this repository's adapter, as published, refuses them; the
re-collect described above ran with the original pins. The test fixture `tests/fixtures/cad/real-arm/` carries the
published hashes (and `/srv/ptc` paths) so that it matches the published pins. Comments also changed in
`runtime/run_cad_nemo.py`, `runtime/make_data.py`, `runtime/gen_cad.py`, `runtime/sft_cpu_pilot.py`,
`runtime/executor/run-executor.sh` and `runtime/executor/Containerfile`: the launch records and the data metadata
(`script_sha256` in `runtime/data/data_meta.json` and `runtime/items-*/item_set_meta.json`) name the pre-cleanup
versions of the first two; the executor image is unchanged (the Containerfile's comments do not enter it).

## Executor hardening after publication (M3, 2026-11-16)

A security review (M3) found that executor jobs could forge or corrupt a sibling job's result: every child wrote
`result.json` into `<batch>/<index>/`, a predictable path that all children (same uid) could reach, and the zygote
read the result back from that file. `runtime/executor/cad_sandbox.py` now gives each job a private `mkdtemp`
directory, takes the result only from the job's own pipe, closes inherited descriptors in the child, makes the zygote
and its children non-dumpable, and checks each kept STL mesh against the SHA-256 its job reported
(`tests/test_executor_isolation.py`). Re-executing the 300 programs of a 2026-11-14 evaluation (`eval-grpo-base-1`)
with the new file gives byte-identical program records and meshes; only `code_sha256` in `exec_meta.json` differs.

| Pin | Before (published) | After M3 |
|---|---|---|
| `runtime/executor/cad_sandbox.py` | `sha256:c1ef3014aab81f71af898e766199bca5158cc2f2fa657e845fc7da11f5838577` | `sha256:b5f19863d54ed35177f1479a805179f0338ca6a4186631ba221e8946be77862e` |
| scorer `code_sha256` (both evaluator rows) | `sha256:b2632011e5332e6a46e25ec3679798bc39872b7481a023fb863280a6ddba72fc` | `sha256:bc7a7fd64d0e60479ec25d01073b99c8afc5f462a307a90989e1d6b311891ae0` |

The historical runs (results/2026-11-14/) used the earlier executor (not published, see the table above) and
are unchanged; as before, this repository's adapter refuses their arms unless the original pins are supplied. The
test fixture `tests/fixtures/cad/real-arm/` carries the new hashes (and, in `env-runtime.json`, the new
`exec_meta.json` hash).

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-14/ (the gate files and the pilot arm `pilot/cut30-sftgate-score/arm-iou/`) keep the names ptc-skel
0.1.0 wrote; they are read by ptc `78eb913` (this and the other framework commits cited here, `e20736e` among them,
are unpublished pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the published framework is
`gatedtrain` `v0.3.0`). The gate definitions are carried forward by
results/2026-11-14/gate-sft/registration-ptc-hub.yaml (amends
sha256:b6e40c5a34bf191aa54c3bfef85a1f8b88c98c1d64f4b97136d4b53438782745, post_hoc: true) with spec-ptc-hub.yaml,
and results/2026-11-14/gate-grpo/registration-ptc-hub.yaml (amends
sha256:7f4618cc25ef989e79eb2e490f6e2867221227e373067c038c2d32dd512b6c0c, post_hoc: true) with spec-ptc-hub.yaml.
`registration.yaml` and `spec.yaml` stay as the gates ran them (the specs as the redacted copies above).

Why post hoc: the deciding evaluator's identity covers the composite scorer hash, and the scorer changed after both
gates (the publication cleanup and M3 above, and now the schema id `ptc_hub.item_eval` in `score_cad.py`). ptc's
rename-only proof (design R17-3) starts from the gate-time scorer, and the gate-time files with the name map applied
are not today's files, so the amendments record the current scorer's `manifest_hash`
(sha256:018c81cbcf599adb8be6cec787acdb0589e4ac85c3b05b5222c69f0d3ed4a337, an evaluator change, not only a rename) and
`post_hoc: true`. Under ptc-hub 0.2.0 a post-hoc gate cannot PASS: both gates report INCONCLUSIVE with reason
`post_hoc`. The PASS verdicts above are the gates' verdicts under their original registrations; they reproduce with
ptc `78eb913`, the original `registration.yaml`/`spec.yaml` and the original pins, as described in "Re-collect with
this repository's adapter".

Re-pinned for the rename and the wording cleanup (only the new pin is accepted; the historical runs used the
original versions, listed in the tables above and marked below):

| File | Reason | Before (published) | Now |
|---|---|---|---|
| `runtime/score_cad.py` | schema id `ptc_hub.item_eval` | `sha256:f1115f08d3aea50837a49e3356dba411e20ea9c43485ec50424464ff7fb2e9fa` | `sha256:162c4cb3fd741dfe5baeefc07473d3322570c6101366245e5018d94ab5d06f64` |
| scorer `code_sha256` (both evaluator rows; composite) | follows `score_cad.py` | `sha256:bc7a7fd64d0e60479ec25d01073b99c8afc5f462a307a90989e1d6b311891ae0` | `sha256:989506df861f0c1b42d458c93b4c07e5639c730cf3725a29c26f6546ec675e31` |
| `runtime/cad_env.py` (`env_sha256` in `run_args.json`, checked at collect) | "untested on this host" wording | the original, not published | `sha256:a06bc0385933a9e5867151c13837e96e8799791ea4ea1b71f1081dfa9eb4b779` |
| `runtime/ptc_cad_hooks.py` (`hooks_sha256` in `run_args.json`, checked at collect) | same wording | the original, not published | `sha256:b91218d4c32ab5cde66262356b1040e0c1d14dcde46a8202be238d8e5dd9cf97` |

The original composite scorer hash is in the table above (`sha256:c5b19935…`).
Comments changed, no pin involved: `runtime/executor/pyproject.toml` (an internal decision-document name; the lock is
unchanged), `runtime/executor/run-executor.sh` and `runtime/README.md` (operator wording). The test fixture
`tests/fixtures/cad/real-arm/` follows the published code as before: its two `eval_meta.json` now carry
`ptc_hub.item_eval` and the new composite hash, and `env-runtime.json` the new `score_cad.py` and composite hashes.

### Re-judgement under ptc-hub 0.2.0 (scratch replay)

The historical records carry the old ids, which ptc-hub 0.2.0 refuses, so each gate was re-judged on a scratch replay
with ptc `e20736e`, this repository installed and the runtime override naming the original area directory, against
the amended spec (`<data-root>` substituted) and registration:

- copies of the launch record, the smoke record and both evaluation outputs (`gen/`, `exec/`, `score/`); in the copies
  only the schema ids (`ptc_skel.launch_record`, `ptc_skel.smoke_record`, `ptc_skel.item_eval` -> `ptc_hub.*`) and
  the scorer's code pins were replaced: the arms' `scorer.code_sha256` and `env-runtime.json`
  `scorer_code_sha256` (`c5b19935…` -> `989506df…`), its `code_sha256` members `score_cad.py` (the original ->
  `162c4cb3…`) and `cad_sandbox.py` (the original -> `b5f19863…`), the same `cad_sandbox.py` member in both
  `exec_meta.json` copies, and `env-runtime.json` `exec_meta_sha256` recomputed for the edited copy;
- the launch record's label checksum covers the record and the GPU log accounts time by label, so (as in the TRL
  recipe's replay) the replay used a scratch host-work directory holding a copy of the host's `gpu-step.sh` (its
  `WORK=` line naming that directory) and the run's own two `gpu-guard.log` lines under the re-hashed label (the
  host's own log was only read);
  `host_work`, `gpu_step_sha256`, `host_config_sha256` and `label` were recomputed with ptc's own functions;
- the SFT run additionally answers the launch-time `run_cad_nemo.py` hash from its launch record, as in the
  re-collect above.

Compared with the canonical 2026-11-14 reports, every statistic is equal; the differences are exactly the expected
ones: `status` PASS -> INCONCLUSIVE with `status_reasons` [`post_hoc`], `post_hoc` true, `registration_hash` (the
amendment), `registration_amends` (the original registration), `registration_amendment` null (new field),
`evaluator_hash` and `evaluator_hash_registered` (the current scorer), `analysis_code_version`, the report `schema`
id, and the run-id-derived `run_id`, `created_at` and `lockfile_hash`. The ledgers are equal except their `writer_id`
(`ptc-hub.ledger@0.2.0`) and run-id columns, the lineage equal.

| Gate | Canonical (ptc-skel 0.1.0) | Replay (ptc-hub 0.2.0) | delta_hat, 95% interval, p | Ledger rows |
|---|---|---|---|---|
| SFT | PASS | INCONCLUSIVE (`post_hoc`) | 0.588064 [0.536319, 0.636721], p <= 1.57e-04 (equal) | 1,920 (equal) |
| GRPO | PASS | INCONCLUSIVE (`post_hoc`) | 0.247790 [0.209506, 0.289136], p <= 6.24e-05 (equal) | 7,680 (equal) |

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub 0.2.0 to gatedtrain 0.3.0 (distribution and import `gatedtrain`, schema
ids `gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`); this repository is now `gatedtrain-recipe-nemo-rl-cad`
(import `gatedtrain_recipe_nemo_rl_cad`, entry point group `gatedtrain.adapters`, runtime override
`GATEDTRAIN_RECIPE_NEMO_RL_CAD_RUNTIME`). The files under results/2026-11-14/ keep their bytes: `registration.yaml` and
`spec.yaml` as the gates ran them (ptc-skel 0.1.0), `registration-ptc-hub.yaml` and `spec-ptc-hub.yaml` as the
ptc-hub 0.2.0 amendments. gatedtrain 0.3.0 reads neither old version's records nor registrations. The sections above describe
what was done at the time and keep the names of their time (ptc, ptc-hub, `ptc-a6-nemo-rl:1`, `ptc-cadexec`).

Kept names. Some identifiers that predate the rename stay, because the published runs or the committed metadata carry
them: `runtime/ptc_cad_hooks.py` (the adapter checks a run's `hooks_sha256` against the file of that name beside the
launched script, so a renamed file would refuse the published runs), the runtime's own schema ids `ptc_mm.cad_cut`,
`ptc_mm.cad_exec` (checked in the published `cut.json` and `exec_meta.json`), `ptc_mm.cad_data`,
`ptc_mm.cad_item_set` and `ptc_mm.cad_models` (in committed metadata whose hashes the registrations pin), and
`/etc/ptc-cadexec-packages.txt` inside the executor image: the `Containerfile`'s comments and documented tag changed
(`gatedtrain-cadexec:area6`), its `RUN` instructions did not, and the image stays pinned by its ID. The executor
sidecar's container name prefix is now `gatedtrain-cadexec-` (row `name_prefix` and `run-executor.sh serve`, which
also prints `docker --context gatedtrain` and reads `GATEDTRAIN_CADEXEC_*`), as the framework prints it
(`<name_prefix><hashed label>`); the name is not part of any record.

Re-pinned (only the new pin is accepted; the published runs used the versions in the earlier tables):

| Pin | Reason | Before (ptc-hub 0.2.0) | Now |
|---|---|---|---|
| `runtime/score_cad.py` | name map only: writes `gatedtrain.item_eval` | `sha256:162c4cb3fd741dfe5baeefc07473d3322570c6101366245e5018d94ab5d06f64` | `sha256:f7215cf1cf621d46da88e6ffab6faf1bceb24d9e5a246d23e65c3f995377547b` |
| scorer `code_sha256` (both evaluator rows; composite) | follows `score_cad.py` | `sha256:989506df861f0c1b42d458c93b4c07e5639c730cf3725a29c26f6546ec675e31` | `sha256:6f44fcf8187644e6f9e5db5c31c0bd64516eaa28ba2fa7eb39c5817c394d3e42` |
| `runtime/executor/uv.lock` (`EXECUTOR_LOCK_SHA256`) | project renamed `gatedtrain-recipe-nemo-rl-cad-executor` (`uv lock`, no upgrade: the same 46 packages and versions; only the project entry changed) | `sha256:cd0d0b24d23111599740e7c3d630befd347bc12ba6f12840a30162c8959eb110` (also the published runs') | `sha256:8f4264b967785942044058ecd8fcd66468bbb8f4421dbb2e6d941998c3366c55` |

`runtime/run_cad_nemo.py` changed in one comment (not pinned by the adapter; each run records its own
`script_sha256`). `cad_sandbox.py`, `cad_protocol.py`, `geom_metrics.py`, `cad_env.py` and `ptc_cad_hooks.py` are
unchanged. The test fixtures follow the published code as before: `tests/fixtures/cad/executor-uv.lock` is the new
lock, the real-arm fixture carries `gatedtrain.item_eval`, the new scorer and lock hashes, and the neutral path
`/srv/gatedtrain` (with `gen_meta_sha256` and `exec_meta_sha256` recomputed for the edited copies).

### Amendments

Both gates are carried forward by a version-only amendment of the ptc-hub 0.2.0 amendment (design 05 section 3.7,
R17-1/R17-3): results/2026-11-14/gate-sft/registration-gatedtrain.yaml (SHA-256
`e2fec59732a0acb475c24433bf47c2df8e50b66657365c69e9b1bb737a144e0e`, amends
sha256:ae7770b4b0a6ad66a5242df9c74336b1fb2adc566fd7164eead7d804bf6a5ad1) with spec-gatedtrain.yaml
(`4192d08f34f2176d4f695b6b7fa2e78b92b6342083ed2dc851984b411af2fc03`), and
results/2026-11-14/gate-grpo/registration-gatedtrain.yaml (`f76afbfebc755dfecf0afe85a026c39f87495867ce170c8ab6608d152b1dea91`,
amends sha256:c735bac51bf3ca82f17e0e0d3d345377cea7247f1a4f26c9cfd276298bef71ad) with spec-gatedtrain.yaml
(`efccb8d5ca75b6f298d0f8b8fdd5d3f14fb8695468c4d323bd86f03a1425c3b3`). Each differs from the registration it amends only
in `analysis_code_version: gatedtrain 0.3.0`, `amends`, `created_at` (2026-11-18T06:08:45Z), the deciding evaluator's
`manifest_hash` (`sha256:018c81cbcf599adb8be6cec787acdb0589e4ac85c3b05b5222c69f0d3ed4a337` ->
`sha256:b60b904db58cf3a953f57401fab65827cb795d2aadaa8071d2a44da26828d6d7`) and its comment, a `rename_proof`, and a
header comment; `post_hoc` stays `true`, as the amended registration has it (the scorer changed after the gates
before the earlier rename, see above), so both gates still report INCONCLUSIVE with reason `post_hoc`. The
`rename_proof` (`composite: true`) names the four ptc-hub-era scorer files, byte copies from git kept in
results/2026-11-14/scorer-ptc-hub/ (they hash to the composite `989506df...`); the framework's `apply_name_map` of
them gives today's files (only `score_cad.py` changes). The specs differ from the ptc-hub ones only in the
registration uri, comments, and the placeholder `<mm-root>` for the runtime's data root. `gatedtrain validate` accepts
both (with `<mm-root>` filled in); the ptc-hub specs are refused with `error[REG-FIELD]
registration.analysis_code_version` naming the amendment.

### Re-judgement under gatedtrain 0.3.0 (scratch replay)

Each gate was re-judged on a scratch replay with gatedtrain 0.3.0 (framework HEAD of 2026-11-18) and this repository
installed, the runtime override and input roots naming the original area directory, as in the earlier replay:

- copies of the launch record, the smoke record and both evaluation outputs (`gen/`, `exec/`, `score/`), made from
  the original records (ptc-skel ids); in the copies only the schema ids (`ptc_skel.launch_record`,
  `ptc_skel.smoke_record`, `ptc_skel.item_eval` -> `gatedtrain.*`) and the pins were replaced: the arms'
  `scorer.code_sha256` and `env-runtime.json` `scorer_code_sha256` (`c5b19935…` -> `6f44fcf8…`), its `code_sha256`
  members `score_cad.py` (the original -> `f7215cf1…`) and `cad_sandbox.py` (the original -> `b5f19863…`), the same
  `cad_sandbox.py` member in both `exec_meta.json` copies, `env-runtime.json` `uv_lock_sha256` (`cd0d0b24…` ->
  `8f4264b9…`), and `env-runtime.json` `exec_meta_sha256` recomputed for the edited copy;
- the launch record moved to a scratch host-work directory (a copy of the host's `gpu-step.sh` with its `WORK=` line
  naming it, and the run's two `gpu-guard.log` lines under the re-hashed label; the host's own log was only read);
  `host_work`, `gpu_step_sha256`, `host_config_sha256` and `label` were recomputed with the framework's own functions;
- the SFT run additionally answers the launch-time `run_cad_nemo.py` hash from its launch record, as before.

`ext-collect` on `spec-gatedtrain.yaml` reported `registration_amendment: version_only` for both gates. Compared with
the canonical 2026-11-14 reports every statistic is equal; the differences are exactly `status` PASS -> INCONCLUSIVE
with `status_reasons` [`post_hoc`] and `post_hoc` true (both as under ptc-hub 0.2.0), `registration_hash`,
`registration_amends` (the ptc-hub amendment), `registration_amendment` (`version_only`), `evaluator_hash` and
`evaluator_hash_registered` (`b60b904d…`), `analysis_code_version`, the report `schema` id, and the run-id-derived
`run_id`, `created_at` and `lockfile_hash`. Compared with the ptc-hub 0.2.0 replay, only the version, schema, amendment
and evaluator-identity fields and the run-id-derived ones differ. The ledgers are equal except `writer_id`
(`gatedtrain.ledger@0.3.0`) and the run-id-derived columns; the lineage is equal.

| Gate | Canonical (ptc-skel 0.1.0) | Replay (gatedtrain 0.3.0) | delta_hat, 95% interval, p | Ledger rows |
|---|---|---|---|---|
| SFT | PASS | INCONCLUSIVE (`post_hoc`) | 0.588064 [0.536319, 0.636721], p <= 1.57e-04 (equal) | 1,920 (equal) |
| GRPO | PASS | INCONCLUSIVE (`post_hoc`) | 0.247790 [0.209506, 0.289136], p <= 6.24e-05 (equal) | 7,680 (equal) |

## Re-run of the smokes (2026-11-18)

Before publication the smokes were re-run with gatedtrain 0.3.0 and this repository at its 2026-11-18 head, each request
mirroring the published smoke's launch record (same seeds, data, item sets and arguments, images by ID except as noted
below; only the runtime and output paths differ). The published records, gate files and results above are unchanged. Two
limitations showed up; neither comes from the renames (the SFT training data and the per-step sample logs are
byte-identical to the published ones, and the runtime scripts differ only in comments and in `make_data.py`, whose
output is identical).

| Run (20 items, greedy, eval seed 11) | Published | Re-run |
|---|---|---|
| SFT smoke, cut 10 on `items-sft`: `iou`, `valid` | 0.834, 1.00 | 0.834, 1.00 |
| GRPO smoke from SFT cut 60, peak GPU memory | 22.27 GB (21,239 MiB) | 23.39 GB (22,303 MiB) |
| Baseline arm (SFT cut 60 on `items-grpo`): `iou`, `valid` | 0.793, 0.95 | 0.144, 0.65 |

GPU memory. The GRPO smoke's peak is within run-to-run variation of the smoke's 22.5 GB `gpu_memory` limit: the
published smoke passed by 0.23 GB, the re-run on the published path (a full SFT run, then the GRPO smoke from its
cut 60) failed by 0.89 GB with nearly the same sequence lengths (mean 643 to 662 tokens per step), and a GRPO smoke
from the SFT smoke's cut 10 peaked at 24.70 GB. The full GRPO run (Stage 2) peaked at 24.2 GB. Lowering vLLM's
`gpu_memory_utilization` (0.25) would add headroom but changes the published configuration.

Reproducibility. Training is not bit-reproducible on the same host: the re-run's SFT cuts (smoke cut 10 and full cut
60) have other weights than the published ones although data, sample order and arguments are equal. The SFT smoke
scores the same, but the full SFT cut 60 scores far lower on the held-out GRPO families (18 of the 20 generations
differ; e.g. the published cut writes a plain shell for a four-hole plate, the re-run's cut tries to cut the holes and
fails to execute). One environment difference remains: the published SFT runs used the Tier B NeMo RL image, the
re-run its uid-1000 derivative (the image the published GRPO run used). Whether that or run-to-run divergence
explains the gap was not investigated further. Results of a re-run of either stage can therefore differ from the
published ones well beyond their intervals.
