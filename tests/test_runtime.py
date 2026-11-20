"""The runtime code that runs inside the trainer, generation and scoring containers (runtime/ptc_cad_hooks.py,
runtime/cad_env.py, runtime/gen_cad.py, runtime/score_cad.py, runtime/geom_metrics.py, runtime/run_cad_nemo.py), on
CPU and without NeMo RL, Ray, vLLM, transformers or CadQuery, with inputs shaped as the trainer passes them and built
from committed data rows (tests/fixtures/cad/data/{sft,grpo}/first3.jsonl: the first three lines of the generated
data/{sft,grpo}/train.jsonl, make_data.py --seed 20261113).

Why: the img2cad recipe, derived from this one, reached the GPU with two faults that CPU tests would have caught: the
SFT sample-log hook joined the user content with str(), which fails on a list of content parts, and the derived
prompt had lost this recipe's instruction v2 (API reminder and two worked examples).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shutil
import socket
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import _nemo_rl_ext as C  # noqa: E402  (first: sets the runtime override before gatedtrain's registry loads the rows)

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
FIX = Path(__file__).resolve().parent / "fixtures" / "cad" / "data"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "executor"))
import cad_env  # noqa: E402
import cad_protocol  # noqa: E402
import gen_cad  # noqa: E402
import ptc_cad_hooks as hooks  # noqa: E402
import run_cad_nemo  # noqa: E402

SFT_ROWS = FIX / "sft" / "first3.jsonl"
GRPO_ROWS = FIX / "grpo" / "first3.jsonl"


def rows_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def lines_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def stub_modules(test: unittest.TestCase, mods: dict[str, types.ModuleType]) -> None:
    """Put stand-in modules into sys.modules (parents linked to children) for the duration of `test`."""
    for name, mod in mods.items():
        parent, _, child = name.rpartition(".")
        if parent in mods:
            setattr(mods[parent], child, mod)
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    test.addCleanup(lambda: [sys.modules.pop(k) if v is None else sys.modules.__setitem__(k, v)
                             for k, v in saved.items()])


class TestTextOf(unittest.TestCase):
    def test_instruction_does_not_survive_str_of_a_parts_list(self):
        # The instruction has newlines and quotes, so str() of a list of content parts does not contain it.
        self.assertIn("\n", cad_protocol.INSTRUCTION)
        self.assertIn('"', cad_protocol.INSTRUCTION)
        self.assertNotIn(cad_protocol.INSTRUCTION, str([{"type": "text", "text": cad_protocol.INSTRUCTION}]))

    def test_shapes(self):
        text = rows_of(SFT_ROWS)[0]["input"]
        self.assertEqual(cad_env.text_of(text), text)
        self.assertEqual(cad_env.text_of([{"type": "text", "text": text}]), text)
        self.assertEqual(cad_env.text_of([{"type": "text", "text": "a"}, {"type": "image", "image": "x.png"},
                                          {"type": "text", "text": "b"}]), "ab")
        self.assertEqual(cad_env.text_of(None), "")
        self.assertEqual(cad_env.text_of([{"type": "image", "image": "x.png"}]), "")

    def test_responses_of_string_and_list_assistant_content(self):
        code = rows_of(SFT_ROWS)[0]["output"]
        batch = [[{"role": "user", "content": "u"}, {"role": "assistant", "content": code}],
                 [{"role": "user", "content": "u"}, {"role": "assistant", "content": [{"type": "text", "text": code}]}],
                 [{"role": "user", "content": "u"}]]
        self.assertEqual(cad_env.responses_of(batch), [code, code, ""])
        self.assertEqual(cad_protocol.extract_program(cad_env.responses_of(batch)[1]),
                         cad_protocol.extract_program(code))


class TestSftSampleLogHook(unittest.TestCase):
    """install_sft_sample_log against a stand-in `nemo_rl.algorithms.sft`, fed batches shaped as NeMo RL's
    (`idx` = the 0-based line of train.jsonl, `message_log` = one list of messages per example)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.train = self.tmp / "train.jsonl"
        shutil.copyfile(SFT_ROWS, self.train)
        self.rows = rows_of(SFT_ROWS)
        self.seen = []
        sft = types.ModuleType("nemo_rl.algorithms.sft")
        sft.prepare_sft_batch = lambda batch, **kw: self.seen.append((batch, kw)) or "prepared"
        stub_modules(self, {"nemo_rl": types.ModuleType("nemo_rl"),
                            "nemo_rl.algorithms": types.ModuleType("nemo_rl.algorithms"),
                            "nemo_rl.algorithms.sft": sft})
        self.sft = sft

    def install(self, batch_size: int = 3) -> None:
        hooks.install_sft_sample_log(self.tmp, self.train, batch_size=batch_size, attempt=2)

    def batch(self, user_content, idx=(0, 1, 2)) -> dict:
        logs = [[{"role": "user", "content": user_content(self.rows[i]["input"])},
                 {"role": "assistant", "content": self.rows[i]["output"]}] for i in idx]
        return {"idx": list(idx), "message_log": logs}

    def test_string_content(self):
        # CAD's real shape: sft_processor keeps the row's `input` as the user message's string content.
        self.install()
        self.assertEqual(self.sft.prepare_sft_batch(self.batch(lambda t: t), mbs=4), "prepared")
        self.assertEqual(self.seen[0][1], {"mbs": 4})
        lines = lines_of(self.tmp / "samples.jsonl")
        self.assertEqual([x["row_id"] for x in lines], [r["row_id"] for r in self.rows])
        self.assertEqual([(x["step"], x["epoch"], x["position"], x["idx"], x["attempt"]) for x in lines],
                         [(1, 0, 0, 0, 2), (1, 0, 1, 1, 2), (1, 0, 2, 2, 2)])
        self.assertEqual(lines[0]["row"], "sha256:" + hashlib.sha256(self.rows[0]["row_id"].encode()).hexdigest())
        steps = lines_of(self.tmp / "train_steps.jsonl")
        self.assertEqual(steps, [{"step": 1, "epoch": 0, "n_examples": 3,
                                  "example_ids": [r["row_id"] for r in self.rows]}])

    def test_epoch_advances_after_one_pass(self):
        self.install(batch_size=3)              # 3 rows, drop_last: one step per epoch
        self.sft.prepare_sft_batch(self.batch(lambda t: t))
        self.sft.prepare_sft_batch(self.batch(lambda t: t))
        self.assertEqual([x["epoch"] for x in lines_of(self.tmp / "train_steps.jsonl")], [0, 1])

    def test_list_of_text_parts_content(self):
        self.install()
        self.sft.prepare_sft_batch(self.batch(lambda t: [{"type": "text", "text": t}]))
        self.assertEqual(len(lines_of(self.tmp / "samples.jsonl")), 3)

    def test_rendered_string_content(self):
        self.install()
        rendered = lambda t: f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"   # noqa: E731
        self.sft.prepare_sft_batch(self.batch(rendered))
        self.assertEqual(len(self.seen), 1)

    def test_text_of_another_row_refused(self):
        self.install()
        b = self.batch(lambda t: t)
        b["idx"] = [1, 0, 2]                    # the batch's text is rows 0, 1, 2; the idx names 1, 0, 2
        with self.assertRaisesRegex(RuntimeError, "does not match the batch text"):
            self.sft.prepare_sft_batch(b)
        self.assertEqual(self.seen, [])
        self.assertFalse((self.tmp / "samples.jsonl").exists())

    def test_fewer_rows_than_the_batch_refused(self):
        with self.assertRaisesRegex(RuntimeError, "3 rows, fewer than the batch size 4"):
            self.install(batch_size=4)


class TestEnvironmentWithFakeExecutor(unittest.TestCase):
    """CadEnvCore.score over a real Unix socket, the executor replaced by a stand-in that speaks cad_protocol and
    returns the reference row's own OCCT features for any program."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir="/tmp"))     # AF_UNIX paths are short
        self.addCleanup(shutil.rmtree, self.tmp)
        self.sock_path = str(self.tmp / "x.sock")
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.sock_path)
        self.srv.listen(1)
        self.addCleanup(self.srv.close)
        self.requests = []
        self.rows = rows_of(GRPO_ROWS)
        self.refs = [json.loads(r["output"]) for r in self.rows]
        self.reply = None

    def serve(self, reply):
        def run():
            conn, _ = self.srv.accept()
            with conn:
                req = cad_protocol.recv_msg(conn)
                self.requests.append(req)
                cad_protocol.send_msg(conn, reply(req))
        threading.Thread(target=run, daemon=True).start()

    def ok_reply(self, req):
        ref = self.refs[0]
        return {"results": [{"id": p["id"], "status": "ok", "volume": ref["volume"], "bbox": ref["bbox"]}
                            if p["code"] else {"id": p["id"], "status": "no_code"} for p in req["programs"]]}

    def batch(self, *assistant_contents):
        return [[{"role": "user", "content": self.rows[i]["input"]}, {"role": "assistant", "content": c}]
                for i, c in enumerate(assistant_contents)]

    def test_scores_and_logs_each_sample(self):
        self.serve(self.ok_reply)
        good = "Here it is.\n```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)\n```"
        log = self.tmp / "env_calls.jsonl"
        core = cad_env.CadEnvCore({"socket": self.sock_path, "timeout_s": 10, "log_path": str(log)})
        responses = cad_env.responses_of(self.batch(good, [{"type": "text", "text": good}], "no code here"))
        rewards, results = core.score(responses, [r["output"] for r in self.rows])
        # Row 0 gets its own features back: reward 1; row 1 another part's features: between 0.2 and 1; row 2 none.
        self.assertAlmostEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], cad_protocol.reward(results[1], self.refs[1]))
        self.assertTrue(0.2 < rewards[1] < 1.0)
        self.assertEqual(rewards[2], 0.0)
        req = self.requests[0]
        self.assertEqual(req["op"], "run")
        self.assertEqual([sorted(p) for p in req["programs"]], [["code", "id"]] * 3)
        self.assertEqual([p["id"] for p in req["programs"]], ["1:0", "1:1", "1:2"])
        self.assertEqual([p["code"] for p in req["programs"]], [cad_protocol.extract_program(good)] * 2 + [None])
        wire = json.dumps(req)                  # the executor never sees reference data
        for ref in self.refs:
            self.assertNotIn(ref["row_id"], wire)
            self.assertNotIn(json.dumps(ref["volume"]), wire)
        self.assertNotIn("volume", wire)
        self.assertNotIn("bbox", wire)
        lines = lines_of(log)
        self.assertEqual([(x["call"], x["position"], x["row_id"], x["status"], x["reward"]) for x in lines],
                         [(1, i, r["row_id"], s, rw) for i, (r, s, rw)
                          in enumerate(zip(self.refs, ("ok", "ok", "no_code"), rewards))])
        self.assertEqual(lines[0]["program_sha256"], cad_protocol.sha256_text(cad_protocol.extract_program(good)))
        self.assertIsNone(lines[2]["program_sha256"])
        self.assertEqual([x["response_sha256"] for x in lines], [cad_protocol.sha256_text(t) for t in responses])

    def test_executor_fault_raises(self):
        self.serve(lambda req: {"results": [{"id": "other", "status": "ok"} for _ in req["programs"]]})
        core = cad_env.CadEnvCore({"socket": self.sock_path, "timeout_s": 10})
        with self.assertRaisesRegex(RuntimeError, "do not match the request"):
            core.score(["x"], [self.rows[0]["output"]])

    def test_executor_error_raises(self):
        self.serve(lambda req: {"error": "zygote died"})
        core = cad_env.CadEnvCore({"socket": self.sock_path, "timeout_s": 10})
        with self.assertRaisesRegex(RuntimeError, "zygote died"):
            core.score(["x"], [self.rows[0]["output"]])


class TestPromptPinned(unittest.TestCase):
    """The prompt the rows, items and evaluation requests carry: instruction v2 in the pinned cad_protocol.py."""

    # Measured once (2026-11-20) with the base's tokenizer.json (Qwen/Qwen2.5-Coder-0.5B-Instruct @ ea3f247) and
    # `tokenizers` over the 600 committed item texts: at most 628 tokens (1,865 characters), at least 2.97 characters
    # per token. Without `tokenizers` and the base, the bound assumes 2.75 characters per token (8 % more tokens).
    CHARS_PER_TOKEN = 2.75
    # Qwen2.5's chat template around one user message: its default system message, the role markers and the
    # generation prompt (29 tokens measured).
    TEMPLATE_TOKENS = 32

    def test_committed_protocol_is_the_pinned_one(self):
        from gatedtrain_recipe_nemo_rl_cad import rows
        for name in ("cad_protocol.py", "cad_sandbox.py"):
            self.assertEqual(cad_protocol.code_sha256_of(RUNTIME / "executor" / name),
                             rows.EXECUTOR_CODE_SHA256[name], name)
        self.assertEqual(cad_protocol.prompt_block_sha256(), C.PROMPT_BLOCK_SHA256)
        meta = json.loads((RUNTIME / "data" / "data_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["generator"]["prompt_block_sha256"], cad_protocol.prompt_block_sha256())

    def test_instruction_v2(self):
        ins = cad_protocol.INSTRUCTION
        self.assertIn("Useful CadQuery calls", ins)
        self.assertEqual(ins.count("Example part:"), 2)
        self.assertEqual(ins.count("```python\nimport cadquery as cq\n"), 2)
        for body in ins.split("Example part:")[1:]:     # each example parses as Python
            compile(cad_protocol.extract_program(body), "<example>", "exec")

    def test_rows_and_items_carry_the_instruction(self):
        for row in rows_of(SFT_ROWS) + rows_of(GRPO_ROWS):
            self.assertTrue(row["input"].startswith(cad_protocol.INSTRUCTION + "\n\nPart: "), row["row_id"])
            self.assertEqual(row["input"], row["text"])
        for iset in ("items-sft", "items-grpo"):
            for it in rows_of(RUNTIME / iset / "items.jsonl"):
                self.assertEqual(it["text"], cad_protocol.user_prompt(it["description"]), it["item"])

    def test_prompt_and_answer_fit_the_sequence_length(self):
        texts = [it["text"] for iset in ("items-sft", "items-grpo") for it in rows_of(RUNTIME / iset / "items.jsonl")]
        texts += [r["input"] for r in rows_of(SFT_ROWS) + rows_of(GRPO_ROWS)]
        need = max(self.counter()(t) for t in texts) + self.TEMPLATE_TOKENS + gen_cad.SAMPLER["max_new_tokens"]
        self.assertLessEqual(need, run_cad_nemo.MAX_SEQ)

    def counter(self):
        tok = RUNTIME / "models" / "qwen2.5-coder-0.5b-instruct" / "tokenizer.json"
        try:
            from tokenizers import Tokenizer
        except ImportError:
            Tokenizer = None
        if Tokenizer is None or not tok.is_file():
            return lambda text: math.ceil(len(text) / self.CHARS_PER_TOKEN)
        t = Tokenizer.from_file(str(tok))
        return lambda text: len(t.encode(text).ids)

    def test_trainer_passes_max_seq_and_the_sampler_bound(self):
        a = types.SimpleNamespace(init=None, base=Path("/b"), seed=1, socket="/s")
        for stage, prof in run_cad_nemo.PROFILES.items():
            ov = run_cad_nemo.overrides(stage[0], prof, a, Path("/o"))
            self.assertIn(f"policy.max_total_sequence_length={run_cad_nemo.MAX_SEQ}", ov)
            if stage[0] == "grpo":
                self.assertIn(f"policy.generation.max_new_tokens={gen_cad.SAMPLER['max_new_tokens']}", ov)
                self.assertIn("data.default.processor=math_hf_data_processor", ov)
            else:
                self.assertIn(f"policy.train_global_batch_size={prof['batch']}", ov)
            self.assertIn("data.shuffle=false", ov)


def safetensors_bytes(tag: str) -> bytes:
    header = json.dumps({"__metadata__": {"tag": tag}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    head = header.encode("utf-8")
    return struct.pack("<Q", len(head)) + head + b"\0\0\0\0"


class TestWriteCut(unittest.TestCase):
    """write_cut on a NeMo RL step directory stand-in (tiny byte files): one consolidated file (the result checked by
    the adapter's own cut check), several shards or none (refused)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.base = self.tmp / "models" / "qwen2.5-coder-0.5b-instruct"
        self.base.mkdir(parents=True)
        (self.base / "generation_config.json").write_text('{"do_sample": false}\n', encoding="utf-8")
        shutil.copyfile(RUNTIME / "models" / "MODELS.json", self.tmp / "models" / "MODELS.json")
        self.out = self.tmp / "run"
        self.out.mkdir()
        self.step_dir = self.out / "nemo" / "checkpoints" / "step_3"
        self.cons = self.step_dir / "policy" / "weights" / "model" / "consolidated"
        self.cons.mkdir(parents=True)
        (self.cons / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}\n', encoding="utf-8")
        tok = self.step_dir / "policy" / "tokenizer"
        tok.mkdir(parents=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
            (tok / name).write_text(f"stand-in {name}\n", encoding="utf-8")
        (self.step_dir / "training_info.json").write_text('{"step": 3}\n', encoding="utf-8")
        self.meta = {"seed": 6000, "profile": "smoke", "stage": "sft", "script_sha256": "sha256:" + "3" * 64,
                     "init_checkpoint": None, "examples_consumed": lambda s: s * 32}

    def write(self) -> Path:
        return hooks.write_cut(self.step_dir, 3, self.out, self.base, self.meta)

    def test_one_consolidated_file(self):
        (self.cons / "model-00001-of-00001.safetensors").write_bytes(safetensors_bytes("one"))
        (self.cons / "model.safetensors.index.json").write_text('{"weight_map": {}}\n', encoding="utf-8")
        cut = self.write()
        self.assertEqual(cut, self.out / "cuts" / "checkpoint-3")
        doc = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(doc["files"]), ["base_ref.json", "config.json", "generation_config.json",
                                                 "model.safetensors", "sampler.json", "tokenizer.json",
                                                 "tokenizer_config.json", "training_info.json", "vocab.json"])
        self.assertEqual(sorted(p.name for p in cut.iterdir()), sorted([*doc["files"], "cut.json"]))
        self.assertEqual(doc["deployable"], ["config.json", "generation_config.json", "model.safetensors",
                                             "sampler.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"])
        self.assertEqual((cut / "model.safetensors").stat().st_ino,
                         (self.cons / "model-00001-of-00001.safetensors").stat().st_ino)   # a hard link
        self.assertEqual(doc["files"]["model.safetensors"], hooks.sha256_file(cut / "model.safetensors"))
        self.assertEqual((doc["global_step"], doc["examples_consumed"], doc["nemo_rl_step_dir"]), (3, 96, "step_3"))
        self.assertEqual(json.loads((cut / "sampler.json").read_text(encoding="utf-8")), gen_cad.SAMPLER)
        self.assertFalse((self.out / "cuts" / ".checkpoint-3.partial").exists())
        self.assertEqual([x["cut"] for x in lines_of(self.out / "cuts.jsonl")], ["checkpoint-3"])
        self.assertIsNone(registry_adapter().cut_problem(cut, 3))

    def test_shards_refused_at_once(self):
        # The adapter's cut checks, rows.py components and gen_cad.py read one model.safetensors: a sharded model
        # must fail at the first cut, not at collect after the whole run.
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for i, n in enumerate(names):
            (self.cons / n).write_bytes(safetensors_bytes(f"shard {i}"))
        (self.cons / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": names[0], "b": names[1]}}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "several consolidated weight files .*read one model.safetensors; "
                                                  "adapt rows.py components"):
            self.write()
        self.assertFalse((self.out / "cuts" / "checkpoint-3").exists())
        self.assertFalse((self.out / "cuts.jsonl").exists())

    def test_no_weights_refused(self):
        with self.assertRaisesRegex(RuntimeError, "no consolidated safetensors file"):
            self.write()


def registry_adapter():
    from gatedtrain.ext import registry
    return registry.adapter(C.ADAPTER)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, **kw):
        self.calls.append((msgs, kw))
        return "<chat>" + msgs[0]["content"]


class TestGenCad(unittest.TestCase):
    """gen_cad.py request construction and its record, the model backend replaced (no transformers, no vLLM)."""

    def test_chat_prompts(self):
        items = rows_of(RUNTIME / "items-sft" / "items.jsonl")[:2]
        tok = FakeTokenizer()
        self.assertEqual(gen_cad.chat_prompts(tok, items), ["<chat>" + it["text"] for it in items])
        self.assertEqual([c[0] for c in tok.calls], [[{"role": "user", "content": it["text"]}] for it in items])
        self.assertEqual([c[1] for c in tok.calls], [{"tokenize": False, "add_generation_prompt": True,
                                                      "enable_thinking": False}] * 2)

    def test_read_items(self):
        items_dir = RUNTIME / "items-sft"
        everything = rows_of(items_dir / "items.jsonl")
        self.assertEqual(gen_cad.read_items(items_dir, None, False), everything)
        self.assertEqual(gen_cad.read_items(items_dir, 3, False), everything[:3])
        picked = gen_cad.read_items(items_dir, 12, True)
        fams = sorted({it["family"] for it in everything})
        self.assertEqual(len(fams), 10)
        self.assertEqual([it["family"] for it in picked], fams + fams[:2])
        self.assertEqual(len({it["item"] for it in picked}), 12)

    def test_main_writes_responses_and_meta(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        model = tmp / "model"
        model.mkdir()
        for n in ("config.json", "model.safetensors", "tokenizer.json"):
            (model / n).write_text(f"stand-in {n}\n", encoding="utf-8")
        tok = FakeTokenizer()
        tf = types.ModuleType("transformers")
        tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kw: tok)
        stub_modules(self, {"transformers": tf})
        seen = []

        def fake_hf(model_dir, prompts, seed, batch_size, threads):
            seen.append((model_dir, prompts, seed))
            return [(f"```python\nresult = {i}\n```", 7, "stop") for i in range(len(prompts))], {"transformers": "x"}
        orig = gen_cad.gen_hf
        gen_cad.gen_hf = fake_hf
        self.addCleanup(setattr, gen_cad, "gen_hf", orig)
        out = tmp / "out"
        argv = ["--backend", "hf", "--model", str(model), "--items", str(RUNTIME / "items-grpo"), "--out", str(out),
                "--limit", "2", "--eval-seeds", "11,12"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_cad.main(argv), 0)
        items = rows_of(RUNTIME / "items-grpo" / "items.jsonl")[:2]
        self.assertEqual([s[1] for s in seen], [["<chat>" + it["text"] for it in items]] * 2)
        self.assertEqual([s[2] for s in seen], [11, 12])
        doc = json.loads((out / "gen" / f"{items[1]['item']}__s12.json").read_text(encoding="utf-8"))
        self.assertEqual(doc, {"item": items[1]["item"], "seed": 12, "repeat": 1, "text": "```python\nresult = 1\n```",
                               "prompt_sha256": cad_protocol.sha256_text(items[1]["text"]), "n_new_tokens": 7,
                               "finish_reason": "stop"})
        meta = json.loads((out / "gen_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["model_files"], {n: gen_cad.sha256_file(model / n)
                                               for n in ("config.json", "model.safetensors", "tokenizer.json")})
        self.assertEqual((meta["sampler"], meta["eval_seeds"], meta["items"], meta["cut"]),
                         (gen_cad.SAMPLER, [11, 12], 2, None))
        self.assertEqual(meta["prompt_block_sha256"], C.PROMPT_BLOCK_SHA256)
        with self.assertRaisesRegex(SystemExit, "is not empty"):
            gen_cad.main(argv)

    def test_main_refuses_a_cut_whose_files_changed(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        model = tmp / "cut"
        model.mkdir()
        (model / "model.safetensors").write_text("weights\n", encoding="utf-8")
        (model / "cut.json").write_text(json.dumps({"files": {"model.safetensors": "sha256:" + "0" * 64}}),
                                        encoding="utf-8")
        tf = types.ModuleType("transformers")
        tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kw: FakeTokenizer())
        stub_modules(self, {"transformers": tf})
        with self.assertRaisesRegex(SystemExit, r"hash mismatch \['model.safetensors'\]"):
            gen_cad.main(["--backend", "hf", "--model", str(model), "--items", str(RUNTIME / "items-grpo"),
                          "--out", str(tmp / "out"), "--limit", "1"])


try:
    import numpy as np
except ImportError:   # pragma: no cover  (gatedtrain depends on numpy, so CI has it)
    np = None


def box_stl(path: Path, lo, hi) -> Path:
    """A binary STL of an axis-aligned box, outward normals (counter-clockwise seen from outside)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6), (1, 2, 6),
             (1, 6, 5), (3, 0, 4), (3, 4, 7)]
    data = bytearray(b"\0" * 80) + struct.pack("<I", len(faces))
    for f in faces:
        data += struct.pack("<3f", 0, 0, 0)
        for i in f:
            data += struct.pack("<3f", *v[i])
        data += b"\0\0"
    path.write_bytes(bytes(data))
    return path


@unittest.skipIf(np is None, "numpy is not installed")
class TestScoreHelpers(unittest.TestCase):
    """score_cad.py's pure helpers and geom_metrics on tiny meshes (no CadQuery: the meshes are written here)."""

    @classmethod
    def setUpClass(cls):
        import geom_metrics
        import score_cad
        cls.geom, cls.sc = geom_metrics, score_cad

    def test_seed_of(self):
        self.assertEqual(self.sc.seed_of("iou:x"), int.from_bytes(hashlib.sha256(b"iou:x").digest()[:8], "big"))

    def test_stats_and_mm38(self):
        st = self.sc.stats([0.0, 0.5, 0.5, 1.0, None])
        self.assertEqual((st["n"], st["mean"], st["min"], st["max"], st["distinct"]), (4, 0.5, 0.0, 1.0, 3))
        self.assertAlmostEqual(st["sd"], math.sqrt(1 / 6))
        self.assertEqual((st["mode"], st["mode_share"], st["share_zero"], st["share_one"]), (0.5, 0.5, 0.25, 0.25))
        self.assertEqual(self.sc.stats([None]), {"n": 0})
        self.assertIsNone(self.sc.stats([0.3])["sd"])
        self.assertFalse(self.sc.mm38(st, 0, 1, False)["pass"])          # mode share 0.5 is not below 0.5
        self.assertTrue(self.sc.mm38(self.sc.stats([0.1, 0.4, 0.5, 0.9]), 0, 1, False)["pass"])
        self.assertFalse(self.sc.mm38(self.sc.stats([0.01, 0.02, 0.03]), 0, 1, False)["mean_inside_by_5pct"])
        self.assertEqual(self.sc.mm38({"n": 0}, 0, 1, False), {"pass": False, "reason": "no scores"})
        self.assertTrue(self.sc.mm38(self.sc.stats([0, 1, 1]), 0, 1, True)["pass"])
        self.assertFalse(self.sc.mm38(self.sc.stats([1, 1]), 0, 1, True)["pass"])

    def test_arm_sub_metrics(self):
        sub = {"status": "runtime_error", "executes": False, "chamfer": None, "volume_ratio": float("nan"),
               "reward": 0.0, "n_new_tokens": 512, "finish_reason": "length", "detail": "NameError"}
        self.assertEqual(self.sc.arm_sub_metrics(sub), {"executes": 0, "reward": 0.0, "n_new_tokens": 512,
                                                        "status_code": 3, "truncated": 1})
        ok = self.sc.arm_sub_metrics({"status": "ok", "executes": True, "chamfer": 0.01, "finish_reason": "stop"})
        self.assertEqual(ok, {"executes": 1, "chamfer": 0.01, "status_code": 0, "truncated": 0})
        self.assertIsNone(self.sc.arm_sub_metrics(None))
        self.assertTrue(all(isinstance(v, (int, float)) for v in self.sc.arm_sub_metrics(sub).values()))

    def test_framework_of(self):
        self.assertEqual(self.sc.framework_of({"backend": "vllm", "versions": {"vllm": "0.29.0"}}),
                         {"name": "vllm", "version": "0.29.0"})
        self.assertEqual(self.sc.framework_of({"backend": "hf", "versions": {"transformers": "5.1"}}),
                         {"name": "transformers", "version": "5.1"})
        self.assertEqual(self.sc.framework_of({}), {"name": "None", "version": None})

    def test_iou_and_chamfer_on_boxes(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        a = box_stl(tmp / "a.stl", (0, 0, 0), (2, 2, 2))
        b = box_stl(tmp / "b.stl", (1, 0, 0), (3, 2, 2))     # half of a: intersection 4, union 12
        tri_a = self.geom.read_stl(str(a))
        self.assertEqual(tri_a.shape, (12, 3, 3))
        same = self.geom.volumetric_iou(tri_a, tri_a, seed=1, n=2048)
        self.assertAlmostEqual(same["iou"], 1.0)
        self.assertEqual(same["unbalanced_columns"], 0)
        self.assertAlmostEqual(same["cand_volume_mc"], 8.0, delta=0.05)
        res = self.sc.score_one((a, b, "item-1"))
        self.assertAlmostEqual(res["iou"], 1 / 3, delta=0.03)
        self.assertEqual(res["n_columns"], self.geom.IOU_COLUMNS)
        self.assertGreater(res["chamfer"], 0)
        self.assertEqual(res, self.sc.score_one((a, b, "item-1")))     # deterministic per item
        self.assertAlmostEqual(self.sc.score_one((a, a, "item-1"))["chamfer"], 0.0, delta=0.02)
        self.assertIn("error", self.sc.score_one((a, tmp / "missing.stl", "item-1")))

    def test_ascii_stl(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        tri = self.geom.read_stl(str(box_stl(tmp / "a.stl", (0, 0, 0), (1, 1, 1))))
        text = "solid t\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(
            f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n" for t in tri) + "endsolid t\n"
        (tmp / "b.stl").write_text(text, encoding="ascii")
        self.assertTrue(np.array_equal(self.geom.read_stl(str(tmp / "b.stl")), tri))


if __name__ == "__main__":
    unittest.main()
