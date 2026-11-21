"""The scorer's canonical JSON (runtime/score_cad.py `canonical`) must write the bytes of the framework's
`gatedtrain.canonical.canonical_json` (RFC 8785): ext-collect recomputes `sampler_hash` with it, and json.dumps writes
the sampler's 0.0 and 1.0 as `0.0`/`1.0` where RFC 8785 writes `0`/`1`."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "executor"))
import gen_cad  # noqa: E402
import score_cad  # noqa: E402

try:
    from gatedtrain.canonical import canonical_json
except ImportError:   # the framework is not installed
    canonical_json = None

PROMPT_BLOCK_SHA256 = "sha256:b9aea19b65e447006db92eb9e868357f519a37455d59169a778075c18ffd7e69"
# The deployment sampler as the scorer writes it (gen_cad.SAMPLER plus the backend and the prompt block hash), as
# registered in results/2026-11-14/gate-*/registration*.yaml.
SAMPLER = {
    "backend": "vllm",
    "chat_template": "checkpoint tokenizer_config chat template, add_generation_prompt=True, enable_thinking=False",
    "max_new_tokens": 512,
    "messages": "[{role: user, content: cad_protocol.user_prompt(description)}]",
    "prompt_block_sha256": PROMPT_BLOCK_SHA256,
    "samples_per_seed": 1,
    "strategy": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
}
# RFC 8785 hash of SAMPLER: the value gatedtrain expects (and the gated arms carry after ext-arm-fill).
SAMPLER_HASH = "sha256:9ed4c488cf016ca42124f3da32622bae9bfa1180c54afbc7544a4eef341a2223"
# The compact sorted JSON hash the earlier scorer wrote for the same sampler.
OLD_SAMPLER_HASH = "sha256:82bf58b12a9bc3e231b648db99f25b09aa43bac7e5bf1e539dffd776ef582858"

EDGE_VALUES = [0.0, 1.0, -0.0, 0.5, -2.5, 1e-7, 1.5e-7, 1e21, 1e22, 123456789.125, 0.1, 100, 2**53 - 1, -(2**53 - 1),
               1e-6, 5e-324, 1.7976931348623157e308, "\U0001f600",
               True, False, None, "", "café 中文 €", "q\"b\\s\n\t\x01\x7f/", " ",
               [], {}, [1, 2.0, [3.0]], {"b": 1.0, "a": {"z": 0.0, "y": [1e-7]}, "é": 2, "B": 3, "￿": 4}]


class ScorerCanonical(unittest.TestCase):
    def test_sampler_is_gen_cad_sampler(self):
        self.assertEqual(SAMPLER, {**gen_cad.SAMPLER, "backend": "vllm", "prompt_block_sha256": PROMPT_BLOCK_SHA256})

    def test_real_sampler_hash(self):
        self.assertEqual(score_cad.sha256_bytes(score_cad.canonical(SAMPLER)), SAMPLER_HASH)
        old = json.dumps(SAMPLER, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.assertEqual(score_cad.sha256_bytes(old), OLD_SAMPLER_HASH)

    def test_integral_floats_have_no_fraction(self):
        self.assertEqual(score_cad.canonical({"a": 0.0, "b": 1.0}), b'{"a":0,"b":1}')

    def test_scorer_code_hash_encoding_unchanged(self):
        """The composite scorer hash hashes a dict of `sha256:<hex>` strings, for which RFC 8785 and the compact sorted
        JSON the adapter checks (`gatedtrain.ext.sdk.compact_sha256`) give the same bytes."""
        code = {"score_cad.py": "sha256:" + "a" * 64, "geom_metrics.py": "sha256:" + "b" * 64,
                "cad_sandbox.py": "sha256:" + "c" * 64, "cad_protocol.py": "sha256:" + "d" * 64}
        self.assertEqual(score_cad.canonical(code),
                         json.dumps(code, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    @unittest.skipIf(canonical_json is None, "gatedtrain is not importable")
    def test_equals_framework_canonical_json(self):
        for v in [SAMPLER, *EDGE_VALUES]:
            with self.subTest(value=v):
                self.assertEqual(score_cad.canonical(v), canonical_json(v))

    @unittest.skipIf(canonical_json is None, "gatedtrain is not importable")
    def test_rejects_what_the_framework_rejects(self):
        for v in (float("nan"), float("inf"), 2**53, {1: 2}, "\ud800", {"\U0001f600": 1}, object()):
            with self.subTest(value=repr(v)):
                with self.assertRaises(ValueError):
                    canonical_json(v)
                with self.assertRaises(ValueError):
                    score_cad.canonical(v)

    def test_hash_matches_hashlib(self):
        b = score_cad.canonical(SAMPLER)
        self.assertEqual(score_cad.sha256_bytes(b), "sha256:" + hashlib.sha256(b).hexdigest())


if __name__ == "__main__":
    unittest.main()
