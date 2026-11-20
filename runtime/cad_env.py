"""NeMo RL environment of area "CAD" (`rl.nemo_rl_env@0.1`): rewards a response by executing its CadQuery program
in the B18 executor container.

Runs as a Ray actor in the NeMo RL image's system Python (registered by run_grpo_cad.py through
`nemo_rl.environments.utils.register_env` and `ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM`,
utils.py:132-136, ray_actor_environment_registry.py:64-80). It never imports CadQuery and never executes model
output: it extracts the program (cad_protocol.extract_program) and sends the batch over the executor's Unix socket
(`env.cad.socket`, a file in a host directory mounted into both containers). The reward is cad_protocol.reward
(OCCT validity, volume ratio and bounding-box IoU against the row's reference features, which NeMo RL passes as
`metadata[i]["ground_truth"]`, the row's `output` field: math_hf_data_processor, processors.py:563).

The executor never receives reference data (review 11): the request carries programs only, the executor returns
the candidate's own status and OCCT measurements, and the reward against the reference is computed here, in the
trainer container. A transport or executor fault raises: the GRPO step fails rather than training on a reward of 0 that was not the
policy's fault (MM-4). Each call appends one line per sample to `env.cad.log_path` (`env_calls.jsonl`, fsynced):
`{call, position, row_id, status, reward, program_sha256, response_sha256, utc}` - the observed consumption record
of a GRPO sample (README "Consumption").

The Ray actor needs the NeMo RL image (the GPU smoke run exercises it); `--self-test` of cad_env_core below runs on
any Python.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))   # cad_protocol, cad_sandbox (the executor mount)
import cad_protocol  # noqa: E402


def text_of(content) -> str:
    """The text of a message's content: a string as is, or the text parts of a list of content parts
    (`[{type: text, text: ...}]`). str() of such a list would give its repr, with escaped newlines and quotes."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def responses_of(message_log_batch) -> list[str]:
    """Each conversation's assistant text (what the environment scores)."""
    return ["".join(text_of(m["content"]) for m in conv if m["role"] == "assistant") for conv in message_log_batch]


class CadEnvCore:
    """Framework-free part of the environment (testable without NeMo RL)."""

    def __init__(self, cfg: dict):
        self.socket_path = cfg["socket"]
        self.timeout = float(cfg.get("timeout_s", 900))
        self.log_path = cfg.get("log_path")
        self.calls = 0

    def score(self, responses: list[str], ground_truths: list[str]) -> tuple[list[float], list[dict]]:
        self.calls += 1
        refs = [json.loads(g) for g in ground_truths]
        progs = [{"id": f"{self.calls}:{i}", "code": cad_protocol.extract_program(t)} for i, t in enumerate(responses)]
        t0 = time.time()
        results = cad_protocol.call_executor(self.socket_path, progs, self.timeout)
        rewards = [cad_protocol.reward(r, ref) for r, ref in zip(results, refs)]
        if self.log_path:
            utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            with open(self.log_path, "a", encoding="utf-8") as f:
                for i, (resp, p, r, ref, rw) in enumerate(zip(responses, progs, results, refs, rewards)):
                    f.write(json.dumps({
                        "call": self.calls, "position": i, "row_id": ref["row_id"], "status": r["status"],
                        "reward": rw, "program_sha256": cad_protocol.sha256_text(p["code"]) if p["code"] else None,
                        "response_sha256": cad_protocol.sha256_text(resp), "exec_batch_s": round(time.time() - t0, 3),
                        "utc": utc}, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rewards, results


try:  # the Ray actor exists only inside the NeMo RL image
    import ray
    import torch
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

    @ray.remote(max_restarts=0, max_task_retries=0)  # pragma: no cover
    class CadQueryEnvironment(EnvironmentInterface):
        def __init__(self, cfg: dict):
            self.core = CadEnvCore(cfg)

        def step(self, message_log_batch, metadata):
            responses = responses_of(message_log_batch)
            rewards, results = self.core.score(responses, [m["ground_truth"] for m in metadata])
            observations = [{"role": "environment", "content": f"Environment: {r['status']}"} for r in results]
            rw = torch.tensor(rewards, dtype=torch.float32).cpu()
            return EnvironmentReturn(observations=observations, metadata=metadata,
                                     next_stop_strings=[None] * len(responses), rewards=rw,
                                     terminateds=torch.ones_like(rw).cpu(), answers=None)

        def global_post_process_and_metrics(self, batch):
            rewards = batch["rewards"] * batch["is_end"]
            metrics = {"cad_reward": rewards.mean().item(),
                       "cad_executes": (rewards > 0).float().mean().item(),
                       "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
                       "generation_lengths": batch["generation_lengths"].float().mean().item()}
            return batch, metrics

        def shutdown(self):
            return None
except ImportError:  # host: no ray / nemo_rl
    CadQueryEnvironment = None
