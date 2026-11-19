"""Shared, dependency-free pieces of area "CAD": prompt text, program extraction, the executor wire protocol and
the training reward.

Imported by the executor side (`cad_sandbox.py`, `score_cad.py`, `make_data.py`, executor venv) and by the
trainer side (`cad_env.py` inside the NeMo RL image's system Python, `gen_cad.py`). Standard library only, so the
NeMo RL environment actor never imports CadQuery: it sends programs to the executor container over a Unix socket.

The "SHARED PROMPT" block below is the one place the instruction text lives; `make_data.py` writes it into every
row and item, and its SHA-256 (`PROMPT_SHA256`) is recorded in the data metadata, the evaluator sampler and the cuts.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import struct

# ---- SHARED PROMPT (begin) -------------------------------------------------------------------------------------
# Instruction v2 (README "Task and data"): API reminder plus two worked examples outside the 15 families.
INSTRUCTION = 'Write CadQuery (Python) code that builds the part described below. Units are millimetres. Import cadquery as cq and assign the final solid to a variable named `result`. Reply with one short ```python code block and nothing else.\nUseful CadQuery calls (use only what the part needs): cq.Workplane("XY"); .box(x, y, z) (centred at the origin; centered=(True, True, False) puts the bottom on z = 0); .cylinder(height, radius) (Z axis, centred); .circle(r) / .rect(x, y) / .polygon(n, circumscribed_diameter) then .extrude(h) (upwards from the workplane) or .extrude(h / 2, both=True); .faces(">Z").workplane() (new workplane on the top face, on the axis); .hole(diameter) or .hole(diameter, depth); .cboreHole(diameter, cbore_diameter, cbore_depth); .slot2D(length, width); .cutThruAll(); .rect(x, y, forConstruction=True).vertices(); .polarArray(radius, 0, 360, count); .edges("|Z").fillet(r); .edges(">Z").chamfer(c); .faces(">Z").shell(-thickness); .union(other); .translate((x, y, z)).\n\nExample part: A square plate 30 x 30 mm and 4 mm thick in Z, centred at the origin, with a 10 mm square hole through its centre.\n```python\nimport cadquery as cq\n\nresult = cq.Workplane("XY").box(30, 30, 4).faces(">Z").workplane().rect(10, 10).cutThruAll()\n```\n\nExample part: A cylinder of diameter 24 mm on the Z axis from z = 0 to z = 10, with a 12 x 12 mm square block 8 mm tall centred on top of it.\n```python\nimport cadquery as cq\n\nbase = cq.Workplane("XY").circle(12).extrude(10)\ntop = cq.Workplane("XY").box(12, 12, 8, centered=(True, True, False)).translate((0, 0, 10))\nresult = base.union(top)\n```'


def user_prompt(description: str) -> str:
    """The user message of every training row, item and evaluation request."""
    return f"{INSTRUCTION}\n\nPart: {description}"


def completion(code: str) -> str:
    """The SFT target: the reference program in one fenced block."""
    return f"```python\n{code.strip()}\n```"
# ---- SHARED PROMPT (end) ---------------------------------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*)\Z", re.DOTALL | re.IGNORECASE)


def extract_program(text: str) -> str | None:
    """Program of a model response: the first complete fenced block; else an unterminated final fence (a response
    cut by the token bound); else None (status `no_code`). Text outside the block is ignored."""
    if text is None:
        return None
    m = _FENCE.search(text)
    if m:
        code = m.group(1)
    else:
        m = _OPEN_FENCE.search(text)
        if not m:
            return None
        code = m.group(1)
    code = code.strip("\n")
    return code if code.strip() else None


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_sha256_of(path) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def prompt_block_sha256() -> str:
    """SHA-256 of the SHARED PROMPT block of this file (the bytes between the markers, inclusive)."""
    src = open(__file__, encoding="utf-8").read()
    a = src.index("# ---- SHARED PROMPT (begin)")
    b = src.index("# ---- SHARED PROMPT (end)")
    return sha256_text(src[a:b])


# ---- wire protocol: 8-byte big-endian length + UTF-8 JSON, one request and one response per connection --------
MAX_MESSAGE = 256 << 20


def send_msg(sock: socket.socket, obj) -> None:
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(data)) + data)


def recv_msg(sock: socket.socket):
    head = _recv_exact(sock, 8)
    (n,) = struct.unpack(">Q", head)
    if n > MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds {MAX_MESSAGE}")
    return json.loads(_recv_exact(sock, n).decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("executor closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def call_executor(socket_path: str, programs: list[dict], timeout: float) -> list[dict]:
    """Send one batch to the executor sidecar; returns one result per program, in order. Raises on any transport
    or protocol fault (an evaluator/infrastructure error, never a policy score)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(socket_path)
        send_msg(s, {"op": "run", "programs": programs})
        resp = recv_msg(s)
    if not isinstance(resp, dict) or resp.get("error"):
        raise RuntimeError(f"executor error: {resp.get('error') if isinstance(resp, dict) else resp!r}")
    results = resp["results"]
    if len(results) != len(programs) or [r["id"] for r in results] != [p["id"] for p in programs]:
        raise RuntimeError("executor returned results that do not match the request")
    return results


# ---- training reward (OCCT quantities only; the deciding metric is computed elsewhere, see score_cad.py) --------
REWARD_VERSION = "cad_reward_v1: 0 unless ok; else 0.2 + 0.4*volume_ratio + 0.4*bbox_iou"


def bbox_iou(a: list[float], b: list[float]) -> float:
    """Axis-aligned box IoU; boxes are [xmin, ymin, zmin, xmax, ymax, zmax]."""
    inter = 1.0
    for i in range(3):
        lo, hi = max(a[i], b[i]), min(a[i + 3], b[i + 3])
        if hi <= lo:
            return 0.0
        inter *= hi - lo
    va = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
    vb = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def reward(result: dict, ref: dict) -> float:
    """Training reward from the executor's OCCT features (`volume`, `bbox`) and the row's reference features."""
    if result.get("status") != "ok":
        return 0.0
    v, vr = float(result["volume"]), float(ref["volume"])
    vol_ratio = min(v, vr) / max(v, vr) if v > 0 and vr > 0 else 0.0
    return 0.2 + 0.4 * vol_ratio + 0.4 * bbox_iou(result["bbox"], ref["bbox"])
