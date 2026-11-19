#!/usr/bin/env python
"""CadQuery executor of area "CAD": the program-level half of the B18 sandbox (area 6).

One subprocess per program. A single-threaded "zygote" imports CadQuery/OCP once and forks one child per program;
the child sets its own resource limits, executes the program, measures the result with OCCT and exits. The zygote
never executes model code itself. Container-level isolation (no network, non-root, all capabilities dropped,
read-only root filesystem, CPU/memory/pids limits) comes from the executor container (`executor/run-executor.sh`);
this file does not and cannot provide it. README "Executor sandbox" lists which property lives where.

Per child (all set after fork, before the program is compiled):
- RLIMIT_CPU (default 10 s; SIGXCPU then SIGKILL), RLIMIT_AS (address space at fork + 2 GiB), RLIMIT_FSIZE 64 MiB,
  RLIMIT_NOFILE 64, RLIMIT_CORE 0; wall-clock deadline (default 15 s) enforced by the zygote with SIGKILL to the
  child's process group;
- its own process group, cwd = a fresh empty work directory, stdin from /dev/null, stdout/stderr to files there;
- isolation from sibling jobs (M3): every job gets its own private work directory (`mkdtemp`: mode 0700, random
  name, no index in the path), and its result travels to the zygote over a pipe of its own, never through a file, so
  a sibling that writes into another job's directory cannot forge or replace that job's result. The child closes
  every inherited descriptor except its own pipe end (other jobs' pipes, the `--serve` client connection). The
  zygote and its children are not dumpable (PR_SET_DUMPABLE 0), so a sibling cannot reach another job's or the
  zygote's `/proc/<pid>/{cwd,fd,mem}`. The child reports the SHA-256 of its STL mesh over the pipe and `--run-batch`
  verifies the copy it keeps. All jobs still run under one uid: per-job uids (or containers) would be needed for a
  directory a sibling cannot write at all;
- a Python audit hook that refuses sockets, subprocesses, exec/fork/spawn, signals to other processes, ctypes,
  file deletion/renaming, opening files for writing outside the work directory, and opening or listing anything
  outside the read roots (the Python tree, this executor directory with its venv, the work directory, /usr, /lib,
  /dev). The hook is defence in depth for host CPU tests; it is not a security boundary (native code can bypass it).
  The container is the boundary.

No reference data reaches a process that runs generated code (decided after review 11):
- the executor container mounts only the Python tree and this directory (`<area>/executor`: this file,
  cad_protocol.py, the venv); never `items-*/`, `data/`, `pilot/`, `models/` (`run-executor.sh`);
- the zygote never loads reference data: `--serve` receives programs only, `--run-batch` reads the generation
  outputs (responses) only; children are forked from it, so their memory holds no references either;
- the zygote returns the candidate's own measurements (status, OCCT volume and bounding box, and with `--run-batch`
  its STL mesh). Rewards (cad_env.py, in the trainer container, which holds ground truth) and IoU (score_cad.py, a
  separate scoring step that runs no generated code) are computed by the caller.

Child outcome (`status`), all scores of the policy (MM-4):
  ok | no_code | syntax_error | runtime_error | no_result | not_solid | invalid_solid | timeout | resource_limit |
  crash (the child died on a signal or exited without a result, e.g. an OCCT segfault on degenerate geometry).
Executor faults (fork failure, a zygote that is not single-threaded, an unreadable work directory) raise
`ExecutorError`; callers turn them into evaluator errors, never into scores.

Modes:
  cad_sandbox.py --serve /spool/cadexec.sock [--workers 8]    # sidecar for GRPO training (Unix socket)
  cad_sandbox.py --run-batch --gen DIR --out DIR [--workers 8] # evaluation step 2a: execute every response of a
                                                               # gen_cad.py output; per program result.json (+ mesh.stl)
  cad_sandbox.py --self-test                                   # CPU checks: statuses, limits, escapes, reference
                                                               # files unreadable, no reference bytes in zygote memory
Library use: `Zygote(...).run_batch([{"id", "code", "mesh"}])` (make_data.py, for trusted reference programs only).
"""

from __future__ import annotations

import os

# Single-threaded zygote: numerical libraries must not start thread pools before fork (checked below).
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "VTK_SMP_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import errno
import hashlib
import json
import re
import resource
import shutil
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <area>/executor: the only area directory the executor mounts
sys.path.insert(0, str(HERE))
import cad_protocol  # noqa: E402

DEFAULT_LIMITS = {"cpu_s": 10, "wall_s": 15.0, "as_extra_mb": 2048, "fsize_mb": 64, "nofile": 64}
MESH_TOL_REL = 0.002          # STL linear deflection, relative to the bounding-box diagonal (min 0.001 mm)
MESH_ANGULAR_TOL = 0.1        # radians
TIMING_KEYS = ("wall_s", "cpu_s", "maxrss_mb", "exec_cpu_s")    # kept out of deterministic records
MESH_SHA_KEY = "mesh_sha256"  # reported by the child over its pipe; checked against the kept mesh, not recorded
MAX_RESULT_BYTES = 1 << 20
_ADDR = re.compile(r"0x[0-9a-fA-F]{6,}")


def _clean(text: str) -> str:
    """Failure texts without memory addresses (reproducible records)."""
    return _ADDR.sub("0x?", text)


STATUSES = ("ok", "no_code", "syntax_error", "runtime_error", "no_result", "not_solid", "invalid_solid",
            "timeout", "resource_limit", "crash")


class ExecutorError(RuntimeError):
    """Infrastructure fault of the executor (an evaluator error, MM-4)."""


def _thread_count() -> int:
    return len(os.listdir("/proc/self/task"))


def _set_dumpable(flag: int) -> None:
    """PR_SET_DUMPABLE 0: processes of the same uid (sibling jobs) may not open this process's /proc/<pid>/cwd, fd/,
    mem or ptrace it (and the process itself may not open its own /proc/self/mem). Inherited by forked children."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, flag, 0, 0, 0) != 0:           # PR_SET_DUMPABLE = 4
        raise ExecutorError(f"prctl(PR_SET_DUMPABLE, {flag}) failed: errno {ctypes.get_errno()}")


def _vm_size_bytes() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("VmSize:"):
            return int(line.split()[1]) * 1024
    raise ExecutorError("no VmSize in /proc/self/status")


# ------------------------------------------------------------------------------------------------- child side ---
_DENY_PREFIXES = ("socket.", "subprocess.", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                  "os.kill", "os.killpg", "signal.pthread_kill", "ctypes.", "os.remove", "os.rmdir", "os.rename",
                  "os.unlink", "shutil.rmtree", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate",
                  "os.system", "pty.", "webbrowser.", "urllib.Request", "http.client.", "ftplib.", "smtplib.",
                  "imaplib.", "poplib.", "nntplib.", "telnetlib.", "os.setuid")


def _read_roots(workdir: str) -> tuple[str, ...]:
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, str(HERE), workdir)}
    roots |= {"/usr", "/lib", "/lib64", "/dev"}
    return tuple(sorted(roots))


def _under(real: str, roots: tuple[str, ...]) -> bool:
    return any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def _install_audit_hook(workdir: str) -> None:
    workdir = os.path.realpath(workdir)
    roots = _read_roots(workdir)

    def hook(event, args):
        if event.startswith(_DENY_PREFIXES):
            raise PermissionError(f"sandbox: {event} is not allowed")
        if event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            target = args[0] if args else "."
            if isinstance(target, int):
                raise PermissionError("sandbox: listing by descriptor is not allowed")
            if not _under(os.path.realpath(os.fsdecode(target if target is not None else ".")), roots):
                raise PermissionError("sandbox: listing outside the read roots is not allowed")
            return
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if isinstance(path, int) or path is None:
                return
            if not _under(os.path.realpath(os.fsdecode(path)), roots):
                raise PermissionError("sandbox: reading outside the read roots is not allowed")
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                real = os.path.realpath(os.fsdecode(path))
                if not (real == workdir or real.startswith(workdir + os.sep)):
                    raise PermissionError("sandbox: writing outside the work directory is not allowed")

    sys.addaudithook(hook)


def _to_shape(result, cq):
    """Accept a Workplane, a Shape, or a list/tuple of either; return one Shape or None."""
    if isinstance(result, cq.Workplane):
        vals = [v for v in result.vals() if isinstance(v, cq.Shape)]
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(result, cq.Shape):
        return result
    if isinstance(result, (list, tuple)):
        shapes = [s for s in (_to_shape(r, cq) for r in result) if s is not None]
        if not shapes:
            return None
        return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
    return None


def _serial_occt() -> None:
    """Make OCCT run its parallel algorithms (booleans, checks, meshing) on the calling thread only. In a forked
    child OCCT's first-use thread pool segfaulted on about 1 in 400 runs of the flange references (polarArray holes);
    with a one-thread default pool: 0 in 30,000 (README "Executor sandbox")."""
    from OCP.OSD import OSD_Parallel, OSD_ThreadPool
    OSD_Parallel.SetUseOcctThreads_s(True)
    pool = OSD_ThreadPool.DefaultPool_s(1)
    pool.SetNbDefaultThreadsToLaunch(1)


def _child_main(code: str | None, workdir: str, limits: dict, mesh: bool, cq, result_fd: int) -> None:
    """Runs in the forked child. Never returns: always os._exit. The result goes to `result_fd`, the write end of
    this job's own pipe; every other inherited descriptor above stderr is closed first."""
    try:
        os.closerange(3, result_fd)
        os.closerange(result_fd + 1, max(resource.getrlimit(resource.RLIMIT_NOFILE)[0], result_fd + 1))
        os.setpgid(0, 0)
        cpu = int(limits["cpu_s"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits["fsize_mb"] << 20, limits["fsize_mb"] << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits["nofile"], limits["nofile"]))
        as_limit = _vm_size_bytes() + (int(limits["as_extra_mb"]) << 20)
        resource.setrlimit(resource.RLIMIT_AS, (as_limit, as_limit))
        os.chdir(workdir)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        so = os.open("stdout.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        se = os.open("stderr.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(so, 1)
        os.dup2(se, 2)
        for fd in (devnull, so, se):
            os.close(fd)
        sys.stdout = os.fdopen(1, "w", buffering=1)
        sys.stderr = os.fdopen(2, "w", buffering=1)
    except BaseException:
        os._exit(70)

    def finish(obj):
        data = json.dumps(obj, sort_keys=True).encode("utf-8")
        while data:
            data = data[os.write(result_fd, data):]
        os.close(result_fd)
        os._exit(0)

    try:
        _serial_occt()
    except BaseException:
        os._exit(72)
    if code is None:
        finish({"status": "no_code"})
    try:
        compiled = compile(code, "<generated>", "exec")
    except (SyntaxError, ValueError) as e:
        finish({"status": "syntax_error", "detail": f"{type(e).__name__}: {e}"[:300]})
    t0 = time.process_time()
    ns = {"__name__": "__generated__", "cq": cq, "cadquery": cq}
    _install_audit_hook(workdir)
    try:
        exec(compiled, ns)
    except MemoryError:
        finish({"status": "resource_limit", "detail": "MemoryError"})
    except BaseException as e:  # noqa: BLE001 - every program failure is a score
        finish({"status": "runtime_error", "detail": f"{type(e).__name__}: {e}"[:300]})
    exec_cpu = time.process_time() - t0
    try:
        if "result" not in ns:
            finish({"status": "no_result"})
        shape = _to_shape(ns["result"], cq)
        if shape is None or not shape.Solids():
            finish({"status": "not_solid", "detail": type(ns["result"]).__name__})
        volume = float(shape.Volume())
        bb = shape.BoundingBox()
        bbox = [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]
        valid = bool(shape.isValid())
        base = {"volume": volume, "bbox": bbox, "n_solids": len(shape.Solids()), "exec_cpu_s": round(exec_cpu, 4)}
        if not valid:
            finish({"status": "invalid_solid", **base})
        if not volume > 1e-9:
            finish({"status": "not_solid", **base})
        if mesh:
            diag = max(1e-9, ((bb.xlen ** 2) + (bb.ylen ** 2) + (bb.zlen ** 2)) ** 0.5)
            tol = max(1e-3, MESH_TOL_REL * diag)
            # Serial meshing: OCCT's parallel BRepMesh in a forked child segfaulted once in 3,000 runs (README).
            mesh_path = os.path.join(workdir, "mesh.stl")
            if not shape.exportStl(mesh_path, tolerance=tol,
                                   angularTolerance=MESH_ANGULAR_TOL, ascii=False, relative=False, parallel=False):
                finish({"status": "runtime_error", "detail": "STL export failed", **base})
            base["mesh_tolerance"] = tol
            with open(mesh_path, "rb") as f:             # the zygote's check of the kept mesh (M3), not a record field
                base[MESH_SHA_KEY] = "sha256:" + hashlib.sha256(f.read()).hexdigest()
        finish({"status": "ok", **base})
    except MemoryError:
        finish({"status": "resource_limit", "detail": "MemoryError"})
    except BaseException as e:  # noqa: BLE001
        finish({"status": "runtime_error", "detail": f"measure: {type(e).__name__}: {e}"[:300]})


# ------------------------------------------------------------------------------------------------ zygote side ---
class Zygote:
    def __init__(self, workers: int = 8, limits: dict | None = None, tmp_root: str | None = None):
        import cadquery as cq  # preloaded once; children inherit it copy-on-write
        self.cq = cq
        self.workers = int(workers)
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.tmp_root = tmp_root or tempfile.gettempdir()
        _set_dumpable(0)
        self.versions = {"cadquery": cq.__version__, "python": sys.version.split()[0]}
        try:
            import OCP
            self.versions["ocp"] = getattr(OCP, "__version__", None)
        except Exception:  # noqa: BLE001
            self.versions["ocp"] = None

    def run_batch(self, programs: list[dict], keep_dir: str | None = None) -> list[dict]:
        """programs: [{"id": str, "code": str | None, "mesh": bool}]. Returns [{"id", "status", ...}] in order.
        With `keep_dir`, each program's work directory is moved to `<keep_dir>/<index>/` (meshes for scoring)."""
        if _thread_count() != 1:
            raise ExecutorError(f"zygote has {_thread_count()} threads; fork is only safe single-threaded")
        batch_dir = tempfile.mkdtemp(prefix="cadexec-", dir=self.tmp_root)
        results: list[dict | None] = [None] * len(programs)
        workdirs: list[str | None] = [None] * len(programs)
        pending = list(range(len(programs)))
        running: dict[int, tuple[int, str, float, int]] = {}
        try:
            while pending or running:
                while pending and len(running) < self.workers:
                    i = pending.pop(0)
                    # Private per-job directory (0700, random name): no sibling can derive its path from the index.
                    wd = workdirs[i] = tempfile.mkdtemp(prefix="job-", dir=batch_dir)
                    p = programs[i]
                    r_fd, w_fd = os.pipe()
                    try:
                        pid = os.fork()
                    except OSError as e:
                        os.close(r_fd)
                        os.close(w_fd)
                        raise ExecutorError(f"fork failed: {e}") from e
                    if pid == 0:  # child
                        try:
                            _child_main(p.get("code"), wd, self.limits, bool(p.get("mesh")), self.cq, w_fd)
                        finally:
                            os._exit(71)
                    os.close(w_fd)
                    os.set_blocking(r_fd, False)
                    running[pid] = (i, wd, time.monotonic(), r_fd)
                done_any = False
                for pid in list(running):
                    try:
                        rpid, wstatus, ru = os.wait4(pid, os.WNOHANG)
                    except ChildProcessError as e:
                        raise ExecutorError(f"lost child {pid}") from e
                    i, wd, t0, r_fd = running[pid]
                    if rpid == 0:
                        if time.monotonic() - t0 > self.limits["wall_s"]:
                            self._kill(pid)
                            os.wait4(pid, 0)
                            os.close(r_fd)
                            results[i] = {"status": "timeout", "wall_s": round(time.monotonic() - t0, 3)}
                            del running[pid]
                            done_any = True
                        continue
                    del running[pid]
                    done_any = True
                    self._kill(pid)              # anything the program left in its process group
                    try:
                        data = self._read_result(r_fd)
                    finally:
                        os.close(r_fd)
                    results[i] = self._collect(wd, data, wstatus, ru, time.monotonic() - t0)
                if not done_any:
                    time.sleep(0.003)
        except BaseException:
            for pid, (_, _, _, r_fd) in running.items():
                self._kill(pid)
                try:
                    os.wait4(pid, 0)
                except ChildProcessError:
                    pass
                os.close(r_fd)
            raise
        finally:
            if keep_dir is not None:
                os.makedirs(keep_dir, exist_ok=True)
                for i in range(len(programs)):
                    src = workdirs[i]
                    if src is not None and os.path.isdir(src):
                        dst = os.path.join(keep_dir, str(i))
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.move(src, dst)
            shutil.rmtree(batch_dir, ignore_errors=True)
        out = []
        for p, r in zip(programs, results):
            out.append({"id": p["id"], **r})
        return out

    @staticmethod
    def _kill(pid: int) -> None:
        for target in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
            try:
                target()
            except (ProcessLookupError, PermissionError):
                pass

    @staticmethod
    def _read_result(r_fd: int) -> bytes:
        """What the exited child wrote to its own pipe (non-blocking: a leftover writer cannot stall the zygote)."""
        data = b""
        while len(data) <= MAX_RESULT_BYTES:
            try:
                block = os.read(r_fd, 65536)
            except BlockingIOError:
                break
            if not block:
                break
            data += block
        return data

    def _collect(self, wd: str, data: bytes, wstatus: int, ru, wall: float) -> dict:
        """`data`: the bytes the child wrote to its own pipe, the only source of its result (never a file in `wd`)."""
        info = {"wall_s": round(wall, 3), "cpu_s": round(ru.ru_utime + ru.ru_stime, 3),
                "maxrss_mb": round(ru.ru_maxrss / 1024, 1)}
        if os.WIFEXITED(wstatus) and os.WEXITSTATUS(wstatus) == 0 and data:
            if len(data) > MAX_RESULT_BYTES:
                raise ExecutorError(f"child result exceeds {MAX_RESULT_BYTES} bytes")
            try:
                res = json.loads(data.decode("utf-8"))
            except ValueError as e:
                raise ExecutorError(f"unreadable child result: {e}") from e
            if not isinstance(res, dict):
                raise ExecutorError("child result is not an object")
            if res.get("status") not in STATUSES:
                raise ExecutorError(f"unknown child status {res.get('status')!r}")
            if "detail" in res:
                res["detail"] = _clean(res["detail"])
            return {**res, **info}
        if os.WIFEXITED(wstatus) and os.WEXITSTATUS(wstatus) in (70, 71, 72):
            raise ExecutorError(f"child setup failed (exit {os.WEXITSTATUS(wstatus)})")
        if os.WIFSIGNALED(wstatus) and os.WTERMSIG(wstatus) in (signal.SIGXCPU, signal.SIGKILL) \
                and info["cpu_s"] >= self.limits["cpu_s"] - 0.5:
            return {"status": "resource_limit", "detail": "cpu", **info}
        sig = os.WTERMSIG(wstatus) if os.WIFSIGNALED(wstatus) else None
        tail = ""
        try:
            tail = open(os.path.join(wd, "stderr.txt"), errors="replace").read()[-300:]
        except OSError:
            pass
        return {"status": "crash", "detail": f"signal {sig}" if sig else f"exit {os.WEXITSTATUS(wstatus)}",
                "stderr_tail": _clean(tail), **info}


# ------------------------------------------------------------------------------------------------------ serve ---
def serve(sock_path: str, workers: int, limits: dict) -> int:
    z = Zygote(workers=workers, limits=limits)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o077)
    srv.bind(sock_path)
    os.umask(old)
    srv.listen(16)
    print(json.dumps({"event": "listening", "socket": sock_path, "workers": workers, "limits": z.limits,
                      "versions": z.versions}), flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    srv.settimeout(1.0)
    n_batches = 0
    while not stop["flag"]:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError as e:
            if e.errno == errno.EINTR:
                continue
            raise
        with conn:
            conn.settimeout(600)
            try:
                req = cad_protocol.recv_msg(conn)
                if req.get("op") == "ping":
                    cad_protocol.send_msg(conn, {"ok": True, "versions": z.versions, "limits": z.limits})
                    continue
                if req.get("op") != "run":
                    cad_protocol.send_msg(conn, {"error": f"unknown op {req.get('op')!r}"})
                    continue
                progs = [{"id": str(p["id"]), "code": p.get("code"), "mesh": False} for p in req["programs"]]
                t0 = time.monotonic()
                res = z.run_batch(progs)
                n_batches += 1
                print(json.dumps({"event": "batch", "n": len(progs), "seconds": round(time.monotonic() - t0, 3),
                                  "batches": n_batches}), flush=True)
                cad_protocol.send_msg(conn, {"results": res})
            except ExecutorError as e:
                cad_protocol.send_msg(conn, {"error": f"executor_error: {e}"})
            except (ConnectionError, ValueError, KeyError, TypeError) as e:
                try:
                    cad_protocol.send_msg(conn, {"error": f"protocol_error: {type(e).__name__}: {e}"})
                except OSError:
                    pass
    srv.close()
    os.unlink(sock_path)
    return 0


# ------------------------------------------------------------------------------------------------ run-batch ---
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def run_batch_dir(gen: Path, out: Path, workers: int, probe: Path | None = None) -> int:
    """Evaluation step 2a: execute every response in `<gen>/gen/*.json` (gen_cad.py output) once, with STL export.

    Reads responses only (no items, no references). Writes into the empty `out`:
      programs/<id>/result.json  - deterministic record {id, status, detail?, volume?, bbox?, n_solids?,
                                   mesh_tolerance?, program_sha256}; `<id>` = the gen file's stem (`<item>__s<seed>`)
      programs/<id>/mesh.stl     - the candidate's OCCT mesh, only when status is ok
      exec_timing.jsonl          - wall / CPU time and peak RSS per program (not part of any hash)
      exec_meta.json (last)      - versions, limits, gen_meta.json SHA-256, ids, and the SHA-256 of every file above
                                   except exec_timing.jsonl (reproducible: two runs give identical bytes).
    `out` is checked to be empty after all children have exited and before anything is written, so a program that
    wrote into it (possible only past the audit hook, inside the container) is an executor error.
    """
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((gen / "gen").glob("*.json"))
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    programs = [{"id": f.stem, "code": cad_protocol.extract_program(d.get("text")), "mesh": True}
                for f, d in zip(files, docs)]
    z = Zygote(workers=workers)
    keep = Path(tempfile.mkdtemp(prefix="cadexec-keep-", dir=z.tmp_root))
    try:
        t0 = time.monotonic()
        results = z.run_batch(programs, keep_dir=str(keep))
        seconds = time.monotonic() - t0
        if any(out.iterdir()):
            raise ExecutorError(f"{out} was written to while programs ran")
        timing = []
        for k, (p, r) in enumerate(zip(programs, results)):
            d = out / "programs" / p["id"]
            d.mkdir(parents=True)
            rec = {key: v for key, v in r.items() if key not in TIMING_KEYS and key != MESH_SHA_KEY}
            rec["program_sha256"] = cad_protocol.sha256_text(p["code"]) if p["code"] is not None else None
            (d / "result.json").write_text(json.dumps(rec, sort_keys=True, indent=1) + "\n", encoding="utf-8")
            mesh = keep / str(k) / "mesh.stl"
            if r["status"] == "ok":
                if not mesh.is_file():
                    raise ExecutorError(f"{p['id']}: status ok but no mesh")
                shutil.copyfile(mesh, d / "mesh.stl")
                if _sha256_file(d / "mesh.stl") != r.get(MESH_SHA_KEY):   # M3: not the mesh the job itself wrote
                    raise ExecutorError(f"{p['id']}: mesh.stl differs from the one the program's child reported")
            timing.append({"id": p["id"], **{key: r.get(key) for key in TIMING_KEYS}})
        (out / "exec_timing.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in timing))
        if probe is not None:
            (out / "probe.json").write_text(json.dumps(_memory_probe(probe), sort_keys=True, indent=1) + "\n")
    finally:
        shutil.rmtree(keep, ignore_errors=True)
    # exec_timing.jsonl is the one file outside the manifest (wall-clock values; keeps exec_meta.json reproducible)
    manifest = {str(f.relative_to(out)): _sha256_file(f) for f in sorted(out.rglob("*"))
                if f.is_file() and f.name != "exec_timing.jsonl"}
    meta = {"schema": "ptc_mm.cad_exec", "schema_version": "1.0", "ids": [p["id"] for p in programs],
            "gen_meta_sha256": _sha256_file(gen / "gen_meta.json"), "versions": z.versions, "limits": z.limits,
            "code_sha256": {"cad_sandbox.py": _sha256_file(Path(__file__).resolve()),
                            "cad_protocol.py": _sha256_file(HERE / "cad_protocol.py")},
            "files": manifest}
    (out / "exec_meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"programs": len(programs), "seconds": round(seconds, 2),
                      "statuses": {st: sum(r["status"] == st for r in results) for st in STATUSES if
                                   any(r["status"] == st for r in results)}}))
    return 0


def _memory_probe(spec_path: Path) -> dict:
    """Self-test only. Fork a trusted child of the zygote (no program, no audit hook) that reads every readable
    mapping of its own memory (/proc/self/mem) and searches for marker byte strings. The markers arrive XOR-ed with a
    one-byte key, so the plain reference bytes are never placed in this process's memory by the probe itself; a hit
    is a plain occurrence in memory inherited from the zygote."""
    spec = json.loads(spec_path.read_text())
    key = int(spec["key"])
    table = bytes(b ^ key for b in range(256))
    groups = {g: [bytes.fromhex(h) for h in spec[g]] for g in ("positive", "negative")}
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        _set_dumpable(1)    # this trusted child reads its own /proc/self/mem; no program runs while it exists
        hits = {g: [0] * len(v) for g, v in groups.items()}
        scanned = 0
        overlap = max(len(m) for v in groups.values() for m in v)
        with open("/proc/self/maps") as maps, open("/proc/self/mem", "rb", 0) as mem:
            regions = [ln.split() for ln in maps]
            for parts in regions:
                if "r" not in parts[1] or (len(parts) > 5 and parts[5] in ("[vvar]", "[vsyscall]", "[vvar_vclock]")):
                    continue
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                pos, tail = lo, b""
                while pos < hi:
                    n = min(hi - pos, 16 << 20)
                    try:
                        mem.seek(pos)
                        chunk = mem.read(n)
                    except (OSError, ValueError, OverflowError):
                        break
                    buf = (tail + chunk).translate(table)
                    for g, ms in groups.items():
                        for j, m in enumerate(ms):
                            hits[g][j] += buf.count(m)
                    tail = chunk[-overlap:]
                    scanned += len(chunk)
                    pos += n
        os.write(w, json.dumps({"hits": hits, "bytes_scanned": scanned}).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while True:
        block = os.read(r, 65536)
        if not block:
            break
        data += block
    os.close(r)
    os.waitpid(pid, 0)
    res = json.loads(data)
    res["positive_found"] = all(h > 0 for h in res["hits"]["positive"])
    res["negative_found"] = any(h > 0 for h in res["hits"]["negative"])
    return res


# -------------------------------------------------------------------------------------------------- self-test ---
# -------------------------------------------------------------------------------------------------- self-test ---
SELF_TEST = [
    ("ok_box", "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 20, 30)", "ok"),
    ("no_code", None, "no_code"),
    ("syntax", "result = cq.Workplane('XY').box(1,2", "syntax_error"),
    ("runtime", "result = cq.Workplane('XY').box(1, 2, 3).faces('>Z').fillet(5)", "runtime_error"),
    ("no_result", "x = cq.Workplane('XY').box(1, 2, 3)", "no_result"),
    ("not_solid", "result = cq.Workplane('XY').rect(10, 10)", "not_solid"),
    ("timeout_sleep", "import time\ntime.sleep(100)\nresult = None", "timeout"),
    ("cpu_loop", "while True:\n    pass", "resource_limit"),
    ("memory", "x = bytearray(8 << 30)", "resource_limit"),
    ("socket", "import socket\nsocket.socket()", "runtime_error"),
    ("subprocess", "import subprocess\nsubprocess.run(['true'])", "runtime_error"),
    ("write_outside", "open('/tmp/cadexec-escape-test.txt', 'w').write('x')", "runtime_error"),
    ("os_system", "import os\nos.system('true')", "runtime_error"),
    ("segv", "import ctypes\nctypes.string_at(0)", "runtime_error"),
    ("hard_crash", "import os\nos.abort()", "crash"),
]
# Reference files at their area paths (review 11). On the host they exist and the audit hook refuses the read; in the
# executor container they are not mounted at all (FileNotFoundError). Either way: runtime_error, nothing read.
_AREA = HERE.parent
_REF_PATHS = [_AREA / "items-grpo" / "items.jsonl", _AREA / "items-sft" / "items.jsonl",
              _AREA / "data" / "grpo" / "train.jsonl", _AREA / "data" / "sft" / "train.jsonl"]
SELF_TEST += [(f"read_{p.parent.name}_{p.stem}", f"data = open({str(p)!r}).read()\nresult = None", "runtime_error")
              for p in _REF_PATHS]
SELF_TEST += [("read_ref_mesh", f"import glob\nm = glob.glob({str(_AREA / 'items-grpo' / 'ref' / '*.stl')!r})\n"
                                f"data = open(m[0], 'rb').read()\nresult = None", "runtime_error"),
              ("read_ref_mesh_direct", f"data = open({str(_AREA / 'items-grpo' / 'ref')!r} + "
                                       f"'/cad-grpo-blind_hole-0000.stl', 'rb').read()\nresult = None",
               "runtime_error"),
              ("list_area", f"import os\nnames = os.listdir({str(_AREA)!r})\nresult = None", "runtime_error"),
              ("read_proc_mem", "data = open('/proc/self/mem', 'rb')\nresult = None", "runtime_error")]


def self_test(workers: int) -> int:
    z = Zygote(workers=workers, limits={"cpu_s": 3, "wall_s": 5.0})
    progs = [{"id": n, "code": c, "mesh": n == "ok_box"} for n, c, _ in SELF_TEST]
    t0 = time.monotonic()
    res = z.run_batch(progs)
    bad = 0
    for (name, _, want), r in zip(SELF_TEST, res):
        ok = r["status"] == want
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {r['status']} (want {want}) "
              f"{r.get('detail', '')[:80]} wall {r.get('wall_s')} s", flush=True)
    print(f"{len(res)} programs, {time.monotonic() - t0:.2f} s, {bad} failures; versions {z.versions}", flush=True)
    bad += _self_test_memory(workers)
    return 1 if bad else 0


def _self_test_memory(workers: int) -> int:
    """Run `--run-batch` in a fresh process (the production path) on a small generated batch, then probe the zygote's
    memory for reference bytes of an item (negative markers) and for bytes of a response it did read (positive
    control). The markers are prepared here, XOR-ed, and never given to the zygote in plain form."""
    import subprocess
    items = _AREA / "items-grpo" / "items.jsonl"
    if not items.exists():
        print("SKIP memory probe: reference files are not mounted here (the filesystem cases above cover it)")
        return 0
    it = json.loads(items.open(encoding="utf-8").readline())
    ref_mesh = (_AREA / "items-grpo" / it["ref_mesh"]).read_bytes()
    key = 0x5A
    x = lambda b: bytes(c ^ key for c in b).hex()  # noqa: E731
    positive_text = "result = cq.Workplane('XY').box(11, 13, 17)  # probe-positive-control"
    negative = [it["reference_code"].encode(), repr(it["ref_volume"]).encode(), it["ref_mesh_sha256"].encode(),
                ref_mesh[84:84 + 150]]
    tmp = Path(tempfile.mkdtemp(prefix="cadexec-selftest-"))
    try:
        (tmp / "gen" / "gen").mkdir(parents=True)
        responses = {"probe__s11": cad_protocol.completion(positive_text),
                     "read__s11": cad_protocol.completion(f"result = open({str(items)!r}).read()")}
        for name, text in responses.items():
            (tmp / "gen" / "gen" / f"{name}.json").write_text(json.dumps({"text": text}))
        (tmp / "gen" / "gen_meta.json").write_text("{}")
        (tmp / "probe.json").write_text(json.dumps({"key": key, "positive": [x(positive_text.encode())],
                                                    "negative": [x(m) for m in negative]}))
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--run-batch", "--gen", str(tmp / "gen"),
               "--out", str(tmp / "out"), "--workers", str(workers), "--probe-xor", str(tmp / "probe.json")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        pr = json.loads((tmp / "out" / "probe.json").read_text())
        rd = json.loads((tmp / "out" / "programs" / "read__s11" / "result.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = 0
    ok = pr["positive_found"] and not pr["negative_found"]
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} zygote_memory: positive control found {pr['positive_found']} "
          f"{pr['hits']['positive']}, reference markers found {pr['negative_found']} {pr['hits']['negative']} "
          f"({pr['bytes_scanned'] / 2**20:.0f} MiB scanned)", flush=True)
    ok = rd["status"] == "runtime_error"
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} run_batch_read_items: {rd['status']} {rd.get('detail', '')[:70]}", flush=True)
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--serve", metavar="SOCKET")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run-batch", action="store_true")
    p.add_argument("--gen", type=Path, help="--run-batch: gen_cad.py output directory (gen/, gen_meta.json)")
    p.add_argument("--out", type=Path, help="--run-batch: new, empty output directory")
    p.add_argument("--probe-xor", type=Path, help=argparse.SUPPRESS)  # self-test only
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-s", type=int, default=DEFAULT_LIMITS["cpu_s"])
    p.add_argument("--wall-s", type=float, default=DEFAULT_LIMITS["wall_s"])
    a = p.parse_args(argv)
    if a.self_test:
        return self_test(a.workers)
    if a.serve:
        return serve(a.serve, a.workers, {"cpu_s": a.cpu_s, "wall_s": a.wall_s})
    if a.run_batch:
        if not (a.gen and a.out):
            p.error("--run-batch needs --gen and --out")
        return run_batch_dir(a.gen, a.out, a.workers, a.probe_xor)
    p.error("give --serve SOCKET, --run-batch or --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
