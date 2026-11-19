"""CAD executor (runtime/executor/cad_sandbox.py): a job cannot forge or replace a sibling job's result (M3).

The zygote forks one child per program, all under one uid. The Python audit hook keeps a program's Python-level writes
inside its own work directory, but it is not a security boundary: native code (OCCT through the OCP bindings) writes
files without raising audit events. These tests model that bypass by disabling the hook and check that the zygote
takes each job's result only from a channel that job alone controls.

CadQuery is not needed: a stub `cadquery` / `OCP` (written to a temporary directory) stands in for the CAD kernel, so
the real fork, pipe and work-directory handling of the zygote runs on any Linux Python. Each case runs in a fresh
subprocess (the zygote must be single-threaded and makes its process non-dumpable).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXECUTOR = Path(__file__).resolve().parent.parent / "runtime" / "executor"

STUB_CADQUERY = '''
__version__ = "stub"


class _BB:
    def __init__(self, side):
        self.xmin = self.ymin = self.zmin = 0.0
        self.xmax = self.ymax = self.zmax = self.xlen = self.ylen = self.zlen = side


class Shape:
    def __init__(self, volume=1.0):
        self._volume = float(volume)

    def Solids(self):
        return [self]

    def Volume(self):
        return self._volume

    def BoundingBox(self):
        return _BB(self._volume ** (1.0 / 3.0))

    def isValid(self):
        return True

    def exportStl(self, path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"genuine mesh %r" % self._volume)
        return True


class Compound(Shape):
    @staticmethod
    def makeCompound(shapes):
        return Compound(sum(s.Volume() for s in shapes))


class Workplane:
    def vals(self):
        return []
'''

STUB_OCP_OSD = '''
class OSD_Parallel:
    @staticmethod
    def SetUseOcctThreads_s(flag):
        pass


class _Pool:
    def SetNbDefaultThreadsToLaunch(self, n):
        pass


class OSD_ThreadPool:
    @staticmethod
    def DefaultPool_s(n):
        return _Pool()
'''

# The attacker: for 3 s, rewrites `result.json` in every directory next to its own work directory, found by listing
# or behind /proc/<pid>/cwd of nearby processes (atomic replace, so the last write wins over the victim's own file).
# Only targets inside the batch directory are touched: other processes of the test's uid must never be written to.
FORGE_RESULT = textwrap.dedent('''
    import glob, json, os, time
    forged = json.dumps({"status": "ok", "volume": 999.0, "bbox": [0, 0, 0, 1, 1, 1], "n_solids": 1})
    me = os.path.realpath(os.getcwd())
    batch = os.path.dirname(me) + os.sep
    deadline = time.time() + 3.0
    while time.time() < deadline:
        targets = [d for d in glob.glob(os.path.join(os.path.dirname(me), "*")) if os.path.realpath(d) != me]
        targets += [f"/proc/{pid}/cwd" for pid in range(max(1, os.getpid() - 64), os.getpid() + 64)
                    if pid != os.getpid()]
        for d in targets:
            try:
                real = os.path.realpath(d)
                if real == me or not real.startswith(batch) or not os.path.isdir(d):
                    continue
                tmp = os.path.join(me, "forged.json")
                with open(tmp, "w") as f:
                    f.write(forged)
                os.replace(tmp, os.path.join(d, "result.json"))
            except OSError:
                pass
        time.sleep(0.01)
    result = cq.Shape(1.0)
''')

VICTIM = "import time\ntime.sleep(1.0)\nresult = cq.Shape(6.0)"

# The attacker waits until its sibling has finished, then overwrites every mesh.stl it can reach.
FORGE_MESH = textwrap.dedent('''
    import glob, os, time
    time.sleep(1.5)
    me = os.path.realpath(os.getcwd())
    for m in glob.glob(os.path.join(os.path.dirname(me), "*", "mesh.stl")):
        if os.path.dirname(os.path.realpath(m)) != me:
            with open(m, "wb") as f:
                f.write(b"forged mesh")
    result = cq.Shape(1.0)
''')

# Reports (as a runtime_error detail) the descriptors the child holds and whether it can list the zygote's fds.
PROBE_FDS = textwrap.dedent('''
    import os
    fds = sorted(int(x) for x in os.listdir("/proc/self/fd"))
    try:
        os.listdir(f"/proc/{os.getppid()}/fd")
        parent = "readable"
    except PermissionError:
        parent = "denied"
    raise RuntimeError(f"fds={fds} parent_fd={parent}")
''')

DRIVER = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import cad_sandbox
    cad_sandbox._install_audit_hook = lambda workdir: None   # model native code that bypasses the audit hook
    case = json.loads(sys.argv[2])
    if case["mode"] == "batch":
        z = cad_sandbox.Zygote(workers=len(case["programs"]), limits={"wall_s": 10.0}, tmp_root=case["tmp"])
        print(json.dumps({"results": z.run_batch(case["programs"])}))
    else:
        try:
            cad_sandbox.run_batch_dir(Path(case["gen"]), Path(case["out"]), workers=2)
            print(json.dumps({"ok": True}))
        except cad_sandbox.ExecutorError as e:
            print(json.dumps({"executor_error": str(e)}))
''')


@unittest.skipUnless(sys.platform.startswith("linux"), "the executor is Linux-only (fork, /proc, prctl)")
class ExecutorIsolationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cadexec-m3-")
        self.tmp = Path(self._tmp.name)
        stubs = self.tmp / "stubs"
        (stubs / "OCP").mkdir(parents=True)
        (stubs / "cadquery.py").write_text(STUB_CADQUERY)
        (stubs / "OCP" / "__init__.py").write_text('__version__ = "stub"\n')
        (stubs / "OCP" / "OSD.py").write_text(STUB_OCP_OSD)
        (self.tmp / "work").mkdir()
        self.env = {**os.environ, "PYTHONPATH": str(stubs), "TMPDIR": str(self.tmp / "work")}

    def tearDown(self):
        self._tmp.cleanup()

    def _drive(self, case: dict) -> dict:
        proc = subprocess.run([sys.executable, "-c", DRIVER, str(EXECUTOR), json.dumps(case)], env=self.env,
                              cwd=self.tmp, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def _batch(self, programs: list[dict]) -> list[dict]:
        return self._drive({"mode": "batch", "tmp": str(self.tmp / "work"), "programs": programs})["results"]

    def test_job_cannot_forge_sibling_result(self):
        res = self._batch([{"id": "attacker", "code": FORGE_RESULT, "mesh": False},
                           {"id": "victim", "code": VICTIM, "mesh": False}])
        victim = res[1]
        self.assertEqual(victim["id"], "victim")
        self.assertEqual(victim["status"], "ok")
        self.assertEqual(victim["volume"], 6.0, "the victim's result was replaced by its sibling's forgery")

    def test_job_cannot_replace_sibling_mesh(self):
        gen = self.tmp / "gen"
        (gen / "gen").mkdir(parents=True)
        (gen / "gen_meta.json").write_text("{}")
        for name, code in (("a_attacker__s0", FORGE_MESH), ("b_victim__s0", "result = cq.Shape(6.0)")):
            (gen / "gen" / f"{name}.json").write_text(json.dumps({"text": f"```python\n{code}\n```"}))
        out = self.tmp / "out"
        res = self._drive({"mode": "dir", "gen": str(gen), "out": str(out)})
        kept = out / "programs" / "b_victim__s0" / "mesh.stl"
        if "executor_error" in res:   # the forgery was detected and nothing was recorded for the batch
            self.assertIn("mesh.stl", res["executor_error"])
            self.assertFalse((out / "exec_meta.json").exists())
        else:
            self.assertEqual(kept.read_bytes(), b"genuine mesh 6.0", "the victim's mesh was replaced by its sibling")

    def test_child_holds_only_its_own_pipe(self):
        res = self._batch([{"id": "a", "code": PROBE_FDS, "mesh": False},
                           {"id": "b", "code": PROBE_FDS, "mesh": False}])
        for r in res:
            self.assertEqual(r["status"], "runtime_error")
            fds = json.loads(r["detail"].split("fds=")[1].split(" parent_fd=")[0])
            # 0-2, the result pipe, and the descriptor listdir itself opened
            self.assertLessEqual(len([fd for fd in fds if fd > 2]), 2, r["detail"])
            self.assertIn("parent_fd=denied", r["detail"])


if __name__ == "__main__":
    unittest.main()
