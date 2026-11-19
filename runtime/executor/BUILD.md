# Executor image build and in-container checks (2026-11-13)

- Built with `docker --context gatedtrain build --network host -t gatedtrain-cadexec:area6 -f executor/Containerfile executor/`.
  Image id `sha256:9b0cfb6e125b8986ad562ef18cff5268460b833d046374b292ca164c4558b89a`, 315 MB.
- First build failed: the base image has no CA bundle and snapshot.ubuntu.com is HTTPS only ("Certificate
  verification failed"). Fix in the Containerfile: install `ca-certificates` (20260816ubuntu0.24.04.2, with openssl
  3.0.13-0ubuntu3.16) from the image's own HTTP archive first, then the snapshot. The installed package list is in
  `/etc/ptc-cadexec-packages.txt` in the image.
- Self-test in the container with the `run-executor.sh` flags (`--network none`, `--user 1000:1000`,
  `--cap-drop ALL`, `no-new-privileges`, `--read-only`, tmpfs /tmp, CPU / memory / pids / nofile / core limits):
  `cad_sandbox.py --self-test` 15 programs, 0 failures, 5.02 s; cadquery 2.8.0, OCP 7.9.3.1, Python 3.12.16.
- Isolation checks in the same container: uid/gid 1000; network interfaces `['lo']` only, an outbound connection
  fails; writes to `/etc` and to the read-only area mount refused; host directories outside the mounts (the user's
  home and other project directories) not visible; CapPrm and CapEff 0, NoNewPrivs 1; `import cadquery` works (the first import took about 2 min on a
  cold page cache).

## Re-check after the review-11 redesign (2026-11-14)

- Same image `gatedtrain-cadexec:area6` (unchanged). Mount set of the executing containers (serve, exec): the Python tree and
  `area6-cad/executor` only, both read-only.
- `executor/cad_sandbox.py --self-test` in the container with the B18 flags: 23 programs, 0 failures, 5.02 s; the 8 new
  reference-read attempts (items-grpo, items-sft, both training files, reference mesh by glob and by path, listing the
  area root, /proc/self/mem) end in runtime_error; the memory probe SKIPs (no reference files are mounted).
- In the same mount set, `os.listdir('<area6-cad>')` shows only `['executor']`; `items-grpo` does not exist.
