#!/bin/bash
# Prints (never runs) the docker commands of area "CAD"'s executor containers. README "Executor sandbox" and
# "Launch request facts". Run the printed commands yourself.
#
#   run-executor.sh serve <run_dir> <label>      GRPO sidecar: Unix socket <run_dir>/spool/cadexec.sock; container name
#                                                gatedtrain-cadexec-<label> (unique per launch; the printed stop names it)
#   run-executor.sh exec  <gen_dir> <exec_dir>   evaluation step 2a: execute every response (cad_sandbox --run-batch)
#   run-executor.sh score <gen_dir> <exec_dir> <out_dir> <items-sft|items-grpo>
#                                                evaluation step 2b: score against the references (score_cad.py);
#                                                runs no generated code
#
# Containers that run generated code (serve, exec) mount ONLY the Python tree and <area>/executor (cad_sandbox.py,
# cad_protocol.py, the venv), plus the spool or the gen input and the exec output: never items-*/, data/, pilot/,
# models/ (review 11: generated code must not be able to read reference data). The score container mounts the area
# read-only (it needs items-*/ and score_cad.py) and runs no generated code.
# All three: --network none, uid 1000, no capabilities, no-new-privileges, read-only root, tmpfs /tmp,
# CPU / memory / swap / pids limits.
# The area (runtime) directory is this script's parent directory, the Python tree the directory holding the interpreter
# the executor venv was made from (.venv/pyvenv.cfg `home` = <tree>/bin); both are mounted at their own host paths.
# GATEDTRAIN_CADEXEC_AREA and GATEDTRAIN_CADEXEC_PYTHON override them (the recipe's tests print the commands for a mirrored layout).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AREA=${GATEDTRAIN_CADEXEC_AREA:-$(dirname "$HERE")}
if [ -z "${GATEDTRAIN_CADEXEC_PYTHON:-}" ]; then
  HOME_BIN=$(sed -n 's/^home *= *//p' "$AREA/executor/.venv/pyvenv.cfg" 2>/dev/null || true)
  [ -n "$HOME_BIN" ] || { echo "no executor venv: $AREA/executor/.venv/pyvenv.cfg (uv sync in executor/)" >&2; exit 2; }
  PY=$(dirname "$(dirname "$HOME_BIN")")
else
  PY=$GATEDTRAIN_CADEXEC_PYTHON
fi
EXE=$AREA/executor
IMAGE=${GATEDTRAIN_CADEXEC_IMAGE:-gatedtrain-cadexec:area6}
B18=(--rm --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g --cpus 8 --memory 8g --memory-swap 8g --pids-limit 256
  --ulimit nofile=1024:1024 --ulimit core=0
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -v "$PY:$PY:ro")
case "${1:-}" in
  serve)
    RUN=${2:?run dir}; LABEL=${3:?label (e.g. mm-a6-cad-grpo-full-1)}
    [[ "$LABEL" =~ ^[a-z0-9][a-z0-9_.-]*$ ]] || { echo "label must match [a-z0-9][a-z0-9_.-]*" >&2; exit 2; }
    NAME=gatedtrain-cadexec-$LABEL
    printf '%q ' docker --context gatedtrain run -d --name "$NAME" "${B18[@]}" -v "$EXE:$EXE:ro" -v "$RUN/spool:/spool:rw" \
      "$IMAGE" "$EXE/.venv/bin/python" -I "$EXE/cad_sandbox.py" --serve /spool/cadexec.sock --workers 8; echo
    printf '%q ' docker --context gatedtrain stop "$NAME"; echo "   # after the GRPO container has exited" ;;
  exec)
    GEN=${2:?gen dir}; OUT=${3:?exec out dir}
    printf '%q ' docker --context gatedtrain run "${B18[@]}" -v "$EXE:$EXE:ro" -v "$GEN:/inputs/gen:ro" -v "$OUT:/out:rw" \
      "$IMAGE" "$EXE/.venv/bin/python" -I "$EXE/cad_sandbox.py" --run-batch --gen /inputs/gen --out /out --workers 8; echo ;;
  score)
    GEN=${2:?gen dir}; EXD=${3:?exec dir}; OUT=${4:?out dir}; ITEMS=${5:?item set: items-sft or items-grpo}
    case "$ITEMS" in items-sft|items-grpo) ;; *) echo "item set must be items-sft or items-grpo" >&2; exit 2 ;; esac
    printf '%q ' docker --context gatedtrain run "${B18[@]}" -v "$AREA:$AREA:ro" -v "$GEN:/inputs/gen:ro" \
      -v "$EXD:/inputs/exec:ro" -v "$OUT:/out:rw" "$IMAGE" "$EXE/.venv/bin/python" -I "$AREA/score_cad.py" \
      --gen /inputs/gen --exec /inputs/exec --items "$AREA/$ITEMS" --out /out --eval-seeds 11 --workers 8; echo ;;
  *) echo "usage: $0 serve <run_dir> <label> | exec <gen_dir> <exec_dir> | score <gen_dir> <exec_dir> <out_dir> <items-sft|items-grpo>" >&2; exit 2 ;;
esac
