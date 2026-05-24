#!/bin/bash -l
#SBATCH --job-name=COBAYA_SPA_smoke
#SBATCH --account=project_465002956
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=00:15:00
#SBATCH --output=%j.COBAYA_SPA_smoke.out
#SBATCH --error=%j.COBAYA_SPA_smoke.err
#SBATCH --mail-type=END,FAIL

module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

set -euo pipefail

# Safe smoke test for containerized Cobaya SPA benchmarking.
# - Uses <=8 CPU cores by default.
# - Always writes to a unique debug output directory.
# - Never uses --force.
# - Rejects non-Bench config by default to avoid touching production runs.

CLASSDIR="${CLASSDIR:-/project/project_465002956/iDM}"
SIF_PATH="${SIF_PATH:-/project/project_465002956/cobaya.sif}"
BIND_PATH="${BIND_PATH:-/project/project_465002956}"
SCRATCH_BIND="${SCRATCH_BIND:-/scratch/project_465002956}"
COBAYA_CONFIG="${COBAYA_CONFIG:-/project/project_465002956/iDM/Cobaya/MCMC/Bench_SPA.yml}"

# Safety guard: only allow Bench config unless explicitly overridden.
ALLOW_NON_BENCH_CONFIG="${ALLOW_NON_BENCH_CONFIG:-0}"
if [[ "$(basename "$COBAYA_CONFIG")" != "Bench_SPA.yml" && "$ALLOW_NON_BENCH_CONFIG" != "1" ]]; then
  echo "[ERROR] Refusing to run non-benchmark config: $COBAYA_CONFIG"
  echo "[ERROR] Set ALLOW_NON_BENCH_CONFIG=1 only if you are sure this is safe."
  exit 2
fi

TASKS="${TASKS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
N_REPEATS="${N_REPEATS:-1}"
NODE_CORE_CAPACITY="${NODE_CORE_CAPACITY:-8}"

BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-80}"
BENCHMARK_RMINUS1_STOP="${BENCHMARK_RMINUS1_STOP:-0.0}"
BENCHMARK_RMINUS1_CL_STOP="${BENCHMARK_RMINUS1_CL_STOP:-0.0}"
BENCHMARK_OUTPUT_EVERY="${BENCHMARK_OUTPUT_EVERY:-20}"
BENCHMARK_LEARN_EVERY="${BENCHMARK_LEARN_EVERY:-20}"

PYTHON_EXE="${PYTHON_EXE:-python3}"

total_cores=$((TASKS * CPUS_PER_TASK))
if (( total_cores < 1 || total_cores > NODE_CORE_CAPACITY )); then
  echo "[ERROR] Requested cores=$total_cores are outside safe smoke-test limit NODE_CORE_CAPACITY=$NODE_CORE_CAPACITY"
  exit 2
fi

for required in "$SIF_PATH" "$COBAYA_CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required"
    exit 2
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$CLASSDIR/benchmark/scalability/${STAMP}_container_cobaya_spa_smoke_job${SLURM_JOB_ID:-noid}"
OUT_DIR="$RESULT_DIR/output"
LOG_DIR="$RESULT_DIR/logs"
TMP_CFG_DIR="$RESULT_DIR/tmp_cfg"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$TMP_CFG_DIR"

TMP_CFG="$TMP_CFG_DIR/Bench_SPA_smoke.yml"
# Use Cobaya's own YAML tools inside the container to deep-merge overrides into
# a copy of the source config. This avoids the duplicate-key error that Cobaya's
# strict YAML loader raises when a second 'sampler:' block is appended.
singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" python3 - \
    "$COBAYA_CONFIG" "$TMP_CFG" "$OUT_DIR/chain" \
    "$BENCHMARK_OUTPUT_EVERY" "$BENCHMARK_LEARN_EVERY" \
    "$BENCHMARK_MAX_SAMPLES" "$BENCHMARK_RMINUS1_STOP" \
    "$BENCHMARK_RMINUS1_CL_STOP" <<'PY'
import sys
from cobaya.yaml import yaml_load_file, yaml_dump
from cobaya.tools import recursive_update

src, dst, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
output_every  = int(sys.argv[4])
learn_every   = int(sys.argv[5])
max_samples   = int(sys.argv[6])
rminus1_stop  = float(sys.argv[7])
rminus1_cl    = float(sys.argv[8])

info = yaml_load_file(src)
overrides = {
    "output": out_root,
    "resume": False,
    "sampler": {"mcmc": {
        "output_every": output_every,
        "learn_every":  learn_every,
        "max_samples":  max_samples,
        "Rminus1_stop":    rminus1_stop,
        "Rminus1_cl_stop": rminus1_cl,
    }},
}
info = recursive_update(info, overrides)
with open(dst, "w") as f:
    f.write(yaml_dump(info))
print(f"[prepare_cfg] Wrote merged config to {dst}")
PY

SYSTEM_INFO="$RESULT_DIR/system_info.txt"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "host=$(hostname)"
  echo "date=$(date -Iseconds)"
  echo "sif_path=$SIF_PATH"
  echo "bind_path=$BIND_PATH"
  echo "scratch_bind=$SCRATCH_BIND"
  echo "source_config=$COBAYA_CONFIG"
  echo "smoke_config=$TMP_CFG"
  echo "tasks=$TASKS"
  echo "cpus_per_task=$CPUS_PER_TASK"
  echo "total_cores=$total_cores"
  echo "repeats=$N_REPEATS"
} > "$SYSTEM_INFO"

run_container_python() {
  singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" "$PYTHON_EXE" "$@"
}

echo "[INFO] Smoke test result directory: $RESULT_DIR"
echo "[INFO] Running preflight checks"

run_container_python - <<'PY' > "$LOG_DIR/preflight_python.log" 2>&1
import importlib
mods = ["cobaya", "classy"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK import {m}")
    except Exception as e:
        print(f"FAIL import {m}: {type(e).__name__}: {e}")
PY

echo "[INFO] Preflight python module report written to $LOG_DIR/preflight_python.log"

set +e
for ((r=1; r<=N_REPEATS; r++)); do
  RUN_LOG="$LOG_DIR/run_r${r}.log"
  START_NS=$(date +%s%N)

  srun --nodes=1 --ntasks="$TASKS" --mpi=pmi2 --exclusive --cpus-per-task="$CPUS_PER_TASK" --cpu-bind=threads \
    singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" \
    cobaya-run "$TMP_CFG" > "$RUN_LOG" 2>&1
  CODE=$?

  END_NS=$(date +%s%N)
  ELAPSED=$(awk -v s="$START_NS" -v e="$END_NS" 'BEGIN{printf "%.6f", (e-s)/1e9}')

  echo "repeat=$r wall_seconds=$ELAPSED exit_code=$CODE" | tee -a "$RESULT_DIR/run_summary.txt"
  echo "[INFO] Run $r finished with exit_code=$CODE (log: $RUN_LOG)"

  if [[ $CODE -ne 0 ]]; then
    echo "[WARN] Cobaya failed. Showing last 80 lines of $RUN_LOG"
    tail -n 80 "$RUN_LOG"
    break
  fi
done
set -e

echo "[INFO] Smoke test complete"
echo "[INFO] Result dir: $RESULT_DIR"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Modified config copy: $TMP_CFG"
