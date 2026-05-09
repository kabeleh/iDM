#!/bin/bash -l
#SBATCH --job-name=CLASS_hyper_scaling
#SBATCH --account=project_465002956
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=%j.CLASS_hyper_scaling.out
#SBATCH --error=%j.CLASS_hyper_scaling.err
#SBATCH --mail-type=END,FAIL

module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

set -euo pipefail

# ============================================================================
# Container-only CLASS scalability benchmark for hyperbolic potential on LUMI
#
# Usage:
#   sbatch iDM/SLURM/CLASS_scalability_hyper_container_fixed.slurm
#
# Optional overrides (export before sbatch):
#   CLASSDIR, SIF_PATH, BIND_PATH, CLASS_BIN,
#   CORES_LIST, N_REPEATS,
#   STRONG_BASE_INI, WEAK_BASE_INI
# ============================================================================

CLASSDIR="${CLASSDIR:-/project/project_465002956/iDM}"
SIF_PATH="${SIF_PATH:-/project/project_465002956/cobaya.sif}"
BIND_PATH="${BIND_PATH:-/project/project_465002956}"
CLASS_BIN="${CLASS_BIN:-/project/project_465002956/iDM/class}"

# Full-node scaling set (LUMI CPU node)
CORES_LIST_STR="${CORES_LIST:-1 2 4 8 16 32 64 96 128}"
read -r -a CORES_LIST <<< "$CORES_LIST_STR"

N_REPEATS="${N_REPEATS:-3}"

# Hyperbolic-focused user cases
USER_CASES=(
  "hyper_cmb:pgo_hyperbolic_ic_cmb.ini"
  "hyper_bao:pgo_hyperbolic_ic_bao.ini"
)

STRONG_BASE_INI="${STRONG_BASE_INI:-pgo_hyperbolic_ic_cmb.ini}"
WEAK_BASE_INI="${WEAK_BASE_INI:-pgo_hyperbolic_ic_bao.ini}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$CLASSDIR/benchmark/scalability/${STAMP}_container_hyper_job${SLURM_JOB_ID:-noid}"
RAW_DIR="$RESULT_DIR/raw"
OUT_DIR="$RESULT_DIR/output"
TMP_INI_DIR="$RESULT_DIR/tmp_ini"
mkdir -p "$RAW_DIR" "$OUT_DIR" "$TMP_INI_DIR"

SYSTEM_INFO="$RESULT_DIR/system_info.txt"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "host=$(hostname)"
  echo "date=$(date -Iseconds)"
  echo "classdir=$CLASSDIR"
  echo "class_bin=$CLASS_BIN"
  echo "sif_path=$SIF_PATH"
  echo "bind_path=$BIND_PATH"
  echo "cores_list=${CORES_LIST[*]}"
  echo "repeats=$N_REPEATS"
  echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unknown}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unknown}"
  echo "slurm_cpus_on_node=${SLURM_CPUS_ON_NODE:-unknown}"
} > "$SYSTEM_INFO"

USER_RAW="$RAW_DIR/user_cases_raw.csv"
STRONG_RAW="$RAW_DIR/strong_raw.csv"
WEAK_RAW="$RAW_DIR/weak_raw.csv"

echo "case_id,ini_file,cores,repeat,wall_seconds,nodes,processes,exit_code" > "$USER_RAW"
echo "cores,repeat,wall_seconds,nodes,processes,exit_code" > "$STRONG_RAW"
echo "cores,repeat,wall_seconds,nodes,processes,exit_code" > "$WEAK_RAW"

ensure_exists() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
}

prepare_ini() {
  local src_ini="$1"
  local dst_ini="$2"
  local out_root="$3"

  cp "$src_ini" "$dst_ini"
  sed -i -E 's/^(root\s*=)/#\1/' "$dst_ini"
  sed -i -E 's/^(overwrite_root\s*=)/#\1/' "$dst_ini"
  sed -i -E 's/^(write_parameters\s*=)/#\1/' "$dst_ini"
  {
    echo ""
    echo "# benchmark overrides"
    echo "root = $out_root"
    echo "overwrite_root = yes"
    echo "write_parameters = no"
  } >> "$dst_ini"
}

run_timed_quiet() {
  local start_ns end_ns elapsed rc
  start_ns=$(date +%s%N)
  set +e
  "$@" >/dev/null 2>&1
  rc=$?
  set -e
  end_ns=$(date +%s%N)
  elapsed=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.6f", (e-s)/1e9}')
  printf '%s,%s\n' "$elapsed" "$rc"
}

cd "$CLASSDIR"

ensure_exists "$SIF_PATH"
ensure_exists "$CLASS_BIN"
ensure_exists "$CLASSDIR/$STRONG_BASE_INI"
ensure_exists "$CLASSDIR/$WEAK_BASE_INI"

for pair in "${USER_CASES[@]}"; do
  ini_file="${pair#*:}"
  ensure_exists "$CLASSDIR/$ini_file"
done

echo "[INFO] Running typical user cases (Section 7.3A)"
for pair in "${USER_CASES[@]}"; do
  case_id="${pair%%:*}"
  ini_file="${pair#*:}"

  for c in "${CORES_LIST[@]}"; do
    for ((r=1; r<=N_REPEATS; r++)); do
      tmp_ini="$TMP_INI_DIR/user_${case_id}_c${c}_r${r}.ini"
      out_root="$OUT_DIR/user_${case_id}_c${c}_r${r}"
      prepare_ini "$CLASSDIR/$ini_file" "$tmp_ini" "$out_root"

      result=$(run_timed_quiet srun --nodes=1 --ntasks=1 --cpus-per-task="$c" --cpu-bind=cores --hint=nomultithread --exclusive \
        singularity exec -B "$BIND_PATH" "$SIF_PATH" "$CLASS_BIN" "$tmp_ini")
      elapsed="${result%,*}"
      code="${result#*,}"

      echo "$case_id,$ini_file,$c,$r,$elapsed,1,1,$code" >> "$USER_RAW"
      echo "[USER] case=$case_id cores=$c repeat=$r wall=${elapsed}s exit=$code"
    done
  done
done

echo "[INFO] Running strong scaling (Section 7.3B)"
for c in "${CORES_LIST[@]}"; do
  for ((r=1; r<=N_REPEATS; r++)); do
    tmp_ini="$TMP_INI_DIR/strong_c${c}_r${r}.ini"
    out_root="$OUT_DIR/strong_c${c}_r${r}"
    prepare_ini "$CLASSDIR/$STRONG_BASE_INI" "$tmp_ini" "$out_root"

    result=$(run_timed_quiet srun --nodes=1 --ntasks=1 --cpus-per-task="$c" --cpu-bind=cores --hint=nomultithread --exclusive \
      singularity exec -B "$BIND_PATH" "$SIF_PATH" "$CLASS_BIN" "$tmp_ini")
    elapsed="${result%,*}"
    code="${result#*,}"

    echo "$c,$r,$elapsed,1,1,$code" >> "$STRONG_RAW"
    echo "[STRONG] cores=$c repeat=$r wall=${elapsed}s exit=$code"
  done
done

echo "[INFO] Running weak scaling (Section 7.3C, ensemble throughput mode)"
for c in "${CORES_LIST[@]}"; do
  for ((r=1; r<=N_REPEATS; r++)); do
    weak_dir="$OUT_DIR/weak_c${c}_r${r}"
    mkdir -p "$weak_dir"

    export WEAK_INI="$CLASSDIR/$WEAK_BASE_INI"
    export WEAK_TMP_DIR="$TMP_INI_DIR"
    export WEAK_OUT_DIR="$weak_dir"
    export WEAK_CORES="$c"
    export WEAK_REPEAT="$r"
    export SIF_PATH BIND_PATH CLASS_BIN

    result=$(run_timed_quiet srun --nodes=1 --ntasks="$c" --cpus-per-task=1 --cpu-bind=cores --hint=nomultithread --exclusive bash -lc '
      set -euo pipefail
      rid=${SLURM_PROCID:-0}
      ini="$WEAK_TMP_DIR/weak_c${WEAK_CORES}_r${WEAK_REPEAT}_task${rid}.ini"
      out_root="$WEAK_OUT_DIR/task${rid}"
      cp "$WEAK_INI" "$ini"
      sed -i -E "s/^(root\\s*=)/#\\1/" "$ini"
      sed -i -E "s/^(overwrite_root\\s*=)/#\\1/" "$ini"
      sed -i -E "s/^(write_parameters\\s*=)/#\\1/" "$ini"
      {
        echo ""
        echo "# weak scaling benchmark overrides"
        echo "root = $out_root"
        echo "overwrite_root = yes"
        echo "write_parameters = no"
      } >> "$ini"
      singularity exec -B "$BIND_PATH" "$SIF_PATH" "$CLASS_BIN" "$ini" >/dev/null 2>&1
    ')
    elapsed="${result%,*}"
    code="${result#*,}"

    echo "$c,$r,$elapsed,1,$c,$code" >> "$WEAK_RAW"
    echo "[WEAK] cores=$c repeat=$r wall=${elapsed}s exit=$code"
  done
done

python3 - "$RESULT_DIR" <<'PY'
import csv
import os
import statistics
import sys

result_dir = os.path.abspath(sys.argv[1])
raw_dir = os.path.join(result_dir, "raw")

user_raw = os.path.join(raw_dir, "user_cases_raw.csv")
strong_raw = os.path.join(raw_dir, "strong_raw.csv")
weak_raw = os.path.join(raw_dir, "weak_raw.csv")

out_a = os.path.join(result_dir, "section_7_3A_typical_user_cases.csv")
out_b = os.path.join(result_dir, "section_7_3B_strong_scaling.csv")
out_c = os.path.join(result_dir, "section_7_3C_weak_scaling.csv")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_ok_row(row, value_key="wall_seconds"):
    try:
        if int(row["exit_code"]) != 0:
            return None
        return float(row[value_key])
    except (KeyError, TypeError, ValueError):
        return None


def median_ok(rows, value_key="wall_seconds"):
    vals = []
    for row in rows:
        value = parse_ok_row(row, value_key)
        if value is not None:
            vals.append(value)
    return statistics.median(vals) if vals else None


# 7.3A Typical user cases
rows = read_csv(user_raw)
by_case_core = {}
for row in rows:
    try:
        key = (row["case_id"], row["ini_file"], int(row["cores"]))
    except (KeyError, TypeError, ValueError):
        continue
    by_case_core.setdefault(key, []).append(row)

case_ref = {}
for case_id, ini_file, cores in sorted(by_case_core):
    if case_id in case_ref:
        continue
    median = median_ok(by_case_core[(case_id, ini_file, cores)])
    if median is not None:
        case_ref[case_id] = median

with open(out_a, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "ini_file", "Number of cores", "Wall clock time", "Speed-up vs the first one", "Number of nodes", "Number of processes"])
    for case_id, ini_file, cores in sorted(by_case_core):
        median = median_ok(by_case_core[(case_id, ini_file, cores)])
        if median is None:
            continue
        ref = case_ref.get(case_id, median)
        speedup = ref / median if median > 0 else 0.0
        writer.writerow([case_id, ini_file, cores, f"{median:.6f}", f"{speedup:.6f}", 1, 1])

# 7.3B Strong scaling
rows = read_csv(strong_raw)
by_core = {}
for row in rows:
    try:
        cores = int(row["cores"])
    except (KeyError, TypeError, ValueError):
        continue
    by_core.setdefault(cores, []).append(row)

strong_ref = None
for cores in sorted(by_core):
    median = median_ok(by_core[cores])
    if median is not None:
        strong_ref = median
        break

with open(out_b, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Number of cores", "Wall clock time", "Speed-up vs the first one", "Number of nodes", "Number of processes"])
    for cores in sorted(by_core):
        median = median_ok(by_core[cores])
        if median is None:
            continue
        speedup = (strong_ref / median) if (strong_ref and median > 0) else 0.0
        writer.writerow([cores, f"{median:.6f}", f"{speedup:.6f}", 1, 1])

# 7.3C Weak scaling
rows = read_csv(weak_raw)
by_core = {}
for row in rows:
    try:
        cores = int(row["cores"])
    except (KeyError, TypeError, ValueError):
        continue
    by_core.setdefault(cores, []).append(row)

weak_ref = None
for cores in sorted(by_core):
    median = median_ok(by_core[cores])
    if median is not None:
        weak_ref = median
        break

with open(out_c, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Number of cores", "Wall clock time", "Speed-up vs the first one", "Number of nodes", "Number of processes"])
    for cores in sorted(by_core):
        median = median_ok(by_core[cores])
        if median is None:
            continue
        speedup = (weak_ref / median) if (weak_ref and median > 0) else 0.0
        writer.writerow([cores, f"{median:.6f}", f"{speedup:.6f}", 1, cores])

print("Wrote:")
print(out_a)
print(out_b)
print(out_c)
PY

echo "[DONE] Results in: $RESULT_DIR"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  sacct -j "$SLURM_JOB_ID" -o jobid,jobname,partition,account,state,consumedenergyraw
fi
