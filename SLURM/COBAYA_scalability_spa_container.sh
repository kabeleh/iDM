#!/bin/bash -l
#SBATCH --job-name=COBAYA_SPA_scaling
#SBATCH --account=project_465002956
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=%j.COBAYA_SPA_scaling.out
#SBATCH --error=%j.COBAYA_SPA_scaling.err
#SBATCH --mail-type=END,FAIL

module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

set -euo pipefail

# ============================================================================
# Container-only Cobaya scalability benchmark for SPA MCMC configuration.
#
# Usage:
#   sbatch SLURM/COBAYA_scalability_spa_container.sh
#
# Optional overrides (export before sbatch):
#   CLASSDIR, SIF_PATH, BIND_PATH, SCRATCH_BIND,
#   COBAYA_CONFIG, TASKS_LIST, CPUS_PER_TASK_LIST, N_REPEATS,
#   BENCHMARK_MAX_SAMPLES, BENCHMARK_RMINUS1_STOP, BENCHMARK_RMINUS1_CL_STOP,
#   BENCHMARK_OUTPUT_EVERY, BENCHMARK_LEARN_EVERY,
#   ESS_PARAMS, PYTHON_EXE, REQUIRE_GETDIST
#
# Notes:
#   - Default config points to Bench_SPA.yml to keep production SPA output safe.
#   - For textbook-style fixed-work scaling, convergence stopping is disabled by
#     default (Rminus thresholds set to 0) and runs stop by max_samples.
# ============================================================================

CLASSDIR="${CLASSDIR:-/project/project_465002956/iDM}"
SIF_PATH="${SIF_PATH:-/project/project_465002956/cobaya.sif}"
BIND_PATH="${BIND_PATH:-/project/project_465002956}"
SCRATCH_BIND="${SCRATCH_BIND:-/scratch/project_465002956}"

# Separate benchmark config to avoid writing into production SPA output.
COBAYA_CONFIG="${COBAYA_CONFIG:-/project/project_465002956/iDM/Cobaya/MCMC/Bench_SPA.yml}"

# Number of Cobaya chain tasks to benchmark.
TASKS_LIST_STR="${TASKS_LIST:-2 4 8 16 32 64 128}"
read -r -a TASKS_LIST <<< "$TASKS_LIST_STR"

# Cores per MPI task (per chain) to benchmark.
CPUS_PER_TASK_LIST_STR="${CPUS_PER_TASK_LIST:-128 64 32 16 8 4 2 1}"
read -r -a CPUS_PER_TASK_LIST <<< "$CPUS_PER_TASK_LIST_STR"

# One LUMI-C node has 128 physical cores in this setup.
NODE_CORE_CAPACITY="${NODE_CORE_CAPACITY:-128}"
N_REPEATS="${N_REPEATS:-3}"

# Benchmark controls to keep runs finite and comparable.
BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-4000}"
BENCHMARK_RMINUS1_STOP="${BENCHMARK_RMINUS1_STOP:-0.0}"
BENCHMARK_RMINUS1_CL_STOP="${BENCHMARK_RMINUS1_CL_STOP:-0.0}"
BENCHMARK_OUTPUT_EVERY="${BENCHMARK_OUTPUT_EVERY:-200}"
BENCHMARK_LEARN_EVERY="${BENCHMARK_LEARN_EVERY:-200}"

# Parameters used for ESS diagnostics (if found in chain outputs).
ESS_PARAMS_STR="${ESS_PARAMS:-n_s H0 omega_b omega_cdm cdm_c}"

# Python executable inside the Singularity container.
PYTHON_EXE="${PYTHON_EXE:-python3}"

# If set to 1, fail early when getdist is missing inside the container.
REQUIRE_GETDIST="${REQUIRE_GETDIST:-0}"

if singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" "$PYTHON_EXE" -c "import getdist" >/dev/null 2>&1; then
    GETDIST_AVAILABLE=1
else
    GETDIST_AVAILABLE=0
fi

if [[ "$REQUIRE_GETDIST" == "1" && "$GETDIST_AVAILABLE" != "1" ]]; then
    echo "[ERROR] getdist is not available inside the container, but REQUIRE_GETDIST=1"
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$CLASSDIR/benchmark/scalability/${STAMP}_container_cobaya_spa_job${SLURM_JOB_ID:-noid}"
RAW_DIR="$RESULT_DIR/raw"
OUT_DIR="$RESULT_DIR/output"
TMP_CFG_DIR="$RESULT_DIR/tmp_cfg"
mkdir -p "$RAW_DIR" "$OUT_DIR" "$TMP_CFG_DIR"

SYSTEM_INFO="$RESULT_DIR/system_info.txt"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "host=$(hostname)"
  echo "date=$(date -Iseconds)"
  echo "classdir=$CLASSDIR"
  echo "sif_path=$SIF_PATH"
  echo "bind_path=$BIND_PATH"
  echo "scratch_bind=$SCRATCH_BIND"
  echo "cobaya_config=$COBAYA_CONFIG"
  echo "tasks_list=${TASKS_LIST[*]}"
  echo "cpus_per_task_list=${CPUS_PER_TASK_LIST[*]}"
  echo "node_core_capacity=$NODE_CORE_CAPACITY"
  echo "repeats=$N_REPEATS"
  echo "benchmark_max_samples=$BENCHMARK_MAX_SAMPLES"
  echo "benchmark_rminus1_stop=$BENCHMARK_RMINUS1_STOP"
  echo "benchmark_rminus1_cl_stop=$BENCHMARK_RMINUS1_CL_STOP"
  echo "benchmark_output_every=$BENCHMARK_OUTPUT_EVERY"
  echo "benchmark_learn_every=$BENCHMARK_LEARN_EVERY"
  echo "ess_params=$ESS_PARAMS_STR"
    echo "getdist_available=$GETDIST_AVAILABLE"
    echo "require_getdist=$REQUIRE_GETDIST"
  echo "python_exe=$PYTHON_EXE"
  echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unknown}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unknown}"
  echo "slurm_cpus_on_node=${SLURM_CPUS_ON_NODE:-unknown}"
} > "$SYSTEM_INFO"

RAW_CSV="$RAW_DIR/cobaya_spa_raw.csv"
SUMMARY_CSV="$RESULT_DIR/section_7_3A_typical_user_cases_cobaya_spa.csv"
STRONG_CSV="$RESULT_DIR/section_7_3B_strong_scaling_cobaya_spa.csv"
WEAK_CSV="$RESULT_DIR/section_7_3C_weak_scaling_cobaya_spa.csv"

echo "tasks,cpus_per_task,cores,repeat,wall_seconds,nodes,processes,status,exit_code,accepted_steps,acceptance_rate,rminus1,rminus1_cl,ess_min,ess_mean,ess_params_found,config" > "$RAW_CSV"

ensure_exists() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
}

prepare_cfg() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local out_root="$3"

  cp "$src_cfg" "$dst_cfg"
  sed -i -E 's/^(\s*output\s*:)/#\1/' "$dst_cfg"
  {
    echo ""
    echo "# benchmark overrides"
    echo "output: $out_root"
    echo "resume: false"
    echo "sampler:"
    echo "  mcmc:"
    echo "    output_every: $BENCHMARK_OUTPUT_EVERY"
    echo "    learn_every: $BENCHMARK_LEARN_EVERY"
    echo "    max_samples: $BENCHMARK_MAX_SAMPLES"
    echo "    Rminus1_stop: $BENCHMARK_RMINUS1_STOP"
    echo "    Rminus1_cl_stop: $BENCHMARK_RMINUS1_CL_STOP"
  } >> "$dst_cfg"
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

run_container_python() {
    singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" "$PYTHON_EXE" "$@"
}

collect_cobaya_metrics() {
  local root_prefix="$1"
    run_container_python - "$root_prefix" "$ESS_PARAMS_STR" "$GETDIST_AVAILABLE" <<'PY'
import math
import os
import sys

root = sys.argv[1]
ess_params = [p for p in sys.argv[2].split() if p]
getdist_available = sys.argv[3] == "1"

accepted_steps = float("nan")
acceptance_rate = float("nan")
rminus1 = float("nan")
rminus1_cl = float("nan")
ess_min = float("nan")
ess_mean = float("nan")
ess_found = 0

progress_path = root + ".progress"
if os.path.exists(progress_path):
    with open(progress_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) >= 2:
        header = lines[0].split()
        values = lines[-1].split()
        if len(values) >= len(header):
            row = dict(zip(header, values))

            def safe_float(name):
                try:
                    return float(row.get(name, "nan"))
                except Exception:
                    return float("nan")

            accepted_steps = safe_float("N")
            acceptance_rate = safe_float("acceptance_rate")
            rminus1 = safe_float("Rminus1")
            rminus1_cl = safe_float("Rminus1_cl")

if getdist_available:
    try:
        from getdist.mcsamples import loadMCSamples

        samples = loadMCSamples(root, no_cache=True)
        names = samples.getParamNames()
        ess_vals = []
        for pname in ess_params:
            idx = names.numberOfName(pname)
            if idx is None:
                continue
            try:
                val = float(samples.getEffectiveSamples(j=idx))
                if math.isfinite(val) and val > 0:
                    ess_vals.append(val)
            except Exception:
                continue
        if ess_vals:
            ess_found = len(ess_vals)
            ess_min = min(ess_vals)
            ess_mean = sum(ess_vals) / len(ess_vals)
    except Exception:
        pass

def fmt(x):
    return "nan" if not math.isfinite(x) else f"{x:.6f}"

print(",".join([
    fmt(accepted_steps),
    fmt(acceptance_rate),
    fmt(rminus1),
    fmt(rminus1_cl),
    fmt(ess_min),
    fmt(ess_mean),
    str(ess_found),
]))
PY
}

ensure_exists "$SIF_PATH"
ensure_exists "$COBAYA_CONFIG"

for t in "${TASKS_LIST[@]}"; do
  if (( t < 2 )); then
    echo "[ERROR] TASKS_LIST entries must be >= 2 (1-chain case intentionally excluded). Got: $t"
    exit 1
  fi
done

for c in "${CPUS_PER_TASK_LIST[@]}"; do
  if (( c < 1 )); then
    echo "[ERROR] CPUS_PER_TASK_LIST entries must be >= 1. Got: $c"
    exit 1
  fi
done

echo "[INFO] Running Cobaya SPA scaling benchmark"
for t in "${TASKS_LIST[@]}"; do
  for c in "${CPUS_PER_TASK_LIST[@]}"; do
    total_cores=$((t * c))

    if (( total_cores > NODE_CORE_CAPACITY )); then
      echo "$t,$c,$total_cores,0,0.000000,1,$t,skipped_invalid,NA,nan,nan,nan,nan,nan,nan,0,$(basename "$COBAYA_CONFIG")" >> "$RAW_CSV"
      echo "[SKIP] tasks=$t cpus_per_task=$c total_cores=$total_cores exceeds node capacity $NODE_CORE_CAPACITY"
      continue
    fi

    for ((r=1; r<=N_REPEATS; r++)); do
      cfg_run_dir="$OUT_DIR/tasks${t}_c${c}_r${r}"
      mkdir -p "$cfg_run_dir"

      tmp_cfg="$TMP_CFG_DIR/spa_tasks${t}_c${c}_r${r}.yml"
      prepare_cfg "$COBAYA_CONFIG" "$tmp_cfg" "$cfg_run_dir/chain"

      result=$(run_timed_quiet srun --nodes=1 --ntasks="$t" --mpi=pmi2 --exclusive --cpus-per-task="$c" --cpu-bind=threads \
        singularity exec -B "$BIND_PATH" -B "$SCRATCH_BIND" "$SIF_PATH" \
        cobaya-run "$tmp_cfg" --allow-changes)

      elapsed="${result%,*}"
      code="${result#*,}"
      metrics=$(collect_cobaya_metrics "$cfg_run_dir/chain")

      echo "$t,$c,$total_cores,$r,$elapsed,1,$t,ran,$code,$metrics,$(basename "$COBAYA_CONFIG")" >> "$RAW_CSV"
      echo "[COBAYA] tasks=$t cpus_per_task=$c cores=$total_cores repeat=$r wall=${elapsed}s exit=$code metrics=[$metrics]"
    done
  done

done

run_container_python - "$RAW_CSV" "$SUMMARY_CSV" "$STRONG_CSV" "$WEAK_CSV" <<'PY'
import csv
import math
import statistics
import sys

raw_path = sys.argv[1]
summary_path = sys.argv[2]
strong_path = sys.argv[3]
weak_path = sys.argv[4]

with open(raw_path, newline="") as f:
    rows = list(csv.DictReader(f))

groups = {}
metric_groups = {}
for row in rows:
    try:
        if row.get("status") != "ran":
            continue
        if int(row["exit_code"]) != 0:
            continue
        key = (int(row["tasks"]), int(row["cpus_per_task"]), int(row["cores"]), row["config"])
        wall = float(row["wall_seconds"])
        groups.setdefault(key, []).append(wall)

        def safe_float(name):
            try:
                x = float(row.get(name, "nan"))
            except Exception:
                return None
            return x if math.isfinite(x) else None

        metric_groups.setdefault(key, {
            "accepted_steps": [],
            "acceptance_rate": [],
            "rminus1": [],
            "rminus1_cl": [],
            "ess_min": [],
            "ess_mean": [],
        })
        for metric_name in metric_groups[key]:
            v = safe_float(metric_name)
            if v is not None:
                metric_groups[key][metric_name].append(v)
    except Exception:
        continue

ordered = sorted(groups.keys(), key=lambda k: k[2])

common_headers = [
    "Number of cores",
    "Wall clock time",
    "Speed-up vs the first one",
    "Number of nodes",
    "Number of processes",
    "Tasks",
    "Cores per chain",
    "Repeats used",
    "Wall stddev",
    "Wall stderr",
    "Relative error (%)",
    "Accepted steps (median)",
    "Acceptance rate (median)",
    "Rminus1 (median)",
    "Rminus1_cl (median)",
    "ESS min (median)",
    "ESS mean (median)",
    "Config",
]

def write_empty(path, headers):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

def med_or_nan(values):
    return statistics.median(values) if values else float("nan")

if not ordered:
    write_empty(summary_path, common_headers)
    write_empty(strong_path, common_headers + ["Strong efficiency (%)", "Strong reference cores"])
    write_empty(weak_path, common_headers + ["Weak constancy T_ref/T", "Weak throughput speed-up", "Weak reference tasks"])
    sys.exit(0)

baseline = statistics.median(groups[ordered[0]])
summary_rows = []

with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(common_headers)

    for key in ordered:
        tasks, cpus_per_task, cores, config = key
        samples = groups[key]
        n = len(samples)
        med = statistics.median(samples)
        stddev = statistics.stdev(samples) if n > 1 else 0.0
        stderr = stddev / math.sqrt(n) if n > 0 else 0.0
        rel_err_pct = (100.0 * stderr / med) if med > 0 else 0.0
        speedup = baseline / med if med > 0 else 0.0

        metrics = metric_groups.get(key, {})
        accepted_med = med_or_nan(metrics.get("accepted_steps", []))
        accrate_med = med_or_nan(metrics.get("acceptance_rate", []))
        rminus_med = med_or_nan(metrics.get("rminus1", []))
        rminuscl_med = med_or_nan(metrics.get("rminus1_cl", []))
        essmin_med = med_or_nan(metrics.get("ess_min", []))
        essmean_med = med_or_nan(metrics.get("ess_mean", []))

        w.writerow([
            cores,
            f"{med:.6f}",
            f"{speedup:.6f}",
            1,
            tasks,
            tasks,
            cpus_per_task,
            n,
            f"{stddev:.6f}",
            f"{stderr:.6f}",
            f"{rel_err_pct:.4f}",
            f"{accepted_med:.6f}" if math.isfinite(accepted_med) else "nan",
            f"{accrate_med:.6f}" if math.isfinite(accrate_med) else "nan",
            f"{rminus_med:.6f}" if math.isfinite(rminus_med) else "nan",
            f"{rminuscl_med:.6f}" if math.isfinite(rminuscl_med) else "nan",
            f"{essmin_med:.6f}" if math.isfinite(essmin_med) else "nan",
            f"{essmean_med:.6f}" if math.isfinite(essmean_med) else "nan",
            config,
        ])

        summary_rows.append({
            "tasks": tasks,
            "cpus_per_task": cpus_per_task,
            "cores": cores,
            "wall": med,
            "n": n,
            "stddev": stddev,
            "stderr": stderr,
            "rel_err_pct": rel_err_pct,
            "accepted_med": accepted_med,
            "accrate_med": accrate_med,
            "rminus_med": rminus_med,
            "rminuscl_med": rminuscl_med,
            "essmin_med": essmin_med,
            "essmean_med": essmean_med,
            "config": config,
        })

with open(strong_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(common_headers + ["Strong efficiency (%)", "Strong reference cores"])

    by_tasks = {}
    for row in summary_rows:
        by_tasks.setdefault((row["tasks"], row["config"]), []).append(row)

    for (tasks, config), rows_t in sorted(by_tasks.items(), key=lambda kv: kv[0][0]):
        rows_t.sort(key=lambda r: r["cores"])
        ref = rows_t[0]
        ref_wall = ref["wall"]
        ref_cores = ref["cores"]
        for r in rows_t:
            strong_speedup = ref_wall / r["wall"] if r["wall"] > 0 else 0.0
            core_ratio = r["cores"] / ref_cores if ref_cores > 0 else 0.0
            eff = 100.0 * strong_speedup / core_ratio if core_ratio > 0 else 0.0

            w.writerow([
                r["cores"],
                f"{r['wall']:.6f}",
                f"{strong_speedup:.6f}",
                1,
                r["tasks"],
                r["tasks"],
                r["cpus_per_task"],
                r["n"],
                f"{r['stddev']:.6f}",
                f"{r['stderr']:.6f}",
                f"{r['rel_err_pct']:.4f}",
                f"{r['accepted_med']:.6f}" if math.isfinite(r["accepted_med"]) else "nan",
                f"{r['accrate_med']:.6f}" if math.isfinite(r["accrate_med"]) else "nan",
                f"{r['rminus_med']:.6f}" if math.isfinite(r["rminus_med"]) else "nan",
                f"{r['rminuscl_med']:.6f}" if math.isfinite(r["rminuscl_med"]) else "nan",
                f"{r['essmin_med']:.6f}" if math.isfinite(r["essmin_med"]) else "nan",
                f"{r['essmean_med']:.6f}" if math.isfinite(r["essmean_med"]) else "nan",
                config,
                f"{eff:.4f}",
                ref_cores,
            ])

with open(weak_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(common_headers + ["Weak constancy T_ref/T", "Weak throughput speed-up", "Weak reference tasks"])

    by_cpt = {}
    for row in summary_rows:
        by_cpt.setdefault((row["cpus_per_task"], row["config"]), []).append(row)

    for (cpus_per_task, config), rows_c in sorted(by_cpt.items(), key=lambda kv: kv[0][0]):
        rows_c.sort(key=lambda r: r["tasks"])
        ref = rows_c[0]
        ref_wall = ref["wall"]
        ref_tasks = ref["tasks"]
        for r in rows_c:
            constancy = ref_wall / r["wall"] if r["wall"] > 0 else 0.0
            throughput = (r["tasks"] / ref_tasks) * constancy if ref_tasks > 0 else 0.0

            w.writerow([
                r["cores"],
                f"{r['wall']:.6f}",
                f"{baseline / r['wall'] if r['wall'] > 0 else 0.0:.6f}",
                1,
                r["tasks"],
                r["tasks"],
                r["cpus_per_task"],
                r["n"],
                f"{r['stddev']:.6f}",
                f"{r['stderr']:.6f}",
                f"{r['rel_err_pct']:.4f}",
                f"{r['accepted_med']:.6f}" if math.isfinite(r["accepted_med"]) else "nan",
                f"{r['accrate_med']:.6f}" if math.isfinite(r["accrate_med"]) else "nan",
                f"{r['rminus_med']:.6f}" if math.isfinite(r["rminus_med"]) else "nan",
                f"{r['rminuscl_med']:.6f}" if math.isfinite(r["rminuscl_med"]) else "nan",
                f"{r['essmin_med']:.6f}" if math.isfinite(r["essmin_med"]) else "nan",
                f"{r['essmean_med']:.6f}" if math.isfinite(r["essmean_med"]) else "nan",
                config,
                f"{constancy:.6f}",
                f"{throughput:.6f}",
                ref_tasks,
            ])
PY

echo "[INFO] Benchmark complete"
echo "[INFO] Raw data:    $RAW_CSV"
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Strong CSV:  $STRONG_CSV"
echo "[INFO] Weak CSV:    $WEAK_CSV"
echo "[INFO] Result dir:  $RESULT_DIR"

# Energy accounting from SLURM
sacct -j "$SLURM_JOB_ID" -o jobid,jobname,partition,account,state,consumedenergyraw
