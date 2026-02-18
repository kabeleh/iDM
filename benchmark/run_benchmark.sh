#!/usr/bin/env bash
# ============================================================================
# CLASS compiler benchmark — run_benchmark.sh
#
# Usage:
#   ./benchmark/run_benchmark.sh <compiler_label>   (e.g. "gcc" or "clang")
#
# Prerequisites:
#   - CLASS must already be compiled (./class binary must exist)
#   - Run from the CLASS root directory
#
# What it does:
#   1. Runs iDM.ini and all pgo_*.ini files N_RUNS times each
#   2. Records wall-clock time for every run
#   3. Saves output data files under benchmark/<label>/output/
#   4. Writes timings to benchmark/<label>/timings.csv
#   5. Prints per-file averages; if results for another compiler already
#      exist, prints a side-by-side comparison table
# ============================================================================
set -euo pipefail

# ---- configurable parameters ------------------------------------------------
N_RUNS=5
# ini files to benchmark (relative to CLASS root)
INI_FILES=(
    iDM.ini
    pgo_doubleexp_bao.ini
    pgo_doubleexp_cmb.ini
    pgo_doubleexp_cmb_shooting_fails.ini
    pgo_hyperbolic_bao.ini
    pgo_hyperbolic_cmb.ini
    pgo_segfault.ini
)
# ini files that are *expected* to fail (CLASS returns non-zero)
EXPECTED_FAIL=(
    pgo_segfault.ini
    pgo_doubleexp_cmb_shooting_fails.ini
)
# ------------------------------------------------------------------------------

CLASSDIR="$(cd "$(dirname "$0")/.." && pwd)"
CLASS_BIN="${CLASSDIR}/class"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <compiler_label>  (e.g. gcc, clang)"
    exit 1
fi

LABEL="$1"
BENCH_DIR="${CLASSDIR}/benchmark/${LABEL}"
OUT_DIR="${BENCH_DIR}/output"
TIMINGS_CSV="${BENCH_DIR}/timings.csv"
TMP_DIR="${BENCH_DIR}/tmp_ini"

# Check that the CLASS binary exists
if [[ ! -x "$CLASS_BIN" ]]; then
    echo "ERROR: CLASS binary not found at ${CLASS_BIN}"
    echo "       Compile CLASS first, then re-run this script."
    exit 1
fi

# Helper: is this ini expected to fail?
is_expected_fail() {
    local ini="$1"
    for f in "${EXPECTED_FAIL[@]}"; do
        [[ "$ini" == "$f" ]] && return 0
    done
    return 1
}

# ---- set up directories -----------------------------------------------------
mkdir -p "$OUT_DIR" "$TMP_DIR"
echo "ini_file,run,wall_seconds,exit_code" > "$TIMINGS_CSV"

echo "============================================================"
echo " CLASS compiler benchmark"
echo " Compiler label : ${LABEL}"
echo " Runs per file  : ${N_RUNS}"
echo " Output dir     : ${OUT_DIR}"
echo " Timings file   : ${TIMINGS_CSV}"
echo "============================================================"
echo ""

# ---- prepare temporary ini files with controlled root ------------------------
# CLASS parser rejects duplicate parameter names, so we comment out any existing
# root / overwrite_root / write_parameters lines, then append our overrides.
for ini in "${INI_FILES[@]}"; do
    base="${ini%.ini}"
    tmp_ini="${TMP_DIR}/${ini}"
    cp "${CLASSDIR}/${ini}" "$tmp_ini"
    # Comment out existing lines that we need to override
    sed -i -E 's/^(root\s*=)/#\1/'              "$tmp_ini"
    sed -i -E 's/^(overwrite_root\s*=)/#\1/'    "$tmp_ini"
    sed -i -E 's/^(write_parameters\s*=)/#\1/'  "$tmp_ini"
    # Append our overrides
    {
        echo ""
        echo "# --- benchmark overrides ---"
        echo "root = ${OUT_DIR}/${base}"
        echo "overwrite_root = yes"
        echo "write_parameters = yes"
    } >> "$tmp_ini"
done

# ---- run benchmarks ----------------------------------------------------------
declare -A SUM_TIMES   # associative array:  ini -> cumulative seconds
declare -A RUN_COUNTS  # associative array:  ini -> number of successful timings
declare -A AVG_TIMES   # filled after the loop

for ini in "${INI_FILES[@]}"; do
    SUM_TIMES["$ini"]=0
    RUN_COUNTS["$ini"]=0

    echo "---- ${ini} ----"
    tmp_ini="${TMP_DIR}/${ini}"

    for ((r = 1; r <= N_RUNS; r++)); do
        printf "  run %d/%d ... " "$r" "$N_RUNS"

        # time the run (wall clock, seconds with 3 decimals)
        start_ns=$(date +%s%N)
        set +e
        "${CLASS_BIN}" "$tmp_ini" > /dev/null 2>&1
        exit_code=$?
        set -e
        end_ns=$(date +%s%N)

        elapsed=$(awk "BEGIN {printf \"%.3f\", ($end_ns - $start_ns) / 1e9}")

        # record
        echo "${ini},${r},${elapsed},${exit_code}" >> "$TIMINGS_CSV"

        if [[ $exit_code -eq 0 ]]; then
            printf "%.3fs  (ok)\n" "$elapsed"
        else
            if is_expected_fail "$ini"; then
                printf "%.3fs  (expected failure, exit %d)\n" "$elapsed" "$exit_code"
            else
                printf "%.3fs  (UNEXPECTED failure, exit %d)\n" "$elapsed" "$exit_code"
            fi
        fi

        # accumulate for average (always, even for failures — we're timing the run)
        SUM_TIMES["$ini"]=$(awk "BEGIN {printf \"%.3f\", ${SUM_TIMES[$ini]} + $elapsed}")
        RUN_COUNTS["$ini"]=$((RUN_COUNTS["$ini"] + 1))
    done
    echo ""
done

# ---- compute & display averages ---------------------------------------------
echo "============================================================"
echo " Average wall-clock times  [${LABEL}]"
echo "============================================================"
printf "%-45s %10s\n" "ini file" "avg (s)"
printf "%-45s %10s\n" "---------------------------------------------" "----------"

for ini in "${INI_FILES[@]}"; do
    n="${RUN_COUNTS[$ini]}"
    if [[ "$n" -gt 0 ]]; then
        avg=$(awk "BEGIN {printf \"%.3f\", ${SUM_TIMES[$ini]} / $n}")
    else
        avg="N/A"
    fi
    AVG_TIMES["$ini"]="$avg"
    printf "%-45s %10s\n" "$ini" "$avg"
done
echo ""

# ---- cross-compiler comparison (if another label's timings exist) ------------
echo "============================================================"
echo " Cross-compiler comparison"
echo "============================================================"

found_other=0
for other_dir in "${CLASSDIR}"/benchmark/*/; do
    other_label="$(basename "$other_dir")"
    [[ "$other_label" == "$LABEL" ]] && continue
    other_csv="${other_dir}/timings.csv"
    [[ -f "$other_csv" ]] || continue

    found_other=1
    echo ""
    echo "  Comparing: ${LABEL}  vs  ${other_label}"
    echo ""
    printf "%-45s %10s %10s %10s\n" "ini file" "${LABEL}(s)" "${other_label}(s)" "speedup"
    printf "%-45s %10s %10s %10s\n" "---------------------------------------------" "----------" "----------" "----------"

    for ini in "${INI_FILES[@]}"; do
        my_avg="${AVG_TIMES[$ini]}"

        # compute average from the other CSV
        other_avg=$(awk -F',' -v ini="$ini" '
            NR > 1 && $1 == ini {sum += $3; n++}
            END {if (n > 0) printf "%.3f", sum/n; else print "N/A"}
        ' "$other_csv")

        if [[ "$my_avg" != "N/A" && "$other_avg" != "N/A" ]]; then
            speedup=$(awk "BEGIN {printf \"%.2fx\", $other_avg / $my_avg}")
        else
            speedup="N/A"
        fi

        printf "%-45s %10s %10s %10s\n" "$ini" "$my_avg" "$other_avg" "$speedup"
    done
done

if [[ "$found_other" -eq 0 ]]; then
    echo "  (No other compiler results found yet."
    echo "   Recompile CLASS with the other compiler and run this script again.)"
fi

echo ""
echo "Done. Timings saved to ${TIMINGS_CSV}"
echo "Output data saved to ${OUT_DIR}/"
