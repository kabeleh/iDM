#!/bin/bash -l
#SBATCH --job-name=pgo_build
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 01:00:00
#SBATCH --output %j.pgo_build.out
#SBATCH --error %j.pgo_build.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Profile-Guided Optimization (PGO) build for CLASS
##
## Three phases:
##   1. Instrumented build (compile with -fprofile-generate)
##   2. Profile generation (run training workloads to create .gcda profiles)
##   3. Optimized build   (compile with -fprofile-use)
##
## Training set: 25 .ini files covering all potentials (IC + tracking + bug cases).
## Normal workloads exercise both BAO (mPk only) and CMB (Cls + lensing + mPk).
## Bug workloads exercise error-handling paths (expected to fail with exit 1).

# set -euo pipefail

## Load software environment
module load Python foss
source my_2025-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CLASS_DIR="/home/users/u103677/iDM"
cd "$CLASS_DIR"

# PROFILE_DIR="${CLASS_DIR}/pgo_profiles"

# echo "============================================"
# echo "PGO Build — Phase 1: Instrumented Build"
# echo "============================================"

# make clean
# make class -j
# # make class -j OPTFLAG="-O2 -fprofile-generate=${PROFILE_DIR}"

# echo "Instrumented binary built."
# echo ""

# # Remove stale profiles from previous runs
# rm -rf "${PROFILE_DIR}"
# mkdir -p "${PROFILE_DIR}"

echo "============================================"
echo "PGO Build — Phase 2: Profile Generation"
echo "============================================"

# --- Normal workloads (should succeed) ---
NORMAL_INIS=(
    # IC BAO (8 potentials)
    "pgo_Bean_ic_bao.ini"
    "pgo_cosine_ic_bao.ini"
    "pgo_DoubleExp_ic_bao.ini"
    "pgo_exponential_ic_bao.ini"
    "pgo_hyperbolic_ic_bao.ini"
    "pgo_pNG_ic_bao.ini"
    "pgo_power-law_ic_bao.ini"
    "pgo_SqE_ic_bao.ini"
    # IC CMB (8 potentials)
    "pgo_Bean_ic_cmb.ini"
    "pgo_cosine_ic_cmb.ini"
    "pgo_DoubleExp_ic_cmb.ini"
    "pgo_exponential_ic_cmb.ini"
    "pgo_hyperbolic_ic_cmb.ini"
    "pgo_pNG_ic_cmb.ini"
    "pgo_power-law_ic_cmb.ini"
    "pgo_SqE_ic_cmb.ini"
    # Tracking (2 potentials × 2 workloads)
    "pgo_hyperbolic_tracking_bao.ini"
    "pgo_hyperbolic_tracking_cmb.ini"
    "pgo_BeanAdS_tracking_bao.ini"
    "pgo_BeanAdS_tracking_cmb.ini"
)

# --- Bug workloads (expected to fail; exercise error paths) ---
BUG_INIS=(
    "pgo_bug_bean_ads_vacuum.ini"
    "pgo_bug_large_cdm_c.ini"
    "pgo_bug_shooting_fails.ini"
    "pgo_bug_tanh_underflow.ini"
    "pgo_bug_zriddr_not_bracketed.ini"
)

TOTAL=${#NORMAL_INIS[@]}
PASSED=0
FAILED=0
FAILED_LIST=""

echo ""
echo "Running ${TOTAL} normal training workloads..."
echo ""

for ini in "${NORMAL_INIS[@]}"; do
    echo -n "  ${ini}: "
    if ./class "${ini}" > /dev/null 2>&1; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${ini}"
    fi
done

echo ""
echo "Normal workloads: ${PASSED}/${TOTAL} passed, ${FAILED}/${TOTAL} failed"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed:${FAILED_LIST}"
    echo ""
    echo "ERROR: Some normal workloads failed. PGO profiles may be incomplete."
    echo "Continuing anyway — check failures above."
fi

echo ""
echo "Running ${#BUG_INIS[@]} bug workloads (expected failures)..."
echo ""

for ini in "${BUG_INIS[@]}"; do
    echo -n "  ${ini}: "
    if ./class "${ini}" > /dev/null 2>&1; then
        echo "UNEXPECTED PASS (expected failure)"
    else
        echo "FAIL (expected)"
    fi
done

# echo ""
# echo "============================================"
# echo "PGO Build — Phase 3: Optimized Build"
# echo "============================================"

# make clean
# make class -j OPTFLAG="-O2 -fprofile-use=${PROFILE_DIR} -fprofile-correction"

# echo ""
# echo "============================================"
# echo "PGO build complete."
# echo "Optimized binary: ${CLASS_DIR}/class"
# echo "============================================"

# # Quick smoke test with a BAO workload
# echo ""
# echo "Smoke test..."
# if ./class pgo_Bean_ic_bao.ini > /dev/null 2>&1; then
#     echo "Smoke test PASSED."
# else
#     echo "Smoke test FAILED — check the optimized binary."
#     exit 1
# fi


#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw