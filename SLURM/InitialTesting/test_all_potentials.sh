#!/bin/bash -l
#SBATCH --job-name=test_all_potentials
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 11
#SBATCH --ntasks-per-node 11
#SBATCH --cpus-per-task 8
#SBATCH --time 06:00:00
#SBATCH --output %j.test_all_potentials.out
#SBATCH --error %j.test_all_potentials.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

# Number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

YAML_DIR="/home/users/u103677/iDM/Cobaya/MCMC"

# List of all YAML files to test
YAMLS=(
    # Attractor (tracking) runs — potentials with viable tracking solutions
    "hyperbolic_Planck_tracking_MCMC.yml"
    "BeanAdS_Planck_tracking_MCMC.yml"
    # IC (Initial Conditions) runs — all non-LCDM potentials
    "cosine_Planck_InitCond_MCMC.yml"
    "hyperbolic_Planck_InitCond_MCMC.yml"
    "exponential_Planck_InitCond_MCMC.yml"
    "Bean_Planck_InitCond_MCMC.yml"
    "BeanAdS_Planck_InitCond_MCMC.yml"
    "DoubleExp_Planck_InitCond_MCMC.yml"
    "SqE_Planck_InitCond_MCMC.yml"
    "power-law_Planck_InitCond_MCMC.yml"
    "pNG_Planck_InitCond_MCMC.yml"
)

TOTAL=${#YAMLS[@]}

echo "============================================"
echo "Testing all ${TOTAL} potential configurations (parallel)"
echo "============================================"
echo ""

# Launch all tests in parallel, each using 1 MPI task
PIDS=()
for i in "${!YAMLS[@]}"; do
    YAML_FILE="${YAMLS[$i]}"
    FULL_PATH="${YAML_DIR}/${YAML_FILE}"
    LOG_FILE="${SLURM_JOB_ID}_test_${YAML_FILE%.yml}.out"

    srun -n 1 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK \
        cobaya-run "${FULL_PATH}" --test --debug > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

# Wait for all and collect exit codes
EXIT_CODES=()
for PID in "${PIDS[@]}"; do
    wait "$PID"
    EXIT_CODES+=($?)
done

# Report results
PASSED=0
FAILED=0
FAILED_LIST=""

for i in "${!YAMLS[@]}"; do
    YAML_FILE="${YAMLS[$i]}"
    EC=${EXIT_CODES[$i]}
    LOG_FILE="${SLURM_JOB_ID}_test_${YAML_FILE%.yml}.out"

    echo "--------------------------------------------"
    echo "Result: ${YAML_FILE}"
    echo "--------------------------------------------"

    if [ "$EC" -eq 0 ]; then
        echo "  => PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  => FAILED (exit code ${EC})"
        echo "  Log output (last 20 lines):"
        tail -20 "${LOG_FILE}" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${YAML_FILE} (exit code ${EC})"
    fi
    echo ""
done

echo "============================================"
echo "SUMMARY: ${PASSED}/${TOTAL} passed, ${FAILED}/${TOTAL} failed"
echo "============================================"

if [ $FAILED -gt 0 ]; then
    echo -e "Failed tests:${FAILED_LIST}"
    echo ""
fi

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
