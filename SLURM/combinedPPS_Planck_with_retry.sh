#!/bin/bash -l
#SBATCH --job-name=combined
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --ntasks-per-node 24
#SBATCH --cpus-per-task 6
#SBATCH --time 48:00:00
#SBATCH --output %j.combined.out
#SBATCH --error %j.combined.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_foss-env/bin/activate

#Number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Function to run a task with retry logic for segfaults and other errors
run_with_retry() {
    local config_file="$1"
    local output_file="$2"
    local ntasks="$3"
    local MAX_RETRIES=999
    local RETRY_COUNT=0
    local EXIT_CODE=1  # Initialize with error code to enter loop
    
    echo "Starting task: $config_file" >> "$output_file"
    
    while [ $EXIT_CODE -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -gt 1 ]; then
            echo "============================================" >> "$output_file"
            echo "Task $config_file: Attempt $RETRY_COUNT of $MAX_RETRIES" >> "$output_file"
            echo "Previous exit code: $EXIT_CODE" >> "$output_file"
            echo "============================================" >> "$output_file"
        fi
        
        srun -n "$ntasks" --exact --cpus-per-task=$SLURM_CPUS_PER_TASK \
            cobaya-run "$config_file" --resume --allow-changes >> "$output_file" 2>&1
        EXIT_CODE=$?
        
        # Exit successfully if code is 0
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Task $config_file completed successfully" >> "$output_file"
            break
        fi
        
        # Check for specific error codes that should trigger retry
        # 139 = SIGSEGV (segmentation fault)
        # 143 = SIGTERM (terminated)
        # 134 = SIGABRT (abort)
        if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 134 ]; then
            echo "Task $config_file: Received exit code $EXIT_CODE, will retry..." >> "$output_file"
        else
            echo "Task $config_file: Exit code $EXIT_CODE - not retrying (non-recoverable error)" >> "$output_file"
            break
        fi
    done
    
    if [ $EXIT_CODE -ne 0 ] && [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Task $config_file: Max retries ($MAX_RETRIES) reached with exit code $EXIT_CODE" >> "$output_file"
    fi
    
    return $EXIT_CODE
}

# Run all tasks in parallel, each with its own retry logic
run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_DESI_DoubleExp_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_DoubleExp_PP.txt" 4 &
PID1=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_DESI_hyperbolic_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_hyperbolic_PP.txt" 4 &
PID2=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_S_DESI_DoubleExp_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_DoubleExp_PPS.txt" 4 &
PID3=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_S_DESI_hyperbolic_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_hyperbolic_PPS.txt" 4 &
PID4=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CMB_DoubleExp_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_DoubleExp_CMB.txt" 4 &
PID5=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CMB_hyperbolic_InitCond_uncoupled.yml" \
    "${SLURM_JOB_ID}_hyperbolic_CMB.txt" 4 &
PID6=$!

# run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CMB_LCDM.yml" \
#     "${SLURM_JOB_ID}_LCDM_CMB.txt" 4 &
# PID7=$!

# Wait for all background tasks to complete
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?
wait $PID3
EXIT3=$?
wait $PID4
EXIT4=$?
wait $PID5
EXIT5=$?
wait $PID6
EXIT6=$?
# wait $PID7
# EXIT7=$?

echo "============================================"
echo "All tasks completed"
echo "DoubleExp_PP exit code: $EXIT1"
echo "hyperbolic_PP exit code: $EXIT2"
echo "DoubleExp_PPS exit code: $EXIT3"
echo "hyperbolic_PPS exit code: $EXIT4"
echo "DoubleExp_CMB exit code: $EXIT5"
echo "hyperbolic_CMB exit code: $EXIT6"
# echo "LCDM_CMB exit code: $EXIT7"
echo "============================================"

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
