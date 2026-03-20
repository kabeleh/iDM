#!/bin/bash -l
#SBATCH --job-name=HyperExp
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --ntasks 62
#SBATCH --ntasks-per-node 62
#SBATCH --cpus-per-task 4
#SBATCH --time 48:00:00
#SBATCH --output %j.HyperExp.out
#SBATCH --error %j.HyperExp.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

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
            cobaya-run "$config_file" --resume >> "$output_file" 2>&1
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

# Function to run a multi-stage pipeline sequentially.
# Each stage uses run_with_retry; pipeline stops on first non-zero exit code.
run_pipeline_with_retry() {
    local pipeline_log_file="$1"
    local ntasks="$2"
    shift 2

    local total_stages=$#
    local stage_index=0
    local stage_config
    local stage_output_file
    local stage_name

    echo "Pipeline started with $total_stages stages" >> "$pipeline_log_file"
    for stage_config in "$@"; do
        stage_index=$((stage_index + 1))
        stage_name=$(basename "$stage_config" .yml)
        stage_output_file="${SLURM_JOB_ID}_${stage_name}.txt"

        echo "Pipeline stage ${stage_index}/${total_stages}: starting $stage_config" >> "$pipeline_log_file"
        echo "Pipeline stage ${stage_index}/${total_stages}: output file $stage_output_file" >> "$pipeline_log_file"

        run_with_retry "$stage_config" "$stage_output_file" "$ntasks"
        local stage_exit=$?
        if [ $stage_exit -ne 0 ]; then
            echo "Pipeline failed at stage: $stage_config (exit code: $stage_exit)" >> "$pipeline_log_file"
            return $stage_exit
        fi
        echo "Pipeline stage ${stage_index}/${total_stages}: completed successfully $stage_config" >> "$pipeline_log_file"
    done

    echo "Pipeline completed successfully" >> "$pipeline_log_file"
    return 0
}

# Run all tasks in parallel, each with its own retry logic
run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/DoubleExp_Planck_InitCond_MCMC.yml" \
    "${SLURM_JOB_ID}_DoubleExp.txt" 10 &
PID1=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/exponential_Planck_InitCond_MCMC.yml" \
    "${SLURM_JOB_ID}_exponential.txt" 10 &
PID2=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_Planck_tracking_MCMC.yml" \
    "${SLURM_JOB_ID}_hyperbolicTracking.txt" 10 &
PID3=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_hyperbolic_PP.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_D_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_D_InitCond_MCMC_minimizer.yml" &
PID4=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_hyperbolic_PPS.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_S_D_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_S_D_InitCond_MCMC_minimizer.yml" &
PID5=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_hyperbolic_SPA.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_SPA_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_SPA_InitCond_MCMC_minimizer.yml" &
PID6=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_pNG_PP.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_PP_D_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_PP_D_InitCond_MCMC_minimizer.yml" &
PID7=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_pNG_PPS.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_PP_S_D_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_PP_S_D_InitCond_MCMC_minimizer.yml" &
PID8=$!

run_pipeline_with_retry "${SLURM_JOB_ID}_pNG_SPA.txt" 4 \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_SPA_InitCond_MCMC.yml" \
    "/home/users/u103677/iDM/Cobaya/MCMC/pNG_SPA_InitCond_MCMC_minimizer.yml" &
PID9=$!



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
wait $PID7
EXIT7=$?
wait $PID8
EXIT8=$?
wait $PID9=$?
EXIT9=$?

echo "============================================"
echo "All tasks completed"
echo "DoubleExp exit code: $EXIT1"
echo "Exponential exit code: $EXIT2"
echo "Hyperbolic Tracking exit code: $EXIT3"
echo "Hyperbolic PP DESI pipeline exit code: $EXIT4"
echo "Hyperbolic PPS DESI pipeline exit code: $EXIT5"
echo "Hyperbolic SPA pipeline exit code: $EXIT6"
echo "pNG PP pipeline exit code: $EXIT7"
echo "pNG PPS pipeline exit code: $EXIT8"
echo "pNG SPA pipeline exit code: $EXIT9"
echo "============================================"

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
