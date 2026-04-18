#!/bin/bash -l
#SBATCH --job-name=HyperMin
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 8
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 32
#SBATCH --time 3:00:00
#SBATCH --output %j.HyperMin.out
#SBATCH --error %j.HyperMin.err
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
    local MAX_RETRIES=1
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
            cobaya-run "$config_file" --force >> "$output_file" 2>&1
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

# 2*16 chains with 8 CPUS

# run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC_minimizer.yml" \
#     "${SLURM_JOB_ID}_hyperbolic_PPS_Swamp.txt" 16 &
# PID1=$!

# run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC_minimizer.yml" \
#     "${SLURM_JOB_ID}_hyperbolic_PP_Swamp.txt" 16 &
# PID2=$!

# 3*4 chains with 21 CPUs

# run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_D_InitCond_MCMC_minimizer.yml" \
#     "${SLURM_JOB_ID}_hyperbolic_PP.txt" 4 &
# PID3=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_PP_S_D_InitCond_MCMC_minimizer.yml" \
    "${SLURM_JOB_ID}_hyperbolic_PP_S_D_InitCond.txt" 4 &
PID4=$!

run_with_retry "/home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_SPA_InitCond_MCMC_minimizer.yml" \
    "${SLURM_JOB_ID}_hyperbolic_SPA_InitCond.txt" 4 &
PID5=$!




# Wait for all background tasks to complete
# wait $PID1
# EXIT1=$?
# wait $PID2
# EXIT2=$?
# wait $PID3
# EXIT3=$?
wait $PID4
EXIT4=$?
wait $PID5
EXIT5=$?


echo "============================================"
echo "All tasks completed"
# echo "Hyperbolic Planck PP Swamp exit code: $EXIT1"
# echo "Hyperbolic Planck PPS Swamp exit code: $EXIT2"
# echo "Hyperbolic PP exit code: $EXIT3"
echo "Hyperbolic PPS pipeline exit code: $EXIT4"
echo "Hyperbolic SPA pipeline exit code: $EXIT5"
echo "============================================"

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
