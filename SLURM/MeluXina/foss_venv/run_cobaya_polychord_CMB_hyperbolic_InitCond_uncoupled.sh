#!/bin/bash -l
#SBATCH --job-name=polychord_CMB_hyperbolic_InitCond_uncoupled
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 8
#SBATCH --time 48:00:00
#SBATCH --output %j.polychord_CMB_hyperbolic_InitCond_uncoupled.out
#SBATCH --error %j.polychord_CMB_hyperbolic_InitCond_uncoupled.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_foss-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# srun -n 32 --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_polychord_CMB_hyperbolic_InitCond_uncoupled.yml

# Retry logic for exit code 143 (SIGTERM)
MAX_RETRIES=99
RETRY_COUNT=0
EXIT_CODE=143

while [ $EXIT_CODE -eq 143 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT of $MAX_RETRIES"
    srun -n 32 --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_polychord_CMB_hyperbolic_InitCond_uncoupled.yml --resume
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    if [ $EXIT_CODE -eq 143 ]; then
        echo "Received exit code 143, will retry..."
    fi
done

if [ $EXIT_CODE -eq 143 ]; then
    echo "Max retries ($MAX_RETRIES) reached with exit code 143"
fi

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
