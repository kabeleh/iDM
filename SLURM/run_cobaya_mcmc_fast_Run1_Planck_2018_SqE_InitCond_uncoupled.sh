#!/bin/bash -l
#SBATCH --job-name=run_cobaya_mcmc_fast_Run1_Planck_2018_SqE_InitCond_uncoupled
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 64
#SBATCH --time 06:00:00
#SBATCH --output run_cobaya_mcmc_fast_Run1_Planck_2018_SqE_InitCond_uncoupled.%j.out
#SBATCH --error run_cobaya_mcmc_fast_Run1_Planck_2018_SqE_InitCond_uncoupled.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS
#Activate Python virtual environment
source my_python-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Retry logic for exit code 143 (SIGTERM)
MAX_RETRIES=199
RETRY_COUNT=0
EXIT_CODE=143

while [ $EXIT_CODE -eq 143 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT of $MAX_RETRIES"
    srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_fast_Run1_Planck_2018_SqE_InitCond_uncoupled.yml --resume
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
