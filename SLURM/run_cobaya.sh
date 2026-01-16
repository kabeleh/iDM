#!/bin/bash -l
#SBATCH --job-name=cobaya_run
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 8
#SBATCH --time 06:00:00
#SBATCH --output cobaya_run.%j.out
#SBATCH --error cobaya_run.%j.err
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
srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/cobaya_mcmc_fast_Run1_Planck_2018_DoubleExp_tracking_uncoupled.yml --debug --force

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw