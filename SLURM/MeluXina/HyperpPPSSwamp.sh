#!/bin/bash -l
#SBATCH --job-name=HyperPPSwamp
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --ntasks-per-node 12
#SBATCH --cpus-per-task 20
#SBATCH --time 6:00:00
#SBATCH --output %j.HyperPPSwamp.out
#SBATCH --error %j.HyperPPSwamp.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

#Number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/hyperbolic_Planck_PP_DESI_InitCond_MCMC_Swamp_minimizer.yml --force



#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
