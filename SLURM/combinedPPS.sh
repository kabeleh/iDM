#!/bin/bash -l
#SBATCH --job-name=combined_PPS
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 8
#SBATCH --time 48:00:00
#SBATCH --output %j.combined_PPS.out
#SBATCH --error %j.combined_PPS.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_foss-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_DESI_DoubleExp_InitCond_uncoupled.yml --resume > "${SLURM_JOB_ID}_DoubleExp_PP.txt" &
srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_DESI_hyperbolic_InitCond_uncoupled.yml --resume > "${SLURM_JOB_ID}_hyperbolic_PP.txt" &
srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_S_DESI_DoubleExp_InitCond_uncoupled.yml --resume > "${SLURM_JOB_ID}_DoubleExp_PPS.txt" &
srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_S_DESI_hyperbolic_InitCond_uncoupled.yml --resume > "${SLURM_JOB_ID}_hyperbolic_PPS.txt" &
wait
#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
