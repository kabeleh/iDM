#!/bin/bash -l
#SBATCH --job-name=LCDM_SPA_PP_DESI_S8_minimize
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 64
#SBATCH --time 06:00:00
#SBATCH --output %j.LCDM_SPA_PP_DESI_S8_minimize.out
#SBATCH --error %j.LCDM_SPA_PP_DESI_S8_minimize.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_foss-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_PP_DESI_LCDM_S8_minimize.yml


#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
