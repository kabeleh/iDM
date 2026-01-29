#!/bin/bash -l
#SBATCH --job-name=run_cobaya_polychord_CV_PP_DESI_LCDM
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 8
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 16
#SBATCH --time 06:00:00
#SBATCH --output %j.run_cobaya_polychord_CV_PP_DESI_LCDM.out
#SBATCH --error %j.run_cobaya_polychord_CV_PP_DESI_LCDM.err
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


srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_polychord_CV_PP_DESI_LCDM.yml --resume > "${SLURM_JOB_ID}_PolyChord_CV_PP_DESI_LCDM.txt" &
srun -n 4 --exact --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_polychord_CV_PP_S_DESI_LCDM.yml --resume > "${SLURM_JOB_ID}_PolyChord_CV_PP_S_DESI_LCDM.txt" &
wait
#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
