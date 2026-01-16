#!/bin/bash -l
#SBATCH --job-name=install_numba
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --time 00:05:00
#SBATCH --output install_numba.%j.out
#SBATCH --error install_numba.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS
module load libpciaccess

#Activate Python virtual environment
source my_python-env/bin/activate

#Install numba
python -m pip install numba

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw