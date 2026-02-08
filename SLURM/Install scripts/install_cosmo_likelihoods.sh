#!/bin/bash -l
#SBATCH --job-name=install_cosmo_likelihoods
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --time 00:15:00
#SBATCH --output install_cosmo_likelihoods.%j.out
#SBATCH --error install_cosmo_likelihoods.%j.err
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

#Install cosmo likelihoods for cobaya
srun cobaya-install cosmo -p $HOME/cobaya_cosmo_packages/

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw