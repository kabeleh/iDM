#!/bin/bash -l
#SBATCH --job-name=uninstall_jax
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --time 00:30:00
#SBATCH --output uninstall_jax.%j.out
#SBATCH --error uninstall_jax.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss

source my_python-env/bin/activate

python -m pip uninstall -y jax jaxlib

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw