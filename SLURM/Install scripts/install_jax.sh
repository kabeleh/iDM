#!/bin/bash -l
#SBATCH --job-name=install_jax
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --time 00:30:00
#SBATCH --output install_jax.%j.out
#SBATCH --error install_jax.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss

source my_python-env/bin/activate

python -m pip install "numpy>=1.24,<2.0"
python -m pip install jax==0.7.1 jaxlib==0.7.1
python -m pip install "numpy>=1.24,<2.0"

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw