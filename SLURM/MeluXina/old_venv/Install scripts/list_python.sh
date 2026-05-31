#!/bin/bash -l
#SBATCH --job-name=list_python
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 00:30:00
#SBATCH --output %j.list_python.out
#SBATCH --error %j.list_python.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss

source my_python-env/bin/activate

echo "full list:\n"
python -m pip list

echo "user list:\n"
python -m pip list --user

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw