#!/bin/bash -l
#SBATCH --job-name=gccComparison
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos short
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 256
#SBATCH --time 00:10:00
#SBATCH --output %j.gccComparison.out
#SBATCH --error %j.gccComparison.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
#Activate Python virtual environment
source my_foss-env/bin/activate

# srun /home/users/u103677/iDM/benchmark/run_benchmark.sh gccNEW
python3 /home/users/u103677/iDM/benchmark/compare_outputs.py gccNEW gccOLD


#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
