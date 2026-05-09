#!/bin/bash -l
#SBATCH --job-name=benchCray
#SBATCH --account=project_465002956
#SBATCH --partition standard
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --time 01:30:00
#SBATCH --output %j.benchCray.out
#SBATCH --error %j.benchCray.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

#Navigate to CLASS source code directory
cd /project/project_465002956/iDM

## Test task execution
srun ./benchmark/run_benchmark.sh pgo

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw