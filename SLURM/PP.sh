#!/bin/bash -l
#SBATCH --job-name=PP
#SBATCH --account=project_465002956
#SBATCH --partition small
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu=1750
#SBATCH --time 72:00:00
#SBATCH --output %j.PP.out
#SBATCH --error %j.PP.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## Task execution
srun -n 4  --mpi=pmi2 --exclusive --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores singularity exec -B /project/project_465002956 -B /scratch/project_465002956 cobaya.sif cobaya-run /project/project_465002956/iDM/Cobaya/MCMC/hyperbolic_PP_D_InitCond_MCMC.yml --resume --allow-changes

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw