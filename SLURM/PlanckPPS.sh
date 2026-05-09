#!/bin/bash -l
#SBATCH --job-name=PlanckPPS
#SBATCH --account=project_465002956
#SBATCH --partition standard
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 16
#SBATCH --threads-per-core 2
#SBATCH --hint=multithread
#SBATCH --time 48:00:00
#SBATCH --output %j.PlanckPPS.out
#SBATCH --error %j.PlanckPPS.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

## Task execution
srun -n 16 --mpi=pmi2 --exclusive --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=threads singularity exec -B /project/project_465002956 -B /scratch/project_465002956 cobaya.sif cobaya-run /project/project_465002956/iDM/Cobaya/MCMC/hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC.yml --resume --allow-changes

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw