#!/bin/bash -l
#SBATCH --job-name=PPPlanck
#SBATCH --account=project_465002956
#SBATCH --partition standard
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 16
#SBATCH --threads-per-core 2
#SBATCH --hint=multithread
#SBATCH --time 48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output %j.PPPlanck.out
#SBATCH --error %j.PPPlanck.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load CrayEnv LUMI cpeCray cray-libsci cray-mpich lumi-CrayPath

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Maximum number of automatic requeues (can be overridden at submit time).
MAX_REQUEUES=${MAX_REQUEUES:-10}

cobaya_done=0
handle_timelimit() {
	trap - USR1

	if [ "$cobaya_done" -eq 1 ]; then
		echo "[$(date)] Cobaya already finished successfully; skipping requeue."
		exit 0
	fi

	restart_count=${SLURM_RESTART_COUNT:-0}
	if [ "$restart_count" -ge "$MAX_REQUEUES" ]; then
		echo "[$(date)] Requeue limit reached (${restart_count}/${MAX_REQUEUES}); not requeueing."
		exit 0
	fi

	echo "[$(date)] Time limit approaching for job $SLURM_JOB_ID. Requeueing (${restart_count}/${MAX_REQUEUES})..."
	scontrol requeue "$SLURM_JOB_ID"
	exit 0
}

# USR1 is sent 180s before time limit (configured above).
trap handle_timelimit USR1

## Task execution
srun -n 16 --mpi=pmi2 --exclusive --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=threads singularity exec -B /project/project_465002956 -B /scratch/project_465002956 cobaya.sif cobaya-run /project/project_465002956/iDM/Cobaya/MCMC/hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC.yml --resume --allow-changes
srun_exit=$?

if [ "$srun_exit" -eq 0 ]; then
	cobaya_done=1
	trap - USR1
	echo "[$(date)] Cobaya run completed successfully; no requeue requested."
fi

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw

exit "$srun_exit"