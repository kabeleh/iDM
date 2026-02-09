#!/bin/bash -l
#SBATCH --job-name=compile_CLASS
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --time 00:30:00
#SBATCH --output %j.compile_CLASS.out
#SBATCH --error %j.compile_CLASS.err
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

#Check GCC version
gcc --version

#Navigate to CLASS source code directory
cd $HOME/iDM/

#Compile C program with GCC (in parallel)
make clean; make -j

## Test task execution
srun ./class iDM.ini

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw