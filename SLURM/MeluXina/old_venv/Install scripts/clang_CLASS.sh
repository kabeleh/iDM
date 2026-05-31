#!/bin/bash -l
#SBATCH --job-name=compile_CLASS
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 256
#SBATCH --time 00:30:00
#SBATCH --output %j.compile_CLASS.out
#SBATCH --error %j.compile_CLASS.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
# module load GCC
# module load Python
# module load Cython
# module load OpenMPI/5.0.3-GCC-13.3.0
# module load OpenBLAS
module load Python foss AOCC
#Activate Python virtual environment
source my_foss-env/bin/activate

#Check GCC version
# gcc --version

#Check AOCC version
# clang --version

#Navigate to CLASS source code directory
cd $HOME/iDM/

#Compile C program with GCC (in parallel)
make clean; make class -j

## Test task execution
# ./class pgo_doubleexp_bao.ini
# ./class pgo_doubleexp_cmb_shooting_fails.ini
# ./class pgo_doubleexp_cmb.ini
# ./class pgo_hyperbolic_bao.ini
# ./class pgo_hyperbolic_cmb.ini
# ./class pgo_segfault.ini
# ./class test_segfault.ini

# llvm-profdata merge -output=pgo_profiles/merged.profdata pgo_profiles/*.profraw

# #Change GPO compile flags first.
# make clean; make -j

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw