#!/bin/bash -l
#SBATCH --job-name=install_cobaya
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 128
#SBATCH --cpus-per-task 1
#SBATCH --time 00:15:00
#SBATCH --output install_cobaya.%j.out
#SBATCH --error install_cobaya.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI
module load OpenBLAS

#Check Python version
python -c  'import sys; print(sys.version)'

#Check OpenMPI version
mpicc --version

#Create a virtual environment
python3 -m venv my_python-env
source my_python-env/bin/activate

#Upgrade pip
python -m pip install pip --upgrade

#Install MPI for Python from source to ensure compatibility with OpenMPI
python -m pip install "mpi4py>=3" --upgrade --no-binary :all:

#Check mpi4py installation
srun -n 2 python -c "from mpi4py import MPI, __version__; print(__version__ if MPI.COMM_WORLD.Get_rank() else '')"

# Check OpenBLAS installation
python -c "from numpy import show_config; show_config()" | grep 'mkl\|openblas_info' -A 1

#Reinstall numpy and scipy to ensure they are linked against OpenBLAS
python -m pip install --force-reinstall numpy --upgrade

#Install cobaya and dependencies
python -m pip install cobaya --upgrade

#Check cobaya installation
python -c "import cobaya"

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw