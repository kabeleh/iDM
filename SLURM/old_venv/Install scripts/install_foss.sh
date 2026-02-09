#!/bin/bash -l
#SBATCH --job-name=install_cobaya
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 1
#SBATCH --time 00:35:00
#SBATCH --output install_cobaya.%j.out
#SBATCH --error install_cobaya.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
# module load GCC
module load Python foss
# module load Cython
# module load OpenMPI/5.0.3-GCC-13.3.0
# module load OpenBLAS

#Check MPI compiler
which mpicc

#Check Python version
python -c  'import sys; print(sys.version)'

#Check OpenMPI version
mpicc --version

#Create a virtual environment
python3 -m venv my_foss-env
source my_foss-env/bin/activate

#Upgrade pip
python -m pip install pip --upgrade

#Install MPI for Python from source to ensure compatibility with OpenMPI
python -m pip install "mpi4py>=3" --upgrade --no-binary :all:

#Check mpi4py installation
srun -n 2 python -c "from mpi4py import MPI, __version__; print(__version__ if MPI.COMM_WORLD.Get_rank() else '')"

# Check OpenBLAS installation
python -c "from numpy import show_config; show_config()" | grep 'mkl\|openblas_info' -A 1

#Reinstall numpy and scipy to ensure they are linked against OpenBLAS
# python -m pip install --force-reinstall numpy --upgrade
# python -m pip install "numpy>=1.24,<2.0"

#Install python packages for cobaya and cosmological likelihoods
cd /home/users/u103677/cobaya_packages_2026/data
git clone https://github.com/SouthPoleTelescope/spt_candl_data.git
cd spt_candl_data
pip install .

cd /home/users/u103677/cobaya_packages_2026/data
git clone https://github.com/Lbalkenhol/candl_data.git
cd candl_data
pip install .

cd $HOME
python -m pip install numba iminuit Cython setuptools wheel cobaya candl-like "numpy>=1.22,<2.0" sacc llvmlite muse3glike spt_candl_data --upgrade

#Check cobaya installation
python -c "import cobaya"

#Install likelihoods
python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_CMB_SPA_PP_S_DESI_hyperbolic_InitCond_uncoupled.yml --p /home/users/u103677/cobaya_packages_2026 --upgrade

#Check GCC version
gcc --version

#Navigate to CLASS source code directory
cd $HOME/iDM/
#Compile C program with GCC (in parallel)
make clean && make class -j
./class pgo_hyperbolic_bao.ini
./class pgo_doubleexp_bao.ini
./class pgo_hyperbolic_cmb.ini
./class pgo_doubleexp_cmb.ini
./class pgo_doubleexp_cmb_shooting_fails.ini
#then change makefile to use PGO results and recompile
make clean; make -j
## Test task execution
srun ./class iDM.ini

#Now test cobaya with the new CLASS executable
cd /home/users/u103677

srun cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_PP_S_DESI_DoubleExp_InitCond_uncoupled.yml --test --debug --force
srun cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_mcmc_CV_CMB_SPA_PP_S_DESI_hyperbolic_InitCond_uncoupled.yml --test --debug --force



#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw