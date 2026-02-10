#!/bin/bash -l
#SBATCH --job-name=install_PolyChord
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 256
#SBATCH --time 00:30:00
#SBATCH --output install_PolyChord.%j.out
#SBATCH --error install_PolyChord.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
# module load GCC
module load Python foss
# module load Cython
# module load OpenMPI/5.0.3-GCC-13.3.0
# module load OpenBLAS


#Activate a virtual environment
source my_foss-env/bin/activate

cd /home/users/u103677/cobaya_packages_2026/code/
git clone https://github.com/PolyChord/PolyChordLite.git
cd /home/users/u103677/cobaya_packages_2026/code/PolyChordLite
make MPI=1 COMPILER_TYPE=gnu
python setup.py build
pip install .


srun cobaya-run /home/users/u103677/iDM/Cobaya/MCMC/cobaya_polychord_CMB_hyperbolic_InitCond_uncoupled.yml --test --debug --force



#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw