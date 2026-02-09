#!/bin/bash -l
#SBATCH --job-name=install_CosmoVerse
#SBATCH --account p201176
#SBATCH --partition cpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --time 00:15:00
#SBATCH --output install_CosmoVerse.%j.out
#SBATCH --error install_CosmoVerse.%j.err
#SBATCH --mail-user kay.lehnert.2023@mumail.ie
#SBATCH --mail-type END,FAIL

## Load software environment
module load Python foss
# module load GCC
# module load libpciaccess
# module load Python
# module load Cython
# module load OpenMPI/5.0.3-GCC-13.3.0
# module load OpenBLAS


#Activate Python virtual environment
source my_python-env/bin/activate



# python -m pip install candl_like
# cd /home/users/u103677/cobaya_cosmo_packages/data/
# git clone https://github.com/SouthPoleTelescope/spt_candl_data.git
# cd spt_candl_data
# pip install .
# python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/CV_PP_DESI_LCDM.yml --p /home/users/u103677/cobaya_cosmo_packages/
# python -m cobaya /home/users/u103677/iDM/Cobaya/MCMC/CV_PP_DESI_LCDM.yml --test
# python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/CV_PP_S_DESI_LCDM.yml --p /home/users/u103677/cobaya_cosmo_packages/
# python -m cobaya /home/users/u103677/iDM/Cobaya/MCMC/CV_PP_S_DESI_LCDM.yml --test
python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_LCDM.yml --p /home/users/u103677/cobaya_cosmo_packages/
python -m cobaya /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_LCDM.yml --test --force

# python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_PP_DESI_LCDM.yml --p /home/users/u103677/cobaya_cosmo_packages/
# python -m cobaya /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_PP_DESI_LCDM.yml --test
python -m cobaya install /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_PP_S_DESI_LCDM.yml --p /home/users/u103677/cobaya_cosmo_packages/
python -m cobaya /home/users/u103677/iDM/Cobaya/MCMC/CV_CMB_SPA_PP_S_DESI_LCDM.yml --test

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw