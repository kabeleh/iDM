#!/usr/bin/env python3
"""Test script to reproduce Cobaya segfault."""
import sys

sys.path.insert(0, "build/lib.macosx-10.13-universal2-cpython-313")
from classy import Class

# Test 1: Using h directly (should work)
print("=" * 60)
print("TEST 1: Using 'h' directly (like iDM.ini)")
print("=" * 60)
cosmo1 = Class()
params1 = {
    "scf_parameters": "1e-07,75.0,1.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.001,0.1",
    "Omega_fld": 0.0,
    "Omega_scf": -0.7,
    "Omega_Lambda": 0.0,
    "A_s": 2.110156399853534e-09,
    "n_s": 0.9679902618072788,
    "h": 0.6781,  # Use h directly
    "omega_b": 0.022228317084779656,
    "omega_cdm": 0.1207562665618809,
    "m_ncdm": 0.06,
    "tau_reio": 0.0605413523700832,
    "N_ncdm": 1,
    "N_ur": 2.0328,
    "tol_initial_Omega_r": 0.01,
    "scf_potential": "DoubleExp",
    "scf_tuning_index": 0,
    "gauge": "newtonian",
    "attractor_ic_scf": False,
    "output": "lCl tCl mPk pCl",
    "non_linear": "halofit",
    "l_max_scalars": 2500.0,
    "lensing": "yes",
    "P_k_max_1/Mpc": 1,
}
print("Setting parameters...")
cosmo1.set(params1)
print("Computing...")
try:
    cosmo1.compute()
    print("Done! H0 =", cosmo1.h() * 100)
except Exception as e:
    print("Error:", e)
cosmo1.struct_cleanup()

# Test 2: Using theta_s_100 (segfaults)
print()
print("=" * 60)
print("TEST 2: Using 'theta_s_100' (like Cobaya)")
print("=" * 60)
cosmo2 = Class()
params2 = {
    "scf_parameters": "1e-07,75.0,1.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.001,0.1",
    "Omega_fld": 0.0,
    "Omega_scf": -0.7,
    "Omega_Lambda": 0.0,
    "A_s": 2.110156399853534e-09,
    "n_s": 0.9679902618072788,
    "theta_s_100": 1.0411566776867598,  # This triggers nested shooting
    "omega_b": 0.022228317084779656,
    "omega_cdm": 0.1207562665618809,
    "m_ncdm": 0.06,
    "tau_reio": 0.0605413523700832,
    "N_ncdm": 1,
    "N_ur": 2.0328,
    "tol_initial_Omega_r": 0.01,
    "scf_potential": "DoubleExp",
    "scf_tuning_index": 0,
    "gauge": "newtonian",
    "attractor_ic_scf": False,
    "output": "lCl tCl mPk pCl",
    "non_linear": "halofit",
    "l_max_scalars": 2500.0,
    "lensing": "yes",
    "P_k_max_1/Mpc": 1,
}
print("Setting parameters...")
cosmo2.set(params2)
print("Computing...")
try:
    cosmo2.compute()
    print("Done! H0 =", cosmo2.h() * 100)
except Exception as e:
    print("Error:", e)
