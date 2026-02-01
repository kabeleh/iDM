import os
import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load

# Set the correct packages path
packages_path = "/Users/klmba/Cosmology"

# --- 1. Define the input information (YAML as a string) ---
info_yaml = f"""
packages_path: {packages_path}

theory:
  classy:
    extra_args:
      gauge: newtonian
      nonlinear_min_k_max: 25
      N_ncdm: 1
      N_ur: 2.046
      sBBN file: sBBN_2017.dat
      non linear: hmcode
      hmcode_version: 2020

likelihood:
  muse3glike:
    class: muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse
    # package_install block is only needed for installation, not runtime
    components:
    - "\\u03D5\\u03D5" # Correct Python string representation for phi-phi (lensing)
    
    # Nuisance parameters for muse3glike are typically internal/inherited, 
    # but we include A_planck for completeness if it's required.
    params:
        A_planck: {{value: 1.0}}

params:
  # --- LCDM parameters (must be fixed for initialization) ---
  logA: {{value: 3.05}}
  n_s: {{value: 0.965}}
  H0: {{value: 67}}
  tau_reio: {{value: 0.051}}
  omega_b: {{value: 0.0224}}
  omega_cdm: {{value: 0.12}}

  # --- SPT (candl) nuisance parameters (must be fixed) ---
  Tcal: {{value: 1.0}}
  Ecal: {{value: 1.0}}

  # --- Global nuisance parameters (must be fixed) ---
  A_planck: {{value: 1.0}} 
  A_act: {{value: 1.0}}
  P_act: {{value: 1.0}}

  # Derived parameter placeholder value (only needed if they were sampled inputs)
  m_ncdm: {{value: 0.06}}
"""

# Load the YAML string into a Python dictionary
info = yaml_load(info_yaml)

# --- 2. Instantiate the model ---
like_name = "muse3glike"

try:
    model = get_model(info)
    print(f"[Success] Model initialized successfully.")
except Exception as e:
    print(f"[Failure] Error initializing model: {e}")
    print("Likely causes: 1. Missing data download. 2. Incorrect component path.")
    exit()

# Access the likelihood instance
like = model.likelihood[like_name]
# Best to add a break point here and inspect this object by hand in the debugger. There is usually a cov object or something similar that contains the information about how many data points are actually used.


# --- 4. Clean up ---
model.close()
