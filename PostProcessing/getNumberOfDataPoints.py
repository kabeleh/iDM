from typing import Any, Callable, cast

from cobaya.model import get_model  # type: ignore[import-untyped]
from cobaya.yaml import yaml_load  # type: ignore[import-untyped]

get_model = cast(Callable[[Any], Any], get_model)
yaml_load = cast(Callable[[str], dict[str, Any]], yaml_load)

# Set the correct packages path
# packages_path = "/Users/klmba/Cosmology"
packages_path = "/home/kl/kDrive/Sci/PhD/Research/HDM/cobaya_klarch/"

# --- 1. Define the input information (YAML as a string) ---
info_yaml = f"""
packages_path: {packages_path}

theory:
  classy:
    path: /home/kl/kDrive/Sci/PhD/Research/HDM/class_public
    extra_args:
      gauge: newtonian
      N_ncdm: 1
      N_ur: 2.046
      sBBN file: sBBN_2017.dat
      non linear: halofit
likelihood:
  # planck_2018_lowl.TT:
  # planck_2018_lowl.EE:
  # planck_2018_highl_plik.TTTEEE_lite_native:
  planck_2018_lensing.native:

params:
  logA: {{value: 3.05}}
  n_s: {{value: 0.965}}
  H0: {{value: 67}}
  tau_reio: {{value: 0.051}}
  omega_b: {{value: 0.0224}}
  omega_cdm: {{value: 0.12}}
  m_ncdm: {{value: 0.06}}
  A_planck: {{value: 1.0}}
"""

# Load the YAML string into a Python dictionary
info: dict[str, Any] = yaml_load(info_yaml)

# --- 2. Instantiate the model ---
like_name: str = "planck_2018_lensing.native"

try:
    model = get_model(info)
    print(f"[Success] Model initialized successfully.")
except Exception as e:
    print(f"[Failure] Error initializing model: {e}")
    print("Likely causes: 1. Missing data download. 2. Incorrect component path.")
    exit()

# Access the likelihood instance
like: Any = model.likelihood[like_name]
# Best to add a break point here and inspect this object by hand in the debugger. There is usually a cov object or something similar that contains the information about how many data points are actually used.


# --- 4. Clean up ---
model.close()
