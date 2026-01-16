# This routine creates the configuration files for the Cobaya runs to assess the iDM model.
# Ouput: Cobaya_<sampler>_<likelihoods>_<potential>_<attractor>_<coupling>.yml

from ruamel.yaml import YAML
import re

# Specify the parameters
sampler = "mcmc_fast"
likelihood = "Run1_Planck_2018"
potential = "DoubleExp"
attractor = "yes"
coupling = "uncoupled"

yaml = YAML()
# yaml.version = (1, 2)  # Specify YAML version


def create_cobaya_yaml(
    sampler: str,
    likelihood: str,
    potential: str,
    attractor: str = "no",
    coupling: str = "uncoupled",
) -> dict:
    """
    Create a Cobaya YAML configuration dictionary for iDM model runs.

    Parameters:
    - sampler (str): Sampling method. Options: 'polychord', 'mcmc', 'mcmc_fast'.
    - likelihood (str): Likelihood combination. Options: 'Run1_Planck_2018', 'Run2_PP_SH0ES_DESIDR2', 'Run3_Planck_PP_SH0ES_DESIDR2'.
    - potential (str): Scalar field potential. Options: 'power-law', 'cosine', 'hyperbolic', 'pNG', 'iPL', 'SqE', 'exponential', 'Bean', 'DoubleExp'.
      Note: 'power-law', 'cosine', 'pNG', 'iPL', 'SqE' do not support attractor initial conditions.
    - attractor (str): Initial condition type. Options: 'yes', 'Yes', 'YES' (tracking), 'no', 'No', 'NO' (phi_ini).
    - coupling (str): Coupling type. Options: 'uncoupled', 'coupled'.

    Returns:
    - dict: Cobaya configuration dictionary.
    """

    # Validate inputs
    if sampler not in ["polychord", "mcmc", "mcmc_fast"]:
        raise ValueError(
            f"Unknown sampler '{sampler}'. Must be one of: polychord, mcmc, mcmc_fast"
        )
    if likelihood not in [
        "Run1_Planck_2018",
        "Run2_PP_SH0ES_DESIDR2",
        "Run3_Planck_PP_SH0ES_DESIDR2",
    ]:
        raise ValueError(
            f"Unknown likelihood '{likelihood}'. Must be one of: Run1_Planck_2018, Run2_PP_SH0ES_DESIDR2, Run3_Planck_PP_SH0ES_DESIDR2"
        )
    if attractor not in ["yes", "Yes", "YES", "no", "No", "NO"]:
        raise ValueError(
            f"attractor must be 'yes', 'Yes', 'YES', 'no', 'No', or 'NO', got '{attractor}'"
        )
    if coupling not in ["uncoupled", "coupled"]:
        raise ValueError(f"coupling must be 'uncoupled' or 'coupled', got '{coupling}'")
    if attractor in ("yes", "Yes", "YES") and potential in (
        "power-law",
        "cosine",
        "pNG",
        "iPL",
        "SqE",
    ):
        raise ValueError(
            f"Attractor initial conditions are not implemented for potential '{potential}'"
        )

    # Define samplers

    polychord = {
        "sampler": {"polychord": {}},
    }

    mcmc = {
        "sampler": {
            "mcmc": {
                "Rminus1_cl_stop": 0.2,
                "Rminus1_stop": 0.01,
                "covmat": "auto",
                "drag": True,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "learn_proposal": True,
                "measure_speeds": True,
            }
        },
    }

    mcmc_fast = {
        "sampler": {
            "mcmc": {
                "Rminus1_cl_stop": 0.3,
                "Rminus1_stop": 0.05,
                "covmat": "auto",
                "drag": True,
                "oversample_power": 0.4,
                "learn_proposal": True,
                "measure_speeds": True,
            }
        },
    }

    SAMPLERS = {"polychord": polychord, "mcmc": mcmc, "mcmc_fast": mcmc_fast}

    # Define Likelihoods

    Run1_Planck_2018 = {
        "likelihood": {
            "planck_2018_lowl.EE": None,
            "planck_2018_lowl.TT": None,
            "planck_NPIPE_highl_CamSpec.TTTEEE": None,
            "planckpr4lensing": {
                "package_install": {
                    "github_repository": "carronj/planck_PR4_lensing",
                    "min_version": "1.0.2",
                }
            },
        },
    }

    Run2_PP_SH0ES_DESIDR2 = {
        "likelihood": {
            "H0.riess2020Mb": None,
            "bao.desi_dr2": None,
            "sn.pantheon": {"use_abs_mag": True},
            "sn.pantheonplus": None,
        },
    }

    Run3_Planck_PP_SH0ES_DESIDR2 = {
        "likelihood": {
            "H0.riess2020Mb": None,
            "bao.desi_dr2": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_lowl.TT": None,
            "planck_NPIPE_highl_CamSpec.TTTEEE": None,
            "planckpr4lensing": {
                "package_install": {
                    "github_repository": "carronj/planck_PR4_lensing",
                    "min_version": "1.0.2",
                }
            },
            "sn.pantheon": {"use_abs_mag": True},
            "sn.pantheonplus": None,
        },
    }

    # # Request sigma_R(z) for R=12 Mpc at z=0 as a dummy likelihood to compute derived parameters
    # requires = {
    #     "likelihood": {
    #         "one": {
    #             "requires": {
    #                 "sigma_R": {
    #                     "z": [0.0],
    #                     "R": [12.0],
    #                     "k_max": 5.0,  # Ensure k_max is large enough for the integral
    #                 }
    #             }
    #         }
    #     }
    # }

    LIKELIHOODS = {
        "Run1_Planck_2018": Run1_Planck_2018,
        "Run2_PP_SH0ES_DESIDR2": Run2_PP_SH0ES_DESIDR2,
        "Run3_Planck_PP_SH0ES_DESIDR2": Run3_Planck_PP_SH0ES_DESIDR2,
    }

    # Define cosmological Parameters

    # Parameters
    parameters_base = {
        "A_s": {"latex": "A_\\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"},
        "H0": {
            "latex": "H_0",
            "prior": {"max": 100, "min": 20},
            "proposal": 2,
            "ref": {"dist": "norm", "loc": 67, "scale": 2},
        },
        "Omega_m": {"latex": "\\Omega_\\mathrm{m}"},
        "YHe": {"latex": "Y_\\mathrm{P}"},
        "logA": {
            "drop": True,
            "latex": "\\log(10^{10} A_\\mathrm{s})",
            "prior": {"max": 3.91, "min": 1.61},
            "proposal": 0.001,
            "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
        },
        "m_ncdm": {"renames": "mnu", "value": 0.06},
        "n_s": {
            "latex": "n_\\mathrm{s}",
            "prior": {"max": 1.2, "min": 0.8},
            "proposal": 0.002,
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
        },
        "omega_b": {
            "latex": "\\Omega_\\mathrm{b} h^2",
            "prior": {"max": 0.1, "min": 0.005},
            "proposal": 0.0001,
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
        },
        "omega_cdm": {
            "latex": "\\Omega_\\mathrm{c} h^2",
            "prior": {"max": 0.99, "min": 0.001},
            "proposal": 0.0005,
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
        },
        "omegamh2": {
            "derived": "lambda Omega_m, H0: Omega_m*(H0/100)**2",
            "latex": "\\Omega_\\mathrm{m} h^2",
        },
        "tau_reio": {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"max": 0.8, "min": 0.01},
            "proposal": 0.003,
            "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
        },
        "z_reio": {"latex": "z_\\mathrm{re}"},
    }

    parameters_planck = {
        "A": {"derived": "lambda A_s: 1e9*A_s", "latex": "10^9 A_\\mathrm{s}"},
        "age": {"latex": "{\\rm{Age}}/\\mathrm{Gyr}"},
        "clamp": {
            "derived": "lambda A_s, tau_reio: 1e9*A_s*np.exp(-2*tau_reio)",
            "latex": "10^9 A_\\mathrm{s} e^{-2\\tau}",
        },
        "rs_drag": {"latex": "r_\\mathrm{drag}"},
        "s8h5": {
            "derived": "lambda sigma8, H0: (sigma8*(H0*1e-2)**(-0.5) if (H0>0) else 0)",
            "latex": "\\sigma_8/h^{0.5}",
        },
        "s8omegamp25": {
            "derived": "lambda sigma8, Omega_m: (sigma8*Omega_m**0.25 if (Omega_m>0) else 0)",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.25}",
        },
        "s8omegamp5": {
            "derived": "lambda sigma8, Omega_m: (sigma8*Omega_m**0.5 if (Omega_m>0) else 0)",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.5}",
        },
        "sigma8": {"latex": "\\sigma_8"},
        "theta_s_100": {
            "latex": "100\\theta_\\mathrm{s}",
            "prior": {"max": 10, "min": 0.5},
            "proposal": 0.0002,
            "ref": {"dist": "norm", "loc": 1.0416, "scale": 0.0004},
        },
        "H0": {"latex": "H_0"},  # Override to remove prior/proposal
        "Omega_m": {"latex": "\\Omega_\\mathrm{m}"},  # Ensure latex only
    }

    # # Custom Derived Parameters for sigma12 and S12
    # parameters_S12 = {
    #     # Lambda function to extract sigma_R at z=0, R=12 Mpc.
    #     # The provider.get_sigma_R() returns (z, R, sigma_R_grid).
    #     # We access the grid [2] at index [0, 0]
    #     "sigma12": {
    #         "derived": "lambda _self: _self.provider.get_sigma_R()[2][0, 0]",
    #         "latex": "\\sigma_{12}",
    #     },
    #     "S12": {
    #         "derived": "lambda sigma12, omegamh2: sigma12 * ((omegamh2/0.14)**0.4 if (omegamh2>0) else 0)",
    #         "latex": "S_{12}",
    #     },
    # }

    # Scalar Field Parameters for iDM model
    cdm_c = {"prior": {"min": -3, "max": 3}}

    # power-law:    V(phi) = c_1^(4-c_2) * phi^(c_2) + c_3
    # cosine:       V(phi) = c_1 * cos(phi*c_2)
    # hyperbolic:   V(phi) = c_1 * [1-tanh(c_2*phi)]
    # pNG:          V(phi) = c_1^4 * [1 + cos(phi/c_2)]
    # iPL:          V(phi) = c_1^(4+c_2) * phi^(-c_2)
    # exponential:  V(phi) = c_1 * exp(-c_2*phi)
    # SqE:          V(phi) = c_1^(c_2+4) * phi^(-c_2) * exp(c_1*phi^2)
    # Bean:         V(phi) = c_1 * [(c_4-phi)^2 + c_2] * exp(-c_3*phi)
    # DoubleExp:    V(phi) = c_1 * (exp(-c_2*phi) + c_3 * exp(-c_4*phi))
    if potential in ("power-law",):
        scf_c1 = {"value": 1e-2, "drop": True}
        scf_c2 = {"prior": {"min": 0.5, "max": 3.5}, "drop": True}
        scf_c3 = {"prior": {"dist": "loguniform", "a": 1e-6, "b": 1e4}, "drop": True}
        scf_c4 = 0.0
    elif potential in ("cosine",):
        scf_c1 = {"value": 1e-7, "drop": True}
        scf_c2 = {"prior": {"min": 0.0, "max": 6.2832}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("hyperbolic",):
        scf_c1 = {"value": 1e-7, "drop": True}
        scf_c2 = {"prior": {"min": 0.0, "max": 3.0}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("pNG",):
        scf_c1 = {"value": 1e-1, "drop": True}
        scf_c2 = {"prior": {"dist": "loguniform", "a": 1e-6, "b": 1e1}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("iPL",):
        scf_c1 = {"value": 1e-2, "drop": True}
        scf_c2 = {"prior": {"min": 0.0, "max": 4.0}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("exponential",):
        scf_c1 = {"value": 1e-7, "drop": True}
        scf_c2 = {"prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("SqE",):
        scf_c1 = {"prior": {"dist": "loguniform", "a": 1e-10, "b": 1e1}, "drop": True}
        scf_c2 = {"prior": {"min": -4.0, "max": 4.0}, "drop": True}
        scf_c3 = 0.0
        scf_c4 = 0.0
    elif potential in ("Bean",):
        scf_c1 = {"value": 1e-7, "drop": True}
        scf_c2 = {"prior": {"dist": "loguniform", "a": 1e-24, "b": 1e6}, "drop": True}
        scf_c3 = {"prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1}, "drop": True}
        scf_c4 = {"prior": {"min": 0.0, "max": 4.0}, "drop": True}
    elif potential in ("DoubleExp",):
        scf_c1 = {"value": 1e-7, "drop": True}
        scf_c2 = {"prior": {"min": 0.0, "max": 500}, "drop": True}
        scf_c3 = {"prior": {"min": 0.0, "max": 10.0}, "drop": True}
        scf_c4 = {"prior": {"min": 0.0, "max": 2.0}, "drop": True}
    else:
        raise ValueError(
            "potential must be one of: 'power-law', 'cosine', 'hyperbolic','pNG', 'iPL', 'SqE', 'exponential', 'Bean', 'DoubleExp'"
        )

    scf_exp_f = {}  # default: no extra prior
    if coupling in ("uncoupled",):
        scf_q1 = {"value": 0, "drop": True}
        scf_q2 = {"value": 0, "drop": True}
        scf_q3 = {"value": 0, "drop": True}
        scf_q4 = {"value": 0, "drop": True}
        scf_exp1 = {"value": 0, "drop": True}
        scf_exp2 = {"value": 0, "drop": True}
    elif coupling in ("coupled",):
        scf_q1 = {"prior": {"dist": "loguniform", "a": 1e-80, "b": 1e-30}, "drop": True}
        scf_q2 = {"prior": {"dist": "loguniform", "a": 1e-80, "b": 1e-30}, "drop": True}
        scf_q3 = {"prior": {"min": -10, "max": 10}, "drop": True}
        scf_q4 = {"prior": {"min": -10, "max": 10}, "drop": True}
        scf_exp1 = {"prior": {"min": 0, "max": 10}, "drop": True}
        scf_exp2 = {"prior": {"min": 0, "max": 5.5}, "drop": True}
        # # We have an exchange symmetry between DE and DM in the coupling, such that we only need to explore scf_exp2 in (0 , scf_exp1/2).
        scf_exp_f = {
            "prior": {
                "scf_exp2_constraint": "lambda scf_exp1, scf_exp2: 0.0 if scf_exp2 < scf_exp1/2 else -np.inf"
            }
        }
    else:
        raise ValueError("coupling must be 'uncoupled' or 'coupled'")

    if attractor in ("yes", "Yes", "YES"):
        scf_phi_ini = {"value": 0.001, "drop": True}
        scf_phi_prime_ini = {"value": 0.1, "drop": True}
    elif attractor in ("no", "No", "NO"):
        scf_phi_ini = {
            "prior": {"dist": "loguniform", "a": 1e-12, "b": 1e3},
            "drop": True,
        }
        scf_phi_prime_ini = {
            "prior": {"min": -10.0, "max": 10.0},
            "drop": True,
        }
    else:
        raise ValueError("attractor must be 'yes' or 'no'")

    parameters_iDM = {
        "cdm_c": cdm_c,
        "scf_c1": scf_c1,
        "scf_c2": scf_c2,
        "scf_c3": scf_c3,
        "scf_c4": scf_c4,
        "scf_q1": scf_q1,
        "scf_q2": scf_q2,
        "scf_q3": scf_q3,
        "scf_q4": scf_q4,
        "scf_exp1": scf_exp1,
        "scf_exp2": scf_exp2,
        "scf_phi_ini": scf_phi_ini,
        "scf_phi_prime_ini": scf_phi_prime_ini,
        "scf_parameters": {
            "value": 'lambda scf_c1,scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(scf_c1),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
            "derived": False,
        },
        "Omega_fld": 0.00,
        "Omega_scf": {"value": -0.7, "latex": "\\Omega_\\phi"},
        "Omega_Lambda": {"value": 0.0, "latex": "\\Omega_\\Lambda"},
    }

    # Combine all parameters
    params = parameters_iDM.copy()
    params.update(parameters_base)
    if likelihood in ["Run1_Planck_2018", "Run3_Planck_PP_SH0ES_DESIDR2"]:
        params.update(parameters_planck)
    # params.update(parameters_S12)

    parameters = {"params": params}

    # CLASS settings
    # class_path = "/Users/klmba/kDrive/Sci/PhD/Research/HDM/class_public"
    class_path = "/home/users/u103677/iDM/"

    extra_args = {
        "N_ncdm": 1,
        "N_ur": 2.0328,
        "model_cdm": "i",
        "tol_initial_Omega_r": 1e-3,
        "scf_tuning_index": 0,
        "gauge": "newtonian",
        "P_k_max_h/Mpc": 1.0,
        "scf_potential": potential,
        "attractor_ic_scf": attractor,
    }
    if likelihood in ["Run1_Planck_2018", "Run3_Planck_PP_SH0ES_DESIDR2"]:
        extra_args["non linear"] = "halofit"

    theorycode = {
        "theory": {
            "classy": {
                "path": class_path,
                "extra_args": extra_args,
                # "requires": requires,
            }
        }
    }

    # Return one dict that represents user choice
    config = {}
    config.update(SAMPLERS[sampler])  # Adds "sampler" key
    config.update(LIKELIHOODS[likelihood])  # Adds "likelihood" key
    # config.update(requires)  # Adds dummy likelihood for sigma_R
    config.update(parameters)  # Add "params"
    if coupling in ("coupled",):
        config.update(scf_exp_f)  # Add constraint on scf_exp2 < scf_exp1 / 2
    config.update(theorycode)  # Add "theory"

    return config


configuration = create_cobaya_yaml(sampler, likelihood, potential, attractor, coupling)

# Generate filename
attractor_name = "InitCond" if attractor in ("no", "No", "NO") else "tracking"
filename = f"cobaya_{sampler}_{likelihood}_{potential}_{attractor_name}_{coupling}.yml"

# Specify the output filename in the YAML file
output = {
    "output": "/project/home/p201176/" + filename.replace(".yml", ""),
}
configuration.update(output)

# Writing nested data to a YAML file
try:
    with open(filename, "w") as file:
        yaml.dump(configuration, file)

    # Post-process to use bracket notation for lists
    with open(filename, "r") as file:
        content = file.read()
    # The sigma_R() function requires z and R to be in brackets, yet standard YAML notation is a new line with a dash.
    # So we use regex to replace that pattern with bracket notation.
    # It only triggers in lines that start with spaces followed by 'z:' or 'R:' and then a new line with spaces, a dash and a number.
    # This should not appear somewhere else in the file.
    content = re.sub(r"(\s+)z:\s*\n\s*-\s*([0-9.]+)", r"\1z: [\2]", content)
    content = re.sub(r"(\s+)R:\s*\n\s*-\s*([0-9.]+)", r"\1R: [\2]", content)
    with open(filename, "w") as file:
        file.write(content)
except IOError as e:
    print(f"Error handling YAML file: {e}")
    raise
except Exception as e:
    print(f"Unexpected error during YAML processing: {e}")
    raise

print(f"cobaya configuration has been written to '{filename}'")
