# This routine creates the configuration files for the Cobaya runs to assess the iDM model and compare it to LCDM.
# Output: cobaya_<sampler>_<likelihood>_<potential>_<attractor>_<coupling>.yml
# For LCDM: cobaya_<sampler>_<likelihood>_LCDM.yml

from typing import Any, Union
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import re

# Specify the parameters
sampler: str = "mcmc"  # MCMC or Polychord
likelihood: str = "CV_PP_S_DESI"  # likelihood combination
potential: str = "hyperbolic"  # LCDM or iDM potential for scalar field models
attractor: str = "no"  # Scaling Solution; Ignored for LCDM
coupling: str = "uncoupled"  # Coupling; Ignored for LCDM

yaml: YAML = YAML()
# Configure YAML formatting: standard 2-space indentation for mappings and sequences
# offset=2 ensures list items are indented from their parent key (output_params:\n  - item)
yaml.indent(mapping=2, sequence=2, offset=2)
# yaml.version = (1, 2)  # Specify YAML version


def flow_seq(lst: list[Any]) -> CommentedSeq:
    """Create a YAML sequence that will be serialized in flow style [a, b, c]"""
    cs = CommentedSeq(lst)
    cs.fa.set_flow_style()  # type: ignore[attr-defined]
    return cs


def create_cobaya_yaml(
    sampler: str,
    likelihood: str,
    potential: str,
    attractor: str = "no",
    coupling: str = "uncoupled",
) -> dict[str, Any]:
    """
    Create a Cobaya YAML configuration dictionary.

    Any potential (LCDM or iDM) can be combined with any likelihood.

    Parameters:
        - sampler (str): Sampling method. Options: 'polychord', 'mcmc', 'mcmc_fast', 'minimize_polychord', 'minimize_mcmc', 'minimize_mcmc_fast', 'post_mcmc', 'post_polychord'.
            Note: 'minimize_*' samplers use the minimize sampler but output paths point to corresponding sampler chains.
            Note: 'post_*' samplers create a post-processing configuration with the static swampland params block; only the output path is dynamic.
    - likelihood (str): Likelihood combination. Options:
      'Run1_Planck_2018', 'Run2_PP_SH0ES_DESIDR2', 'Run3_Planck_PP_SH0ES_DESIDR2',
      'CV_CMB_SPA', 'CV_CMB_SPA_PP_DESI', 'CV_CMB_SPA_PP_S_DESI', 'CV_PP_DESI', 'CV_PP_S_DESI'.
      Note: 'Run3_Planck_PP_SH0ES_DESIDR2' is a post-processing run that adds likelihoods to Run 1 chains.
    - potential (str): Model. Options: 'LCDM', 'power-law', 'cosine', 'hyperbolic', 'pNG', 'iPL', 'SqE', 'exponential', 'Bean', 'DoubleExp'.
      Note: 'LCDM' uses standard CLASS - attractor and coupling settings are ignored.
      Note: 'power-law', 'cosine', 'pNG', 'iPL', 'SqE' do not support attractor initial conditions.
    - attractor (str): Initial condition type. Options: 'yes'/'Yes'/'YES' (tracking), 'no'/'No'/'NO' (phi_ini).
      Ignored for 'LCDM' potential.
    - coupling (str): Coupling type. Options: 'uncoupled', 'coupled'.
      Ignored for 'LCDM' potential.

    Returns:
    - dict: Cobaya configuration dictionary.
    """

    # Check if this is a CosmoVerse LCDM run
    is_lcdm = potential == "LCDM"

    # Validate inputs
    if sampler not in [
        "polychord",
        "mcmc",
        "mcmc_fast",
        "minimize_polychord",
        "minimize_mcmc",
        "minimize_mcmc_fast",
        "post_mcmc",
        "post_polychord",
    ]:
        raise ValueError(
            f"Unknown sampler '{sampler}'. Must be one of: polychord, mcmc, mcmc_fast, minimize_polychord, minimize_mcmc, minimize_mcmc_fast, post_mcmc, post_polychord"
        )

    valid_likelihoods = [
        "Run1_Planck_2018",
        "Run2_PP_SH0ES_DESIDR2",
        "Run3_Planck_PP_SH0ES_DESIDR2",
        "CV_CMB_SPA",
        "CV_CMB_SPA_PP_DESI",
        "CV_CMB_SPA_PP_S_DESI",
        "CV_PP_DESI",
        "CV_PP_S_DESI",
    ]
    if likelihood not in valid_likelihoods:
        raise ValueError(
            f"Unknown likelihood '{likelihood}'. Must be one of: {', '.join(valid_likelihoods)}"
        )

    valid_potentials = [
        "LCDM",
        "power-law",
        "cosine",
        "hyperbolic",
        "pNG",
        "iPL",
        "SqE",
        "exponential",
        "Bean",
        "DoubleExp",
    ]
    if potential not in valid_potentials:
        raise ValueError(
            f"Unknown potential '{potential}'. Must be one of: {', '.join(valid_potentials)}"
        )

    # Validate attractor and coupling only for non-LCDM potentials
    if not is_lcdm:
        if attractor not in ["yes", "Yes", "YES", "no", "No", "NO"]:
            raise ValueError(
                f"attractor must be 'yes', 'Yes', 'YES', 'no', 'No', or 'NO', got '{attractor}'"
            )
        if coupling not in ["uncoupled", "coupled"]:
            raise ValueError(
                f"coupling must be 'uncoupled' or 'coupled', got '{coupling}'"
            )
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

    polychord: dict[str, Any] = {
        "sampler": {
            "polychord": {
                "nprior": "20nlive",
                "num_repeats": "5d",
                "synchronous": False,
            }
        },
    }

    mcmc: dict[str, Any] = {
        "sampler": {
            "mcmc": {
                "covmat": "auto",
                "drag": True,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "Rminus1_stop": 0.02,
                "Rminus1_cl_stop": 0.2,
                "learn_proposal": True,
                "measure_speeds": True,
                "max_tries": float("inf"),
            }
        },
    }

    mcmc_fast: dict[str, Any] = {
        "sampler": {
            "mcmc": {
                "covmat": "auto",
                "drag": True,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "Rminus1_stop": 0.05,
                "Rminus1_cl_stop": 0.3,
                "learn_proposal": True,
                "measure_speeds": True,
            }
        },
    }

    minimize: dict[str, Any] = {
        "sampler": {
            "minimize": {
                "best_of": 4,
                "ignore_prior": True,  # needed for AIC and BIC computation, where prior is not taken into account
            }
        },
    }

    SAMPLERS: dict[str, dict[str, Any]] = {
        "polychord": polychord,
        "mcmc": mcmc,
        "mcmc_fast": mcmc_fast,
        "minimize_polychord": minimize,
        "minimize_mcmc": minimize,
        "minimize_mcmc_fast": minimize,
    }

    # Define Likelihoods

    Run1_Planck_2018: dict[str, Any] = {
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

    Run2_PP_SH0ES_DESIDR2: dict[str, Any] = {
        "likelihood": {
            "H0.riess2020": None,
            "bao.desi_dr2": None,
            # "sn.pantheon": {"use_abs_mag": True},
            "sn.pantheonplus": None,
        },
    }

    # Run 3 is a post-processing run that adds the likelihoods from Run 2 to Run 1 chains
    Run3_Planck_PP_SH0ES_DESIDR2_add: dict[str, Any] = Run2_PP_SH0ES_DESIDR2

    # CosmoVerse Likelihoods
    # CMB-only: Planck low-l TT + ACT DR6 + ACT lensing + SPT lensing (MUSE)
    CV_CMB_SPA: dict[str, Any] = {
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "act_dr6_cmbonly.PlanckActCut": {
                "package_install": {
                    "github_repository": "ACTCollaboration/DR6-ACT-lite"
                },
                "dataset_params": {
                    "use_cl": "tt te ee",
                    "lmin_cuts": "0 0 0",
                    "lmax_cuts": "1000 600 600",
                },
                "params": {
                    "A_planck": {
                        "prior": {"min": 0.5, "max": 1.5},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.1},
                        "latex": "A_{\\rm planck}",
                        "proposal": 0.003,
                    }
                },
            },
            "act_dr6_cmbonly.ACTDR6CMBonly": {
                "input_file": "dr6_data_cmbonly.fits",
                "lmax_theory": 9000,
                "ell_cuts": {
                    "TT": flow_seq([600, 8500]),
                    "TE": flow_seq([600, 8500]),
                    "EE": flow_seq([600, 8500]),
                },
                "stop_at_error": True,
                "params": {
                    "A_act": {
                        "value": "lambda A_planck: A_planck",
                        "latex": "A_{\\rm ACT}",
                    },
                    "P_act": {
                        "prior": {"min": 0.9, "max": 1.1},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
                        "proposal": 0.01,
                        "latex": "p_{\\rm ACT}",
                    },
                },
            },
            "act_dr6_lenslike.ACTDR6LensLike": {
                "package_install": {
                    "github_repository": "ACTCollaboration/act_dr6_lenslike"
                },
                "lens_only": False,
                "stop_at_error": True,
                "lmax": 4000,
                "variant": "actplanck_baseline",
            },
            "spt3g_d1_tne": {
                "package_install": {
                    "github_repository": "SouthPoleTelescope/spt_candl_data"
                },
                "class": "candl.interface.CandlCobayaLikelihood",
                "additional_args": {},
                "clear_internal_priors": True,
                "data_set_file": "spt_candl_data.SPT3G_D1_TnE",
                "variant": "lite",
                "feedback": True,
                "wrapper": None,
            },
            "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": {
                "package_install": {
                    "download_url": "https://pole.uchicago.edu/public/data/ge25/muse_3g_like_march_2025.zip"
                },
                "components": flow_seq(["\u03d5\u03d5"]),
            },
        },
    }

    # CMB + Pantheon+ + DESI DR2
    CV_CMB_SPA_PP_DESI: dict[str, Any] = {
        "likelihood": {
            "bao.desi_dr2": None,
            "sn.pantheonplus": None,
            "planck_2018_lowl.TT": None,
            "act_dr6_cmbonly.PlanckActCut": {
                "package_install": {
                    "github_repository": "ACTCollaboration/DR6-ACT-lite"
                },
                "dataset_params": {
                    "use_cl": "tt te ee",
                    "lmin_cuts": "0 0 0",
                    "lmax_cuts": "1000 600 600",
                },
                "params": {
                    "A_planck": {
                        "prior": {"min": 0.5, "max": 1.5},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.1},
                        "latex": "A_{\\rm planck}",
                        "proposal": 0.003,
                    }
                },
            },
            "act_dr6_cmbonly.ACTDR6CMBonly": {
                "input_file": "dr6_data_cmbonly.fits",
                "lmax_theory": 9000,
                "ell_cuts": {
                    "TT": flow_seq([600, 8500]),
                    "TE": flow_seq([600, 8500]),
                    "EE": flow_seq([600, 8500]),
                },
                "stop_at_error": True,
                "params": {
                    "A_act": {
                        "value": "lambda A_planck: A_planck",
                        "latex": "A_{\\rm ACT}",
                    },
                    "P_act": {
                        "prior": {"min": 0.9, "max": 1.1},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
                        "proposal": 0.01,
                        "latex": "p_{\\rm ACT}",
                    },
                },
            },
            "act_dr6_lenslike.ACTDR6LensLike": {
                "package_install": {
                    "github_repository": "ACTCollaboration/act_dr6_lenslike"
                },
                "lens_only": False,
                "stop_at_error": True,
                "lmax": 4000,
                "variant": "actplanck_baseline",
            },
            "spt3g_d1_tne": {
                "package_install": {
                    "github_repository": "SouthPoleTelescope/spt_candl_data"
                },
                "class": "candl.interface.CandlCobayaLikelihood",
                "additional_args": {},
                "clear_internal_priors": True,
                "data_set_file": "spt_candl_data.SPT3G_D1_TnE",
                "variant": "lite",
                "feedback": True,
                "wrapper": None,
            },
            "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": {
                "package_install": {
                    "download_url": "https://pole.uchicago.edu/public/data/ge25/muse_3g_like_march_2025.zip"
                },
                "components": flow_seq(["\u03d5\u03d5"]),
            },
        },
    }

    # CMB + PantheonPlusSHoES + DESI DR2
    CV_CMB_SPA_PP_S_DESI: dict[str, Any] = {
        "likelihood": {
            "bao.desi_dr2": None,
            "sn.pantheonplusshoes": None,
            "planck_2018_lowl.TT": None,
            "act_dr6_cmbonly.PlanckActCut": {
                "package_install": {
                    "github_repository": "ACTCollaboration/DR6-ACT-lite"
                },
                "dataset_params": {
                    "use_cl": "tt te ee",
                    "lmin_cuts": "0 0 0",
                    "lmax_cuts": "1000 600 600",
                },
                "params": {
                    "A_planck": {
                        "prior": {"min": 0.5, "max": 1.5},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.1},
                        "latex": "A_{\\rm planck}",
                        "proposal": 0.003,
                    }
                },
            },
            "act_dr6_cmbonly.ACTDR6CMBonly": {
                "input_file": "dr6_data_cmbonly.fits",
                "lmax_theory": 9000,
                "ell_cuts": {
                    "TT": flow_seq([600, 8500]),
                    "TE": flow_seq([600, 8500]),
                    "EE": flow_seq([600, 8500]),
                },
                "stop_at_error": True,
                "params": {
                    "A_act": {
                        "value": "lambda A_planck: A_planck",
                        "latex": "A_{\\rm ACT}",
                    },
                    "P_act": {
                        "prior": {"min": 0.9, "max": 1.1},
                        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
                        "proposal": 0.01,
                        "latex": "p_{\\rm ACT}",
                    },
                },
            },
            "act_dr6_lenslike.ACTDR6LensLike": {
                "package_install": {
                    "github_repository": "ACTCollaboration/act_dr6_lenslike"
                },
                "lens_only": False,
                "stop_at_error": True,
                "lmax": 4000,
                "variant": "actplanck_baseline",
            },
            "spt3g_d1_tne": {
                "package_install": {
                    "github_repository": "SouthPoleTelescope/spt_candl_data"
                },
                "class": "candl.interface.CandlCobayaLikelihood",
                "additional_args": {},
                "clear_internal_priors": True,
                "data_set_file": "spt_candl_data.SPT3G_D1_TnE",
                "variant": "lite",
                "feedback": True,
                "wrapper": None,
            },
            "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": {
                "package_install": {
                    "download_url": "https://pole.uchicago.edu/public/data/ge25/muse_3g_like_march_2025.zip"
                },
                "components": flow_seq(["\u03d5\u03d5"]),
            },
        },
    }

    # Pantheon+ + DESI DR2 only (no CMB)
    CV_PP_DESI: dict[str, Any] = {
        "likelihood": {
            "bao.desi_dr2": None,
            "sn.pantheonplus": None,
        },
    }

    # PantheonPlusSHoES + DESI DR2 only (no CMB)
    CV_PP_S_DESI: dict[str, Any] = {
        "likelihood": {
            "bao.desi_dr2": None,
            "sn.pantheonplusshoes": None,
        },
    }

    LIKELIHOODS: dict[str, dict[str, Any]] = {
        "Run1_Planck_2018": Run1_Planck_2018,
        "Run2_PP_SH0ES_DESIDR2": Run2_PP_SH0ES_DESIDR2,
        "Run3_Planck_PP_SH0ES_DESIDR2": Run3_Planck_PP_SH0ES_DESIDR2_add,
        "CV_CMB_SPA": CV_CMB_SPA,
        "CV_CMB_SPA_PP_DESI": CV_CMB_SPA_PP_DESI,
        "CV_CMB_SPA_PP_S_DESI": CV_CMB_SPA_PP_S_DESI,
        "CV_PP_DESI": CV_PP_DESI,
        "CV_PP_S_DESI": CV_PP_S_DESI,
    }

    # Check if likelihood includes CMB data (for tau_reio handling)
    has_cmb = likelihood in [
        "Run1_Planck_2018",
        "Run3_Planck_PP_SH0ES_DESIDR2",
        "CV_CMB_SPA",
        "CV_CMB_SPA_PP_DESI",
        "CV_CMB_SPA_PP_S_DESI",
    ]
    # Check if likelihood includes SPT candl (for nuisance parameters)
    has_spt_candl = likelihood in [
        "CV_CMB_SPA",
        "CV_CMB_SPA_PP_DESI",
        "CV_CMB_SPA_PP_S_DESI",
    ]

    # Define cosmological Parameters

    # Base parameters (updated to match CosmoVerse LCDM conventions)
    # tau_reio handling: Gaussian prior for CMB runs, fixed for non-CMB runs
    tau_reio_param: Union[dict[str, Any], float]
    if has_cmb:
        tau_reio_param = {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"dist": "norm", "loc": 0.051, "scale": 0.006},
            "ref": 0.051,
            "proposal": 0.003,
        }
    else:
        tau_reio_param = 0.06  # Fixed value for non-CMB runs

    parameters_base: dict[str, Any] = {
        "logA": {
            "drop": True,
            "latex": "\\log(10^{10} A_\\mathrm{s})",
            "prior": {"max": 3.91, "min": 1.61},
            "proposal": 0.001,
            "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
        },
        "n_s": {
            "latex": "n_\\mathrm{s}",
            "prior": {"max": 1.2, "min": 0.8},
            "proposal": 0.002,
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
        },
        "H0": {
            "latex": "H_0",
            "prior": {"max": 100, "min": 40},
            "proposal": 2,
            "ref": {"dist": "norm", "loc": 67, "scale": 2},
        },
        "tau_reio": tau_reio_param,
        "omega_b": {
            "latex": "\\Omega_\\mathrm{b} h^2",
            "prior": {"max": 0.03, "min": 0.0075},
            "proposal": 0.0001,
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
        },
        "omega_cdm": {
            "latex": "\\Omega_\\mathrm{c} h^2",
            "prior": {"max": 0.99, "min": 0.001},
            "proposal": 0.0005,
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
        },
        "m_ncdm": {"value": 0.06},
        "Omega_m": {"latex": "\\Omega_\\mathrm{m}", "derived": True},
        "omegamh2": {
            "derived": "lambda Omega_m, H0: Omega_m*(H0/100)**2",
            "latex": "\\Omega_\\mathrm{m} h^2",
        },
        "Omega_Lambda": {"latex": "\\Omega_\\Lambda", "derived": True},
        "A_s": {
            "value": "lambda logA: 1e-10*np.exp(logA)",
            "derived": True,
            "latex": "A_\\mathrm{s}",
        },
        "YHe": {"latex": "Y_\\mathrm{P}", "derived": True},
        "z_reio": {"latex": "z_\\mathrm{re}", "derived": True},
        "sigma8": {"latex": "\\sigma_8", "derived": True},
        "s8h5": {
            "derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)",
            "latex": "\\sigma_8/h^{0.5}",
        },
        "s8omegamp5": {
            "derived": "lambda sigma8, Omega_m: sigma8*Omega_m**0.5",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.5}",
        },
        "s8omegamp25": {
            "derived": "lambda sigma8, Omega_m: sigma8*Omega_m**0.25",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.25}",
        },
        "A": {"derived": "lambda A_s: 1e9*A_s", "latex": "10^9 A_\\mathrm{s}"},
        "clamp": {
            "derived": "lambda A_s, tau_reio: 1e9*A_s*np.exp(-2*tau_reio)",
            "latex": "10^9 A_\\mathrm{s} e^{-2\\tau}",
        },
        "age": {"latex": "{\\rm{Age}}/\\mathrm{Gyr}", "derived": True},
        "rs_drag": {"latex": "r_\\mathrm{drag}", "derived": True},
        "theta_star_100": {"latex": "100\\theta_\\star", "derived": True},
        "z_star": {"latex": "z_\\star", "derived": True},
        "rs_star": {"latex": "r_\\star", "derived": True},
        "da_star": {"latex": "D_A(z_\\star)", "derived": True},
        "z_d": {"latex": "z_\\mathrm{d}", "derived": True},
        "z_eq": {"latex": "z_\\mathrm{eq}", "derived": True},
        "k_eq": {"latex": "k_\\mathrm{eq}", "derived": True},
        "S8": {
            "derived": "lambda sigma8, omegam: (omegam/0.3)**0.5/sigma8",
            "latex": "S_8",
        },
        "M8": {
            "derived": "lambda sigma8, omegam: (omegam/0.3)**0.5*sigma8",
            "latex": "M_8",
        },
        "omegamh3": {
            "derived": "lambda omegam, H0: omegam*(H0/100)**3",
            "latex": "\\Omega_\\mathrm{m} h^3",
        },
        "rs_d_h": {"latex": "r_\\mathrm{drag}", "derived": True},
    }

    # SPT candl nuisance parameters (only for CV_CMB_SPA runs)
    parameters_spt_candl: dict[str, Any] = {
        "Tcal": {
            "latex": "T_{\\rm cal}",
            "prior": {"dist": "norm", "loc": 1.0, "scale": 0.0036},
            "ref": 1.0,
        },
        "Ecal": {
            "latex": "E_{\\rm cal}",
            "prior": {"max": 1.2, "min": 0.8},
            "ref": 1.0,
        },
    }

    # Planck-specific derived parameters (for runs with theta_s_100 as sampled parameter)
    parameters_planck: dict[str, Any] = {
        "theta_s_100": {
            "latex": "100\\theta_\\mathrm{s}",
            "prior": {"max": 10, "min": 0.5},
            "proposal": 0.0002,
            "ref": {"dist": "norm", "loc": 1.0416, "scale": 0.0004},
        },
        "H0": {
            "latex": "H_0"
        },  # Override to remove prior/proposal (derived from theta_s)
    }

    # Scalar Field Parameters for iDM model (only for non-LCDM potentials)
    parameters_iDM: dict[str, Any] = {}
    scf_exp_f: dict[str, Any] = {}  # default: no extra prior

    if not is_lcdm:
        cdm_c: dict[str, Any] = {
            "prior": {"min": -3, "max": 3},
            "latex": "c_\\mathrm{DM}",
        }
        if potential == "hyperbolic":
            cdm_c["ref"] = {
                "dist": "norm",
                "loc": 0.0,
                "scale": 0.28,
            }
        if potential == "DoubleExp":
            cdm_c["ref"] = {
                "dist": "norm",
                "loc": -0.071,
                "scale": 0.38,
            }

        # power-law:    V(phi) = c_1^(4-c_2) * phi^(c_2) + c_3
        # cosine:       V(phi) = c_1 * cos(phi*c_2)
        # hyperbolic:   V(phi) = c_1 * [1-tanh(c_2*phi)]
        # pNG:          V(phi) = c_1^4 * [1 + cos(phi/c_2)]
        # iPL:          V(phi) = c_1^(4+c_2) * phi^(-c_2)
        # exponential:  V(phi) = c_1 * exp(-c_2*phi)
        # SqE:          V(phi) = c_1^(c_2+4) * phi^(-c_2) * exp(c_1*phi^2)
        # Bean:         V(phi) = c_1 * [(c_4-phi)^2 + c_2] * exp(-c_3*phi)
        # DoubleExp:    V(phi) = c_1 * (exp(-c_2*phi) + c_3 * exp(-c_4*phi))
        scf_c1: dict[str, Any]
        scf_c2: dict[str, Any]
        scf_c3: dict[str, Any]
        scf_c4: dict[str, Any]
        if potential in ("power-law",):
            scf_c1 = {"value": 1e-2, "drop": True, "latex": "c_1"}
            scf_c2 = {"prior": {"min": 0.5, "max": 3.5}, "drop": True, "latex": "c_2"}
            scf_c3 = {
                "prior": {"dist": "loguniform", "a": 1e-6, "b": 1e4},
                "drop": True,
                "latex": "c_3",
            }
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("cosine",):
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.0, "max": 6.2832},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("hyperbolic",):
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.0, "max": 3.0},
                "drop": True,
                "latex": "c_2",
                "ref": {"dist": "norm", "loc": 0.98, "scale": 0.77},
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("pNG",):
            scf_c1 = {"value": 1e-1, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"dist": "loguniform", "a": 1e-6, "b": 1e1},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("iPL",):
            scf_c1 = {"value": 1e-2, "drop": True, "latex": "c_1"}
            scf_c2 = {"prior": {"min": 0.0, "max": 4.0}, "drop": True, "latex": "c_2"}
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("exponential",):
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("SqE",):
            scf_c1 = {
                "prior": {"dist": "loguniform", "a": 1e-10, "b": 1e1},
                "drop": True,
                "latex": "c_1",
            }
            scf_c2 = {"prior": {"min": -4.0, "max": 4.0}, "drop": True, "latex": "c_2"}
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("Bean",):
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"dist": "loguniform", "a": 1e-24, "b": 1e6},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {
                "prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1},
                "drop": True,
                "latex": "c_3",
            }
            scf_c4 = {"prior": {"min": 0.0, "max": 4.0}, "drop": True, "latex": "c_4"}
        elif potential in ("DoubleExp",):
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.0, "max": 500},
                "drop": True,
                "latex": "c_2",
                "ref": {"dist": "uniform", "min": 198.0, "max": 500.0},
            }
            scf_c3 = {
                "prior": {"min": 0.0, "max": 10.0},
                "drop": True,
                "latex": "c_3",
                "ref": {"dist": "uniform", "min": 0.0, "max": 4.96},
            }
            scf_c4 = {
                "prior": {"min": 0.0, "max": 2.0},
                "drop": True,
                "latex": "c_4",
                "ref": {"dist": "norm", "loc": 0.71, "scale": 0.26},
            }
        else:
            # Should not reach here due to validation, but provide defaults for type checker
            raise ValueError(f"Unknown potential: {potential}")

        scf_q1: dict[str, Any]
        scf_q2: dict[str, Any]
        scf_q3: dict[str, Any]
        scf_q4: dict[str, Any]
        scf_exp1: dict[str, Any]
        scf_exp2: dict[str, Any]
        if coupling == "uncoupled":
            scf_q1 = {"value": 0, "drop": True, "latex": "q_1"}
            scf_q2 = {"value": 0, "drop": True, "latex": "q_2"}
            scf_q3 = {"value": 0, "drop": True, "latex": "q_3"}
            scf_q4 = {"value": 0, "drop": True, "latex": "q_4"}
            scf_exp1 = {"value": 0, "drop": True, "latex": "\\exp_1"}
            scf_exp2 = {"value": 0, "drop": True, "latex": "\\exp_2"}
        else:  # coupling == "coupled"
            scf_q1 = {
                "prior": {"dist": "loguniform", "a": 1e-80, "b": 1e-30},
                "drop": True,
                "latex": "q_1",
            }
            scf_q2 = {
                "prior": {"dist": "loguniform", "a": 1e-80, "b": 1e-30},
                "drop": True,
                "latex": "q_2",
            }
            scf_q3 = {"prior": {"min": -10, "max": 10}, "drop": True, "latex": "q_3"}
            scf_q4 = {"prior": {"min": -10, "max": 10}, "drop": True, "latex": "q_4"}
            scf_exp1 = {
                "prior": {"min": 0, "max": 10},
                "drop": True,
                "latex": "\\exp_1",
            }
            scf_exp2 = {
                "prior": {"min": 0, "max": 5.5},
                "drop": True,
                "latex": "\\exp_2",
            }
            # # We have an exchange symmetry between DE and DM in the coupling, such that we only need to explore scf_exp2 in (0 , scf_exp1/2).
            scf_exp_f = {
                "prior": {
                    "scf_exp2_constraint": "lambda scf_exp1, scf_exp2: 0.0 if scf_exp2 < scf_exp1/2 else -np.inf"
                }
            }

        scf_phi_ini: dict[str, Any]
        scf_phi_prime_ini: dict[str, Any]
        if attractor in ("yes", "Yes", "YES"):
            scf_phi_ini = {"value": 0.001, "drop": True, "latex": "\\phi_\\mathrm{ini}"}
            scf_phi_prime_ini = {
                "value": 0.1,
                "drop": True,
                "latex": "\\phi\\prime_\\mathrm{ini}",
            }
        else:  # attractor in ("no", "No", "NO")
            scf_phi_ini = {
                "prior": {"dist": "loguniform", "a": 1e-12, "b": 1e3},
                "drop": True,
                "latex": "\\phi_\\mathrm{ini}",
            }
            scf_phi_prime_ini = {
                "prior": {"min": -10.0, "max": 10.0},
                "drop": True,
                "latex": "\\phi\\prime_\\mathrm{ini}",
            }

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
            # Swampland parameters (derived from CLASS output)
            "phi_ini_scf_ic": {"latex": "\\phi_{\\mathrm{ini}}"},
            "phi_prime_scf_ic": {"latex": "\\phi^{\\prime}_{\\mathrm{ini}}"},
            "phi_scf_min": {"latex": "\\phi_{\\mathrm{min}}"},
            "phi_scf_max": {"latex": "\\phi_{\\mathrm{max}}"},
            "phi_scf_range": {"latex": "\\Delta \\phi"},
            "dV_V_scf_min": {"latex": "\\mathfrak{s}_{1,\\mathrm{min}}"},
            "ddV_V_scf_max": {"latex": "-\\mathfrak{s}_{2,\\mathrm{max}}"},
            "ddV_V_at_dV_V_min": {
                "latex": "-\\mathfrak{s}_{2@\\mathfrak{s}_{1,\\mathrm{min}}}"
            },
            "dV_V_at_ddV_V_max": {
                "latex": "\\mathfrak{s}_{1@\\mathfrak{s}_{2,\\mathrm{max}}}"
            },
            "swgc_expr_min": {"latex": "\\mathrm{SWGC}_\\phi"},
            "sswgc_min": {"latex": "\\mathrm{SSWGC}_\\mathrm{DM}"},
            "attractor_regime_scf": None,
            "AdSDC2_max": {"latex": "m_\\mathrm{DM,min (no scale separation)}"},
            "AdSDC4_max": {"latex": "m_\\mathrm{DM,min (scale separation)}"},
            "combined_dSC_min": {
                "latex": "\\mathrm{(FLB–SSWGC) combined dSC}_\\mathrm{min}"
            },
            "conformal_age": {"latex": "\\tau_0"},
        }

    # Combine all parameters
    params: dict[str, Any] = parameters_base.copy()
    if not is_lcdm:
        params.update(parameters_iDM)
    if has_spt_candl:
        params.update(parameters_spt_candl)
    # For iDM Planck runs, use theta_s_100 as sampled parameter instead of H0
    if likelihood in [
        "Run1_Planck_2018",
        "Run3_Planck_PP_SH0ES_DESIDR2",
    ]:
        params.update(parameters_planck)

    parameters: dict[str, Any] = {"params": params}

    # CLASS settings
    # For non-SPT_candl runs, use explicit path to custom CLASS installation
    # For SPT_candl runs, omit path to use Cobaya's installed CLASS
    class_path: str | None = "/home/users/u103677/iDM/" if not has_spt_candl else None

    # Different extra_args for LCDM/CosmoVerse vs iDM runs
    extra_args: dict[str, Any]
    if is_lcdm:
        # CosmoVerse LCDM settings (standard CLASS)
        extra_args = {
            "gauge": "newtonian",
            "nonlinear_min_k_max": 25,
            "N_ncdm": 1,
            "N_ur": 2.046,
            "sBBN file": "sBBN_2017.dat",
            "non linear": "halofit",
            # "hmcode_version": 2020,
        }
    else:
        # iDM scalar field settings
        extra_args = {
            "gauge": "newtonian",
            "nonlinear_min_k_max": 25,
            "N_ncdm": 1,
            "N_ur": 2.046,
            "sBBN file": "sBBN_2017.dat",
            "non linear": "halofit",
            # "hmcode_version": 2020,
            "model_cdm": "i",
            "tol_initial_Omega_r": 1e-2,
            "scf_tuning_index": 0,
            "scf_potential": potential,
            "attractor_ic_scf": attractor,
            "output_params": [
                "phi_ini_scf_ic",
                "phi_prime_scf_ic",
                "phi_scf_min",
                "phi_scf_max",
                "phi_scf_range",
                "dV_V_scf_min",
                "ddV_V_scf_max",
                "ddV_V_at_dV_V_min",
                "dV_V_at_ddV_V_max",
                "swgc_expr_min",
                "sswgc_min",
                "attractor_regime_scf",
                "AdSDC2_max",
                "AdSDC4_max",
                "combined_dSC_min",
                "conformal_age",
            ],
        }

    theorycode: dict[str, Any] = {
        "theory": {
            "classy": {
                "extra_args": extra_args,
            }
        }
    }
    # Add explicit path for non-SPT_candl runs
    if class_path is not None:
        theorycode["theory"]["classy"]["path"] = class_path

    # Return one dict that represents user choice
    config: dict[str, Any] = {}

    # Check if this is a post-processing run (either Run3 or post_* sampler)
    is_postprocessing = (
        likelihood == "Run3_Planck_PP_SH0ES_DESIDR2" or sampler.startswith("post_")
    )

    swampland_post_block: dict[str, Any] = {
        "suffix": "swampland",
        "add": {
            "theory": {
                "classy": {
                    "path": "/home/kl/kDrive/Sci/PhD/Research/HDM/class_public",
                    "output_params": [
                        "phi_ini_scf_ic",
                        "phi_prime_scf_ic",
                        "phi_scf_min",
                        "phi_scf_max",
                        "phi_scf_range",
                        "dV_V_scf_min",
                        "ddV_V_scf_max",
                        "ddV_V_at_dV_V_min",
                        "dV_V_at_ddV_V_max",
                        "swgc_expr_min",
                        "sswgc_min",
                        "attractor_regime_scf",
                        "AdSDC2_max",
                        "AdSDC4_max",
                        "combined_dSC_min",
                        "conformal_age",
                    ],
                },
            },
            "params": {
                "phi_ini_scf_ic": {"latex": "\\phi_{\\mathrm{ini}}"},
                "phi_prime_scf_ic": {"latex": "\\phi^{\\prime}_{\\mathrm{ini}}"},
                "phi_scf_min": {"latex": "\\phi_{\\mathrm{min}}"},
                "phi_scf_max": {"latex": "\\phi_{\\mathrm{max}}"},
                "phi_scf_range": {"latex": "\\Delta \\phi"},
                "dV_V_scf_min": {"latex": "\\mathfrak{s}_{1,\\mathrm{min}}"},
                "ddV_V_scf_max": {"latex": "-\\mathfrak{s}_{2,\\mathrm{max}}"},
                "ddV_V_at_dV_V_min": {
                    "latex": "-\\mathfrak{s}_{2@\\mathfrak{s}_{1,\\mathrm{min}}}"
                },
                "dV_V_at_ddV_V_max": {
                    "latex": "\\mathfrak{s}_{1@\\mathfrak{s}_{2,\\mathrm{max}}}"
                },
                "swgc_expr_min": {"latex": "\\mathrm{SWGC}_\\phi"},
                "sswgc_min": {"latex": "\\mathrm{SSWGC}_\\mathrm{DM}"},
                "attractor_regime_scf": None,
                "AdSDC2_max": {"latex": "m_\\mathrm{DM,min (no scale separation)}"},
                "AdSDC4_max": {"latex": "m_\\mathrm{DM,min (scale separation)}"},
                "combined_dSC_min": {
                    "latex": "\\mathrm{(FLB--SSWGC) combined dSC}_\\mathrm{min}"
                },
                "conformal_age": {"latex": "\\tau_0"},
            },
        },
    }

    if is_postprocessing:
        # Post-processing configuration
        if likelihood == "Run3_Planck_PP_SH0ES_DESIDR2":
            # Run 3 is a special post-processing run
            attractor_name = (
                "tracking" if attractor in ("yes", "Yes", "YES") else "InitCond"
            )
            run1_output_path = f"/project/home/p201176/cobaya_{sampler}_Run1_Planck_2018_{potential}_{attractor_name}_{coupling}"
            config["output"] = run1_output_path  # Points to Run 1 chains
            config["post"] = {
                "suffix": "SN_BAO",
                "add": LIKELIHOODS[likelihood],  # Adds the new likelihoods
            }
        else:
            # post_mcmc or post_polychord samplers
            # Output points to the base sampler chain (without the "post_" prefix)
            base_sampler = sampler.replace("post_", "")
            attractor_name = (
                "tracking" if attractor in ("yes", "Yes", "YES") else "InitCond"
            )
            if is_lcdm:
                base_output_path = (
                    f"/project/home/p201176/cobaya_{base_sampler}_{likelihood}_LCDM"
                )
            else:
                base_output_path = f"/project/home/p201176/cobaya_{base_sampler}_{likelihood}_{potential}_{attractor_name}_{coupling}"
            config["output"] = base_output_path
            config["post"] = swampland_post_block
    else:
        # Standard sampling run
        config.update(SAMPLERS[sampler])
        # For non-CMB runs, disable drag sampling (no fast/slow parameter hierarchy)
        if not has_cmb and "mcmc" in config.get("sampler", {}):
            config["sampler"]["mcmc"]["drag"] = False
        config.update(LIKELIHOODS[likelihood])  # Adds "likelihood" key
        config.update(parameters)  # Add "params"
        if not is_lcdm and coupling == "coupled":
            config.update(scf_exp_f)  # Add constraint on scf_exp2 < scf_exp1 / 2
        config.update(theorycode)  # Add "theory"

    return config


configuration = create_cobaya_yaml(sampler, likelihood, potential, attractor, coupling)

# Compute attractor name once for filename generation
is_lcdm: bool = potential == "LCDM"  # type: ignore[comparison-overlap]
attractor_name: str = "tracking" if attractor in ("yes", "Yes", "YES") else "InitCond"
is_postprocessing_run: bool = sampler.startswith("post_")

# Generate filename based on potential type
filename: str
if is_lcdm:
    filename = f"cobaya_{sampler}_{likelihood}_LCDM.yml"
else:
    filename = (
        f"cobaya_{sampler}_{likelihood}_{potential}_{attractor_name}_{coupling}.yml"
    )

# Specify the output path in the YAML file
# For post-processing runs (Run 3 or post_* samplers), output is already set in the config
if likelihood != "Run3_Planck_PP_SH0ES_DESIDR2" and not is_postprocessing_run:  # type: ignore[comparison-overlap]
    # For minimize_ samplers, output goes to the base sampler's chain directory
    if sampler.startswith("minimize_"):
        base_sampler = sampler.replace("minimize_", "")
        if is_lcdm:
            output_filename = f"cobaya_{base_sampler}_{likelihood}_LCDM"
        else:
            output_filename = f"cobaya_{base_sampler}_{likelihood}_{potential}_{attractor_name}_{coupling}"
    else:
        output_filename = filename.replace(".yml", "")
    configuration["output"] = "/project/home/p201176/" + output_filename

# Writing nested data to a YAML file
yaml_path = f"Cobaya/MCMC/{filename}"
try:
    with open(yaml_path, "w") as file:
        yaml.dump(configuration, file)  # type: ignore[misc]

    # Post-process to use bracket notation for z and R lists (used by sigma_R())
    # Standard YAML serializes lists with dashes, but these need inline bracket notation
    with open(yaml_path, "r") as file:
        content = file.read()
    content = re.sub(r"(\s+)z:\s*\n\s*-\s*([0-9.]+)", r"\1z: [\2]", content)
    content = re.sub(r"(\s+)R:\s*\n\s*-\s*([0-9.]+)", r"\1R: [\2]", content)
    with open(yaml_path, "w") as file:
        file.write(content)
except IOError as e:
    print(f"Error handling YAML file: {e}")
    raise
except Exception as e:
    print(f"Unexpected error during YAML processing: {e}")
    raise

print(f"cobaya configuration has been written to '{yaml_path}'")

# Now, we create the bash script to run the cobaya test job on the cluster and save it as test_<filename>.sh


def create_slurm_test_script(
    yaml_filename: str,
    account: str = "p201176",
    partition: str = "cpu",
    qos: str = "test",
    nodes: int = 1,
    ntasks: int = 1,
    ntasks_per_node: int = 1,
    cpus_per_task: int = 64,
    time: str = "00:05:00",
    mail_user: str = "kay.lehnert.2023@mumail.ie",
    yaml_base_path: str = "/home/users/u103677/iDM/Cobaya/MCMC/",
) -> str:
    """
    Create a SLURM bash script for running a Cobaya test job.

    Parameters:
    - yaml_filename (str): The YAML configuration filename (e.g., 'cobaya_mcmc_fast_Run1_Planck_2018_Bean_tracking_uncoupled.yml').
    - account (str): SLURM account.
    - partition (str): SLURM partition.
    - qos (str): SLURM quality of service.
    - nodes (int): Number of nodes.
    - ntasks (int): Number of tasks.
    - ntasks_per_node (int): Tasks per node.
    - cpus_per_task (int): CPUs per task.
    - time (str): Job time limit (HH:MM:SS).
    - mail_user (str): Email for notifications.
    - yaml_base_path (str): Base path where YAML files are located on the cluster.

    Returns:
    - str: The shell script filename that was created.
    """
    # Derive job name from yaml filename
    job_name: str = "test_" + yaml_filename.replace(".yml", "")

    # Generate script filename based on yaml filename
    script_filename: str = f"SLURM/test_{yaml_filename.replace('.yml', '.sh')}"

    # Full path to the YAML file on the cluster
    yaml_full_path: str = yaml_base_path + yaml_filename

    script_content: str = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --account {account}
#SBATCH --partition {partition}
#SBATCH --qos {qos}
#SBATCH --nodes {nodes}
#SBATCH --ntasks {ntasks}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --time {time}
#SBATCH --output %j.{job_name}.out
#SBATCH --error %j.{job_name}.err
#SBATCH --mail-user {mail_user}
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS
#Activate Python virtual environment
source my_python-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run {yaml_full_path} --test --debug

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
"""

    try:
        with open(script_filename, "w") as file:
            file.write(script_content)
    except IOError as e:
        print(f"Error writing shell script: {e}")
        raise

    return script_filename


# Create the SLURM test script
slurm_script_filename: str = create_slurm_test_script(yaml_filename=filename)
print(f"SLURM test script has been written to '{slurm_script_filename}'")

# Next, we create the SLURM production script to run the full Cobaya job on the cluster and save it as run_<filename>.sh


def create_slurm_run_script(
    yaml_filename: str,
    account: str = "p201176",
    partition: str = "cpu",
    qos: str = "short",
    nodes: int = 1,
    ntasks: int = 4,
    ntasks_per_node: int = 4,
    cpus_per_task: int = 64,
    time: str = "06:00:00",
    mail_user: str = "kay.lehnert.2023@mumail.ie",
    yaml_base_path: str = "/home/users/u103677/iDM/Cobaya/MCMC/",
    max_retries: int = 99,
) -> str:
    """
    Create a SLURM bash script for running a full Cobaya production job.

    Parameters:
    - yaml_filename (str): The YAML configuration filename (e.g., 'cobaya_mcmc_fast_Run1_Planck_2018_Bean_tracking_uncoupled.yml').
    - account (str): SLURM account.
    - partition (str): SLURM partition.
    - qos (str): SLURM quality of service.
    - nodes (int): Number of nodes.
    - ntasks (int): Number of tasks.
    - ntasks_per_node (int): Tasks per node.
    - cpus_per_task (int): CPUs per task.
    - time (str): Job time limit (HH:MM:SS).
    - mail_user (str): Email for notifications.
    - yaml_base_path (str): Base path where YAML files are located on the cluster.
    - max_retries (int): Maximum number of retries for exit code 143 (SIGTERM).

    Returns:
    - str: The shell script filename that was created.
    """
    # Derive job name from yaml filename
    job_name: str = "run_" + yaml_filename.replace(".yml", "")

    # Generate script filename based on yaml filename
    script_filename: str = f"SLURM/run_{yaml_filename.replace('.yml', '.sh')}"

    # Full path to the YAML file on the cluster
    yaml_full_path: str = yaml_base_path + yaml_filename

    script_content: str = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --account {account}
#SBATCH --partition {partition}
#SBATCH --qos {qos}
#SBATCH --nodes {nodes}
#SBATCH --ntasks {ntasks}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --time {time}
#SBATCH --output %j.{job_name}.out
#SBATCH --error %j.{job_name}.err
#SBATCH --mail-user {mail_user}
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS
#Activate Python virtual environment
source my_python-env/bin/activate

#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Retry logic for exit code 143 (SIGTERM)
MAX_RETRIES={max_retries}
RETRY_COUNT=0
EXIT_CODE=143

while [ $EXIT_CODE -eq 143 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT of $MAX_RETRIES"
    srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run {yaml_full_path} --resume
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    if [ $EXIT_CODE -eq 143 ]; then
        echo "Received exit code 143, will retry..."
    fi
done

if [ $EXIT_CODE -eq 143 ]; then
    echo "Max retries ($MAX_RETRIES) reached with exit code 143"
fi

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
"""

    try:
        with open(script_filename, "w") as file:
            file.write(script_content)
    except IOError as e:
        print(f"Error writing shell script: {e}")
        raise

    return script_filename


def create_slurm_minimize_script(
    yaml_filename: str,
    account: str = "p201176",
    partition: str = "cpu",
    qos: str = "short",
    nodes: int = 1,
    ntasks: int = 4,
    ntasks_per_node: int = 4,
    cpus_per_task: int = 64,
    time: str = "01:00:00",
    mail_user: str = "kay.lehnert.2023@mumail.ie",
    yaml_base_path: str = "/home/users/u103677/iDM/Cobaya/MCMC/",
) -> str:
    """
    Create a SLURM bash script for running a Cobaya minimize job.

    Parameters:
    - yaml_filename (str): The YAML configuration filename (e.g., 'cobaya_minimize_Run1_Planck_2018_hyperbolic_tracking_uncoupled.yml').
    - account (str): SLURM account.
    - partition (str): SLURM partition.
    - qos (str): SLURM quality of service.
    - nodes (int): Number of nodes.
    - ntasks (int): Number of tasks.
    - ntasks_per_node (int): Tasks per node.
    - cpus_per_task (int): CPUs per task.
    - time (str): Job time limit (HH:MM:SS).
    - mail_user (str): Email for notifications.
    - yaml_base_path (str): Base path where YAML files are located on the cluster.

    Returns:
    - str: The shell script filename that was created.
    """
    # Derive job name from yaml filename
    job_name: str = "run_" + yaml_filename.replace(".yml", "")

    # Generate script filename based on yaml filename
    script_filename: str = f"SLURM/run_{yaml_filename.replace('.yml', '.sh')}"

    # Full path to the YAML file on the cluster
    yaml_full_path: str = yaml_base_path + yaml_filename

    script_content: str = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --account {account}
#SBATCH --partition {partition}
#SBATCH --qos {qos}
#SBATCH --nodes {nodes}
#SBATCH --ntasks {ntasks}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --time {time}
#SBATCH --output %j.{job_name}.out
#SBATCH --error %j.{job_name}.err
#SBATCH --mail-user {mail_user}
#SBATCH --mail-type END,FAIL

## Load software environment
module load GCC
module load Python
module load Cython
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS
#Activate Python virtual environment
source my_python-env/bin/activate

#Number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run minimize (no retry logic needed for optimization)
srun --cpus-per-task=$SLURM_CPUS_PER_TASK cobaya-run {yaml_full_path}

#Check energy consumption after job completion
sacct -j $SLURM_JOB_ID -o jobid,jobname,partition,account,state,consumedenergyraw
"""

    try:
        with open(script_filename, "w") as file:
            file.write(script_content)
    except IOError as e:
        print(f"Error writing shell script: {e}")
        raise

    return script_filename


# Create the SLURM production run script (use minimize script for minimize sampler variants)
slurm_run_script_filename: str
if sampler.startswith("minimize_"):
    slurm_run_script_filename = create_slurm_minimize_script(yaml_filename=filename)
else:
    slurm_run_script_filename = create_slurm_run_script(yaml_filename=filename)
print(f"SLURM run script has been written to '{slurm_run_script_filename}'")
