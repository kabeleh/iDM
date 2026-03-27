"""Generate Cobaya YAML configs and matching SLURM scripts.

Terminal usage (from repository root):

1) Default generation (uses main() defaults):
    python3 Cobaya/create_cobaya_yaml.py

2) Custom generation without editing this file:
    python3 -c "from Cobaya.create_cobaya_yaml import main; main(sampler='mcmc_fast', likelihood='CMB', potential='DoubleExp', attractor='no', coupling='uncoupled')"

3) Example: LCDM run (attractor/coupling ignored for LCDM):
    python3 -c "from Cobaya.create_cobaya_yaml import main; main(sampler='mcmc_fast', likelihood='CV_PP_DESI', potential='LCDM')"

What gets written:
- YAML files in Cobaya/MCMC/
- SLURM test scripts in SLURM/test_*.sh
- SLURM run scripts in SLURM/run_*.sh

Filename convention:
- Non-LCDM: <potential>_<likelihoods>_<attractor>_<coupling>_<sampler>.yml
- LCDM:     <potential>_<likelihoods>_<sampler>.yml
- Coupling tag is omitted when coupling='uncoupled'.
"""

from typing import Any
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import re

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


# ============================================================================
# FILENAME NAMING CONVENTION
# ============================================================================
# New scheme: <potential>_<likelihoods>_<attractor>_<coupling>_<sampler>
# For LCDM: <potential>_<likelihoods>_<sampler> (attractor and coupling omitted)
# Coupling is omitted when 'uncoupled'.

# Mapping from internal likelihood name to filename data tag
LIKELIHOOD_FILE_TAG: dict[str, str] = {
    "CMB": "Planck",
    "Run1_Planck_2018": "Planck",
    "Run2_PP_SH0ES_DESIDR2": "PP_S_D",
    "Run3_Planck_PP_SH0ES_DESIDR2": "Planck_PP_S_D",
    "CV_CMB_SPA": "SPA",
    "CV_CMB_SPA_PP_DESI": "SPA_PP_D",
    "CV_CMB_SPA_PP_S_DESI": "SPA_PP_S_D",
    "CV_PP_DESI": "PP_D",
    "CV_PP_S_DESI": "PP_S_D",
}

# Mapping from internal sampler name to filename sampler tag
SAMPLER_FILE_TAG: dict[str, str] = {
    "polychord": "Polychord",
    "mcmc": "MCMC",
    "mcmc_fast": "MCMC",
    "minimize_polychord": "Polychord_minimizer",
    "minimize_mcmc": "MCMC_minimizer",
    "minimize_mcmc_fast": "MCMC_minimizer",
    "post_mcmc": "MCMC_swamp",
    "post_polychord": "Polychord_swamp",
}


def build_filename_stem(
    potential: str,
    likelihood: str,
    attractor: str,
    coupling: str,
    sampler: str,
) -> str:
    """
    Build the filename stem (without extension) using the naming convention:
    <potential>_<likelihoods>_<attractor>_<coupling>_<sampler>

    For LCDM, attractor and coupling are omitted.
    Coupling is omitted when 'uncoupled'.
    """
    parts: list[str] = [potential, LIKELIHOOD_FILE_TAG[likelihood]]
    if potential != "LCDM":
        parts.append("tracking" if attractor.lower() == "yes" else "InitCond")
        if coupling == "coupled":
            parts.append("coupled")
    parts.append(SAMPLER_FILE_TAG[sampler])
    return "_".join(parts)


def build_chain_output_stem(
    potential: str,
    likelihood: str,
    attractor: str,
    coupling: str,
    sampler: str,
) -> str:
    """
    Build the chain output stem (used for Cobaya's output path).
    For minimize_ and post_ samplers, uses the base sampler tag.
    """
    base_sampler = sampler
    if sampler.startswith("minimize_"):
        base_sampler = sampler.replace("minimize_", "")
    elif sampler.startswith("post_"):
        base_sampler = sampler.replace("post_", "")
    return build_filename_stem(potential, likelihood, attractor, coupling, base_sampler)


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
            'CMB', 'Run1_Planck_2018', 'Run2_PP_SH0ES_DESIDR2', 'Run3_Planck_PP_SH0ES_DESIDR2',
            'CV_CMB_SPA', 'CV_CMB_SPA_PP_DESI', 'CV_CMB_SPA_PP_S_DESI', 'CV_PP_DESI', 'CV_PP_S_DESI'.
            Note: 'CMB' is Planck low-l TT/EE + plik high-l TTTEEE lite native + native lensing.
            Note: 'Run3_Planck_PP_SH0ES_DESIDR2' is a post-processing run that adds likelihoods to Run 1 chains.
    - potential (str): Model. Options: 'LCDM', 'power-law', 'cosine', 'hyperbolic', 'pNG', 'SqE', 'exponential', 'Bean', 'BeanSingleWell', 'BeanAdS', 'DoubleExp'.
      Note: 'LCDM' uses standard CLASS - attractor and coupling settings are ignored.
      Note: 'power-law', 'cosine', 'pNG', 'exponential', 'SqE', 'Bean', 'BeanSingleWell', 'BeanAdS', 'DoubleExp' do not support attractor initial conditions.
      Note: iPL (inverse power-law) is subsumed by 'power-law' with c2 in [-6, 6].
      Note: 'Bean' generates both single-well (c2>0) and double-well/AdS (c2<0) YAMLs. Use 'BeanSingleWell' or 'BeanAdS' to generate only one.
      Note: Attractor runs restrict priors to the tracking-attractor domain:
            hyperbolic: |c2|>2.
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
        "CMB",
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
        "SqE",
        "exponential",
        "Bean",
        "BeanSingleWell",
        "BeanAdS",
        "DoubleExp",
    ]
    if potential not in valid_potentials:
        raise ValueError(
            f"Unknown potential '{potential}'. Must be one of: {', '.join(valid_potentials)}"
        )

    # Validate attractor and coupling only for non-LCDM potentials
    is_attractor = attractor.lower() == "yes"
    if not is_lcdm:
        if attractor.lower() not in ("yes", "no"):
            raise ValueError(
                f"attractor must be 'yes' or 'no' (case-insensitive), got '{attractor}'"
            )
        if coupling not in ["uncoupled", "coupled"]:
            raise ValueError(
                f"coupling must be 'uncoupled' or 'coupled', got '{coupling}'"
            )
        if is_attractor and potential in (
            "power-law",
            "cosine",
            "pNG",
            "exponential",
            "SqE",
            "Bean",
            "BeanSingleWell",
            "BeanAdS",
            "DoubleExp",
        ):
            raise ValueError(
                f"Attractor initial conditions are not supported for potential '{potential}'. "
                "Either no attractor exists or tracking is degenerate with a single exponential "
                "(c1 cancels from Omega_scf on the attractor). "
                "Use attractor='no' with phi_ini sampling instead."
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
                "proposal_scale": 3.4,
                "Rminus1_stop": 0.02,
                "Rminus1_cl_stop": 0.2,
                "learn_proposal": True,
                "measure_speeds": True,
                "max_tries": float("inf"),
                # "temperature": 2.0,
            }
        },
    }

    minimize: dict[str, Any] = {
        "sampler": {
            "minimize": {
                "best_of": 10,
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

    # Planck-only CMB (no SPT candl)
    CMB: dict[str, Any] = {
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE_lite_native": None,
            "planck_2018_lensing.native": None,
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
    # Shared core: Planck low-l TT + ACT DR6 + ACT lensing + SPT candl + SPT lensing (MUSE)
    _cmb_spa_core: dict[str, Any] = {
        "planck_2018_lowl.TT": None,
        "act_dr6_cmbonly.PlanckActCut": {
            "package_install": {"github_repository": "ACTCollaboration/DR6-ACT-lite"},
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
    }

    # CMB-only
    CV_CMB_SPA: dict[str, Any] = {"likelihood": {**_cmb_spa_core}}
    # CMB + Pantheon+ + DESI DR2
    CV_CMB_SPA_PP_DESI: dict[str, Any] = {
        "likelihood": {"bao.desi_dr2": None, "sn.pantheonplus": None, **_cmb_spa_core}
    }
    # CMB + PantheonPlusSHoES + DESI DR2
    CV_CMB_SPA_PP_S_DESI: dict[str, Any] = {
        "likelihood": {
            "bao.desi_dr2": None,
            "sn.pantheonplusshoes": None,
            **_cmb_spa_core,
        }
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
        "CMB": CMB,
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
        "CMB",
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
    # The CMB likelihood uses defaults for nonlinear_min_k_max
    skip_nonlinear_min_k_max = likelihood == "CMB"

    # Define cosmological Parameters

    # Base parameters (updated to match CosmoVerse LCDM conventions)
    # tau_reio handling: Gaussian prior for CMB runs, fixed for non-CMB runs
    tau_reio_param: dict[str, Any] | float
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
            "derived": "lambda sigma8, Omega_m: (Omega_m/0.3)**0.5*sigma8",
            "latex": "S_8",
        },
        "omegamh3": {
            "derived": "lambda Omega_m, H0: Omega_m*(H0/100)**3",
            "latex": "\\Omega_\\mathrm{m} h^3",
        },
        "rs_d_h": {"latex": "r_\\mathrm{drag}", "derived": True},
    }

    # SPT candl nuisance parameters (only for CV_CMB_SPA* runs)
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
    attractor_prior: dict[str, Any] = {}  # default: no attractor constraint
    cdm_c_floor_prior: dict[str, Any] = {}  # default: no cdm_c floor
    scf_c2_floor_prior: dict[str, Any] = {}  # default: no scf_c2 floor

    # Swampland derived parameters (shared between standard runs and post-processing)
    swampland_params: dict[str, Any] = {
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
        "cdm_f_phi0": {"latex": "f_\\mathrm{DM}(\\phi_0)"},
    }

    if not is_lcdm:
        # Default cdm_c: always overridden by attractor/non-attractor branches
        # below (both set [-18, 18] with ref N(0, 3)).  Keep [-10, 10] as a
        # safe fallback in case a new branch is added without setting cdm_c.
        cdm_c: dict[str, Any] = {
            "prior": {"min": -10, "max": 10},
            "latex": "c_\\mathrm{DM}",
        }

        # power-law:    V(phi) = c_1^(4-c_2) * phi^(c_2) + c_3
        # cosine:       V(phi) = c_1 * cos(phi*c_2)
        # hyperbolic:   V(phi) = c_1 * [1-tanh(c_2*phi)]
        # pNG:          V(phi) = c_1^4 * [1 + cos(phi/c_2)]
        # iPL subsumed by power-law with c_2 in [-6, 6]
        # exponential:  V(phi) = c_1 * exp(-c_2*phi)
        # SqE:          V(phi) = c_1^(c_2+4) * phi^(-c_2) * exp(c_1*phi^2)
        # Bean:         V(phi) = c_1 * [(c_4-phi)^2 + c_2] * exp(-c_3*phi)
        # DoubleExp:    V(phi) = c_1 * (exp(-c_2*phi) + c_3 * exp(-c_4*phi))
        scf_c1: dict[str, Any]
        scf_c2: dict[str, Any]
        scf_c3: dict[str, Any]
        scf_c4: dict[str, Any]
        if potential == "power-law":
            scf_c1 = {"value": 1e-2, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": -6.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 0, "scale": 2},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {
                "value": 0.0,
                "drop": True,
                "latex": "c_3",
            }
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential == "cosine":
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.0, "max": 6.2832},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential == "hyperbolic":
            scf_c1 = {"value": 1e-8, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": -10, "max": 10},
                "ref": {"dist": "norm", "loc": 0, "scale": 3},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential == "pNG":
            scf_c1 = {"value": 1e-1, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.05, "max": 10.0},
                "ref": {"dist": "norm", "loc": 1, "scale": 2},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential == "exponential":
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"dist": "loguniform", "a": 1e-3, "b": 1e1},
                "ref": {"dist": "norm", "loc": 1, "scale": 0.5},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential == "SqE":
            # SqE: c1 is determined by shooting (log-space, now enabled in input.c).
            # No need to sample scf_c1_exp — fix c1 as initial guess for the shooter.
            scf_c1 = {"value": 1e-2, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": -4.0, "max": 4.0},
                "ref": {"dist": "norm", "loc": 1, "scale": 1},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 0.0, "drop": True, "latex": "c_3"}
            scf_c4 = {"value": 0.0, "drop": True, "latex": "c_4"}
        elif potential in ("Bean", "BeanSingleWell"):
            # Bean single-well (c2 > 0): smooth positive-definite potential
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"dist": "loguniform", "a": 1e-6, "b": 1e3},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {
                "prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1},
                "drop": True,
                "latex": "c_3",
            }
            scf_c4 = {"prior": {"min": 0.0, "max": 4.0}, "drop": True, "latex": "c_4"}
        elif potential == "BeanAdS":
            # Bean double-well (c2 < 0): AdS-like vacuum, barrier crossing
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": -100.0, "max": 0.0},
                "ref": {"dist": "norm", "loc": -1, "scale": 5},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {
                "prior": {"dist": "loguniform", "a": 1e-2, "b": 1e1},
                "drop": True,
                "latex": "c_3",
            }
            scf_c4 = {"prior": {"min": -4.0, "max": 4.0}, "drop": True, "latex": "c_4"}
        elif potential == "DoubleExp":
            # DoubleExp: V = c1*(exp(-c2*phi) + c3*exp(-c4*phi))
            # c3 fixed to 1: shift phi -> phi+delta absorbs c3 into c1 (shooter handles).
            # Coupling m(c_DM*phi) breaks the shift for non-zero c_DM, but the
            # remaining parameters (c2, c4, c_DM, phi_ini) are over-complete
            # for the ~2-3 observable modes — no distinguishable physics is lost.
            # c2 >= 0 breaks the sign-flip symmetry (c2,c4) <-> (-c2,-c4).
            # Exchange symmetry (c2,c4) <-> (c4,c2) broken by prior constraint
            # c4 >= c2 (when c4 >= 0); c4 < 0 is freely allowed (mixed regime).
            scf_c1 = {"value": 1e-7, "drop": True, "latex": "c_1"}
            scf_c2 = {
                "prior": {"min": 0.0, "max": 20.0},
                "ref": {"dist": "norm", "loc": 5, "scale": 3},
                "drop": True,
                "latex": "c_2",
            }
            scf_c3 = {"value": 1.0, "drop": True, "latex": "c_3"}
            scf_c4 = {
                "prior": {"min": -5.0, "max": 20.0},
                "ref": {"dist": "norm", "loc": 1, "scale": 2},
                "drop": True,
                "latex": "c_4",
            }
        else:
            # Should not reach here due to validation, but provide defaults for type checker
            raise ValueError(f"Unknown potential: {potential}")

        # Attractor-specific prior overrides: restrict parameter space to
        # where a genuine tracking attractor exists (|lambda_eff| > 2).
        if is_attractor:
            if potential == "hyperbolic":
                # lambda_eff = -c2, need |c2| > 2 for tracking attractor
                attractor_prior = {
                    "prior": {
                        "hyperbolic_attractor": "lambda scf_c2: 0.0 if abs(scf_c2) > 2.0 else -np.inf"
                    }
                }

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
        cdm_c = {
            "prior": {"min": -18, "max": 18},
            "ref": {"dist": "norm", "loc": 0, "scale": 3},
            "latex": "c_\\mathrm{DM}",
        }
        if is_attractor:
            scf_phi_ini = {"value": 0.001, "drop": True, "latex": "\\phi_\\mathrm{ini}"}
            scf_phi_prime_ini = {
                "value": 0.1,
                "drop": True,
                "latex": "\\phi\\prime_\\mathrm{ini}",
            }
        else:  # non-attractor initial conditions
            # phi'_ini is irrelevant: Hubble friction damps it by a^{-2}
            # from a_ini ~ 1e-14, so any initial velocity is erased long
            # before dark energy matters. Fix phi'_ini = 0.
            scf_phi_prime_ini = {
                "value": 0.0,
                "drop": True,
                "latex": "\\phi\\prime_\\mathrm{ini}",
            }
            if potential == "exponential":
                # Exponential: shift symmetry in V = c1*exp(-c2*phi) means
                # phi_ini only matters through the coupling sigmoid
                # f(phi) = 1/(1+exp(2*cdm_c*phi)).  The natural variable
                # is psi_ini = cdm_c * phi_ini; the sigmoid transition
                # occupies psi ~ [-3, 3] and saturates by |psi| ~ 5.
                # psi_ini dict is built in the "Build iDM parameter block" below.
                scf_phi_ini = {
                    "derived": "lambda psi_ini, cdm_c: psi_ini / cdm_c if abs(cdm_c) > 1e-6 else 0.0",
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
                # Reject |cdm_c| < 1e-6 to match the psi_ini/cdm_c guard.
                # At |cdm_c| < 1e-6 the sigmoid is constant and psi_ini
                # is degenerate, creating a flat volume in the posterior.
                cdm_c_floor_prior = {
                    "prior": {
                        "cdm_c_floor": "lambda cdm_c: 0.0 if abs(cdm_c) > 1e-6 else -np.inf"
                    }
                }
            elif potential in ("power-law", "SqE"):
                # Power-law / SqE: no shift symmetry.
                # phi_ini matters because slow-roll parameters depend on
                # c2 and phi independently. phi must be > 0
                # (phi^c2 / phi^(-c2) undefined for phi<0 with non-integer c2).
                scf_phi_ini = {
                    "prior": {"dist": "loguniform", "a": 0.1, "b": 100.0},
                    "ref": {"dist": "norm", "loc": 1, "scale": 0.5},
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential in ("Bean", "BeanSingleWell"):
                # Bean single-well: psi_ini = c3 * phi_ini normalizes to
                # the exponential decay length, matching exploration of
                # both the quadratic well and exponential tail for all c3.
                # psi_ini dict is built in the "Build iDM parameter block" below.
                scf_phi_ini = {
                    "derived": "lambda psi_ini, scf_c3: psi_ini / scf_c3",
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential == "BeanAdS":
                # Bean double-well (AdS): sample phi_ini directly.
                # The well feature at phi ~ c4 must be resolved, so
                # the range is matched to the well width sqrt(|c2|) <= 10.
                scf_phi_ini = {
                    "prior": {"min": -10.0, "max": 10.0},
                    "ref": {"dist": "norm", "loc": 0, "scale": 2},
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential == "DoubleExp":
                # DoubleExp: phi_ini sampled directly.
                # Field rolls O(10) in Planck units during DE era;
                # attractor-mode phi ~ 1e63 is irrelevant here.
                scf_phi_ini = {
                    "prior": {"min": -10.0, "max": 10.0},
                    "ref": {"dist": "norm", "loc": 0, "scale": 2},
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential == "cosine":
                # Cosine: V = c1*cos(c2*phi).  V > 0 requires |c2*phi| < pi/2.
                # The natural variable is xi_ini = c2 * phi_ini (argument
                # of cosine).  Shooting fails around xi ~ 0.8 (V too small).
                # xi_ini in [0, 0.7] covers up to ~3% Cl effect.
                # V is even in phi; cdm_c in [-18,18] covers coupling sign.
                # xi_ini dict is built in the "Build iDM parameter block" below.
                scf_phi_ini = {
                    "derived": "lambda xi_ini, scf_c2: xi_ini / scf_c2",
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential == "pNG":
                # pNG: V = c1^4*(1+cos(phi/c2)).  V >= 0 always, max at
                # phi=0.  The natural variable is xi_ini = phi_ini / c2
                # (argument of cosine).  V = 0 at xi = pi; shooting fails
                # around xi ~ 2 (V/Vmax ~ 29%).  xi_ini in [0, 1.5] gives
                # V/Vmax >= 54% and up to ~8.5% Cl effect.
                # V is even in phi; cdm_c in [-18,18] covers coupling sign.
                # xi_ini dict is built in the "Build iDM parameter block" below.
                scf_phi_ini = {
                    "derived": "lambda xi_ini, scf_c2: xi_ini * scf_c2",
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
            elif potential == "hyperbolic":
                # Hyperbolic: V = c1*[1-tanh(c2*phi)].
                # chi_ini = c2*phi_ini is the natural variable for the
                # tanh transition, but coupling f(phi) depends on phi
                # directly.  Sample chi_ini in [-5, 5] and derive phi.
                scf_phi_ini = {
                    "derived": "lambda chi_ini, scf_c2: chi_ini / scf_c2 if abs(scf_c2) > 1e-6 else 0.0",
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }
                # Reject |scf_c2| < 1e-6 to match the chi_ini/scf_c2 guard.
                # At |scf_c2| < 1e-6 the potential is constant (V = c1)
                # and chi_ini is degenerate, creating a flat volume.
                scf_c2_floor_prior = {
                    "prior": {
                        "scf_c2_floor": "lambda scf_c2: 0.0 if abs(scf_c2) > 1e-6 else -np.inf"
                    }
                }
            else:
                # Fallback: fix phi_ini = 1.  All 9 potentials have
                # explicit branches above; this only triggers if a new
                # potential is added without defining its own IC.
                scf_phi_ini = {
                    "value": 1.0,
                    "drop": True,
                    "latex": "\\phi_\\mathrm{ini}",
                }

        # ── Analytical c1 initial guess for non-attractor mode ──────────
        #
        # The CLASS shooter tunes c1 so that rho_scf(z=0) matches
        # Omega_scf.  For a slow-rolling field, rho_scf ≈ V(phi)/3, so
        # an educated first bracket for the shooter is
        #
        #   c1_guess = [ V_target / g(phi_ini) ]^(1/p)
        #
        # where V_target = 3 * Omega_scf * H0^2  (≈ 1.04e-7 Mpc^-2 for
        # Planck 2018), g() is the non-c1 factor of V at phi_ini, and p
        # is the power-law exponent of c1 in V.
        #
        # The formula is approximate (the field rolls, so phi_today ≠
        # phi_ini) but typically lands within ~0.3 dex of the true c1.
        # This replaces the old hardcoded defaults (often 5-7 decades
        # off) and saves ~25 % of shooting iterations.
        #
        # V_target is nearly constant across the posterior (H0 and
        # Omega_scf vary by <10 %), so we fix it at the Planck 2018
        # best-fit.  The shooter corrects the residual exactly.
        #
        # NOTE: for attractor-mode IC the field quickly forgets phi_ini,
        # so the formula has no anchoring advantage; we keep the original
        # hardcoded values there.
        if not is_attractor:
            # V_target = 3 * Omega_scf * (H0/c)^2
            # h = 0.6736, Omega_scf ≈ 0.685, c = 299792.458 km/s
            _V_T = 3.0 * 0.685 * (67.36 / 299792.458) ** 2  # ≈ 1.037e-7

            if potential == "cosine":
                # V = c1*cos(c2*phi) = c1*cos(xi_ini)  [xi = c2*phi]
                scf_c1 = {
                    "derived": f"lambda xi_ini: {_V_T} / np.cos(xi_ini)"
                    f" if np.cos(xi_ini) > 0.01 else 1e-7",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "hyperbolic":
                # V = c1*[1-tanh(c2*phi)] = c1*[1-tanh(chi_ini)]
                # Guard chi_ini < 15 to avoid 1-tanh(x) losing all digits.
                scf_c1 = {
                    "derived": f"lambda chi_ini: {_V_T} / (1 - np.tanh(chi_ini))"
                    f" if chi_ini < 15 else 1e-8",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "pNG":
                # V = c1^4*(1+cos(phi/c2)) = c1^4*(1+cos(xi_ini))
                scf_c1 = {
                    "derived": f"lambda xi_ini: ({_V_T} / (1 + np.cos(xi_ini)))**0.25"
                    f" if (1 + np.cos(xi_ini)) > 0.01 else 1e-1",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "exponential":
                # V = c1*exp(-c2*phi) → c1 = V_T*exp(c2*phi)
                # Rewrite as multiplication to avoid 1/exp → 0 division.
                # Clamp exponent to [-500, 500] against overflow.
                scf_c1 = {
                    "derived": f"lambda psi_ini, scf_c2, cdm_c: {_V_T}"
                    f" * np.exp(np.clip(scf_c2 * psi_ini / cdm_c, -500, 500))"
                    f" if abs(cdm_c) > 1e-6 else 1e-7",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential in ("Bean", "BeanSingleWell"):
                # V = c1*[(c4-phi)^2+c2]*exp(-c3*phi)
                # phi = psi_ini/c3, so exp(-c3*phi) = exp(-psi_ini)
                # Clamp result to [1e-15, 1e5] against tiny/huge denominators.
                scf_c1 = {
                    "derived": f"lambda psi_ini, scf_c2, scf_c3, scf_c4:"
                    f" np.clip({_V_T}"
                    f" / (max(((scf_c4 - psi_ini/scf_c3)**2 + scf_c2), 1e-30)"
                    f" * np.exp(np.clip(-psi_ini, -500, 500))), 1e-15, 1e5)",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "BeanAdS":
                # V = c1*[(c4-phi)^2+c2]*exp(-c3*phi), phi sampled directly
                # Clamp result to [1e-15, 1e5]; clamp exp argument.
                scf_c1 = {
                    "derived": f"lambda scf_phi_ini, scf_c2, scf_c3, scf_c4:"
                    f" np.clip({_V_T}"
                    f" / (max(((scf_c4 - scf_phi_ini)**2 + scf_c2), 1e-30)"
                    f" * np.exp(np.clip(-scf_c3 * scf_phi_ini, -500, 500))),"
                    f" 1e-15, 1e5)",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "DoubleExp":
                # V = c1*(exp(-c2*phi)+c3*exp(-c4*phi)), phi sampled directly
                # Clamp exp arguments and result against overflow.
                scf_c1 = {
                    "derived": f"lambda scf_phi_ini, scf_c2, scf_c3, scf_c4:"
                    f" np.clip({_V_T}"
                    f" / (np.exp(np.clip(-scf_c2 * scf_phi_ini, -500, 500))"
                    f" + scf_c3 * np.exp(np.clip(-scf_c4 * scf_phi_ini, -500, 500))),"
                    f" 1e-15, 1e5)",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "power-law":
                # V = c1^(4-c2)*phi^c2  (c3=0)
                # c1 = (V_t / phi^c2)^(1/(4-c2))
                scf_c1 = {
                    "derived": f"lambda scf_phi_ini, scf_c2:"
                    f" ({_V_T} / scf_phi_ini**scf_c2)**(1.0/(4.0 - scf_c2))"
                    f" if abs(4 - scf_c2) > 0.01 and scf_phi_ini > 0 else 1e-2",
                    "drop": True,
                    "latex": "c_1",
                }
            elif potential == "SqE":
                # V ≈ c1^(c2+4)*phi^(-c2)  (exp(c1*phi^2) ≈ 1 for small c1)
                # c1 = (V_t * phi^c2)^(1/(c2+4))
                scf_c1 = {
                    "derived": f"lambda scf_phi_ini, scf_c2:"
                    f" ({_V_T} * scf_phi_ini**scf_c2)**(1.0/(scf_c2 + 4.0))"
                    f" if (scf_c2 + 4) > 0.01 and scf_phi_ini > 0 else 1e-2",
                    "drop": True,
                    "latex": "c_1",
                }

        # Build iDM parameter block
        parameters_iDM_ordered: dict[str, Any] = {}
        if not is_attractor:
            if potential in ("Bean", "BeanSingleWell"):
                psi_ini: dict[str, Any] = {
                    "prior": {"min": -10.0, "max": 10.0},
                    "ref": {"dist": "norm", "loc": 0, "scale": 2},
                    "drop": True,
                    "latex": "\\psi_\\mathrm{ini}",
                }
                parameters_iDM_ordered["psi_ini"] = psi_ini
            elif potential == "exponential":
                psi_ini_exp: dict[str, Any] = {
                    "prior": {"min": -5.0, "max": 5.0},
                    "ref": {"dist": "norm", "loc": 0, "scale": 1.5},
                    "drop": True,
                    "latex": "\\psi_\\mathrm{ini}",
                }
                parameters_iDM_ordered["psi_ini"] = psi_ini_exp
            elif potential == "cosine":
                xi_ini_cos: dict[str, Any] = {
                    "prior": {"min": 0.0, "max": 0.7},
                    "ref": {"dist": "norm", "loc": 0.1, "scale": 0.1},
                    "drop": True,
                    "latex": "\\xi_\\mathrm{ini}",
                }
                parameters_iDM_ordered["xi_ini"] = xi_ini_cos
            elif potential == "pNG":
                xi_ini_pNG: dict[str, Any] = {
                    "prior": {"min": 0.0, "max": 1.5},
                    "ref": {"dist": "norm", "loc": 0.5, "scale": 0.3},
                    "drop": True,
                    "latex": "\\xi_\\mathrm{ini}",
                }
                parameters_iDM_ordered["xi_ini"] = xi_ini_pNG
            elif potential == "hyperbolic":
                chi_ini: dict[str, Any] = {
                    "prior": {"min": -5.0, "max": 5.0},
                    "ref": {"dist": "norm", "loc": 0, "scale": 1.5},
                    "drop": True,
                    "latex": "\\chi_\\mathrm{ini}",
                }
                parameters_iDM_ordered["chi_ini"] = chi_ini
        parameters_iDM_ordered["cdm_c"] = cdm_c
        parameters_iDM_ordered["scf_c1"] = scf_c1
        parameters_iDM_ordered["scf_c2"] = scf_c2
        parameters_iDM_ordered["scf_c3"] = scf_c3
        parameters_iDM_ordered["scf_c4"] = scf_c4
        parameters_iDM_ordered["scf_q1"] = scf_q1
        parameters_iDM_ordered["scf_q2"] = scf_q2
        parameters_iDM_ordered["scf_q3"] = scf_q3
        parameters_iDM_ordered["scf_q4"] = scf_q4
        parameters_iDM_ordered["scf_exp1"] = scf_exp1
        parameters_iDM_ordered["scf_exp2"] = scf_exp2
        parameters_iDM_ordered["scf_phi_ini"] = scf_phi_ini
        parameters_iDM_ordered["scf_phi_prime_ini"] = scf_phi_prime_ini

        # scf_parameters lambda: assembles the comma-separated string for CLASS.
        # In non-attractor mode scf_c1 is a derived parameter (computed via
        # lambda), so Cobaya won't pass it as an argument to another lambda.
        # We must inline the scf_c1 formula directly into the scf_parameters
        # lambda for every non-attractor potential.
        # In attractor mode scf_c1 has a fixed value and can be referenced
        # normally (the final else branch).
        if not is_attractor and potential in ("Bean", "BeanSingleWell"):
            # Bean: phi_ini = psi_ini / scf_c3
            # c1 = clip(V_T / [((c4-phi)^2+c2)*exp(-psi_ini)], 1e-15, 1e5)
            scf_parameters_entry: dict[str, Any] = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,psi_ini,scf_phi_prime_ini: ",".join([str(np.clip({_V_T} / (max(((scf_c4 - psi_ini/scf_c3)**2 + scf_c2), 1e-30) * np.exp(np.clip(-psi_ini, -500, 500))), 1e-15, 1e5)),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(psi_ini/scf_c3),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "exponential":
            # Exponential: phi_ini = psi_ini / cdm_c (guard: cdm_c ~ 0 → phi = 0)
            # c1 = V_T * exp(clip(c2*psi_ini/cdm_c, -500, 500))
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,psi_ini,scf_phi_prime_ini,cdm_c: ",".join([str({_V_T} * np.exp(np.clip(scf_c2 * psi_ini / cdm_c, -500, 500)) if abs(cdm_c) > 1e-6 else 1e-7),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(psi_ini/cdm_c if abs(cdm_c)>1e-6 else 0.0),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "cosine":
            # Cosine: phi_ini = xi_ini / scf_c2
            # c1 = V_T / cos(xi_ini)
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,xi_ini,scf_phi_prime_ini: ",".join([str({_V_T} / np.cos(xi_ini) if np.cos(xi_ini) > 0.01 else 1e-7),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(xi_ini/scf_c2),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "pNG":
            # pNG: phi_ini = xi_ini * scf_c2
            # c1 = (V_T / (1+cos(xi_ini)))^0.25
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,xi_ini,scf_phi_prime_ini: ",".join([str(({_V_T} / (1 + np.cos(xi_ini)))**0.25 if (1 + np.cos(xi_ini)) > 0.01 else 1e-1),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(xi_ini*scf_c2),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "hyperbolic":
            # Hyperbolic: phi_ini = chi_ini / scf_c2
            # c1 = V_T / (1-tanh(chi_ini))
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,chi_ini,scf_phi_prime_ini: ",".join([str({_V_T} / (1 - np.tanh(chi_ini)) if chi_ini < 15 else 1e-8),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(chi_ini/scf_c2 if abs(scf_c2)>1e-6 else 0.0),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "BeanAdS":
            # BeanAdS: phi_ini sampled directly
            # c1 = clip(V_T / [((c4-phi)^2+c2)*exp(-c3*phi)], 1e-15, 1e5)
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(np.clip({_V_T} / (max(((scf_c4 - scf_phi_ini)**2 + scf_c2), 1e-30) * np.exp(np.clip(-scf_c3 * scf_phi_ini, -500, 500))), 1e-15, 1e5)),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "DoubleExp":
            # DoubleExp: phi_ini sampled directly
            # c1 = clip(V_T / (exp(-c2*phi)+c3*exp(-c4*phi)), 1e-15, 1e5)
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(np.clip({_V_T} / (np.exp(np.clip(-scf_c2 * scf_phi_ini, -500, 500)) + scf_c3 * np.exp(np.clip(-scf_c4 * scf_phi_ini, -500, 500))), 1e-15, 1e5)),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "power-law":
            # power-law: phi_ini sampled directly (loguniform)
            # c1 = (V_T / phi^c2)^(1/(4-c2))
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(({_V_T} / scf_phi_ini**scf_c2)**(1.0/(4.0 - scf_c2)) if abs(4 - scf_c2) > 0.01 and scf_phi_ini > 0 else 1e-2),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        elif not is_attractor and potential == "SqE":
            # SqE: phi_ini sampled directly (loguniform)
            # c1 = (V_T * phi^c2)^(1/(c2+4))
            scf_parameters_entry = {
                "value": f'lambda scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(({_V_T} * scf_phi_ini**scf_c2)**(1.0/(scf_c2 + 4.0)) if (scf_c2 + 4) > 0.01 and scf_phi_ini > 0 else 1e-2),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        else:
            # Attractor mode: scf_c1 has a fixed value and can be referenced directly
            scf_parameters_entry = {
                "value": 'lambda scf_c1,scf_c2,scf_c3,scf_c4,scf_q1,scf_q2,scf_q3,scf_q4,scf_exp1,scf_exp2,scf_phi_ini,scf_phi_prime_ini: ",".join([str(scf_c1),str(scf_c2),str(scf_c3),str(scf_c4),str(scf_q1),str(scf_q2),str(scf_q3),str(scf_q4),str(scf_exp1),str(scf_exp2),str(scf_phi_ini),str(scf_phi_prime_ini)])',
                "derived": False,
            }
        parameters_iDM_ordered["scf_parameters"] = scf_parameters_entry

        parameters_iDM = {
            **parameters_iDM_ordered,
            "Omega_fld": 0.00,
            "Omega_scf": {"value": -0.7, "latex": "\\Omega_\\phi"},
            "Omega_Lambda": {"value": 0.0, "latex": "\\Omega_\\Lambda"},
            **swampland_params,
        }

    # Combine all parameters
    params: dict[str, Any] = parameters_base.copy()
    if not is_lcdm:
        params.update(parameters_iDM)
    if has_spt_candl:
        params.update(parameters_spt_candl)
    # For Planck runs with theta_s_100 sampled, use it instead of H0
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
            "N_ncdm": 1,
            "N_ur": 2.046,
            "sBBN file": "sBBN_2017.dat",
            "non linear": "halofit",
            # "hmcode_version": 2020,
            "model_cdm": "i",
            "scf_tuning_index": 0,
            "scf_potential": (
                "Bean" if potential in ("BeanSingleWell", "BeanAdS") else potential
            ),
            "attractor_ic_scf": attractor.lower(),
        }

    if not skip_nonlinear_min_k_max:
        extra_args["nonlinear_min_k_max"] = 25

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
                    "output_params": list(swampland_params.keys()),
                },
            },
            "params": swampland_params,
        },
    }

    if is_postprocessing:
        # Post-processing configuration
        if likelihood == "Run3_Planck_PP_SH0ES_DESIDR2":
            # Run 3 is a special post-processing run that adds SN+BAO to Run 1 chains
            run1_chain_stem = build_chain_output_stem(
                potential, "Run1_Planck_2018", attractor, coupling, sampler
            )
            config["output"] = f"/project/home/p201176/{run1_chain_stem}"
            config["post"] = {
                "suffix": "SN_BAO",
                "add": LIKELIHOODS[likelihood],  # Adds the new likelihoods
            }
        else:
            # post_mcmc or post_polychord samplers (swampland post-processing)
            # Output points to the base sampler chain
            base_chain_stem = build_chain_output_stem(
                potential, likelihood, attractor, coupling, sampler
            )
            config["output"] = f"/project/home/p201176/{base_chain_stem}"
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
        if not is_lcdm and attractor_prior:
            # Merge attractor prior into existing prior block or create new one
            if "prior" in config:
                config["prior"].update(attractor_prior["prior"])
            else:
                config.update(attractor_prior)
        if not is_lcdm and cdm_c_floor_prior:
            if "prior" in config:
                config["prior"].update(cdm_c_floor_prior["prior"])
            else:
                config.update(cdm_c_floor_prior)
        if not is_lcdm and scf_c2_floor_prior:
            if "prior" in config:
                config["prior"].update(scf_c2_floor_prior["prior"])
            else:
                config.update(scf_c2_floor_prior)
        if not is_lcdm and potential == "DoubleExp":
            # Exchange symmetry constraint: (c2,c4) <-> (c4,c2) is exact.
            # Break by requiring c4 >= c2 when both are non-negative.
            # c4 < 0 (mixed-sign regime) is freely allowed.
            doubleexp_exchange = {
                "doubleexp_exchange": "lambda scf_c2, scf_c4: 0.0 if (scf_c4 < 0 or scf_c4 >= scf_c2) else -np.inf"
            }
            if "prior" in config:
                config["prior"].update(doubleexp_exchange)
            else:
                config["prior"] = doubleexp_exchange
        config.update(theorycode)  # Add "theory"

    # Add packages_path at top level
    config["packages_path"] = "/home/users/u103677/cobaya_packages_2026"

    return config


def main(
    sampler: str = "mcmc_fast",
    likelihood: str = "CMB",
    potential: str = "DoubleExp",
    attractor: str = "no",
    coupling: str = "uncoupled",
) -> list[str]:
    """
    Generate Cobaya YAML configuration files and SLURM scripts.

    Parameters:
        sampler: MCMC(_fast), minimizer_*, post_*, or Polychord.
        likelihood: Likelihood combination (e.g. 'CMB', 'Run1_Planck_2018', etc.).
        potential: Options: 'LCDM', 'power-law', 'cosine', 'hyperbolic', 'pNG',
                   'SqE', 'exponential', 'Bean', 'BeanSingleWell', 'BeanAdS', 'DoubleExp'.
        attractor: 'yes' or 'no'. Ignored for LCDM.
        coupling: 'uncoupled' or 'coupled'. Ignored for LCDM.

    Returns:
        List of generated YAML filenames.
    """
    # Determine which variants to generate
    # 'Bean' generates both single-well (Bean) and double-well (BeanAdS) configs
    _variants: list[str] = ["Bean", "BeanAdS"] if potential == "Bean" else [potential]
    _filenames: list[str] = []

    for _variant in _variants:
        _config = create_cobaya_yaml(sampler, likelihood, _variant, attractor, coupling)

        _is_postprocessing_run: bool = sampler.startswith("post_")

        _filename: str = (
            build_filename_stem(_variant, likelihood, attractor, coupling, sampler)
            + ".yml"
        )
        _filenames.append(_filename)

        # Specify the output path in the YAML file
        # For post-processing runs (Run 3 or post_* samplers), output is already set in the config
        if likelihood != "Run3_Planck_PP_SH0ES_DESIDR2" and not _is_postprocessing_run:
            _output_stem = build_chain_output_stem(
                _variant, likelihood, attractor, coupling, sampler
            )
            _config["output"] = "/project/home/p201176/" + _output_stem

        # Writing nested data to a YAML file
        _yaml_path = f"Cobaya/MCMC/{_filename}"
        try:
            with open(_yaml_path, "w") as file:
                yaml.dump(_config, file)  # type: ignore[misc]

            # Post-process to use bracket notation for z and R lists (used by sigma_R())
            # Standard YAML serializes lists with dashes, but these need inline bracket notation
            with open(_yaml_path, "r") as file:
                content = file.read()
            content = re.sub(r"(\s+)z:\s*\n\s*-\s*([0-9.]+)", r"\1z: [\2]", content)
            content = re.sub(r"(\s+)R:\s*\n\s*-\s*([0-9.]+)", r"\1R: [\2]", content)
            with open(_yaml_path, "w") as file:
                file.write(content)
        except IOError as e:
            print(f"Error handling YAML file: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during YAML processing: {e}")
            raise

        print(f"cobaya configuration has been written to '{_yaml_path}'")

    # Create the SLURM test scripts
    for _filename in _filenames:
        slurm_script_filename: str = create_slurm_test_script(yaml_filename=_filename)
        print(f"SLURM test script has been written to '{slurm_script_filename}'")

    # Create the SLURM production run scripts (use minimize script for minimize sampler variants)
    for _filename in _filenames:
        slurm_run_script_filename: str
        if sampler.startswith("minimize_"):
            slurm_run_script_filename = create_slurm_minimize_script(
                yaml_filename=_filename
            )
        else:
            slurm_run_script_filename = create_slurm_run_script(yaml_filename=_filename)
        print(f"SLURM run script has been written to '{slurm_run_script_filename}'")

    return _filenames


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
    - yaml_filename (str): The YAML configuration filename (e.g., 'Bean_Planck_tracking_MCMC.yml').
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
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

# Number of OpenMP threads
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
    - yaml_filename (str): The YAML configuration filename (e.g., 'Bean_Planck_tracking_MCMC.yml').
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
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

# Number of OpenMP threads
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
    - yaml_filename (str): The YAML configuration filename (e.g., 'hyperbolic_Planck_tracking_MCMC_minimizer.yml').
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
module load Python foss
#Activate Python virtual environment
source my_2025-env/bin/activate

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


if __name__ == "__main__":
    main()
