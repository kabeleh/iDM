# %%
# Import required libraries:
# - matplotlib.pyplot for plotting
# - cmcrameri.cm for perceptually uniform colormaps
# - getdist.plots for MCMC chain plotting
from typing import Any, Callable, Mapping, Sequence, cast

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from matplotlib.patches import Patch
from cmcrameri import cm  # type: ignore[import-untyped]
from getdist import plots  # type: ignore[import-untyped]

plt = cast(Any, plt)
cm = cast(Any, cm)
plots = cast(Any, plots)

Color = Any

# ============================================================================
# COMMON CONFIGURATION
# ============================================================================

# Directory where MCMC chain files are stored
CHAIN_DIR: str = r"/Users/klmba/kDrive/Sci/PhD/Research/HDM/MCMCfast/"
ANALYSIS_SETTINGS: dict[str, float] = {"ignore_rows": 0.33}

# Define the root names of the MCMC chains (file prefixes without extensions)
ROOTS: list[str] = [
    # "Cobaya_mcmc_Run3_Planck_PP_SH0ES_DESIDR2_DoubleExp_tracking_uncoupled",
    # "cobaya_iDM_20251230_dexp",
    # "cobaya_mcmc_fast_Run1_Planck_2018_DoubleExp_tracking_uncoupled",
    # "cobaya_mcmc_fast_Run1_Planck_2018_hyperbolic_tracking_uncoupled",
    # "cobaya_mcmc_fast_Run1_Planck_2018_LCDM",
    # "cobaya_mcmc_Run2_PP_SH0ES_DESIDR2_hyperbolic_tracking_uncoupled",
    # "cobaya_mcmc_Run2_PP_SH0ES_DESIDR2_DoubleExp_tracking_uncoupled",
    # "cobaya_mcmc_Run2_PP_SH0ES_DESIDR2_LCDM",
    "cobaya_mcmc_CV_PP_DESI_DoubleExp_tracking_uncoupled",
    "cobaya_mcmc_CV_PP_DESI_hyperbolic_tracking_uncoupled",
    "cobaya_mcmc_CV_PP_S_DESI_DoubleExp_tracking_uncoupled",
    "cobaya_mcmc_CV_PP_S_DESI_LCDM",
    "cobaya_mcmc_CV_PP_S_DESI_hyperbolic_tracking_uncoupled",
    "cobaya_polychord_CV_PP_DESI_LCDM",
    # "cobaya_polychord_CV_PP_S_DESI_LCDM",
]

# Extract a list of colors from the categorical batlowKS colourmap
# Reserve indices 0, 1 for observational bands; 2+ for MCMC chains
ALL_COLOURS: list[Color] = [tuple(c) for c in cm.batlowKS.colors]
BAND_COLOURS: list[Color] = ALL_COLOURS[:2]  # colours[0] for H0, colours[1] for S8
CHAIN_COLOURS: list[Color] = ALL_COLOURS[2 : 2 + len(ROOTS)]  # consistent chain colours


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def make_triangle_plot(
    params: Sequence[str],
    annotations: Callable[[Any], list[Patch]] | None = None,
    param_labels: Mapping[str, str] | None = None,
    title: str | None = None,
) -> Any:
    """
    Create a triangle plot for the given parameters.

    Parameters
    ----------
    params : list of str
        Parameter names to plot.
    annotations : callable or None
        A function with signature `annotations(g) -> list[Patch]` that:
        - Receives the GetDistPlotter `g` after the triangle plot is drawn
        - Adds any desired annotations (bands, lines, shading, masks, etc.)
        - Returns a list of matplotlib Patch/Line2D handles for the legend
        If None, no extra annotations are added.
    param_labels : dict or None
        A dictionary mapping parameter names to custom LaTeX labels.
        E.g., {'cdm_c': r'$c_\text{DM}$', 'scf_c2': r'$c_2$'}
        If None, default labels from the chain files are used.
    title : str, optional
        Title for the figure.

    Returns
    -------
    g : getdist.plots.GetDistPlotter
        The plotter object.
    """
    g: Any = plots.get_subplot_plotter(  # type: ignore[misc]
        chain_dir=CHAIN_DIR,
        analysis_settings=ANALYSIS_SETTINGS,
    )

    # Load samples, drop roots with none of the requested params,
    # and keep only params present in all remaining roots.
    samples_by_root: list[tuple[str, Any]] = []
    for root in ROOTS:
        samples = g.sample_analyser.samples_for_root(root)
        if any(samples.paramNames.parWithName(p) is not None for p in params):
            samples_by_root.append((root, samples))

    if not samples_by_root:
        raise ValueError(
            "None of the requested parameters are available in the provided roots."
        )

    available_params: list[str] = [
        p
        for p in params
        if all(
            samples.paramNames.parWithName(p) is not None
            for _, samples in samples_by_root
        )
    ]

    if not available_params:
        raise ValueError(
            "No common parameters found across the selected roots; cannot plot."
        )

    # Apply custom labels if provided
    if param_labels:
        for _, samples in samples_by_root:
            for param_name, label in param_labels.items():
                p = samples.paramNames.parWithName(param_name)
                if p is not None:
                    p.label = label

    used_roots: list[str] = [root for root, _ in samples_by_root]
    roots_to_plot: Sequence[Any] = [samples for _, samples in samples_by_root]
    root_to_color: dict[str, Color] = {
        root: CHAIN_COLOURS[i] for i, root in enumerate(ROOTS) if root in used_roots
    }
    chain_colors: list[Color] = [root_to_color[root] for root in used_roots]

    # Generate the triangle plot
    g.triangle_plot(
        roots_to_plot,
        available_params,
        filled=True,
        colors=chain_colors,
        diag1d_kwargs={"colors": chain_colors},
        contour_lws=3,
        legend_loc="lower left",
        figure_legend_outside=True,
    )

    fig: Any = g.fig

    # Build legend handles for MCMC chains
    chain_handles: list[Patch] = [
        Patch(facecolor=root_to_color[root], label=root) for root in used_roots
    ]

    # Apply custom annotations and collect their legend handles
    annotation_handles: list[Patch] = []
    if annotations is not None:
        annotation_handles = annotations(g) or []

    all_handles: list[Patch] = chain_handles + annotation_handles
    all_labels: list[str] = [str(h.get_label()) for h in all_handles]

    # Remove any existing legends
    for legend in fig.legends:
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # Adjust layout: plot on left, make room for legend on right
    fig.subplots_adjust(left=0.1, right=0.6)

    # Position legend aligned to top-right of first subplot
    first_ax = g.subplots[0, 0]
    ax_bbox = first_ax.get_position()
    fig.legend(
        all_handles,
        all_labels,
        loc="upper left",
        bbox_to_anchor=(ax_bbox.x1, ax_bbox.y1 + 0.017),
        frameon=True,
    )

    if title:
        fig.suptitle(title, y=1.02)

    return g


# ============================================================================
# ANNOTATION FUNCTIONS
# ============================================================================


def annotate_H0_S8(g: Any) -> list[Patch]:
    """
    Add H0 (SH0ES) and S8 (KiDS-1000) observational bands.
    Returns legend handles for these annotations.
    """
    # SH0ES 2020b: H0 = 73.2 ± 1.3 km/s/Mpc
    g.add_x_bands(73.2, 1.3, ax=0, color=BAND_COLOURS[0])
    g.add_x_bands(73.2, 1.3, ax=2, color=BAND_COLOURS[0])

    # KiDS-1000 2023: S8 = 0.776 ± 0.031
    # CosmoVerse asks for 1/S8, so the error is d(1/S8)=dS8/S8^2
    g.add_x_bands(1.289, 0.051, ax=3, color=BAND_COLOURS[1])
    g.add_y_bands(1.289, 0.051, ax=2, color=BAND_COLOURS[1])

    return [
        Patch(facecolor=BAND_COLOURS[0], alpha=0.5, label=r"$H_0$ SH0ES 2020"),
        Patch(facecolor=BAND_COLOURS[1], label=r"$1/S_8$ KiDS-1000"),
    ]


def annotate_scf_constraints(g: Any) -> list[Patch]:
    """
    Add constraints for scalar field parameters.
    Customize this function to add lines, shaded regions, masks, etc.
    Returns legend handles for these annotations.

    Examples of what you can do here:
    - g.add_x_bands(value, sigma, ax=idx, color=...)  # vertical band
    - g.add_y_bands(value, sigma, ax=idx, color=...)  # horizontal band
    - ax = g.subplots[row, col]; ax.axvline(x, ...)   # vertical line
    - ax = g.subplots[row, col]; ax.axhline(y, ...)   # horizontal line
    - ax = g.subplots[row, col]; ax.fill_between(...) # shaded region
    - ax = g.subplots[row, col]; ax.fill_betweenx(...)# shaded region (vertical)
    """
    handles: list[Patch] = []

    # # c_2 should be of order 1. As a visual aid, we add a line at c_2=1 and another line at c_2=0.5
    # g.add_x_marker(1.0, ax=3, color="black", ls="--")
    # g.add_x_marker(0.5, ax=3, color="black", ls="--")

    # Example: add a vertical line constraint on cdm_c (param index 0, 1D plot at ax=0)
    # ax = g.subplots[0, 0]
    # line = ax.axvline(0.5, color=BAND_COLOURS[0], ls='--', lw=2, label='cdm_c limit')
    # handles.append(Line2D([0], [0], color=BAND_COLOURS[0], ls='--', lw=2, label='cdm_c limit'))

    # Example: shade excluded region on a 2D subplot
    # ax = g.subplots[1, 0]  # scf_c2 vs cdm_c
    # ax.fill_betweenx([y_min, y_max], x_excluded_min, x_excluded_max,
    #                  color='gray', alpha=0.3, label='Excluded')
    # handles.append(Patch(facecolor='gray', alpha=0.3, label='Excluded'))

    return handles


# ============================================================================
# PLOT 1: H0 & S8 with observational bands
# ============================================================================
# %%
params_cosmology = ["H0", "S8"]
g1 = make_triangle_plot(params_cosmology, annotations=annotate_H0_S8, title=None)

# Export example:
# g1.fig.savefig("plot_H0_S8.png", bbox_inches="tight", dpi=300)

plt.show()

# ============================================================================
# PLOT 2: Scalar field parameters with constraints
# ============================================================================
# %%
params_scf = ["cdm_c", "scf_c2", "scf_c3", "scf_c4"]
scf_labels = {
    "cdm_c": r"c_\mathrm{DM}",
    "scf_c2": r"c_2",
    "scf_c3": r"c_3",
    "scf_c4": r"c_4",
}
g2 = make_triangle_plot(
    params_scf,
    annotations=annotate_scf_constraints,
    param_labels=scf_labels,
    title=None,
)

# Export example:
# g2.fig.savefig("plot_scf_params.png", bbox_inches="tight", dpi=300)

plt.show()

# ============================================================================
# TABLE GENERATION: Extract data from chains and create LaTeX tables
# ============================================================================
# %%
import re
import os
import numpy as np
from getdist import MCSamples, loadMCSamples  # type: ignore[import-untyped]

MCSamples = cast(Any, MCSamples)
loadMCSamples = cast(Callable[..., Any], loadMCSamples)


def parse_minimum_file(filepath: str) -> dict[str, Any]:
    """
    Parse a .minimum file to extract best-fit values, chi-sq, and LaTeX labels.

    Parameters
    ----------
    filepath : str
        Path to the .minimum file.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'neg_log_like': float, -log(Like) value
        - 'chi_sq': float, total chi-squared value
        - 'chi_sq_components': dict mapping likelihood_name -> chi_sq value
        - 'params': dict mapping param_name -> {'value': float, 'latex': str}
    """
    result: dict[str, Any] = {
        "neg_log_like": None,
        "chi_sq": None,
        "chi_sq_components": {},
        "params": {},
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse -log(Like) line
        if line.startswith("-log(Like)"):
            match = re.search(r"-log\(Like\)\s*=\s*([\d.eE+-]+)", line)
            if match:
                result["neg_log_like"] = float(match.group(1))
            continue

        # Parse chi-sq line
        if line.startswith("chi-sq"):
            match = re.search(r"chi-sq\s*=\s*([\d.eE+-]+)", line)
            if match:
                result["chi_sq"] = float(match.group(1))
            continue

        # Parse parameter lines: index  value  name  latex_label
        # Format: "   14  7.820503228e-01   sigma8                       \sigma_8"
        match = re.match(r"\s*\d+\s+([\d.eE+-]+)\s+(\S+)\s+(.*)", line)
        if match:
            value = float(match.group(1))
            param_name = match.group(2)
            latex_label = match.group(3).strip()
            result["params"][param_name] = {"value": value, "latex": latex_label}

            # Track chi2 components (e.g., chi2__bao.desi_dr2, chi2__sn.pantheonplusshoes)
            if param_name.startswith("chi2__") and not param_name in [
                "chi2__BAO",
                "chi2__SN",
                "chi2__CMB",
            ]:
                # Extract the specific likelihood name (after chi2__)
                likelihood_name = param_name[6:]  # Remove "chi2__" prefix
                result["chi_sq_components"][likelihood_name] = value

    return result


def parse_cobaya_yaml(filepath: str) -> list[str]:
    """
    Parse a Cobaya .input.yaml or .updated.yaml file to extract likelihood names.

    Parameters
    ----------
    filepath : str
        Path to the YAML file.

    Returns
    -------
    list
        List of likelihood names used in the run.
    """
    try:
        import yaml

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        likelihoods: list[str] = []
        if "likelihood" in config and config["likelihood"]:
            likelihoods = list(config["likelihood"].keys())
        return likelihoods
    except Exception as e:
        print(f"Warning: Could not parse YAML file {filepath}: {e}")
        return []


def get_likelihoods_for_chain(root: str, chain_dir: str = CHAIN_DIR) -> list[str]:
    """
    Get the likelihoods used in a chain, trying multiple sources.

    Tries in order:
    1. Parse .input.yaml file for likelihood section (preferred)
    2. Parse .minimum file for chi2__ components (fallback)

    Parameters
    ----------
    root : str
        Chain root name.
    chain_dir : str
        Directory containing chain files.

    Returns
    -------
    list
        List of likelihood names.
    """
    likelihoods: list[str] = []

    # Try YAML file first (more reliable)
    for yaml_suffix in [".input.yaml", ".updated.yaml"]:
        yaml_file = os.path.join(chain_dir, f"{root}{yaml_suffix}")
        if os.path.exists(yaml_file):
            likelihoods = parse_cobaya_yaml(yaml_file)
            if likelihoods:
                break

    # If no likelihoods found in YAML, try .minimum file as fallback
    if not likelihoods:
        minimum_file = os.path.join(chain_dir, f"{root}.minimum")
        if os.path.exists(minimum_file):
            min_data = parse_minimum_file(minimum_file)
            if min_data.get("chi_sq_components"):
                likelihoods = list(min_data["chi_sq_components"].keys())

    return likelihoods


# Lookup table for number of data points per likelihood
# These are the effective number of data points used in the chi-squared calculation
# The values are extracted from the covmat and mean.txt files from the cobaya data folder.
LIKELIHOOD_DATA_POINTS: dict[str, int] = {
    "bao.desi_dr2": 13,  # DESI DR2 BAO (13 data points)
    "sn.pantheonplus": 1701,  # Pantheon+ without SH0ES (1701 SNe)
    "sn.pantheonplusshoes": 1701,  # Pantheon+ with SH0ES calibration (1701 SNe + SH0ES prior)
    "planck_2018_lowl.TT": 28,  # Low-l TT (l=2-29)
    "act_dr6_cmbonly.PlanckActCut": 613,  # Combined Planck+ACT DR6 CMB-only (613 data points)
    "act_dr6_cmbonly.ACTDR6CMBonly": 135,  # ACT DR6 CMB-only (135 data points)
    "act_dr6_lenslike.ACTDR6LensLike": 19,  # ACT DR6 lensing-like (19 data points)
    "spt3g_d1_tne": 196,  # SPT-3G D1 temperature and E-mode polarization (196 data points)
    "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": 16,  # Since only phi phi component is used.
}


def estimate_n_data_from_likelihoods(
    chi_sq_components: Mapping[str, float] | None,
) -> int | None:
    """
    Estimate the total number of data points from the chi-squared components.

    Parameters
    ----------
    chi_sq_components : dict
        Dictionary mapping likelihood names to chi-squared values.

    Returns
    -------
    int or None
        Estimated number of data points, or None if unknown likelihoods.
    """
    if not chi_sq_components:
        return None

    total_n_data: int = 0
    unknown_likelihoods: list[str] = []

    for likelihood in chi_sq_components.keys():
        # Normalize likelihood name (lowercase, handle variations)
        likelihood_lower = likelihood.lower()

        # Try exact match first
        if likelihood_lower in LIKELIHOOD_DATA_POINTS:
            total_n_data += LIKELIHOOD_DATA_POINTS[likelihood_lower]
        else:
            # Try partial matching for common patterns
            matched = False
            for known_like, n_data in LIKELIHOOD_DATA_POINTS.items():
                if known_like in likelihood_lower or likelihood_lower in known_like:
                    total_n_data += n_data
                    matched = True
                    break

            if not matched:
                unknown_likelihoods.append(likelihood)

    if unknown_likelihoods:
        print(
            f"Warning: Unknown likelihoods (data points not counted): {unknown_likelihoods}"
        )

    return total_n_data if total_n_data > 0 else None


def get_chain_statistics(
    root: str,
    params: Sequence[str],
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> dict[str, Any]:
    """
    Get mean, std, and confidence limits for parameters from a chain.

    Parameters
    ----------
    root : str
        Chain root name.
    params : list of str
        Parameter names to extract.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings (e.g., burn-in).

    Returns
    -------
    dict
        Dictionary mapping param_name -> {
            'mean': float,
            'std': float,
            'lower_1sigma': float,
            'upper_1sigma': float,
            'lower_2sigma': float,
            'upper_2sigma': float
        }
    """
    samples: Any = loadMCSamples(os.path.join(chain_dir, root), settings=settings)
    stats: dict[str, Any] = {}

    for param in params:
        p = samples.paramNames.parWithName(param)
        if p is None:
            stats[param] = None
            continue

        # Get 1D marginalized statistics
        mean = samples.mean(param)
        std = samples.std(param)

        # Get confidence limits using twoTailLimits (68% = 1sigma, 95% = 2sigma)
        # twoTailLimits returns (lower, upper) for symmetric tail probability
        marge = samples.getMargeStats()
        par_marge = marge.parWithName(param)

        if par_marge is not None:
            # Get limits from marginalized statistics
            lim_68 = par_marge.limits[0]  # 68% limits (index 0)
            lim_95 = par_marge.limits[1]  # 95% limits (index 1)
            lower_1, upper_1 = lim_68.lower, lim_68.upper
            lower_2, upper_2 = lim_95.lower, lim_95.upper
        else:
            # Fallback: approximate from mean ± n*sigma
            lower_1, upper_1 = mean - std, mean + std
            lower_2, upper_2 = mean - 2 * std, mean + 2 * std

        stats[param] = {
            "mean": mean,
            "std": std,
            "lower_1sigma": lower_1,
            "upper_1sigma": upper_1,
            "lower_2sigma": lower_2,
            "upper_2sigma": upper_2,
        }

    return stats


def count_free_parameters(
    root: str,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> int:
    """
    Count the number of free (sampled) parameters in a chain.

    Parameters
    ----------
    root : str
        Chain root name.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.

    Returns
    -------
    int
        Number of free parameters.
    """
    samples: Any = loadMCSamples(os.path.join(chain_dir, root), settings=settings)
    # paramNames.names includes all parameters; we want only sampled ones
    # The .paramNames.list() returns sampled params, .paramNames.numberOfName() for derived
    return len([p for p in samples.paramNames.names if not p.isDerived])


def compute_aic_bic(
    chi_sq: float, n_params: int, n_data: int | None = None
) -> tuple[float, float | None]:
    """
    Compute AIC and BIC.

    Parameters
    ----------
    chi_sq : float
        Chi-squared value.
    n_params : int
        Number of free parameters.
    n_data : int, optional
        Number of data points (required for BIC).

    Returns
    -------
    tuple
        (AIC, BIC) where BIC is None if n_data not provided.
    """
    aic = chi_sq + 2 * n_params
    bic = chi_sq + n_params * np.log(n_data) if n_data is not None else None
    return aic, bic


def identify_dataset_from_root(root: str) -> tuple[str, str, bool]:
    """
    Identify the dataset combination from a chain root name.

    Returns a tuple (dataset_key, model_name, is_lcdm) for grouping chains by dataset.
    """
    root_lower = root.lower()

    # Identify dataset components
    has_pp = "pp" in root_lower or "pantheon" in root_lower
    has_sh0es = "sh0es" in root_lower or "shoes" in root_lower or "_s_" in root_lower
    has_desi = "desi" in root_lower
    has_planck = "planck" in root_lower

    # Build dataset key
    parts: list[str] = []
    if has_planck:
        parts.append("Planck")
    if has_pp:
        parts.append("PP")
    if has_sh0es:
        parts.append("SH0ES")
    if has_desi:
        parts.append("DESI")

    dataset_key = "+".join(parts) if parts else "Unknown"

    # Identify model type and get LaTeX name
    is_lcdm = "lcdm" in root_lower
    if is_lcdm:
        model_name = r"\lcdm"
    elif "doubleexp" in root_lower or "dexp" in root_lower:
        model_name = r"\Nref{pot:dexp}"
    elif "hyperbolic" in root_lower or "tanh" in root_lower:
        model_name = r"\Nref{pot:tanh}"
    else:
        # Fallback: use shortened root name
        model_name = (
            root.replace("cobaya_", "")
            .replace("mcmc_", "")
            .replace("polychord_", "")
            .replace("_", r"\_")
        )

    return dataset_key, model_name, is_lcdm


# LaTeX labels for parameters (used in table headers and rows)
PARAM_LATEX_LABELS: dict[str, str] = {
    "H0": r"H_0",
    "S8": r"S_8",
    "Omega_m": r"\Omega_\mathrm{m}",
    "sigma8": r"\sigma_8",
    "omega_b": r"\Omega_\mathrm{b} h^2",
    "omega_cdm": r"\Omega_\mathrm{c} h^2",
    "n_s": r"n_\mathrm{s}",
    "logA": r"\log(10^{10} A_\mathrm{s})",
    "tau_reio": r"\tau_\mathrm{reio}",
    "cdm_c": r"\mathfrak{c}_{\textnormal{\gls{dm}}}",
    "scf_c2": r"\mathfrak{c}_2",
    "scf_c3": r"\mathfrak{c}_3",
    "scf_c4": r"\mathfrak{c}_4",
}


def format_value_with_errors(
    mean: float, lower: float, upper: float, precision: int = 2
) -> str:
    """
    Format a value with asymmetric errors as LaTeX string.

    Parameters
    ----------
    mean : float
        Central value.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    precision : int
        Number of decimal places.

    Returns
    -------
    str
        LaTeX formatted string like "67.1^{+2.7}_{-4.0}" (without outer $)
    """
    err_up = upper - mean
    err_down = mean - lower
    fmt = f".{precision}f"
    # Check if errors are symmetric (within 5% tolerance)
    if abs(err_up - err_down) < 0.05 * max(abs(err_up), abs(err_down), 0.001):
        avg_err = (err_up + err_down) / 2
        return f"{mean:{fmt}} \\pm {avg_err:{fmt}}"
    else:
        return f"{mean:{fmt}}^{{+{err_up:{fmt}}}}_{{-{err_down:{fmt}}}}"


def format_symmetric_error(mean: float, std: float, precision: int = 2) -> str:
    """Format value with symmetric error (without outer $)."""
    fmt = f".{precision}f"
    return f"{mean:{fmt}} \\pm {std:{fmt}}"


def get_param_latex(param: str, chain_data: Mapping[str, Any] | None = None) -> str:
    """Get LaTeX label for a parameter, checking multiple sources."""
    # First check our predefined labels
    if param in PARAM_LATEX_LABELS:
        return PARAM_LATEX_LABELS[param]
    # Then check chain data if provided
    if chain_data:
        for root_data in chain_data.values():
            if param in root_data.get("minimum", {}).get("params", {}):
                return root_data["minimum"]["params"][param]["latex"]
    # Fallback
    return param


def generate_cosmology_table(
    roots: Sequence[str],
    params: Sequence[str] = ("H0", "S8"),
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
    n_data: int | None = None,
) -> str:
    """
    Generate a LaTeX table for cosmological parameters (H0, S8) with chi^2, AIC, BIC.
    Uses sidewaystable for rotation and booktabs for professional formatting.

    Parameters
    ----------
    roots : list of str
        Chain root names.
    params : list of str
        Parameters to include in table.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.
    n_data : int, optional
        Number of data points for BIC calculation. If None, will be estimated
        automatically from the chi2 components in the .minimum files.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Group chains by dataset to identify LCDM baselines
    dataset_groups: dict[str, dict[str, Any]] = {}
    model_names: dict[str, str] = {}  # Store model names for each root
    for root in roots:
        dataset_key, model_name, is_lcdm = identify_dataset_from_root(root)
        model_names[root] = model_name
        if dataset_key not in dataset_groups:
            dataset_groups[dataset_key] = {"lcdm": None, "others": []}
        if is_lcdm:
            dataset_groups[dataset_key]["lcdm"] = root
        else:
            dataset_groups[dataset_key]["others"].append(root)

    # Collect data for all chains
    chain_data: dict[str, Any] = {}
    for root in roots:
        minimum_file = os.path.join(chain_dir, f"{root}.minimum")
        if os.path.exists(minimum_file):
            min_data: dict[str, Any] = parse_minimum_file(minimum_file)
        else:
            min_data = {
                "neg_log_like": None,
                "chi_sq": None,
                "chi_sq_components": {},
                "params": {},
            }

        stats = get_chain_statistics(root, params, chain_dir, settings)
        n_params = count_free_parameters(root, chain_dir, settings)

        chi_sq = min_data["chi_sq"]

        # Estimate n_data from chi2 components if not provided
        chain_n_data = n_data
        if chain_n_data is None and min_data.get("chi_sq_components"):
            chain_n_data = estimate_n_data_from_likelihoods(
                min_data["chi_sq_components"]
            )
            if chain_n_data:
                print(
                    f"  {root}: Estimated N_data = {chain_n_data} from likelihoods: {list(min_data['chi_sq_components'].keys())}"
                )

        aic, bic = (
            compute_aic_bic(chi_sq, n_params, chain_n_data) if chi_sq else (None, None)
        )

        chain_data[root] = {
            "minimum": min_data,
            "stats": stats,
            "n_params": n_params,
            "n_data": chain_n_data,
            "chi_sq": chi_sq,
            "aic": aic,
            "bic": bic,
        }

    # Build column spec with math mode columns
    # Model | param1 (68% limits) | ... | χ² | ΔAIC | ΔBIC
    n_param_cols = len(params)
    # Use >{$}c<{$} for automatic math mode in data columns
    col_spec = "l" + " >{$}c<{$}" * n_param_cols + " >{$}c<{$} >{$}c<{$} >{$}c<{$}"

    lines: list[str] = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(r"\caption{Cosmological parameters from MCMC analysis.}")
    lines.append(r"\label{tab:cosmology}")
    lines.append(r"\tagpdfsetup{table/header-rows={1}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row - need to escape math mode for headers since columns are in math mode
    header_parts: list[str] = ["Model"]
    for param in params:
        label = get_param_latex(param, chain_data)
        header_parts.append(r"\text{$" + label + r"$ (68\% CI)}")
    header_parts.extend([r"\chi^2", r"\Delta\text{AIC}", r"\Delta\text{BIC}"])
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows grouped by dataset
    for dataset_key in sorted(dataset_groups.keys()):
        group = dataset_groups[dataset_key]
        lcdm_root = group["lcdm"]

        # Get baseline values for ΔAIC and ΔBIC
        if lcdm_root and lcdm_root in chain_data and chain_data[lcdm_root]["aic"]:
            baseline_aic = chain_data[lcdm_root]["aic"]
            baseline_bic = chain_data[lcdm_root]["bic"]
        else:
            baseline_aic = None
            baseline_bic = None

        # Add dataset separator
        n_total_cols = n_param_cols + 4
        lines.append(
            r"\multicolumn{"
            + str(n_total_cols)
            + r"}{l}{\textbf{"
            + dataset_key
            + r"}} \\"
        )

        # Add LCDM first, then others
        group_roots: list[str] = []
        if lcdm_root:
            group_roots.append(lcdm_root)
        group_roots.extend(group["others"])

        for root in group_roots:
            if root not in chain_data:
                continue
            data = chain_data[root]
            row_parts: list[str] = [model_names[root]]

            for param in params:
                # Use 68% confidence limits with asymmetric errors
                if data["stats"] and data["stats"].get(param):
                    s = data["stats"][param]
                    row_parts.append(
                        format_value_with_errors(
                            s["mean"], s["lower_1sigma"], s["upper_1sigma"], precision=2
                        )
                    )
                else:
                    row_parts.append("--")

            # Chi-squared
            if data["chi_sq"]:
                row_parts.append(f"{data['chi_sq']:.2f}")
            else:
                row_parts.append("--")

            # ΔAIC
            if data["aic"] and baseline_aic:
                delta_aic = data["aic"] - baseline_aic
                row_parts.append(f"{delta_aic:+.2f}")
            else:
                row_parts.append("--")

            # ΔBIC
            if data["bic"] and baseline_bic:
                delta_bic = data["bic"] - baseline_bic
                row_parts.append(f"{delta_bic:+.2f}")
            else:
                row_parts.append("--")

            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")

    return "\n".join(lines)


def get_dataset_label(root: str) -> str:
    """
    Extract a human-readable dataset label from a chain root name.

    Returns a LaTeX-formatted string describing the dataset combination.
    """
    root_lower = root.lower()

    parts: list[str] = []
    if "planck" in root_lower:
        parts.append("Planck")
    if "pp" in root_lower or "pantheon" in root_lower:
        parts.append("PP")
    if "sh0es" in root_lower or "shoes" in root_lower:
        parts.append("SH0ES")
    elif "_s_" in root_lower:  # Check for _S_ pattern (short for SH0ES)
        parts.append("SH0ES")
    if "desi" in root_lower:
        if "dr2" in root_lower:
            parts.append("DESI DR2")
        elif "dr1" in root_lower:
            parts.append("DESI DR1")
        else:
            parts.append("DESI")

    return " + ".join(parts) if parts else root


def generate_scf_table(
    roots: Sequence[str],
    params: Sequence[str] = ("cdm_c", "scf_c2", "scf_c3", "scf_c4"),
    param_labels: Mapping[str, str] | None = None,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> str:
    """
    Generate LaTeX tables for scalar field parameters, one per model type.

    Creates separate tables for Double Exponential and Hyperbolic models,
    with each row representing a different dataset combination.

    Parameters
    ----------
    roots : list of str
        Chain root names (typically only non-LCDM models).
    params : list of str
        Scalar field parameters to include.
    param_labels : dict, optional
        Custom LaTeX labels for parameters.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.

    Returns
    -------
    str
        LaTeX table code (multiple tables concatenated).
    """
    # Group roots by model type
    model_groups: dict[str, list[str]] = {
        "dexp": [],
        "hyperbolic": [],
    }  # Double Exponential  # Hyperbolic

    for root in roots:
        root_lower = root.lower()
        _, _, is_lcdm = identify_dataset_from_root(root)
        if is_lcdm:
            continue  # Skip LCDM

        if "doubleexp" in root_lower or "dexp" in root_lower:
            model_groups["dexp"].append(root)
        elif "hyperbolic" in root_lower or "tanh" in root_lower:
            model_groups["hyperbolic"].append(root)

    # Model display names and labels for captions
    model_info: dict[str, dict[str, str]] = {
        "dexp": {
            "name": r"\Nref{pot:dexp} potential",
            "label": "tab:scf_dexp",
        },
        "hyperbolic": {
            "name": r"\Nref{pot:tanh} potential",
            "label": "tab:scf_hyperbolic",
        },
    }

    all_tables: list[str] = []

    for model_key, model_roots in model_groups.items():
        if not model_roots:
            continue

        # Collect data for this model's chains
        chain_data: dict[str, Any] = {}
        for root in model_roots:
            minimum_file = os.path.join(chain_dir, f"{root}.minimum")
            if os.path.exists(minimum_file):
                min_data: dict[str, Any] = parse_minimum_file(minimum_file)
            else:
                min_data = {"params": {}}

            stats = get_chain_statistics(root, params, chain_dir, settings)
            chain_data[root] = {"minimum": min_data, "stats": stats}

        # Build table with math mode columns
        n_param_cols = len(params)
        col_spec = "l" + " >{$}c<{$}" * n_param_cols

        lines: list[str] = []
        lines.append(r"\begin{sidewaystable}")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Scalar field parameters for the "
            + model_info[model_key]["name"]
            + r".}"
        )
        lines.append(r"\label{" + model_info[model_key]["label"] + "}")
        lines.append(r"\tagpdfsetup{table/header-rows={1}}")
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header - use "Dataset" instead of "Model"
        header_parts: list[str] = ["Dataset"]
        for param in params:
            label = get_param_latex(param, chain_data)
            header_parts.append(r"\text{$" + label + r"$ (68\% CI)}")
        lines.append(" & ".join(header_parts) + r" \\")
        lines.append(r"\midrule")

        # Data rows - one per dataset
        for root in model_roots:
            data = chain_data[root]
            dataset_label = get_dataset_label(root)
            row_parts: list[str] = [dataset_label]

            for param in params:
                # Use 68% confidence limits with asymmetric errors
                if data["stats"] and data["stats"].get(param):
                    s = data["stats"][param]
                    row_parts.append(
                        format_value_with_errors(
                            s["mean"], s["lower_1sigma"], s["upper_1sigma"], precision=2
                        )
                    )
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{sidewaystable}")

        all_tables.append("\n".join(lines))

    if not all_tables:
        return "% No scalar field models found in the provided roots."

    return "\n\n".join(all_tables)


def generate_detailed_table(
    roots: Sequence[str],
    params: Sequence[str],
    param_labels: Mapping[str, str] | None = None,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
    caption: str = "Parameter constraints from MCMC analysis.",
    label: str = "tab:params",
) -> str:
    """
    Generate a detailed LaTeX table with best-fit, mean, and 1σ/2σ ranges.
    Uses sidewaystable for rotation and booktabs for professional formatting.

    Parameters
    ----------
    roots : list of str
        Chain root names.
    params : list of str
        Parameters to include.
    param_labels : dict, optional
        Custom LaTeX labels.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.
    caption : str
        Table caption.
    label : str
        Table label.

    Returns
    -------
    str
        LaTeX table code.
    """
    # Collect data and model names
    chain_data: dict[str, Any] = {}
    model_names: dict[str, str] = {}
    for root in roots:
        _, model_name, _ = identify_dataset_from_root(root)
        model_names[root] = model_name

        minimum_file = os.path.join(chain_dir, f"{root}.minimum")
        if os.path.exists(minimum_file):
            min_data: dict[str, Any] = parse_minimum_file(minimum_file)
        else:
            min_data = {"params": {}}

        stats = get_chain_statistics(root, params, chain_dir, settings)
        chain_data[root] = {"minimum": min_data, "stats": stats}

    # Build table: rows are parameters, columns are chains
    # For each chain: 68% limits | 95% limits
    n_chains = len(roots)
    # Use math mode columns
    col_spec = "l" + " >{$}c<{$} >{$}c<{$}" * n_chains

    lines: list[str] = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\tagpdfsetup{table/header-rows={1}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: model names spanning 2 columns each
    header1_parts: list[str] = [""]
    for root in roots:
        header1_parts.append(r"\multicolumn{2}{c}{" + model_names[root] + "}")
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: 68% | 95% for each chain
    header2_parts: list[str] = ["Parameter"]
    for _ in roots:
        header2_parts.extend([r"\text{68\% CI}", r"\text{95\% CI}"])
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for param in params:
        label = get_param_latex(param, chain_data)
        row_parts: list[str] = [f"${label}$"]

        for root in roots:
            data = chain_data[root]

            if data["stats"] and data["stats"].get(param):
                s = data["stats"][param]
                # 68% limits
                row_parts.append(
                    format_value_with_errors(
                        s["mean"], s["lower_1sigma"], s["upper_1sigma"], precision=2
                    )
                )
                # 95% limits
                row_parts.append(
                    format_value_with_errors(
                        s["mean"], s["lower_2sigma"], s["upper_2sigma"], precision=2
                    )
                )
            else:
                row_parts.extend(["--", "--"])

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")

    return "\n".join(lines)


# ============================================================================
# Generate tables for the current analysis
# ============================================================================
# %%
# Table 1: H0 and S8 with chi^2, AIC, BIC
# n_data is automatically estimated from the chi2 components in the .minimum files
print("=" * 80)
print("TABLE 1: Cosmological Parameters (H0, S8)")
print("=" * 80)
cosmology_table = generate_cosmology_table(
    ROOTS,
    params=["H0", "S8"],
    chain_dir=CHAIN_DIR,
    settings=ANALYSIS_SETTINGS,
    # n_data is estimated automatically from likelihoods in .minimum files
)
print(cosmology_table)

# %%
# Table 2: Scalar field parameters
print("\n" + "=" * 80)
print("TABLE 2: Scalar Field Parameters")
print("=" * 80)
scf_table = generate_scf_table(
    ROOTS,
    params=["cdm_c", "scf_c2", "scf_c3", "scf_c4"],
    param_labels=scf_labels,
    chain_dir=CHAIN_DIR,
    settings=ANALYSIS_SETTINGS,
)
print(scf_table)

# %%
# Table 3: Detailed table with confidence intervals
print("\n" + "=" * 80)
print("TABLE 3: Detailed Parameter Constraints")
print("=" * 80)
detailed_table = generate_detailed_table(
    ROOTS,
    params=["H0", "S8", "Omega_m", "sigma8"],
    chain_dir=CHAIN_DIR,
    settings=ANALYSIS_SETTINGS,
    caption="Detailed cosmological parameter constraints.",
    label="tab:detailed_cosmo",
)
print(detailed_table)
