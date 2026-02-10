# %%
# Import required libraries:
# - matplotlib.pyplot for plotting
# - cmcrameri.cm for perceptually uniform colormaps
# - getdist.plots for MCMC chain plotting
from typing import Any, Callable, Mapping, Sequence, cast
import os
import re
import glob
import numpy as np  # type: ignore[import-untyped]

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from matplotlib.patches import Patch
from cmcrameri import cm  # type: ignore[import-untyped]
from getdist import plots  # type: ignore[import-untyped]

plt = cast(Any, plt)
cm = cast(Any, cm)
plots = cast(Any, plots)

Color = Any

# Global cache for loaded samples to avoid redundant loading
_SAMPLES_CACHE: dict[str, Any] = {}
_ROOT_PATH_CACHE: dict[str, str] = {}

# GetDist imports for MCMC analysis
from getdist import MCSamples, loadMCSamples  # type: ignore[import-untyped]

MCSamples = cast(Any, MCSamples)
loadMCSamples = cast(Callable[..., Any], loadMCSamples)


def preload_all_chains(
    roots: Sequence[str],
    chain_dir: str,
    settings: Mapping[str, Any],
    verbose: bool = True,
) -> None:
    """
    Preload all chains into the cache to minimize redundant loading.

    Call this once before any analysis to ensure all subsequent operations
    use cached data instead of reloading from disk.

    Parameters
    ----------
    roots : list of str
        Chain root names to preload.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings (e.g., burn-in).
    verbose : bool
        If True, print loading progress.
    """
    if verbose:
        print(f"Preloading {len(roots)} chain(s) into cache...")

    for i, root in enumerate(roots, 1):
        resolved_root = resolve_chain_root(root, chain_dir)
        cache_key = _cache_key(chain_dir, resolved_root)
        if cache_key not in _SAMPLES_CACHE:
            try:
                if verbose:
                    print(f"  [{i}/{len(roots)}] Loading {root}...")
                samples = loadMCSamples(
                    _root_base_path(resolved_root, chain_dir), settings=settings
                )
                _SAMPLES_CACHE[cache_key] = samples
            except Exception as e:
                if verbose:
                    print(f"  [{i}/{len(roots)}] Failed to load {root}: {e}")
                _SAMPLES_CACHE[cache_key] = None
        elif verbose:
            print(f"  [{i}/{len(roots)}] {root} already cached")

    if verbose:
        successful = sum(1 for v in _SAMPLES_CACHE.values() if v is not None)
        print(
            f"Cache preload complete: {successful}/{len(roots)} chains loaded successfully.\n"
        )


# ============================================================================
# COMMON CONFIGURATION
# ============================================================================

# Directory where MCMC chain files are stored
# Try both paths and use the one that works
_path1 = r"/Users/klmba/kDrive/Sci/PhD/Research/HDM/MCMC_archive/"
_path2 = r"/home/kl/kDrive/Sci/PhD/Research/HDM/MCMC_archive/"
CHAIN_DIR: str = _path1 if os.path.exists(_path1) else _path2
ANALYSIS_SETTINGS: dict[str, float] = {"ignore_rows": 0.33}

# Define the root names of the MCMC chains (file prefixes without extensions)
ROOTS: list[str] = [
    # "cobaya_polychord_CV_PP_DESI_LCDM.post.S8",
    "cobaya_mcmc_fast_CMB_LCDM",
    "cobaya_mcmc_CV_PP_S_DESI_LCDM.post.S8",
    "cobaya_mcmc_CV_CMB_SPA_LCDM.post.S8",
    "cobaya_mcmc_CV_CMB_SPA_PP_DESI_LCDM.post.S8",
    "cobaya_mcmc_CV_CMB_SPA_PP_S_DESI_LCDM.post.S8",
]

# Extract a list of colors from the categorical batlowKS colourmap
# Reserve indices 0, 1 for observational bands; 2+ for MCMC chains
ALL_COLOURS: list[Color] = [tuple(c) for c in cm.batlowKS.colors]
BAND_COLOURS: list[Color] = ALL_COLOURS[:2]  # colours[0] for H0, colours[1] for S8
CHAIN_COLOURS: list[Color] = ALL_COLOURS[2 : 2 + len(ROOTS)]  # consistent chain colours


def _root_from_filepath(path: str, chain_dir: str) -> str:
    """Return a chain root path (relative to chain_dir) from a file path."""
    rel = os.path.relpath(path, chain_dir)
    rel = rel.replace(os.sep, "/")
    rel = re.sub(r"\.\d+\.txt$", "", rel)
    rel = re.sub(r"\.txt$", "", rel)
    rel = re.sub(r"\.bestfit$", "", rel)
    rel = re.sub(r"\.input\.yaml$", "", rel)
    rel = re.sub(r"\.updated\.yaml$", "", rel)
    return rel


def resolve_chain_root(root: str, chain_dir: str = CHAIN_DIR) -> str:
    """
    Resolve a root name to a subfoldered root path if needed.

    If root already contains a path segment, it is returned as-is.
    Otherwise, search under chain_dir for matching chain files and cache the result.
    """
    if os.path.isabs(root):
        try:
            common = os.path.commonpath([chain_dir, root])
        except ValueError:
            common = None
        if common == chain_dir:
            return _root_from_filepath(root, chain_dir)
        return root

    if "/" in root or os.sep in root:
        base = _root_base_path(root, chain_dir)
        if (
            os.path.exists(f"{base}.txt")
            or os.path.exists(f"{base}.bestfit")
            or os.path.exists(f"{base}.input.yaml")
            or os.path.exists(f"{base}.updated.yaml")
            or bool(glob.glob(f"{base}.*.txt"))
        ):
            return root
        # Path-like root not found; try resolving under subfolders
        search_root = root.replace(os.sep, "/")
        patterns = [
            f"**/{search_root}.*.txt",
            f"**/{search_root}.txt",
            f"**/{search_root}.bestfit",
            f"**/{search_root}.input.yaml",
            f"**/{search_root}.updated.yaml",
        ]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(chain_dir, pattern), recursive=True))

        if not matches:
            return root

        matches = sorted(set(matches))
        resolved = _root_from_filepath(matches[0], chain_dir)
        _ROOT_PATH_CACHE[root] = resolved

        if len(matches) > 1:
            rel_matches = [os.path.relpath(p, chain_dir) for p in matches[:5]]
            print(
                "Warning: Multiple chain matches for root "
                f"'{root}', using '{resolved}'. Examples: {rel_matches}"
            )

        return resolved

    if root in _ROOT_PATH_CACHE:
        return _ROOT_PATH_CACHE[root]

    patterns = [
        f"**/{root}.*.txt",
        f"**/{root}.txt",
        f"**/{root}.bestfit",
        f"**/{root}.input.yaml",
        f"**/{root}.updated.yaml",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(chain_dir, pattern), recursive=True))

    if not matches:
        _ROOT_PATH_CACHE[root] = root
        return root

    matches = sorted(set(matches))
    resolved = _root_from_filepath(matches[0], chain_dir)
    _ROOT_PATH_CACHE[root] = resolved

    if len(matches) > 1:
        rel_matches = [os.path.relpath(p, chain_dir) for p in matches[:5]]
        print(
            "Warning: Multiple chain matches for root "
            f"'{root}', using '{resolved}'. Examples: {rel_matches}"
        )

    return resolved


def _cache_key(chain_dir: str, root: str) -> str:
    if os.path.isabs(root):
        return root
    return f"{chain_dir}/{root}"


def _root_base_path(root: str, chain_dir: str) -> str:
    """Return absolute root base path without extensions."""
    if os.path.isabs(root):
        return root
    return os.path.join(chain_dir, root)


def get_samples_for_root(
    root: str,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> Any | None:
    """Load samples for a root with caching and subfolder resolution."""
    resolved_root = resolve_chain_root(root, chain_dir)
    cache_key = _cache_key(chain_dir, resolved_root)
    if cache_key in _SAMPLES_CACHE:
        return _SAMPLES_CACHE[cache_key]
    try:
        samples = loadMCSamples(
            _root_base_path(resolved_root, chain_dir), settings=settings
        )
    except Exception as e:
        print(f"Note: GetDist failed for {root}: {e}")
        samples = None
    _SAMPLES_CACHE[cache_key] = samples
    return samples


def build_legend_label(root: str) -> str:
    """Build a legend label from the chain root name."""
    root_lower = root.lower()

    if "lcdm" in root_lower:
        model_label = r"$\Lambda$CDM"
    elif "hyperbolic" in root_lower:
        model_label = "Hyperbolic"
    elif "doubleexp" in root_lower or "doubleexponential" in root_lower:
        model_label = "Double Exponential"
    else:
        model_label = "Model"

    likelihoods: list[str] = []
    if "fast" in root_lower:
        likelihoods.append("Planck 2018")
    if "spa" in root_lower:
        likelihoods.append("SPA")
    if "pp" in root_lower:
        likelihoods.append("Pantheon+")
    if "_s_" in root_lower:
        likelihoods.append("SH0ES")
    if "desi" in root_lower:
        likelihoods.append("DESI DR2")

    if likelihoods:
        return f"{model_label}: " + " | ".join(likelihoods)
    return f"{model_label}: (no likelihood tags)"


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
    samples_by_root: list[tuple[str, str, Any]] = []
    for root in ROOTS:
        samples = get_samples_for_root(root, CHAIN_DIR, ANALYSIS_SETTINGS)
        if samples is None:
            continue
        if any(samples.paramNames.parWithName(p) is not None for p in params):
            resolved_root = resolve_chain_root(root, CHAIN_DIR)
            samples_by_root.append((root, resolved_root, samples))

    if not samples_by_root:
        raise ValueError(
            "None of the requested parameters are available in the provided roots."
        )

    available_params: list[str] = [
        p
        for p in params
        if all(
            samples.paramNames.parWithName(p) is not None
            for _, _, samples in samples_by_root
        )
    ]

    if not available_params:
        raise ValueError(
            "No common parameters found across the selected roots; cannot plot."
        )

    # Apply custom labels if provided
    if param_labels:
        for _, _, samples in samples_by_root:
            for param_name, label in param_labels.items():
                p = samples.paramNames.parWithName(param_name)
                if p is not None:
                    p.label = label

    used_roots: list[str] = [root for root, _, _ in samples_by_root]
    roots_to_plot: Sequence[Any] = [samples for _, _, samples in samples_by_root]
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
        Patch(facecolor=root_to_color[root], label=build_legend_label(root))
        for root in used_roots
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
    g.add_x_bands(73.18, 0.88, ax=0, color=BAND_COLOURS[0])
    g.add_x_bands(73.18, 0.88, ax=2, color=BAND_COLOURS[0])

    # KiDS-1000 2023: S8 = 0.776 ± 0.031
    g.add_x_bands(0.776, 0.031, ax=3, color=BAND_COLOURS[1])
    g.add_y_bands(0.776, 0.031, ax=2, color=BAND_COLOURS[1])

    return [
        Patch(facecolor=BAND_COLOURS[0], alpha=0.5, label=r"$H_0$ SH0ES 2025"),
        Patch(facecolor=BAND_COLOURS[1], label=r"$S_8$ KiDS-1000"),
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
# PRELOAD CHAINS INTO CACHE
# ============================================================================
# Preload all chains before any analysis to maximize performance
# This ensures both plots and tables benefit from cached data
preload_all_chains(ROOTS, CHAIN_DIR, ANALYSIS_SETTINGS, verbose=True)

# ============================================================================
# PLOT 1: H0 & S8 with observational bands
# ============================================================================
# %%
params_cosmology = ["H0", "S8"]
g1 = make_triangle_plot(params_cosmology, annotations=annotate_H0_S8, title=None)

# Export example:
# g1.fig.savefig("plot_H0_S8_PP_S_DESI_LCDM_hyperbolic.png", bbox_inches="tight", dpi=300)

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
# g2.fig.savefig("plot_scf_params_PP_S_DESI_hyperbolic.png", bbox_inches="tight", dpi=300)

plt.show()

# ============================================================================
# TABLE GENERATION: Extract data from chains and create LaTeX tables
# ============================================================================
# %%


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
    resolved_root = resolve_chain_root(root, chain_dir)
    for yaml_suffix in [".input.yaml", ".updated.yaml"]:
        yaml_file = _root_base_path(resolved_root, chain_dir) + yaml_suffix
        if os.path.exists(yaml_file):
            likelihoods = parse_cobaya_yaml(yaml_file)
            if likelihoods:
                break

    # If no likelihoods found in YAML, try .minimum file as fallback
    if not likelihoods:
        minimum_file = _root_base_path(resolved_root, chain_dir) + ".bestfit"
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
    stats: dict[str, Any] = {}
    samples: Any = None
    chain_data: dict[str, Any] = {}

    # Try GetDist first (with caching)
    resolved_root = resolve_chain_root(root, chain_dir)
    cache_key = _cache_key(chain_dir, resolved_root)
    if cache_key in _SAMPLES_CACHE:
        samples = _SAMPLES_CACHE[cache_key]
        use_getdist = samples is not None
    else:
        try:
            samples = loadMCSamples(
                _root_base_path(resolved_root, chain_dir), settings=settings
            )
            _SAMPLES_CACHE[cache_key] = samples
            use_getdist = True
        except Exception as e:
            print(f"Note: GetDist failed for {root}, using direct chain reading: {e}")
            _SAMPLES_CACHE[cache_key] = None
            use_getdist = False

    if not use_getdist:
        # Read chain data directly
        try:
            chain_data = read_chain_data_directly(root, params, chain_dir, settings)
        except Exception as e2:
            print(f"Warning: Could not read chain data for {root}: {e2}")
            return {param: None for param in params}

    for param in params:
        if use_getdist and samples is not None:
            # Use GetDist
            p = samples.paramNames.parWithName(param)
            if p is None:
                # Try fallback for this specific parameter
                try:
                    chain_data_single = read_chain_data_directly(
                        root, [param], chain_dir, settings
                    )
                    if param in chain_data_single:
                        values = chain_data_single[param]
                        weights = chain_data_single["weights"]
                        stats[param] = calculate_statistics_from_samples(
                            values, weights
                        )
                    else:
                        stats[param] = None
                except Exception:
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
        else:
            # Use direct reading fallback
            if param in chain_data:
                values = chain_data[param]
                weights = chain_data["weights"]
                stats[param] = calculate_statistics_from_samples(values, weights)
            else:
                stats[param] = None

    return stats


def calculate_statistics_from_samples(
    values: Any,
    weights: Any,
) -> dict[str, float]:
    """
    Calculate mean, std, and confidence intervals from weighted samples.

    Parameters
    ----------
    values : array-like
        Parameter values.
    weights : array-like
        Sample weights.

    Returns
    -------
    dict
        Dictionary with mean, std, and confidence limits.
    """
    # Calculate weighted mean and std
    mean = float(np.average(values, weights=weights))  # type: ignore[no-untyped-call]
    variance = float(np.average((values - mean) ** 2, weights=weights))  # type: ignore[no-untyped-call]
    std = float(np.sqrt(variance))  # type: ignore[no-untyped-call]

    # Calculate confidence limits using weighted percentiles
    # Sort values and weights
    sorted_indices = np.argsort(values)  # type: ignore[no-untyped-call]
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Calculate cumulative weights
    cumsum = np.cumsum(sorted_weights)  # type: ignore[no-untyped-call]
    cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]

    # Find percentiles for confidence intervals
    # 68% CI: 16th and 84th percentiles
    # 95% CI: 2.5th and 97.5th percentiles
    lower_1 = float(np.interp(0.16, cumsum, sorted_values))  # type: ignore[no-untyped-call]
    upper_1 = float(np.interp(0.84, cumsum, sorted_values))  # type: ignore[no-untyped-call]
    lower_2 = float(np.interp(0.025, cumsum, sorted_values))  # type: ignore[no-untyped-call]
    upper_2 = float(np.interp(0.975, cumsum, sorted_values))  # type: ignore[no-untyped-call]

    return {
        "mean": mean,
        "std": std,
        "lower_1sigma": lower_1,
        "upper_1sigma": upper_1,
        "lower_2sigma": lower_2,
        "upper_2sigma": upper_2,
    }


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
    # Use cache if available
    resolved_root = resolve_chain_root(root, chain_dir)
    cache_key = _cache_key(chain_dir, resolved_root)
    if cache_key in _SAMPLES_CACHE and _SAMPLES_CACHE[cache_key] is not None:
        samples: Any = _SAMPLES_CACHE[cache_key]
    else:
        samples = loadMCSamples(
            _root_base_path(resolved_root, chain_dir), settings=settings
        )
        _SAMPLES_CACHE[cache_key] = samples
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
        model_name = r"\gls{lcdm}"
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

    # Use scientific notation for very small numbers (|value| < 0.01)
    if abs(mean) < 0.01 and mean != 0:
        # Format in scientific notation
        mean_str = f"{mean:.{precision}e}"
        err_up_str = f"{err_up:.{precision}e}"
        err_down_str = f"{err_down:.{precision}e}"

        # Check if errors are symmetric
        if abs(err_up - err_down) < 0.05 * max(abs(err_up), abs(err_down), 1e-10):
            avg_err = (err_up + err_down) / 2
            avg_err_str = f"{avg_err:.{precision}e}"
            return f"{mean_str} \\pm {avg_err_str}"
        else:
            return f"{mean_str}^{{+{err_up_str}}}_{{-{err_down_str}}}"

    # Standard formatting for normal-sized numbers
    fmt = f".{precision}f"
    # Check if errors are symmetric (within 5% tolerance)
    if abs(err_up - err_down) < 0.05 * max(abs(err_up), abs(err_down), 0.001):
        avg_err = (err_up + err_down) / 2
        return f"{mean:{fmt}} \\pm {avg_err:{fmt}}"
    else:
        return f"{mean:{fmt}}^{{+{err_up:{fmt}}}}_{{-{err_down:{fmt}}}}"


def format_symmetric_error(mean: float, std: float, precision: int = 2) -> str:
    """Format value with symmetric error (without outer $)."""
    # Use scientific notation for very small numbers
    if abs(mean) < 0.01 and mean != 0:
        return f"{mean:.{precision}e} \\pm {std:.{precision}e}"

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
        resolved_root = resolve_chain_root(root, chain_dir)
        minimum_file = _root_base_path(resolved_root, chain_dir) + ".bestfit"
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
            resolved_root = resolve_chain_root(root, chain_dir)
            minimum_file = _root_base_path(resolved_root, chain_dir) + ".bestfit"
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


def read_chain_data_directly(
    root: str,
    param_names: Sequence[str],
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> dict[str, Any]:
    """
    Read chain data directly from text files as a fallback when GetDist fails.

    Reads all *.txt files matching the root pattern and extracts the specified parameters.

    Parameters
    ----------
    root : str
        Chain root name.
    param_names : list of str
        Parameter names to extract.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings (e.g., ignore_rows for burn-in).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'weights': array of sample weights
        - 'param_name': array of parameter values for each param_name
    """
    import glob

    resolved_root = resolve_chain_root(root, chain_dir)
    base_path = _root_base_path(resolved_root, chain_dir)

    # Find all chain files matching the root
    pattern = f"{base_path}.*.txt"
    chain_files = sorted(glob.glob(pattern))

    if not chain_files:
        # Try without the .* pattern (single file)
        single_file = f"{base_path}.txt"
        if os.path.exists(single_file):
            chain_files = [single_file]
        else:
            fallback = glob.glob(
                os.path.join(chain_dir, f"**/{root}.*.txt"), recursive=True
            )
            chain_files = sorted(fallback)
            if not chain_files:
                raise FileNotFoundError(f"No chain files found for root {root}")

    all_data: list[Any] = []
    param_indices: dict[str, int] = {}
    weight_idx: int | None = None

    for chain_file in chain_files:
        try:
            # Read the file, looking for header line
            with open(chain_file, "r") as f:
                lines = f.readlines()

            # Find header line (starts with #)
            header_line = None
            data_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("#"):
                    header_line = line.strip()[1:].strip()  # Remove # and whitespace
                    data_start_idx = i + 1
                    break

            if header_line is None:
                print(f"Warning: No header found in {chain_file}")
                continue

            # Parse header to get column names
            col_names = header_line.split()

            # Find column indices for requested parameters (only first time)
            if not param_indices:
                for param in param_names:
                    if param in col_names:
                        param_indices[param] = col_names.index(param)

                if "weight" in col_names:
                    weight_idx = col_names.index("weight")

            # Read numerical data
            data_lines = lines[data_start_idx:]
            chain_data: list[Any] = []
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    chain_data.append(values)
                except ValueError:
                    continue

            if chain_data:
                all_data.append(np.array(chain_data))  # type: ignore[no-untyped-call]

        except Exception as e:
            print(f"Warning: Could not read {chain_file}: {e}")
            continue

    if not all_data:
        raise ValueError(f"No valid data could be read from chain files for {root}")

    # Concatenate all chain data
    combined_data: Any = np.vstack(all_data)  # type: ignore[no-untyped-call]

    # Apply burn-in if specified
    ignore_rows = settings.get("ignore_rows", 0.0)
    if ignore_rows > 0:
        if ignore_rows < 1:  # Fraction
            n_ignore = int(len(combined_data) * ignore_rows)  # type: ignore[arg-type]
        else:  # Absolute number
            n_ignore = int(ignore_rows)
        combined_data = combined_data[n_ignore:]

    # Extract weights and parameters
    result: dict[str, Any] = {}

    if weight_idx is not None:
        result["weights"] = combined_data[:, weight_idx]
    else:
        result["weights"] = np.ones(len(combined_data))  # type: ignore[arg-type,no-untyped-call]

    for param, idx in param_indices.items():
        result[param] = combined_data[:, idx]

    return result


def get_integer_parameter_mode(
    root: str,
    param_name: str,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> tuple[float | None, int | None]:
    """
    Extract mean and mode for an integer parameter from raw samples.

    For discrete integer parameters like attractor_regime_scf, this function
    bypasses KDE smoothing and calculates the weighted mode directly.

    Parameters
    ----------
    root : str
        Chain root name.
    param_name : str
        Name of the integer parameter.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.

    Returns
    -------
    tuple
        (mean_value, mode_value) where mode_value is an integer, or (None, None) on error.
    """
    try:
        # First try GetDist (with caching)
        resolved_root = resolve_chain_root(root, chain_dir)
        cache_key = _cache_key(chain_dir, resolved_root)
        if cache_key in _SAMPLES_CACHE and _SAMPLES_CACHE[cache_key] is not None:
            samples: Any = _SAMPLES_CACHE[cache_key]
        else:
            samples = loadMCSamples(
                _root_base_path(resolved_root, chain_dir), settings=settings
            )
            _SAMPLES_CACHE[cache_key] = samples
        param_values = samples[param_name]
        weights = samples.weights
    except Exception as e:
        # Fallback: read directly from chain files
        try:
            chain_data = read_chain_data_directly(
                root, [param_name], chain_dir, settings
            )
            param_values = chain_data[param_name]
            weights = chain_data["weights"]
        except Exception as e2:
            print(f"Warning: Could not extract mode for {param_name} in {root}: {e2}")
            return None, None

    try:
        # Calculate mean
        if weights is not None:
            mean_value = float(np.average(param_values, weights=weights))  # type: ignore[no-untyped-call]
        else:
            mean_value = float(np.mean(param_values))  # type: ignore[no-untyped-call]

        # Calculate discrete mode
        discrete_values = np.round(param_values).astype(int)  # type: ignore[no-untyped-call]
        unique_vals, indices = np.unique(discrete_values, return_inverse=True)  # type: ignore[no-untyped-call]
        weighted_counts = np.bincount(indices, weights=weights)  # type: ignore[no-untyped-call]
        mode_index = np.argmax(weighted_counts)  # type: ignore[no-untyped-call]
        mode_value = int(unique_vals[mode_index])  # type: ignore[no-untyped-call]

        return mean_value, mode_value
    except Exception as e:
        print(f"Warning: Could not calculate mode for {param_name} in {root}: {e}")
        return None, None


def check_parameter_identity(
    root: str,
    param1: str,
    param2: str,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check if two parameters are always identical in a chain.

    Parameters
    ----------
    root : str
        Chain root name.
    param1, param2 : str
        Parameter names to compare.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.
    tolerance : float
        Tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if the parameters are identical within tolerance across all samples.
    """
    try:
        # First try GetDist (with caching)
        resolved_root = resolve_chain_root(root, chain_dir)
        cache_key = _cache_key(chain_dir, resolved_root)
        if cache_key in _SAMPLES_CACHE and _SAMPLES_CACHE[cache_key] is not None:
            samples: Any = _SAMPLES_CACHE[cache_key]
        else:
            samples = loadMCSamples(
                _root_base_path(resolved_root, chain_dir), settings=settings
            )
            _SAMPLES_CACHE[cache_key] = samples
        values1 = samples[param1]
        values2 = samples[param2]
    except Exception as e:
        # Fallback: read directly from chain files
        try:
            chain_data = read_chain_data_directly(
                root, [param1, param2], chain_dir, settings
            )
            values1 = chain_data[param1]
            values2 = chain_data[param2]
        except Exception as e2:
            print(f"Warning: Could not compare {param1} and {param2} in {root}: {e2}")
            return False

    try:
        return np.allclose(values1, values2, atol=tolerance)  # type: ignore[no-untyped-call]
    except Exception as e:
        print(f"Warning: Could not compare {param1} and {param2} in {root}: {e}")
        return False


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

        resolved_root = resolve_chain_root(root, chain_dir)
        minimum_file = _root_base_path(resolved_root, chain_dir) + ".bestfit"
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


def generate_swampland_table(
    roots: Sequence[str],
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> str:
    """
    Generate a LaTeX table for swampland constraint parameters.

    Filters chains to only those containing 'swampland' in the name.
    Organizes by dataset and model type (Hyperbolic, Double Exponential).
    Shows best-fit ± 1σ values.

    Parameters
    ----------
    roots : list of str
        Chain root names.
    chain_dir : str
        Directory containing chain files.
    settings : dict
        Analysis settings.

    Returns
    -------
    str
        LaTeX table code.
    """

    # Filter for swampland chains
    swampland_roots = [r for r in roots if "swampland" in r.lower()]

    if not swampland_roots:
        return "% No swampland constraint chains found."

    # Swampland parameters to extract
    swampland_params = [
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
        "attractor_regime_scf",  # Integer parameter
        "AdSDC2_max",
        "AdSDC4_max",
        "combined_dSC_min",
        "conformal_age",
    ]

    # Check if phi_ini_scf_ic and phi_prime_scf_ic are always identical
    duplicate_phi = True
    for root in swampland_roots:
        if not check_parameter_identity(
            root, "phi_ini_scf_ic", "phi_prime_scf_ic", chain_dir, settings
        ):
            duplicate_phi = False
            break

    if duplicate_phi:
        print(
            "Note: phi_ini_scf_ic and phi_prime_scf_ic are identical across all swampland chains."
        )
        # Remove the duplicate parameter
        swampland_params.remove("phi_prime_scf_ic")

    # Group chains by dataset and model type
    dataset_model_groups: dict[str, dict[str, list[str]]] = {}

    for root in swampland_roots:
        dataset_key, _, _ = identify_dataset_from_root(root)
        root_lower = root.lower()

        if "hyperbolic" in root_lower or "tanh" in root_lower:
            model_type = "hyperbolic"
        elif "doubleexp" in root_lower or "dexp" in root_lower:
            model_type = "dexp"
        else:
            model_type = "unknown"

        if dataset_key not in dataset_model_groups:
            dataset_model_groups[dataset_key] = {"hyperbolic": [], "dexp": []}

        dataset_model_groups[dataset_key][model_type].append(root)

    # Collect parameter data
    chain_data: dict[str, dict[str, Any]] = {}
    integer_modes: dict[str, tuple[float, int]] = {}

    for root in swampland_roots:
        resolved_root = resolve_chain_root(root, chain_dir)
        minimum_file = _root_base_path(resolved_root, chain_dir) + ".bestfit"
        if os.path.exists(minimum_file):
            min_data: dict[str, Any] = parse_minimum_file(minimum_file)
        else:
            min_data = {"params": {}}

        stats = get_chain_statistics(root, swampland_params, chain_dir, settings)

        # Extract mode for integer parameter
        if "attractor_regime_scf" in swampland_params:
            mean_val, mode_val = get_integer_parameter_mode(
                root, "attractor_regime_scf", chain_dir, settings
            )
            if mean_val is not None and mode_val is not None:
                integer_modes[root] = (mean_val, mode_val)

        chain_data[root] = {"minimum": min_data, "stats": stats}

    # Build LaTeX table
    # Parameters as column headers, datasets as sub-headers, within each dataset: Hyperbolic then Double Exp

    # Prepare parameter labels
    param_latex_labels: dict[str, str] = {
        "phi_ini_scf_ic": r"\phi_{\text{ini}}",
        "phi_prime_scf_ic": r"\dot{\phi}_{\text{ini}}",
        "phi_scf_min": r"\phi_{\min}",
        "phi_scf_max": r"\phi_{\max}",
        "phi_scf_range": r"\Delta\phi",
        "dV_V_scf_min": r"\left(\frac{dV}{V}\right)_{\min}",
        "ddV_V_scf_max": r"\left(\frac{d^2V}{V}\right)_{\max}",
        "ddV_V_at_dV_V_min": r"\left(\frac{d^2V}{V}\right)_{\text{dV/V}_{\min}}",
        "dV_V_at_ddV_V_max": r"\left(\frac{dV}{V}\right)_{\text{d}^2\text{V/V}_{\max}}",
        "swgc_expr_min": r"\text{SWGC}_{\text{expr,min}}",
        "sswgc_min": r"\text{SSWGC}_{\min}",
        "attractor_regime_scf": r"n_{\text{attr}}",
        "AdSDC2_max": r"\text{AdSDC}_2",
        "AdSDC4_max": r"\text{AdSDC}_4",
        "combined_dSC_min": r"\text{dSC}_{\min}",
        "conformal_age": r"t_{\text{conf}}",
    }

    # Column spec: 1 for row label + 1 for each parameter
    n_params = len(swampland_params)
    col_spec = "l" + " >{$}c<{$}" * n_params

    lines: list[str] = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(r"\caption{Swampland constraint parameters from MCMC analysis.}")
    lines.append(r"\label{tab:swampland_params}")
    lines.append(r"\tagpdfsetup{table/header-rows={1}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row: parameter names
    header_parts: list[str] = ["Model"]
    for param in swampland_params:
        label = param_latex_labels.get(param, param)
        header_parts.append(r"\text{$" + label + r"$}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows grouped by dataset
    for dataset_key in sorted(dataset_model_groups.keys()):
        models_dict = dataset_model_groups[dataset_key]

        # Add dataset sub-header
        lines.append(
            r"\multicolumn{"
            + str(n_params + 1)
            + r"}{l}{\textbf{"
            + dataset_key
            + r"}} \\"
        )

        # Add Hyperbolic first, then Double Exponential
        for model_type in ["hyperbolic", "dexp"]:
            model_roots = models_dict[model_type]

            model_labels = {
                "hyperbolic": r"\Nref{pot:tanh}",
                "dexp": r"\Nref{pot:dexp}",
            }

            for root in model_roots:
                if root not in chain_data:
                    continue

                data = chain_data[root]
                model_name = model_labels[model_type]
                row_parts: list[str] = [model_name]

                for param in swampland_params:
                    if param == "attractor_regime_scf":
                        # Format as "mean (mode)" for integer parameter
                        if root in integer_modes:
                            mean_val, mode_val = integer_modes[root]
                            row_parts.append(
                                f"{mean_val:.2f} \\text{{ (mode:{mode_val})}}"
                            )
                        else:
                            row_parts.append("--")
                    elif data["stats"] and data["stats"].get(param):
                        s = data["stats"][param]
                        row_parts.append(
                            format_value_with_errors(
                                s["mean"],
                                s["lower_1sigma"],
                                s["upper_1sigma"],
                                precision=2,
                            )
                        )
                    else:
                        row_parts.append("--")

                lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = r"\bottomrule"
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
# Table 3: Swampland constraint parameters
print("\n" + "=" * 80)
print("TABLE 3: Swampland Parameters")
print("=" * 80)
swampland_table = generate_swampland_table(
    ROOTS,
    chain_dir=CHAIN_DIR,
    settings=ANALYSIS_SETTINGS,
)
print(swampland_table)
