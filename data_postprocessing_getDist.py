# %%
# Import required libraries:
# - matplotlib.pyplot for plotting
# - cmcrameri.cm for perceptually uniform colormaps
# - getdist.plots for MCMC chain plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cmcrameri import cm
from getdist import plots

# ============================================================================
# COMMON CONFIGURATION
# ============================================================================

# Directory where MCMC chain files are stored
CHAIN_DIR = r"/Users/klmba/kDrive/Sci/PhD/Research/HDM/MCMCfast/p201176"
ANALYSIS_SETTINGS = {"ignore_rows": 0.33}

# Define the root names of the MCMC chains (file prefixes without extensions)
ROOTS = [
    # "Cobaya_mcmc_Run3_Planck_PP_SH0ES_DESIDR2_DoubleExp_tracking_uncoupled",
    # "cobaya_iDM_20251230_dexp",
    "cobaya_mcmc_fast_Run1_Planck_2018_DoubleExp_tracking_uncoupled"
]

# Extract a list of colors from the categorical batlowKS colourmap
# Reserve indices 0, 1 for observational bands; 2+ for MCMC chains
ALL_COLOURS = [tuple(c) for c in cm.batlowKS.colors]
BAND_COLOURS = ALL_COLOURS[:2]  # colours[0] for H0, colours[1] for S8
CHAIN_COLOURS = ALL_COLOURS[2 : 2 + len(ROOTS)]  # consistent chain colours


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def make_triangle_plot(params, annotations=None, param_labels=None, title=None):
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
    g = plots.get_subplot_plotter(
        chain_dir=CHAIN_DIR,
        analysis_settings=ANALYSIS_SETTINGS,
    )

    # Load samples and apply custom labels if provided
    if param_labels:
        samples_list = [g.sample_analyser.samples_for_root(root) for root in ROOTS]
        for samples in samples_list:
            for param_name, label in param_labels.items():
                p = samples.paramNames.parWithName(param_name)
                if p is not None:
                    p.label = label
        roots_to_plot = samples_list
    else:
        roots_to_plot = ROOTS

    # Generate the triangle plot
    g.triangle_plot(
        roots_to_plot,
        params,
        filled=True,
        colors=CHAIN_COLOURS,
        diag1d_kwargs={"colors": CHAIN_COLOURS},
        contour_lws=3,
        legend_loc="lower left",
        figure_legend_outside=True,
    )

    fig = g.fig

    # Build legend handles for MCMC chains
    chain_handles = [
        Patch(facecolor=CHAIN_COLOURS[i], label=ROOTS[i]) for i in range(len(ROOTS))
    ]

    # Apply custom annotations and collect their legend handles
    annotation_handles = []
    if annotations is not None:
        annotation_handles = annotations(g) or []

    all_handles = chain_handles + annotation_handles
    all_labels = [h.get_label() for h in all_handles]

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


def annotate_H0_S8(g):
    """
    Add H0 (SH0ES) and S8 (KiDS-1000) observational bands.
    Returns legend handles for these annotations.
    """
    # SH0ES 2020b: H0 = 73.2 ± 1.3 km/s/Mpc
    g.add_x_bands(73.2, 1.3, ax=0, color=BAND_COLOURS[0])
    g.add_x_bands(73.2, 1.3, ax=2, color=BAND_COLOURS[0])

    # KiDS-1000 2023: S8 = 0.776 ± 0.031
    g.add_x_bands(0.776, 0.031, ax=3, color=BAND_COLOURS[1])
    g.add_y_bands(0.776, 0.031, ax=2, color=BAND_COLOURS[1])

    return [
        Patch(facecolor=BAND_COLOURS[0], alpha=0.5, label=r"$H_0$ SH0ES 2020"),
        Patch(facecolor=BAND_COLOURS[1], label=r"$S_8$ KiDS-1000"),
    ]


def annotate_scf_constraints(g):
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
    handles = []

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
params_cosmology = ["H0", "s8h5"]
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

# %%
