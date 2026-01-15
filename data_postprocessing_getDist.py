# %%
# Import required libraries:
# - matplotlib.pyplot for plotting
# - cmcrameri.cm for perceptually uniform colormaps
# - getdist.plots for MCMC chain plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cmcrameri import cm
from getdist import plots

# Initialize the getdist subplot plotter
# - chain_dir: directory where MCMC chain files are stored
# - analysis_settings: ignore_rows to skip burn-in (first 20% of samples)
g = plots.get_subplot_plotter(
    chain_dir=r"/Users/klmba",
    analysis_settings={"ignore_rows": 0.2},
)

# Define the root names of the MCMC chains (file prefixes without extensions)
roots = [
    "Cobaya_mcmc_Run3_Planck_PP_SH0ES_DESIDR2_DoubleExp_tracking_uncoupled",
    "cobaya_iDM_20251230_dexp",
]

# Specify the parameters to plot in the triangle plot
params = ["H0", "s8h5"]  # Hubble constant and S8

# Extract a list of colors from the categorical bamako colourmap for coloring the chains
colours = [tuple(c) for c in cm.batlowKS.colors]

# %%

# Generate the triangle plot showing:
# - 1D marginalized distributions on the diagonal
# - 2D contour plots on the off-diagonal
g.triangle_plot(
    roots,  # List of chain roots to include
    params,  # Parameters to plot
    filled=True,  # Fill the contour regions
    colors=colours[2:],  # Colors for contour lines and fills in 2D plot
    diag1d_kwargs={"colors": colours[2:]},  # Colors for 1D plots
    contour_lws=3,  # Line width for contours
    legend_loc="lower left",  # Legend position
    figure_legend_outside=True,  # Place legend outside the plot area
)

# Add reference bands for observational data
# SH0ES 2020b measurement of H0: 73.2 ± 1.3 km/s/Mpc
g.add_x_bands(73.2, 1.3, ax=0, color=colours[0])  # Vertical band on H0 1D plot
g.add_x_bands(73.2, 1.3, ax=2, color=colours[0])  # Vertical band on H0 axis of 2D plot

# KiDS-1000 2023 measurement of S8: 0.776 ± 0.031
g.add_x_bands(0.776, 0.031, ax=3, color=colours[1])  # Vertical band on S8 1D plot
g.add_y_bands(
    0.776, 0.031, ax=2, color=colours[1]
)  # Horizontal band on S8 axis of 2D plot

# Create legend entries manually
fig = g.fig

# Create patches for the MCMC chains (matching the contour colors)
chain_handles = [
    Patch(facecolor=colours[i + 2], label=roots[i]) for i in range(len(roots))
]

# Create patches for the observational bands
band_handles = [
    Patch(facecolor=colours[0], alpha=0.5, label=r"$H_0$ SH0ES 2020"),
    Patch(facecolor=colours[1], label=r"$S_8$ KiDS-1000"),
]

# Combine all handles and labels
all_handles = chain_handles + band_handles
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

# %%
# Get the position of the first subplot (H0 1D plot) to align the legend
first_ax = g.subplots[0, 0]
ax_bbox = first_ax.get_position()

# Position legend with upper left corner aligned to upper right corner of first subplot
leg = fig.legend(
    all_handles,
    all_labels,
    loc="upper left",
    bbox_to_anchor=(ax_bbox.x1, ax_bbox.y1 + 0.017),  # Slight gap from subplot edge
    frameon=True,
)

# Export the plot to default file format (usually PDF/PNG)
# Use bbox_extra_artists and bbox_inches='tight' to include the legend
fig.savefig("test.png", bbox_extra_artists=[leg], bbox_inches="tight", dpi=300)

# Display the plot in the output
plt.show()

# %%
