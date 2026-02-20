import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples
from getdist import plots

# Load the raw data (assuming standard GetDist/CosmoMC format)
# Columns: weight, -log(Post), param1, param2...
data = np.loadtxt(
    "/home/kl/kDrive/Sci/PhD/Research/HDM/MCMC_chains/cobaya_mcmc_fast_CMB_hyperbolic_InitCond_uncoupled.1.txt"
)
weights = data[:, 0]
loglikes = data[:, 1]
samples_array = data[:, 2:]

# 3. Determine the number of parameters
num_params = samples_array.shape[1]

# 4. Generate parameter names that match the column count
# You can replace these with your actual names, but the length must be num_params
names = [f"x{i}" for i in range(num_params)]
labels = [f"x_{i}" for i in range(num_params)]

# Initialize MCSamples
# ignore_rows=0.3 removes the first 30% as burn-in
samples = MCSamples(
    samples=samples_array,
    weights=weights,
    loglikes=loglikes,
    names=names,
    labels=labels,
    settings={"ignore_rows": 0.0},
)

# # Get the mean of the first parameter 'x0'
# mean_x0 = samples.mean("x2")
# print(f"Mean of H0: {mean_x0}")

# mean_x31 = samples.mean("x31")
# print(f"Mean of S8: {mean_x31}")


# Update settings on your samples object
samples.updateSettings(
    {
        "fine_bins_2D": 512,  # Increase from default 256
        "fine_bins": 2048,  # Increase from default 1024
        "smooth_scale_2D": 0.3,  # Manual smoothing for 2D
        "smooth_scale_1D": 0.3,  # Manual smoothing for 1D
    }
)
# This requires multiple chains (e.g., .1.txt, .2.txt) to be loaded
# If you only have one chain, it will assess internal split-convergence
# print(f"Gelman-Rubin R-1: {samples.getGelmanRubin()}")


# Extract the parameter vector
p_data = samples["x31"]

plt.figure(figsize=(10, 4))
plt.plot(p_data)
plt.xlabel("Sample Index")
plt.ylabel("x2 Value")
plt.title("Trace Plot for x2")
plt.show()

g = plots.get_single_plotter(width_inch=5)
# This plots the actual points from the chain
g.plot_2d_scatter(samples, "x2", "x31", color="blue", alpha=0.3)
plt.show()

# Now regenerate the plot
g = plots.get_subplot_plotter()
g.triangle_plot(samples, params=["x2", "x31"], filled=True)
plt.show()
# 4. Optional: Save the plot to a file
# g.export('my_triangle_plot.pdf')
