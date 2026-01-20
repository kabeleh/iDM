import glob
import os
from cobaya.samplers.mcmc import plot_progress
import matplotlib.pyplot as plt


def has_progress_data(progress_file):
    """Check if a .progress file has data beyond the header line."""
    with open(progress_file, "r") as f:
        lines = f.readlines()
        # File needs at least 2 lines (header + data)
        return len(lines) >= 2


# Folder containing MCMC chains
mcmc_folder = "/Users/klmba/kDrive/Sci/PhD/Research/HDM/MCMCfast"

# Find all .progress files in the folder
progress_files = glob.glob(os.path.join(mcmc_folder, "*.progress"))

if not progress_files:
    print(f"No .progress files found in {mcmc_folder}")
else:
    print(f"Found {len(progress_files)} chain(s) with .progress files:")

    # Separate chains with data from empty ones
    chains_with_data = []
    chains_empty = []

    for pf in progress_files:
        chain_path = pf.rsplit(".progress", 1)[0]
        chain_name = os.path.basename(chain_path)
        if has_progress_data(pf):
            chains_with_data.append((pf, chain_path, chain_name))
            print(f"  - {chain_name} (has data)")
        else:
            chains_empty.append(chain_name)
            print(f"  - {chain_name} (empty - header only)")

    if chains_empty:
        print(f"\nSkipping {len(chains_empty)} empty chain(s)")

    if not chains_with_data:
        print("\nNo chains with progress data to plot.")
    else:
        # Plot each chain that has data
        for pf, chain_path, chain_name in chains_with_data:
            try:
                print(f"\nPlotting progress for: {chain_name}")
                plot_progress(chain_path)
                plt.suptitle(chain_name, fontsize=10)
                plt.tight_layout()
            except Exception as e:
                print(f"Error plotting {chain_name}: {e}")

        plt.show()
