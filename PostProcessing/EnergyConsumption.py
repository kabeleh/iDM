#!/usr/bin/env python3
"""
Calculate total energy consumption and CO2 emissions from SLURM output files.

This script parses .out files containing SLURM job statistics, extracts
ConsumedEnergyRaw values (in Joules), and calculates total energy consumption
in kWh and equivalent CO2 emissions.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List


def parse_energy_from_file(filepath: str) -> List[int]:
    """
    Parse a SLURM output file and extract all ConsumedEnergyRaw values.

    Args:
        filepath: Path to the .out file

    Returns:
        List of energy values in Joules
    """
    energy_values: List[int] = []
    in_table = False

    try:
        with open(filepath, "r") as f:
            for line in f:
                # Check if we've found the header line
                if "ConsumedEnergyRaw" in line and "JobID" in line:
                    in_table = True
                    continue

                # Skip the separator line
                if in_table and "----" in line:
                    continue

                # If we're in a table and hit an empty line, we're done with this table
                if in_table and line.strip() == "":
                    in_table = False
                    continue

                # Parse data lines
                if in_table:
                    parts = line.split()
                    if len(parts) >= 6:  # Should have at least 6 columns
                        try:
                            energy_str = parts[-1]  # ConsumedEnergyRaw is last column
                            energy = int(energy_str)
                            energy_values.append(energy)
                        except (ValueError, IndexError):
                            # Skip lines that don't have valid energy values
                            pass

    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}", file=sys.stderr)

    return energy_values


def main() -> None:
    """Main function to process folder and calculate emissions."""

    # Get folder path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python EnergyConsumption.py <folder_path>")
        print("\nExample:")
        print("  python EnergyConsumption.py /path/to/compute_logs")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all .out files
    out_files = glob.glob(os.path.join(folder_path, "*.out"))

    if not out_files:
        print(f"No .out files found in {folder_path}")
        sys.exit(0)

    print(f"Found {len(out_files)} .out file(s) in {folder_path}")
    print()

    # Process all files
    total_energy_joules = 0
    file_count = 0

    for filepath in sorted(out_files):
        energy_values = parse_energy_from_file(filepath)
        if energy_values:
            file_energy = sum(energy_values)
            total_energy_joules += file_energy
            file_count += 1
            print(
                f"{Path(filepath).name:60s}: {file_energy:15,} J  ({len(energy_values)} entries)"
            )

    print()
    print("=" * 80)
    print(f"Files processed: {file_count}")
    print()

    # Convert Joules to kWh
    # 1 Joule = 1 Watt-second
    # 1 kWh = 1000 Watt-hours = 1000 * 3600 Watt-seconds = 3,600,000 Joules
    joules_per_kwh = 3_600_000
    total_energy_kwh = total_energy_joules / joules_per_kwh

    # Calculate CO2 equivalent
    # 1 kWh = 0.326 kg CO2e for 2025 Luxembourg electricity mix (https://lowcarbonpower.org/region/Luxembourg)
    kg_co2_per_kwh = 0.326
    total_co2_kg = total_energy_kwh * kg_co2_per_kwh
    total_co2_tons = total_co2_kg / 1000

    # Print results
    print(f"Total energy consumed:  {total_energy_joules:15,} J")
    print(f"                        {total_energy_kwh:15.3f} kWh")
    print()
    print(f"CO2 equivalent:         {total_co2_kg:15.3f} kg CO2e")
    print(f"                        {total_co2_tons:15.6f} tons CO2e")
    print("=" * 80)


if __name__ == "__main__":
    main()
