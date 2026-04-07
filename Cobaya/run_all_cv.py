"""Generate all YAML configs for mcmc_fast / CMB / uncoupled across potentials."""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))

from create_cobaya_yaml import (
    create_cobaya_yaml,
    build_filename_stem,
    build_chain_output_stem,
)
from ruamel.yaml import YAML

yaml = YAML()
yaml.indent(mapping=2, sequence=2, offset=2)

SAMPLER = "mcmc_fast"
LIKELIHOOD = "CMB"
COUPLING = "uncoupled"

# All potentials with attractor=no
NON_ATTRACTOR = [
    "power-law",
    "cosine",
    "hyperbolic",
    "pNG",
    "SqE",
    "exponential",
    "Bean",
    "DoubleExp",
]

# Potentials that support attractor=yes
ATTRACTOR = ["hyperbolic"]


def write_yaml(potential, attractor):
    # Bean generates both BeanSingleWell and BeanAdS
    variants = ["Bean", "BeanAdS"] if potential == "Bean" else [potential]

    for variant in variants:
        config = create_cobaya_yaml(SAMPLER, LIKELIHOOD, variant, attractor, COUPLING)

        filename = (
            build_filename_stem(variant, LIKELIHOOD, attractor, COUPLING, SAMPLER)
            + ".yml"
        )

        # Set output path
        output_stem = build_chain_output_stem(
            variant, LIKELIHOOD, attractor, COUPLING, SAMPLER
        )
        config["output"] = "/project/home/p201176/" + output_stem

        yaml_path = os.path.join(os.path.dirname(__file__), "MCMC", filename)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        # Post-process bracket notation for z and R lists
        with open(yaml_path, "r") as f:
            content = f.read()
        content = re.sub(r"(\s+)z:\s*\n\s*-\s*([0-9.]+)", r"\1z: [\2]", content)
        content = re.sub(r"(\s+)R:\s*\n\s*-\s*([0-9.]+)", r"\1R: [\2]", content)
        with open(yaml_path, "w") as f:
            f.write(content)

        print(f"  Written: MCMC/{filename}")


print("=== Non-attractor runs (attractor=no) ===")
for pot in NON_ATTRACTOR:
    print(f"\n--- {pot} ---")
    write_yaml(pot, "no")

print("\n=== Attractor runs (attractor=yes) ===")
for pot in ATTRACTOR:
    print(f"\n--- {pot} ---")
    write_yaml(pot, "yes")

print("\nDone.")
