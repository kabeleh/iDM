#!/usr/bin/env python3
"""Generate additional Cobaya YAML files from existing base MCMC YAMLs.

This script scans a directory of base YAML runs (e.g. *_Planck_*_MCMC.yml) and
creates the additional files requested for each base run:

1) Base minimizer YAML
2) PP_DESI post-processing YAML
3) PPS_DESI post-processing YAML
4) Minimizer YAMLs for PP_DESI and PPS_DESI
5) Swampland post-processing YAMLs for base, PP_DESI, and PPS_DESI runs
6) Minimizer YAMLs for all swampland post-processing runs

Output paths are always derived from the base YAML's top-level `output:` value.
"""

import argparse
import copy
from collections import namedtuple
from pathlib import Path

import yaml


PostFlavor = namedtuple("PostFlavor", ["suffix", "extra_likelihoods"])

PP_DESI = PostFlavor("PP_DESI", ("bao.desi_dr2", "sn.pantheonplus"))
PPS_DESI = PostFlavor("PPS_DESI", ("bao.desi_dr2", "sn.pantheonplusshoes"))


SWAMP_PARAMS = {
    "dSC_tolerance": 0.1,
    "dSC_target_abs": 1.0,
    "dSC_sigma": 0.5,
}

SWAMP_LIKELIHOODS = {
    "DC": "lambda phi_scf_range: 0.0 if phi_scf_range <= 10.0 else -1e10",
    "SWGC": "lambda swgc_expr_min: 0.0 if swgc_expr_min >= 0.0 else -1e10",
    "dSC": (
        "lambda dV_V_scf_min, ddV_V_at_dV_V_min, ddV_V_scf_max, dV_V_at_ddV_V_max, "
        "dSC_tolerance, dSC_target_abs, dSC_sigma: (-1e10 if (dSC_tolerance <= 0.0 or "
        "dSC_target_abs <= dSC_tolerance or dSC_sigma <= 0.0) else (-0.5 * (max(0.0, "
        "np.log10(dSC_target_abs / max(1e-300, max(max(max(1e-300, dV_V_scf_min), "
        "max(1e-300, -ddV_V_at_dV_V_min)), max(max(1e-300, -ddV_V_scf_max), "
        "max(1e-300, dV_V_at_ddV_V_max)))))) / dSC_sigma) ** 2))"
    ),
}


def is_base_mcmc_yaml(path):
    """Return True if filename matches a base MCMC run YAML."""
    name = path.name
    if not name.endswith("_MCMC.yml"):
        return False
    disallowed = ("_minimizer", "_Swamp", "_PP_", "_PPS_", "_SPA_", "_PP_D", "_PP_S_D")
    return not any(token in name for token in disallowed)


def potential_from_stem(stem):
    """Extract potential name from '<potential>_Planck_...' stem."""
    if "_Planck_" not in stem:
        raise ValueError("Cannot infer potential from stem without '_Planck_': {}".format(stem))
    return stem.split("_Planck_", 1)[0]


def post_filename_from_base(base_stem, flavor):
    """Build post-processing filename from base stem.

    Uses existing pNG naming convention in this workspace:
    - PP_DESI post:  pNG_Planck_PP_... (without _DESI)
    - PPS_DESI post: pNG_Planck_PPS_... (without _DESI)
    """
    potential = potential_from_stem(base_stem)
    if potential == "pNG":
        tag = "PP" if flavor.suffix == "PP_DESI" else "PPS"
    else:
        tag = flavor.suffix
    return base_stem.replace("_Planck_", "_Planck_{}_".format(tag), 1) + ".yml"


def post_minimizer_filename_from_base(base_stem, flavor):
    """Build minimizer filename for PP/PPS post runs.

    These minimizer filenames consistently use *_PP_DESI_* / *_PPS_DESI_*.
    """
    post_stem = base_stem.replace("_Planck_", "_Planck_{}_".format(flavor.suffix), 1)
    return "{}_minimizer.yml".format(post_stem)


def make_post_config(base_output, flavor):
    return {
        "post": {
            "suffix": flavor.suffix,
            "add": {
                "likelihood": dict((lk, None) for lk in flavor.extra_likelihoods),
            },
        },
        "output": base_output,
    }


def make_minimizer_config(base_cfg, output):
    cfg = copy.deepcopy(base_cfg)
    cfg["sampler"] = {
        "minimize": {
            "best_of": 10,
            "ignore_prior": True,
            "method": "scipy",
        }
    }
    cfg["output"] = output
    cfg.pop("post", None)
    return cfg


def add_likelihoods(cfg, names):
    like = cfg.setdefault("likelihood", {})
    if not isinstance(like, dict):
        raise ValueError("Expected 'likelihood' to be a mapping")
    for key in names:
        like[key] = None


def add_swamp_to_minimizer(cfg):
    params = cfg.setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError("Expected 'params' to be a mapping")
    params.update(copy.deepcopy(SWAMP_PARAMS))

    like = cfg.setdefault("likelihood", {})
    if not isinstance(like, dict):
        raise ValueError("Expected 'likelihood' to be a mapping")
    like.update(SWAMP_LIKELIHOODS)


def make_swamp_post_config(target_output):
    return {
        "post": {
            "suffix": "Swamp",
            "add": {
                "params": copy.deepcopy(SWAMP_PARAMS),
                "likelihood": copy.deepcopy(SWAMP_LIKELIHOODS),
            },
        },
        "output": target_output,
    }


def write_yaml(path, data, overwrite):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    return True


def generate_for_base(base_path, overwrite):
    """Generate all derived YAMLs for one base run.

    Returns (created, skipped) file path lists.
    """
    with base_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    if not isinstance(base_cfg, dict):
        raise ValueError("Unexpected YAML root type in {}".format(base_path))

    if "output" not in base_cfg:
        raise ValueError("Base YAML has no top-level 'output': {}".format(base_path))

    base_output = str(base_cfg["output"])
    base_stem = base_path.stem
    parent = base_path.parent

    created = []
    skipped = []

    targets = []

    # 1) Base minimizer
    base_min_file = parent / "{}_minimizer.yml".format(base_stem)
    base_min_cfg = make_minimizer_config(base_cfg, output=base_output)
    targets.append((base_min_file, base_min_cfg))

    # 2) PP_DESI and PPS_DESI post + minimizer
    for flavor in (PP_DESI, PPS_DESI):
        post_file = parent / post_filename_from_base(base_stem, flavor)
        post_cfg = make_post_config(base_output, flavor)
        targets.append((post_file, post_cfg))

        post_output = "{}.post.{}".format(base_output, flavor.suffix)
        post_min_file = parent / post_minimizer_filename_from_base(base_stem, flavor)
        post_min_cfg = make_minimizer_config(base_cfg, output=post_output)
        add_likelihoods(post_min_cfg, flavor.extra_likelihoods)
        targets.append((post_min_file, post_min_cfg))

        # 3) Swampland post for PP/PPS run
        post_swamp_file = parent / "{}_Swamp.yml".format(post_file.stem)
        post_swamp_cfg = make_swamp_post_config(target_output=post_output)
        targets.append((post_swamp_file, post_swamp_cfg))

        # 4) Minimizer for PP/PPS Swampland run
        post_swamp_min_file = parent / "{}_Swamp_minimizer.yml".format(post_file.stem)
        post_swamp_min_cfg = make_minimizer_config(
            base_cfg, output="{}.post.Swamp".format(post_output)
        )
        add_likelihoods(post_swamp_min_cfg, flavor.extra_likelihoods)
        add_swamp_to_minimizer(post_swamp_min_cfg)
        targets.append((post_swamp_min_file, post_swamp_min_cfg))

    # 5) Swampland post for base run
    base_swamp_file = parent / "{}_Swamp.yml".format(base_stem)
    base_swamp_cfg = make_swamp_post_config(target_output=base_output)
    targets.append((base_swamp_file, base_swamp_cfg))

    # 6) Minimizer for base Swampland run
    base_swamp_min_file = parent / "{}_Swamp_minimizer.yml".format(base_stem)
    base_swamp_min_cfg = make_minimizer_config(base_cfg, output="{}.post.Swamp".format(base_output))
    add_swamp_to_minimizer(base_swamp_min_cfg)
    targets.append((base_swamp_min_file, base_swamp_min_cfg))

    for path, cfg in targets:
        changed = write_yaml(path, cfg, overwrite=overwrite)
        if changed:
            created.append(path)
        else:
            skipped.append(path)

    return created, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate minimizer/post/swampland Cobaya YAMLs from base MCMC YAMLs"
    )
    parser.add_argument(
        "--dir",
        default="iDM/Cobaya/MCMC",
        help="Directory containing base YAML files (default: iDM/Cobaya/MCMC)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing generated files",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of base YAML stems to process (without .yml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mcmc_dir = Path(args.dir).resolve()
    if not mcmc_dir.exists():
        raise FileNotFoundError("Directory does not exist: {}".format(mcmc_dir))

    base_files = sorted([p for p in mcmc_dir.glob("*.yml") if is_base_mcmc_yaml(p)])
    if args.only:
        only_set = set(args.only)
        base_files = [p for p in base_files if p.stem in only_set]

    print("Found {} base YAML file(s) in {}".format(len(base_files), mcmc_dir))
    if not base_files:
        return 0

    total_created = 0
    total_skipped = 0

    for base in base_files:
        created, skipped = generate_for_base(base, overwrite=args.overwrite)
        total_created += len(created)
        total_skipped += len(skipped)
        print("[{}] created={} skipped={}".format(base.name, len(created), len(skipped)))

    print("Done. created={} skipped={}".format(total_created, total_skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
