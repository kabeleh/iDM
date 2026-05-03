#!/usr/bin/env python3
"""
Posterior background heatmaps from MCMC chains.

Overview
--------
This script builds posterior-density heatmaps for background quantities from
chain samples using mode-aware resampling plus persistent CLASS background
caching. For each requested dataset root, the pipeline is:

1. Load post-burn rows from chain files.
2. Build or reuse a sampling plan (mode-aware weighted resampling).
3. Replay trajectories through background-only CLASS (with cache).
4. Accumulate 2D histograms and pointwise weighted summaries.
5. Overlay best-fit trajectory and save figure bundle (.png/.pdf/.pgf).
6. Save NPZ payload with histogram and summary arrays.

Default behavior
----------------
If none of --preset, --quantities, --quantity is provided, the script uses
the publication default preset: --preset all.

Precedence is:
1) --preset
2) --quantities
3) --quantity
4) implicit default --preset all

Layout defaults for multi-panel diagnostics:
- dSC pair (s1 + minus_s2): --dsc-layout split
- SWGC triplet (swgc_lhs + swgc_rhs + swgc_residual): --swgc-layout stacked

Quantity names
--------------
Valid quantity tokens for --quantity/--quantities:
- phi_scf
- w
- Omega_cdm
- Omega_scf
- s1
- minus_s2
- swampland_expr
- swgc_lhs
- swgc_rhs
- swgc_residual

Preset names
------------
Valid --preset options:
- phi
- eos
- omega
- swampland
- swgc
- all

Clean examples
--------------
Minimal (uses default preset=all):
python3 PostProcessing/posterior_background_heatmaps.py \
        --roots hyperbolic_PP_D_InitCond_MCMC

Single preset:
python3 PostProcessing/posterior_background_heatmaps.py \
        --roots hyperbolic_PP_D_InitCond_MCMC \
        --preset swgc \
        --max-samples 3000 \
        --num-threads 0

Explicit quantities (single pass with shared cached trajectories):
python3 PostProcessing/posterior_background_heatmaps.py \
        --roots hyperbolic_PP_D_InitCond_MCMC \
        --quantities phi_scf w swgc_lhs swgc_rhs swgc_residual \
        --max-samples 2000

Two roots with parallel root and trajectory workers:
python3 PostProcessing/posterior_background_heatmaps.py \
        --roots hyperbolic_PP_D_InitCond_MCMC hyperbolic_PP_S_D_InitCond_MCMC \
        --preset all \
        --num-roots 2 \
        --num-threads 8 \
        --max-samples 5000

Dry-run (inspect sampling stats, skip CLASS/figures):
python3 PostProcessing/posterior_background_heatmaps.py \
        --roots hyperbolic_PP_D_InitCond_MCMC \
        --dry-run

Complete flag reference
-----------------------
Core selection and geometry:
- --roots ROOT [ROOT ...]
    Chain roots to process (without extension). Default: DEFAULT_ROOTS.
- --quantity {phi_scf,w,Omega_cdm,Omega_scf,s1,minus_s2,swampland_expr,swgc_lhs,swgc_rhs,swgc_residual}
    Single quantity (legacy override).
- --quantities QUANTITY [QUANTITY ...]
    One or more quantities (same choices as --quantity); overrides --quantity.
- --preset {phi,eos,omega,swampland,swgc,all}
    Named preset; overrides --quantity and --quantities.
- --x-bins INT
    Number of bins in x=log10(1+z). Default: 320.
- --y-bins INT
    Number of bins in y. Default: 320.
- --x-min FLOAT
    Minimum x=log10(1+z). Default: 0.0.
- --x-max FLOAT
    Maximum x=log10(1+z). Default: 14.0.

Resampling and chain handling:
- --ignore-rows FLOAT
    Burn-in fraction removed from each chain file. Default: 0.33.
- --max-samples INT
    Number of posterior-resampled trajectories per dataset. Default: 2000.
- --seed INT
    Random seed for weighted posterior resampling. Default: 12345.

Mode-aware sampling controls:
- --mode-detect-bins INT
    Bins per axis for connected-component mode detection. Default: 20.
- --mode-min-mass-frac FLOAT
    Minimum posterior mass for guaranteed floor allocation. Default: 0.003.
- --mode-floor-abs INT
    Absolute guaranteed draws per eligible mode. Default: 12.
- --mode-floor-frac FLOAT
    Fractional guaranteed draws per mode as a fraction of --max-samples.
    Default: 0.003.
- --mode-floor-cap-frac FLOAT
    Cap on total guaranteed floor allocation (fraction of --max-samples).
    Default: 0.20.
- --sample-plan-cache-mode {auto,refresh,locked}
    Sampling-plan cache policy:
    auto    -> reuse plan when fingerprint matches (default)
    refresh -> force redraw and overwrite cached plan
    locked  -> require matching cached plan, fail otherwise

Backward-compatible HPD parse/fingerprint inputs (no HPD prefiltering):
- --hpd-params P1 P2 P3
    Default: scf_c2 cdm_c phi_ini_scf_ic.
- --hpd-mass FLOAT
    Default: 0.68.
- --hpd-bins INT
    Default: 48.

I/O and execution:
- --cache-dir PATH
    Persistent background cache directory.
    Default: PostProcessing/background_posterior_cache.
- --output-dir PATH
    Output directory for figure bundles and NPZ files.
    Default: PostProcessing/PosteriorHeatmaps.
- --failure-audit-dir PATH
    Directory for structured failure audit artifacts (.ini + .json).
    Default: PostProcessing/background_failure_audit.
- --dry-run
    Print selection/sampling stats only; skip CLASS runs and plotting.
- --num-threads INT
    Trajectory-level workers per dataset. 0 means all cores. Default: 1.
- --num-roots INT
    Root-level worker processes. 0 means all cores. Default: 1.

Plot styling and overlays:
- --zero-label-overlap-margin-px FLOAT
    Pixel margin for pruning overlapping y labels near zero on phi symlog axes.
    Default: 0.6.
- --phi-y-scale {symlog,symlog2}
    y-scaling policy for phi_scf when sign changes occur. Default: symlog2.
- --include-legends-in-plots
    If set, draw legends inside plots. Otherwise save standalone legend files.

dSC and SWGC layout options:
- --dsc-layout {split,combined}
    split    -> separate s1 and -s2 outputs (default)
    combined -> legacy overlay panel
- --swgc-layout {stacked,combined}
    stacked  -> four-panel SWGC layout (lhs, rhs, residual, probability) (default)
    combined -> legacy SWGC combined figure

SWGC robust sign/probability controls:
- --swgc-crossing-epsilon-abs FLOAT
    Absolute epsilon floor for SWGC residual sign classification. Default: 1e-20.
- --swgc-crossing-epsilon-rel FLOAT
    Relative epsilon multiplier applied to local median max(|lhs|,|rhs|).
    Default: 0.02.
- --swgc-crossing-epsilon-quantile FLOAT
    Weighted quantile term for adaptive epsilon from |Delta_SWGC|. Default: 0.02.

Phi crossing diagnostics:
- --phi-crossing-overlay {none,probability,binary}
    none        -> no overlay
    probability -> overlay crossing probability curve (default)
    binary      -> highlight intervals above threshold
- --phi-crossing-binary-threshold FLOAT
    Threshold used by binary mode. Default: 0.5.
- --phi-crossing-epsilon-mode {adaptive,fixed,linthresh}
    adaptive  -> max(abs floor, linthresh-scaled floor, weighted quantile)
    fixed     -> absolute epsilon only
    linthresh -> linthresh-scaled epsilon floor
- --phi-crossing-epsilon-abs FLOAT
    Absolute epsilon floor for phi sign classification. Default: 1e-14.
- --phi-crossing-epsilon-linthresh-frac FLOAT
    Linthresh multiplier used in linthresh/adaptive modes. Default: 0.02.
- --phi-crossing-epsilon-quantile FLOAT
    Weighted |phi| quantile used in adaptive mode. Default: 0.02.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc
from matplotlib import patheffects as pe
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)

from BestFitPlot import (
    extract_class_parameters,
    load_background_dataset,
    load_bestfit_file,
    load_input_yaml,
)

# Keep plotting dependencies lightweight and robust on clusters/headless runs.
mpl.use("Agg")

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11.5,
        "axes.labelsize": 12,
        "axes.formatter.useoffset": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.85,
        "axes.linewidth": 0.95,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.8,
        "ytick.major.size": 4.8,
        "xtick.minor.size": 2.4,
        "ytick.minor.size": 2.4,
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.minor.width": 0.72,
        "ytick.minor.width": 0.72,
        "lines.markersize": 4,
        "errorbar.capsize": 3,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        **(
            {
                "text.usetex": True,
                "pgf.preamble": (
                    r"\usepackage{fontspec}"
                    r"\usepackage{mathtools}"
                    r"\usepackage{amsfonts}"
                    r"\usepackage{amssymb}"
                    r"\usepackage[warnings-off={mathtools-overbracket,mathtools-colon}]{unicode-math}"
                    r"\setmainfont{IBM Plex Serif}"
                    r"\setsansfont{IBM Plex Sans}"
                    r"\setmonofont{IBM Plex Mono}"
                    r"\setmathfont{IBM Plex Math}"
                ),
                "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
                "pgf.texsystem": "lualatex",
                "pgf.rcfonts": False,
            }
            if shutil.which("latex")
            else {
                "text.usetex": False,
                "mathtext.fontset": "cm",
            }
        ),
    }
)

_HEATMAP_CMAP = getattr(cmc, "batlowK")
_OMEGA_DM_CMAP = getattr(cmc, "lajolla", _HEATMAP_CMAP)
_OMEGA_PHI_CMAP = getattr(cmc, "oslo", _HEATMAP_CMAP)
_HIGH_CONTRAST_PALETTE: dict[str, str] = {
    "black": "#000000",
    "blue": "#004488",
    "red": "#BB5566",
    "teal": "#009988",
    "yellow": "#DDAA33",
    "white": "#FFFFFF",
}


@dataclass
class ChainBundle:
    root: str
    bestfit_file: Path
    input_yaml: Path
    chain_files: list[Path]


@dataclass
class SelectionData:
    # Per-row arrays after burn-in, concatenated over files in file order.
    weights: np.ndarray
    minuslogpost: np.ndarray
    hpd_values: np.ndarray  # shape (N, 3)
    file_counts_postburn: list[int]


@dataclass
class DrawRecord:
    file_index: int
    row_index_postburn: int
    multiplicity: int


@dataclass
class TrajectoryResult:
    """Result from processing a single trajectory: interpolated quantities and stats."""

    qty_interps: dict[str, np.ndarray]
    multiplicity: int
    cache_hit: bool


class ClassBackgroundRunError(RuntimeError):
    """Structured error for a failed CLASS background-only replay."""

    def __init__(
        self,
        message: str,
        *,
        cache_key: str,
        ini_path: Path,
        ini_text: str,
        returncode: int,
        stdout_tail: str,
        stderr_tail: str,
    ) -> None:
        super().__init__(message)
        self.cache_key = cache_key
        self.ini_path = ini_path
        self.ini_text = ini_text
        self.returncode = returncode
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail


_FIELDS_TO_CACHE: tuple[str, ...] = (
    "z",
    "a",
    "phi_scf",
    "w",
    "Omega_cdm",
    "Omega_scf",
    "Omega_m_class",
    "rho_cdm",
    "rho_scf",
    "rho_crit",
    "p_scf",
    "V",
    "dV",
    "d2V",
    "d3V",
    "d4V",
    "s1",
    "minus_s2",
    "swampland_expr",
    "swgc_lhs",
    "swgc_rhs",
    "swgc_residual",
)

# Default roots mirror the current hyperbolic set used in data_postprocessing_getDist.py
DEFAULT_ROOTS: tuple[str, ...] = (
    "hyperbolic_PP_D_InitCond_MCMC",
    "hyperbolic_PP_S_D_InitCond_MCMC",
    "hyperbolic_Planck_InitCond_MCMC.post.Swamp",
    "hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC",
    "hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC",
)

_ALL_QUANTITIES: list[str] = [
    "phi_scf",
    "w",
    "Omega_cdm",
    "Omega_scf",
    "s1",
    "minus_s2",
    "swampland_expr",
    # Strong Scalar Weak Gravity Conjecture (SWGC / plot 6 in BestFitPlot.py)
    # Condition: (V'')^2 <= 2(V''')^2 - V''*V'''' i.e. swgc_residual >= 0
    "swgc_lhs",  # (V'')^2
    "swgc_rhs",  # 2(V''')^2 - V''*V''''
    "swgc_residual",  # rhs - lhs  (positive = SWGC satisfied)
]

# Named presets for --preset; each expands to a list of quantities plotted in one pass.
_PRESET_QUANTITIES: dict[str, list[str]] = {
    "phi": ["phi_scf"],
    "eos": ["w"],
    "omega": ["Omega_cdm", "Omega_scf"],
    "swampland": ["s1", "minus_s2"],
    "swgc": ["swgc_lhs", "swgc_rhs", "swgc_residual"],
    "all": [
        "phi_scf",
        "w",
        "Omega_cdm",
        "Omega_scf",
        "s1",
        "minus_s2",
        "swgc_lhs",
        "swgc_rhs",
        "swgc_residual",
    ],
}

_OUTPUT_QUANTITY_ALIASES: dict[str, str] = {
    "phi_scf": "phi",
    "w": "w",
    "Omega_cdm": "Omega",
    "Omega_scf": "Omega",
    "s1": "dSC_s1",
    "minus_s2": "dSC_s2",
    "swampland_expr": "dSC_expr",
    "swgc_lhs": "swgc_lhs",
    "swgc_rhs": "swgc_rhs",
    "swgc_residual": "swgc",
}

_MODE_SAMPLING_SCHEMA_VERSION = "mode-aware-defaults-v1-20260502"


def _resolve_mode_sampling_tuning(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve concrete mode-aware sampling defaults from CLI knobs.

    These diagnostics are logged at startup so tuning is transparent and
    reproducible across reruns.
    """
    max_samples = int(max(1, args.max_samples))
    floor_abs = int(max(1, args.mode_floor_abs))
    floor_frac = float(max(0.0, args.mode_floor_frac))
    floor_cap_frac = float(min(1.0, max(0.0, args.mode_floor_cap_frac)))

    floor_from_frac = int(np.round(floor_frac * max_samples))
    floor_per_mode = int(max(floor_abs, floor_from_frac))
    floor_total_cap = int(max(1, np.floor(floor_cap_frac * max_samples)))
    max_floor_modes = int(max(1, floor_total_cap // max(floor_per_mode, 1)))

    return {
        "schema_version": _MODE_SAMPLING_SCHEMA_VERSION,
        "max_samples": max_samples,
        "mode_detect_bins": int(max(4, args.mode_detect_bins)),
        "mode_min_mass_frac": float(max(0.0, args.mode_min_mass_frac)),
        "mode_floor_abs": floor_abs,
        "mode_floor_frac": floor_frac,
        "mode_floor_from_frac": floor_from_frac,
        "mode_floor_per_mode": floor_per_mode,
        "mode_floor_cap_frac": floor_cap_frac,
        "mode_floor_total_cap": floor_total_cap,
        "mode_floor_max_modes": max_floor_modes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-dataset posterior heatmaps for background quantities "
            "using mode-aware posterior resampling, persistent CLASS caching, "
            "best-fit overlays, and pointwise credible bands."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Chain roots to process (without extension).",
    )
    parser.add_argument(
        "--hpd-params",
        nargs=3,
        default=["scf_c2", "cdm_c", "phi_ini_scf_ic"],
        help="Three parameters used for HPD-like region selection.",
    )
    parser.add_argument(
        "--hpd-mass",
        type=float,
        default=0.68,
        help="Target posterior mass for HPD-like region (default: 0.68).",
    )
    parser.add_argument(
        "--hpd-bins",
        type=int,
        default=48,
        help="Bins per axis for 3D histogram density surrogate (default: 48).",
    )
    parser.add_argument(
        "--ignore-rows",
        type=float,
        default=0.33,
        help="Burn-in fraction removed from each chain file (default: 0.33).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help=(
            "Posterior-resampled trajectories per dataset. "
            "Larger values increase runtime approximately linearly."
        ),
    )
    parser.add_argument(
        "--mode-detect-bins",
        type=int,
        default=20,
        help=(
            "Bins per axis for connected-component mode detection in the "
            "3-parameter control space (default: 20)."
        ),
    )
    parser.add_argument(
        "--mode-min-mass-frac",
        type=float,
        default=0.003,
        help=(
            "Minimum posterior mass fraction for a mode to receive a guaranteed "
            "floor allocation (default: 0.003)."
        ),
    )
    parser.add_argument(
        "--mode-floor-abs",
        type=int,
        default=12,
        help=(
            "Absolute floor for guaranteed per-mode draws before mass-proportional "
            "allocation (default: 12)."
        ),
    )
    parser.add_argument(
        "--mode-floor-frac",
        type=float,
        default=0.003,
        help=(
            "Fractional floor for guaranteed per-mode draws, as a fraction of "
            "--max-samples (default: 0.003)."
        ),
    )
    parser.add_argument(
        "--mode-floor-cap-frac",
        type=float,
        default=0.20,
        help=(
            "Upper cap on total guaranteed floor allocation as a fraction of "
            "--max-samples (default: 0.20)."
        ),
    )
    parser.add_argument(
        "--sample-plan-cache-mode",
        choices=["auto", "refresh", "locked"],
        default="auto",
        help=(
            "Sampling-plan cache policy: auto=reuse when chain fingerprint matches, "
            "refresh=force redraw and overwrite cached plan, "
            "locked=require matching cached plan and fail otherwise."
        ),
    )
    parser.add_argument(
        "--quantity",
        choices=_ALL_QUANTITIES,
        default=None,
        help=(
            "Single background quantity (legacy override). "
            "If omitted and no --preset/--quantities are given, "
            "defaults to publication set (--preset all)."
        ),
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        choices=_ALL_QUANTITIES,
        default=None,
        metavar="QUANTITY",
        help=(
            "One or more quantities to plot in a single pass, reusing cached trajectories. "
            "Overrides --quantity."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=list(_PRESET_QUANTITIES.keys()),
        default=None,
        help=(
            "Named quantity preset: phi | eos | omega | swampland | swgc | all. "
            "Overrides --quantity and --quantities."
        ),
    )
    parser.add_argument(
        "--x-bins",
        type=int,
        default=320,
        help="Number of bins in log10(1+z) for the heatmap.",
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        default=320,
        help="Number of bins on the quantity axis for the heatmap.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=0.0,
        help="Minimum x = log10(1+z). Default 0 (z=0).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=14.0,
        help="Maximum x = log10(1+z). Default 14 (z~1e14).",
    )
    parser.add_argument(
        "--cache-dir",
        default="PostProcessing/background_posterior_cache",
        help="Persistent background cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="PostProcessing/PosteriorHeatmaps",
        help="Output directory for figures and histogram arrays.",
    )
    parser.add_argument(
        "--failure-audit-dir",
        default="PostProcessing/background_failure_audit",
        help=(
            "Directory for structured audit artifacts from failed CLASS background replays. "
            "Each failure writes a preserved .ini plus a JSON metadata record."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for weighted posterior resampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selection statistics; skip CLASS runs and plotting.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help=(
            "Number of worker threads for trajectory-level parallelization (CLASS runs per dataset). "
            "Default: 1 (serial). Recommended: min(N_cores, max_samples). "
            "Set to 0 for all available cores."
        ),
    )
    parser.add_argument(
        "--num-roots",
        type=int,
        default=1,
        help=(
            "Number of worker processes for root-level parallelization (datasets). "
            "Default: 1 (serial). Recommended: min(N_cores, len(roots)). "
            "Set to 0 for all available cores."
        ),
    )
    parser.add_argument(
        "--zero-label-overlap-margin-px",
        type=float,
        default=0.6,
        help=(
            "Pixel margin for hiding y-axis labels that overlap the y=0 label "
            "on phi_scf symlog plots. Smaller values hide fewer labels."
        ),
    )
    parser.add_argument(
        "--phi-y-scale",
        choices=["symlog", "symlog2"],
        default="symlog2",
        help=(
            "Y-axis scale for phi_scf when the range crosses zero. "
            "symlog=single symmetric log (default), "
            "symlog2=stronger far-outlier compression using a nested symmetric log."
        ),
    )
    parser.add_argument(
        "--include-legends-in-plots",
        action="store_true",
        help=(
            "Include legends inside plots (default: False). "
            "When False, legends are extracted to separate PNG/PDF/PGF files "
            "suitable for inclusion as standalone subfigures in LaTeX layouts."
        ),
    )
    parser.add_argument(
        "--dsc-layout",
        choices=["split", "combined"],
        default="split",
        help=(
            "Layout for dSC outputs when both s1 and minus_s2 are requested: "
            "split=default, produce separate s1 and -s2 figures with their "
            "assigned colormaps; combined=legacy single-panel overlay."
        ),
    )
    parser.add_argument(
        "--swgc-layout",
        choices=["stacked", "combined"],
        default="stacked",
        help=(
            "Layout for SWGC outputs when swgc_lhs, swgc_rhs, and swgc_residual are "
            "all requested: stacked=default four-panel layout (lhs, rhs, residual, "
            "crossing probability); combined=legacy two-panel overlay."
        ),
    )
    parser.add_argument(
        "--swgc-crossing-epsilon-abs",
        type=float,
        default=1e-20,
        help=(
            "Absolute epsilon floor for robust SWGC residual sign classification "
            "(default: 1e-20)."
        ),
    )
    parser.add_argument(
        "--swgc-crossing-epsilon-rel",
        type=float,
        default=0.02,
        help=(
            "Relative epsilon floor factor for SWGC residual crossing diagnostics; "
            "multiplies the local median max(|lhs|,|rhs|) (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--swgc-crossing-epsilon-quantile",
        type=float,
        default=0.02,
        help=(
            "Weighted quantile of |Delta_SWGC| used as an adaptive epsilon term at "
            "each z for SWGC crossing diagnostics (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--phi-crossing-overlay",
        choices=["none", "probability", "binary"],
        default="probability",
        help=(
            "Overlay robust zero-crossing diagnostics on phi plots: "
            "none=disable overlay, probability=plot crossing probability vs z, "
            "binary=highlight redshift intervals where crossing probability exceeds "
            "--phi-crossing-binary-threshold."
        ),
    )
    parser.add_argument(
        "--phi-crossing-binary-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for binary crossing overlay mode (default: 0.5). "
            "Intervals with crossing probability >= threshold are highlighted."
        ),
    )
    parser.add_argument(
        "--phi-crossing-epsilon-mode",
        choices=["adaptive", "fixed", "linthresh"],
        default="adaptive",
        help=(
            "Robust sign threshold policy for phi crossing diagnostics: "
            "adaptive=max(abs floor, linthresh-scaled floor, weighted |phi| quantile), "
            "fixed=constant absolute floor, linthresh=linthresh-scaled floor."
        ),
    )
    parser.add_argument(
        "--phi-crossing-epsilon-abs",
        type=float,
        default=1e-14,
        help=(
            "Absolute epsilon floor for robust sign classification in phi crossing "
            "diagnostics (default: 1e-14)."
        ),
    )
    parser.add_argument(
        "--phi-crossing-epsilon-linthresh-frac",
        type=float,
        default=0.02,
        help=(
            "Fraction of the phi symlog linthresh used as an epsilon floor in "
            "linthresh/adaptive modes (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--phi-crossing-epsilon-quantile",
        type=float,
        default=0.02,
        help=(
            "Weighted quantile of |phi| used as an adaptive epsilon term at each z "
            "(default: 0.02)."
        ),
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hdm_root() -> Path:
    # class_public/PostProcessing -> class_public -> HDM
    return Path(__file__).resolve().parents[2]


def _ordered_chain_search_dirs(hdm_root: Path) -> list[Path]:
    candidates = [hdm_root / "MCMC_archive", hdm_root / "MCMC_chains"]
    existing = [p for p in candidates if p.is_dir()]
    if hdm_root.is_dir():
        existing.append(hdm_root)
    return existing


def _root_from_match(path: Path, chain_dir: Path) -> str:
    rel = str(path.relative_to(chain_dir)).replace(os.sep, "/")
    rel = re.sub(r"\.\d+\.txt$", "", rel)
    rel = re.sub(r"\.txt$", "", rel)
    rel = re.sub(r"\.bestfit$", "", rel)
    rel = re.sub(r"\.input\.yaml$", "", rel)
    rel = re.sub(r"\.updated\.yaml$", "", rel)
    return rel


def resolve_root(root: str, hdm_root: Path) -> str:
    if "/" in root or os.sep in root:
        return root.replace(os.sep, "/")

    patterns = [
        f"**/{root}.*.txt",
        f"**/{root}.txt",
        f"**/{root}.bestfit",
        f"**/{root}.input.yaml",
    ]

    for search_dir in _ordered_chain_search_dirs(hdm_root):
        matches: list[Path] = []
        for pat in patterns:
            matches.extend(search_dir.glob(pat))
        if not matches:
            continue

        uniq = sorted(
            {_root_from_match(m, search_dir) for m in matches},
            key=lambda r: (1 if "/initialTesting/" in f"/{r}" else 0, r.count("/"), r),
        )

        basename_exact = [r for r in uniq if Path(r).name == root]
        if basename_exact:
            return basename_exact[0]
        return uniq[0]

    raise FileNotFoundError(f"Could not resolve chain root '{root}' under {hdm_root}")


def discover_chain_bundle(root: str, hdm_root: Path) -> ChainBundle:
    resolved = resolve_root(root, hdm_root)

    chain_files: list[Path] = []
    bestfit_file: Path | None = None
    input_yaml: Path | None = None

    for base in _ordered_chain_search_dirs(hdm_root):
        root_abs = base / resolved
        segment_files = sorted(root_abs.parent.glob(f"{root_abs.name}.[0-9]*.txt"))
        if segment_files:
            # Keep the first discovered chain segment set and avoid resetting it
            # in later base-directory iterations.
            if not chain_files:
                chain_files = segment_files

            # Do not use with_suffix here: roots can contain dots (e.g. .post.Swamp),
            # and with_suffix would incorrectly replace only the trailing component.
            bestfit_candidate = root_abs.parent / f"{root_abs.name}.bestfit"
            input_candidate = root_abs.parent / f"{root_abs.name}.input.yaml"
            if bestfit_candidate.exists():
                bestfit_file = bestfit_candidate
            if input_candidate.exists():
                input_yaml = input_candidate
            if bestfit_file is not None and input_yaml is not None:
                break

    if not chain_files:
        raise FileNotFoundError(f"No chain segment files found for root '{resolved}'")
    if bestfit_file is None:
        raise FileNotFoundError(f"No .bestfit file found for root '{resolved}'")
    if input_yaml is None:
        raise FileNotFoundError(f"No .input.yaml file found for root '{resolved}'")

    return ChainBundle(
        root=resolved,
        bestfit_file=bestfit_file,
        input_yaml=input_yaml,
        chain_files=chain_files,
    )


def read_chain_header(file_path: Path) -> list[str]:
    with open(file_path, "r") as f:
        header = f.readline().lstrip("#").strip().split()
    if not header:
        raise ValueError(f"Failed to read header from {file_path}")
    return header


def _load_selection_data(
    bundle: ChainBundle,
    hpd_params: list[str],
    ignore_rows: float,
) -> SelectionData:
    header = read_chain_header(bundle.chain_files[0])
    idx = {name: i for i, name in enumerate(header)}

    needed = ["weight", "minuslogpost", *hpd_params]
    missing = [name for name in needed if name not in idx]
    if missing:
        raise KeyError(
            f"Root '{bundle.root}' missing required columns: {missing}. "
            "Check parameter names in --hpd-params."
        )

    usecols = [idx[name] for name in needed]

    weights_parts: list[np.ndarray] = []
    minuslogpost_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    counts_postburn: list[int] = []

    for fp in bundle.chain_files:
        arr = np.loadtxt(fp, comments="#", usecols=usecols)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        burn = int(max(0, min(arr.shape[0], ignore_rows * arr.shape[0])))
        arr = arr[burn:, :]
        counts_postburn.append(arr.shape[0])

        if arr.shape[0] == 0:
            continue

        weights_parts.append(arr[:, 0])
        minuslogpost_parts.append(arr[:, 1])
        values_parts.append(arr[:, 2:5])

    if not weights_parts:
        raise ValueError(
            f"No post-burn rows available for root '{bundle.root}'. "
            "Lower --ignore-rows or inspect chain files."
        )

    weights = np.concatenate(weights_parts).astype(float, copy=False)
    minuslogpost = np.concatenate(minuslogpost_parts).astype(float, copy=False)
    hpd_values = np.concatenate(values_parts).astype(float, copy=False)

    return SelectionData(
        weights=weights,
        minuslogpost=minuslogpost,
        hpd_values=hpd_values,
        file_counts_postburn=counts_postburn,
    )


def _sanitize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0
    w[w < 0.0] = 0.0
    return w


def _draw_systematic_indices(
    indices: np.ndarray,
    probs: np.ndarray,
    draw_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draw_count = int(max(0, draw_count))
    if draw_count == 0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)

    cdf = np.cumsum(probs, dtype=float)
    if cdf[-1] <= 0:
        return np.empty(0, dtype=np.int64)
    cdf /= cdf[-1]
    positions = (rng.random() + np.arange(draw_count, dtype=float)) / float(draw_count)
    picked = np.searchsorted(cdf, positions, side="right")
    picked = np.clip(picked, 0, indices.size - 1)
    return indices[picked].astype(np.int64, copy=False)


def _connected_components_3d(mask: np.ndarray) -> np.ndarray:
    labels = np.full(mask.shape, -1, dtype=np.int32)
    comp_id = 0
    sx, sy, sz = mask.shape

    neighbors: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((dx, dy, dz))

    coords = np.argwhere(mask)
    for x, y, z in coords:
        if labels[x, y, z] != -1:
            continue
        stack = [(int(x), int(y), int(z))]
        labels[x, y, z] = comp_id
        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in neighbors:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= sx or ny >= sy or nz >= sz:
                    continue
                if not mask[nx, ny, nz] or labels[nx, ny, nz] != -1:
                    continue
                labels[nx, ny, nz] = comp_id
                stack.append((nx, ny, nz))
        comp_id += 1

    return labels


def _detect_modes_from_control_space(
    values: np.ndarray,
    probs: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Mode detection requires control-space values shape (N, 3)")

    bins = int(max(4, bins))
    n = values.shape[0]
    mode_id_by_row = np.full(n, -1, dtype=np.int32)

    valid = np.all(np.isfinite(values), axis=1) & (probs > 0.0)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return mode_id_by_row, {
            "mode_count": 0,
            "valid_rows_for_mode_detection": 0,
            "occupied_bins": 0,
            "total_bins": int(bins**3),
            "invalid_rows": int(n),
        }

    vals = values[valid]
    q_lo = np.nanpercentile(vals, 0.5, axis=0)
    q_hi = np.nanpercentile(vals, 99.5, axis=0)
    for i in range(3):
        if not np.isfinite(q_lo[i]) or not np.isfinite(q_hi[i]) or q_lo[i] == q_hi[i]:
            q_lo[i] = float(np.nanmin(vals[:, i]))
            q_hi[i] = float(np.nanmax(vals[:, i]))
            if q_lo[i] == q_hi[i]:
                q_hi[i] = q_lo[i] + 1e-12

    H, edges = np.histogramdd(
        vals,
        bins=(bins, bins, bins),
        range=((q_lo[0], q_hi[0]), (q_lo[1], q_hi[1]), (q_lo[2], q_hi[2])),
        weights=probs[valid],
    )

    occupied = H > 0.0
    labels = _connected_components_3d(occupied)
    mode_count = int(np.max(labels) + 1) if np.any(labels >= 0) else 0

    bx = np.clip(np.digitize(vals[:, 0], edges[0]) - 1, 0, bins - 1)
    by = np.clip(np.digitize(vals[:, 1], edges[1]) - 1, 0, bins - 1)
    bz = np.clip(np.digitize(vals[:, 2], edges[2]) - 1, 0, bins - 1)
    mode_id_by_row[valid_idx] = labels[bx, by, bz]

    diag = {
        "mode_count": mode_count,
        "valid_rows_for_mode_detection": int(valid_idx.size),
        "occupied_bins": int(np.count_nonzero(occupied)),
        "total_bins": int(occupied.size),
        "invalid_rows": int(n - valid_idx.size),
    }
    return mode_id_by_row, diag


def _draw_mode_aware_indices(
    control_values: np.ndarray,
    weights: np.ndarray,
    draw_count: int,
    rng: np.random.Generator,
    mode_tuning: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    draw_count = int(max(1, draw_count))
    w = _sanitize_weights(weights)
    total_w = float(np.sum(w))
    if not np.isfinite(total_w) or total_w <= 0.0:
        raise ValueError("No positive posterior weight available for resampling")

    probs = w / total_w
    n_rows = probs.size
    all_idx = np.arange(n_rows, dtype=np.int64)

    mode_ids, mode_diag = _detect_modes_from_control_space(
        values=control_values,
        probs=probs,
        bins=int(mode_tuning["mode_detect_bins"]),
    )

    # Posterior mass per detected mode.
    mode_masses: dict[int, float] = {}
    valid_mode_rows = mode_ids >= 0
    if np.any(valid_mode_rows):
        uniq_modes, _ = (
            np.unique(
                mode_ids[valid_mode_rows],
                return_counts=False,
            ),
            None,
        )
        for mid in uniq_modes.tolist():
            mask = mode_ids == mid
            mode_masses[int(mid)] = float(np.sum(probs[mask]))

    eligible_modes = [
        m
        for m, mass in mode_masses.items()
        if mass >= float(mode_tuning["mode_min_mass_frac"])
    ]
    eligible_modes = sorted(eligible_modes, key=lambda m: mode_masses[m], reverse=True)

    floor_per_mode = int(mode_tuning["mode_floor_per_mode"])
    floor_total_cap = int(mode_tuning["mode_floor_total_cap"])
    max_floor_modes = int(mode_tuning["mode_floor_max_modes"])
    floor_budget = min(draw_count, floor_total_cap)

    supported_modes = min(len(eligible_modes), max_floor_modes)
    if floor_per_mode > 0:
        supported_modes = min(supported_modes, floor_budget // floor_per_mode)
    selected_floor_modes = eligible_modes[:supported_modes]

    floor_draws: list[np.ndarray] = []
    for mid in selected_floor_modes:
        row_idx = np.flatnonzero(mode_ids == mid)
        if row_idx.size == 0:
            continue
        p_local = probs[row_idx]
        p_sum = float(np.sum(p_local))
        if p_sum <= 0.0:
            continue
        p_local = p_local / p_sum
        floor_draws.append(
            _draw_systematic_indices(
                indices=row_idx,
                probs=p_local,
                draw_count=floor_per_mode,
                rng=rng,
            )
        )

    floor_total = int(sum(arr.size for arr in floor_draws))
    remaining = int(max(0, draw_count - floor_total))
    base_draws = _draw_systematic_indices(
        indices=all_idx,
        probs=probs,
        draw_count=remaining,
        rng=rng,
    )

    if floor_draws:
        drawn = np.concatenate([base_draws, *floor_draws])
    else:
        drawn = base_draws

    uniq, counts = np.unique(drawn, return_counts=True)

    sampled_mass_by_mode: dict[str, float] = {}
    for mid in selected_floor_modes:
        sampled_mass_by_mode[str(mid)] = mode_masses.get(mid, 0.0)

    diag = {
        **mode_diag,
        "draw_count": int(draw_count),
        "rows_with_positive_weight": int(np.count_nonzero(w > 0.0)),
        "eligible_modes": int(len(eligible_modes)),
        "floor_supported_modes": int(len(selected_floor_modes)),
        "floor_per_mode": int(floor_per_mode),
        "floor_total_draws": int(floor_total),
        "remaining_draws": int(remaining),
        "unique_trajectories": int(uniq.size),
        "eligible_mode_ids": [int(m) for m in eligible_modes],
        "selected_floor_mode_ids": [int(m) for m in selected_floor_modes],
        "mode_posterior_mass": {str(k): v for k, v in sorted(mode_masses.items())},
        "selected_mode_posterior_mass": sampled_mass_by_mode,
    }
    return uniq.astype(np.int64), counts.astype(np.int64), diag


def _sampling_plan_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "sampling_plans"


def _chain_fingerprint_payload(
    bundle: ChainBundle,
    selection: SelectionData,
    args: argparse.Namespace,
    mode_tuning: dict[str, Any],
) -> dict[str, Any]:
    files_meta = []
    for fp, postburn_rows in zip(bundle.chain_files, selection.file_counts_postburn):
        st = fp.stat()
        files_meta.append(
            {
                "path": str(fp),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "postburn_rows": int(postburn_rows),
            }
        )

    return {
        "schema_version": _MODE_SAMPLING_SCHEMA_VERSION,
        "root": bundle.root,
        "ignore_rows": float(args.ignore_rows),
        "max_samples": int(args.max_samples),
        "hpd_params": [str(x) for x in args.hpd_params],
        "mode_tuning": mode_tuning,
        "chain_files": files_meta,
    }


def _chain_fingerprint_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _sampling_plan_cache_path(cache_dir: Path, root: str, fp_hash: str) -> Path:
    d = _sampling_plan_cache_dir(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_sanitize_label(root)}__{fp_hash}.json"


def _load_sampling_plan(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    uniq = np.asarray(payload["unique_global_indices"], dtype=np.int64)
    mult = np.asarray(payload["multiplicities"], dtype=np.int64)
    diag = dict(payload.get("diagnostics", {}))
    return uniq, mult, diag


def _save_sampling_plan(
    path: Path,
    *,
    unique_global_indices: np.ndarray,
    multiplicities: np.ndarray,
    diagnostics: dict[str, Any],
    fingerprint_payload: dict[str, Any],
    fingerprint_hash: str,
) -> None:
    payload = {
        "schema_version": _MODE_SAMPLING_SCHEMA_VERSION,
        "fingerprint_hash": fingerprint_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint_payload": fingerprint_payload,
        "unique_global_indices": [int(x) for x in unique_global_indices.tolist()],
        "multiplicities": [int(x) for x in multiplicities.tolist()],
        "diagnostics": diagnostics,
    }
    _write_json_atomic(path, payload)


def _resolve_sampling_plan(
    *,
    bundle: ChainBundle,
    selection: SelectionData,
    args: argparse.Namespace,
    mode_tuning: dict[str, Any],
    rng: np.random.Generator,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    fingerprint_payload = _chain_fingerprint_payload(
        bundle, selection, args, mode_tuning
    )
    fingerprint_hash = _chain_fingerprint_hash(fingerprint_payload)
    plan_path = _sampling_plan_cache_path(cache_dir, bundle.root, fingerprint_hash)

    if args.sample_plan_cache_mode != "refresh" and plan_path.exists():
        uniq, mult, diag = _load_sampling_plan(plan_path)
        diag = {
            **diag,
            "sampling_plan_cache": "hit",
            "sampling_plan_path": str(plan_path),
            "sampling_plan_fingerprint": fingerprint_hash,
        }
        return uniq, mult, diag, fingerprint_hash

    if args.sample_plan_cache_mode == "locked" and not plan_path.exists():
        raise FileNotFoundError(
            "Sampling plan cache mode is 'locked' but no matching plan was found: "
            f"{plan_path}"
        )

    uniq, mult, diag = _draw_mode_aware_indices(
        control_values=selection.hpd_values,
        weights=selection.weights,
        draw_count=args.max_samples,
        rng=rng,
        mode_tuning=mode_tuning,
    )
    diag = {
        **diag,
        "sampling_plan_cache": "miss",
        "sampling_plan_path": str(plan_path),
        "sampling_plan_fingerprint": fingerprint_hash,
        "sample_plan_cache_mode": args.sample_plan_cache_mode,
    }
    _save_sampling_plan(
        plan_path,
        unique_global_indices=uniq,
        multiplicities=mult,
        diagnostics=diag,
        fingerprint_payload=fingerprint_payload,
        fingerprint_hash=fingerprint_hash,
    )
    return uniq, mult, diag, fingerprint_hash


def _global_to_file_rows(
    unique_global_indices: np.ndarray,
    multiplicities: np.ndarray,
    counts_postburn: list[int],
) -> list[DrawRecord]:
    cumulative = np.cumsum(np.asarray(counts_postburn, dtype=np.int64))
    records: list[DrawRecord] = []

    for gidx, mult in zip(unique_global_indices.tolist(), multiplicities.tolist()):
        file_idx = int(np.searchsorted(cumulative, gidx, side="right"))
        prev = 0 if file_idx == 0 else int(cumulative[file_idx - 1])
        row_local = int(gidx - prev)
        records.append(
            DrawRecord(
                file_index=file_idx,
                row_index_postburn=row_local,
                multiplicity=int(mult),
            )
        )

    return records


def _extract_rows_by_index(
    file_path: Path,
    needed_rows_postburn: set[int],
    ignore_rows: float,
) -> dict[int, list[float]]:
    """Extract specific post-burn row numbers from one chain file.

    Returns mapping post-burn row index -> full numeric row values.
    """
    if not needed_rows_postburn:
        return {}

    # Determine burn-in count from total data rows.
    total_rows = 0
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            total_rows += 1
    burn = int(max(0, min(total_rows, ignore_rows * total_rows)))

    keep_sorted = sorted(needed_rows_postburn)
    keep_ptr = 0
    keep_target = keep_sorted[keep_ptr]
    found: dict[int, list[float]] = {}

    postburn_idx = -1
    data_idx = -1

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            data_idx += 1
            if data_idx < burn:
                continue

            postburn_idx += 1
            if postburn_idx < keep_target:
                continue

            parts = line.strip().split()
            found[postburn_idx] = [float(x) for x in parts]

            keep_ptr += 1
            if keep_ptr >= len(keep_sorted):
                break
            keep_target = keep_sorted[keep_ptr]

    missing = needed_rows_postburn.difference(found.keys())
    if missing:
        raise KeyError(
            f"Failed to extract {len(missing)} requested rows from {file_path.name}. "
            f"Example missing post-burn row: {min(missing)}"
        )

    return found


def _normalize_sbbn_path(class_params: dict[str, Any]) -> None:
    key = "sBBN file"
    if key not in class_params:
        return
    val = str(class_params[key]).strip()
    if not val:
        return
    if "/" not in val and "\\" not in val:
        class_params[key] = f"/external/bbn/{val}"
    elif val.startswith("bbn/"):
        class_params[key] = f"/external/{val}"


def _hash_class_params(class_params: dict[str, Any]) -> str:
    canonical = {k: str(v) for k, v in sorted(class_params.items())}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_failure_audit(
    audit_dir: Path,
    *,
    bundle_root: str,
    chain_file: Path,
    trajectory_index: int,
    record: DrawRecord,
    row_map: dict[str, float],
    class_params: dict[str, Any],
    error: ClassBackgroundRunError,
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = (
        f"{_sanitize_label(bundle_root)}"
        f"__{chain_file.stem}"
        f"__traj{trajectory_index:05d}"
        f"__row{record.row_index_postburn}"
        f"__{error.cache_key}"
        f"__{uuid.uuid4().hex[:8]}"
    )

    ini_copy_path = audit_dir / f"{stem}.ini"
    json_path = audit_dir / f"{stem}.json"
    summary_path = audit_dir / "failures.jsonl"

    ini_copy_path.write_text(error.ini_text, encoding="utf-8")

    payload = {
        "timestamp_utc": stamp,
        "bundle_root": bundle_root,
        "chain_file": str(chain_file),
        "trajectory_index": trajectory_index,
        "file_index": record.file_index,
        "row_index_postburn": record.row_index_postburn,
        "multiplicity": record.multiplicity,
        "cache_key": error.cache_key,
        "generated_ini_file": str(ini_copy_path),
        "generated_ini_original_path": str(error.ini_path),
        "sampled_parameter_values": {
            key: _serialize_value(value) for key, value in sorted(row_map.items())
        },
        "class_parameters": {
            key: _serialize_value(value) for key, value in sorted(class_params.items())
        },
        "error": {
            "message": str(error),
            "returncode": error.returncode,
            "stdout_tail": error.stdout_tail,
            "stderr_tail": error.stderr_tail,
        },
    }

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

    return json_path


def _run_class_background_only(
    class_params: dict[str, Any],
    cache_dir: Path,
    cache_key: str,
) -> dict[str, np.ndarray]:
    repo_root = _repo_root()
    class_exec = repo_root / "class"
    if not class_exec.exists():
        raise FileNotFoundError(
            f"CLASS executable not found at {class_exec}. Run 'make class -j'."
        )

    tmp_label = f"tmp_bg_{cache_key}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    out_root = cache_dir / tmp_label
    ini_path = cache_dir / f"{tmp_label}.ini"

    run_params = dict(class_params)
    run_params["write_background"] = "yes"
    run_params["lensing"] = "no"
    run_params["overwrite_root"] = "yes"
    run_params["root"] = str(out_root)

    # Background-only mode: remove expensive spectrum options when present.
    run_params.pop("output", None)
    run_params.pop("non linear", None)
    run_params.pop("non_linear", None)

    _normalize_sbbn_path(run_params)

    ini_lines = ["# Auto-generated: posterior_background_heatmaps.py\n"]
    for key in sorted(run_params):
        ini_lines.append(f"{key} = {run_params[key]}\n")
    ini_text = "".join(ini_lines)

    with open(ini_path, "w") as f:
        f.write(ini_text)

    proc = subprocess.run(
        [str(class_exec), str(ini_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    background_file = Path(f"{out_root}_background.dat")

    try:
        if proc.returncode != 0:
            stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
            stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
            raise ClassBackgroundRunError(
                "CLASS background-only run failed.\n"
                f"Return code: {proc.returncode}\n"
                f"INI: {ini_path}\n"
                f"STDOUT tail:\n{stdout_tail}\n"
                f"STDERR tail:\n{stderr_tail}",
                cache_key=cache_key,
                ini_path=ini_path,
                ini_text=ini_text,
                returncode=proc.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )

        if not background_file.exists():
            raise ClassBackgroundRunError(
                f"Expected background output missing: {background_file}",
                cache_key=cache_key,
                ini_path=ini_path,
                ini_text=ini_text,
                returncode=proc.returncode,
                stdout_tail="\n".join(proc.stdout.splitlines()[-20:]),
                stderr_tail="\n".join(proc.stderr.splitlines()[-20:]),
            )

        dataset = load_background_dataset(str(background_file))
        return {k: np.asarray(dataset[k]) for k in _FIELDS_TO_CACHE if k in dataset}
    finally:
        # Keep persistent cache lean: remove transient CLASS outputs.
        for suffix in ("_background.dat", "_pk.dat", "_cl_lensed.dat"):
            p = Path(f"{out_root}{suffix}")
            if p.exists():
                p.unlink()
        if ini_path.exists():
            ini_path.unlink()


def _load_cached_background(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as arr:
        return {k: np.asarray(arr[k]) for k in arr.files if k in _FIELDS_TO_CACHE}


def _write_npz_atomic(npz_path: Path, data: dict[str, np.ndarray]) -> None:
    tmp_path = npz_path.with_name(f"{npz_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp_path, npz_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def get_or_compute_background(
    class_params: dict[str, Any],
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], str, str]:
    cache_key = _hash_class_params(class_params)
    npz_path = cache_dir / f"{cache_key}.background.npz"
    meta_path = cache_dir / f"{cache_key}.meta.json"

    if npz_path.exists():
        try:
            return _load_cached_background(npz_path), "cache", cache_key
        except (zipfile.BadZipFile, OSError, ValueError, EOFError, KeyError) as exc:
            # Heal corrupted or partially-written cache entries and recompute.
            print(
                f"  [WARNING] Corrupted cache entry {npz_path.name}: {exc}. "
                "Deleting and recomputing."
            )
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    data = _run_class_background_only(class_params, cache_dir, cache_key)

    _write_npz_atomic(npz_path, data)
    _write_json_atomic(
        meta_path,
        {
            "cache_key": cache_key,
            "class_params": {k: str(v) for k, v in sorted(class_params.items())},
            "fields": sorted(list(data.keys())),
        },
    )

    return data, "computed", cache_key


def _dataset_label_from_root(root: str) -> str:
    r = root.lower()
    parts: list[str] = []

    if "spa" in r:
        parts.append("SPA")
    elif "planck" in r or "_cmb_" in r:
        parts.append("Planck")

    if "desi" in r:
        parts.append("DESI")

    # Project convention: PP/PPS implies Pantheon+ and DESI.
    if any(tag in r for tag in ("_pp_", ".post.pp", "_pps_", ".post.pps")):
        if "Pantheon+" not in parts:
            parts.append("Pantheon+")
    if any(tag in r for tag in ("_pp_s_", "sh0es", "_pps_")):
        parts.append("SH0ES")

    return " + ".join(parts) if parts else root


def _build_common_x_grid(x_min: float, x_max: float, n_points: int = 700) -> np.ndarray:
    x_min = float(min(x_min, x_max))
    x_max = float(max(x_min, x_max))
    return np.linspace(x_min, x_max, int(max(128, n_points)))


def _interp_background_quantity(
    dataset: dict[str, np.ndarray],
    quantity: str,
    x_grid: np.ndarray,
) -> np.ndarray:
    z = np.asarray(dataset["z"], dtype=float)
    y = np.asarray(dataset[quantity], dtype=float)

    valid = np.isfinite(z) & np.isfinite(y) & (z >= 0.0)
    if np.count_nonzero(valid) < 2:
        return np.full_like(x_grid, np.nan)

    z = z[valid]
    y = y[valid]

    x = np.log10(1.0 + z)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate x values for stable interpolation.
    x_unique, uniq_idx = np.unique(x, return_index=True)
    y_unique = y[uniq_idx]
    if x_unique.size < 2:
        return np.full_like(x_grid, np.nan)

    out = np.full_like(x_grid, np.nan)
    inside = (x_grid >= x_unique[0]) & (x_grid <= x_unique[-1])
    out[inside] = np.interp(x_grid[inside], x_unique, y_unique)
    return out


def _sanitize_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _output_quantity_name(quantity: str) -> str:
    return _OUTPUT_QUANTITY_ALIASES.get(quantity, quantity)


def _output_base_path(output_dir: Path, root: str, quantity_name: str) -> Path:
    return output_dir / f"{_sanitize_label(root)}_heatmap_{quantity_name}"


def _save_figure_bundle(fig: Figure, base_path: Path) -> None:
    """Save each figure as PNG/PDF/PGF for quick view and LaTeX workflows."""
    fig.savefig(str(base_path.parent / f"{base_path.name}.png"))
    fig.savefig(str(base_path.parent / f"{base_path.name}.pdf"))
    fig.savefig(str(base_path.parent / f"{base_path.name}.pgf"))


def _style_redshift_axis(ax: Axes, z: np.ndarray) -> None:
    """Apply BestFitPlot-style redshift axis formatting with z=0 at right."""
    finite_z = z[np.isfinite(z)]
    zmax = float(np.nanmax(finite_z)) if finite_z.size else 1.0

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(zmax, 0.0)

    if zmax >= 1.0:
        max_pow = int(np.floor(np.log10(zmax)))
        major_ticks = [10.0**p for p in range(max_pow, -1, -1)] + [0.0]
    else:
        major_ticks = [zmax, 0.0]

    major_ticks = sorted(set(float(t) for t in major_ticks))

    major_labels: list[str] = []
    for tick in major_ticks:
        if np.isclose(tick, 0.0):
            major_labels.append("0")
        elif tick >= 1.0:
            power = int(np.round(np.log10(tick)))
            major_labels.append(rf"$10^{{{power}}}$")
        else:
            major_labels.append("")

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(major_labels))

    minor_ticks: list[float] = []
    if zmax >= 10.0:
        for power in range(max_pow - 1, -1, -1):
            decade = 10.0**power
            for multiplier in (2, 5):
                tick = multiplier * decade
                if 1.0 < tick < zmax:
                    minor_ticks.append(tick)

    minor_ticks.extend([0.2, 0.4, 0.6, 0.8])
    minor_ticks = sorted(set(minor_ticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"Redshift $z$")
    ax.grid(True, which="major", alpha=0.32, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.09, linewidth=0.4)


def _build_colorbar_ticks(vmin: float, vmax: float) -> list[float]:
    """Choose at least two readable labeled ticks over [vmin, vmax]."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin <= 0.0 or vmax <= vmin:
        return [vmin, vmax]

    lo = np.log10(vmin)
    hi = np.log10(vmax)

    # For narrow dynamic ranges, prefer rounded linear ticks over awkward
    # geometric midpoints such as 9.16515 for a range that visually reads 6..14.
    if hi - lo < 0.5:
        locator = MaxNLocator(nbins=5, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10])
        ticks = [
            float(tick)
            for tick in locator.tick_values(vmin, vmax)
            if vmin <= float(tick) <= vmax
        ]
        if len(ticks) >= 2:
            return ticks

    decade_ticks = [
        10.0**power
        for power in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        if vmin <= 10.0**power <= vmax
    ]
    if len(decade_ticks) >= 2:
        return decade_ticks

    nice_log_ticks: list[float] = []
    min_power = int(np.floor(lo)) - 1
    max_power = int(np.ceil(hi)) + 1
    for power in range(min_power, max_power + 1):
        decade = 10.0**power
        for sub in (1.0, 2.0, 5.0):
            tick = sub * decade
            if vmin <= tick <= vmax:
                nice_log_ticks.append(float(tick))
    if len(nice_log_ticks) >= 2:
        return sorted(set(nice_log_ticks))

    ticks = sorted(set([vmin, vmax]))
    if len(ticks) < 2:
        ticks = [vmin, vmax]
    return ticks


def _format_colorbar_tick(tick: float) -> str:
    if tick >= 1e-2 and tick < 1e3:
        if np.isclose(tick, round(tick), rtol=0.0, atol=1e-10):
            return f"{int(round(tick))}"
        if np.isclose(tick, round(tick, 1), rtol=0.0, atol=1e-10):
            return f"{tick:.1f}"
        return f"{tick:g}"

    exponent = np.log10(tick)
    rounded = int(np.round(exponent))
    if np.isclose(exponent, rounded, atol=1e-10):
        return rf"$10^{{{rounded}}}$"
    return f"{tick:.1e}"


def _style_colorbar(cb: Colorbar, vmin: float, vmax: float) -> None:
    """Style the heatmap colorbar with the same tick density/geometry family."""
    major_ticks = _build_colorbar_ticks(vmin, vmax)
    if len(major_ticks) < 2 and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        mid = float(np.sqrt(vmin * vmax))
        major_ticks = sorted({float(vmin), mid, float(vmax)})
    if len(major_ticks) < 2 and np.isfinite(vmin) and np.isfinite(vmax):
        upper = float(vmax if vmax > vmin else (vmin * (1.0 + 1e-6)))
        major_ticks = [float(vmin), upper]
    cb.ax.yaxis.set_major_locator(FixedLocator(major_ticks))
    cb.ax.yaxis.set_major_formatter(
        FixedFormatter([_format_colorbar_tick(tick) for tick in major_ticks])
    )
    cb.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        length=4.8,
        width=0.95,
        labelsize=11,
    )
    cb.ax.tick_params(
        axis="y",
        which="minor",
        direction="in",
        length=2.4,
        width=0.72,
    )


def _symlog_transform(values: np.ndarray, linthresh: float) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)


def _symlog_inverse(values: np.ndarray, linthresh: float) -> np.ndarray:
    return np.sign(values) * linthresh * (np.power(10.0, np.abs(values)) - 1.0)


def _symlog2_transform(values: np.ndarray, linthresh: float) -> np.ndarray:
    inner = np.log10(1.0 + np.abs(values) / linthresh)
    return np.sign(values) * np.log10(1.0 + inner)


def _symlog2_inverse(values: np.ndarray, linthresh: float) -> np.ndarray:
    inner = np.power(10.0, np.abs(values)) - 1.0
    return np.sign(values) * linthresh * (np.power(10.0, inner) - 1.0)


def _phi_linthresh_from_range(y_low: float, y_high: float) -> float:
    max_abs = max(abs(y_low), abs(y_high))
    # Keep the linear core tied to the dynamic range while avoiding collapse.
    return max(1e-18, 0.02 * max_abs)


def _build_symlog_y_edges(
    y_min: float,
    y_max: float,
    y_bins: int,
    pad_frac: float = 0.03,
) -> tuple[np.ndarray, float]:
    pad = pad_frac * (y_max - y_min)
    y_low = y_min - pad
    y_high = y_max + pad
    linthresh = _phi_linthresh_from_range(y_low, y_high)
    t_low = _symlog_transform(np.array([y_low]), linthresh)[0]
    t_high = _symlog_transform(np.array([y_high]), linthresh)[0]
    t_edges = np.linspace(t_low, t_high, y_bins + 1)
    return _symlog_inverse(t_edges, linthresh), linthresh


def _build_symlog2_y_edges(
    y_min: float,
    y_max: float,
    y_bins: int,
    pad_frac: float = 0.03,
) -> tuple[np.ndarray, float]:
    pad = pad_frac * (y_max - y_min)
    y_low = y_min - pad
    y_high = y_max + pad
    linthresh = _phi_linthresh_from_range(y_low, y_high)
    t_low = _symlog2_transform(np.array([y_low]), linthresh)[0]
    t_high = _symlog2_transform(np.array([y_high]), linthresh)[0]
    t_edges = np.linspace(t_low, t_high, y_bins + 1)
    return _symlog2_inverse(t_edges, linthresh), linthresh


def _hide_overlaps_with_zero_ytick_label(
    fig: Figure,
    ax: Axes,
    overlap_margin_px: float = 0.6,
) -> None:
    """Keep the y=0 major label visible and hide only labels that overlap it.

    This preserves Matplotlib's default symlog tick selection while resolving the
    specific visual collision near zero.
    """

    fig.canvas.draw()
    renderer_getter = getattr(fig.canvas, "get_renderer", None)
    if renderer_getter is None:
        return
    renderer = renderer_getter()

    ticks = np.asarray(ax.get_yticks(), dtype=float)
    labels = list(ax.yaxis.get_majorticklabels())
    n = min(len(ticks), len(labels))
    if n < 2:
        return

    max_abs = float(np.nanmax(np.abs(ticks[:n]))) if n > 0 else 1.0
    atol = max(1e-14, 1e-12 * max(max_abs, 1.0))

    zero_idx: int | None = None
    for i in range(n):
        if bool(np.isclose(ticks[i], 0.0, atol=atol, rtol=0.0)):
            zero_idx = i
            break
    if zero_idx is None:
        return

    zero_label = labels[zero_idx]
    if not zero_label.get_text().strip():
        return
    zero_label.set_visible(True)

    margin = max(0.0, float(overlap_margin_px))
    x0, y0, x1, y1 = zero_label.get_window_extent(renderer=renderer).extents
    zero_box = type(zero_label.get_window_extent(renderer=renderer)).from_extents(
        x0 - margin,
        y0 - margin,
        x1 + margin,
        y1 + margin,
    )
    for i in range(n):
        if i == zero_idx:
            continue
        label = labels[i]
        if not label.get_visible() or not label.get_text().strip():
            continue
        if zero_box.overlaps(label.get_window_extent(renderer=renderer)):
            label.set_visible(False)


def _ensure_min_labeled_yticks(ax: Axes, min_labels: int = 2) -> None:
    """Guarantee at least `min_labels` major y-tick labels are available."""
    ticks = np.asarray(ax.get_yticks(), dtype=float)
    if ticks.size == 0:
        return

    y0, y1 = ax.get_ylim()
    lo = float(min(y0, y1))
    hi = float(max(y0, y1))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    tol = max(1e-18, 1e-12 * max(abs(lo), abs(hi), 1.0))
    in_view = np.isfinite(ticks) & (ticks >= lo - tol) & (ticks <= hi + tol)
    visible_ticks = ticks[in_view]
    if visible_ticks.size >= min_labels:
        return

    forced: list[float]
    if lo < 0.0 < hi:
        forced = [lo, 0.0, hi]
    else:
        forced = [lo, hi]
        if min_labels >= 3:
            forced.insert(1, 0.5 * (lo + hi))

    # Keep deterministic ordering and avoid duplicate ticks in degenerate ranges.
    forced = sorted({float(v) for v in forced})
    if len(forced) >= 2:
        ax.yaxis.set_major_locator(FixedLocator(forced))


def _build_quantity_y_edges(
    y_min: float,
    y_max: float,
    y_bins: int,
    qty: str,
    phi_y_scale: str,
) -> tuple[np.ndarray, str | None, float | None]:
    """Construct y-bin edges and optional y-axis scale metadata."""
    pad = 0.03 * (y_max - y_min)
    y_low = y_min - pad
    y_high = y_max + pad

    if qty == "phi_scf" and y_low < 0.0 < y_high:
        if phi_y_scale == "symlog2":
            edges, linthresh = _build_symlog2_y_edges(y_min, y_max, y_bins)
            return edges, "symlog2", linthresh
        edges, linthresh = _build_symlog_y_edges(y_min, y_max, y_bins)
        return edges, "symlog", linthresh

    return np.linspace(y_low, y_high, y_bins + 1), None, None


def _build_positive_log_y_edges(
    y_min: float,
    y_max: float,
    y_bins: int,
    pad_frac: float = 0.06,
) -> np.ndarray:
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
        raise ValueError("Positive log y-edges require finite y_min < y_max")

    positive_floor = max(np.finfo(float).tiny, 1e-300)
    y_low = max(positive_floor, y_min * (1.0 - pad_frac))
    y_high = max(y_low * (1.0 + 1e-12), y_max * (1.0 + pad_frac))
    return np.geomspace(y_low, y_high, int(y_bins) + 1)


def _weighted_quantile_1d(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(valid) == 0:
        return np.full(quantiles.shape, np.nan, dtype=float)

    vals = np.asarray(values[valid], dtype=float)
    w = np.asarray(weights[valid], dtype=float)
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]

    total = float(np.sum(w))
    if total <= 0.0:
        return np.full(quantiles.shape, np.nan, dtype=float)

    cdf = np.cumsum(w, dtype=float) / total
    return np.interp(quantiles, cdf, vals, left=vals[0], right=vals[-1])


def _compute_pointwise_weighted_summary(
    series_matrix: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...] = (0.16, 0.5, 0.84),
) -> dict[str, np.ndarray]:
    if series_matrix.ndim != 2:
        raise ValueError("series_matrix must have shape (n_series, n_grid)")

    q_probs = np.asarray(quantiles, dtype=float)
    q_out = np.full((q_probs.size, series_matrix.shape[1]), np.nan, dtype=float)
    valid_weight = np.zeros(series_matrix.shape[1], dtype=float)

    weights = np.asarray(weights, dtype=float)
    for ix in range(series_matrix.shape[1]):
        column = series_matrix[:, ix]
        valid = np.isfinite(column) & np.isfinite(weights) & (weights > 0.0)
        if np.count_nonzero(valid) == 0:
            continue
        valid_weight[ix] = float(np.sum(weights[valid]))
        q_out[:, ix] = _weighted_quantile_1d(column[valid], weights[valid], q_probs)

    return {
        "q16": q_out[0],
        "q50": q_out[1],
        "q84": q_out[2],
        "valid_weight": valid_weight,
    }


def _build_quantity_summary(
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]],
    qty: str,
) -> dict[str, np.ndarray] | None:
    series_list: list[np.ndarray] = []
    weights: list[float] = []

    for qty_interps, multiplicity in trajectory_payload:
        y = qty_interps.get(qty)
        if y is None:
            continue
        series_list.append(np.asarray(y, dtype=float))
        weights.append(float(multiplicity))

    if not series_list:
        return None

    matrix = np.vstack(series_list)
    return _compute_pointwise_weighted_summary(matrix, np.asarray(weights, dtype=float))


def _build_quantity_matrix(
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]],
    qty: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    series_list: list[np.ndarray] = []
    weights: list[float] = []

    for qty_interps, multiplicity in trajectory_payload:
        y = qty_interps.get(qty)
        if y is None:
            continue
        series_list.append(np.asarray(y, dtype=float))
        weights.append(float(multiplicity))

    if not series_list:
        return None

    return np.vstack(series_list), np.asarray(weights, dtype=float)


def _compute_phi_crossing_profile(
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]],
    *,
    epsilon_mode: str,
    epsilon_abs: float,
    epsilon_linthresh_frac: float,
    epsilon_quantile: float,
    y_linthresh: float | None,
) -> dict[str, np.ndarray] | None:
    matrix_and_weights = _build_quantity_matrix(trajectory_payload, "phi_scf")
    if matrix_and_weights is None:
        return None

    matrix, weights = matrix_and_weights
    n_grid = matrix.shape[1]

    q = float(np.clip(epsilon_quantile, 0.0, 0.5))
    eps_abs = float(max(0.0, epsilon_abs))
    eps_linthresh = float(
        max(0.0, epsilon_linthresh_frac) * max(0.0, float(y_linthresh or 0.0))
    )

    eps = np.full(n_grid, np.nan, dtype=float)
    p_pos = np.full(n_grid, np.nan, dtype=float)
    p_neg = np.full(n_grid, np.nan, dtype=float)
    p_near = np.full(n_grid, np.nan, dtype=float)
    p_cross = np.full(n_grid, np.nan, dtype=float)
    valid_weight = np.zeros(n_grid, dtype=float)

    for ix in range(n_grid):
        col = matrix[:, ix]
        valid = np.isfinite(col) & np.isfinite(weights) & (weights > 0.0)
        if np.count_nonzero(valid) == 0:
            continue

        col_v = col[valid]
        w_v = weights[valid]
        w_sum = float(np.sum(w_v))
        if w_sum <= 0.0:
            continue

        eps_q = float(
            _weighted_quantile_1d(np.abs(col_v), w_v, np.asarray([q], dtype=float))[0]
        )
        if epsilon_mode == "fixed":
            eps_i = eps_abs
        elif epsilon_mode == "linthresh":
            eps_i = max(eps_abs, eps_linthresh)
        else:
            eps_i = max(eps_abs, eps_linthresh, eps_q)

        pos = col_v > eps_i
        neg = col_v < -eps_i
        near = ~(pos | neg)

        p_pos_i = float(np.sum(w_v[pos]) / w_sum)
        p_neg_i = float(np.sum(w_v[neg]) / w_sum)
        p_near_i = float(np.sum(w_v[near]) / w_sum)
        p_cross_i = float(np.clip(2.0 * min(p_pos_i, p_neg_i), 0.0, 1.0))

        eps[ix] = eps_i
        p_pos[ix] = p_pos_i
        p_neg[ix] = p_neg_i
        p_near[ix] = p_near_i
        p_cross[ix] = p_cross_i
        valid_weight[ix] = w_sum

    return {
        "epsilon": eps,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "p_near": p_near,
        "p_cross": p_cross,
        "valid_weight": valid_weight,
    }


def _compute_phi_sign_branch_summary(
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]],
    crossing_profile: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    if crossing_profile is None:
        return None

    matrix_and_weights = _build_quantity_matrix(trajectory_payload, "phi_scf")
    if matrix_and_weights is None:
        return None

    matrix, weights = matrix_and_weights
    eps = np.asarray(crossing_profile.get("epsilon", np.array([])), dtype=float)
    if eps.size != matrix.shape[1]:
        return None

    n_grid = matrix.shape[1]
    pos_q16 = np.full(n_grid, np.nan, dtype=float)
    pos_q50 = np.full(n_grid, np.nan, dtype=float)
    pos_q84 = np.full(n_grid, np.nan, dtype=float)
    neg_q16 = np.full(n_grid, np.nan, dtype=float)
    neg_q50 = np.full(n_grid, np.nan, dtype=float)
    neg_q84 = np.full(n_grid, np.nan, dtype=float)
    pos_valid_weight = np.zeros(n_grid, dtype=float)
    neg_valid_weight = np.zeros(n_grid, dtype=float)

    for ix in range(n_grid):
        eps_i = eps[ix]
        if not np.isfinite(eps_i):
            continue

        col = matrix[:, ix]
        valid = np.isfinite(col) & np.isfinite(weights) & (weights > 0.0)
        if np.count_nonzero(valid) == 0:
            continue

        col_v = col[valid]
        w_v = weights[valid]

        pos = col_v > eps_i
        neg = col_v < -eps_i

        if np.any(pos):
            w_pos = w_v[pos]
            v_pos = col_v[pos]
            pos_valid_weight[ix] = float(np.sum(w_pos))
            q_pos = _weighted_quantile_1d(
                v_pos,
                w_pos,
                np.asarray([0.16, 0.5, 0.84], dtype=float),
            )
            pos_q16[ix], pos_q50[ix], pos_q84[ix] = q_pos

        if np.any(neg):
            w_neg = w_v[neg]
            v_neg = col_v[neg]
            neg_valid_weight[ix] = float(np.sum(w_neg))
            q_neg = _weighted_quantile_1d(
                v_neg,
                w_neg,
                np.asarray([0.16, 0.5, 0.84], dtype=float),
            )
            neg_q16[ix], neg_q50[ix], neg_q84[ix] = q_neg

    return {
        "pos_q16": pos_q16,
        "pos_q50": pos_q50,
        "pos_q84": pos_q84,
        "pos_valid_weight": pos_valid_weight,
        "neg_q16": neg_q16,
        "neg_q50": neg_q50,
        "neg_q84": neg_q84,
        "neg_valid_weight": neg_valid_weight,
    }


def _compute_swgc_crossing_profile(
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]],
    *,
    epsilon_abs: float,
    epsilon_rel: float,
    epsilon_quantile: float,
) -> dict[str, np.ndarray] | None:
    residual_matrix_and_weights = _build_quantity_matrix(
        trajectory_payload, "swgc_residual"
    )
    lhs_matrix_and_weights = _build_quantity_matrix(trajectory_payload, "swgc_lhs")
    rhs_matrix_and_weights = _build_quantity_matrix(trajectory_payload, "swgc_rhs")

    if (
        residual_matrix_and_weights is None
        or lhs_matrix_and_weights is None
        or rhs_matrix_and_weights is None
    ):
        return None

    matrix_res, weights = residual_matrix_and_weights
    matrix_lhs, _ = lhs_matrix_and_weights
    matrix_rhs, _ = rhs_matrix_and_weights
    if matrix_res.shape != matrix_lhs.shape or matrix_res.shape != matrix_rhs.shape:
        return None

    n_grid = matrix_res.shape[1]
    eps_abs = float(max(0.0, epsilon_abs))
    eps_rel = float(max(0.0, epsilon_rel))
    q = float(np.clip(epsilon_quantile, 0.0, 0.5))

    eps = np.full(n_grid, np.nan, dtype=float)
    p_pos = np.full(n_grid, np.nan, dtype=float)
    p_neg = np.full(n_grid, np.nan, dtype=float)
    p_near = np.full(n_grid, np.nan, dtype=float)
    p_lt_zero = np.full(n_grid, np.nan, dtype=float)
    kappa_median = np.full(n_grid, np.nan, dtype=float)
    valid_weight = np.zeros(n_grid, dtype=float)

    for ix in range(n_grid):
        col_res = matrix_res[:, ix]
        col_lhs = matrix_lhs[:, ix]
        col_rhs = matrix_rhs[:, ix]

        valid = (
            np.isfinite(col_res)
            & np.isfinite(col_lhs)
            & np.isfinite(col_rhs)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        if np.count_nonzero(valid) == 0:
            continue

        res_v = col_res[valid]
        lhs_v = col_lhs[valid]
        rhs_v = col_rhs[valid]
        w_v = weights[valid]
        w_sum = float(np.sum(w_v))
        if w_sum <= 0.0:
            continue

        scale_v = np.maximum(np.abs(lhs_v), np.abs(rhs_v))
        scale_med = float(
            _weighted_quantile_1d(scale_v, w_v, np.asarray([0.5], dtype=float))[0]
        )
        eps_q = float(
            _weighted_quantile_1d(np.abs(res_v), w_v, np.asarray([q], dtype=float))[0]
        )
        eps_i = max(eps_abs, eps_rel * max(0.0, scale_med), eps_q)

        pos = res_v > eps_i
        neg = res_v < -eps_i
        near = ~(pos | neg)

        p_pos_i = float(np.sum(w_v[pos]) / w_sum)
        p_neg_i = float(np.sum(w_v[neg]) / w_sum)
        p_near_i = float(np.sum(w_v[near]) / w_sum)
        p_lt_zero_i = p_neg_i

        ratio_v = scale_v / np.maximum(np.abs(res_v), eps_i)
        kappa_i = float(
            _weighted_quantile_1d(ratio_v, w_v, np.asarray([0.5], dtype=float))[0]
        )

        eps[ix] = eps_i
        p_pos[ix] = p_pos_i
        p_neg[ix] = p_neg_i
        p_near[ix] = p_near_i
        p_lt_zero[ix] = p_lt_zero_i
        kappa_median[ix] = kappa_i
        valid_weight[ix] = w_sum

    return {
        "epsilon": eps,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "p_near": p_near,
        "p_lt_zero": p_lt_zero,
        "p_cross": p_lt_zero,
        "kappa_median": kappa_median,
        "valid_weight": valid_weight,
    }


def _overlay_phi_crossing_diagnostic(
    ax: Axes,
    z_grid: np.ndarray,
    crossing_profile: dict[str, np.ndarray] | None,
    *,
    overlay_mode: str,
    binary_threshold: float,
) -> None:
    if crossing_profile is None or overlay_mode == "none":
        return

    p_cross = np.asarray(crossing_profile["p_cross"], dtype=float)
    valid = np.isfinite(z_grid) & np.isfinite(p_cross)
    if np.count_nonzero(valid) < 2:
        return

    if overlay_mode == "probability":
        ax_prob = ax.twinx()
        ax_prob.plot(
            z_grid[valid],
            p_cross[valid],
            color=_HIGH_CONTRAST_PALETTE["red"],
            linewidth=1.2,
            linestyle="-",
            alpha=0.95,
            zorder=5.2,
        )
        ax_prob.set_ylim(0.0, 1.0)
        ax_prob.set_ylabel(r"$P_{\rm cross}$")
        ax_prob.grid(False)
        ax_prob.tick_params(axis="y", which="major", labelsize=10)
        return

    thresh = float(np.clip(binary_threshold, 0.0, 1.0))
    mask = valid & (p_cross >= thresh)
    if not np.any(mask):
        return

    ax.fill_between(
        z_grid,
        0.96,
        1.00,
        where=mask.tolist(),
        transform=ax.get_xaxis_transform(),
        color=_HIGH_CONTRAST_PALETTE["red"],
        alpha=0.22,
        linewidth=0.0,
        zorder=5.1,
    )


def _overlay_phi_sign_branch_summary(
    ax: Axes,
    z_grid: np.ndarray,
    branch_summary: dict[str, np.ndarray] | None,
) -> None:
    if branch_summary is None:
        return

    pos_q16 = np.asarray(branch_summary["pos_q16"], dtype=float)
    pos_q50 = np.asarray(branch_summary["pos_q50"], dtype=float)
    pos_q84 = np.asarray(branch_summary["pos_q84"], dtype=float)
    neg_q16 = np.asarray(branch_summary["neg_q16"], dtype=float)
    neg_q50 = np.asarray(branch_summary["neg_q50"], dtype=float)
    neg_q84 = np.asarray(branch_summary["neg_q84"], dtype=float)

    pos_band = np.isfinite(z_grid) & np.isfinite(pos_q16) & np.isfinite(pos_q84)
    if np.count_nonzero(pos_band) >= 2:
        ax.fill_between(
            z_grid[pos_band],
            pos_q16[pos_band],
            pos_q84[pos_band],
            color=_HIGH_CONTRAST_PALETTE["blue"],
            alpha=0.22,
            linewidth=0.0,
            zorder=3.25,
        )

    neg_band = np.isfinite(z_grid) & np.isfinite(neg_q16) & np.isfinite(neg_q84)
    if np.count_nonzero(neg_band) >= 2:
        ax.fill_between(
            z_grid[neg_band],
            neg_q16[neg_band],
            neg_q84[neg_band],
            color=_HIGH_CONTRAST_PALETTE["red"],
            alpha=0.22,
            linewidth=0.0,
            zorder=3.25,
        )

    _line_with_contrast(
        ax,
        z_grid,
        pos_q50,
        color=_HIGH_CONTRAST_PALETTE["blue"],
        linewidth=1.15,
        linestyle="--",
        zorder=3.95,
        outline_color=_HIGH_CONTRAST_PALETTE["white"],
        outline_width=1.0,
    )
    _line_with_contrast(
        ax,
        z_grid,
        neg_q50,
        color=_HIGH_CONTRAST_PALETTE["red"],
        linewidth=1.15,
        linestyle="--",
        zorder=3.95,
        outline_color=_HIGH_CONTRAST_PALETTE["white"],
        outline_width=1.0,
    )


def _transform_phi_for_histogram(
    values: np.ndarray,
    *,
    y_scale: str | None,
    y_linthresh: float | None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if y_scale is None or y_linthresh is None:
        return arr
    linthresh = float(y_linthresh)
    if y_scale == "symlog":
        return _symlog_transform(arr, linthresh)
    if y_scale == "symlog2":
        return _symlog2_transform(arr, linthresh)
    return arr


def _compute_bestfit_interpolations(
    bestfit_values: dict[str, float],
    yaml_config: dict[str, Any],
    quantities: list[str],
    x_grid: np.ndarray,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], str, str]:
    class_params = _build_sample_class_params({}, bestfit_values, yaml_config)
    bg, source, cache_key = get_or_compute_background(class_params, cache_dir)

    interps: dict[str, np.ndarray] = {}
    for qty in quantities:
        if qty not in bg:
            continue
        interps[qty] = _interp_background_quantity(bg, qty, x_grid)

    return interps, source, cache_key


def _line_with_contrast(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str,
    zorder: float,
    outline_color: str,
    outline_width: float,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return

    line = ax.plot(
        x[valid],
        y[valid],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )[0]
    line.set_path_effects(
        [
            pe.Stroke(linewidth=linewidth + outline_width, foreground=outline_color),
            pe.Normal(),
        ]
    )


def _overlay_summary_on_axis(
    ax: Axes,
    z_grid: np.ndarray,
    summary: dict[str, np.ndarray] | None,
    *,
    bestfit: np.ndarray | None,
    band_color: str,
    band_alpha: float,
    median_color: str,
    bestfit_color: str,
) -> None:
    if summary is not None:
        q16 = np.asarray(summary["q16"], dtype=float)
        q50 = np.asarray(summary["q50"], dtype=float)
        q84 = np.asarray(summary["q84"], dtype=float)
        valid_band = np.isfinite(z_grid) & np.isfinite(q16) & np.isfinite(q84)
        if np.count_nonzero(valid_band) >= 2:
            ax.fill_between(
                z_grid[valid_band],
                q16[valid_band],
                q84[valid_band],
                color=band_color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=3.1,
            )
        _line_with_contrast(
            ax,
            z_grid,
            q50,
            color=median_color,
            linewidth=1.35,
            linestyle="--",
            zorder=3.8,
            outline_color=_HIGH_CONTRAST_PALETTE["black"],
            outline_width=1.2,
        )

    if bestfit is not None:
        _line_with_contrast(
            ax,
            z_grid,
            np.asarray(bestfit, dtype=float),
            color=bestfit_color,
            linewidth=1.15,
            linestyle="-",
            zorder=4.5,
            outline_color=_HIGH_CONTRAST_PALETTE["white"],
            outline_width=1.6,
        )


def _transform_summary_for_scale(
    summary: dict[str, np.ndarray] | None,
    *,
    scale: float,
) -> dict[str, np.ndarray] | None:
    """Apply a linear scale to summary curves while preserving q16<=q84."""
    if summary is None:
        return None
    if np.isclose(scale, 1.0):
        return summary

    q16_raw = np.asarray(summary["q16"], dtype=float) * scale
    q50 = np.asarray(summary["q50"], dtype=float) * scale
    q84_raw = np.asarray(summary["q84"], dtype=float) * scale
    q16 = np.minimum(q16_raw, q84_raw)
    q84 = np.maximum(q16_raw, q84_raw)
    valid_weight = np.asarray(summary.get("valid_weight", np.array([])), dtype=float)

    return {
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "valid_weight": valid_weight,
    }


def _summary_legend_handles() -> list[Any]:
    return [
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["black"],
            linewidth=1.15,
            label="Best-fit trajectory",
        ),
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["teal"],
            linewidth=1.35,
            linestyle="--",
            label="Posterior median",
        ),
        Patch(
            facecolor=_HIGH_CONTRAST_PALETTE["teal"],
            edgecolor=_HIGH_CONTRAST_PALETTE["blue"],
            linewidth=0.8,
            alpha=0.35,
            label=r"68\% confidence band",
        ),
    ]


def _summary_line_legend_handles() -> list[Any]:
    """Legend handles for shared line semantics without a generic band patch."""
    return [
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["black"],
            linewidth=1.15,
            label="Best-fit trajectory",
        ),
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["teal"],
            linewidth=1.35,
            linestyle="--",
            label="Posterior median",
        ),
    ]


def _save_legend_figure(
    handles: list[Any],
    preset: str,
    output_dir: Path,
) -> None:
    """Save legend as standalone figure(s) for inclusion in LaTeX layouts.

    Args:
        handles: List of matplotlib legend handles (Patch, Line2D, etc.)
        preset: Preset name (phi, eos, omega, swampland, swgc)
        output_dir: Directory to save legend files
    """
    fig, ax = plt.subplots(figsize=(6.0, 1.0))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=max(1, len(handles) // 3),
        frameon=False,
        fontsize=11,
        handlelength=2.2,
    )
    fig.patch.set_facecolor(_HIGH_CONTRAST_PALETTE["white"])
    fig.tight_layout(pad=0.2)

    base_name = f"legend_{preset}"
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    pgf_path = output_dir / f"{base_name}.pgf"

    try:
        fig.savefig(
            str(png_path),
            dpi=150,
            bbox_inches="tight",
            facecolor=_HIGH_CONTRAST_PALETTE["white"],
        )
        fig.savefig(
            str(pdf_path),
            bbox_inches="tight",
            facecolor=_HIGH_CONTRAST_PALETTE["white"],
        )
        fig.savefig(str(pgf_path), bbox_inches="tight")
        print(f"  Saved legend: {png_path.name}, {pdf_path.name}, {pgf_path.name}")
    finally:
        plt.close(fig)


def _get_phi_legend_handles() -> list[Any]:
    """Legend handles for phi_scf plot."""
    return [
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["black"],
            linewidth=1.15,
            label="Best-fit trajectory",
        ),
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["blue"],
            linewidth=1.15,
            linestyle="--",
            label=r"Positive-branch median $\phi>0$",
        ),
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["red"],
            linewidth=1.15,
            linestyle="--",
            label=r"Negative-branch median $\phi<0$",
        ),
        Patch(
            facecolor=_HIGH_CONTRAST_PALETTE["blue"],
            edgecolor=_HIGH_CONTRAST_PALETTE["blue"],
            linewidth=0.8,
            alpha=0.22,
            label=r"68\% band $\phi>0$",
        ),
        Patch(
            facecolor=_HIGH_CONTRAST_PALETTE["red"],
            edgecolor=_HIGH_CONTRAST_PALETTE["red"],
            linewidth=0.8,
            alpha=0.22,
            label=r"68\% band $\phi<0$",
        ),
    ]


def _get_eos_legend_handles() -> list[Any]:
    """Legend handles for w (equation of state) plot."""
    return _summary_legend_handles()


def _get_omega_legend_handles() -> list[Any]:
    """Legend handles for combined Omega (Omega_dm + Omega_scf) plot."""
    return [
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$\Omega_{\phi}$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$\Omega_{\rm DM}$",
        ),
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $\Omega_{\phi}$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $\Omega_{\rm DM}$",
        ),
        *_summary_line_legend_handles(),
    ]


def _get_dsc_legend_handles() -> list[Any]:
    """Legend handles for combined dSC (swampland coupling s1 + s2) plot."""
    return [
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$\mathfrak{s}_1$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$\mathfrak{s}_2$",
        ),
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $\mathfrak{s}_1$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $\mathfrak{s}_2$",
        ),
        *_summary_line_legend_handles(),
    ]


def _get_swgc_legend_handles() -> list[Any]:
    """Legend handles for combined SWGC (3-component, 2-panel) plot."""
    return [
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$\left(V_{\phi\phi}\right)^2$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.78),
            edgecolor="none",
            alpha=0.64,
            label=r"$2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$",
        ),
        Patch(
            facecolor=_OMEGA_PHI_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $\left(V_{\phi\phi}\right)^2$",
        ),
        Patch(
            facecolor=_OMEGA_DM_CMAP(0.58),
            edgecolor="none",
            alpha=0.22,
            label=r"68\% confidence band $2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$",
        ),
        Patch(
            facecolor=_HEATMAP_CMAP(0.58),
            edgecolor="none",
            linewidth=0.8,
            alpha=0.22,
            label=r"68\% confidence band $\Delta_{\rm SWGC}$",
        ),
        Line2D(
            [0],
            [0],
            color=_HIGH_CONTRAST_PALETTE["black"],
            linewidth=1.3,
            label=(
                r"$\Delta_{\rm SWGC}$"
                r"$=2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}"
                r"-\left(V_{\phi\phi}\right)^2$"
            ),
        ),
        *_summary_line_legend_handles(),
    ]


def _legend_presets_for_quantities(quantities: list[str]) -> list[str]:
    """Return ordered legend presets needed for the requested quantities."""
    qset = set(quantities)
    presets: list[str] = []

    if "phi_scf" in qset:
        presets.append("phi")
    if "w" in qset:
        presets.append("eos")
    if {"Omega_cdm", "Omega_scf"} & qset:
        presets.append("omega")
    if {"s1", "minus_s2"} & qset:
        presets.append("swampland")
    if {"swgc_lhs", "swgc_rhs", "swgc_residual"} & qset:
        presets.append("swgc")

    return presets


def _save_requested_legends_once(quantities: list[str], output_dir: Path) -> None:
    """Save standalone legends once per preset for the requested quantity set."""
    for preset in _legend_presets_for_quantities(quantities):
        if preset == "phi":
            _save_legend_figure(_get_phi_legend_handles(), preset, output_dir)
        elif preset == "eos":
            _save_legend_figure(_get_eos_legend_handles(), preset, output_dir)
        elif preset == "omega":
            _save_legend_figure(_get_omega_legend_handles(), preset, output_dir)
        elif preset == "swampland":
            _save_legend_figure(_get_dsc_legend_handles(), preset, output_dir)
        elif preset == "swgc":
            _save_legend_figure(_get_swgc_legend_handles(), preset, output_dir)


def _build_sample_class_params(
    row_map: dict[str, float],
    bestfit_values: dict[str, float],
    yaml_config: dict[str, Any],
) -> dict[str, Any]:
    sample_bestfit = dict(bestfit_values)
    sample_bestfit.update(row_map)

    warnings: list[str] = []
    class_params = extract_class_parameters(
        sample_bestfit, yaml_config, warnings=warnings
    )

    # Ensure stable yes/no serialization for bool-like flags.
    for k, v in list(class_params.items()):
        if isinstance(v, bool):
            class_params[k] = "yes" if v else "no"

    return class_params


def _process_single_trajectory(
    bundle_root: str,
    chain_files: list[Path],
    rec_index: tuple[int, DrawRecord],
    header: list[str],
    extracted_by_file: dict[int, dict[int, list[float]]],
    bestfit_values: dict[str, float],
    yaml_config: dict[str, Any],
    quantities: list[str],
    x_grid: np.ndarray,
    cache_dir: Path,
    failure_audit_dir: Path,
) -> TrajectoryResult | None:
    """Process a single trajectory: extract, run CLASS, interpolate quantities.

    Returns TrajectoryResult if valid (>= 2 finite points per quantity), else None.
    On CLASS failure, logs error and returns None (skipped).
    """
    i, rec = rec_index
    chain_file = chain_files[rec.file_index]
    row_values = extracted_by_file[rec.file_index][rec.row_index_postburn]
    row_map = {name: float(row_values[j]) for j, name in enumerate(header)}

    class_params = _build_sample_class_params(row_map, bestfit_values, yaml_config)

    try:
        bg, source, _ = get_or_compute_background(class_params, cache_dir)
    except ClassBackgroundRunError as e:
        audit_json = _write_failure_audit(
            failure_audit_dir,
            bundle_root=bundle_root,
            chain_file=chain_file,
            trajectory_index=i,
            record=rec,
            row_map=row_map,
            class_params=class_params,
            error=e,
        )
        print(
            f"    [WARNING] Trajectory {i}: CLASS background run failed. "
            f"Skipping. Audit: {audit_json}"
        )
        return None

    cache_hit = source == "cache"

    qty_interps: dict[str, np.ndarray] = {}
    for qty in quantities:
        if qty not in bg:
            continue
        y_interp = _interp_background_quantity(bg, qty, x_grid)
        valid = np.isfinite(y_interp)
        if np.count_nonzero(valid) >= 2:
            qty_interps[qty] = y_interp

    if not qty_interps:
        return None

    return TrajectoryResult(
        qty_interps=qty_interps,
        multiplicity=rec.multiplicity,
        cache_hit=cache_hit,
    )


def process_dataset(
    bundle: ChainBundle,
    args: argparse.Namespace,
    quantities: list[str],
    rng: np.random.Generator,
    mode_tuning: dict[str, Any],
    cache_dir: Path,
    output_dir: Path,
    failure_audit_dir: Path,
) -> None:
    """Run the full heatmap pipeline for one dataset root.

    All quantities in `quantities` share a single CLASS run per resampled point,
    reusing the persistent background cache.  One figure + NPZ is saved per quantity.
    """
    print(f"\n=== Dataset root: {bundle.root} ===")
    print(f"  Quantities: {quantities}")

    selection = _load_selection_data(
        bundle=bundle,
        hpd_params=args.hpd_params,
        ignore_rows=args.ignore_rows,
    )

    n_rows = selection.weights.size
    neff = (selection.weights.sum() ** 2) / max(np.sum(selection.weights**2), 1e-300)
    print(
        f"Rows(post-burn)={n_rows:,}, sum(weights)={selection.weights.sum():.3g}, "
        f"N_eff~{neff:,.0f}"
    )

    uniq_global_idx, multiplicities, sampling_diag, sampling_fp = (
        _resolve_sampling_plan(
            bundle=bundle,
            selection=selection,
            args=args,
            mode_tuning=mode_tuning,
            rng=rng,
            cache_dir=cache_dir,
        )
    )

    print(
        "Sampling plan: "
        f"cache={sampling_diag.get('sampling_plan_cache', 'unknown')}, "
        f"fingerprint={sampling_fp}, "
        f"path={sampling_diag.get('sampling_plan_path', 'n/a')}"
    )
    print(
        "Mode-aware resampling: "
        f"draws={sampling_diag.get('draw_count', int(np.sum(multiplicities))):,}, "
        f"unique={sampling_diag.get('unique_trajectories', int(uniq_global_idx.size)):,}, "
        f"modes={sampling_diag.get('mode_count', 0)}, "
        f"eligible={sampling_diag.get('eligible_modes', 0)}, "
        f"floor_modes={sampling_diag.get('floor_supported_modes', 0)}, "
        f"floor_draws={sampling_diag.get('floor_total_draws', 0)}"
    )

    draw_records = _global_to_file_rows(
        uniq_global_idx,
        multiplicities,
        selection.file_counts_postburn,
    )

    total_draws = int(np.sum(multiplicities))
    print(
        f"Resampling: total draws={total_draws:,}, unique trajectories={len(draw_records):,}"
    )

    if args.dry_run:
        return

    header = read_chain_header(bundle.chain_files[0])

    # Group requested row indices by source file.
    rows_by_file: dict[int, set[int]] = {}
    for rec in draw_records:
        rows_by_file.setdefault(rec.file_index, set()).add(rec.row_index_postburn)

    extracted_by_file: dict[int, dict[int, list[float]]] = {}
    for file_idx, row_set in sorted(rows_by_file.items()):
        fp = bundle.chain_files[file_idx]
        extracted_by_file[file_idx] = _extract_rows_by_index(
            fp,
            row_set,
            ignore_rows=args.ignore_rows,
        )

    bestfit_values = load_bestfit_file(str(bundle.bestfit_file))
    yaml_config = load_input_yaml(str(bundle.input_yaml))

    x_grid = _build_common_x_grid(
        args.x_min, args.x_max, n_points=max(256, args.x_bins * 2)
    )
    z_grid = np.power(10.0, x_grid) - 1.0

    try:
        bestfit_interps, bestfit_source, bestfit_cache_key = (
            _compute_bestfit_interpolations(
                bestfit_values,
                yaml_config,
                quantities,
                x_grid,
                cache_dir,
            )
        )
    except ClassBackgroundRunError as exc:
        raise RuntimeError(
            f"Best-fit CLASS background replay failed for root '{bundle.root}'. "
            "The overlay layer cannot be generated consistently."
        ) from exc

    print(
        "Best-fit background: "
        f"source={bestfit_source}, cache_key={bestfit_cache_key}, "
        f"quantities={sorted(bestfit_interps)}"
    )

    # Pass 1: run CLASS (or load cache) once per unique sample; interpolate all quantities.
    # With parallelization, process trajectories concurrently (threads safe for I/O-bound CLASS runs).
    # y_ranges[qty] = [min, max] accumulated across trajectories.
    y_ranges: dict[str, list[float]] = {qty: [np.inf, -np.inf] for qty in quantities}
    cache_hits = 0
    cache_misses = 0
    skipped = 0

    # Each element: per-quantity interpolated y-values on x_grid, plus multiplicity.
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]] = []

    # Determine thread pool size: 0 = use all cores, >0 = explicit limit.
    num_threads = args.num_threads if args.num_threads > 0 else None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks in order.
        futures = [
            (
                i,
                executor.submit(
                    _process_single_trajectory,
                    bundle.root,
                    bundle.chain_files,
                    (i, rec),
                    header,
                    extracted_by_file,
                    bestfit_values,
                    yaml_config,
                    quantities,
                    x_grid,
                    cache_dir,
                    failure_audit_dir,
                ),
            )
            for i, rec in enumerate(draw_records, 1)
        ]

        # Collect results maintaining order.
        for i, future in futures:
            traj_result = future.result()

            if traj_result is None:
                skipped += 1
            else:
                if traj_result.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1

                for qty, y_interp in traj_result.qty_interps.items():
                    valid = np.isfinite(y_interp)
                    if np.count_nonzero(valid) >= 2:
                        y_ranges[qty][0] = min(
                            y_ranges[qty][0], float(np.nanmin(y_interp[valid]))
                        )
                        y_ranges[qty][1] = max(
                            y_ranges[qty][1], float(np.nanmax(y_interp[valid]))
                        )

                trajectory_payload.append(
                    (traj_result.qty_interps, traj_result.multiplicity)
                )

            if i % 50 == 0 or i == len(draw_records):
                print(
                    f"  processed {i}/{len(draw_records)} unique trajectories "
                    f"(cache hit={cache_hits}, miss={cache_misses}, skipped={skipped})"
                )

    if not trajectory_payload:
        raise RuntimeError(
            f"No valid trajectories for {bundle.root} after interpolation"
        )

    quantity_summaries: dict[str, dict[str, np.ndarray]] = {}
    for qty in quantities:
        summary = _build_quantity_summary(trajectory_payload, qty)
        if summary is not None:
            quantity_summaries[qty] = summary

    dataset_label = _dataset_label_from_root(bundle.root)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_label_map: dict[str, str] = {
        "phi_scf": r"$\phi$",
        "w": r"$w$",
        "Omega_cdm": r"$\Omega_{\rm cdm}$",
        "Omega_scf": r"$\Omega_{\phi}$",
        "s1": r"$\mathfrak{s}_1 = |V_{\phi}|/V$",
        "minus_s2": r"$\mathfrak{s}_2 = -V_{\phi\phi}/V$",
        "swampland_expr": r"$1+w-0.15\,\mathfrak{s}_1^2$",
        "swgc_lhs": r"$(V'')^2$",
        "swgc_rhs": r"$2(V''')^2 - V''V''''$",
        "swgc_residual": r"$\Delta_{\rm SWGC} = 2(V''')^2 - V''V'''' - (V'')^2$",
    }

    omega_combined_only = {"Omega_cdm", "Omega_scf"}.issubset(set(quantities))
    dsc_requested = {"s1", "minus_s2"}.issubset(set(quantities))
    dsc_combined_only = dsc_requested and args.dsc_layout == "combined"
    swgc_requested = {"swgc_lhs", "swgc_rhs", "swgc_residual"}.issubset(set(quantities))
    swgc_combined_only = swgc_requested and args.swgc_layout == "combined"
    swgc_stacked_only = swgc_requested and args.swgc_layout == "stacked"

    # Pass 2: for each quantity, accumulate histogram and produce figure + NPZ.
    for qty in quantities:
        if omega_combined_only and qty in {"Omega_cdm", "Omega_scf"}:
            continue
        if dsc_combined_only and qty in {"s1", "minus_s2"}:
            continue
        if swgc_requested and qty in {"swgc_lhs", "swgc_rhs", "swgc_residual"}:
            continue

        dsc_split_minus_s2 = (
            qty == "minus_s2" and dsc_requested and args.dsc_layout == "split"
        )
        qty_scale = -1.0 if dsc_split_minus_s2 else 1.0

        y_min_qty, y_max_qty = y_ranges[qty]
        if dsc_split_minus_s2:
            y_min_qty, y_max_qty = (-y_max_qty, -y_min_qty)
        if (
            not np.isfinite(y_min_qty)
            or not np.isfinite(y_max_qty)
            or y_min_qty == y_max_qty
        ):
            print(
                f"  [WARN] Skipping {qty}: invalid y-range [{y_min_qty}, {y_max_qty}]"
            )
            continue

        y_edges, y_scale, y_linthresh = _build_quantity_y_edges(
            y_min_qty,
            y_max_qty,
            args.y_bins,
            qty,
            args.phi_y_scale,
        )
        phi_crossing_profile: dict[str, np.ndarray] | None = None
        phi_branch_summary: dict[str, np.ndarray] | None = None
        if qty == "phi_scf":
            phi_crossing_profile = _compute_phi_crossing_profile(
                trajectory_payload,
                epsilon_mode=args.phi_crossing_epsilon_mode,
                epsilon_abs=args.phi_crossing_epsilon_abs,
                epsilon_linthresh_frac=args.phi_crossing_epsilon_linthresh_frac,
                epsilon_quantile=args.phi_crossing_epsilon_quantile,
                y_linthresh=y_linthresh,
            )
            phi_branch_summary = _compute_phi_sign_branch_summary(
                trajectory_payload,
                phi_crossing_profile,
            )
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)
        z_edges = np.power(10.0, x_edges) - 1.0

        # For phi, accumulate directly in transformed y-space so density and display
        # live in the same coordinate system.
        y_edges_hist = np.asarray(y_edges, dtype=float)
        if qty == "phi_scf":
            y_edges_hist = _transform_phi_for_histogram(
                y_edges_hist,
                y_scale=y_scale,
                y_linthresh=y_linthresh,
            )

        H = np.zeros((args.x_bins, args.y_bins), dtype=float)
        for qty_interps, multiplicity in trajectory_payload:
            if qty not in qty_interps:
                continue
            y_interp = np.asarray(qty_interps[qty], dtype=float) * qty_scale
            y_for_hist = np.asarray(y_interp, dtype=float)
            if qty == "phi_scf":
                y_for_hist = _transform_phi_for_histogram(
                    y_for_hist,
                    y_scale=y_scale,
                    y_linthresh=y_linthresh,
                )
            valid = np.isfinite(y_for_hist)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 2:
                continue
            w = np.full(n_valid, float(multiplicity), dtype=float)
            h2d, _, _ = np.histogram2d(
                x_grid[valid],
                y_for_hist[valid],
                bins=(x_edges, y_edges_hist),  # type: ignore[arg-type]
                weights=w,
            )
            H += h2d

        if not np.any(H > 0):
            print(f"  [WARN] Empty histogram for {qty}, skipping.")
            continue

        ax_cross: Axes | None = None
        if qty == "phi_scf" and args.phi_crossing_overlay == "probability":
            fig, (ax, ax_cross) = plt.subplots(
                2,
                1,
                figsize=(6.4, 5.25),
                sharex=True,
                gridspec_kw={"height_ratios": [4.2, 1.15], "hspace": 0.06},
            )
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.1))

        H_plot = H.T
        positive = H_plot[H_plot > 0]
        vmin = max(float(np.percentile(positive, 5.0)), 1e-12)
        vmax = float(np.percentile(positive, 99.8))
        if vmax <= vmin:
            vmax = float(np.max(positive))

        quantity_cmap = _HEATMAP_CMAP
        if dsc_requested and args.dsc_layout == "split":
            if qty == "s1":
                quantity_cmap = _OMEGA_PHI_CMAP
            elif qty == "minus_s2":
                quantity_cmap = _OMEGA_DM_CMAP

        mesh = ax.pcolormesh(
            z_edges,
            y_edges,
            H_plot,
            cmap=quantity_cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )

        if ax_cross is not None:
            # Place one shared colorbar strictly spanning the combined subplot stack.
            pos_top = ax.get_position()
            pos_bottom = ax_cross.get_position()
            y0 = min(pos_top.y0, pos_bottom.y0)
            y1 = max(pos_top.y1, pos_bottom.y1)
            x1 = max(pos_top.x1, pos_bottom.x1)
            cbar_rect: tuple[float, float, float, float] = (
                float(x1) + 0.018,
                float(y0),
                0.030,
                float(y1 - y0),
            )
            cax = fig.add_axes(cbar_rect)
            cb = fig.colorbar(mesh, cax=cax)
        else:
            cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("Posterior Path Density")
        _style_colorbar(cb, vmin, vmax)

        qty_label = y_label_map.get(qty, qty)
        if dsc_split_minus_s2:
            qty_label = r"$-\mathfrak{s}_2 = V_{\phi\phi}/V$"
        ax.set_ylabel(qty_label)
        if not (
            qty == "phi_scf"
            and args.phi_crossing_overlay == "probability"
            and ax_cross is not None
        ):
            _style_redshift_axis(ax, z_edges)
        if y_scale == "symlog" and y_linthresh is not None:
            ax.set_yscale("symlog", linthresh=y_linthresh)
        elif y_scale == "symlog2" and y_linthresh is not None:
            linthresh_value = cast(float, y_linthresh)
            ax.set_yscale(
                "function",
                functions=(
                    lambda values: _symlog2_transform(
                        np.asarray(values), linthresh_value
                    ),
                    lambda values: _symlog2_inverse(
                        np.asarray(values), linthresh_value
                    ),
                ),
            )

        single_band_color = _HIGH_CONTRAST_PALETTE["teal"]
        single_band_alpha = 0.30
        single_median_color = _HIGH_CONTRAST_PALETTE["white"]
        single_bestfit_color = _HIGH_CONTRAST_PALETTE["black"]
        if qty == "w":
            # Improve contrast for w(z): dark band stands out on the warm heatmap.
            single_band_color = _HIGH_CONTRAST_PALETTE["blue"]
            single_band_alpha = 0.40
            single_median_color = _HIGH_CONTRAST_PALETTE["black"]
        if dsc_requested and args.dsc_layout == "split":
            if qty == "s1":
                single_band_color = _OMEGA_PHI_CMAP(0.58)
                single_band_alpha = 0.22
                single_median_color = _OMEGA_PHI_CMAP(0.97)
                single_bestfit_color = _OMEGA_PHI_CMAP(0.32)
            elif qty == "minus_s2":
                single_band_color = _OMEGA_DM_CMAP(0.58)
                single_band_alpha = 0.22
                single_median_color = _OMEGA_DM_CMAP(0.97)
                single_bestfit_color = _OMEGA_DM_CMAP(0.28)

        summary_for_plot = _transform_summary_for_scale(
            quantity_summaries.get(qty),
            scale=qty_scale,
        )
        bestfit_for_plot = (
            (np.asarray(bestfit_interps.get(qty), dtype=float) * qty_scale)
            if bestfit_interps.get(qty) is not None
            else None
        )

        if qty == "phi_scf":
            _overlay_summary_on_axis(
                ax,
                z_grid,
                None,
                bestfit=bestfit_for_plot,
                band_color=single_band_color,
                band_alpha=single_band_alpha,
                median_color=single_median_color,
                bestfit_color=single_bestfit_color,
            )
            _overlay_phi_sign_branch_summary(
                ax,
                z_grid,
                phi_branch_summary,
            )
            if args.phi_crossing_overlay == "probability" and ax_cross is not None:
                p_cross = np.asarray(
                    (
                        phi_crossing_profile["p_cross"]
                        if phi_crossing_profile is not None
                        else np.full_like(z_grid, np.nan)
                    ),
                    dtype=float,
                )
                valid_cross = np.isfinite(z_grid) & np.isfinite(p_cross)
                if np.count_nonzero(valid_cross) >= 2:
                    ax_cross.plot(
                        z_grid[valid_cross],
                        p_cross[valid_cross],
                        color=_HIGH_CONTRAST_PALETTE["red"],
                        linewidth=1.25,
                        linestyle="-",
                        alpha=0.95,
                        zorder=3.0,
                    )
                ax_cross.set_ylim(0.0, 1.0)
                ax_cross.set_ylabel(r"$P_{\rm cross}$")
                ax_cross.tick_params(axis="y", which="major", labelsize=10)

                # Use one common shared redshift axis definition (lower panel only).
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both", labelbottom=False)
                _style_redshift_axis(ax_cross, z_edges)
            else:
                _overlay_phi_crossing_diagnostic(
                    ax,
                    z_grid,
                    phi_crossing_profile,
                    overlay_mode=args.phi_crossing_overlay,
                    binary_threshold=args.phi_crossing_binary_threshold,
                )
        else:
            _overlay_summary_on_axis(
                ax,
                z_grid,
                summary_for_plot,
                bestfit=bestfit_for_plot,
                band_color=single_band_color,
                band_alpha=single_band_alpha,
                median_color=single_median_color,
                bestfit_color=single_bestfit_color,
            )

        if args.include_legends_in_plots:
            if qty == "phi_scf":
                ax.legend(handles=_get_phi_legend_handles(), loc="best", frameon=False)
            else:
                ax.legend(handles=_summary_legend_handles(), loc="best", frameon=False)

        if not (qty == "phi_scf" and args.phi_crossing_overlay == "probability"):
            fig.tight_layout()
        if y_scale in {"symlog", "symlog2"} and qty == "phi_scf":
            _hide_overlaps_with_zero_ytick_label(
                fig,
                ax,
                overlap_margin_px=args.zero_label_overlap_margin_px,
            )

        output_quantity = _output_quantity_name(qty)
        output_base = _output_base_path(output_dir, bundle.root, output_quantity)
        png_path = output_dir / f"{output_base.name}.png"
        pdf_path = output_dir / f"{output_base.name}.pdf"
        pgf_path = output_dir / f"{output_base.name}.pgf"
        npz_path = output_dir / f"{output_base.name}.npz"

        _save_figure_bundle(fig, output_base)
        plt.close(fig)

        np.savez_compressed(
            npz_path,
            H=H,
            x_edges=x_edges,
            y_edges=y_edges,
            x_grid=x_grid,
            z_grid=z_grid,
            quantity=np.array([output_quantity], dtype=str),
            root=np.array([bundle.root], dtype=str),
            dataset_label=np.array([dataset_label], dtype=str),
            total_draws=np.array([total_draws], dtype=np.int64),
            unique_trajectories=np.array([len(draw_records)], dtype=np.int64),
            cache_hits=np.array([cache_hits], dtype=np.int64),
            cache_misses=np.array([cache_misses], dtype=np.int64),
            bestfit_curve=np.asarray(
                (
                    bestfit_for_plot
                    if bestfit_for_plot is not None
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            q16=np.asarray(
                (summary_for_plot or {}).get("q16", np.full_like(x_grid, np.nan)),
                dtype=float,
            ),
            q50=np.asarray(
                (summary_for_plot or {}).get("q50", np.full_like(x_grid, np.nan)),
                dtype=float,
            ),
            q84=np.asarray(
                (summary_for_plot or {}).get("q84", np.full_like(x_grid, np.nan)),
                dtype=float,
            ),
            summary_valid_weight=np.asarray(
                (summary_for_plot or {}).get(
                    "valid_weight", np.zeros_like(x_grid, dtype=float)
                ),
                dtype=float,
            ),
            sample_plan_fingerprint=np.array([sampling_fp], dtype=str),
            sample_plan_cache=np.array(
                [str(sampling_diag.get("sampling_plan_cache", "unknown"))], dtype=str
            ),
            bestfit_background_source=np.array([bestfit_source], dtype=str),
            bestfit_background_cache_key=np.array([bestfit_cache_key], dtype=str),
            phi_crossing_overlay_mode=np.array(
                [args.phi_crossing_overlay if qty == "phi_scf" else "none"],
                dtype=str,
            ),
            phi_crossing_epsilon_mode=np.array(
                [args.phi_crossing_epsilon_mode if qty == "phi_scf" else "none"],
                dtype=str,
            ),
            phi_crossing_binary_threshold=np.array(
                [args.phi_crossing_binary_threshold if qty == "phi_scf" else np.nan],
                dtype=float,
            ),
            phi_crossing_epsilon_abs=np.array(
                [args.phi_crossing_epsilon_abs if qty == "phi_scf" else np.nan],
                dtype=float,
            ),
            phi_crossing_epsilon_linthresh_frac=np.array(
                [
                    (
                        args.phi_crossing_epsilon_linthresh_frac
                        if qty == "phi_scf"
                        else np.nan
                    )
                ],
                dtype=float,
            ),
            phi_crossing_epsilon_quantile=np.array(
                [args.phi_crossing_epsilon_quantile if qty == "phi_scf" else np.nan],
                dtype=float,
            ),
            phi_crossing_probability=np.asarray(
                (
                    phi_crossing_profile["p_cross"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_probability_positive=np.asarray(
                (
                    phi_crossing_profile["p_pos"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_probability_negative=np.asarray(
                (
                    phi_crossing_profile["p_neg"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_probability_near_zero=np.asarray(
                (
                    phi_crossing_profile["p_near"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_crossing_epsilon=np.asarray(
                (
                    phi_crossing_profile["epsilon"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_crossing_valid_weight=np.asarray(
                (
                    phi_crossing_profile["valid_weight"]
                    if (qty == "phi_scf" and phi_crossing_profile is not None)
                    else np.zeros_like(x_grid, dtype=float)
                ),
                dtype=float,
            ),
            phi_histogram_space=np.array(
                [
                    (
                        "transformed"
                        if (qty == "phi_scf" and y_scale in {"symlog", "symlog2"})
                        else "physical"
                    )
                ],
                dtype=str,
            ),
            phi_branch_pos_q16=np.asarray(
                (
                    phi_branch_summary["pos_q16"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_pos_q50=np.asarray(
                (
                    phi_branch_summary["pos_q50"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_pos_q84=np.asarray(
                (
                    phi_branch_summary["pos_q84"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_neg_q16=np.asarray(
                (
                    phi_branch_summary["neg_q16"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_neg_q50=np.asarray(
                (
                    phi_branch_summary["neg_q50"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_neg_q84=np.asarray(
                (
                    phi_branch_summary["neg_q84"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.full_like(x_grid, np.nan)
                ),
                dtype=float,
            ),
            phi_branch_pos_valid_weight=np.asarray(
                (
                    phi_branch_summary["pos_valid_weight"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.zeros_like(x_grid, dtype=float)
                ),
                dtype=float,
            ),
            phi_branch_neg_valid_weight=np.asarray(
                (
                    phi_branch_summary["neg_valid_weight"]
                    if (qty == "phi_scf" and phi_branch_summary is not None)
                    else np.zeros_like(x_grid, dtype=float)
                ),
                dtype=float,
            ),
        )

        print(
            f"  [{qty}] Saved: {png_path.name}, {pdf_path.name}, {pgf_path.name}, {npz_path.name}"
        )

    # Combined Omega figure (analogous to BestFitPlot omega panel):
    # Omega_phi and Omega_DM in one panel with fixed y-range [0, 1].
    if omega_combined_only:
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)
        z_edges = np.power(10.0, x_edges) - 1.0
        y_edges_omega = np.linspace(0.0, 1.0, args.y_bins + 1)

        H_omega_dm = np.zeros((args.x_bins, args.y_bins), dtype=float)
        H_omega_scf = np.zeros((args.x_bins, args.y_bins), dtype=float)

        for qty_interps, multiplicity in trajectory_payload:
            y_dm = qty_interps.get("Omega_cdm")
            y_scf = qty_interps.get("Omega_scf")

            if y_dm is not None:
                valid_dm = np.isfinite(y_dm)
                if np.count_nonzero(valid_dm) >= 2:
                    w_dm = np.full(int(np.count_nonzero(valid_dm)), float(multiplicity))
                    h_dm, _, _ = np.histogram2d(
                        x_grid[valid_dm],
                        y_dm[valid_dm],
                        bins=(x_edges, y_edges_omega),  # type: ignore[arg-type]
                        weights=w_dm,
                    )
                    H_omega_dm += h_dm

            if y_scf is not None:
                valid_scf = np.isfinite(y_scf)
                if np.count_nonzero(valid_scf) >= 2:
                    w_scf = np.full(
                        int(np.count_nonzero(valid_scf)), float(multiplicity)
                    )
                    h_scf, _, _ = np.histogram2d(
                        x_grid[valid_scf],
                        y_scf[valid_scf],
                        bins=(x_edges, y_edges_omega),  # type: ignore[arg-type]
                        weights=w_scf,
                    )
                    H_omega_scf += h_scf

        if np.any(H_omega_dm > 0) or np.any(H_omega_scf > 0):
            fig_omega, ax_top = plt.subplots(figsize=(6.4, 4.1))
            mesh_omega_phi = None
            mesh_omega_dm = None
            omega_phi_limits: tuple[float, float] | None = None
            omega_dm_limits: tuple[float, float] | None = None

            if np.any(H_omega_scf > 0):
                pos_scf = H_omega_scf[H_omega_scf > 0]
                vmin_scf = max(float(np.percentile(pos_scf, 5.0)), 1e-12)
                vmax_scf = float(np.percentile(pos_scf, 99.8))
                if vmax_scf <= vmin_scf:
                    vmax_scf = float(np.max(pos_scf))
                mesh_omega_phi = ax_top.pcolormesh(
                    z_edges,
                    y_edges_omega,
                    H_omega_scf.T,
                    cmap=_OMEGA_PHI_CMAP,
                    norm=LogNorm(vmin=vmin_scf, vmax=vmax_scf),
                    shading="auto",
                    alpha=0.64,
                )
                omega_phi_limits = (vmin_scf, vmax_scf)

            if np.any(H_omega_dm > 0):
                pos_dm = H_omega_dm[H_omega_dm > 0]
                vmin_dm = max(float(np.percentile(pos_dm, 5.0)), 1e-12)
                vmax_dm = float(np.percentile(pos_dm, 99.8))
                if vmax_dm <= vmin_dm:
                    vmax_dm = float(np.max(pos_dm))
                mesh_omega_dm = ax_top.pcolormesh(
                    z_edges,
                    y_edges_omega,
                    H_omega_dm.T,
                    cmap=_OMEGA_DM_CMAP,
                    norm=LogNorm(vmin=vmin_dm, vmax=vmax_dm),
                    shading="auto",
                    alpha=0.64,
                )
                omega_dm_limits = (vmin_dm, vmax_dm)

            _style_redshift_axis(ax_top, z_edges)
            ax_top.set_ylim(0.0, 1.0)
            ax_top.set_ylabel(r"Relative Energy Density $\Omega$")

            _overlay_summary_on_axis(
                ax_top,
                z_grid,
                quantity_summaries.get("Omega_scf"),
                bestfit=bestfit_interps.get("Omega_scf"),
                band_color=_OMEGA_PHI_CMAP(0.58),
                band_alpha=0.22,
                median_color=_OMEGA_PHI_CMAP(0.97),
                bestfit_color=_OMEGA_PHI_CMAP(0.32),
            )
            _overlay_summary_on_axis(
                ax_top,
                z_grid,
                quantity_summaries.get("Omega_cdm"),
                bestfit=bestfit_interps.get("Omega_cdm"),
                band_color=_OMEGA_DM_CMAP(0.58),
                band_alpha=0.22,
                median_color=_OMEGA_DM_CMAP(0.97),
                bestfit_color=_OMEGA_DM_CMAP(0.28),
            )

            if args.include_legends_in_plots:
                ax_top.legend(
                    handles=_get_omega_legend_handles(),
                    loc="best",
                    frameon=False,
                )

            # Add both reference colorbars (one per overlaid Omega component).
            divider = make_axes_locatable(ax_top)
            cax_dm = None
            dm_pad = 0.26
            if mesh_omega_dm is not None and omega_dm_limits is not None:
                cax_dm = divider.append_axes("right", size="2.8%", pad=dm_pad)
                cb_dm = fig_omega.colorbar(mesh_omega_dm, cax=cax_dm)
                cb_dm.set_label(r"$\Omega_{\rm DM}$ Path Density")
                cb_dm.ax.yaxis.set_label_position("left")
                _style_colorbar(cb_dm, omega_dm_limits[0], omega_dm_limits[1])

            if mesh_omega_phi is not None and omega_phi_limits is not None:
                # Keep the DE bar nearly fixed while nudging the DM bar outward.
                phi_pad = 0.8 - dm_pad if cax_dm is not None else 0.20
                cax_phi = divider.append_axes("right", size="2.8%", pad=phi_pad)
                cb_phi = fig_omega.colorbar(mesh_omega_phi, cax=cax_phi)
                cb_phi.set_label(r"$\Omega_{\phi}$ Path Density")
                cb_phi.ax.yaxis.set_label_position("left")
                _style_colorbar(cb_phi, omega_phi_limits[0], omega_phi_limits[1])

            fig_omega.tight_layout()

            omega_base = _output_base_path(output_dir, bundle.root, "Omega")
            _save_figure_bundle(fig_omega, omega_base)
            plt.close(fig_omega)

            omega_npz_path = output_dir / f"{omega_base.name}.npz"
            save_payload: dict[str, Any] = {
                "H_omega_dm": H_omega_dm,
                "H_omega_scf": H_omega_scf,
                "x_edges": x_edges,
                "y_edges_omega": y_edges_omega,
                "x_grid": x_grid,
                "z_grid": z_grid,
                "quantity": np.array(["Omega"], dtype=str),
                "root": np.array([bundle.root], dtype=str),
                "dataset_label": np.array([dataset_label], dtype=str),
                "total_draws": np.array([total_draws], dtype=np.int64),
                "unique_trajectories": np.array([len(draw_records)], dtype=np.int64),
                "sample_plan_fingerprint": np.array([sampling_fp], dtype=str),
                "omega_cdm_bestfit": np.asarray(
                    bestfit_interps.get("Omega_cdm", np.full_like(x_grid, np.nan)),
                    dtype=float,
                ),
                "omega_cdm_q16": np.asarray(
                    quantity_summaries.get("Omega_cdm", {}).get(
                        "q16", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_cdm_q50": np.asarray(
                    quantity_summaries.get("Omega_cdm", {}).get(
                        "q50", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_cdm_q84": np.asarray(
                    quantity_summaries.get("Omega_cdm", {}).get(
                        "q84", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_cdm_valid_weight": np.asarray(
                    quantity_summaries.get("Omega_cdm", {}).get(
                        "valid_weight", np.zeros_like(x_grid, dtype=float)
                    ),
                    dtype=float,
                ),
                "omega_scf_bestfit": np.asarray(
                    bestfit_interps.get("Omega_scf", np.full_like(x_grid, np.nan)),
                    dtype=float,
                ),
                "omega_scf_q16": np.asarray(
                    quantity_summaries.get("Omega_scf", {}).get(
                        "q16", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_scf_q50": np.asarray(
                    quantity_summaries.get("Omega_scf", {}).get(
                        "q50", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_scf_q84": np.asarray(
                    quantity_summaries.get("Omega_scf", {}).get(
                        "q84", np.full_like(x_grid, np.nan)
                    ),
                    dtype=float,
                ),
                "omega_scf_valid_weight": np.asarray(
                    quantity_summaries.get("Omega_scf", {}).get(
                        "valid_weight", np.zeros_like(x_grid, dtype=float)
                    ),
                    dtype=float,
                ),
                "bestfit_background_source": np.array([bestfit_source], dtype=str),
                "bestfit_background_cache_key": np.array(
                    [bestfit_cache_key], dtype=str
                ),
            }

            np.savez_compressed(omega_npz_path, **save_payload)

            print(
                "  [omega-combined] Saved: "
                f"{omega_base.name}.png, {omega_base.name}.pdf, {omega_base.name}.pgf, "
                f"{omega_npz_path.name}"
            )

    # Combined dSC figure: s1 and s2 (minus_s2) overlaid in one symlog-y panel.
    if dsc_combined_only:
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)
        z_edges = np.power(10.0, x_edges) - 1.0

        y_min_dsc = min(y_ranges["s1"][0], y_ranges["minus_s2"][0])
        y_max_dsc = max(y_ranges["s1"][1], y_ranges["minus_s2"][1])
        if np.isfinite(y_min_dsc) and np.isfinite(y_max_dsc) and y_min_dsc < y_max_dsc:
            y_edges_dsc, y_linthresh_dsc = _build_symlog_y_edges(
                y_min_dsc,
                y_max_dsc,
                args.y_bins,
            )

            H_dsc_s1 = np.zeros((args.x_bins, args.y_bins), dtype=float)
            H_dsc_s2 = np.zeros((args.x_bins, args.y_bins), dtype=float)

            for qty_interps, multiplicity in trajectory_payload:
                y_s1 = qty_interps.get("s1")
                y_s2 = qty_interps.get("minus_s2")

                if y_s1 is not None:
                    valid_s1 = np.isfinite(y_s1)
                    n_valid_s1 = int(np.count_nonzero(valid_s1))
                    if n_valid_s1 >= 2:
                        w_s1 = np.full(n_valid_s1, float(multiplicity), dtype=float)
                        h_s1, _, _ = np.histogram2d(
                            x_grid[valid_s1],
                            y_s1[valid_s1],
                            bins=(x_edges, y_edges_dsc),  # type: ignore[arg-type]
                            weights=w_s1,
                        )
                        H_dsc_s1 += h_s1

                if y_s2 is not None:
                    valid_s2 = np.isfinite(y_s2)
                    n_valid_s2 = int(np.count_nonzero(valid_s2))
                    if n_valid_s2 >= 2:
                        w_s2 = np.full(n_valid_s2, float(multiplicity), dtype=float)
                        h_s2, _, _ = np.histogram2d(
                            x_grid[valid_s2],
                            y_s2[valid_s2],
                            bins=(x_edges, y_edges_dsc),  # type: ignore[arg-type]
                            weights=w_s2,
                        )
                        H_dsc_s2 += h_s2

            if np.any(H_dsc_s1 > 0) or np.any(H_dsc_s2 > 0):
                fig_dsc, ax_dsc = plt.subplots(figsize=(6.4, 4.1))
                mesh_dsc_s1 = None
                mesh_dsc_s2 = None
                dsc_s1_limits: tuple[float, float] | None = None
                dsc_s2_limits: tuple[float, float] | None = None

                if np.any(H_dsc_s1 > 0):
                    pos_s1 = H_dsc_s1[H_dsc_s1 > 0]
                    vmin_s1 = max(float(np.percentile(pos_s1, 5.0)), 1e-12)
                    vmax_s1 = float(np.percentile(pos_s1, 99.8))
                    if vmax_s1 <= vmin_s1:
                        vmax_s1 = float(np.max(pos_s1))
                    mesh_dsc_s1 = ax_dsc.pcolormesh(
                        z_edges,
                        y_edges_dsc,
                        H_dsc_s1.T,
                        cmap=_OMEGA_PHI_CMAP,
                        norm=LogNorm(vmin=vmin_s1, vmax=vmax_s1),
                        shading="auto",
                        alpha=0.64,
                    )
                    dsc_s1_limits = (vmin_s1, vmax_s1)

                if np.any(H_dsc_s2 > 0):
                    pos_s2 = H_dsc_s2[H_dsc_s2 > 0]
                    vmin_s2 = max(float(np.percentile(pos_s2, 5.0)), 1e-12)
                    vmax_s2 = float(np.percentile(pos_s2, 99.8))
                    if vmax_s2 <= vmin_s2:
                        vmax_s2 = float(np.max(pos_s2))
                    mesh_dsc_s2 = ax_dsc.pcolormesh(
                        z_edges,
                        y_edges_dsc,
                        H_dsc_s2.T,
                        cmap=_OMEGA_DM_CMAP,
                        norm=LogNorm(vmin=vmin_s2, vmax=vmax_s2),
                        shading="auto",
                        alpha=0.64,
                    )
                    dsc_s2_limits = (vmin_s2, vmax_s2)

                _style_redshift_axis(ax_dsc, z_edges)
                ax_dsc.set_yscale("symlog", linthresh=y_linthresh_dsc)
                ax_dsc.set_ylabel(r"dSC Parameters")

                _overlay_summary_on_axis(
                    ax_dsc,
                    z_grid,
                    quantity_summaries.get("s1"),
                    bestfit=bestfit_interps.get("s1"),
                    band_color=_OMEGA_PHI_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_PHI_CMAP(0.97),
                    bestfit_color=_OMEGA_PHI_CMAP(0.32),
                )
                _overlay_summary_on_axis(
                    ax_dsc,
                    z_grid,
                    quantity_summaries.get("minus_s2"),
                    bestfit=bestfit_interps.get("minus_s2"),
                    band_color=_OMEGA_DM_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_DM_CMAP(0.97),
                    bestfit_color=_OMEGA_DM_CMAP(0.28),
                )

                if args.include_legends_in_plots:
                    ax_dsc.legend(
                        handles=_get_dsc_legend_handles(),
                        loc="center left",
                        frameon=False,
                    )

                divider = make_axes_locatable(ax_dsc)
                cax_s1 = None
                dsc_s1_pad = 0.26
                if mesh_dsc_s1 is not None and dsc_s1_limits is not None:
                    cax_s1 = divider.append_axes("right", size="2.8%", pad=dsc_s1_pad)
                    cb_s1 = fig_dsc.colorbar(mesh_dsc_s1, cax=cax_s1)
                    cb_s1.set_label(r"$\mathfrak{s}_1$ Path Density")
                    cb_s1.ax.yaxis.set_label_position("left")
                    _style_colorbar(cb_s1, dsc_s1_limits[0], dsc_s1_limits[1])

                if mesh_dsc_s2 is not None and dsc_s2_limits is not None:
                    s2_pad = 0.8 - dsc_s1_pad if cax_s1 is not None else 0.20
                    cax_s2 = divider.append_axes("right", size="2.8%", pad=s2_pad)
                    cb_s2 = fig_dsc.colorbar(mesh_dsc_s2, cax=cax_s2)
                    cb_s2.set_label(r"$\mathfrak{s}_2$ Path Density")
                    cb_s2.ax.yaxis.set_label_position("left")
                    _style_colorbar(cb_s2, dsc_s2_limits[0], dsc_s2_limits[1])

                fig_dsc.tight_layout()
                _hide_overlaps_with_zero_ytick_label(
                    fig_dsc,
                    ax_dsc,
                    overlap_margin_px=args.zero_label_overlap_margin_px,
                )

                dsc_base = _output_base_path(output_dir, bundle.root, "dSC")
                _save_figure_bundle(fig_dsc, dsc_base)
                plt.close(fig_dsc)

                dsc_npz_path = output_dir / f"{dsc_base.name}.npz"
                np.savez_compressed(
                    dsc_npz_path,
                    H_dsc_s1=H_dsc_s1,
                    H_dsc_s2=H_dsc_s2,
                    x_edges=x_edges,
                    y_edges_dsc=y_edges_dsc,
                    x_grid=x_grid,
                    z_grid=z_grid,
                    quantity=np.array(["dSC"], dtype=str),
                    root=np.array([bundle.root], dtype=str),
                    dataset_label=np.array([dataset_label], dtype=str),
                    total_draws=np.array([total_draws], dtype=np.int64),
                    unique_trajectories=np.array([len(draw_records)], dtype=np.int64),
                    s1_bestfit=np.asarray(
                        bestfit_interps.get("s1", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    s1_q16=np.asarray(
                        quantity_summaries.get("s1", {}).get(
                            "q16", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    s1_q50=np.asarray(
                        quantity_summaries.get("s1", {}).get(
                            "q50", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    s1_q84=np.asarray(
                        quantity_summaries.get("s1", {}).get(
                            "q84", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    s1_valid_weight=np.asarray(
                        quantity_summaries.get("s1", {}).get(
                            "valid_weight", np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    minus_s2_bestfit=np.asarray(
                        bestfit_interps.get("minus_s2", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    minus_s2_q16=np.asarray(
                        quantity_summaries.get("minus_s2", {}).get(
                            "q16", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    minus_s2_q50=np.asarray(
                        quantity_summaries.get("minus_s2", {}).get(
                            "q50", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    minus_s2_q84=np.asarray(
                        quantity_summaries.get("minus_s2", {}).get(
                            "q84", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    minus_s2_valid_weight=np.asarray(
                        quantity_summaries.get("minus_s2", {}).get(
                            "valid_weight", np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    bestfit_background_source=np.array([bestfit_source], dtype=str),
                    bestfit_background_cache_key=np.array(
                        [bestfit_cache_key], dtype=str
                    ),
                )

                print(
                    "  [dSC-combined] Saved: "
                    f"{dsc_base.name}.png, {dsc_base.name}.pdf, {dsc_base.name}.pgf, "
                    f"{dsc_npz_path.name}"
                )

    # Stacked SWGC layout (default): lhs, rhs, residual, crossing probability.
    if swgc_stacked_only:
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)
        z_edges = np.power(10.0, x_edges) - 1.0

        lhs_positive_parts: list[np.ndarray] = []
        rhs_positive_parts: list[np.ndarray] = []
        for qty_interps, _ in trajectory_payload:
            lhs_vals = qty_interps.get("swgc_lhs")
            if lhs_vals is not None:
                lhs_valid = np.isfinite(lhs_vals) & (lhs_vals > 0.0)
                if np.any(lhs_valid):
                    lhs_positive_parts.append(
                        np.asarray(lhs_vals[lhs_valid], dtype=float)
                    )
            rhs_vals = qty_interps.get("swgc_rhs")
            if rhs_vals is not None:
                rhs_valid = np.isfinite(rhs_vals) & (rhs_vals > 0.0)
                if np.any(rhs_valid):
                    rhs_positive_parts.append(
                        np.asarray(rhs_vals[rhs_valid], dtype=float)
                    )

        y_min_lhs = (
            float(np.nanmin(np.concatenate(lhs_positive_parts)))
            if lhs_positive_parts
            else np.nan
        )
        y_max_lhs = (
            float(np.nanmax(np.concatenate(lhs_positive_parts)))
            if lhs_positive_parts
            else np.nan
        )
        y_min_rhs = (
            float(np.nanmin(np.concatenate(rhs_positive_parts)))
            if rhs_positive_parts
            else np.nan
        )
        y_max_rhs = (
            float(np.nanmax(np.concatenate(rhs_positive_parts)))
            if rhs_positive_parts
            else np.nan
        )
        y_min_res = y_ranges["swgc_residual"][0]
        y_max_res = y_ranges["swgc_residual"][1]

        if (
            np.isfinite(y_min_lhs)
            and np.isfinite(y_max_lhs)
            and y_min_lhs > 0.0
            and y_max_lhs > y_min_lhs
            and np.isfinite(y_min_rhs)
            and np.isfinite(y_max_rhs)
            and y_min_rhs > 0.0
            and y_max_rhs > y_min_rhs
            and np.isfinite(y_min_res)
            and np.isfinite(y_max_res)
            and y_max_res > y_min_res
        ):
            y_edges_lhs = _build_positive_log_y_edges(y_min_lhs, y_max_lhs, args.y_bins)
            y_edges_rhs = _build_positive_log_y_edges(y_min_rhs, y_max_rhs, args.y_bins)
            y_edges_res, y_linthresh_res = _build_symlog_y_edges(
                y_min_res,
                y_max_res,
                args.y_bins,
            )

            H_swgc_lhs = np.zeros((args.x_bins, args.y_bins), dtype=float)
            H_swgc_rhs = np.zeros((args.x_bins, args.y_bins), dtype=float)
            H_swgc_residual = np.zeros((args.x_bins, args.y_bins), dtype=float)

            for qty_interps, multiplicity in trajectory_payload:
                lhs_vals = qty_interps.get("swgc_lhs")
                if lhs_vals is not None:
                    valid_lhs = np.isfinite(lhs_vals) & (lhs_vals > 0.0)
                    n_valid_lhs = int(np.count_nonzero(valid_lhs))
                    if n_valid_lhs >= 2:
                        h_lhs, _, _ = np.histogram2d(
                            x_grid[valid_lhs],
                            lhs_vals[valid_lhs],
                            bins=(x_edges, y_edges_lhs),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_lhs, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_lhs += h_lhs

                rhs_vals = qty_interps.get("swgc_rhs")
                if rhs_vals is not None:
                    valid_rhs = np.isfinite(rhs_vals) & (rhs_vals > 0.0)
                    n_valid_rhs = int(np.count_nonzero(valid_rhs))
                    if n_valid_rhs >= 2:
                        h_rhs, _, _ = np.histogram2d(
                            x_grid[valid_rhs],
                            rhs_vals[valid_rhs],
                            bins=(x_edges, y_edges_rhs),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_rhs, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_rhs += h_rhs

                residual_vals = qty_interps.get("swgc_residual")
                if residual_vals is not None:
                    valid_res = np.isfinite(residual_vals)
                    n_valid_res = int(np.count_nonzero(valid_res))
                    if n_valid_res >= 2:
                        h_res, _, _ = np.histogram2d(
                            x_grid[valid_res],
                            residual_vals[valid_res],
                            bins=(x_edges, y_edges_res),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_res, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_residual += h_res

            swgc_crossing_profile = _compute_swgc_crossing_profile(
                trajectory_payload,
                epsilon_abs=args.swgc_crossing_epsilon_abs,
                epsilon_rel=args.swgc_crossing_epsilon_rel,
                epsilon_quantile=args.swgc_crossing_epsilon_quantile,
            )

            if (
                np.any(H_swgc_lhs > 0)
                or np.any(H_swgc_rhs > 0)
                or np.any(H_swgc_residual > 0)
            ):
                fig_swgc = plt.figure(figsize=(7.2, 8.4))
                gs_swgc = fig_swgc.add_gridspec(
                    4,
                    1,
                    height_ratios=[2.2, 2.2, 2.6, 0.9],
                    hspace=0.04,
                )
                ax_lhs = fig_swgc.add_subplot(gs_swgc[0, 0])
                ax_rhs = fig_swgc.add_subplot(gs_swgc[1, 0], sharex=ax_lhs)
                ax_res = fig_swgc.add_subplot(gs_swgc[2, 0], sharex=ax_lhs)
                ax_cross = fig_swgc.add_subplot(gs_swgc[3, 0], sharex=ax_lhs)

                mesh_lhs = None
                mesh_rhs = None
                mesh_residual = None
                lhs_limits: tuple[float, float] | None = None
                rhs_limits: tuple[float, float] | None = None
                residual_limits: tuple[float, float] | None = None

                if np.any(H_swgc_lhs > 0):
                    lhs_positive = H_swgc_lhs[H_swgc_lhs > 0]
                    vmin_lhs = max(float(np.percentile(lhs_positive, 5.0)), 1e-12)
                    vmax_lhs = float(np.percentile(lhs_positive, 99.8))
                    if vmax_lhs <= vmin_lhs:
                        vmax_lhs = float(np.max(lhs_positive))
                    lhs_limits = (vmin_lhs, vmax_lhs)
                    mesh_lhs = ax_lhs.pcolormesh(
                        z_edges,
                        y_edges_lhs,
                        H_swgc_lhs.T,
                        cmap=_OMEGA_PHI_CMAP,
                        norm=LogNorm(vmin=vmin_lhs, vmax=vmax_lhs),
                        shading="auto",
                    )

                if np.any(H_swgc_rhs > 0):
                    rhs_positive = H_swgc_rhs[H_swgc_rhs > 0]
                    vmin_rhs = max(float(np.percentile(rhs_positive, 5.0)), 1e-12)
                    vmax_rhs = float(np.percentile(rhs_positive, 99.8))
                    if vmax_rhs <= vmin_rhs:
                        vmax_rhs = float(np.max(rhs_positive))
                    rhs_limits = (vmin_rhs, vmax_rhs)
                    mesh_rhs = ax_rhs.pcolormesh(
                        z_edges,
                        y_edges_rhs,
                        H_swgc_rhs.T,
                        cmap=_OMEGA_DM_CMAP,
                        norm=LogNorm(vmin=vmin_rhs, vmax=vmax_rhs),
                        shading="auto",
                    )

                if np.any(H_swgc_residual > 0):
                    residual_positive = H_swgc_residual[H_swgc_residual > 0]
                    vmin_res = max(float(np.percentile(residual_positive, 5.0)), 1e-12)
                    vmax_res = float(np.percentile(residual_positive, 99.8))
                    if vmax_res <= vmin_res:
                        vmax_res = float(np.max(residual_positive))
                    residual_limits = (vmin_res, vmax_res)
                    mesh_residual = ax_res.pcolormesh(
                        z_edges,
                        y_edges_res,
                        H_swgc_residual.T,
                        cmap=_HEATMAP_CMAP,
                        norm=LogNorm(vmin=vmin_res, vmax=vmax_res),
                        shading="auto",
                    )

                ax_lhs.set_yscale("log")
                ax_lhs.set_ylabel(r"$(V'')^2$")
                ax_rhs.set_yscale("log")
                ax_rhs.set_ylabel(r"$2(V''')^2 - V''V''''$")
                ax_res.set_ylabel(r"$\Delta_{\rm SWGC}$")
                if y_linthresh_res is not None:
                    ax_res.set_yscale("symlog", linthresh=y_linthresh_res)
                ax_res.axhline(
                    0.0,
                    color=_HIGH_CONTRAST_PALETTE["black"],
                    linestyle=(0, (5, 2.5)),
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=2.0,
                )

                _overlay_summary_on_axis(
                    ax_lhs,
                    z_grid,
                    quantity_summaries.get("swgc_lhs"),
                    bestfit=bestfit_interps.get("swgc_lhs"),
                    band_color=_OMEGA_PHI_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_PHI_CMAP(0.22),
                    bestfit_color=_OMEGA_PHI_CMAP(0.10),
                )
                _overlay_summary_on_axis(
                    ax_rhs,
                    z_grid,
                    quantity_summaries.get("swgc_rhs"),
                    bestfit=bestfit_interps.get("swgc_rhs"),
                    band_color=_OMEGA_DM_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_DM_CMAP(0.22),
                    bestfit_color=_OMEGA_DM_CMAP(0.10),
                )
                _overlay_summary_on_axis(
                    ax_res,
                    z_grid,
                    quantity_summaries.get("swgc_residual"),
                    bestfit=bestfit_interps.get("swgc_residual"),
                    band_color=_HEATMAP_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_HIGH_CONTRAST_PALETTE["white"],
                    bestfit_color=_HIGH_CONTRAST_PALETTE["black"],
                )

                p_lt_zero = np.asarray(
                    (
                        swgc_crossing_profile["p_lt_zero"]
                        if swgc_crossing_profile is not None
                        else np.full_like(z_grid, np.nan)
                    ),
                    dtype=float,
                )
                p_near = np.asarray(
                    (
                        swgc_crossing_profile["p_near"]
                        if swgc_crossing_profile is not None
                        else np.full_like(z_grid, np.nan)
                    ),
                    dtype=float,
                )
                valid_lt_zero = np.isfinite(z_grid) & np.isfinite(p_lt_zero)
                if np.count_nonzero(valid_lt_zero) >= 2:
                    ax_cross.plot(
                        z_grid[valid_lt_zero],
                        p_lt_zero[valid_lt_zero],
                        color=_HIGH_CONTRAST_PALETTE["red"],
                        linewidth=1.25,
                        linestyle="-",
                        alpha=0.95,
                        zorder=3.0,
                    )
                valid_near = np.isfinite(z_grid) & np.isfinite(p_near)
                if np.count_nonzero(valid_near) >= 2:
                    ax_cross.fill_between(
                        z_grid[valid_near],
                        0.0,
                        p_near[valid_near],
                        color=_HIGH_CONTRAST_PALETTE["yellow"],
                        alpha=0.20,
                        linewidth=0.0,
                        zorder=2.0,
                    )
                ax_cross.set_ylim(0.0, 1.0)
                ax_cross.set_ylabel(r"$P_{<0}$")
                ax_cross.tick_params(axis="y", which="major", labelsize=10)

                for axis in (ax_lhs, ax_rhs, ax_res):
                    axis.set_xlabel("")
                    axis.tick_params(axis="x", which="both", labelbottom=False)
                _style_redshift_axis(ax_cross, z_edges)

                # Reserve a right-side strip for external colorbars so all four
                # shared-x panels keep identical widths.
                fig_swgc.subplots_adjust(left=0.10, right=0.86, top=0.98, bottom=0.08)

                def _add_external_colorbar(
                    mesh: QuadMesh,
                    axis: Axes,
                    label: str,
                    vmin: float,
                    vmax: float,
                ) -> None:
                    pos = axis.get_position()
                    cax_rect: tuple[float, float, float, float] = (
                        0.875,
                        float(pos.y0),
                        0.018,
                        float(pos.height),
                    )
                    cax = fig_swgc.add_axes(cax_rect)
                    cb = fig_swgc.colorbar(mesh, cax=cax)
                    cb.set_label(label)
                    _style_colorbar(cb, vmin, vmax)

                if mesh_lhs is not None and lhs_limits is not None:
                    _add_external_colorbar(
                        mesh_lhs,
                        ax_lhs,
                        r"$\left(V_{\phi\phi}\right)^2$ Path Density",
                        lhs_limits[0],
                        lhs_limits[1],
                    )
                if mesh_rhs is not None and rhs_limits is not None:
                    _add_external_colorbar(
                        mesh_rhs,
                        ax_rhs,
                        r"$2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$ Path Density",
                        rhs_limits[0],
                        rhs_limits[1],
                    )
                if mesh_residual is not None and residual_limits is not None:
                    _add_external_colorbar(
                        mesh_residual,
                        ax_res,
                        r"$\Delta_{\rm SWGC}$ Path Density",
                        residual_limits[0],
                        residual_limits[1],
                    )

                if args.include_legends_in_plots:
                    ax_lhs.legend(
                        handles=_get_swgc_legend_handles(),
                        loc="upper left",
                        frameon=False,
                        fontsize=10,
                    )

                # Tiny regression guard: stacked SWGC axes must keep identical x-span
                # and left/right bounds to preserve a visually shared x-axis.
                _swgc_axes = (ax_lhs, ax_rhs, ax_res, ax_cross)
                _ref_xlim = ax_lhs.get_xlim()
                _ref_pos = ax_lhs.get_position()
                _xlim_ok = all(
                    np.isclose(axis.get_xlim()[0], _ref_xlim[0], atol=1e-10)
                    and np.isclose(axis.get_xlim()[1], _ref_xlim[1], atol=1e-10)
                    for axis in _swgc_axes
                )
                _bounds_ok = all(
                    np.isclose(axis.get_position().x0, _ref_pos.x0, atol=1e-10)
                    and np.isclose(axis.get_position().x1, _ref_pos.x1, atol=1e-10)
                    for axis in _swgc_axes
                )
                print(
                    "  [DEBUG][SWGC-stacked] axis alignment: "
                    f"xlim_ok={_xlim_ok}, bounds_ok={_bounds_ok}"
                )
                assert _xlim_ok and _bounds_ok, (
                    "SWGC stacked subplot alignment regression detected "
                    "(shared x-axis span or horizontal bounds differ)."
                )

                swgc_base = _output_base_path(output_dir, bundle.root, "swgc")
                _save_figure_bundle(fig_swgc, swgc_base)
                plt.close(fig_swgc)

                swgc_npz_path = output_dir / f"{swgc_base.name}.npz"
                np.savez_compressed(
                    swgc_npz_path,
                    H_swgc_lhs=H_swgc_lhs,
                    H_swgc_rhs=H_swgc_rhs,
                    H_swgc_residual=H_swgc_residual,
                    x_edges=x_edges,
                    y_edges_swgc_lhs=y_edges_lhs,
                    y_edges_swgc_rhs=y_edges_rhs,
                    y_edges_swgc_residual=y_edges_res,
                    x_grid=x_grid,
                    z_grid=z_grid,
                    quantity=np.array(["swgc"], dtype=str),
                    root=np.array([bundle.root], dtype=str),
                    dataset_label=np.array([dataset_label], dtype=str),
                    total_draws=np.array([total_draws], dtype=np.int64),
                    unique_trajectories=np.array([len(draw_records)], dtype=np.int64),
                    swgc_layout=np.array(["stacked"], dtype=str),
                    swgc_lhs_bestfit=np.asarray(
                        bestfit_interps.get("swgc_lhs", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    swgc_rhs_bestfit=np.asarray(
                        bestfit_interps.get("swgc_rhs", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    swgc_residual_bestfit=np.asarray(
                        bestfit_interps.get(
                            "swgc_residual", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_crossing_probability=np.asarray(
                        (
                            swgc_crossing_profile["p_lt_zero"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_probability_lt_zero=np.asarray(
                        (
                            swgc_crossing_profile["p_lt_zero"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_probability_positive=np.asarray(
                        (
                            swgc_crossing_profile["p_pos"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_probability_negative=np.asarray(
                        (
                            swgc_crossing_profile["p_neg"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_probability_near_zero=np.asarray(
                        (
                            swgc_crossing_profile["p_near"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_crossing_epsilon=np.asarray(
                        (
                            swgc_crossing_profile["epsilon"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_cancellation_kappa_median=np.asarray(
                        (
                            swgc_crossing_profile["kappa_median"]
                            if swgc_crossing_profile is not None
                            else np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_crossing_valid_weight=np.asarray(
                        (
                            swgc_crossing_profile["valid_weight"]
                            if swgc_crossing_profile is not None
                            else np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    sample_plan_fingerprint=np.array([sampling_fp], dtype=str),
                    bestfit_background_source=np.array([bestfit_source], dtype=str),
                    bestfit_background_cache_key=np.array(
                        [bestfit_cache_key], dtype=str
                    ),
                )

                print(
                    "  [SWGC-stacked] Saved: "
                    f"{swgc_base.name}.png, {swgc_base.name}.pdf, {swgc_base.name}.pgf, "
                    f"{swgc_npz_path.name}"
                )

    # Combined SWGC history figure: lhs/rhs terms on top, residual on bottom.
    if swgc_combined_only:
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)
        z_edges = np.power(10.0, x_edges) - 1.0

        term_positive_parts: list[np.ndarray] = []
        for qty_interps, _ in trajectory_payload:
            for key in ("swgc_lhs", "swgc_rhs"):
                vals = qty_interps.get(key)
                if vals is None:
                    continue
                valid = np.isfinite(vals) & (vals > 0.0)
                if np.any(valid):
                    term_positive_parts.append(np.asarray(vals[valid], dtype=float))

        if term_positive_parts:
            term_positive_all = np.concatenate(term_positive_parts)
            y_min_terms = float(np.nanmin(term_positive_all))
            y_max_terms = float(np.nanmax(term_positive_all))
        else:
            y_min_terms = np.nan
            y_max_terms = np.nan

        y_min_res = y_ranges["swgc_residual"][0]
        y_max_res = y_ranges["swgc_residual"][1]

        if (
            np.isfinite(y_min_terms)
            and np.isfinite(y_max_terms)
            and y_min_terms > 0.0
            and y_max_terms > y_min_terms
            and np.isfinite(y_min_res)
            and np.isfinite(y_max_res)
            and y_max_res > y_min_res
        ):
            y_edges_terms = _build_positive_log_y_edges(
                y_min_terms, y_max_terms, args.y_bins
            )
            # Use symlog edges for residuals so positive/negative values are both
            # represented on a log-like scale.
            y_edges_res, y_linthresh_res = _build_symlog_y_edges(
                y_min_res,
                y_max_res,
                args.y_bins,
            )

            H_swgc_lhs = np.zeros((args.x_bins, args.y_bins), dtype=float)
            H_swgc_rhs = np.zeros((args.x_bins, args.y_bins), dtype=float)
            H_swgc_residual = np.zeros((args.x_bins, args.y_bins), dtype=float)

            for qty_interps, multiplicity in trajectory_payload:
                lhs_vals = qty_interps.get("swgc_lhs")
                if lhs_vals is not None:
                    valid_lhs = np.isfinite(lhs_vals) & (lhs_vals > 0.0)
                    n_valid_lhs = int(np.count_nonzero(valid_lhs))
                    if n_valid_lhs >= 2:
                        h_lhs, _, _ = np.histogram2d(
                            x_grid[valid_lhs],
                            lhs_vals[valid_lhs],
                            bins=(x_edges, y_edges_terms),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_lhs, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_lhs += h_lhs

                rhs_vals = qty_interps.get("swgc_rhs")
                if rhs_vals is not None:
                    valid_rhs = np.isfinite(rhs_vals) & (rhs_vals > 0.0)
                    n_valid_rhs = int(np.count_nonzero(valid_rhs))
                    if n_valid_rhs >= 2:
                        h_rhs, _, _ = np.histogram2d(
                            x_grid[valid_rhs],
                            rhs_vals[valid_rhs],
                            bins=(x_edges, y_edges_terms),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_rhs, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_rhs += h_rhs

                residual_vals = qty_interps.get("swgc_residual")
                if residual_vals is not None:
                    valid_res = np.isfinite(residual_vals)
                    n_valid_res = int(np.count_nonzero(valid_res))
                    if n_valid_res >= 2:
                        h_res, _, _ = np.histogram2d(
                            x_grid[valid_res],
                            residual_vals[valid_res],
                            bins=(x_edges, y_edges_res),  # type: ignore[arg-type]
                            weights=np.full(
                                n_valid_res, float(multiplicity), dtype=float
                            ),
                        )
                        H_swgc_residual += h_res

            if (
                np.any(H_swgc_lhs > 0)
                or np.any(H_swgc_rhs > 0)
                or np.any(H_swgc_residual > 0)
            ):
                fig_swgc = plt.figure(figsize=(7.2, 6.0))
                gs_swgc = fig_swgc.add_gridspec(
                    2,
                    1,
                    height_ratios=[3.5, 1.35],
                    hspace=0.04,
                )
                ax_top = fig_swgc.add_subplot(gs_swgc[0, 0])
                ax_bottom = fig_swgc.add_subplot(gs_swgc[1, 0], sharex=ax_top)
                cax_lhs: Axes | None = None
                cax_rhs: Axes | None = None

                mesh_lhs = None
                mesh_rhs = None
                mesh_residual = None
                top_limits: tuple[float, float] | None = None
                residual_limits: tuple[float, float] | None = None

                top_positive = []
                if np.any(H_swgc_lhs > 0):
                    top_positive.append(H_swgc_lhs[H_swgc_lhs > 0])
                if np.any(H_swgc_rhs > 0):
                    top_positive.append(H_swgc_rhs[H_swgc_rhs > 0])

                if top_positive:
                    top_positive_flat = np.concatenate(top_positive)
                    vmin_top = max(float(np.percentile(top_positive_flat, 5.0)), 1e-12)
                    vmax_top = float(np.percentile(top_positive_flat, 99.8))
                    if vmax_top <= vmin_top:
                        vmax_top = float(np.max(top_positive_flat))
                    top_limits = (vmin_top, vmax_top)
                    if np.any(H_swgc_lhs > 0):
                        mesh_lhs = ax_top.pcolormesh(
                            z_edges,
                            y_edges_terms,
                            H_swgc_lhs.T,
                            cmap=_OMEGA_PHI_CMAP,
                            norm=LogNorm(vmin=vmin_top, vmax=vmax_top),
                            shading="auto",
                            alpha=0.62,
                        )
                    if np.any(H_swgc_rhs > 0):
                        mesh_rhs = ax_top.pcolormesh(
                            z_edges,
                            y_edges_terms,
                            H_swgc_rhs.T,
                            cmap=_OMEGA_DM_CMAP,
                            norm=LogNorm(vmin=vmin_top, vmax=vmax_top),
                            shading="auto",
                            alpha=0.62,
                        )

                if np.any(H_swgc_residual > 0):
                    residual_positive = H_swgc_residual[H_swgc_residual > 0]
                    vmin_res = max(float(np.percentile(residual_positive, 5.0)), 1e-12)
                    vmax_res = float(np.percentile(residual_positive, 99.8))
                    if vmax_res <= vmin_res:
                        vmax_res = float(np.max(residual_positive))
                    residual_limits = (vmin_res, vmax_res)
                    mesh_residual = ax_bottom.pcolormesh(
                        z_edges,
                        y_edges_res,
                        H_swgc_residual.T,
                        cmap=_HEATMAP_CMAP,
                        norm=LogNorm(vmin=vmin_res, vmax=vmax_res),
                        shading="auto",
                    )

                _style_redshift_axis(ax_top, z_edges)
                _style_redshift_axis(ax_bottom, z_edges)
                ax_top.set_yscale("log")
                ax_top.set_ylabel("SWGC terms")
                ax_bottom.set_ylabel(r"$\Delta_{\rm SWGC}$")
                if y_linthresh_res is not None:
                    ax_bottom.set_yscale("symlog", linthresh=y_linthresh_res)

                # Avoid scientific offset text (e.g. "2.4e-10") that can look
                # like a detached annotation and clutter the shared-x panel layout.
                bottom_y_formatter = ax_bottom.yaxis.get_major_formatter()
                if isinstance(bottom_y_formatter, ScalarFormatter):
                    bottom_y_formatter.set_useOffset(False)
                ax_bottom.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda val, _: (
                            "0"
                            if np.isclose(val, 0.0, atol=1e-18, rtol=0.0)
                            else (
                                (
                                    rf"$-10^{{{int(np.round(np.log10(abs(val))))}}}$"
                                    if val < 0.0
                                    else rf"$10^{{{int(np.round(np.log10(abs(val))))}}}$"
                                )
                                if np.isclose(
                                    abs(val),
                                    10.0 ** int(np.round(np.log10(abs(val)))),
                                    rtol=1e-8,
                                    atol=0.0,
                                )
                                else f"{val:.1e}"
                            )
                        )
                    )
                )
                ax_bottom.yaxis.get_offset_text().set_visible(False)

                _overlay_summary_on_axis(
                    ax_top,
                    z_grid,
                    quantity_summaries.get("swgc_lhs"),
                    bestfit=bestfit_interps.get("swgc_lhs"),
                    band_color=_OMEGA_PHI_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_PHI_CMAP(0.22),
                    bestfit_color=_OMEGA_PHI_CMAP(0.10),
                )
                _overlay_summary_on_axis(
                    ax_top,
                    z_grid,
                    quantity_summaries.get("swgc_rhs"),
                    bestfit=bestfit_interps.get("swgc_rhs"),
                    band_color=_OMEGA_DM_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_OMEGA_DM_CMAP(0.22),
                    bestfit_color=_OMEGA_DM_CMAP(0.10),
                )
                _overlay_summary_on_axis(
                    ax_bottom,
                    z_grid,
                    quantity_summaries.get("swgc_residual"),
                    bestfit=bestfit_interps.get("swgc_residual"),
                    band_color=_HEATMAP_CMAP(0.58),
                    band_alpha=0.22,
                    median_color=_HIGH_CONTRAST_PALETTE["white"],
                    bestfit_color=_HIGH_CONTRAST_PALETTE["black"],
                )
                ax_bottom.axhline(
                    0.0,
                    color=_HIGH_CONTRAST_PALETTE["black"],
                    linestyle=(0, (5, 2.5)),
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=2.0,
                )

                shared_swgc_legend_handles: list[Any] = [
                    Patch(
                        facecolor=_OMEGA_PHI_CMAP(0.78),
                        edgecolor="none",
                        alpha=0.64,
                        label=r"$\left(V_{\phi\phi}\right)^2$",
                    ),
                    Patch(
                        facecolor=_OMEGA_DM_CMAP(0.78),
                        edgecolor="none",
                        alpha=0.64,
                        label=r"$2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$",
                    ),
                    Patch(
                        facecolor=_OMEGA_PHI_CMAP(0.58),
                        edgecolor="none",
                        alpha=0.22,
                        label=r"68\% confidence band $\left(V_{\phi\phi}\right)^2$",
                    ),
                    Patch(
                        facecolor=_OMEGA_DM_CMAP(0.58),
                        edgecolor="none",
                        alpha=0.22,
                        label=r"68\% confidence band $2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$",
                    ),
                    Patch(
                        facecolor=_HEATMAP_CMAP(0.58),
                        edgecolor="none",
                        linewidth=0.8,
                        alpha=0.22,
                        label=r"68\% confidence band $\Delta_{\rm SWGC}$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=_HIGH_CONTRAST_PALETTE["black"],
                        linewidth=1.3,
                        label=(
                            r"$\Delta_{\rm SWGC}$"
                            r"$=2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}"
                            r"-\left(V_{\phi\phi}\right)^2$"
                        ),
                    ),
                    *_summary_line_legend_handles(),
                ]
                if args.include_legends_in_plots:
                    ax_top.legend(
                        handles=shared_swgc_legend_handles,
                        loc="upper left",
                        ncol=1,
                        frameon=False,
                        fontsize=10,
                        handlelength=2.2,
                        borderaxespad=0.35,
                    )

                ax_top.set_xlabel("")
                ax_top.tick_params(axis="x", which="both", labelbottom=False)

                divider_swgc = make_axes_locatable(ax_top)
                # Reserve room for left-side colorbar titles so each reads
                # "title, then colorbar" from left to right.
                swgc_lhs_pad = 0.34

                if top_limits is not None and mesh_lhs is not None:
                    cax_lhs = divider_swgc.append_axes(
                        "right", size="2.8%", pad=swgc_lhs_pad
                    )
                    cb_lhs = fig_swgc.colorbar(mesh_lhs, cax=cax_lhs)
                    cb_lhs.set_label(r"$\left(V_{\phi\phi}\right)^2$")
                    cb_lhs.ax.yaxis.set_label_position("left")
                    cb_lhs.ax.yaxis.labelpad = 4
                    _style_colorbar(cb_lhs, top_limits[0], top_limits[1])
                if top_limits is not None and mesh_rhs is not None:
                    # Keep enough clearance so oslo tick labels/title do not collide
                    # with the lajolla colorbar/ticks.
                    rhs_pad = 0.5 if cax_lhs is not None else 0.22
                    cax_rhs = divider_swgc.append_axes(
                        "right", size="2.8%", pad=rhs_pad
                    )
                    cb_rhs = fig_swgc.colorbar(mesh_rhs, cax=cax_rhs)
                    cb_rhs.set_label(
                        r"$2\left(V_{\phi\phi\phi}\right)^2 - V_{\phi\phi}V_{\phi\phi\phi\phi}$"
                    )
                    cb_rhs.ax.yaxis.set_label_position("left")
                    cb_rhs.ax.yaxis.labelpad = 4
                    _style_colorbar(cb_rhs, top_limits[0], top_limits[1])

                fig_swgc.subplots_adjust(left=0.09, right=0.96, top=0.97, bottom=0.10)

                # Draw first so the symlog locator has settled, then replace ticks
                # with a well-distributed set: one near the bottom of the range,
                # one near the top, and 0 as a third tick if neither chosen tick is
                # zero and the range straddles zero.
                fig_swgc.canvas.draw()

                # The top axis is resized by append_axes; force the lower axis to use
                # the same x-start and width so both SWGC panels stay aligned.
                top_pos = ax_top.get_position()
                bottom_pos = ax_bottom.get_position()
                ax_bottom.set_position(
                    (top_pos.x0, bottom_pos.y0, top_pos.width, bottom_pos.height)
                )

                fig_swgc.canvas.draw()

                if residual_limits is not None and mesh_residual is not None:
                    bottom_pos = ax_bottom.get_position()
                    # Leave room so the left-side colorbar title sits fully
                    # between the subplot and the bar (title -> colorbar order).
                    residual_pad = 0.047
                    residual_width = 0.028 * bottom_pos.width
                    cax_residual = fig_swgc.add_axes(
                        (
                            bottom_pos.x1 + residual_pad,
                            bottom_pos.y0,
                            residual_width,
                            bottom_pos.height,
                        )
                    )
                    cb_bottom = fig_swgc.colorbar(mesh_residual, cax=cax_residual)
                    cb_bottom.set_label(r"$\Delta_{\rm SWGC}$ path density", labelpad=4)
                    cb_bottom.ax.yaxis.set_label_position("left")
                    _style_colorbar(cb_bottom, residual_limits[0], residual_limits[1])

                bottom_offset = ax_bottom.yaxis.get_offset_text()
                bottom_offset.set_text("")
                bottom_offset.set_visible(False)

                _res_lo, _res_hi = sorted(ax_bottom.get_ylim())
                if np.isfinite(_res_lo) and np.isfinite(_res_hi) and _res_hi > _res_lo:
                    _new_res_ticks: list[float] = []
                    if _res_lo < 0.0 < _res_hi:
                        # Use rounded signed powers of ten, matching publication-style
                        # residual ticks: one negative, zero, one positive.
                        _neg_tick = -(10.0 ** np.ceil(np.log10(abs(_res_lo))))
                        _pos_tick = 10.0 ** np.floor(np.log10(_res_hi))
                        _new_res_ticks = [float(_neg_tick), 0.0, float(_pos_tick)]
                        # Give the rounded ticks visual breathing room in symlog,
                        # preventing -10^n and 0 label collisions.
                        ax_bottom.set_ylim(
                            min(_res_lo, 5.0 * float(_neg_tick)),
                            max(_res_hi, 3.0 * float(_pos_tick)),
                        )
                    else:
                        _curr_res_ticks = np.asarray(
                            [
                                t
                                for t in ax_bottom.get_yticks()
                                if np.isfinite(t) and _res_lo <= t <= _res_hi
                            ],
                            dtype=float,
                        )
                        if _curr_res_ticks.size >= 2:
                            _sorted_rt = np.sort(_curr_res_ticks)
                            _new_res_ticks = [
                                float(_sorted_rt[0]),
                                float(_sorted_rt[-1]),
                            ]
                        else:
                            _ensure_min_labeled_yticks(ax_bottom, min_labels=2)

                    if _new_res_ticks:
                        ax_bottom.yaxis.set_major_locator(
                            FixedLocator(sorted(set(_new_res_ticks)))
                        )

                swgc_base = _output_base_path(output_dir, bundle.root, "swgc")
                _save_figure_bundle(fig_swgc, swgc_base)
                plt.close(fig_swgc)

                swgc_npz_path = output_dir / f"{swgc_base.name}.npz"
                np.savez_compressed(
                    swgc_npz_path,
                    H_swgc_lhs=H_swgc_lhs,
                    H_swgc_rhs=H_swgc_rhs,
                    H_swgc_residual=H_swgc_residual,
                    x_edges=x_edges,
                    y_edges_swgc_terms=y_edges_terms,
                    y_edges_swgc_residual=y_edges_res,
                    x_grid=x_grid,
                    z_grid=z_grid,
                    quantity=np.array(["swgc"], dtype=str),
                    root=np.array([bundle.root], dtype=str),
                    dataset_label=np.array([dataset_label], dtype=str),
                    total_draws=np.array([total_draws], dtype=np.int64),
                    unique_trajectories=np.array([len(draw_records)], dtype=np.int64),
                    swgc_lhs_bestfit=np.asarray(
                        bestfit_interps.get("swgc_lhs", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    swgc_lhs_q16=np.asarray(
                        quantity_summaries.get("swgc_lhs", {}).get(
                            "q16", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_lhs_q50=np.asarray(
                        quantity_summaries.get("swgc_lhs", {}).get(
                            "q50", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_lhs_q84=np.asarray(
                        quantity_summaries.get("swgc_lhs", {}).get(
                            "q84", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_lhs_valid_weight=np.asarray(
                        quantity_summaries.get("swgc_lhs", {}).get(
                            "valid_weight", np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    swgc_rhs_bestfit=np.asarray(
                        bestfit_interps.get("swgc_rhs", np.full_like(x_grid, np.nan)),
                        dtype=float,
                    ),
                    swgc_rhs_q16=np.asarray(
                        quantity_summaries.get("swgc_rhs", {}).get(
                            "q16", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_rhs_q50=np.asarray(
                        quantity_summaries.get("swgc_rhs", {}).get(
                            "q50", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_rhs_q84=np.asarray(
                        quantity_summaries.get("swgc_rhs", {}).get(
                            "q84", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_rhs_valid_weight=np.asarray(
                        quantity_summaries.get("swgc_rhs", {}).get(
                            "valid_weight", np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    swgc_residual_bestfit=np.asarray(
                        bestfit_interps.get(
                            "swgc_residual", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_residual_q16=np.asarray(
                        quantity_summaries.get("swgc_residual", {}).get(
                            "q16", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_residual_q50=np.asarray(
                        quantity_summaries.get("swgc_residual", {}).get(
                            "q50", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_residual_q84=np.asarray(
                        quantity_summaries.get("swgc_residual", {}).get(
                            "q84", np.full_like(x_grid, np.nan)
                        ),
                        dtype=float,
                    ),
                    swgc_residual_valid_weight=np.asarray(
                        quantity_summaries.get("swgc_residual", {}).get(
                            "valid_weight", np.zeros_like(x_grid, dtype=float)
                        ),
                        dtype=float,
                    ),
                    sample_plan_fingerprint=np.array([sampling_fp], dtype=str),
                    bestfit_background_source=np.array([bestfit_source], dtype=str),
                    bestfit_background_cache_key=np.array(
                        [bestfit_cache_key], dtype=str
                    ),
                )

                print(
                    "  [SWGC-combined] Saved: "
                    f"{swgc_base.name}.png, {swgc_base.name}.pdf, {swgc_base.name}.pgf, "
                    f"{swgc_npz_path.name}"
                )

    print(
        f"Background cache usage: hit={cache_hits}, miss={cache_misses}, "
        f"cache_dir={cache_dir}"
    )


def _process_single_root_wrapper(
    root_index_tuple: tuple[int, str],
    args: argparse.Namespace,
    quantities: list[str],
    seed: int,
    mode_tuning: dict[str, Any],
    hdm_root: Path,
    cache_dir: Path,
    output_dir: Path,
    failure_audit_dir: Path,
) -> None:
    """Process a single dataset root (for root-level parallelization).

    Wraps process_dataset with per-root RNG seeding for reproducibility.
    """
    idx, root = root_index_tuple
    # Adjust seed per root for independent randomization across processes.
    root_seed = seed + idx
    rng = np.random.default_rng(root_seed)
    bundle = discover_chain_bundle(root, hdm_root)
    process_dataset(
        bundle,
        args,
        quantities,
        rng,
        mode_tuning,
        cache_dir,
        output_dir,
        failure_audit_dir,
    )


def main() -> None:
    args = parse_args()
    if args.hpd_mass <= 0.0 or args.hpd_mass >= 1.0:
        raise ValueError("--hpd-mass must be in (0, 1)")
    if args.mode_detect_bins < 4:
        raise ValueError("--mode-detect-bins must be >= 4")
    if args.mode_min_mass_frac < 0.0 or args.mode_min_mass_frac >= 1.0:
        raise ValueError("--mode-min-mass-frac must be in [0, 1)")
    if args.mode_floor_abs < 1:
        raise ValueError("--mode-floor-abs must be >= 1")
    if args.mode_floor_frac < 0.0 or args.mode_floor_frac >= 1.0:
        raise ValueError("--mode-floor-frac must be in [0, 1)")
    if args.mode_floor_cap_frac <= 0.0 or args.mode_floor_cap_frac > 1.0:
        raise ValueError("--mode-floor-cap-frac must be in (0, 1]")
    if (
        args.phi_crossing_binary_threshold < 0.0
        or args.phi_crossing_binary_threshold > 1.0
    ):
        raise ValueError("--phi-crossing-binary-threshold must be in [0, 1]")
    if args.phi_crossing_epsilon_abs < 0.0:
        raise ValueError("--phi-crossing-epsilon-abs must be >= 0")
    if args.phi_crossing_epsilon_linthresh_frac < 0.0:
        raise ValueError("--phi-crossing-epsilon-linthresh-frac must be >= 0")
    if (
        args.phi_crossing_epsilon_quantile < 0.0
        or args.phi_crossing_epsilon_quantile > 0.5
    ):
        raise ValueError("--phi-crossing-epsilon-quantile must be in [0, 0.5]")
    if args.swgc_crossing_epsilon_abs < 0.0:
        raise ValueError("--swgc-crossing-epsilon-abs must be >= 0")
    if args.swgc_crossing_epsilon_rel < 0.0:
        raise ValueError("--swgc-crossing-epsilon-rel must be >= 0")
    if (
        args.swgc_crossing_epsilon_quantile < 0.0
        or args.swgc_crossing_epsilon_quantile > 0.5
    ):
        raise ValueError("--swgc-crossing-epsilon-quantile must be in [0, 0.5]")

    mode_tuning = _resolve_mode_sampling_tuning(args)

    # Resolve quantities: --preset > --quantities > --quantity > publication default (all)
    if args.preset is not None:
        quantities: list[str] = list(_PRESET_QUANTITIES[args.preset])
    elif args.quantities is not None:
        quantities = list(args.quantities)
    elif args.quantity is not None:
        quantities = [str(args.quantity)]
    else:
        quantities = list(_PRESET_QUANTITIES["all"])

    cache_dir = (_repo_root() / args.cache_dir).resolve()
    output_dir = (_repo_root() / args.output_dir).resolve()
    failure_audit_dir = (_repo_root() / args.failure_audit_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdm_root = _hdm_root()

    print("Posterior background heatmap pipeline")
    print(f"  HDM root:   {hdm_root}")
    print(f"  Cache dir:  {cache_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Audit dir:  {failure_audit_dir}")
    print(f"  Quantities: {quantities}")
    print(f"  Include legends in plots: {args.include_legends_in_plots}")
    print(f"  Phi y-scale mode: {args.phi_y_scale}")
    print(
        "  Phi crossing diagnostic: "
        f"overlay={args.phi_crossing_overlay}, "
        f"eps_mode={args.phi_crossing_epsilon_mode}, "
        f"eps_abs={args.phi_crossing_epsilon_abs:.3e}, "
        f"eps_linthresh_frac={args.phi_crossing_epsilon_linthresh_frac:.4f}, "
        f"eps_quantile={args.phi_crossing_epsilon_quantile:.4f}, "
        f"binary_threshold={args.phi_crossing_binary_threshold:.3f}"
    )
    print(f"  Max draws:  {args.max_samples}")
    print(f"  Sampling-plan cache mode: {args.sample_plan_cache_mode}")
    print("  Mode-aware sampling defaults:")
    print(f"    Schema version: {_MODE_SAMPLING_SCHEMA_VERSION}")
    print(f"    Detection bins/axis: {mode_tuning['mode_detect_bins']}")
    print(
        "    Guaranteed floor per eligible mode: "
        f"max({mode_tuning['mode_floor_abs']}, "
        f"round({mode_tuning['mode_floor_frac']:.6f} * max_samples)) "
        f"= {mode_tuning['mode_floor_per_mode']}"
    )
    print(
        "    Eligible mode minimum posterior mass fraction: "
        f"{mode_tuning['mode_min_mass_frac']:.6f}"
    )
    print(
        "    Total floor allocation cap: "
        f"{mode_tuning['mode_floor_cap_frac']:.3f} * max_samples "
        f"= {mode_tuning['mode_floor_total_cap']}"
    )
    print(
        "    Max floor-supported modes at current budget: "
        f"{mode_tuning['mode_floor_max_modes']}"
    )
    print(
        "    Tuning snapshot JSON: "
        f"{json.dumps(mode_tuning, sort_keys=True, separators=(',', ':'))}"
    )
    print(f"  Trajectory parallelization: --num-threads={args.num_threads}")
    print(f"  Root-level parallelization: --num-roots={args.num_roots}")

    if not args.include_legends_in_plots:
        _save_requested_legends_once(quantities, output_dir)

    # Resolve bundles up front so missing roots are reported and skipped
    # rather than aborting all remaining datasets.
    resolved_bundles: list[ChainBundle] = []
    skipped_roots: list[str] = []
    for root in args.roots:
        try:
            resolved_bundles.append(discover_chain_bundle(root, hdm_root))
        except FileNotFoundError as exc:
            skipped_roots.append(root)
            print(f"[WARNING] Skipping root '{root}': {exc}")

    if not resolved_bundles:
        roots_preview = ", ".join(args.roots)
        raise FileNotFoundError(
            "No valid chain roots found. Checked roots: "
            f"{roots_preview}. Use --roots to pass available roots."
        )

    if skipped_roots:
        print(
            "Proceeding with "
            f"{len(resolved_bundles)} valid root(s); skipped {len(skipped_roots)} missing root(s)."
        )

    # Determine process pool size: 0 = use all cores, >0 = explicit limit.
    num_roots = args.num_roots if args.num_roots > 0 else None

    # Root-level parallelization: process multiple roots concurrently.
    if num_roots == 1 or len(resolved_bundles) == 1:
        # Serial root processing.
        rng = np.random.default_rng(args.seed)
        for bundle in resolved_bundles:
            process_dataset(
                bundle,
                args,
                quantities,
                rng,
                mode_tuning,
                cache_dir,
                output_dir,
                failure_audit_dir,
            )
    else:
        # Parallel root processing with ProcessPoolExecutor.
        with ProcessPoolExecutor(max_workers=num_roots) as executor:
            futures = [
                executor.submit(
                    _process_single_root_wrapper,
                    (i, bundle.root),
                    args,
                    quantities,
                    args.seed,
                    mode_tuning,
                    hdm_root,
                    cache_dir,
                    output_dir,
                    failure_audit_dir,
                )
                for i, bundle in enumerate(resolved_bundles)
            ]
            # Wait for all to complete.
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
