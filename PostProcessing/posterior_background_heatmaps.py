#!/usr/bin/env python3
"""
Posterior background heatmaps from MCMC chains.

This script builds posterior-density heatmaps for background quantities (phi by default)
using a dedicated pipeline:

1) HPD-like selection in parameter space (histogram-density surrogate)
2) Weighted posterior resampling from the selected HPD region
3) Background-only CLASS runs (no C_l / P(k))
4) Persistent background cache keyed by CLASS parameter hash
5) Streaming 2D histogram accumulation and one figure per dataset

Designed for large chains where running CLASS for every accepted sample is infeasible.

Example:
  python3 PostProcessing/posterior_background_heatmaps.py \
      --roots hyperbolic_PP_D_InitCond_MCMC hyperbolic_PP_S_D_InitCond_MCMC \
      --max-samples 2000 --hpd-mass 0.68
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from BestFitPlot import (
    extract_class_parameters,
    load_background_dataset,
    load_bestfit_file,
    load_input_yaml,
)

# Keep plotting dependencies lightweight and robust on clusters/headless runs.
mpl.use("Agg")


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
    "swgc_lhs",       # (V'')^2
    "swgc_rhs",       # 2(V''')^2 - V''*V''''
    "swgc_residual",  # rhs - lhs  (positive = SWGC satisfied)
]

# Named presets for --preset; each expands to a list of quantities plotted in one pass.
_PRESET_QUANTITIES: dict[str, list[str]] = {
    "phi":       ["phi_scf"],
    "eos":       ["w"],
    "omega":     ["Omega_cdm", "Omega_scf"],
    "swampland": ["s1", "minus_s2", "swampland_expr"],
    "swgc":      ["swgc_lhs", "swgc_rhs", "swgc_residual"],
    "all":       list(_ALL_QUANTITIES),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-dataset posterior heatmaps for background quantities "
            "using HPD-like selection + weighted resampling + persistent CLASS cache."
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
        "--quantity",
        choices=_ALL_QUANTITIES,
        default="phi_scf",
        help="Single background quantity (default: phi_scf). Overridden by --quantities or --preset.",
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
        chain_files = sorted(root_abs.parent.glob(f"{root_abs.name}.[0-9]*.txt"))
        if chain_files:
            bestfit_candidate = root_abs.with_suffix(".bestfit")
            input_candidate = root_abs.with_suffix(".input.yaml")
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

    valid = np.isfinite(weights) & np.all(np.isfinite(hpd_values), axis=1)
    weights = weights[valid]
    minuslogpost = minuslogpost[valid]
    hpd_values = hpd_values[valid]

    return SelectionData(
        weights=weights,
        minuslogpost=minuslogpost,
        hpd_values=hpd_values,
        file_counts_postburn=counts_postburn,
    )


def _hpd_like_mask(
    values: np.ndarray,
    weights: np.ndarray,
    bins: int,
    mass: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """HPD-like region from a weighted 3D histogram density surrogate.

    We estimate local posterior density with equal-volume histogram bins.
    Bins are ranked by weighted occupancy; highest-density bins are included
    until cumulative posterior mass reaches `mass`.
    """
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("HPD-like mask requires values shape (N, 3)")

    mass = float(max(0.01, min(0.999, mass)))
    bins = int(max(8, bins))

    # Robust ranges avoid a few outliers wasting histogram resolution.
    q_lo = np.nanpercentile(values, 0.5, axis=0)
    q_hi = np.nanpercentile(values, 99.5, axis=0)
    for i in range(3):
        if not np.isfinite(q_lo[i]) or not np.isfinite(q_hi[i]) or q_lo[i] == q_hi[i]:
            q_lo[i] = float(np.nanmin(values[:, i]))
            q_hi[i] = float(np.nanmax(values[:, i]))
            if q_lo[i] == q_hi[i]:
                q_hi[i] = q_lo[i] + 1e-12

    H, edges = np.histogramdd(
        values,
        bins=(bins, bins, bins),
        range=((q_lo[0], q_hi[0]), (q_lo[1], q_hi[1]), (q_lo[2], q_hi[2])),
        weights=weights,
    )

    total_mass = float(np.sum(H))
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Invalid total posterior mass in HPD histogram")

    flat = H.ravel()
    order = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[order])
    cutoff_pos = int(np.searchsorted(cumsum, mass * total_mass, side="left"))
    cutoff_pos = min(cutoff_pos, len(order) - 1)
    keep_flat_ids = order[: cutoff_pos + 1]

    keep_bins = np.zeros_like(flat, dtype=bool)
    keep_bins[keep_flat_ids] = True
    keep_bins = keep_bins.reshape(H.shape)

    # Map points -> histogram bin IDs.
    bin_ids = []
    for i in range(3):
        b = np.digitize(values[:, i], edges[i]) - 1
        b = np.clip(b, 0, bins - 1)
        bin_ids.append(b)
    mask = keep_bins[bin_ids[0], bin_ids[1], bin_ids[2]]

    diag = {
        "hpd_bins_kept": int(np.count_nonzero(keep_bins)),
        "hpd_bins_total": int(keep_bins.size),
        "hpd_target_mass": mass,
        "hpd_achieved_mass": float(np.sum(weights[mask]) / np.sum(weights)),
        "selected_count": int(np.count_nonzero(mask)),
        "selected_weight_fraction": float(np.sum(weights[mask]) / np.sum(weights)),
    }
    return mask, diag


def _draw_weighted_indices(
    mask: np.ndarray,
    weights: np.ndarray,
    draw_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected_idx = np.flatnonzero(mask)
    if selected_idx.size == 0:
        raise ValueError("HPD-like selection produced zero rows")

    draw_count = int(max(1, draw_count))
    probs = weights[selected_idx].astype(float, copy=False)
    probs = np.clip(probs, 0.0, None)
    if not np.any(probs > 0):
        probs = np.ones_like(probs)
    probs = probs / probs.sum()

    # Weighted posterior resampling (with replacement) preserves target mass
    # while capping expensive CLASS evaluations.
    drawn = rng.choice(selected_idx, size=draw_count, replace=True, p=probs)
    uniq, counts = np.unique(drawn, return_counts=True)
    return uniq, counts


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

    with open(ini_path, "w") as f:
        f.write("# Auto-generated: posterior_background_heatmaps.py\n")
        for key in sorted(run_params):
            f.write(f"{key} = {run_params[key]}\n")

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
            raise RuntimeError(
                "CLASS background-only run failed.\n"
                f"Return code: {proc.returncode}\n"
                f"INI: {ini_path}\n"
                f"STDOUT tail:\n{stdout_tail}\n"
                f"STDERR tail:\n{stderr_tail}"
            )

        if not background_file.exists():
            raise RuntimeError(f"Expected background output missing: {background_file}")

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


def get_or_compute_background(
    class_params: dict[str, Any],
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], str, str]:
    cache_key = _hash_class_params(class_params)
    npz_path = cache_dir / f"{cache_key}.background.npz"
    meta_path = cache_dir / f"{cache_key}.meta.json"

    if npz_path.exists():
        return _load_cached_background(npz_path), "cache", cache_key

    data = _run_class_background_only(class_params, cache_dir, cache_key)

    np.savez_compressed(npz_path, **data)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "cache_key": cache_key,
                "class_params": {k: str(v) for k, v in sorted(class_params.items())},
                "fields": sorted(list(data.keys())),
            },
            f,
            indent=2,
            sort_keys=True,
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


def process_dataset(
    bundle: ChainBundle,
    args: argparse.Namespace,
    quantities: list[str],
    rng: np.random.Generator,
    cache_dir: Path,
    output_dir: Path,
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

    hpd_mask, hpd_diag = _hpd_like_mask(
        selection.hpd_values,
        selection.weights,
        bins=args.hpd_bins,
        mass=args.hpd_mass,
    )
    print(
        "HPD-like: "
        f"selected={hpd_diag['selected_count']:,} "
        f"({100.0*hpd_diag['selected_weight_fraction']:.2f}% weighted mass), "
        f"achieved_mass={hpd_diag['hpd_achieved_mass']:.4f}"
    )

    uniq_global_idx, multiplicities = _draw_weighted_indices(
        hpd_mask,
        selection.weights,
        draw_count=args.max_samples,
        rng=rng,
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

    # Pass 1: run CLASS (or load cache) once per unique sample; interpolate all quantities.
    # y_ranges[qty] = [min, max] accumulated across trajectories.
    y_ranges: dict[str, list[float]] = {qty: [np.inf, -np.inf] for qty in quantities}
    cache_hits = 0
    cache_misses = 0
    skipped = 0

    # Each element: per-quantity interpolated y-values on x_grid, plus multiplicity.
    trajectory_payload: list[tuple[dict[str, np.ndarray], int]] = []

    for i, rec in enumerate(draw_records, 1):
        row_values = extracted_by_file[rec.file_index][rec.row_index_postburn]
        row_map = {name: float(row_values[j]) for j, name in enumerate(header)}

        class_params = _build_sample_class_params(row_map, bestfit_values, yaml_config)
        bg, source, _ = get_or_compute_background(class_params, cache_dir)
        if source == "cache":
            cache_hits += 1
        else:
            cache_misses += 1

        qty_interps: dict[str, np.ndarray] = {}
        for qty in quantities:
            if qty not in bg:
                continue
            y_interp = _interp_background_quantity(bg, qty, x_grid)
            valid = np.isfinite(y_interp)
            if np.count_nonzero(valid) >= 2:
                y_ranges[qty][0] = min(
                    y_ranges[qty][0], float(np.nanmin(y_interp[valid]))
                )
                y_ranges[qty][1] = max(
                    y_ranges[qty][1], float(np.nanmax(y_interp[valid]))
                )
                qty_interps[qty] = y_interp

        if not qty_interps:
            skipped += 1
            continue

        trajectory_payload.append((qty_interps, rec.multiplicity))

        if i % 50 == 0 or i == len(draw_records):
            print(
                f"  processed {i}/{len(draw_records)} unique trajectories "
                f"(cache hit={cache_hits}, miss={cache_misses}, skipped={skipped})"
            )

    if not trajectory_payload:
        raise RuntimeError(
            f"No valid trajectories for {bundle.root} after interpolation"
        )

    dataset_label = _dataset_label_from_root(bundle.root)
    safe = _sanitize_label(bundle.root)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_label_map: dict[str, str] = {
        "phi_scf":       r"$\phi$",
        "w":             r"$w$",
        "Omega_cdm":     r"$\Omega_{\rm cdm}$",
        "Omega_scf":     r"$\Omega_{\phi}$",
        "s1":            r"$\mathfrak{s}_1 = |V_{\phi}|/V$",
        "minus_s2":      r"$\mathfrak{s}_2 = -V_{\phi\phi}/V$",
        "swampland_expr": r"$1+w-0.15\,\mathfrak{s}_1^2$",
        "swgc_lhs":      r"$(V'')^2$",
        "swgc_rhs":      r"$2(V''')^2 - V''V''''$",
        "swgc_residual": r"$\Delta_{\rm SWGC} = 2(V''')^2 - V''V'''' - (V'')^2$",
    }

    # Pass 2: for each quantity, accumulate histogram and produce figure + NPZ.
    for qty in quantities:
        y_min_qty, y_max_qty = y_ranges[qty]
        if (
            not np.isfinite(y_min_qty)
            or not np.isfinite(y_max_qty)
            or y_min_qty == y_max_qty
        ):
            print(
                f"  [WARN] Skipping {qty}: invalid y-range [{y_min_qty}, {y_max_qty}]"
            )
            continue

        pad = 0.03 * (y_max_qty - y_min_qty)
        y_edges = np.linspace(y_min_qty - pad, y_max_qty + pad, args.y_bins + 1)
        x_edges = np.linspace(args.x_min, args.x_max, args.x_bins + 1)

        H = np.zeros((args.x_bins, args.y_bins), dtype=float)
        for qty_interps, multiplicity in trajectory_payload:
            if qty not in qty_interps:
                continue
            y_interp = qty_interps[qty]
            valid = np.isfinite(y_interp)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 2:
                continue
            w = np.full(n_valid, float(multiplicity), dtype=float)
            h2d, _, _ = np.histogram2d(
                x_grid[valid],
                y_interp[valid],
                bins=(x_edges, y_edges),  # type: ignore[arg-type]
                weights=w,
            )
            H += h2d

        if not np.any(H > 0):
            print(f"  [WARN] Empty histogram for {qty}, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(8.2, 5.2))

        H_plot = H.T
        positive = H_plot[H_plot > 0]
        vmin = max(float(np.percentile(positive, 5.0)), 1e-12)
        vmax = float(np.percentile(positive, 99.8))
        if vmax <= vmin:
            vmax = float(np.max(positive))

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            H_plot,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
        )

        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("posterior path density (weighted resampling)")

        # Show cosmic time direction as in BestFitPlot-style redshift narratives.
        ax.set_xlim(args.x_max, args.x_min)
        z_ticks = np.array([1e14, 1e10, 1e7, 1e5, 1e3, 10.0, 1.0, 0.0])
        x_ticks = np.log10(1.0 + z_ticks)
        x_tick_labels = [
            r"$10^{14}$",
            r"$10^{10}$",
            r"$10^7$",
            r"$10^5$",
            r"$10^3$",
            "10",
            "1",
            "0",
        ]
        inside = (x_ticks >= args.x_min) & (x_ticks <= args.x_max)
        ax.set_xticks(x_ticks[inside])
        ax.set_xticklabels([x_tick_labels[k] for k, ok in enumerate(inside) if ok])

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(y_label_map.get(qty, qty))
        ax.set_title(
            f"{dataset_label} | {qty} posterior heatmap\n"
            f"HPD-like mass={args.hpd_mass:.2f}, draws={total_draws:,}, "
            f"unique={len(draw_records):,}"
        )

        fig.tight_layout()

        png_path = output_dir / f"heatmap_{qty}_{safe}.png"
        pdf_path = output_dir / f"heatmap_{qty}_{safe}.pdf"
        npz_path = output_dir / f"heatmap_{qty}_{safe}.npz"

        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)

        np.savez_compressed(
            npz_path,
            H=H,
            x_edges=x_edges,
            y_edges=y_edges,
            quantity=np.array([qty], dtype=str),
            root=np.array([bundle.root], dtype=str),
            dataset_label=np.array([dataset_label], dtype=str),
            total_draws=np.array([total_draws], dtype=np.int64),
            unique_trajectories=np.array([len(draw_records)], dtype=np.int64),
            cache_hits=np.array([cache_hits], dtype=np.int64),
            cache_misses=np.array([cache_misses], dtype=np.int64),
        )

        print(f"  [{qty}] Saved: {png_path.name}, {pdf_path.name}, {npz_path.name}")

    print(
        f"Background cache usage: hit={cache_hits}, miss={cache_misses}, "
        f"cache_dir={cache_dir}"
    )


def main() -> None:
    args = parse_args()
    if args.hpd_mass <= 0.0 or args.hpd_mass >= 1.0:
        raise ValueError("--hpd-mass must be in (0, 1)")

    # Resolve quantities: --preset > --quantities > --quantity
    if args.preset is not None:
        quantities: list[str] = list(_PRESET_QUANTITIES[args.preset])
    elif args.quantities is not None:
        quantities = list(args.quantities)
    else:
        quantities = [str(args.quantity)]

    cache_dir = (_repo_root() / args.cache_dir).resolve()
    output_dir = (_repo_root() / args.output_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    hdm_root = _hdm_root()

    print("Posterior background heatmap pipeline")
    print(f"  HDM root:   {hdm_root}")
    print(f"  Cache dir:  {cache_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Quantities: {quantities}")
    print(f"  HPD params: {args.hpd_params}")
    print(f"  HPD mass:   {args.hpd_mass}")
    print(f"  Max draws:  {args.max_samples}")

    for root in args.roots:
        bundle = discover_chain_bundle(root, hdm_root)
        process_dataset(bundle, args, quantities, rng, cache_dir, output_dir)


if __name__ == "__main__":
    main()
