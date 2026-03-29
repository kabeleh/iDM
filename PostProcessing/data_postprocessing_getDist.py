"""
GetDist MCMC Chain Post-Processing: Tables and Visualization

Generate publication-ready GetDist plots and LaTeX tables from MCMC chains.
Includes robust font handling, adaptive convergence diagnostics, and marker/style visualization.

Available CLI flags:
  --skip-plots                 Skip all plotting and only generate LaTeX tables.
  --skip-tables                Skip LaTeX table generation and only produce plots.
  --strict-export              Automatically export final plots with strict IBM Plex Math
                               (PDF + PGF via lualatex); requires lualatex installed.
  --strict-export-dir DIR      Output directory for --strict-export files
                               (default: current working directory).
  --show-all-kde-warnings      Show all repetitive GetDist KDE/binning warnings
                               (default: condensed output with 4 representative warnings per category).

Examples:
  python3 data_postprocessing_getDist.py                     # Full pipeline: tables + plots
  python3 data_postprocessing_getDist.py --skip-plots         # Tables only
  python3 data_postprocessing_getDist.py --skip-tables        # Plots only
  python3 data_postprocessing_getDist.py --strict-export      # Tables + plots + PGF export
  python3 data_postprocessing_getDist.py --skip-tables --strict-export  # Plots + PGF export
  python3 data_postprocessing_getDist.py --show-all-kde-warnings  # Debug: show all warnings
"""

# %%
# Import required libraries:
# - matplotlib.pyplot for plotting
# - cmcrameri.cm for perceptually uniform colormaps
# - getdist.plots for MCMC chain plotting
from typing import Any, Callable, Mapping, Sequence, cast
import argparse
import os
import re
import glob
import shutil
import math
import atexit
import logging
import numpy as np  # type: ignore[import-untyped]
from cycler import cycler

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib import colors as mcolors

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cmcrameri import cm  # type: ignore[import-untyped]
from getdist import plots  # type: ignore[import-untyped]

# Initialize module logger for debugging exception handling
_LOGGER = logging.getLogger(__name__)
_DEBUG_EXCEPTIONS = os.environ.get('GETDIST_DEBUG_EXCEPTIONS', '').lower() in ('1', 'true', 'yes')


# Preview font selection: robust detection with fallback and file logging.
def _find_font_file(font_family: str) -> tuple[bool, str]:
    """Check font availability and return (found, file_path_or_status).

    Returns:
        (True, '/path/to/font.ttf') if found
        (False, 'reason') if not found
    """
    # Method 1: Check fontManager.ttflist cache (fast and safe)
    try:
        for font_entry in font_manager.fontManager.ttflist:
            if (
                isinstance(font_entry.name, str)
                and font_entry.name.strip().lower() == font_family.strip().lower()
            ):
                return (True, font_entry.fname)
    except (AttributeError, TypeError) as e:
        # fontManager.ttflist may not exist or behave unexpectedly
        if _DEBUG_EXCEPTIONS:
            _LOGGER.debug(f"Font cache check failed for {font_family}: {e}")
        pass

    # Method 2: Scan system fonts and match exact FT2 family name.
    # This catches user-installed fonts that may not be in matplotlib's cache.
    try:
        for font_path in font_manager.findSystemFonts():
            try:
                ft_font = font_manager.get_font(font_path)
                family_name = getattr(ft_font, "family_name", "")
                if (
                    isinstance(family_name, str)
                    and family_name.strip().lower() == font_family.strip().lower()
                ):
                    return (True, font_path)
            except (AttributeError, OSError, RuntimeError) as e:
                # Font may be corrupted or inaccessible; skip and continue
                if _DEBUG_EXCEPTIONS:
                    _LOGGER.debug(f"Could not parse font at {font_path}: {e}")
                pass
    except (OSError, RuntimeError) as e:
        # findSystemFonts or font access may fail on system
        if _DEBUG_EXCEPTIONS:
            _LOGGER.debug(f"System font scan failed for {font_family}: {e}")
        pass

    # Method 3: Last resort - findfont (may return fallback, not our font)
    try:
        resolved = font_manager.findfont(font_family)
        # Only return True if the path actually contains the font name
        if (
            resolved
            and font_family.replace(" ", "").lower()
            in resolved.replace(" ", "").lower()
        ):
            return (True, resolved)
        # findfont returned a fallback, not what we want
        return (False, "Not found in system; findfont would use fallback")
    except (OSError, RuntimeError, AttributeError) as e:
        # Font resolution API may not be available or fail on system
        return (False, f"Font detection failed: {type(e).__name__}: {str(e)}")


def _register_preview_font(font_name: str, path: str) -> None:
    """Register a discovered font file with matplotlib runtime manager."""
    try:
        font_manager.fontManager.addfont(path)
    except (OSError, RuntimeError, AttributeError) as e:
        # Font may be corrupted or addfont may not be available on this system
        if _DEBUG_EXCEPTIONS:
            _LOGGER.debug(f"Could not register font {font_name} from {path}: {e}")
        pass


# Check each required font and collect results, then register explicit files.
_FONT_CHECK_RESULTS: dict[str, tuple[bool, str]] = {}
for font_name in ("IBM Plex Serif", "IBM Plex Sans", "IBM Plex Mono"):
    found, info = _find_font_file(font_name)
    _FONT_CHECK_RESULTS[font_name] = (found, info)
    if found:
        _register_preview_font(font_name, info)

_HAS_PLEX_PREVIEW = all(found for found, _ in _FONT_CHECK_RESULTS.values())

if _HAS_PLEX_PREVIEW:
    _PREVIEW_SERIF = "IBM Plex Serif"
    _PREVIEW_SANS = "IBM Plex Sans"
    _PREVIEW_MONO = "IBM Plex Mono"
    print("Using IBM Plex fonts for preview:")
    for font_name, (found, path) in _FONT_CHECK_RESULTS.items():
        if found:
            print(f"  {font_name}: {path}")
else:
    _PREVIEW_SERIF = "DejaVu Serif"
    _PREVIEW_SANS = "DejaVu Sans"
    _PREVIEW_MONO = "DejaVu Sans Mono"
    print(
        "Warning: Not all IBM Plex fonts found for interactive preview. "
        "Using DejaVu fallback. Strict --strict-export remains IBM Plex Math via lualatex."
    )
    for font_name, (found, reason) in _FONT_CHECK_RESULTS.items():
        if not found:
            print(f"  {font_name}: {reason}")


# Plot Configuration
mpl.rcParams.update(
    {
        "font.family": [_PREVIEW_SERIF],
        "font.sans-serif": [_PREVIEW_SANS, "DejaVu Sans", "sans-serif"],
        "font.monospace": [_PREVIEW_MONO, "DejaVu Sans Mono", "monospace"],
        "font.serif": [_PREVIEW_SERIF, "DejaVu Serif", "serif"],
        "font.size": 10,  # body text size (most journals use 10 pt)
        "axes.labelsize": 10,  # axis-label size matches body text
        "xtick.labelsize": 9,  # tick labels one point smaller
        "ytick.labelsize": 9,
        "legend.fontsize": 9,  # legend text one point smaller
        "axes.prop_cycle": cycler(
            "color",
            [  # Okabe–Ito colorblind-safe palette
                "#0072B2",
                "#D55E00",
                "#009E73",
                "#E69F00",
                "#CC79A7",
                "#56B4E9",
            ],
        ),
        "lines.linewidth": 1.5,  # slightly thicker for print clarity
        "axes.linewidth": 0.8,  # thinner axis frame
        "xtick.direction": "in",  # inward ticks — journal standard
        "ytick.direction": "in",
        "xtick.minor.visible": True,  # show minor ticks
        "ytick.minor.visible": True,
        "xtick.major.size": 4,  # longer than the 3.5 default
        "ytick.major.size": 4,
        "xtick.minor.size": 2,  # half of major — proportional
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,  # match axes.linewidth
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,  # thinner for visual hierarchy
        "ytick.minor.width": 0.6,
        "lines.markersize": 4,  # smaller markers for print scale
        "errorbar.capsize": 3,  # visible end-caps (default is 0)
        "axes.xmargin": 0.02,  # hug the data (default is 0.05)
        "axes.ymargin": 0.02,
        "legend.frameon": False,  # no legend box
        "savefig.bbox": "tight",  # tight bounding box by default
        "savefig.dpi": 300,  # publication-quality resolution
        # Thesis text width is 13.1 cm. Keep a consistent figure footprint.
        "figure.figsize": (13.1 / 2.54, 13.1 / 2.54 * 0.72),
        # Prefer high-quality static output by default.
        "savefig.format": "pdf",
        # Interactive preview mode: robust on-screen rendering.
        "text.usetex": False,
        "mathtext.fontset": "stix",
    }
)


STRICT_PLEX_PGF_RC: dict[str, Any] = {
    "text.usetex": True,
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False,
    "pgf.preamble": (
        r"\usepackage{fontspec}"
        r"\usepackage{mathtools}"
        r"\usepackage{amssymb}"
        r"\usepackage[warnings-off={mathtools-overbracket,mathtools-colon}]{unicode-math}"
        r"\setmainfont{IBM Plex Serif}"
        r"\setsansfont{IBM Plex Sans}"
        r"\setmonofont{IBM Plex Mono}"
        r"\setmathfont{IBM Plex Math}"
    ),
    "text.latex.preamble": (
        r"\usepackage{fontspec}"
        r"\usepackage{mathtools}"
        r"\usepackage{amssymb}"
        r"\usepackage[warnings-off={mathtools-overbracket,mathtools-colon}]{unicode-math}"
        r"\setmainfont{IBM Plex Serif}"
        r"\setsansfont{IBM Plex Sans}"
        r"\setmonofont{IBM Plex Mono}"
        r"\setmathfont{IBM Plex Math}"
    ),
}


plt = cast(Any, plt)
cm = cast(Any, cm)
plots = cast(Any, plots)

Color = Any

THESIS_TEXTWIDTH_CM = 13.1
FIGURE_WIDTH_IN = THESIS_TEXTWIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_WIDTH_IN * 0.72

LINE_STYLES: list[str] = ["-", "--", "-.", ":"]
MARKERS: list[str] = ["o", "s", "^", "v", "D", "P", "X", "*"]

# Global cache for loaded samples to avoid redundant loading
_SAMPLES_CACHE: dict[str, Any] = {}
_ROOT_PATH_CACHE: dict[str, str] = {}

# GetDist imports for MCMC analysis
from getdist import MCSamples, loadMCSamples  # type: ignore[import-untyped]

MCSamples = cast(Any, MCSamples)
loadMCSamples = cast(Callable[..., Any], loadMCSamples)


def _style_for_chain(index: int) -> tuple[str, str]:
    """Return (linestyle, marker) for deterministic chain styling."""
    return (
        LINE_STYLES[index % len(LINE_STYLES)],
        MARKERS[index % len(MARKERS)],
    )


def save_strict_plex_figure(
    fig: Any,
    pdf_path: str,
    pgf_path: str | None = None,
) -> None:
    """Save a figure with strict IBM Plex Math using lualatex+PGF.

    This function is intended for final thesis figures. It leaves interactive
    preview rendering untouched, so `plt.show()` continues to work normally.
    """
    if shutil.which("lualatex") is None:
        raise RuntimeError(
            "Cannot export strict IBM Plex Math figure: lualatex is not available."
        )

    with mpl.rc_context(STRICT_PLEX_PGF_RC):
        fig.savefig(pdf_path, format="pdf", backend="pgf", bbox_inches="tight", dpi=300)
        if pgf_path is not None:
            fig.savefig(pgf_path, format="pgf", backend="pgf", bbox_inches="tight")


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the postprocessing script."""
    parser = argparse.ArgumentParser(
        description="Generate GetDist plots and LaTeX tables from MCMC chains."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip all plotting and only generate the tables.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip LaTeX table generation and only produce plots.",
    )
    parser.add_argument(
        "--strict-export",
        action="store_true",
        help=(
            "Automatically export final plots with strict IBM Plex Math "
            "(PDF + PGF via lualatex)."
        ),
    )
    parser.add_argument(
        "--strict-export-dir",
        default=".",
        help=(
            "Output directory for --strict-export files (default: current working directory)."
        ),
    )
    parser.add_argument(
        "--show-all-kde-warnings",
        action="store_true",
        help=(
            "Show all repetitive GetDist KDE/binning warnings (default is condensed output)."
        ),
    )
    return parser.parse_args()


CLI_ARGS = parse_cli_args()


class _GetDistWarningLimiter(logging.Filter):
    """Condense repetitive GetDist KDE warning spam while keeping signal."""

    def __init__(self, max_per_bucket: int = 4):
        super().__init__()
        self.max_per_bucket = max_per_bucket
        self.counts: dict[str, int] = {}

    def _bucket(self, message: str) -> str | None:
        if "2D kernel density bandwidth optimizer failed" in message:
            return "kde_2d_optimizer"
        if "auto bandwidth for" in message and "Using fallback" in message:
            return "kde_auto_bandwidth_fallback"
        if "fine_bins not large enough to well sample smoothing scale" in message:
            return "kde_fine_bins_too_small"
        return None

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        bucket = self._bucket(msg)
        if bucket is None:
            return True
        count = self.counts.get(bucket, 0) + 1
        self.counts[bucket] = count
        return count <= self.max_per_bucket


_KDE_WARNING_LIMITER: _GetDistWarningLimiter | None = None


def _install_getdist_warning_limiter() -> None:
    """Install a logging filter that condenses repetitive GetDist warnings."""
    global _KDE_WARNING_LIMITER
    if CLI_ARGS.show_all_kde_warnings:
        return
    root_logger = logging.getLogger()
    limiter = _GetDistWarningLimiter(max_per_bucket=4)
    root_logger.addFilter(limiter)
    _KDE_WARNING_LIMITER = limiter

    def _print_summary() -> None:
        if _KDE_WARNING_LIMITER is None:
            return
        suppressed: list[str] = []
        for bucket, count in _KDE_WARNING_LIMITER.counts.items():
            extra = count - _KDE_WARNING_LIMITER.max_per_bucket
            if extra > 0:
                suppressed.append(f"{bucket}: {extra}")
        if suppressed:
            print(
                "Note: Condensed repetitive GetDist warnings "
                f"(use --show-all-kde-warnings to disable): {', '.join(suppressed)}"
            )

    atexit.register(_print_summary)


_install_getdist_warning_limiter()


def _validate_chain_root_exists(
    root: str,
    chain_dir: str,
) -> tuple[bool, str]:
    """Check if chain root files exist. Return (valid, reason)."""
    base_path = _root_base_path(root, chain_dir)
    required_extensions = [".txt", ".bestfit", ".input.yaml", ".updated.yaml"]

    # Check for at least one required file or numbered chain file
    has_txt = any(
        os.path.exists(f"{base_path}{ext}") for ext in required_extensions
    ) or bool(glob.glob(f"{base_path}.*.txt"))

    if not has_txt:
        return (False, f"No chain files found for root: {base_path}")

    return (True, base_path)


def _suggest_fine_bins(sample_count: int, current_bins: int) -> int:
    """Return a rounded-up fine_bins target based on sample count.

    We target ~2 samples per fine bin and round to the next multiple of 256
    for stable bin counts across runs.
    """
    max_fine_bins = 8192
    if sample_count <= 0:
        return min(current_bins, max_fine_bins)
    target = max(current_bins, int(math.ceil(sample_count / 2.0)))
    rounded = int(math.ceil(target / 256.0) * 256)
    return min(rounded, max_fine_bins)


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
        If True, print loading progress and validation results.
    """
    if verbose:
        print(f"Preflight validation: checking {len(roots)} chain root(s)...")

    # Preflight check: validate all roots exist before attempting loads
    valid_roots = []
    skipped_roots = []
    for i, root in enumerate(roots, 1):
        resolved_root = resolve_chain_root(root, chain_dir)
        valid, reason = _validate_chain_root_exists(resolved_root, chain_dir)
        if valid:
            valid_roots.append((i, root, resolved_root))
        else:
            skipped_roots.append((i, root, reason))
            if verbose:
                print(f"  [{i}/{len(roots)}] Skipping {root}: {reason}")

    # Build mutable active settings and ensure fine_bins is explicit.
    active_settings: dict[str, Any] = dict(settings)
    max_fine_bins = 8192
    current_fine_bins = min(int(active_settings.get("fine_bins", 1024)), max_fine_bins)
    active_settings["fine_bins"] = current_fine_bins

    def _load_pass(
        pass_label: str, pass_settings: Mapping[str, Any]
    ) -> tuple[int, int]:
        if verbose and valid_roots:
            print(
                f"Preloading {len(valid_roots)}/{len(roots)} valid chain(s) "
                f"({pass_label}, fine_bins={int(pass_settings.get('fine_bins', 1024))})..."
            )

        successful_local = 0
        suggested_fine_bins = int(pass_settings.get("fine_bins", 1024))

        for i_orig, root, resolved_root in valid_roots:
            cache_key = _cache_key(chain_dir, resolved_root)
            try:
                if verbose:
                    print(f"  [{i_orig}/{len(roots)}] Loading {root}...")
                samples = loadMCSamples(
                    _root_base_path(resolved_root, chain_dir), settings=pass_settings
                )
                _SAMPLES_CACHE[cache_key] = samples
                successful_local += 1

                sample_count = len(getattr(samples, "samples", []))
                suggested_fine_bins = max(
                    suggested_fine_bins,
                    _suggest_fine_bins(
                        sample_count, int(pass_settings.get("fine_bins", 1024))
                    ),
                )
            except Exception as e:
                if verbose:
                    print(f"  [{i_orig}/{len(roots)}] Failed to load {root}: {e}")
                _SAMPLES_CACHE[cache_key] = None

        return successful_local, suggested_fine_bins

    successful, suggested = _load_pass("pass 1", active_settings)

    # If probing indicates larger bins are needed, restart from scratch with updated bins.
    if suggested > current_fine_bins:
        if verbose:
            print(
                f"Adaptive binning: increasing fine_bins from {current_fine_bins} to {suggested}."
            )
            print("Clearing cache and restarting chain preload from scratch...")

        _SAMPLES_CACHE.clear()
        active_settings["fine_bins"] = suggested

        # Persist to caller settings if mutable (e.g. ANALYSIS_SETTINGS dict).
        if isinstance(settings, dict):
            settings["fine_bins"] = suggested

        successful, _ = _load_pass("pass 2", active_settings)

    if verbose:
        if skipped_roots:
            print(
                f"Cache preload complete: {successful}/{len(valid_roots)} valid chains loaded successfully. "
                f"({len(skipped_roots)} skipped due to missing files). "
                f"Using fine_bins={int(active_settings.get('fine_bins', 1024))}.\n"
            )
        else:
            print(
                f"Cache preload complete: {successful}/{len(roots)} chains loaded successfully. "
                f"Using fine_bins={int(active_settings.get('fine_bins', 1024))}.\n"
            )


# ============================================================================
# COMMON CONFIGURATION
# ============================================================================

# Resolve paths relative to this script so both Linux and macOS layouts work.
# Expected layout:
#   <HDM>/class_public/PostProcessing/data_postprocessing_getDist.py
#   <HDM>/MCMC_archive/
#   <HDM>/MCMC_chains/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASS_PUBLIC_DIR = os.path.dirname(_SCRIPT_DIR)
HDM_DIR: str = os.path.dirname(_CLASS_PUBLIC_DIR)

# Keep a single chain root directory and resolve each root with per-chain
# precedence: MCMC_archive first, then MCMC_chains.
CHAIN_DIR: str = HDM_DIR
_CHAIN_SEARCH_SUBDIRS: tuple[str, ...] = ("MCMC_archive", "MCMC_chains")


def _ordered_chain_search_dirs(chain_dir: str) -> list[str]:
    """Return existing chain search dirs in priority order."""
    prioritized = [os.path.join(chain_dir, sub) for sub in _CHAIN_SEARCH_SUBDIRS]
    existing_prioritized = [d for d in prioritized if os.path.isdir(d)]

    # Fall back to chain_dir itself if the subdirs are missing.
    if os.path.isdir(chain_dir):
        if chain_dir not in existing_prioritized:
            existing_prioritized.append(chain_dir)

    return existing_prioritized


ANALYSIS_SETTINGS: dict[str, float] = {
    "ignore_rows": 0.33,
    # "fine_bins": 2048,
    "fine_bins_2D": 2048,
}

# Centralized observational references used by plots and tension calculations.
# Update these values in one place when new measurements are released.
OBSERVATIONAL_REFERENCES: dict[str, dict[str, Any]] = {
    "H0": {
        "mean": 73.18,
        "sigma": 0.88,
        "label": r"$H_0$ SH0ES 2025",
    },
    "S8": {
        "mean": 0.776,
        "sigma": 0.031,
        "label": r"$S_8$ KiDS-1000 2023",
    },
}

# Define the root names of the MCMC chains (file prefixes without extensions)
# Naming conventions supported:
#   Legacy:  cobaya_<sampler>_<likelihood>_<potential>_<attractor>_<coupling>
#   New:     <potential>_<likelihoods>_<attractor>_<coupling>_<sampler>
ROOTS: list[str] = [
    # --- Legacy naming examples ---
    # "cobaya_mcmc_fast_CMB_LCDM",
    # "cobaya_mcmc_fast_CMB_LCDM.post.PP",
    # "cobaya_mcmc_fast_CMB_LCDM.post.PPS",
    # "cobaya_polychord_CV_PP_DESI_LCDM.post.S8",
    # "cobaya_mcmc_CV_PP_S_DESI_LCDM.post.S8",
    # "cobaya_mcmc_CV_CMB_SPA_LCDM.post.S8",
    # "cobaya_mcmc_CV_CMB_SPA_PP_DESI_LCDM.post.S8",
    # "cobaya_mcmc_CV_CMB_SPA_PP_S_DESI_LCDM.post.S8",
    # "cobaya_mcmc_fast_CMB_DoubleExp_InitCond_uncoupled",
    # "cobaya_mcmc_CV_PP_S_DESI_hyperbolic_InitCond_uncoupled",
    # --- New naming examples ---
    # "LCDM_Planck_MCMC",
    # "LCDM_SPA_PP_S_D_MCMC",
    # "DoubleExp_Planck_InitCond_MCMC",
    # "DoubleExp_SPA_PP_S_D_InitCond_MCMC",
    # "hyperbolic_PP_S_D_InitCond_Polychord",
    # "DoubleExp_Planck_InitCond_MCMC.post.SN_BAO",
    # "DoubleExp_SPA_PP_S_D_InitCond_MCMC.post.swampland",
    # "DoubleExp_SPA_PP_S_D_InitCond_MCMC",
    # "hyperbolic_PP_S_D_InitCond_MCMC",
    # --- LCDM Archive ---
    "cobaya_mcmc_fast_CMB_LCDM",
    "cobaya_mcmc_fast_CMB_LCDM.post.PP",
    "cobaya_mcmc_fast_CMB_LCDM.post.PPS",
    # --- All potentials comparison Planck Data ---
    # "BeanAdS_Planck_InitCond_MCMC",
    # "Bean_Planck_InitCond_MCMC",
    # "cosine_Planck_InitCond_MCMC",
    # "DoubleExp_Planck_InitCond_MCMC",
    # "exponential_Planck_InitCond_MCMC",
    # "hyperbolic_Planck_InitCond_MCMC",
    "hyperbolic_Planck_InitCond_MCMC.post.Swamp",
    # "hyperbolic_Planck_InitCond_MCMC.post.PP_DESI",
    "hyperbolic_Planck_InitCond_MCMC.post.PP_DESI.post.Swamp",
    # "hyperbolic_Planck_InitCond_MCMC.post.PPS_DESI",
    "hyperbolic_Planck_InitCond_MCMC.post.PPS_DESI.post.Swamp",
    # "hyperbolic_Planck_tracking_MCMC",
    # "pNG_Planck_InitCond_MCMC",
    # "pNG_Planck_InitCond_MCMC.post.PP_DESI",
    # "pNG_Planck_InitCond_MCMC.post.PP_DESI.post.Swamp",
    # "pNG_Planck_InitCond_MCMC.post.PPS_DESI",
    # "pNG_Planck_InitCond_MCMC.post.PPS_DESI.post.Swamp",
    # "power-law_Planck_InitCond_MCMC",
    # "SqE_Planck_InitCond_MCMC",
]


# Use Crameri colourmaps directly with evenly spaced sampling to preserve
# their intended hue progression and avoid collapsing to a few dark tones.
def _sample_evenly(colours: Sequence[Color], count: int) -> list[Color]:
    """Sample `count` colours evenly from a colour list, ensuring distinctness.

    Uses evenly-spaced indices to preserve palette hue progression.
    If not enough colours, cycles through the palette.
    """
    if count <= 0:
        return []
    if count == 1:
        return [colours[len(colours) // 2]]
    if len(colours) >= count:
        # Evenly space indices across the palette
        idx = np.linspace(0, len(colours) - 1, num=count, dtype=int)
        return [colours[int(i)] for i in idx]
    else:
        # Cycle through palette if we need more colours than available
        return [colours[i % len(colours)] for i in range(count)]


# Create a permanent colour mapping from ROOTS to colours.
# This mapping persists across all plots and sorting operations.
_RAW_CHAIN_COLOURS: list[Color] = [tuple(c) for c in cm.batlowKS.colors]
ROOT_TO_COLOUR: dict[str, Color] = {
    root: _RAW_CHAIN_COLOURS[i % len(_RAW_CHAIN_COLOURS)]
    for i, root in enumerate(ROOTS)
}

# Pick two distinct band colours from romaO colormap
_BAND_COLOURS_SOURCE: list[Color] = [tuple(c) for c in cm.romaO.colors]
BAND_COLOURS: list[Color] = [
    _BAND_COLOURS_SOURCE[len(_BAND_COLOURS_SOURCE) // 6],
    _BAND_COLOURS_SOURCE[5 * len(_BAND_COLOURS_SOURCE) // 6],
]


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


def _sort_chain_matches(matches: Sequence[str], chain_dir: str) -> list[str]:
    """Prefer canonical top-level matches over deeper duplicates."""

    def _match_key(path: str) -> tuple[int, int, str]:
        rel = os.path.relpath(path, chain_dir).replace(os.sep, "/")
        root = _root_from_filepath(path, chain_dir)
        depth = root.count("/")
        penalty = 1 if "/initialTesting/" in f"/{rel}" else 0
        return (penalty, depth, rel)

    return sorted(set(matches), key=_match_key)


def _unique_roots_from_matches(matches: Sequence[str], chain_dir: str) -> list[str]:
    """Return sorted unique chain roots extracted from file matches."""
    unique = sorted({_root_from_filepath(m, chain_dir) for m in matches})
    return sorted(
        unique,
        key=lambda r: (
            1 if "/initialTesting/" in f"/{r}" else 0,
            r.count("/"),
            r,
        ),
    )


def _prefer_best_root(root: str, unique_roots: Sequence[str]) -> str:
    """Prefer exact basename matches to avoid resolving into .post variants."""
    basename_exact = [r for r in unique_roots if os.path.basename(r) == root]
    if basename_exact:
        return basename_exact[0]

    # If the request does not include post-processing suffix, prefer non-post roots.
    if ".post." not in root:
        non_post = [r for r in unique_roots if ".post." not in os.path.basename(r)]
        if non_post:
            return non_post[0]

    return unique_roots[0]


def _should_warn_ambiguity(root: str, unique_roots: Sequence[str]) -> bool:
    """Warn only for meaningful ambiguities that can change scientific selection."""
    if len(unique_roots) <= 1:
        return False

    # Multiple locations of the exact same basename (e.g., initialTesting duplicates)
    # are expected and handled by priority ordering.
    if all(os.path.basename(r) == root for r in unique_roots):
        return False

    # If root has no post suffix, suppress warning caused only by .post.* variants.
    if ".post." not in root:
        base_names = [os.path.basename(r) for r in unique_roots]
        if any(b == root for b in base_names):
            non_exact = [b for b in base_names if b != root]
            if non_exact and all(b.startswith(f"{root}.post.") for b in non_exact):
                return False

    return True


def _collect_chain_matches(
    patterns: Sequence[str],
    chain_dir: str,
) -> list[str]:
    """Return prioritized matches from archive/chains search dirs."""
    for search_dir in _ordered_chain_search_dirs(chain_dir):
        dir_matches: list[str] = []
        for pattern in patterns:
            dir_matches.extend(
                glob.glob(os.path.join(search_dir, pattern), recursive=True)
            )
        if dir_matches:
            return _sort_chain_matches(dir_matches, chain_dir)
    return []


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
        matches = _collect_chain_matches(patterns, chain_dir)

        if not matches:
            return root

        unique_roots = _unique_roots_from_matches(matches, chain_dir)
        resolved = _prefer_best_root(root, unique_roots)
        _ROOT_PATH_CACHE[root] = resolved

        # Only warn if there are multiple distinct chain roots (truly ambiguous)
        # GetDist naturally has multiple files per chain (.1.txt, .2.txt, .bestfit, etc.)
        if _should_warn_ambiguity(root, unique_roots):
            print(
                f"Warning: Multiple distinct chains match root '{root}'. "
                f"Using '{resolved}'. Found chains: {unique_roots}"
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
    matches = _collect_chain_matches(patterns, chain_dir)

    if not matches:
        _ROOT_PATH_CACHE[root] = root
        return root

    unique_roots = _unique_roots_from_matches(matches, chain_dir)
    resolved = _prefer_best_root(root, unique_roots)
    _ROOT_PATH_CACHE[root] = resolved

    # Only warn if there are multiple distinct chain roots (truly ambiguous)
    # GetDist naturally has multiple files per chain (.1.txt, .2.txt, .bestfit, etc.)
    if _should_warn_ambiguity(root, unique_roots):
        print(
            f"Warning: Multiple distinct chains match root '{root}'. "
            f"Using '{resolved}'. Found chains: {unique_roots}"
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


def infer_dataset_flags_from_root(root: str) -> dict[str, bool]:
    """Infer dataset flags from chain naming conventions.

    PP/PPS chains implicitly include DESI DR2 in this project.
    """
    root_lower = root.lower()

    post_suffix = ""
    post_match = re.search(r"\.post\.(\w+)$", root_lower)
    if post_match:
        post_suffix = post_match.group(1)
    base_lower = re.sub(r"\.post\.\w+$", "", root_lower)
    is_legacy = base_lower.startswith("cobaya_")

    has_planck = "planck" in base_lower or (
        is_legacy and ("_cmb_" in base_lower or base_lower.endswith("_cmb"))
    )
    has_spa = "spa" in base_lower

    has_pantheon = False
    has_sh0es = False
    has_desi = False

    if is_legacy:
        if "pp" in base_lower or "pantheon" in base_lower:
            has_pantheon = True
        if "_s_" in base_lower or "sh0es" in base_lower or "shoes" in base_lower:
            has_sh0es = True
        if "desi" in base_lower:
            has_desi = True
    else:
        if "pp_s_d" in base_lower or "pps_d" in base_lower:
            has_pantheon = True
            has_sh0es = True
            has_desi = True
        elif "pp_d" in base_lower:
            has_pantheon = True
            has_desi = True
        else:
            if re.search(r"(^|_)pp(s)?(_|$)", base_lower) or "pantheon" in base_lower:
                has_pantheon = True
            if (
                re.search(r"(^|_)s(_|$)", base_lower)
                or "sh0es" in base_lower
                or "shoes" in base_lower
            ):
                has_sh0es = True
            if "desi" in base_lower or re.search(r"(^|_)d(_|$)", base_lower):
                has_desi = True

    if post_suffix == "pps":
        has_pantheon = True
        has_sh0es = True
    elif post_suffix == "pp":
        has_pantheon = True
    elif post_suffix == "sn_bao":
        has_pantheon = True
        has_sh0es = True
        has_desi = True

    # Project rule: all Pantheon+ (PP/PPS) chains also include DESI DR2.
    if has_pantheon:
        has_desi = True

    return {
        "has_planck": has_planck,
        "has_spa": has_spa,
        "has_pantheon": has_pantheon,
        "has_sh0es": has_sh0es,
        "has_desi": has_desi,
    }


def build_legend_label(root: str) -> str:
    """Build a legend label from the chain root name.

    Supports both legacy naming (cobaya_<sampler>_<likelihood>_<potential>_...)
    and new naming (<potential>_<likelihoods>_<attractor>_<coupling>_<sampler>).
    """
    root_lower = root.lower()

    # Separate base name from .post.* suffix
    base_lower = re.sub(r"\.post\.\w+$", "", root_lower)
    dataset_flags = infer_dataset_flags_from_root(root)

    # --- Detect model/potential ---
    if "lcdm" in base_lower:
        model_label = r"$\Lambda$CDM"
    elif "hyperbolic" in base_lower and "tracking" in base_lower:
        model_label = "Hyperbolic (tracking)"
    elif "hyperbolic" in base_lower:
        model_label = "Hyperbolic"
    elif "doubleexp" in base_lower or "doubleexponential" in base_lower:
        model_label = "Double Exponential"
    elif "beanads" in base_lower:
        model_label = "BeanAdS"
    elif "bean" in base_lower:
        model_label = "Bean"
    elif "exponential" in base_lower:
        model_label = "Exponential"
    elif "cosine" in base_lower:
        model_label = "Cosine"
    elif "png" in base_lower:
        model_label = "pseudo-Nambu-Goldstone"
    elif re.search(r"power.?law", base_lower):
        model_label = "Power-law"
    elif "sqe" in base_lower:
        model_label = "Squared Exponential"
    else:
        model_label = "Model"

    likelihoods: list[str] = []

    # --- CMB detection ---
    # Legacy: "planck" or "_cmb_" in name; "fast" alone does NOT imply Planck
    # New: "planck" as a data tag
    if dataset_flags["has_planck"]:
        likelihoods.append("Planck 2018")
    if dataset_flags["has_spa"]:
        likelihoods.append("SPA")

    if dataset_flags["has_desi"]:
        likelihoods.append("DESI DR2")
    if dataset_flags["has_pantheon"]:
        likelihoods.append("Pantheon+")
    if dataset_flags["has_sh0es"]:
        likelihoods.append("SH0ES")

    if likelihoods:
        return f"{model_label}: " + " | ".join(likelihoods)
    return f"{model_label}: (no likelihood tags)"


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def _build_chain_line_args(chain_colors: Sequence[Color]) -> list[dict[str, Any]]:
    """Build per-chain plotting styles for improved grayscale legibility."""
    line_args: list[dict[str, Any]] = []
    for i, color in enumerate(chain_colors):
        linestyle, marker = _style_for_chain(i)
        line_args.append(
            {
                "color": color,
                "ls": linestyle,
                "lw": 1.5,
                # Apply markers in post-processing where we can distinguish
                # open 1D lines from closed 2D contour loops.
                "marker": "None",
                "markersize": 3.8,
            }
        )
    return line_args


def _apply_chain_styles_to_axes(g: Any, used_roots: Sequence[str]) -> None:
    """Force chain linestyles/markers on rendered axes lines.

    GetDist sometimes draws lines whose style is not fully controlled by `line_args`
    (especially on diagonal 1D panels). This post-pass keeps legend and plot styles aligned.
    """
    style_by_rgb: dict[tuple[float, float, float], tuple[str, str, int]] = {}
    for i, root in enumerate(used_roots):
        rgb_arr = mcolors.to_rgb(ROOT_TO_COLOUR[root])
        rgb = (float(rgb_arr[0]), float(rgb_arr[1]), float(rgb_arr[2]))
        linestyle, marker = _style_for_chain(i)
        style_by_rgb[rgb] = (linestyle, marker, i)

    def _nearest_style(line_color: Any) -> tuple[str, str, int] | None:
        try:
            rgb_arr = mcolors.to_rgb(line_color)
            rgb = (float(rgb_arr[0]), float(rgb_arr[1]), float(rgb_arr[2]))
        except Exception:
            return None
        best: tuple[str, str, int] | None = None
        best_d2 = 1e9
        for key_rgb, style in style_by_rgb.items():
            d2 = sum((a - b) ** 2 for a, b in zip(rgb, key_rgb))
            if d2 < best_d2:
                best_d2 = d2
                best = style
        if best is not None and best_d2 < 1e-4:
            return best
        return None

    for ax in g.fig.axes:
        for line in ax.get_lines():
            style = _nearest_style(line.get_color())
            if style is None:
                continue
            linestyle, marker, idx = style
            line.set_linestyle(linestyle)
            line.set_linewidth(1.5)

            # Keep markers off on closed contour loops; use sparse markers on
            # open lines so both marker and linestyle remain identifiable.
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            is_closed_loop = (
                x.size > 3
                and y.size > 3
                and np.isfinite(x[0])
                and np.isfinite(x[-1])
                and np.isfinite(y[0])
                and np.isfinite(y[-1])
                and abs(float(x[0] - x[-1])) < 1e-12
                and abs(float(y[0] - y[-1])) < 1e-12
            )

            if is_closed_loop:
                line.set_marker("None")
                continue

            npts = int(x.size)
            step = max(120, npts // 8)
            offset = (idx * 19) % step
            line.set_marker(marker)
            line.set_markersize(3.6)
            line.set_markevery((offset, step))


def _estimate_h0_s8_1sigma_area(samples: Any) -> float:
    """Estimate 68% contour area proxy in the H0-S8 plane.

    Uses marginalized 68% widths when available; falls back to 2*std.
    Returned area is proportional to contour size and is used for draw ordering.
    """
    widths: dict[str, float] = {}

    try:
        marge_stats = samples.getMargeStats()
    except Exception:
        marge_stats = None

    for param in ("H0", "S8"):
        width = None
        if marge_stats is not None:
            try:
                par_marge = marge_stats.parWithName(param)
                if par_marge is not None and len(par_marge.limits) > 0:
                    lim_68 = par_marge.limits[0]
                    lower = float(lim_68.lower)
                    upper = float(lim_68.upper)
                    width = upper - lower
            except Exception:
                width = None

        if width is None:
            try:
                std = float(samples.std(param))
                width = 2.0 * std
            except Exception:
                width = None

        if width is None or not np.isfinite(width) or width <= 0.0:
            return math.inf

        widths[param] = float(width)

    return float(widths["H0"] * widths["S8"])


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

    # For H0-S8 plots, sort chains by 68% contour size so that larger
    # contours are drawn first (background) and tighter contours are on top.
    if "H0" in available_params and "S8" in available_params:
        ranked: list[tuple[tuple[str, str, Any], float]] = []
        for entry in samples_by_root:
            _, _, samples = entry
            area = _estimate_h0_s8_1sigma_area(samples)
            ranked.append((entry, area))

        ranked.sort(key=lambda item: item[1], reverse=True)
        samples_by_root = [entry for entry, _ in ranked]

        ordered_roots = [entry[0] for entry in samples_by_root]
        print(
            "H0-S8 layering order (back -> front, largest -> smallest 68% area): "
            + " -> ".join(ordered_roots)
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
    # Use the permanent ROOT_TO_COLOUR mapping, not the sorted order
    chain_colors: list[Color] = [ROOT_TO_COLOUR[root] for root in used_roots]
    # Keep LCDM as contour-only in 2D while filling non-LCDM chains.
    # GetDist supports a per-root boolean sequence for `filled`.
    filled_per_root: list[bool] = ["lcdm" not in root.lower() for root in used_roots]

    line_args = _build_chain_line_args(chain_colors)

    # Generate the triangle plot
    g.triangle_plot(
        roots_to_plot,
        available_params,
        filled=filled_per_root,
        colors=chain_colors,
        line_args=line_args,
        diag1d_kwargs={"colors": chain_colors},
        contour_lws=1.8,
        legend_loc="lower left",
        figure_legend_outside=True,
    )

    # Ensure rendered lines use the same style coding as the legend.
    _apply_chain_styles_to_axes(g, used_roots)

    fig: Any = g.fig
    fig.set_size_inches(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)

    # Build legend handles for MCMC chains
    chain_handles: list[Line2D] = []
    for i, root in enumerate(used_roots):
        linestyle, marker = _style_for_chain(i)
        chain_handles.append(
            Line2D(
                [0],
                [0],
                color=ROOT_TO_COLOUR[root],
                lw=2.0,
                ls=linestyle,
                marker=marker,
                markersize=4,
                label=build_legend_label(root),
            )
        )

    # Apply custom annotations and collect their legend handles
    annotation_handles: list[Patch] = []
    if annotations is not None:
        annotation_handles = annotations(g) or []

    all_handles: list[Any] = chain_handles + annotation_handles
    all_labels: list[str] = [str(h.get_label()) for h in all_handles]

    # Remove any existing legends
    for legend in fig.legends:
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # Adjust layout: plot on left, make room for legend on right
    fig.subplots_adjust(left=0.1, right=0.68)

    # Position legend aligned to top-right of first subplot
    first_ax = g.subplots[0, 0]
    ax_bbox = first_ax.get_position()
    legend = fig.legend(
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
    h0_mean = float(OBSERVATIONAL_REFERENCES["H0"]["mean"])
    h0_sigma = float(OBSERVATIONAL_REFERENCES["H0"]["sigma"])
    s8_mean = float(OBSERVATIONAL_REFERENCES["S8"]["mean"])
    s8_sigma = float(OBSERVATIONAL_REFERENCES["S8"]["sigma"])

    def _send_new_artists_to_background(
        ax_index: int, add_band: Callable[..., Any]
    ) -> None:
        flat_axes = list(np.ravel(g.subplots))
        if ax_index < 0 or ax_index >= len(flat_axes):
            add_band(ax=ax_index)
            return
        ax = flat_axes[ax_index]
        before = {id(artist) for artist in ax.get_children()}
        add_band(ax=ax_index)
        for artist in ax.get_children():
            if id(artist) in before:
                continue
            try:
                artist.set_zorder(-50)
            except AttributeError as e:
                # Some artist types may not support zorder; this is acceptable
                if _DEBUG_EXCEPTIONS:
                    _LOGGER.debug(f"Artist {type(artist).__name__} does not support zorder: {e}")
                continue
            except (TypeError, ValueError) as e:
                # Zorder value may be invalid for this artist type
                _LOGGER.warning(f"Failed to set zorder on {type(artist).__name__}: {e}")
                continue

    # SH0ES 2020b default reference (configurable in OBSERVATIONAL_REFERENCES)
    _send_new_artists_to_background(
        0,
        lambda ax: g.add_x_bands(h0_mean, h0_sigma, ax=ax, color=BAND_COLOURS[0]),
    )
    _send_new_artists_to_background(
        2,
        lambda ax: g.add_x_bands(h0_mean, h0_sigma, ax=ax, color=BAND_COLOURS[0]),
    )

    # KiDS-1000 2023 default reference (configurable in OBSERVATIONAL_REFERENCES)
    _send_new_artists_to_background(
        3,
        lambda ax: g.add_x_bands(s8_mean, s8_sigma, ax=ax, color=BAND_COLOURS[1]),
    )
    _send_new_artists_to_background(
        2,
        lambda ax: g.add_y_bands(s8_mean, s8_sigma, ax=ax, color=BAND_COLOURS[1]),
    )

    return [
        Patch(
            facecolor=BAND_COLOURS[0],
            edgecolor="black",
            hatch="///",
            alpha=0.35,
            label=str(OBSERVATIONAL_REFERENCES["H0"]["label"]),
        ),
        Patch(
            facecolor=BAND_COLOURS[1],
            edgecolor="black",
            hatch="\\\\",
            alpha=0.35,
            label=str(OBSERVATIONAL_REFERENCES["S8"]["label"]),
        ),
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

scf_labels = {
    "cdm_c": r"c_\mathrm{DM}",
    "scf_c2": r"c_2",
    "scf_c3": r"c_3",
    "scf_c4": r"c_4",
}
params_scf = ["cdm_c", "scf_c2", "scf_c3", "scf_c4"]

if CLI_ARGS.skip_plots:
    print("Skipping plots (--skip-plots); generating tables only.")
else:
    strict_export_dir = os.path.abspath(CLI_ARGS.strict_export_dir)
    if CLI_ARGS.strict_export:
        os.makedirs(strict_export_dir, exist_ok=True)

    # ============================================================================
    # PLOT 1: H0 & S8 with observational bands
    # ============================================================================
    # %%
    params_cosmology = ["H0", "S8"]
    g1 = make_triangle_plot(params_cosmology, annotations=annotate_H0_S8, title=None)

    # Interactive preview (current backend):
    # plt.show()
    # Strict IBM Plex Math export for thesis:
    # save_strict_plex_figure(
    #     g1.fig,
    #     "plot_H0_S8_Planck_LCDM_hyperbolic.pdf",
    #     "plot_H0_S8_Planck_LCDM_hyperbolic.pgf",
    # )
    if CLI_ARGS.strict_export:
        pdf_path = os.path.join(
            strict_export_dir, "plot_H0_S8_Planck_LCDM_hyperbolic.pdf"
        )
        pgf_path = os.path.join(
            strict_export_dir, "plot_H0_S8_Planck_LCDM_hyperbolic.pgf"
        )
        save_strict_plex_figure(g1.fig, pdf_path, pgf_path)
        print(f"Strict export saved: {pdf_path}")
        print(f"Strict export saved: {pgf_path}")

    # Apply legend text style fix only for interactive display
    if g1.fig.legends:
        for text in g1.fig.legends[0].get_texts():
            text.set_style("normal")
            text.set_weight("normal")

    plt.show()

    # ============================================================================
    # PLOT 2: Scalar field parameters with constraints
    # ============================================================================
    # %%
    try:
        g2 = make_triangle_plot(
            params_scf,
            annotations=annotate_scf_constraints,
            param_labels=scf_labels,
            title=None,
        )

        # Strict IBM Plex Math export for thesis:
        # save_strict_plex_figure(
        #     g2.fig,
        #     "plot_scf_params_PP_S_DESI_hyperbolic.pdf",
        #     "plot_scf_params_PP_S_DESI_hyperbolic.pgf",
        # )
        if CLI_ARGS.strict_export:
            pdf_path = os.path.join(
                strict_export_dir, "plot_scf_params_PP_S_DESI_hyperbolic.pdf"
            )
            pgf_path = os.path.join(
                strict_export_dir, "plot_scf_params_PP_S_DESI_hyperbolic.pgf"
            )
            save_strict_plex_figure(g2.fig, pdf_path, pgf_path)
            print(f"Strict export saved: {pdf_path}")
            print(f"Strict export saved: {pgf_path}")

        # Apply legend text style fix only for interactive display
        if g2.fig.legends:
            for text in g2.fig.legends[0].get_texts():
                text.set_style("normal")
                text.set_weight("normal")

        plt.show()
    except ValueError as e:
        print(f"Skipping scalar field parameters plot: {e}")

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
# Keys are stored in lowercase to match the case-insensitive lookup logic
LIKELIHOOD_DATA_POINTS: dict[str, int] = {
    "bao.desi_dr2": 13,  # DESI DR2 BAO (13 data points)
    "sn.pantheonplus": 1701,  # Pantheon+ without SH0ES (1701 SNe)
    "sn.pantheonplusshoes": 1701,  # Pantheon+ with SH0ES calibration (1701 SNe + SH0ES prior)
    "planck_2018_lowl.tt": 28,  # Low-l TT (l=2-29)
    "planck_2018_lowl.ee": 28,  # Low-l EE (l=2-29)
    "planck_2018_highl_plik.ttteee_lite_native": 613,  # Planck 2018 high-l TT, TE, EE (lite version with 613 data points)
    "planck_2018_lensing.native": 9,  # Planck 2018 lensing (9 data points)
    "act_dr6_cmbonly.planckactcut": 613,  # Combined Planck+ACT DR6 CMB-only (613 data points)
    "act_dr6_cmbonly.actdr6cmbonly": 135,  # ACT DR6 CMB-only (135 data points)
    "act_dr6_lenslike.actdr6lenslike": 19,  # ACT DR6 lensing-like (19 data points)
    "spt3g_d1_tne": 196,  # SPT-3G D1 temperature and E-mode polarization (196 data points)
    "muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse": 16,  # MUSE-3G SPT-3G phi-phi component (16 data points)
}


def estimate_n_data_from_likelihoods(
    chi_sq_components: Mapping[str, float] | None,
    root: str | None = None,
) -> int | None:
    """
    Estimate the total number of data points from the chi-squared components.

    Parameters
    ----------
    chi_sq_components : dict
        Dictionary mapping likelihood names to chi-squared values.
    root : str, optional
        Chain root name used to enforce dataset-specific corrections when
        likelihood components are incomplete.

    Returns
    -------
    int or None
        Estimated number of data points, or None if unknown likelihoods.
    """
    if not chi_sq_components:
        return None

    total_n_data: int = 0
    unknown_likelihoods: list[str] = []
    likelihood_keys_lower = [
        likelihood.lower() for likelihood in chi_sq_components.keys()
    ]

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

    # Some post-processed PP/PPS chain names may not explicitly list the BAO
    # likelihood key in the chi2 components. Enforce: Pantheon+ implies DESI DR2.
    has_any_desi_like = any("desi" in key for key in likelihood_keys_lower)
    if root is not None:
        dataset_flags = infer_dataset_flags_from_root(root)
        if dataset_flags["has_desi"] and not has_any_desi_like:
            total_n_data += LIKELIHOOD_DATA_POINTS["bao.desi_dr2"]
            print(
                f"Note: Added DESI DR2 data points (13) for {root} "
                "because PP/PPS chains implicitly include DESI."
            )

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

    # Compute marginalized stats once to avoid repeated expensive calls and warning spam.
    marge_stats = None
    if use_getdist and samples is not None:
        try:
            marge_stats = samples.getMargeStats()
        except Exception as e_marge:
            print(
                f"Note: Marginalized stats failed for {root}; using direct weighted samples fallback: {e_marge}"
            )
            use_getdist = False
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

            # Get confidence limits from cached marginalized stats.
            par_marge = (
                marge_stats.parWithName(param) if marge_stats is not None else None
            )

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
) -> int | None:
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
        try:
            samples = loadMCSamples(
                _root_base_path(resolved_root, chain_dir), settings=settings
            )
            _SAMPLES_CACHE[cache_key] = samples
        except Exception as e:
            print(
                f"Note: Skipping {root} in tables because free parameters could not be loaded: {e}"
            )
            _SAMPLES_CACHE[cache_key] = None
            return None

    if samples is None or getattr(samples, "paramNames", None) is None:
        print(
            f"Note: Skipping {root} in tables because parameter names are unavailable."
        )
        return None
    # paramNames.names includes all parameters; we want only sampled ones
    # The .paramNames.list() returns sampled params, .paramNames.numberOfName() for derived
    return len([p for p in samples.paramNames.names if not p.isDerived])


def _has_any_valid_stats(
    stats: Mapping[str, Any],
    params: Sequence[str],
) -> bool:
    """Return True if at least one requested parameter has usable statistics."""
    return any(stats.get(param) is not None for param in params)


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

    dataset_flags = infer_dataset_flags_from_root(root)

    # Build dataset key
    parts: list[str] = []
    if dataset_flags["has_planck"]:
        parts.append("Planck")
    if dataset_flags["has_desi"]:
        parts.append("DESI")
    if dataset_flags["has_pantheon"]:
        parts.append("PP")
    if dataset_flags["has_sh0es"]:
        parts.append("SH0ES")

    dataset_key = "+".join(parts) if parts else "Unknown"

    # Identify model type and get LaTeX name
    is_lcdm = "lcdm" in root_lower
    if is_lcdm:
        model_name = r"\gls{lcdm}"
    elif "hyperbolic" in root_lower and "tracking" in root_lower:
        model_name = r"\Nref{pot:tanh} (tracking)"
    elif "doubleexp" in root_lower or "dexp" in root_lower:
        model_name = r"\Nref{pot:dexp}"
    elif "hyperbolic" in root_lower or "tanh" in root_lower:
        model_name = r"\Nref{pot:tanh}"
    elif (
        "beanads" in root_lower
        or "bean_" in root_lower
        or root_lower.startswith("bean")
    ):
        model_name = r"\Nref{pot:bexp}"
    elif "cosine" in root_lower:
        model_name = r"\Nref{pot:cosine}"
    elif "exponential" in root_lower:
        model_name = r"\Nref{pot:exp}"
    elif "png" in root_lower:
        model_name = r"\Nref{pot:pNG}"
    elif re.search(r"power.?law", root_lower):
        model_name = r"\Nref{pot:PL}"
    elif "sqe" in root_lower:
        model_name = r"\Nref{pot:sqexp}"
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
    lower_bound = min(lower, upper)
    upper_bound = max(lower, upper)
    err_up = abs(upper_bound - mean)
    err_down = abs(mean - lower_bound)

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


def format_plain_number(value: float | None, precision: int = 2) -> str:
    """Format a scalar number for table cells (without math delimiters)."""
    if value is None:
        return "--"
    return f"{value:.{precision}f}"


def compute_tension(
    model_value: float | None,
    model_sigma: float | None,
    reference_value: float,
    reference_sigma: float,
) -> float | None:
    """Compute Gaussian tension in sigma units."""
    if model_value is None or model_sigma is None:
        return None
    denom = float(np.sqrt(model_sigma**2 + reference_sigma**2))  # type: ignore[no-untyped-call]
    if denom <= 0:
        return None
    return (model_value - reference_value) / denom


def get_accepted_steps(
    root: str,
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> int | None:
    """Estimate accepted steps from .progress (fallback: chain weights)."""
    try:
        resolved_root = resolve_chain_root(root, chain_dir)
        progress_file = _root_base_path(resolved_root, chain_dir) + ".progress"

        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                lines = f.readlines()

            # Use last non-empty, non-comment row: N timestamp acceptance_rate ...
            for raw_line in reversed(lines):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    n_steps = float(parts[0])
                    acceptance_rate = float(parts[2])
                    return int(round(n_steps * acceptance_rate))
                except ValueError as e:
                    # Line format may not match expected pattern; log and skip
                    if _DEBUG_EXCEPTIONS:
                        _LOGGER.debug(f"Malformed stats line in {root}: {line.strip()}: {e}")
                    continue

        chain_data = read_chain_data_directly(root, [], chain_dir, settings)
        weights = chain_data.get("weights")
        if weights is None:
            return None
        return int(round(float(np.sum(weights))))  # type: ignore[no-untyped-call]
    except Exception as e:
        print(f"Note: Could not estimate accepted steps for {root}: {e}")
        return None


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
        if n_params is None or not _has_any_valid_stats(stats, params):
            print(
                f"Note: Skipping {root} in cosmology table because the chain could not be loaded reliably."
            )
            continue

        chi_sq = min_data["chi_sq"]

        # Estimate n_data from chi2 components if not provided
        chain_n_data = n_data
        if chain_n_data is None and min_data.get("chi_sq_components"):
            chain_n_data = estimate_n_data_from_likelihoods(
                min_data["chi_sq_components"], root=root
            )
            if chain_n_data:
                print(
                    f"  {root}: Estimated N_data = {chain_n_data} from likelihoods: {list(min_data['chi_sq_components'].keys())}"
                )

        aic, bic = (
            compute_aic_bic(chi_sq, n_params, chain_n_data)
            if chi_sq is not None
            else (None, None)
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

    if not chain_data:
        return "% No chains with usable data were available for the cosmology table."

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
        if (
            lcdm_root
            and lcdm_root in chain_data
            and chain_data[lcdm_root]["aic"] is not None
        ):
            baseline_aic = chain_data[lcdm_root]["aic"]
            baseline_bic = chain_data[lcdm_root]["bic"]
        else:
            baseline_aic = None
            baseline_bic = None

        if not lcdm_root and not group["others"]:
            continue

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
        if lcdm_root and lcdm_root in chain_data:
            group_roots.append(lcdm_root)
        group_roots.extend(root for root in group["others"] if root in chain_data)

        if not group_roots:
            continue

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
            if data["chi_sq"] is not None:
                row_parts.append(f"{data['chi_sq']:.2f}")
            else:
                row_parts.append("--")

            # ΔAIC
            if data["aic"] is not None and baseline_aic is not None:
                delta_aic = data["aic"] - baseline_aic
                row_parts.append(f"{delta_aic:+.2f}")
            else:
                row_parts.append("--")

            # ΔBIC
            if data["bic"] is not None and baseline_bic is not None:
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
    dataset_flags = infer_dataset_flags_from_root(root)

    parts: list[str] = []
    if dataset_flags["has_planck"]:
        parts.append("Planck")
    if dataset_flags["has_desi"]:
        parts.append("DESI DR2")
    if dataset_flags["has_pantheon"]:
        parts.append("PP")
    if dataset_flags["has_sh0es"]:
        parts.append("SH0ES")

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
            if not _has_any_valid_stats(stats, params):
                print(
                    f"Note: Skipping {root} in scalar field table because the chain could not be loaded reliably."
                )
                continue
            chain_data[root] = {"minimum": min_data, "stats": stats}

        if not chain_data:
            continue

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
            if root not in chain_data:
                continue
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

    def _is_chain_text_file(path: str) -> bool:
        """Return True only for raw chain segment files like root.<int>.txt."""
        basename = os.path.basename(path)
        return re.match(r"^.+\.\d+\.txt$", basename) is not None

    # Find all chain segment files matching the root.
    # Exclude text artifacts such as *.bestfit.txt which are not chain streams.
    pattern = f"{base_path}.*.txt"
    chain_files = sorted(p for p in glob.glob(pattern) if _is_chain_text_file(p))

    if not chain_files:
        # Try without the .* pattern (single file)
        single_file = f"{base_path}.txt"
        if os.path.exists(single_file):
            chain_files = [single_file]
        else:
            fallback = glob.glob(
                os.path.join(chain_dir, f"**/{root}.*.txt"), recursive=True
            )
            chain_files = sorted(p for p in fallback if _is_chain_text_file(p))
            if not chain_files:
                raise FileNotFoundError(f"No chain files found for root {root}")

    all_data: list[Any] = []
    reference_col_names: list[str] | None = None
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

            # Validate header consistency and build column index map once.
            if reference_col_names is None:
                reference_col_names = col_names
                for param in param_names:
                    if param in col_names:
                        param_indices[param] = col_names.index(param)

                if "weight" in col_names:
                    weight_idx = col_names.index("weight")
            elif col_names != reference_col_names:
                print(
                    "Warning: Skipping file with inconsistent header ordering "
                    f"for {root}: {chain_file}"
                )
                continue

            # Read numerical data
            data_lines = lines[data_start_idx:]
            chain_data: list[Any] = []
            for line in data_lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) != len(col_names):
                        continue
                    chain_data.append(values)
                except ValueError as e:
                    # Line contains non-numeric values or format mismatch; log and skip
                    if _DEBUG_EXCEPTIONS:
                        _LOGGER.debug(f"Malformed data line in {chain_file}: {line.strip()}: {e}")
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
    except Exception:
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
    except Exception:
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

    # Filter for non-LCDM chains that actually contain swampland parameters.
    # Do not rely on legacy naming tags such as ".post.swampland".
    swampland_roots: list[str] = []
    for root in roots:
        _, _, is_lcdm = identify_dataset_from_root(root)
        if is_lcdm:
            continue
        samples = get_samples_for_root(root, chain_dir, settings)
        if samples is None or getattr(samples, "paramNames", None) is None:
            continue
        if any(samples.paramNames.parWithName(p) is not None for p in swampland_params):
            swampland_roots.append(root)

    if not swampland_roots:
        return "% No swampland constraint chains found."

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
        if not _has_any_valid_stats(stats, swampland_params):
            print(
                f"Note: Skipping {root} in swampland table because the chain could not be loaded reliably."
            )
            continue

        # Extract mode for integer parameter
        if "attractor_regime_scf" in swampland_params:
            mean_val, mode_val = get_integer_parameter_mode(
                root, "attractor_regime_scf", chain_dir, settings
            )
            if mean_val is not None and mode_val is not None:
                integer_modes[root] = (mean_val, mode_val)

        chain_data[root] = {"minimum": min_data, "stats": stats}

    if not chain_data:
        return "% No swampland chains with usable data were available."

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

    # Pivoted layout: parameters as rows, model configurations as columns.
    ordered_roots: list[str] = []
    for dataset_key in sorted(dataset_model_groups.keys()):
        models_dict = dataset_model_groups[dataset_key]
        for model_type in ["hyperbolic", "dexp"]:
            for root in models_dict[model_type]:
                if root in chain_data:
                    ordered_roots.append(root)

    if not ordered_roots:
        return "% No swampland chains with usable data were available."

    n_models = len(ordered_roots)
    # Use plain centered columns for maximum LaTeX robustness in this table.
    col_spec = "l" + " c" * n_models

    def _table3_decimals(
        mean: float, err_up: float, err_down: float, precision: int
    ) -> int:
        # Keep enough decimals for small values while staying compact for O(1) scales.
        magnitudes = [abs(v) for v in (mean, err_up, err_down) if abs(v) > 0]
        if not magnitudes:
            return precision
        if min(magnitudes) < 1e-2:
            return max(precision, 6)
        return precision

    def _fmt_table3_value(
        mean: float, lower: float, upper: float, precision: int = 2
    ) -> str:
        lower_bound = min(lower, upper)
        upper_bound = max(lower, upper)
        err_up = abs(upper_bound - mean)
        err_down = abs(mean - lower_bound)
        mean_str = f"{mean:.{precision}f}"
        up_str = f"{err_up:.{precision}f}"
        down_str = f"{err_down:.{precision}f}"

        if abs(err_up - err_down) < 0.05 * max(abs(err_up), abs(err_down), 0.001):
            avg_err = (err_up + err_down) / 2
            avg_str = f"{avg_err:.{precision}f}"
            return rf"${mean_str} \pm {avg_str}$"
        return rf"${{{mean_str}}}^{{+{up_str}}}_{{-{down_str}}}$"

    lines: list[str] = []
    lines.append(r"% Requires \\usepackage{graphicx} for \\rotatebox")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Swampland constraint parameters from MCMC analysis.}")
    lines.append(r"\label{tab:swampland_params}")
    lines.append(r"\tagpdfsetup{table/header-rows={1}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Rotated headers with potential + likelihood information.
    header_parts: list[str] = ["Parameter"]
    for root in ordered_roots:
        header_label = build_legend_label(root)
        header_parts.append(
            r"\multicolumn{1}{c}{\rotatebox[origin=c]{90}{\parbox{3.8cm}{\centering "
            + header_label
            + r"}}}"
        )
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # One row per parameter, one column per model.
    for param in swampland_params:
        row_parts: list[str] = [r"$" + param_latex_labels.get(param, param) + r"$"]
        for root in ordered_roots:
            data = chain_data[root]
            if param == "attractor_regime_scf":
                # Keep integer parameter readable by reporting weighted mean and mode.
                if root in integer_modes:
                    mean_val, mode_val = integer_modes[root]
                    row_parts.append(rf"{mean_val:.2f} (mode:{mode_val})")
                else:
                    row_parts.append("--")
            elif data["stats"] and data["stats"].get(param):
                s = data["stats"][param]
                row_parts.append(
                    _fmt_table3_value(
                        s["mean"],
                        s["lower_1sigma"],
                        s["upper_1sigma"],
                        precision=2,
                    )
                )
            else:
                row_parts.append("--")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_tension_summary_table(
    roots: Sequence[str],
    chain_dir: str = CHAIN_DIR,
    settings: Mapping[str, Any] = ANALYSIS_SETTINGS,
) -> str:
    """
    Generate a LaTeX table with model-comparison metrics and tensions.

    Columns:
    Potential, dataset, Delta chi^2, Delta AIC, Delta BIC,
    best-fit H0, H0 tension, best-fit S8, S8 tension.
    """
    h0_ref = float(OBSERVATIONAL_REFERENCES["H0"]["mean"])
    h0_ref_sigma = float(OBSERVATIONAL_REFERENCES["H0"]["sigma"])
    s8_ref = float(OBSERVATIONAL_REFERENCES["S8"]["mean"])
    s8_ref_sigma = float(OBSERVATIONAL_REFERENCES["S8"]["sigma"])

    row_data: list[dict[str, Any]] = []

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

        stats = get_chain_statistics(root, ["H0", "S8"], chain_dir, settings)
        n_params = count_free_parameters(root, chain_dir, settings)
        if n_params is None:
            print(
                f"Note: Skipping {root} in tension summary table because free parameters are unavailable."
            )
            continue

        chi_sq = min_data.get("chi_sq")
        n_data = estimate_n_data_from_likelihoods(
            min_data.get("chi_sq_components"), root=root
        )
        aic, bic = (
            compute_aic_bic(float(chi_sq), n_params, n_data)
            if chi_sq is not None
            else (None, None)
        )

        bestfit_h0 = None
        bestfit_s8 = None
        if min_data.get("params"):
            if "H0" in min_data["params"]:
                bestfit_h0 = float(min_data["params"]["H0"]["value"])
            if "S8" in min_data["params"]:
                bestfit_s8 = float(min_data["params"]["S8"]["value"])

        h0_sigma_model = (
            float(stats["H0"]["std"])
            if stats.get("H0") and stats["H0"].get("std") is not None
            else None
        )
        s8_sigma_model = (
            float(stats["S8"]["std"])
            if stats.get("S8") and stats["S8"].get("std") is not None
            else None
        )

        h0_tension = compute_tension(bestfit_h0, h0_sigma_model, h0_ref, h0_ref_sigma)
        s8_tension = compute_tension(bestfit_s8, s8_sigma_model, s8_ref, s8_ref_sigma)

        dataset_key, model_name, is_lcdm = identify_dataset_from_root(root)
        row_data.append(
            {
                "root": root,
                "dataset_key": dataset_key,
                "model_name": model_name,
                "is_lcdm": is_lcdm,
                "chi_sq": float(chi_sq) if chi_sq is not None else None,
                "aic": aic,
                "bic": bic,
                "bestfit_h0": bestfit_h0,
                "h0_tension": h0_tension,
                "bestfit_s8": bestfit_s8,
                "s8_tension": s8_tension,
            }
        )

    if not row_data:
        return (
            "% No chains with usable data were available for the tension summary table."
        )

    # Build a per-dataset LCDM baseline so deltas are compared within each
    # observational dataset combination.
    dataset_baseline: dict[str, dict[str, Any]] = {}
    for dataset_key in sorted({str(r["dataset_key"]) for r in row_data}):
        lcdm_candidates = [
            r for r in row_data if r["is_lcdm"] and str(r["dataset_key"]) == dataset_key
        ]
        if not lcdm_candidates:
            continue
        # If multiple LCDM chains map to the same dataset, use the best-fit
        # (minimum chi^2 when available) as baseline.
        lcdm_candidates.sort(
            key=lambda r: (
                float("inf") if r.get("chi_sq") is None else float(r["chi_sq"]),
                str(r["root"]),
            )
        )
        dataset_baseline[dataset_key] = lcdm_candidates[0]

    for row in row_data:
        baseline = dataset_baseline.get(str(row["dataset_key"]))
        baseline_chi_sq = baseline["chi_sq"] if baseline else None
        baseline_aic = baseline["aic"] if baseline else None
        baseline_bic = baseline["bic"] if baseline else None

        row["dchi2"] = (
            row["chi_sq"] - baseline_chi_sq
            if baseline_chi_sq is not None and row["chi_sq"] is not None
            else None
        )
        row["daic"] = (
            row["aic"] - baseline_aic
            if baseline_aic is not None and row["aic"] is not None
            else None
        )
        row["dbic"] = (
            row["bic"] - baseline_bic
            if baseline_bic is not None and row["bic"] is not None
            else None
        )

    col_spec = (
        "l l" + " >{$}c<{$} >{$}c<{$} >{$}c<{$} >{$}c<{$} >{$}c<{$} >{$}c<{$} >{$}c<{$}"
    )
    lines: list[str] = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Model comparison metrics and observational tensions. "
        + r"Tensions are computed with references "
        + rf"$H_0={h0_ref:.1f}\pm{h0_ref_sigma:.1f}$ and $S_8={s8_ref:.3f}\pm{s8_ref_sigma:.3f}$."
        + r" Delta metrics are computed relative to the $\Lambda$CDM chain with the same dataset."
        + r"}"
    )
    lines.append(r"\label{tab:model_tension_summary}")
    lines.append(r"\tagpdfsetup{table/header-rows={1}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_parts = ["Potential", "Dataset"]
    header_parts.extend(
        [
            r"\Delta\chi^2",
            r"\Delta\text{AIC}",
            r"\Delta\text{BIC}",
            r"H_0^{\mathrm{bf}}",
            r"T(H_0)\,[\sigma]",
            r"S_8^{\mathrm{bf}}",
            r"T(S_8)\,[\sigma]",
        ]
    )
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    dataset_keys_sorted = sorted({str(r["dataset_key"]) for r in row_data})
    for dataset_idx, dataset_key in enumerate(dataset_keys_sorted):
        dataset_rows = [r for r in row_data if str(r["dataset_key"]) == dataset_key]
        dataset_rows_sorted = sorted(
            dataset_rows,
            key=lambda r: (
                0 if r["is_lcdm"] else 1,
                float("inf") if r.get("daic") is None else float(r["daic"]),
                (
                    float("inf")
                    if r.get("h0_tension") is None
                    else abs(float(r["h0_tension"]))
                ),
                str(r["model_name"]),
            ),
        )

        if dataset_idx > 0:
            lines.append(r"\midrule")

        for row in dataset_rows_sorted:
            dchi2 = row.get("dchi2")
            daic = row.get("daic")
            dbic = row.get("dbic")

            dchi2_str = "--" if dchi2 is None else f"{dchi2:+.2f}"
            daic_str = "--" if daic is None else f"{daic:+.2f}"
            dbic_str = "--" if dbic is None else f"{dbic:+.2f}"

            row_parts = [
                str(row["model_name"]),
                str(row["dataset_key"]),
            ]
            row_parts.extend(
                [
                    dchi2_str,
                    daic_str,
                    dbic_str,
                    format_plain_number(row["bestfit_h0"], precision=2),
                    format_plain_number(row["h0_tension"], precision=2),
                    format_plain_number(row["bestfit_s8"], precision=3),
                    format_plain_number(row["s8_tension"], precision=2),
                ]
            )
            lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")

    return "\n".join(lines)


# ============================================================================
# Generate tables for the current analysis
# ============================================================================
if CLI_ARGS.skip_tables:
    print("Skipping tables (--skip-tables); generating plots only.")
else:
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

    # %%
    # Table 4: Model comparison with tensions
    print("\n" + "=" * 80)
    print("TABLE 4: Model Comparison and Tensions")
    print("=" * 80)
    tension_summary_table = generate_tension_summary_table(
        ROOTS,
        chain_dir=CHAIN_DIR,
        settings=ANALYSIS_SETTINGS,
    )
    print(tension_summary_table)
