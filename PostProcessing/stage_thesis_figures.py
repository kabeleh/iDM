#!/usr/bin/env python3
"""
Stage thesis figures from PostProcessing sources to a mirrored structure.

Purpose: build PostProcessing/thesis_figures/figures/ that exactly mirrors the
thesis figures/ layout for safe rsyncing without renaming/editing.

Sources:
  - PostProcessing/PosteriorHeatmaps/: heatmap_<quantity>_<root>.{png,pdf,pgf}
    and companions, plus legend_<preset>.{png,pdf,pgf}
  - PostProcessing/Plots/: <root>_plot[1-6]_*.* and plot_*.{png,pdf,pgf}

Routes files by basename pattern to destination subdirs (Omega, EoS, phi, dSC,
SWGC, Pk, Cl, or root). Idempotent: copies only on missing or sha256 diff,
preserves mtimes. Detects and ignores old heatmap naming scheme and debris.
Verifies against thesis .tex files in --check mode.
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/home/kl/kDrive/Sci/PhD/Research/HDM/class_public")
SOURCE_HEATMAPS = BASE_DIR / "PostProcessing" / "PosteriorHeatmaps"
SOURCE_PLOTS = BASE_DIR / "PostProcessing" / "Plots"
STAGING_ROOT = BASE_DIR / "PostProcessing" / "thesis_figures" / "figures"
THESIS_ROOT = Path("/home/kl/kDrive/Sci/PhD/Research/HDM/LaTeX/20260504")


# ============================================================================
# Data classes for reporting
# ============================================================================

@dataclass
class StagingStats:
    staged: int = 0
    up_to_date: int = 0
    debris: int = 0
    unmatched: List[str] = None
    debris_list: List[str] = None

    def __post_init__(self):
        if self.unmatched is None:
            self.unmatched = []
        if self.debris_list is None:
            self.debris_list = []


# ============================================================================
# Routing logic (order matters for precedence)
# ============================================================================

def route_file(basename: str) -> Optional[str]:
    """
    Route a filename to its destination subdirectory under thesis_figures/figures/.

    Returns None if file doesn't match any rule (unmatched).
    Returns "." for root of figures/.
    Returns subdirectory name (e.g., "Omega", "EoS") otherwise.

    Order matters: check more specific patterns first.
    """

    # === Heatmap patterns (order by specificity) ===

    # _heatmap_swgc_lhs*, _heatmap_swgc_rhs* (must come before generic _heatmap_swgc)
    if re.search(r'_heatmap_swgc_(lhs|rhs)', basename):
        return "SWGC"

    # _heatmap_swgc* (generic, matches _heatmap_swgc followed by anything or nothing)
    if re.search(r'_heatmap_swgc', basename):
        return "SWGC"

    # _heatmap_dSC_expr* (must come before generic _heatmap_dSC)
    if re.search(r'_heatmap_dSC_expr', basename):
        return "dSC"

    # _heatmap_dSC* (generic, matches _heatmap_dSC followed by anything or nothing)
    if re.search(r'_heatmap_dSC', basename):
        return "dSC"

    # _heatmap_phi*
    if re.search(r'_heatmap_phi', basename):
        return "phi"

    # _heatmap_w* (must not match _heatmap_swgc or _heatmap_swgc_*)
    # This should only match _heatmap_w followed by not 'sgc'
    if re.search(r'_heatmap_w(?!.*sgc)', basename):
        return "EoS"

    # _heatmap_Omega* (includes _heatmap_Omega_cdm, _heatmap_Omega_k, etc.)
    if re.search(r'_heatmap_Omega', basename):
        return "Omega"

    # === Legend patterns ===

    if re.match(r'^legend_omega\.[a-z]+$', basename, re.IGNORECASE):
        return "Omega"

    if re.match(r'^legend_eos\.[a-z]+$', basename, re.IGNORECASE):
        return "EoS"

    if re.match(r'^legend_phi\.[a-z]+$', basename, re.IGNORECASE):
        return "phi"

    if re.match(r'^legend_swampland\.[a-z]+$', basename, re.IGNORECASE):
        return "dSC"

    if re.match(r'^legend_swgc\.[a-z]+$', basename, re.IGNORECASE):
        return "SWGC"

    # === Plot patterns (order by specificity) ===

    # *_plot6_swgc*
    if re.search(r'_plot6_swgc', basename):
        return "SWGC"

    # *_plot5_cl_lensed*
    if re.search(r'_plot5_cl_lensed', basename):
        return "Cl"

    # *_plot4_pk*
    if re.search(r'_plot4_pk', basename):
        return "Pk"

    # *_plot3_swampland_w*
    if re.search(r'_plot3_swampland_w', basename):
        return "EoS"

    # *_plot2_omegas*
    if re.search(r'_plot2_omegas', basename):
        return "Omega"

    # *_plot1_ds*
    if re.search(r'_plot1_ds', basename):
        return "dSC"

    # === Getdist exports (plot_*.{png,pdf,pgf}) ===

    if re.match(r'^plot_[a-zA-Z0-9_]+\.(png|pdf|pgf)$', basename):
        return "."

    # Not matched
    return None


def is_old_heatmap_naming(basename: str) -> bool:
    """
    Detect files with OLD naming scheme: heatmap_<quantity>_<root>.*
    (quantity BEFORE root, e.g., heatmap_Omega_cdm_*, heatmap_w_*, etc.)

    Also catches variants like heatmap_minus_s2_*, heatmap_s1_*, heatmap_swampland_expr_*
    """
    # Old pattern: starts with heatmap_ followed by any quantity-like token (may include digits)
    # Examples: heatmap_Omega_cdm, heatmap_w_, heatmap_minus_s2_, heatmap_s1_, heatmap_swampland_expr_
    if re.match(r'^heatmap_[a-zA-Z0-9_]+_', basename):
        return True
    return False


def is_debris(basename: str) -> bool:
    """
    Identify known debris to ignore.
    """
    # *.npz files are debris
    if basename.endswith('.npz'):
        return True

    # *.dat files (from getdist exports)
    if basename.endswith('.dat'):
        return True

    # *.processed.npz files
    if basename.endswith('.processed.npz'):
        return True

    # .ini config files (autogen)
    if basename.endswith('.ini'):
        return True

    # Old naming scheme
    if is_old_heatmap_naming(basename):
        return True

    return False


# ============================================================================
# File operations
# ============================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_copy(src: Path, dst: Path) -> bool:
    """
    Determine if a file should be copied.

    Returns True if destination missing or content differs (by sha256).
    """
    if not dst.exists():
        return True

    return sha256_file(src) != sha256_file(dst)


def stage_file(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """
    Copy file from src to dst, preserving mtime.

    Returns True if copied, False if skipped (already up-to-date).
    """
    if not should_copy(src, dst):
        return False

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    return True


# ============================================================================
# File collection
# ============================================================================

def collect_source_files(
    heatmap_dir: Path,
    plots_dir: Path
) -> Tuple[List[Path], List[Path]]:
    """
    Collect all non-debris files from source directories.

    Returns (heatmap_files, plot_files) where each is a list of Path objects.
    """
    heatmap_files = []
    plot_files = []

    # Collect heatmaps and legends
    if heatmap_dir.exists():
        for item in heatmap_dir.iterdir():
            if item.is_file() and not is_debris(item.name):
                heatmap_files.append(item)

    # Collect plots
    if plots_dir.exists():
        for item in plots_dir.iterdir():
            if item.is_file() and not is_debris(item.name):
                plot_files.append(item)

    return heatmap_files, plot_files


def collect_debris(
    heatmap_dir: Path,
    plots_dir: Path
) -> List[Path]:
    """
    Collect all debris files (old naming, .npz, .dat, etc.).
    """
    debris = []

    if heatmap_dir.exists():
        for item in heatmap_dir.iterdir():
            if item.is_file() and is_debris(item.name):
                debris.append(item)

    if plots_dir.exists():
        for item in plots_dir.iterdir():
            if item.is_file() and is_debris(item.name):
                debris.append(item)

    return debris


# ============================================================================
# Main staging logic
# ============================================================================

def stage_all(dry_run: bool = False) -> StagingStats:
    """
    Stage all files from sources to destination.

    Returns StagingStats with summary counts and unmatched file list.
    """
    stats = StagingStats()

    # Collect source files
    heatmap_files, plot_files = collect_source_files(SOURCE_HEATMAPS, SOURCE_PLOTS)
    debris = collect_debris(SOURCE_HEATMAPS, SOURCE_PLOTS)

    # Record debris
    stats.debris = len(debris)
    stats.debris_list = [p.name for p in debris]

    # Process all files
    all_files = heatmap_files + plot_files

    for src_path in all_files:
        basename = src_path.name
        dest_subdir = route_file(basename)

        if dest_subdir is None:
            # Unmatched
            stats.unmatched.append(basename)
        else:
            # Route to destination
            if dest_subdir == ".":
                dst_path = STAGING_ROOT / basename
            else:
                dst_path = STAGING_ROOT / dest_subdir / basename

            # Stage the file
            if stage_file(src_path, dst_path, dry_run):
                stats.staged += 1
            else:
                stats.up_to_date += 1

    return stats


# ============================================================================
# Check mode: verify against thesis .tex files
# ============================================================================

def parse_thesis_references() -> Set[str]:
    r"""
    Parse thesis .tex files for figure references.

    Looks for:
      \HeatmapInclude{figures/Omega/xyz_heatmap_Omega_cdm}
      \PlotInclude{figures/dSC/xyz_plot1_ds}
      \StandalonePlotInclude{figures/Cl/xyz_plot5_cl_lensed}
      \input{figures/SWGC/xyz_plot6_swgc.pgf}

    Extracts basenames (without extension) and returns set of referenced basenames.
    """
    references = set()

    if not THESIS_ROOT.exists():
        print(f"WARNING: Thesis root not found: {THESIS_ROOT}")
        return references

    # Find all .tex files in figures/ subdirectories
    figures_dir = THESIS_ROOT / "figures"
    if not figures_dir.exists():
        print(f"WARNING: Thesis figures dir not found: {figures_dir}")
        return references

    for tex_file in figures_dir.rglob("*.tex"):
        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract \HeatmapInclude{figures/...}
        for match in re.finditer(r'\\HeatmapInclude\{figures/[^/]+/([^}]+)\}', content):
            ref = match.group(1)
            references.add(ref)

        # Extract \PlotInclude{figures/...}
        for match in re.finditer(r'\\PlotInclude\{figures/[^/]+/([^}]+)\}', content):
            ref = match.group(1)
            references.add(ref)

        # Extract \StandalonePlotInclude{figures/...}
        for match in re.finditer(r'\\StandalonePlotInclude\{figures/[^/]+/([^}]+)\}', content):
            ref = match.group(1)
            references.add(ref)

        # Extract \input{figures/...}
        for match in re.finditer(r'\\input\{figures/[^/]+/([^}]+)\}', content):
            ref = match.group(1)
            references.add(ref)

    return references


def check_references() -> List[str]:
    """
    Check that all thesis references exist in staging tree as both .pdf and .pgf.

    Returns list of missing references (basename, expected as both .pdf and .pgf).
    """
    references = parse_thesis_references()
    missing = []

    for ref in sorted(references):
        # ref is basename without extension
        # Check if both .pdf and .pgf exist in staging tree
        found_pdf = False
        found_pgf = False

        for subdir in STAGING_ROOT.rglob("*"):
            if subdir.is_file():
                if subdir.name == f"{ref}.pdf":
                    found_pdf = True
                elif subdir.name == f"{ref}.pgf":
                    found_pgf = True

        if not found_pdf or not found_pgf:
            missing.append(ref)

    return sorted(missing)


# ============================================================================
# Unit tests for routing
# ============================================================================

def test_routing():
    """
    Unit-style tests for routing logic.

    Tests tricky cases:
      - _heatmap_w vs _heatmap_swgc vs _heatmap_swgc_lhs
      - _heatmap_dSC_s1 vs _heatmap_dSC_expr
    """
    test_cases = [
        # Heatmaps
        ("xyz_heatmap_Omega_cdm.pdf", "Omega"),
        ("xyz_heatmap_Omega_k.pdf", "Omega"),
        ("xyz_heatmap_w.pdf", "EoS"),
        ("xyz_heatmap_w.png", "EoS"),
        ("xyz_heatmap_phi.pdf", "phi"),
        ("xyz_heatmap_dSC_s1.pdf", "dSC"),
        ("xyz_heatmap_dSC_expr.pdf", "dSC"),
        ("xyz_heatmap_swgc.pdf", "SWGC"),
        ("xyz_heatmap_swgc_lhs.pdf", "SWGC"),
        ("xyz_heatmap_swgc_rhs.pdf", "SWGC"),

        # Legends
        ("legend_omega.pdf", "Omega"),
        ("legend_omega.png", "Omega"),
        ("legend_eos.pdf", "EoS"),
        ("legend_phi.pdf", "phi"),
        ("legend_swampland.pdf", "dSC"),
        ("legend_swgc.pdf", "SWGC"),

        # Plots
        ("xyz_plot1_ds.pdf", "dSC"),
        ("xyz_plot2_omegas.pdf", "Omega"),
        ("xyz_plot3_swampland_w.pdf", "EoS"),
        ("xyz_plot4_pk.pdf", "Pk"),
        ("xyz_plot5_cl_lensed.pdf", "Cl"),
        ("xyz_plot6_swgc.pdf", "SWGC"),

        # Getdist
        ("plot_scf_params.pdf", "."),
        ("plot_H0_S8_contours.png", "."),

        # Companions/rasterized
        ("xyz_heatmap_Omega_cdm-img0.png", "Omega"),
        ("xyz_heatmap_w-img1.png", "EoS"),
    ]

    failures = []
    for basename, expected_route in test_cases:
        actual_route = route_file(basename)
        if actual_route != expected_route:
            failures.append(
                f"FAIL: {basename} -> {actual_route} (expected {expected_route})"
            )

    # Test unmatched
    unmatched_cases = [
        "randomfile.pdf",
        "not_a_plot.png",
        "heatmap_Omega_cdm_old.pdf",  # Old naming
    ]

    for basename in unmatched_cases:
        route = route_file(basename)
        if route is not None:
            failures.append(f"FAIL: {basename} should be unmatched but got {route}")

    # Test debris detection
    debris_cases = [
        ("data.npz", True),
        ("heatmap_w_old.pdf", True),
        ("heatmap_Omega_cdm_root.png", True),
        ("output.dat", True),
        ("processed.processed.npz", True),
        ("xyz_heatmap_w.pdf", False),  # Not debris
    ]

    for basename, is_deb in debris_cases:
        actual = is_debris(basename)
        if actual != is_deb:
            failures.append(f"FAIL: is_debris({basename}) = {actual}, expected {is_deb}")

    if failures:
        print("=== ROUTING TESTS FAILED ===")
        for msg in failures:
            print(msg)
        sys.exit(1)
    else:
        print("✓ All routing tests passed")


# ============================================================================
# Reporting
# ============================================================================

def print_summary(stats: StagingStats, command: str = ""):
    """Print staging summary."""
    print(f"\n{'='*70}")
    print(f"STAGING SUMMARY ({command})")
    print(f"{'='*70}")
    print(f"Staged:      {stats.staged}")
    print(f"Up-to-date:  {stats.up_to_date}")
    print(f"Debris:      {stats.debris}")

    if stats.debris_list:
        print("\nDebris files (ignored):")
        for name in sorted(stats.debris_list)[:10]:
            print(f"  - {name}")
        if len(stats.debris_list) > 10:
            print(f"  ... and {len(stats.debris_list) - 10} more")

    if stats.unmatched:
        print("\nUNMATCHED files (routing gap):")
        for name in sorted(stats.unmatched):
            print(f"  - {name}")

    print()


def print_rsync_command():
    """Print the rsync command for user."""
    print(f"\n{'='*70}")
    print("RSYNC COMMAND")
    print(f"{'='*70}")
    print(f"rsync -av --checksum {STAGING_ROOT}/ /home/kl/kDrive/Sci/PhD/Research/HDM/LaTeX/20260504/figures/")
    print()


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage thesis figures to a staging tree for rsync."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify staging tree against thesis .tex references (requires staging done first)."
    )

    args = parser.parse_args()

    print("stage_thesis_figures.py - Staging script for thesis figures")
    print()

    # Step 1: Run self-tests
    print("Step 1: Running unit tests...")
    test_routing()
    print()

    # Step 2: Stage files (dry-run or real)
    print(f"Step 2: {'DRY-RUN' if args.dry_run else 'STAGING'} files...")
    stats = stage_all(dry_run=args.dry_run)
    print_summary(stats, "DRY-RUN" if args.dry_run else "REAL")

    # Step 3: Print rsync command (if not dry-run)
    if not args.dry_run:
        print_rsync_command()

    # Step 4: Check mode (only works if staging was done)
    if args.check:
        print("Step 3: Checking staging tree against thesis references...")
        missing = check_references()
        if missing:
            print("\nMISSING references (expected to exist as both .pdf and .pgf):")
            for ref in missing[:20]:
                print(f"  - {ref}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        else:
            print("✓ All thesis references found in staging tree")
        print()


if __name__ == "__main__":
    main()
