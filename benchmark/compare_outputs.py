#!/usr/bin/env python3
"""
CLASS compiler benchmark — compare_outputs.py

Compare output data files produced by two different compiler builds of CLASS.
Reports column-by-column max absolute and relative differences.

Usage:
    python3 benchmark/compare_outputs.py <label_A> <label_B>

Example:
    python3 benchmark/compare_outputs.py gcc clang

Prerequisites:
    Run benchmark/run_benchmark.sh for both labels first.
"""

import sys
import os
import glob
import numpy as np
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def parse_class_dat(filepath):
    """
    Read a CLASS .dat output file.
    Returns (header_lines, col_names, data) where data is a 2-D numpy array.
    Skips comment lines (starting with #).  The last comment line before the
    data block is assumed to contain column names.
    """
    header_lines = []
    col_line = ""
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                header_lines.append(stripped)
                col_line = stripped  # keep updating — last one wins
            elif stripped == "":
                continue
            else:
                break  # first data line

    # Parse column names from the last header line
    col_names = col_line.lstrip("#").split()

    # Read numerical data
    try:
        data = np.loadtxt(filepath, comments="#")
    except Exception as e:
        return header_lines, col_names, None, str(e)

    return header_lines, col_names, data, None


def compare_files(file_a, file_b, label_a, label_b):
    """
    Compare two CLASS output files.  Returns a dict of per-column stats, or
    an error string.
    """
    ha, cols_a, da, err_a = parse_class_dat(file_a)
    hb, cols_b, db, err_b = parse_class_dat(file_b)

    if err_a:
        return f"  Cannot read {label_a} file: {err_a}"
    if err_b:
        return f"  Cannot read {label_b} file: {err_b}"

    if da is None or db is None:
        return "  One or both files have no data."

    # Handle 1-D arrays (single column)
    if da.ndim == 1:
        da = da.reshape(-1, 1)
    if db.ndim == 1:
        db = db.reshape(-1, 1)

    if da.shape != db.shape:
        return f"  Shape mismatch: {label_a} {da.shape} vs {label_b} {db.shape}"

    results = []
    ncols = da.shape[1]
    # Use column names if available, else generic labels
    for c in range(ncols):
        col_name = cols_a[c] if c < len(cols_a) else f"col{c}"
        diff = np.abs(da[:, c] - db[:, c])
        max_abs = np.max(diff)

        # Relative difference: avoid division by zero
        denom = np.maximum(np.abs(da[:, c]), np.abs(db[:, c]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, diff / denom, 0.0)
        max_rel = np.max(rel)

        results.append(
            {
                "col": col_name,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
            }
        )

    return results


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <label_A> <label_B>")
        print(f"Example: {sys.argv[0]} gcc clang")
        sys.exit(1)

    label_a = sys.argv[1]
    label_b = sys.argv[2]

    root = Path(__file__).resolve().parent.parent  # CLASS root
    dir_a = root / "benchmark" / label_a / "output"
    dir_b = root / "benchmark" / label_b / "output"

    if not dir_a.is_dir():
        print(f"ERROR: Directory not found: {dir_a}")
        print(f"       Run  benchmark/run_benchmark.sh {label_a}  first.")
        sys.exit(1)
    if not dir_b.is_dir():
        print(f"ERROR: Directory not found: {dir_b}")
        print(f"       Run  benchmark/run_benchmark.sh {label_b}  first.")
        sys.exit(1)

    # Collect all .dat files from label_a, look for counterparts in label_b
    files_a = sorted(glob.glob(str(dir_a / "*.dat")))
    if not files_a:
        print(f"No .dat files found in {dir_a}")
        sys.exit(1)

    print("=" * 78)
    print(f"  Comparing CLASS outputs:  {label_a}  vs  {label_b}")
    print("=" * 78)
    print()

    n_identical = 0
    n_close = 0
    n_differ = 0
    n_missing = 0
    summary_rows = []

    REL_THRESHOLD = 1e-10  # below this we call it "identical"

    for fa in files_a:
        fname = os.path.basename(fa)
        fb = dir_b / fname

        if not fb.exists():
            n_missing += 1
            summary_rows.append((fname, "MISSING", "-", "-"))
            continue

        result = compare_files(fa, str(fb), label_a, label_b)

        if isinstance(result, str):
            # error message
            n_differ += 1
            summary_rows.append((fname, "ERROR", "-", result))
            continue

        # Find worst-case across all columns
        worst_abs = max(r["max_abs_diff"] for r in result)
        worst_rel = max(r["max_rel_diff"] for r in result)

        if worst_rel == 0.0:
            status = "IDENTICAL"
            n_identical += 1
        elif worst_rel < REL_THRESHOLD:
            status = "IDENTICAL"
            n_identical += 1
        elif worst_rel < 1e-6:
            status = "CLOSE"
            n_close += 1
        else:
            status = "DIFFER"
            n_differ += 1

        summary_rows.append((fname, status, f"{worst_abs:.3e}", f"{worst_rel:.3e}"))

    # Check for files in B that are not in A
    files_b_names = {os.path.basename(f) for f in glob.glob(str(dir_b / "*.dat"))}
    files_a_names = {os.path.basename(f) for f in files_a}
    only_in_b = files_b_names - files_a_names
    for fname in sorted(only_in_b):
        summary_rows.append((fname, f"ONLY IN {label_b}", "-", "-"))

    # ── print summary table ──────────────────────────────────────────────
    print(f"{'File':<50} {'Status':<10} {'Max|Δ|':<12} {'Max|Δ|/|x|':<12}")
    print(f"{'-'*50} {'-'*10} {'-'*12} {'-'*12}")
    for fname, status, abs_str, rel_str in summary_rows:
        print(f"{fname:<50} {status:<10} {abs_str:<12} {rel_str:<12}")

    print()
    print("-" * 78)
    print(f"  IDENTICAL (rel diff < {REL_THRESHOLD:.0e}) : {n_identical}")
    print(f"  CLOSE     (rel diff < 1e-6)    : {n_close}")
    print(f"  DIFFER    (rel diff >= 1e-6)    : {n_differ}")
    print(f"  MISSING in {label_b:<22} : {n_missing}")
    if only_in_b:
        print(f"  ONLY IN {label_b:<25} : {len(only_in_b)}")
    print("-" * 78)

    # ── detailed per-column report for files that are not identical ───────
    non_identical = [
        r
        for r in summary_rows
        if r[1] not in ("IDENTICAL", "MISSING")
        and r[1] != f"ONLY IN {label_b}"
        and r[1] != "ERROR"
    ]
    if non_identical:
        print()
        print("=" * 78)
        print("  Detailed per-column breakdown for non-identical files")
        print("=" * 78)
        for fname, status, _, _ in non_identical:
            fa = str(dir_a / fname)
            fb = str(dir_b / fname)
            result = compare_files(fa, fb, label_a, label_b)
            if isinstance(result, str):
                continue
            print(f"\n  {fname}  [{status}]")
            print(f"  {'Column':<30} {'Max|Δ|':<14} {'Max|Δ|/|x|':<14}")
            print(f"  {'-'*30} {'-'*14} {'-'*14}")
            for r in result:
                print(
                    f"  {r['col']:<30} {r['max_abs_diff']:<14.6e} {r['max_rel_diff']:<14.6e}"
                )

    print()
    if n_differ == 0 and n_missing == 0:
        print(
            "All outputs agree within tolerance. The two compilers produce equivalent results."
        )
    elif n_differ == 0:
        print(
            "All matched outputs agree. Some files are missing from one side (expected for failing ini files)."
        )
    else:
        print("Some outputs differ — inspect the detailed breakdown above.")

    return 0 if n_differ == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
