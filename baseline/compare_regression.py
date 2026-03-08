#!/usr/bin/env python3
"""compare_regression.py — Compare two regression output directories.

Usage: python3 baseline/compare_regression.py <reference_dir> <test_dir>
   e.g. python3 baseline/compare_regression.py baseline assessment

Compares:
  1. .dat files: column-by-column numerical comparison with max relative
     and absolute differences reported per file.
  2. .npz files: array-by-array comparison (bit-exact and tolerance-based).
  3. .npy files: array comparison (bit-exact and tolerance-based).

Exit code 0 = all comparisons pass, 1 = differences found or files missing.
"""
import sys
import os
import glob
import re
import numpy as np

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# Shooting and numerical precision can cause tiny differences even between
# identical code on different runs.  We demand bit-exact agreement for
# simple models but allow tiny tolerance for models with shooting (iDM, PGO).
#
# The comparison reports BOTH bit-exact and tolerance-based results so the
# user can judge whether differences are acceptable.
REL_TOL = 1e-12  # relative tolerance for "pass"
ABS_TOL = 1e-30  # absolute tolerance (for values near zero)


def parse_dat(path):
    """Parse a CLASS .dat file, returning (header_lines, data_array).

    Handles comment lines starting with '#' and may return an empty array
    for header-only files.
    """
    header = []
    data_lines = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or stripped == "":
                header.append(line)
            else:
                data_lines.append(stripped)

    if not data_lines:
        return header, np.array([])

    rows = []
    skipped = 0
    ncols_first = None
    ragged = False
    for dl in data_lines:
        try:
            vals = [float(x) for x in dl.split()]
        except ValueError:
            skipped += 1
            continue
        if ncols_first is None:
            ncols_first = len(vals)
        elif len(vals) != ncols_first:
            ragged = True
            skipped += 1
            continue
        rows.append(vals)

    if skipped:
        print(
            f"WARNING: {path}: skipped {skipped}/{len(data_lines)} "
            f"unparseable or ragged data lines"
        )
    if ragged:
        print(f"WARNING: {path}: inconsistent column counts detected")

    if not rows:
        return header, np.array([])

    return header, np.array(rows)


def compare_arrays(a, b, label):
    """Compare two numpy arrays, returning a dict of results."""
    result = {
        "label": label,
        "shape_a": a.shape,
        "shape_b": b.shape,
        "bit_exact": False,
        "max_abs_diff": np.nan,
        "max_rel_diff": np.nan,
        "pass": False,
    }

    if a.shape != b.shape:
        result["error"] = f"shape mismatch: {a.shape} vs {b.shape}"
        return result

    if a.size == 0:
        result["bit_exact"] = True
        result["max_abs_diff"] = 0.0
        result["max_rel_diff"] = 0.0
        result["pass"] = True
        return result

    # --- NaN / Inf check ---
    a_bad = int(np.sum(~np.isfinite(a)))
    b_bad = int(np.sum(~np.isfinite(b)))
    if a_bad or b_bad:
        result["error"] = f"non-finite values: ref has {a_bad}, test has {b_bad}"
        return result

    # --- Integrity checks: detect signs of uninitialized / garbage memory ---
    # These flag the data as CORRUPT rather than skipping silently.
    # CLASS outputs background/thermodynamics with z DECREASING, so we
    # accept monotonic in either direction — only flag truly scrambled data.
    # Denormalized values (|x| > 0, < ~2.2e-308) in col0 indicate
    # uninitialised memory (physical quantities like z are never that tiny).
    # We also check for denormalized values in the first few rows of any
    # column — malloc garbage in an output table is a clear sign the
    # table was never properly initialised.
    DENORM_THRESHOLD = 1e-300
    corrupt_reasons = []

    def _count_denorm(arr):
        return int(np.sum((np.abs(arr) > 0) & (np.abs(arr) < DENORM_THRESHOLD)))

    if a.ndim == 2 and a.shape[1] >= 2:
        for arr, who in [(a, "ref"), (b, "test")]:
            col0 = arr[:, 0]
            d = np.diff(col0)
            mono_inc = np.all(d >= 0)
            mono_dec = np.all(d <= 0)
            col0_denorm = _count_denorm(col0)
            col0_negative = int(np.sum(col0 < 0))

            if col0_denorm:
                corrupt_reasons.append(
                    f"{who} col0 has {col0_denorm} denormalized values"
                )
            if col0_negative:
                corrupt_reasons.append(
                    f"{who} col0 has {col0_negative} negative values"
                )
            if not mono_inc and not mono_dec:
                corrupt_reasons.append(
                    f"{who} col0 not sorted (neither increasing nor decreasing)"
                )

            # Check the first N rows for denormalized values in ANY column.
            # Real CLASS output never starts with denormalized numbers; malloc
            # garbage does.  We only check the head because some legitimate
            # data columns may have near-zero values at large z.
            head = arr[: min(20, arr.shape[0]), :]
            head_denorm = _count_denorm(head)
            if head_denorm:
                corrupt_reasons.append(
                    f"{who} first rows have {head_denorm} denormalized values (uninitialised memory)"
                )

    if corrupt_reasons:
        result["corrupt"] = True
        result["corrupt_reasons"] = corrupt_reasons
        # Still compute comparison metrics where possible
        result["bit_exact"] = np.array_equal(a, b)
        abs_diff = np.abs(a - b)
        result["max_abs_diff"] = float(np.max(abs_diff))
        denom = np.maximum(np.abs(a), np.abs(b))
        mask = denom > ABS_TOL
        if np.any(mask):
            result["max_rel_diff"] = float(np.max(abs_diff[mask] / denom[mask]))
        else:
            result["max_rel_diff"] = 0.0
        # CORRUPT is never a pass — it must be fixed or the write must be
        # guarded so the file is not produced at all.
        result["pass"] = False
        return result

    result["bit_exact"] = np.array_equal(a, b)

    abs_diff = np.abs(a - b)
    result["max_abs_diff"] = float(np.max(abs_diff))

    # Relative diff: use max(|a|, |b|) as denominator, skip where both ~0
    denom = np.maximum(np.abs(a), np.abs(b))
    mask = denom > ABS_TOL
    if np.any(mask):
        rel = abs_diff[mask] / denom[mask]
        result["max_rel_diff"] = float(np.max(rel))
    else:
        result["max_rel_diff"] = 0.0

    # Pass if bit-exact, or if the maximum relative difference is within
    # tolerance.  The relative diff already ignores near-zero values
    # (via the denom > ABS_TOL mask above), so no separate abs check is
    # needed — the old "and abs_diff <= 1e-20" clause made it impossible
    # for any array with values > O(1e-8) to pass via tolerance.
    result["pass"] = result["bit_exact"] or result["max_rel_diff"] <= REL_TOL

    return result


def _extract_column_names(header):
    """Extract column names from the last header line (CLASS convention)."""
    for line in reversed(header):
        stripped = line.strip()
        if stripped.startswith("#") and ":" in stripped:
            # CLASS format: "#  1:col_name  2:col_name  ..."
            cols = re.findall(r"\d+:([^\s]+)", stripped)
            if cols:
                return cols
    return None


def compare_dat_files(ref_path, test_path):
    """Compare two .dat files column-by-column."""
    ref_hdr, ref_data = parse_dat(ref_path)
    test_hdr, test_data = parse_dat(test_path)

    name = os.path.basename(ref_path)
    results = []

    # Compare column names from headers
    ref_cols = _extract_column_names(ref_hdr)
    test_cols = _extract_column_names(test_hdr)
    if ref_cols and test_cols and ref_cols != test_cols:
        results.append(
            {
                "label": f"{name} (columns)",
                "error": f"column name mismatch: {ref_cols} vs {test_cols}",
                "pass": False,
                "shape_a": (),
                "shape_b": (),
            }
        )
        return results

    # Header-only files (e.g. exotic_injection without injection source)
    if ref_data.size == 0 and test_data.size == 0:
        results.append(
            {
                "label": f"{name} (header-only)",
                "bit_exact": True,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "pass": True,
                "shape_a": (0,),
                "shape_b": (0,),
            }
        )
        return results

    if ref_data.size == 0 or test_data.size == 0:
        results.append(
            {
                "label": f"{name}",
                "error": f"one file is header-only, other has data",
                "pass": False,
                "shape_a": ref_data.shape,
                "shape_b": test_data.shape,
            }
        )
        return results

    r = compare_arrays(ref_data, test_data, name)
    results.append(r)
    return results


def compare_npz_files(ref_path, test_path):
    """Compare two .npz files array-by-array."""
    name = os.path.basename(ref_path)
    results = []

    ref = np.load(ref_path)
    test = np.load(test_path)

    ref_keys = set(ref.files)
    test_keys = set(test.files)

    if ref_keys != test_keys:
        results.append(
            {
                "label": f"{name} (keys)",
                "error": f"key mismatch: ref={sorted(ref_keys)}, test={sorted(test_keys)}",
                "pass": False,
                "shape_a": (),
                "shape_b": (),
            }
        )
        return results

    for key in sorted(ref_keys):
        r = compare_arrays(ref[key], test[key], f"{name}[{key}]")
        results.append(r)

    return results


def compare_npy_files(ref_path, test_path):
    """Compare two .npy files."""
    name = os.path.basename(ref_path)
    ref = np.load(ref_path)
    test = np.load(test_path)
    return [compare_arrays(ref, test, name)]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <reference_dir> <test_dir>")
        sys.exit(2)

    ref_dir = sys.argv[1]
    test_dir = sys.argv[2]

    if not os.path.isdir(ref_dir):
        print(f"ERROR: reference directory '{ref_dir}' not found")
        sys.exit(2)
    if not os.path.isdir(test_dir):
        print(f"ERROR: test directory '{test_dir}' not found")
        sys.exit(2)

    # Collect all reference files
    ref_files = set()
    for ext in ("*.dat", "*.npz", "*.npy"):
        for f in glob.glob(os.path.join(ref_dir, ext)):
            ref_files.add(os.path.basename(f))

    test_files = set()
    for ext in ("*.dat", "*.npz", "*.npy"):
        for f in glob.glob(os.path.join(test_dir, ext)):
            test_files.add(os.path.basename(f))

    # Check for missing files
    missing_in_test = ref_files - test_files
    extra_in_test = test_files - ref_files
    common = sorted(ref_files & test_files)

    all_results = []
    n_pass = 0
    n_fail = 0
    n_exact = 0
    n_corrupt = 0

    print(f"\nComparing {ref_dir}/ vs {test_dir}/")
    print(f"  Reference files:  {len(ref_files)}")
    print(f"  Test files:       {len(test_files)}")
    print(f"  Common files:     {len(common)}")
    if missing_in_test:
        print(f"  MISSING in test:  {len(missing_in_test)}")
        for f in sorted(missing_in_test):
            print(f"    - {f}")
        n_fail += len(missing_in_test)
    if extra_in_test:
        print(f"  Extra in test:    {len(extra_in_test)}")
        for f in sorted(extra_in_test):
            print(f"    + {f}")

    print(
        f"\n{'File':<60s} {'Exact':>5s} {'MaxRelDiff':>14s} {'MaxAbsDiff':>14s} {'Status':>8s}"
    )
    print("=" * 105)

    for fname in common:
        ref_path = os.path.join(ref_dir, fname)
        test_path = os.path.join(test_dir, fname)

        if fname.endswith(".dat"):
            results = compare_dat_files(ref_path, test_path)
        elif fname.endswith(".npz"):
            results = compare_npz_files(ref_path, test_path)
        elif fname.endswith(".npy"):
            results = compare_npy_files(ref_path, test_path)
        else:
            continue

        for r in results:
            all_results.append(r)
            if "error" in r:
                status = "FAIL"
                n_fail += 1
                exact_str = "ERR"
                rel_str = r.get("error", "")[:14]
                abs_str = ""
            elif r.get("corrupt"):
                status = "CORRUPT"
                n_corrupt += 1
                exact_str = "---"
                rel_str = (
                    f"{r['max_rel_diff']:.6e}"
                    if not np.isnan(r["max_rel_diff"])
                    else "N/A"
                )
                abs_str = (
                    f"{r['max_abs_diff']:.6e}"
                    if not np.isnan(r["max_abs_diff"])
                    else "N/A"
                )
            else:
                exact_str = "yes" if r["bit_exact"] else "no"
                rel_str = (
                    f"{r['max_rel_diff']:.6e}"
                    if not np.isnan(r["max_rel_diff"])
                    else "N/A"
                )
                abs_str = (
                    f"{r['max_abs_diff']:.6e}"
                    if not np.isnan(r["max_abs_diff"])
                    else "N/A"
                )
                if r["pass"]:
                    status = "PASS"
                    n_pass += 1
                    if r["bit_exact"]:
                        n_exact += 1
                else:
                    status = "FAIL"
                    n_fail += 1

            print(
                f"{r['label']:<60s} {exact_str:>5s} {rel_str:>14s} {abs_str:>14s} {status:>8s}"
            )

    # Summary
    print()
    print("=" * 105)
    total = n_pass + n_fail + n_corrupt
    print(
        f"SUMMARY: {n_pass} passed ({n_exact} bit-exact), {n_fail} failed, "
        f"{n_corrupt} corrupt, out of {total} comparisons"
    )
    print(f"  Tolerances: rel={REL_TOL:.0e}, abs={ABS_TOL:.0e}")

    if n_corrupt > 0:
        print(
            f"\n  CORRUPT files ({n_corrupt}) — data contains uninitialised memory garbage:"
        )
        for r in all_results:
            if r.get("corrupt"):
                reasons = "; ".join(r["corrupt_reasons"])
                print(f"    {r['label']}: {reasons}")
        print("  These must be fixed (guard the write or initialise the table).")

    if n_fail > 0:
        print("\n*** REGRESSION TEST FAILED ***")
        for r in all_results:
            if not r.get("pass", True) and not r.get("corrupt"):
                if "error" in r:
                    print(f"  FAIL: {r['label']}: {r['error']}")
                else:
                    print(
                        f"  FAIL: {r['label']}: max_rel={r['max_rel_diff']:.6e} max_abs={r['max_abs_diff']:.6e}"
                    )
        sys.exit(1)
    elif n_corrupt > 0:
        print(
            "\n*** ALL NON-CORRUPT COMPARISONS PASSED — corrupt files need fixing ***"
        )
        sys.exit(2)
    else:
        print("\n*** ALL COMPARISONS PASSED ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
