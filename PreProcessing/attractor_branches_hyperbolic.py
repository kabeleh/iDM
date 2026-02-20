#!/usr/bin/env python
"""
Attractor branch analysis for the hyperbolic scalar field potential in CLASS.

For the hyperbolic potential V(phi) = c1 * (1 - tanh(c2*phi)), the attractor
initial conditions during radiation domination depend on the value of c2.
This script:

1. Runs a minimal CLASS instance to obtain Omega_g, Omega_ur, H0 (internal
   units) and the ncdm relativistic contribution — these fix rho_rad at the
   initial scale factor a_ini.

2. Analytically evaluates the branch boundaries for c2, reproducing the logic
   in background_initial_conditions() (source/background.c, case 3).

3. Scans c2 over a wide range, calling CLASS for each value with
   attractor_ic_scf = yes, and records the derived parameters
   phi_ini_scf_ic, phi_prime_scf_ic, and attractor_regime_scf.

4. Prints a summary table mapping c2 ranges → attractor regimes → initial
   field values.

Usage:
    python PreProcessing/attractor_branches_hyperbolic.py

Dependencies: classy (the CLASS Python wrapper), numpy, tabulate (optional).

Note: The baseline cosmological parameters are taken from pgo_hyperbolic_cmb.ini.
      Adjust the `cosmo_params` and `scf_base_params` dictionaries below as needed.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt

# ── classy import ──────────────────────────────────────────────────────────
try:
    from classy import Class  # type: ignore[import-untyped]
except ImportError:
    sys.exit("ERROR: could not import classy.  Install with  pip install .")

# ── try to import tabulate for pretty printing; fall back to manual ────────
_has_tabulate = False
try:
    from tabulate import tabulate as _tabulate  # type: ignore[import-untyped]

    _has_tabulate = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Baseline cosmological parameters (from pgo_hyperbolic_cmb.ini)
cosmo_params: dict[str, Any] = {
    "H0": 67,
    "omega_b": 0.0224,
    "omega_cdm": 0.12,
    "tau_reio": 0.051,
    "A_s": 2.1065e-9,
    "n_s": 0.965,
    "N_ur": 2.046,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "YHe": "BBN",
}

# Scalar-field base parameters (c1 = scf_parameters[0], c2 = scf_parameters[1])
# c1 is the shooting parameter: CLASS tunes it to match Omega_scf.
# But the shooting root-finder needs a reasonable starting guess.
# We try multiple orders of magnitude as initial guesses.
c1_default = 1e-7  # reference value for the analytical calculation

# c1 guesses to try for shooting (CLASS will converge to the correct c1
# from any guess that allows the root-finder to bracket the solution).
# Wider range = more robust but slower.
# 1 OOM spacing lets us assess whether coarser (2 OOM) bins suffice.
c1_guesses: list[float] = [10.0**n for n in range(-20, 21)]  # 1e-20 .. 1e20

# Parallelization: number of worker processes for c1 search.
# Each CLASS instance is single-threaded (parallel parts are C++, not OpenMP).
# With N workers: N simultaneous CLASS calls during c1 search at transitions.
# Set to 1 to disable parallelization.
N_PARALLEL_WORKERS = 16

# Other scalar-field settings
scf_base_params: dict[str, Any] = {
    "Omega_Lambda": 0.0,
    "Omega_fld": 0.0,
    "Omega_scf": -0.7,
    "scf_potential": "hyperbolic",
    "scf_tuning_index": 0,
    "tol_initial_Omega_r": 0.01,
    # interacting DM (uncoupled for now)
    "model_cdm": "i",
    "cdm_c": 0.1,
    # Newtonian gauge required by the interactive DM model
    "gauge": "newtonian",
    # Push a_ini earlier so Omega_r ≈ 1 even for large c2 (tracking solution)
    "a_ini_over_a_today_default": 1e-16,
}

# Precision: a_ini / a_today — must match what CLASS uses above
a_ini_over_a_today_default = 1e-16

# c2 scan range (covers the full physical range)
c2_scan_min = 0.01
c2_scan_max = 1e12

# Scan resolution: points per decade in log10(c2)
c2_points_per_decade = 12
# Extra fine points around each analytical boundary
c2_fine_points = 30

# Verbosity for CLASS runs during the scan (0 = silent)
class_verbosity: dict[str, Any] = {
    "input_verbose": 0,
    "background_verbose": 0,
    "thermodynamics_verbose": 0,
    "perturbations_verbose": 0,
    "transfer_verbose": 0,
    "primordial_verbose": 0,
    "harmonic_verbose": 0,
    "fourier_verbose": 0,
    "lensing_verbose": 0,
    "output_verbose": 0,
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Analytical helper: reproduce the C branch logic in Python
# ═══════════════════════════════════════════════════════════════════════════


def compute_rho_rad(
    Omega_g: float,
    Omega_ur: float,
    H0_inv_Mpc: float,
    a_ini: float,
    rho_ncdm_rel: float = 0.0,
) -> float:
    """
    Compute the initial radiation energy density exactly as CLASS does:
        rho_rad = (Omega_g + Omega_ur) * H0**2 / a_ini**4  +  rho_ncdm_rel
    All quantities in CLASS internal units (Mpc^{-2} for densities).

    Parameters
    ----------
    Omega_g : float        Fractional photon density today.
    Omega_ur : float       Fractional ultra-relativistic density today.
    H0_inv_Mpc : float     Hubble constant in 1/Mpc (CLASS internal).
    a_ini : float          Initial scale factor (a_ini / a_today).
    rho_ncdm_rel : float   Relativistic ncdm contribution (default 0).
    """
    Omega_rad: float = Omega_g + Omega_ur
    return Omega_rad * H0_inv_Mpc**2 / a_ini**4 + rho_ncdm_rel


def analytical_attractor_hyperbolic(
    c1: float,
    c2: float,
    rho_rad: float,
    a_ini: float,
    scf_gamma: float = 4.0 / 3.0,
    omega_background: float = 1.0 / 3.0,
) -> tuple[int, float | None, float | None, str]:
    """
    Reproduce the attractor-IC logic from background_initial_conditions(),
    case 3 (hyperbolic), for radiation domination.

    Returns
    -------
    regime : int
        0 = no attractor / fallback
        1 = large-field attractor (exact)
        2 = approximate attractor (exponential-like)
        3 = small-field attractor (Taylor expansion)
    phi_ini : float or None
        Initial scalar field value (None if regime == 0).
    phi_prime_ini : float or None
        Initial scalar field velocity (None if regime == 0).
    info : str
        Human-readable description of which branch was taken.
    """
    scf_lambda: float = -c2  # sign convention: scf_lambda = -c2
    scf_lambda2: float = scf_lambda * scf_lambda  # = c2**2

    # ── Branch 0: no tracking solution ──
    if 1.0 <= 3.0 * scf_gamma / scf_lambda2:
        return (
            0,
            None,
            None,
            (
                f"|c2| too small (c2^2={c2**2:.4g} <= 3*gamma={3*scf_gamma:.4g}): "
                "no tracking solution; uses user-provided phi_ini, phi_prime_ini"
            ),
        )

    # ── Tracking quantities ──
    rho_tracking: float = (
        rho_rad * 3.0 * scf_gamma / scf_lambda2 / (1.0 - 3.0 * scf_gamma / scf_lambda2)
    )
    phi_prime_ini: float = float(
        a_ini * np.sqrt(rho_tracking * (1.0 + omega_background))
    )

    # Helper constants used in the branch conditions
    val_A: float = 16.0 * c1 * c2**2
    val_B: float = 12.0 * c1 * scf_gamma  # = 16*c1
    val_C: float = 3.0 * scf_gamma * rho_tracking  # = 4*rho_tracking
    w: float = omega_background  # = 1/3

    omega_bound: float = (1.0 / (3.0 * scf_gamma * rho_tracking)) * (
        -val_A + val_B + val_C
    )

    # ── Large-field attractor (regime 1) ──
    # Only possible if c2 > 0 (we already have c2**2 > 4 from above)
    large_field = False
    if c2 > 0.0:
        # Evaluate each sub-branch exactly as in C code
        # c1 > 0 sub-branches
        if (
            c1 > 0.0
            and scf_gamma > (4.0 / 3.0) * c2**2
            and rho_tracking > 0.0
            and w > 1.0
            and w < omega_bound
        ):
            large_field = True
        elif (
            c1 > 0.0
            and scf_gamma > 0.0
            and scf_gamma < (4.0 / 3.0) * c2**2
            and rho_tracking > (1.0 / (3.0 * scf_gamma)) * (val_A - val_B)
            and w > omega_bound
            and w < 1.0
        ):
            large_field = True
        elif (
            c1 > 0.0
            and scf_gamma > 0.0
            and scf_gamma < (4.0 / 3.0) * c2**2
            and rho_tracking > 0.0
            and rho_tracking <= (1.0 / (3.0 * scf_gamma)) * (val_A - val_B)
            and w > 0.0
            and w < 1.0
        ):
            large_field = True
        # c1 < 0 sub-branches
        elif (
            c1 < 0.0
            and scf_gamma > (4.0 / 3.0) * c2**2
            and rho_tracking > (1.0 / (3.0 * scf_gamma)) * (val_A - val_B)
            and w > omega_bound
            and w < 1.0
        ):
            large_field = True
        elif (
            c1 < 0.0
            and scf_gamma > (4.0 / 3.0) * c2**2
            and rho_tracking > 0.0
            and rho_tracking <= (1.0 / (3.0 * scf_gamma)) * (val_A - val_B)
            and w > 0.0
            and w < 1.0
        ):
            large_field = True
        elif (
            c1 < 0.0
            and scf_gamma > 0.0
            and scf_gamma < (4.0 / 3.0) * c2**2
            and rho_tracking > 0.0
            and w > 1.0
            and w < omega_bound
        ):
            large_field = True

    if large_field:
        numer: float = (
            8.0 * c1 * c2**2
            - 6.0 * c1 * scf_gamma
            - 3.0 * scf_gamma * rho_tracking
            + 3.0 * scf_gamma * rho_tracking * w
        )
        denom: float = 2.0 * c1 * (4.0 * c2**2 - 3.0 * scf_gamma)
        arg: float = numer / denom
        phi_ini: float
        if abs(arg) >= 1.0:
            # atanh undefined — fallback handled by NaN check in C
            phi_ini = float("nan")
        else:
            phi_ini = float(np.arctanh(arg) / c2)
        return 1, phi_ini, phi_prime_ini, "large-field attractor (exact atanh solution)"

    # ── Approximate attractor (regime 2) ──
    approx = False
    if c1 > 0.0 and scf_gamma > (1.0 / 3.0) * c2**2 and rho_tracking > 0.0 and w > 1.0:
        approx = True
    elif (
        c1 > 0.0
        and scf_gamma > 0.0
        and scf_gamma < (1.0 / 3.0) * c2**2
        and rho_tracking > 0.0
        and w > 0.0
        and w < 1.0
    ):
        approx = True
    elif (
        c1 < 0.0
        and scf_gamma > (1.0 / 3.0) * c2**2
        and rho_tracking > 0.0
        and w > 0.0
        and w < 1.0
    ):
        approx = True
    elif (
        c1 < 0.0
        and scf_gamma > 0.0
        and scf_gamma < (1.0 / 3.0) * c2**2
        and rho_tracking > 0.0
        and w > 1.0
    ):
        approx = True

    if approx:
        log_arg: float = (
            -2.0
            * (c1 * c2**2 - 3.0 * c1 * scf_gamma)
            / (3.0 * scf_gamma * rho_tracking * (-1.0 + w))
        )
        phi_ini_approx: float
        if log_arg <= 0.0:
            phi_ini_approx = float("nan")
        else:
            phi_ini_approx = float(np.log(log_arg) / c2)
        return (
            2,
            phi_ini_approx,
            phi_prime_ini,
            "approximate attractor (exponential-like)",
        )

    # ── Small-field attractor (regime 3) — fallback ──
    numer_sf: float = (
        2.0 * c1 * c2**2
        - 6.0 * c1 * scf_gamma
        - 3.0 * scf_gamma * rho_tracking
        + 3.0 * scf_gamma * rho_tracking * w
    )
    denom_sf: float = 2.0 * c1 * c2**3 - 6.0 * c1 * c2 * scf_gamma
    phi_ini_sf: float
    if denom_sf == 0.0:
        phi_ini_sf = float("nan")
    else:
        phi_ini_sf = numer_sf / denom_sf
    return 3, phi_ini_sf, phi_prime_ini, "small-field attractor (Taylor fallback)"


def find_analytical_boundaries(
    c1: float,
    rho_rad: float,
    a_ini: float,
    c2_range: npt.NDArray[np.floating[Any]] | None = None,
) -> tuple[
    list[tuple[float, int, int]], npt.NDArray[np.intp], npt.NDArray[np.floating[Any]]
]:
    """
    Scan c2 analytically to locate the boundaries between attractor regimes.

    Returns
    -------
    boundaries : list of (c2_boundary, regime_below, regime_above)
    regimes : ndarray of regime integers for each c2 in c2_range
    """
    if c2_range is None:
        c2_range = np.logspace(-1, 3, 10000)
    regimes: npt.NDArray[np.intp] = np.array(
        [
            analytical_attractor_hyperbolic(c1, c2_val, rho_rad, a_ini)[0]
            for c2_val in c2_range
        ]
    )
    boundaries: list[tuple[float, int, int]] = []
    for i in range(1, len(c2_range)):
        if regimes[i] != regimes[i - 1]:
            boundaries.append(
                (
                    float(0.5 * (c2_range[i - 1] + c2_range[i])),
                    int(regimes[i - 1]),
                    int(regimes[i]),
                )
            )
    return boundaries, regimes, c2_range


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLASS interface helpers
# ═══════════════════════════════════════════════════════════════════════════


def get_radiation_density_from_class(
    cosmo_params_dict: dict[str, Any],
) -> tuple[float, float, float, float]:
    """
    Run a minimal CLASS instance (without scalar field) to obtain
    Omega_g, Omega_ur (or Omega_r), H0_internal and the ncdm relativistic
    contribution at a_ini.

    Returns (Omega_g, Omega_ur, H0_internal, rho_ncdm_rel_approx).
    """
    M = Class()  # type: ignore[no-untyped-call]
    # Set up a simple LCDM-like run (no scf, just to get radiation params)
    params: dict[str, Any] = dict(cosmo_params_dict)
    params["output"] = ""  # no spectra needed
    params.update(class_verbosity)
    M.set(params)
    M.compute()

    Omega_g = float(M.Omega_g())
    h = float(M.h())
    H0_internal = h * 100.0 / 299792.458  # [1/Mpc]

    # Omega_ur: CLASS doesn't expose Omega0_ur directly in classy,
    # but we can reconstruct it from N_ur and Omega_g.
    #   Omega_ur = N_ur * (7/8) * (4/11)^{4/3} * Omega_g
    N_ur = cosmo_params_dict.get("N_ur", 3.046)
    Omega_ur = N_ur * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * Omega_g

    # Approximate relativistic ncdm contribution (fully relativistic at a_ini)
    N_ncdm = cosmo_params_dict.get("N_ncdm", 0)
    # Each relativistic ncdm species contributes ~ (7/8)*(T_ncdm/T_gamma)^4 * rho_gamma
    # With default T_ncdm/T_gamma = (4/11)^{1/3}:
    T_ncdm_over_T_gamma = (4.0 / 11.0) ** (1.0 / 3.0)
    rho_ncdm_rel_factor = N_ncdm * (7.0 / 8.0) * T_ncdm_over_T_gamma**4
    # The ncdm contribution scales as 3*p_ncdm = rho_ncdm when relativistic,
    # and is added to rho_rad as rho_ncdm_rel_tot = sum 3*p_ncdm.
    # In CLASS code: rho_ncdm_rel_tot += 3.*p_ncdm, where p_ncdm ~ rho_ncdm/3
    # So rho_ncdm_rel_tot ~ rho_ncdm (sum over species).
    # rho_ncdm_rel ~ N_ncdm * (7/8)*(T_ncdm/T_gamma)^4 * Omega_g * H0^2 / a^4
    Omega_ncdm_rel = rho_ncdm_rel_factor * Omega_g

    M.struct_cleanup()
    M.empty()

    return Omega_g, Omega_ur, H0_internal, Omega_ncdm_rel


def _run_class_single(
    c2: float,
    c1_guess: float,
    cosmo_params_dict: dict[str, Any],
    scf_base_dict: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run CLASS with a single (c1_guess, c2) pair.

    c1_guess is placed in scf_parameters[0]; CLASS tunes it via shooting to
    match Omega_scf.  The closer c1_guess is to the true solution, the more
    likely the root-finder will bracket the root.

    Returns
    -------
    result : dict with keys
        'c2', 'c1_guess', 'regime', 'phi_ini', 'phi_prime_ini', 'success', 'error'
    """
    M = Class()  # type: ignore[no-untyped-call]
    params: dict[str, Any] = dict(cosmo_params_dict)
    params.update(scf_base_dict)
    params.update(class_verbosity)

    # Build scf_parameters: c1, c2, c3, c4, q1..q4, unused, unused, phi_ini, phi_prime_ini
    # The last two are fallback values for when no attractor exists.
    scf_params_str = f"{c1_guess}, {c2}, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.01"
    params["scf_parameters"] = scf_params_str
    params["attractor_ic_scf"] = "yes"
    params["output"] = ""
    if verbose:
        params["background_verbose"] = 3

    M.set(params)

    result: dict[str, Any] = {
        "c2": c2,
        "c1_guess": c1_guess,
        "regime": None,
        "phi_ini": None,
        "phi_prime_ini": None,
        "success": False,
        "error": None,
    }
    try:
        M.compute()
        derived = M.get_current_derived_parameters(
            ["phi_ini_scf_ic", "phi_prime_scf_ic", "attractor_regime_scf"]
        )
        result["regime"] = int(derived["attractor_regime_scf"])
        result["phi_ini"] = derived["phi_ini_scf_ic"]
        result["phi_prime_ini"] = derived["phi_prime_scf_ic"]
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    finally:
        try:
            M.struct_cleanup()
        except Exception:
            pass
        M.empty()

    return result


def _run_class_single_packed(
    args: tuple[float, float, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Unpack arguments for ProcessPoolExecutor.map()."""
    return _run_class_single(args[0], args[1], args[2], args[3])


def run_class_with_c2(
    c2: float,
    c1_guess_list: list[float],
    cosmo_params_dict: dict[str, Any],
    scf_base_dict: dict[str, Any],
    pool: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """
    Try CLASS with all c1 guesses and pick the best successful result.

    'Best' means the lowest attractor regime number (1 > 2 > 3), since lower
    regimes have more accurate analytical solutions.  Among equal regimes,
    the smallest c1 is preferred (larger c1 bins are harder to scan in MCMC).

    When a pool is provided, guesses are tested in parallel batches of
    N_PARALLEL_WORKERS.  We try from smallest c1 upward; once a regime-1
    or regime-2 success is found we can stop early (regime 3 from a larger
    c1 cannot improve on it).

    Returns a single result dict (the best successful one, or a failure
    after all guesses are exhausted).
    """
    ordered = sorted(c1_guess_list)  # smallest c1 first

    best: dict[str, Any] | None = None
    last_error = ""

    if pool is not None and len(ordered) > 1:
        for batch_start in range(0, len(ordered), N_PARALLEL_WORKERS):
            batch = ordered[batch_start : batch_start + N_PARALLEL_WORKERS]
            args_list = [(c2, c1, cosmo_params_dict, scf_base_dict) for c1 in batch]
            batch_results = list(pool.map(_run_class_single_packed, args_list))
            for r in batch_results:
                if r["success"]:
                    if best is None or r["regime"] < best["regime"]:
                        best = r
                elif r.get("error"):
                    last_error = r["error"]
            # Early exit: regime 1 is optimal, regime 2 is good enough
            if best is not None and best["regime"] <= 2:
                return best
    else:
        for c1_g in ordered:
            r = _run_class_single(c2, c1_g, cosmo_params_dict, scf_base_dict)
            if r["success"]:
                if best is None or r["regime"] < best["regime"]:
                    best = r
                if best["regime"] <= 2:
                    return best
            else:
                last_error = r.get("error", "")

    if best is not None:
        return best

    return {
        "c2": c2,
        "c1_guess": None,
        "regime": None,
        "phi_ini": None,
        "phi_prime_ini": None,
        "success": False,
        "error": last_error,
    }


def _last_error_line(err: str | None) -> str:
    """Extract the last meaningful line from a CLASS error chain."""
    if not err:
        return "(no message)"
    lines = err.strip().split("\n")
    # Walk backwards to find the deepest '=>' line (the actual cause)
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("=>"):
            return line
    return lines[-1].strip()


def _classify_error(err: str | None) -> str:
    """Classify a CLASS error into a short category for grouping."""
    if not err:
        return "unknown"
    if "root must be bracketed" in err:
        return "shooting: root not bracketed in zriddr"
    if "tol_initial_Omega_r" in err or "not close enough to 1" in err:
        return "Omega_r tolerance exceeded (tracking energy too large near c2 ≈ 2)"
    return _last_error_line(err)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Pretty printing
# ═══════════════════════════════════════════════════════════════════════════

REGIME_NAMES: dict[int, str] = {
    0: "no attractor (user-provided ICs)",
    1: "large-field attractor (exact)",
    2: "approximate attractor (exp-like)",
    3: "small-field attractor (Taylor)",
}


def print_banner(text: str, char: str = "═", width: int = 80) -> None:
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_analytical_summary(
    boundaries: list[tuple[float, int, int]],
    c1: float,
    rho_rad: float,
    a_ini: float,
    c2_min: float,
    c2_max: float,
) -> None:
    """Print a human-readable summary of the analytical branch boundaries."""
    print_banner("ANALYTICAL BRANCH BOUNDARIES  (hyperbolic potential)")
    print(f"  c1 = {c1:.4e}")
    print(f"  rho_rad(a_ini) = {rho_rad:.6e}  [CLASS internal units]")
    print(f"  a_ini = {a_ini:.4e}")
    print(f"  scf_gamma = 4/3  (radiation domination)")
    print(f"  omega_background = 1/3  (radiation domination)")
    print()

    if not boundaries:
        r0 = analytical_attractor_hyperbolic(c1, c2_min, rho_rad, a_ini)[0]
        print(f"  No transitions found in [{c2_min}, {c2_max}].")
        print(f"  Entire range is regime {r0}: {REGIME_NAMES.get(r0, '?')}")
        return

    # Build intervals
    intervals: list[tuple[float, float, int]] = []
    prev_c2: float = c2_min
    prev_regime: int = analytical_attractor_hyperbolic(c1, c2_min, rho_rad, a_ini)[0]
    for c2_bnd, r_below, r_above in boundaries:
        intervals.append((prev_c2, c2_bnd, prev_regime))
        prev_c2 = c2_bnd
        prev_regime = r_above
    intervals.append((prev_c2, c2_max, prev_regime))

    header = f"  {'c2 range':>30s}  {'regime':>7s}  {'description'}"
    print(header)
    print("  " + "-" * 76)
    for lo, hi, reg in intervals:
        desc = REGIME_NAMES.get(reg, "?")
        print(f"  {lo:12.4f} < c2 < {hi:<12.4f}  {reg:>5d}    {desc}")
    print()


def print_class_scan_table(results: list[dict[str, Any]]) -> None:
    """Print the CLASS numerical scan results as a table."""
    print_banner("CLASS NUMERICAL SCAN  (hyperbolic, attractor_ic_scf = yes)")
    print("  Units: phi [M_Pl], phi' [M_Pl/Mpc] (conformal-time derivative),")
    print("         c1_guess is the shooting starting point; CLASS tunes c1")
    print("         internally to achieve Omega_scf.")
    print()

    rows: list[list[str]] = []
    for r in results:
        if r["success"]:
            c1g = r.get("c1_guess")
            rows.append(
                [
                    f"{r['c2']:<12.4e}",
                    str(r["regime"]),
                    f"{c1g:.1e}" if c1g is not None else "?",
                    f"{r['phi_ini']:.6e}" if r["phi_ini"] is not None else "N/A",
                    (
                        f"{r['phi_prime_ini']:.6e}"
                        if r["phi_prime_ini"] is not None
                        else "N/A"
                    ),
                ]
            )

    headers: list[str] = [
        "c2",
        "reg",
        "c1_guess",
        "phi_ini [M_Pl]",
        "phi'_ini [M_Pl/Mpc]",
    ]

    if _has_tabulate:
        print(_tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        # Manual table
        col_widths: list[int] = [
            max(len(h), max((len(row[i]) for row in rows), default=0))
            for i, h in enumerate(headers)
        ]
        fmt: str = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print("  " + fmt.format(*headers))
        print("  " + "  ".join("-" * w for w in col_widths))
        for row in rows:
            print("  " + fmt.format(*row))
    print()


def print_regime_summary_table(results: list[dict[str, Any]]) -> None:
    """
    Aggregate the scan results into a compact summary:
    one row per contiguous regime, showing the c2 range, representative
    ICs at the midpoint, and the c1_guess sub-bins needed.
    """
    print_banner("SUMMARY: c2 RANGES → ATTRACTOR REGIMES → REPRESENTATIVE ICs")

    if not results or not any(r["success"] for r in results):
        print("  No successful CLASS runs.")
        return

    # Group contiguous regimes
    groups: list[dict[str, Any]] = []
    for r in results:
        if not r["success"]:
            continue
        if groups and groups[-1]["regime"] == r["regime"]:
            groups[-1]["c2_max"] = r["c2"]
            groups[-1]["entries"].append(r)
        else:
            groups.append(
                {
                    "regime": r["regime"],
                    "c2_min": r["c2"],
                    "c2_max": r["c2"],
                    "entries": [r],
                }
            )

    for g in groups:
        entries: list[dict[str, Any]] = g["entries"]
        mid: dict[str, Any] = entries[len(entries) // 2]
        lo_entry: dict[str, Any] = entries[0]
        hi_entry: dict[str, Any] = entries[-1]
        regime: int = g["regime"]
        desc: str = REGIME_NAMES.get(regime, "?")

        # Collect c1_guess sub-bins
        c1_sub: dict[float, list[float]] = {}
        for e in entries:
            c1g = e.get("c1_guess")
            if c1g is not None:
                c1_sub.setdefault(c1g, []).append(e["c2"])

        print(f"\n  c2 ∈ {g['c2_min']:.4f} – {g['c2_max']:.4f}")
        print(f"    Regime {regime}: {desc}")
        print(
            f"    phi_ini  (midpoint) = "
            + (f"{mid['phi_ini']:.6e}" if mid["phi_ini"] is not None else "N/A")
        )
        print(
            f"    phi'_ini (midpoint) = "
            + (
                f"{mid['phi_prime_ini']:.6e}"
                if mid["phi_prime_ini"] is not None
                else "N/A"
            )
        )
        print(
            f"    phi_ini  [lo, hi]   = "
            + (
                f"[{lo_entry['phi_ini']:.4e}, {hi_entry['phi_ini']:.4e}]"
                if (lo_entry["phi_ini"] is not None and hi_entry["phi_ini"] is not None)
                else "N/A"
            )
        )
        if len(c1_sub) > 1:
            c1_strs = ", ".join(f"{c:.0e}" for c in sorted(c1_sub.keys()))
            print(f"    c1_guess bins: {c1_strs}")
            for c1g in sorted(c1_sub.keys()):
                c2s = c1_sub[c1g]
                print(
                    f"      c1={c1g:.0e}: c2 ∈ [{min(c2s):.4e}, {max(c2s):.4e}]"
                    f" ({len(c2s)} pts)"
                )
        elif c1_sub:
            c1g = next(iter(c1_sub))
            print(f"    c1_guess = {c1g:.0e} (single bin)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main analysis
# ═══════════════════════════════════════════════════════════════════════════


def _build_scan_points(
    boundaries: list[tuple[float, int, int]],
    c2_min: float,
    c2_max: float,
    pts_per_decade: int,
    fine_pts: int,
) -> list[float]:
    """
    Build a set of c2 scan points that:
      - skip regime-0 intervals entirely (no attractor → nothing to measure)
      - cover each attractor regime with log-spaced points (~pts_per_decade per
        order of magnitude)
      - add fine points around each boundary

    Returns a sorted list of unique c2 values.
    """
    # Derive the attractor intervals (regime != 0)
    intervals: list[tuple[float, float, int]] = []
    prev_c2 = c2_min
    # regime at the start of the range
    prev_reg = analytical_attractor_hyperbolic(1.0, c2_min, 1.0, 1.0)[0]
    # (dummy c1/rho_rad/a_ini: regime depends only on c2 relation to gamma)
    # Actually we need real rho_rad — but regime 0 depends only on c2**2 vs gamma.
    # Let me just evaluate properly using boundaries.
    if boundaries:
        prev_reg = boundaries[0][1]  # regime_below of first boundary
    for c2_bnd, r_below, r_above in boundaries:
        intervals.append((prev_c2, c2_bnd, prev_reg))
        prev_c2 = c2_bnd
        prev_reg = r_above
    intervals.append((prev_c2, c2_max, prev_reg))

    pts: set[float] = set()

    for lo, hi, reg in intervals:
        if reg == 0:
            continue  # skip — no attractor solution here
        # Log-spaced grid across [lo, hi]
        if hi <= lo or lo <= 0:
            continue
        n_decades = np.log10(hi / lo)
        n_pts = max(10, int(round(n_decades * pts_per_decade)))
        pts.update(float(v) for v in np.logspace(np.log10(lo), np.log10(hi), n_pts))

    # Fine points around each boundary
    for c2_bnd, _, _ in boundaries:
        eps = 0.03 * c2_bnd
        lo_fine = max(c2_bnd - eps, c2_min)
        hi_fine = min(c2_bnd + eps, c2_max)
        pts.update(float(v) for v in np.linspace(lo_fine, hi_fine, fine_pts))

    return sorted(pts)


def main() -> None:
    print_banner("Attractor Branch Analysis — Hyperbolic Potential", char="█")
    print(f"  V(φ) = c1 * (1 − tanh(c2 * φ))")
    print(f"  c1 reference = {c1_default:.4e}  (CLASS tunes c1 via shooting)")
    print(f"  c1 guesses   = {', '.join(f'{g:.0e}' for g in c1_guesses)}")
    print(f"  a_ini = {a_ini_over_a_today_default:.4e}")
    print(f"  c2 range: [{c2_scan_min}, {c2_scan_max:.0e}]")
    print(f"  Resolution: {c2_points_per_decade} points/decade")
    print()
    print("  Units (CLASS internal):")
    print("    φ      — reduced Planck mass M_Pl = (8πG)^{−1/2}")
    print("    φ'     — dφ/dτ in M_Pl/Mpc  (τ = conformal time)")
    print("    V(φ)   — M_Pl²/Mpc²")
    print("    ρ      — Mpc⁻²  (CLASS convention: ρ_class = 8πGρ_phys / 3c²)")
    print()

    # ── Step 1: Obtain radiation density ──────────────────────────────
    print("Step 1: Extracting radiation density from CLASS …")
    Omega_g, Omega_ur, H0_int, Omega_ncdm_rel = get_radiation_density_from_class(
        cosmo_params
    )
    Omega_rad_total = Omega_g + Omega_ur + Omega_ncdm_rel
    a_ini = a_ini_over_a_today_default
    rho_rad = Omega_rad_total * H0_int**2 / a_ini**4

    print(f"  Omega_g       = {Omega_g:.6e}")
    print(f"  Omega_ur      = {Omega_ur:.6e}")
    print(f"  Omega_ncdm_rel≈ {Omega_ncdm_rel:.6e}")
    print(f"  Omega_rad_tot = {Omega_rad_total:.6e}")
    print(f"  H0 (internal) = {H0_int:.6e}  [1/Mpc]")
    print(f"  rho_rad(a_ini)= {rho_rad:.6e}  [Mpc^⁻²]")
    print()

    # ── Step 2: Analytical branch boundaries ──────────────────────────
    print("Step 2: Analytical regime classification …")
    c2_dense = np.logspace(np.log10(c2_scan_min), np.log10(c2_scan_max), 100000)
    boundaries, regimes_dense, _ = find_analytical_boundaries(
        c1_default, rho_rad, a_ini, c2_range=c2_dense
    )
    print_analytical_summary(
        boundaries, c1_default, rho_rad, a_ini, c2_scan_min, c2_scan_max
    )

    # Print analytical values near each boundary
    if boundaries:
        print("  Analytical values near boundaries:")
        for c2_bnd, r_below, r_above in boundaries:
            eps = 1e-3 * c2_bnd
            for c2_test in [c2_bnd - eps, c2_bnd + eps]:
                if c2_test <= 0:
                    continue
                reg, phi, phip, info = analytical_attractor_hyperbolic(
                    c1_default, c2_test, rho_rad, a_ini
                )
                phi_str = f"{phi:.6e}" if phi is not None else "N/A"
                phip_str = f"{phip:.6e}" if phip is not None else "N/A"
                print(
                    f"    c2={c2_test:.6f}  →  regime={reg}  "
                    f"phi_ini={phi_str}  phi'_ini={phip_str}"
                )
        print()

    # ── Step 3: Greedy scan with carry-forward c1 ─────────────────────
    c2_points = _build_scan_points(
        boundaries,
        c2_scan_min,
        c2_scan_max,
        c2_points_per_decade,
        c2_fine_points,
    )
    n_c1 = len(c1_guesses)
    current_c1 = c1_guesses[0]  # start from smallest c1 (favours regime 2)
    n_total = len(c2_points)

    print(f"Step 3: Greedy scan — {n_total} c2 values, carry-forward c1")
    print(f"  Starting c1 = {current_c1:.0e}; {n_c1} guesses available")
    print(f"  Regime preference: 1 (best) > 2 > 3 (worst); 0 skipped")
    if N_PARALLEL_WORKERS > 1:
        print(f"  Parallel c1 search: {N_PARALLEL_WORKERS} workers")
    print()

    pool: ProcessPoolExecutor | None = None
    if N_PARALLEL_WORKERS > 1:
        pool = ProcessPoolExecutor(max_workers=N_PARALLEL_WORKERS)

    results: list[dict[str, Any]] = []
    max_c2_ok = 0.0
    consecutive_fails = 0
    n_skipped = 0

    try:
        for i, c2 in enumerate(c2_points):
            if (i + 1) % 10 == 0 or i == 0 or i == n_total - 1:
                print(
                    f"  [{i+1:4d}/{n_total}]  c2 = {c2:.4e}" f"  c1 = {current_c1:.0e}",
                    end="\r",
                )

            # Phase 1: try carry-forward c1 (single CLASS call)
            r = _run_class_single(c2, current_c1, cosmo_params, scf_base_params)
            if r["success"]:
                results.append(r)
                max_c2_ok = c2
                consecutive_fails = 0
                continue

            # Early skip: deep in failure zone
            if consecutive_fails >= 5 and max_c2_ok > 0 and c2 > 2.0 * max_c2_ok:
                results.append(
                    {
                        "c2": c2,
                        "c1_guess": None,
                        "regime": None,
                        "phi_ini": None,
                        "phi_prime_ini": None,
                        "success": False,
                        "error": r.get("error", ""),
                    }
                )
                consecutive_fails += 1
                n_skipped += 1
                continue

            # Phase 2: search all c1 guesses, pick lowest regime
            r2 = run_class_with_c2(
                c2,
                [g for g in c1_guesses if g != current_c1],
                cosmo_params,
                scf_base_params,
                pool=pool,
            )
            results.append(r2)
            if r2["success"]:
                current_c1 = r2["c1_guess"]
                max_c2_ok = c2
                consecutive_fails = 0
            else:
                consecutive_fails += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    print()  # newline after \r
    n_ok = sum(1 for r in results if r["success"])
    n_fail = sum(1 for r in results if not r["success"])
    print(
        f"  Done: {n_ok} succeeded, {n_fail} failed"
        f" ({n_skipped} skipped in failure tail)."
    )

    # Compact failure summary: group by error category
    if n_fail > 0:
        print()
        fail_cats: dict[str, list[float]] = {}
        for r in results:
            if r["success"]:
                continue
            cat = _classify_error(r.get("error"))
            fail_cats.setdefault(cat, []).append(r["c2"])
        print(f"  FAILURES ({n_fail} c2 points):")
        for cat, c2_list in fail_cats.items():
            if len(c2_list) == 1:
                print(f"    c2 = {c2_list[0]:.4e}: {cat}")
            else:
                print(
                    f"    c2 ∈ [{min(c2_list):.4e}, {max(c2_list):.4e}]"
                    f" ({len(c2_list)} pts): {cat}"
                )
    print()

    # ── Step 4: Print results ─────────────────────────────────────────
    success_results = [r for r in results if r["success"]]

    # Subsample for the detailed table: ~60 rows + boundary-adjacent
    if len(success_results) > 80:
        step = max(1, len(success_results) // 60)
        subset_indices = set(range(0, len(success_results), step))
        subset_indices.add(0)
        subset_indices.add(len(success_results) - 1)
        for i in range(1, len(success_results)):
            if success_results[i]["regime"] != success_results[i - 1]["regime"]:
                subset_indices.update(
                    [max(0, i - 1), i, min(len(success_results) - 1, i + 1)]
                )
        table_results = [success_results[i] for i in sorted(subset_indices)]
    else:
        table_results = success_results

    print_class_scan_table(table_results)
    print_regime_summary_table(success_results)

    # ── Step 5: Practical summary for MCMC priors ─────────────────────
    print_banner("PRACTICAL SUMMARY FOR MCMC PRIORS")
    print("  Units: φ in M_Pl, φ' = dφ/dτ in M_Pl/Mpc")
    print()

    # -- Analytical regime boundaries --
    if boundaries:
        print("  Analytical regime boundaries (c1_ref = {:.0e}):".format(c1_default))
        for c2_bnd, r_below, r_above in boundaries:
            print(
                f"    c2 ≈ {c2_bnd:.4f}:  regime {r_below} ({REGIME_NAMES.get(r_below, '?')}) "
                f"→ regime {r_above} ({REGIME_NAMES.get(r_above, '?')})"
            )
        print()

    # Report analytical regime 0 range
    for c2_bnd, r_below, r_above in boundaries:
        if r_below == 0:
            print(f"  For 0 < c2 < {c2_bnd:.4f}:")
            print(f"    → Regime 0: {REGIME_NAMES[0]}")
            print(f"    → phi_ini and phi'_ini are user-provided (no attractor)")
            print()

    # Report CLASS-validated attractor regimes with c1 sub-bins
    if success_results:
        groups: list[dict[str, Any]] = []
        for r in success_results:
            if groups and groups[-1]["regime"] == r["regime"]:
                groups[-1]["c2_max"] = r["c2"]
                groups[-1]["entries"].append(r)
            else:
                groups.append(
                    {
                        "regime": r["regime"],
                        "c2_min": r["c2"],
                        "c2_max": r["c2"],
                        "entries": [r],
                    }
                )

        for g in groups:
            entries: list[dict[str, Any]] = g["entries"]
            reg = g["regime"]
            desc = REGIME_NAMES.get(reg, "?")
            phi_vals = [e["phi_ini"] for e in entries if e["phi_ini"] is not None]
            phip_vals = [
                e["phi_prime_ini"] for e in entries if e["phi_prime_ini"] is not None
            ]

            print(f"  For {g['c2_min']:.4e} < c2 < {g['c2_max']:.4e}:")
            print(f"    → Regime {reg}: {desc}")
            if phi_vals:
                print(
                    f"    → phi_ini  ∈ [{min(phi_vals):.6e}, {max(phi_vals):.6e}]  [M_Pl]"
                )
            if phip_vals:
                print(
                    f"    → phi'_ini ∈ [{min(phip_vals):.6e}, {max(phip_vals):.6e}]  [M_Pl/Mpc]"
                )

            # c1_guess sub-bins
            c1_sub: dict[float, list[float]] = {}
            for e in entries:
                c1g = e.get("c1_guess")
                if c1g is not None:
                    c1_sub.setdefault(c1g, []).append(e["c2"])

            if len(c1_sub) > 1:
                print(f"    → c1_guess sub-bins (each may need separate MCMC runs):")
                for c1g in sorted(c1_sub.keys()):
                    c2s = c1_sub[c1g]
                    print(
                        f"        c1 = {c1g:.0e}:  "
                        f"c2 ∈ [{min(c2s):.4e}, {max(c2s):.4e}]"
                        f"  ({len(c2s)} pts)"
                    )
            elif c1_sub:
                c1g = next(iter(c1_sub))
                print(f"    → c1_guess = {c1g:.0e} (single bin)")
            print()

    # Shooting limit
    if success_results:
        max_c2_ok_val = max(r["c2"] for r in success_results)
        failed_above = [
            r["c2"] for r in results if not r["success"] and r["c2"] > max_c2_ok_val
        ]
        if failed_above:
            min_c2_fail = min(failed_above)
            print(f"  NOTE: CLASS shooting succeeds up to c2 ≈ {max_c2_ok_val:.4e}")
            print(f"        Fails from c2 ≈ {min_c2_fail:.4e} upward.")
            print(
                f"        (tried {n_c1} c1 guesses per c2:"
                f" {c1_guesses[0]:.0e} .. {c1_guesses[-1]:.0e})"
            )
            print()

    print("Done.")


if __name__ == "__main__":
    main()
