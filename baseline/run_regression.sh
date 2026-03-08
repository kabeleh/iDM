#!/bin/bash
# run_regression.sh — Generate regression reference outputs (C + Python)
#
# Usage: bash baseline/run_regression.sh <output_dir>
#   e.g. bash baseline/run_regression.sh baseline
#        bash baseline/run_regression.sh assessment
#
# Requires: ./class binary compiled, classy installed in active Python env.
# Must be run from the CLASS repo root.
set -euo pipefail

OUTDIR="${1:?Usage: $0 <output_dir>}"
CLASSDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CLASSDIR"

[[ -x ./class ]] || { echo "ERROR: ./class not found or not executable"; exit 1; }
mkdir -p "$OUTDIR"

# Keys to strip from original .ini (replaced by our overrides)
STRIP='^\s*(write_background|write_thermodynamics|write_primordial|write_exotic_injection|write_noninjection|write_distortions|write_parameters|overwrite_root|k_output_values|DM_annihilation_efficiency)\s*='

# --- run_one <ini> <name> [extra .ini files or flag lines ...] ---
# Builds a temp .ini with always-safe write flags plus any extras.
# Extra arguments that are existing files are appended as .ini overrides;
# bare strings (e.g. "write_exotic_injection = yes") are appended verbatim.
#
# Write-flag safety rules (only enable when the prerequisite module is active):
#   write_exotic_injection  → needs has_exotic_injection  (DM_annihilation_efficiency > 0, etc.)
#   write_noninjection      → needs has_distortions       (output contains Sd/sd/SD)
#   write_distortions       → safe always (internal guard), but only useful with Sd
run_one() {
    local ini="$1" name="$2"; shift 2
    local tmpini
    tmpini=$(mktemp /tmp/regression_XXXXXX.ini)

    grep -Ev "$STRIP" "$ini" >> "$tmpini"

    # Always-safe write flags.  The three conditional flags
    # (write_exotic_injection, write_noninjection, write_distortions)
    # are NOT listed here — they default to 'no' in CLASS and are
    # only added by per-run override arguments when safe.
    cat >> "$tmpini" <<'WRITE'
write_background = yes
write_thermodynamics = yes
write_primordial = yes
write_parameters = no
overwrite_root = yes
k_output_values = 0.0001, 0.01, 0.1
WRITE

    # Append extra arguments: files are included as .ini overrides,
    # bare strings are appended verbatim (for per-run flag overrides).
    for extra in "$@"; do
        if [[ -f "$extra" ]]; then
            grep -Ev '^\s*#|^\s*$' "$extra" >> "$tmpini" || true
        else
            echo "$extra" >> "$tmpini"
        fi
    done

    echo "root = ${OUTDIR}/${name}" >> "$tmpini"

    echo "=== Running: $name ==="
    ./class "$tmpini" > /tmp/regression_${name}.log 2>&1 || {
        echo "WARNING: $name exited with code $? (output may still be complete)"
    }
    rm -f "$tmpini"
    echo "=== Done: $name ==="
}

echo "============================================"
echo "  Regression run → ${OUTDIR}/"
echo "============================================"

# 1. ΛCDM — no exotic injection, no Sd → base flags only
run_one explanatory.ini explanatory

# 2. iDM plain — has Sd (sd in output) → enable distortions.
#    No exotic injection source → write_exotic_injection stays off.
#    write_noninjection is also off: output_heating() uses pin->z_size
#    (injection struct) for the noninjection allocation, but injection_init
#    only runs when has_exotic_injection is true. Without it, pin->z_size
#    is uninitialized → crash (free(): invalid pointer).
run_one iDM.ini iDM \
    "write_distortions = yes"

# 3. iDM + exotic injection — has Sd + exotic → enable all three
run_one iDM.ini iDM_exotic \
    regression_exotic.ini \
    "write_exotic_injection = yes" \
    "write_noninjection = yes" \
    "write_distortions = yes"

# 4. PGO suite — no exotic injection, no Sd → base flags only
pgo_count=0
for ini in pgo_*_ic_*.ini pgo_*_tracking_*.ini; do
    [[ -f "$ini" ]] || continue
    name=$(basename "$ini" .ini)
    run_one "$ini" "$name"
    pgo_count=$((pgo_count + 1))
done
echo "PGO runs completed: $pgo_count"

# 5. Python snapshots (classy)
echo "=== Python snapshots ==="
python3 - "$OUTDIR" <<'PYEOF'
import sys, os
import numpy as np
from classy import Class

outdir = sys.argv[1]

COMMON = {
    "h": 0.67810,
    "omega_b": 0.02238280,
    "omega_cdm": 0.1201075,
    "N_ur": 3.044,
    "T_cmb": 2.7255,
    "z_reio": 7.6711,
    "YHe": 0.25,
    "output": "tCl,pCl,lCl,mPk",
    "lensing": "yes",
    "l_max_scalars": 2500,
    "P_k_max_1/Mpc": 1.0,
    "z_max_pk": 10.0,
}

IDM_BASE = {
    **COMMON,
    "model_cdm": "i",
    "Omega_fld": 0,
    "Omega_scf": -0.7,
    "Omega_Lambda": 0.0,
    "scf_potential": "hyperbolic",
    "scf_parameters": "1e-08, 0.59, 0.0, 0.0, 0, 0, 0., 0., 0., 0., 1, 0",
    "attractor_ic_scf": "no",
    "scf_tuning_index": 0,
    "gauge": "newtonian",
}

MODELS = {
    "lcdm": {**COMMON},
    "idm_weak": {**IDM_BASE, "cdm_c": 0.01},
    "idm_strong": {**IDM_BASE, "cdm_c": 0.1},
}

k_values = np.logspace(-4, -0.3, 200)

for name, params in MODELS.items():
    print(f"  Python: {name}")
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    cl = cosmo.lensed_cl(2500)
    np.savez(os.path.join(outdir, f"{name}_cl.npz"),
             ell=cl["ell"], tt=cl["tt"], ee=cl["ee"], te=cl["te"])
    bg = cosmo.get_background()
    np.savez(os.path.join(outdir, f"{name}_bg.npz"), **bg)
    pk = np.array([cosmo.pk(k, 0) for k in k_values])
    np.save(os.path.join(outdir, f"{name}_pk.npy"), pk)

    # Derived parameters — catch regressions in shooting, integration,
    # and post-processing that wouldn't show up in raw table data.
    derived = {}
    # Universal derived quantities
    for key in ["H0", "h", "age", "conformal_age", "Neff",
                "tau_reio", "z_reio",
                "z_rec", "tau_rec", "rs_rec", "da_rec",
                "z_star", "tau_star", "rs_star", "da_star",
                "z_d", "rs_d",
                "100*theta_s", "100*theta_star",
                "sigma8", "sigma8_cb",
                "Omega_m", "omega_m",
                "k_eq", "z_eq"]:
        try:
            val = cosmo.get_current_derived_parameters([key])
            derived[key] = val[key]
        except Exception:
            pass
    # SCF-specific quantities (only for iDM models)
    if "scf" in str(params.get("scf_potential", "")):
        for key in ["phi_ini_scf_ic", "phi_prime_scf_ic",
                     "phi_scf_min", "phi_scf_max", "phi_scf_range",
                     "cdm_f_phi0", "cdm_f_phi0_inv",
                     "attractor_regime_scf"]:
            try:
                val = cosmo.get_current_derived_parameters([key])
                derived[key] = val[key]
            except Exception:
                pass
    # Distance and growth at representative redshifts
    z_probes = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]
    derived["z_probes"] = z_probes
    for z in z_probes:
        try:
            derived[f"H_z{z}"] = cosmo.Hubble(z)
        except Exception:
            pass
        try:
            derived[f"Da_z{z}"] = cosmo.angular_distance(z)
        except Exception:
            pass
        try:
            derived[f"D_z{z}"] = cosmo.scale_independent_growth_factor(z)
        except Exception:
            pass
        try:
            derived[f"f_z{z}"] = cosmo.scale_independent_growth_factor_f(z)
        except Exception:
            pass
    # f*sigma8 at LSS-relevant redshifts
    for z in [0.0, 0.5, 1.0, 2.0]:
        try:
            derived[f"fsig8_z{z}"] = cosmo.scale_independent_f_sigma8(z)
        except Exception:
            pass

    np.savez(os.path.join(outdir, f"{name}_derived.npz"), **{
        k: np.float64(v) for k, v in derived.items()
        if not isinstance(v, list)
    })

    cosmo.struct_cleanup()
    cosmo.empty()

print("  Python snapshots done.")
PYEOF

# Summary
total=$(find "$OUTDIR" -type f | wc -l)
echo ""
echo "============================================"
echo "  Complete: ${total} files in ${OUTDIR}/"
echo "============================================"
