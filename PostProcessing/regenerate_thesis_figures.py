#!/usr/bin/env python3
"""Regenerate every thesis figure/table from the canonical MCMC archive.

This driver is the single recorded entry point for producing the dissertation
artifacts. It encodes the dataset-matched pairing of hyperbolic and LCDM
best-fits (the pairing is otherwise invisible: BestFitPlot.py takes opaque
CLI paths) and writes a provenance manifest next to the outputs so every
figure can be traced back to the exact archive files it was built from.

Steps (subcommands):
  bestfit    BestFitPlot.py per root with the dataset-matched --lcdm_baseline
             (figures 4.8 Pk and 4.9 Cl, plus plot1/2/3/6 line plots)
  heatmaps   posterior_background_heatmaps.py with the production sampling
             settings (figures 4.7, 4.10, 4.11, 4.16, 4.17)
  getdist    data_postprocessing_getDist.py --strict-export (figures 4.13,
             4.14, 5.2, 5.3, 5.4)
  tables     omega_table.py and the getdist LaTeX tables
  all        everything above, in the order listed

All inputs come from MCMC_archive/ (the scripts themselves refuse anything
else); the manifest records path, size, mtime and head-hash per input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_HDM = _SCRIPT_DIR.parents[1]
_ARCHIVE_HYP = _HDM / "MCMC_archive" / "Hyperbolic"
_ARCHIVE_LCDM = _HDM / "MCMC_archive" / "LCDM"
_PLOTS_DIR = _SCRIPT_DIR / "Plots"
_HEATMAPS_DIR = _SCRIPT_DIR / "PosteriorHeatmaps"
_MANIFEST = _SCRIPT_DIR / "thesis_figures" / "manifest.json"

# Thesis captions state 5000 posterior trajectories per dataset; keep in sync.
_HEATMAP_MAX_SAMPLES = 5000

# root -> (hyperbolic bestfit, dataset-matched LCDM bestfit). The LCDM
# baseline MUST come from the same likelihood combination as the hyperbolic
# run (user decision for figs 4.8/4.9; also used by omega_table.py).
CANONICAL_MAP: dict[str, dict[str, Path]] = {
    "hyperbolic_Planck_InitCond_MCMC": {
        "hyp": _ARCHIVE_HYP / "Planck" / "hyperbolic_Planck_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE_LCDM / "Planck" / "cobaya_mcmc_fast_CMB_LCDM.bestfit",
    },
    "hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC": {
        "hyp": _ARCHIVE_HYP
        / "Planck_PantheonPlus"
        / "hyperbolic_Planck_PP_DESI_InitCond_Swamp_MCMC.bestfit",
        "lcdm": _ARCHIVE_LCDM
        / "Planck_PantheonPlus"
        / "cobaya_mcmc_fast_CMB_LCDM.post.PP.bestfit",
    },
    "hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC": {
        "hyp": _ARCHIVE_HYP
        / "Planck_PantheonPlus_SH0ES"
        / "hyperbolic_Planck_PPS_DESI_InitCond_Swamp_MCMC.bestfit",
        "lcdm": _ARCHIVE_LCDM
        / "Planck_PantheonPlus_SH0ES"
        / "cobaya_mcmc_fast_CMB_LCDM.post.PPS.bestfit",
    },
    "hyperbolic_PP_D_InitCond_MCMC": {
        "hyp": _ARCHIVE_HYP / "PantheonPlus" / "hyperbolic_PP_D_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE_LCDM
        / "PantheonPlus"
        / "cobaya_polychord_CV_PP_DESI_LCDM.post.S8.bestfit",
    },
    "hyperbolic_PP_S_D_InitCond_MCMC": {
        "hyp": _ARCHIVE_HYP
        / "PantheonPlus_SH0ES"
        / "hyperbolic_PP_S_D_InitCond_MCMC.bestfit",
        "lcdm": _ARCHIVE_LCDM
        / "PantheonPlus_SH0ES"
        / "cobaya_mcmc_CV_PP_S_DESI_LCDM.post.S8.bestfit",
    },
}


def _head_hash(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:16]


def _input_record(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "head_hash": _head_hash(path),
    }


def _load_manifest() -> dict:
    if _MANIFEST.exists():
        return json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return {"description": "Provenance of thesis figure regenerations", "runs": []}


def _record_run(step: str, argv: list[str], inputs: list[Path], returncode: int) -> None:
    manifest = _load_manifest()
    manifest["runs"].append(
        {
            "step": step,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": argv,
            "returncode": returncode,
            "inputs": [_input_record(p) for p in inputs if p.exists()],
        }
    )
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run(step: str, argv: list[str], inputs: list[Path]) -> int:
    print(f"\n=== [{step}] {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=str(_SCRIPT_DIR))
    _record_run(step, argv, inputs, proc.returncode)
    if proc.returncode != 0:
        print(f"!!! [{step}] exited with {proc.returncode}", file=sys.stderr)
    return proc.returncode


def _validate_map() -> None:
    missing = [
        str(p)
        for entry in CANONICAL_MAP.values()
        for p in entry.values()
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Canonical archive inputs missing:\n  " + "\n  ".join(missing)
        )


def step_bestfit(force_rerun: bool) -> int:
    rc = 0
    for root, entry in CANONICAL_MAP.items():
        argv = [
            sys.executable,
            str(_SCRIPT_DIR / "BestFitPlot.py"),
            "--bestfit_file",
            str(entry["hyp"]),
            "--lcdm_baseline",
            str(entry["lcdm"]),
            "--output_dir",
            str(_PLOTS_DIR),
        ]
        if force_rerun:
            argv += ["--force-rerun", "both"]
        rc |= _run(f"bestfit:{root}", argv, [entry["hyp"], entry["lcdm"]])
    return rc


def step_heatmaps(extra: list[str]) -> int:
    # No --roots override: per-root RNG seeds depend on the position in the
    # default roots list, so subsetting would silently change the drawn
    # samples of the remaining roots.
    argv = [
        sys.executable,
        str(_SCRIPT_DIR / "posterior_background_heatmaps.py"),
        "--max-samples",
        str(_HEATMAP_MAX_SAMPLES),
        "--output-dir",
        str(_HEATMAPS_DIR),
        *extra,
    ]
    inputs = [entry["hyp"] for entry in CANONICAL_MAP.values()]
    return _run("heatmaps", argv, inputs)


def step_getdist(only_plot: int | None) -> int:
    argv = [
        sys.executable,
        str(_SCRIPT_DIR / "data_postprocessing_getDist.py"),
        "--strict-export",
        "--strict-export-dir",
        str(_PLOTS_DIR),
        "--skip-tables",
        "--auto-close-figures",
    ]
    if only_plot is not None:
        argv += ["--only-plot", str(only_plot)]
    inputs = [entry["hyp"] for entry in CANONICAL_MAP.values()]
    return _run("getdist", argv, inputs)


def step_tables() -> int:
    rc = _run(
        "tables:omega",
        [sys.executable, str(_SCRIPT_DIR / "omega_table.py")],
        [p for e in CANONICAL_MAP.values() for p in e.values()],
    )
    tables_out = _SCRIPT_DIR / "thesis_figures" / "getdist_tables.txt"
    tables_out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(_SCRIPT_DIR / "data_postprocessing_getDist.py"),
        "--skip-plots",
        "--auto-close-figures",
    ]
    print(f"\n=== [tables:getdist] {' '.join(argv)} > {tables_out}", flush=True)
    with open(tables_out, "w", encoding="utf-8") as fh:
        proc = subprocess.run(argv, cwd=str(_SCRIPT_DIR), stdout=fh)
    _record_run(
        "tables:getdist",
        argv,
        [e["hyp"] for e in CANONICAL_MAP.values()],
        proc.returncode,
    )
    return rc | proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)

    p_bestfit = sub.add_parser("bestfit", help="Pk/Cl and line plots per root")
    p_bestfit.add_argument("--force-rerun", action="store_true")

    p_heat = sub.add_parser("heatmaps", help="posterior heatmaps (long run)")
    p_heat.add_argument("extra", nargs="*", help="extra args passed through")

    p_getdist = sub.add_parser("getdist", help="getdist strict-export plots")
    p_getdist.add_argument("--only-plot", type=int, choices=[1, 2, 3, 4])

    sub.add_parser("tables", help="omega table + getdist LaTeX tables")

    p_all = sub.add_parser("all", help="everything (heatmaps take hours)")
    p_all.add_argument("--force-rerun", action="store_true")

    args = parser.parse_args()
    _validate_map()

    if args.step == "bestfit":
        return step_bestfit(args.force_rerun)
    if args.step == "heatmaps":
        return step_heatmaps(args.extra)
    if args.step == "getdist":
        return step_getdist(args.only_plot)
    if args.step == "tables":
        return step_tables()
    rc = step_bestfit(args.force_rerun)
    rc |= step_heatmaps([])
    rc |= step_getdist(None)
    rc |= step_tables()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
